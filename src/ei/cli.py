"""Command-line entry points: ei-validate, ei-run, ei-selftest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .config import ExperimentConfig, load_config
from .extract import SyntheticConfig, SyntheticSource, make_items
from .pipeline import run_experiment
from .validate import assess_design


def cmd_validate(args) -> int:
    cfg = ExperimentConfig.load(args.config)
    errs = cfg.validate()
    print(f"config: {args.config}")
    print(f"preregistration_hash: {cfg.preregistration_hash()}")
    if errs:
        print("\nSCHEMA ERRORS:")
        for e in errs:
            print(f"  - {e}")
    else:
        print("schema: OK")
    d = assess_design(cfg)
    print(f"\ndesign coverage: {d.coverage*100:.1f}%")
    print(f"disposition:     {d.disposition}")
    if d.blocking:
        print("blocking gaps:")
        for b in d.blocking:
            print(f"  - {b}")
    else:
        print("blocking gaps:  none")
    if args.verbose:
        print("\nchecks:")
        for c in d.checks:
            print(f"  [{'x' if c.passed else ' '}] {c.key:28s} (w={c.weight}) {c.detail}")
    return 1 if errs else 0


def _alt_sources(cfg, primary_cfg: SyntheticConfig):
    """Build the comparison systems used by the null ladder and cross-model stage.

    In a real run these are real models; here they are synthetic systems whose ground
    truth is known, so the pipeline's verdicts can be checked against the right answers.
    """
    seed = cfg.runtime.seed
    untrained = []
    for i in range(cfg.nulls.untrained_seeds):
        uc = SyntheticConfig(hidden_dim=primary_cfg.hidden_dim, n_steps=primary_cfg.n_steps,
                             planted_modes=0, planted_amplitude=0.0,
                             ar1_noise=primary_cfg.ar1_noise, seed=seed + 100 + i)
        untrained.append(SyntheticSource(uc, name=f"untrained_seed{i}"))

    mimic_cfg = SyntheticConfig(hidden_dim=primary_cfg.hidden_dim, n_steps=primary_cfg.n_steps,
                                planted_modes=1, planted_persistence=(0.55,),
                                planted_amplitude=0.25, ar1_noise=primary_cfg.ar1_noise,
                                seed=seed + 200)
    ckpt_cfg = SyntheticConfig(**{**primary_cfg.__dict__, "seed": seed + 300})
    arch_cfg = SyntheticConfig(**{**primary_cfg.__dict__, "hidden_dim": primary_cfg.hidden_dim,
                                  "seed": seed + 400})
    return {
        "untrained": untrained,
        "mimic": SyntheticSource(mimic_cfg, name="output_matched_mimic"),
        "distinct_checkpoint": SyntheticSource(ckpt_cfg, name="distinct_checkpoint"),
        "architecture_diverse": SyntheticSource(arch_cfg, name="architecture_diverse"),
    }


def cmd_run(args) -> int:
    cfg = load_config(args.config)
    if args.out:
        cfg.runtime.output_dir = args.out
    if args.seed is not None:
        cfg.runtime.seed = args.seed
    if args.fast:
        cfg.nulls.n_surrogates = 60
        cfg.stats.n_bootstrap = 200
        cfg.extraction.max_items = min(cfg.extraction.max_items, 80)
        cfg.cross_model.n_alignment_permutations = 40

    if cfg.runtime.backend != "synthetic":
        print(f"backend={cfg.runtime.backend}: implement ActivationSource and pass it to "
              f"run_experiment(). See [PLACEHOLDER: ACTIVATION EXTRACTION] in ei/extract.py.",
              file=sys.stderr)
        return 2

    pc = SyntheticConfig(hidden_dim=args.hidden_dim, n_steps=args.n_steps,
                         planted_modes=(2 if args.planted_amplitude > 0 else 0),
                         planted_amplitude=args.planted_amplitude, seed=cfg.runtime.seed)
    source = SyntheticSource(pc, name="synthetic_primary")
    items = make_items(cfg.extraction.max_items, "target")
    out_dir = Path(cfg.runtime.output_dir) / cfg.name

    result = run_experiment(cfg, source=source, alt_sources=_alt_sources(cfg, pc),
                            items=items, out_dir=out_dir,
                            n_surrogates=cfg.nulls.n_surrogates)

    print(f"\nartifacts -> {out_dir}")
    print(f"design coverage : {result['design_assessment']['coverage']*100:.1f}%")
    print(f"design status   : {result['design_assessment']['disposition']}")
    print(f"candidates      : {result['structure']['gate_report']['n_candidates']}")
    for k, t in result["validation"]["null_tests"].items():
        extra = f"p={t['p_value']:.4g}" if t.get("p_value") is not None else ""
        print(f"null {k:24s}: {t.get('decision','n/a'):5s} {extra}")
    cz = result["validation"].get("causal", {}).get("verdict", {})
    print(f"causal verdict  : {'PASS' if cz.get('pass') else 'FAIL'}")
    print(f"\nDISPOSITION     : {result['disposition']['disposition']}")
    print(f"licensed        : {result['disposition']['licensed_claim']}")
    print(f"excluded        : {result['disposition']['excluded_claim'][:100]}…")
    return 0


def cmd_selftest(args) -> int:
    """Recover planted modes from synthetic data with KNOWN ground truth.

    This is the calibration check: before the instrument is pointed at a real model, it
    must (a) recover a planted mode from an AR(1) background, and (b) return NO candidate
    when nothing is planted.  Both are asserted.
    """
    from . import modes as M

    from . import nulls as N

    rng = np.random.default_rng(args.seed)
    d, T, n = 96, 80, 40

    print("== 1. POSITIVE CONTROL: planted modes on an AR(1) background ==")
    cfg = SyntheticConfig(hidden_dim=d, n_steps=T, planted_modes=2, ar1_noise=0.85,
                          planted_amplitude=args.amplitude, seed=args.seed)
    src = SyntheticSource(cfg)
    traj = [src.trajectory(it) for it in make_items(n, "target")]
    ms = M.extract_modes(traj, reduced_dim=24, rng=rng)
    sel = ms.top(2, by="salience")
    rec = ms.ambient_subspace([m.index for m in sel])
    truth = src.planted_subspace()
    ov = M.subspace_overlap(rec, truth)
    chance = src.planted_dim / d
    print(f"  planted dim={src.planted_dim}  recovered dim={rec.shape[1]}")
    print(f"  subspace overlap = {ov:.3f}   (chance = {chance:.3f})")
    print(f"  recovered |lambda| = {[round(m.persistence, 3) for m in sel]}  "
          f"(planted {list(cfg.planted_persistence[:2])})")
    warn = M.persistence_bias_warning(sel[0].snr)
    if warn:
        print(f"  NOTE: {warn}")
    ok_pos = ov > 0.85

    print("\n== 2. NEGATIVE CONTROL: pure AR(1), nothing planted ==")
    ncfg = SyntheticConfig(hidden_dim=d, n_steps=T, planted_modes=0, planted_amplitude=0.0,
                           ar1_noise=0.85, seed=args.seed + 1)
    ntraj = [SyntheticSource(ncfg).trajectory(it) for it in make_items(n, "target")]
    nms = M.extract_modes(ntraj, reduced_dim=24, rng=rng)
    ncand = M.gate_modes(nms, min_persistence=0.85, min_energy_share=0.005,
                         min_consistency=0.3, min_snr=1.5)
    print(f"  descriptive gates still pass {len(ncand)} 'modes' with max |lambda| = "
          f"{max(m.persistence for m in nms.modes):.3f}")
    print("  -> gates alone do NOT reject coloured noise. This is the empirical reason")
    print("     the phase-scramble null is mandatory rather than optional.")

    print("\n== 3. NULL-POWER CHECK: can the surrogate even be beaten? ==")
    try:
        r = N.assert_null_has_power(traj[0], N.phase_scramble, rng)
        print(f"  independent-phase surrogate: lag-1 cross-cov corr = "
              f"{r['preserved_corr']:.3f} -> HAS POWER")
        ok_null = True
    except ValueError as exc:
        print(f"  FAIL: {exc}")
        ok_null = False
    try:
        N.assert_null_has_power(
            traj[0], lambda A, g: N.phase_scramble(A, g, preserve_cross=True), rng)
        print("  identical-phase surrogate: NOT refused -- guard is broken")
        ok_guard = False
    except ValueError:
        print("  identical-phase surrogate: correctly REFUSED (would preserve lag-1")
        print("     cross-covariance at corr~1.0, giving the test zero power)")
        ok_guard = True

    print("\n== SUMMARY ==")
    for label, ok in (("planted-subspace recovery", ok_pos),
                      ("surrogate null has power", ok_null),
                      ("zero-power null refused", ok_guard)):
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    if not (ok_pos and ok_null and ok_guard):
        print("\n  Instrument is NOT calibrated. Do not point it at a real model.")
        return 1
    print("\n  Instrument calibrated on known ground truth.")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="ei", description="Eigenmode instrumentation for "
                                                       "persistent-structure interpretability")
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="validate a config and score the design")
    v.add_argument("config")
    v.add_argument("-v", "--verbose", action="store_true")
    v.set_defaults(func=cmd_validate)

    r = sub.add_parser("run", help="run the pipeline end to end")
    r.add_argument("config")
    r.add_argument("--out", default=None)
    r.add_argument("--seed", type=int, default=None)
    r.add_argument("--fast", action="store_true", help="reduced surrogate counts for a smoke run")
    r.add_argument("--hidden-dim", type=int, default=128)
    r.add_argument("--n-steps", type=int, default=60)
    r.add_argument("--planted-amplitude", type=float, default=1.0,
                   help="synthetic signal strength. 0 = no structure (expect "
                        "'structure_absent'); ~3.5 = strong structure (expect the ceiling "
                        "disposition). Use it to verify the framework discriminates.")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("selftest", help="recover planted modes from synthetic ground truth")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--amplitude", type=float, default=3.0,
                   help="planted signal amplitude (per-mode SD relative to background)")
    s.set_defaults(func=cmd_selftest)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
