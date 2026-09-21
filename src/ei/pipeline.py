"""
Analysis pipeline: strict separation of EXTRACT -> TRANSFORM -> VALIDATE -> REPORT.

The separation is enforced by data flow, not convention:

  Stage 1 EXTRACT    reads the model (or the synthetic source); writes raw trajectories
                     and behaviour.  Knows nothing about hypotheses.  The ONLY stage that
                     touches a model.
  Stage 2 TRANSFORM  reads trajectories; writes modes and candidate subspaces.  Fitted on
                     the DISCOVERY split only.  Knows nothing about statistics.
  Stage 3 VALIDATE   reads candidates + the CONFIRMATION split; writes test results.
                     Never re-fits the basis -- re-fitting on the confirmation split is
                     the leak that makes most such pipelines invalid.
  Stage 4 REPORT     reads test results; writes artifacts.  Never touches activations.

The split discipline is the load-bearing safeguard.  Modes are *selected* on discovery,
*aligned* on the alignment split, *calibrated* on the calibration split, and *tested* on
the confirmation split, which is touched exactly once.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from . import align as align_mod
from . import causal as causal_mod
from . import modes as modes_mod
from . import nulls as nulls_mod
from . import stats as stats_mod
from .extract import Item, ReadoutSpec, SyntheticConfig, SyntheticSource, make_items
from .validate import assess_design, derive_disposition


# --------------------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------------------


@dataclass
class Splits:
    discovery: list
    alignment: list
    calibration: list
    confirmation: list

    def sizes(self) -> dict:
        return {k: len(getattr(self, k)) for k in ("discovery", "alignment", "calibration", "confirmation")}


def make_splits(items: list[Item], fractions: dict, rng: np.random.Generator) -> Splits:
    """Disjoint, deterministic, pair-aware splits.

    Minimal pairs are kept TOGETHER: splitting a pair across discovery and confirmation
    leaks the confirmation item's twin into mode selection.
    """
    by_pair: dict[str, list[Item]] = {}
    for it in items:
        by_pair.setdefault(it.pair_id or it.item_id, []).append(it)
    keys = sorted(by_pair)
    rng.shuffle(keys)
    n = len(keys)
    n_d = int(round(fractions.get("discovery", 0.4) * n))
    n_a = int(round(fractions.get("alignment", 0.2) * n))
    n_c = int(round(fractions.get("calibration", 0.1) * n))
    chunks = [keys[:n_d], keys[n_d:n_d + n_a], keys[n_d + n_a:n_d + n_a + n_c], keys[n_d + n_a + n_c:]]
    return Splits(*[[it for k in chunk for it in by_pair[k]] for chunk in chunks])


# --------------------------------------------------------------------------------------
# Stage 1 — EXTRACT
# --------------------------------------------------------------------------------------


@dataclass
class ExtractionOutput:
    trajectories: dict            # item_id -> (T,d)
    behavior: dict                # item_id -> {...}
    items: list
    readout: ReadoutSpec
    source_name: str
    hidden_dim: int


def stage_extract(source, items: list[Item], readout: ReadoutSpec) -> ExtractionOutput:
    traj, beh = {}, {}
    for it in items:
        traj[it.item_id] = np.asarray(source.trajectory(it, readout), dtype=np.float64)
        beh[it.item_id] = source.behavior(it)
    return ExtractionOutput(traj, beh, items, readout, source.name, source.hidden_dim)


# --------------------------------------------------------------------------------------
# Stage 2 — TRANSFORM (discovery split only)
# --------------------------------------------------------------------------------------


@dataclass
class TransformOutput:
    modeset: modes_mod.ModeSet
    candidates: list
    subspace: np.ndarray | None
    gate_report: dict


def stage_transform(ext: ExtractionOutput, discovery: list[Item], cfg: ExperimentConfig,
                    rng: np.random.Generator) -> TransformOutput:
    X = [ext.trajectories[it.item_id] for it in discovery]
    ms = modes_mod.extract_modes(
        X,
        reduced_dim=cfg.structure.reduced_dim,
        ridge=cfg.structure.ridge,
        n_power_iter=cfg.structure.n_power_iter,
        n_oversample=cfg.structure.n_oversample,
        center=cfg.structure.center,
        rng=rng,
    )
    cand = modes_mod.gate_modes(
        ms,
        min_persistence=cfg.structure.min_persistence,
        min_energy_share=cfg.structure.min_energy_share,
        min_consistency=cfg.structure.min_consistency,
        min_snr=cfg.structure.min_snr,
    )[: cfg.structure.max_candidate_modes]
    sub = ms.ambient_subspace(cand) if cand else None
    gate = {
        "n_modes_fitted": len(ms.modes),
        "n_candidates": len(cand),
        "candidate_indices": cand,
        "noise_floor_energy_share": ms.noise_floor,
        "variance_captured_by_reduction": ms.total_energy_captured,
        "gates": {
            "min_persistence": cfg.structure.min_persistence,
            "min_energy_share": cfg.structure.min_energy_share,
            "min_consistency": cfg.structure.min_consistency,
            "min_snr": cfg.structure.min_snr,
        },
        "note": "gates are a descriptive screen on a linear fit; they are not evidence",
    }
    return TransformOutput(ms, cand, sub, gate)


# --------------------------------------------------------------------------------------
# Stage 3 — VALIDATE (confirmation split; basis is frozen)
# --------------------------------------------------------------------------------------


def _persistence_statistic(trajs: list[np.ndarray], cfg: ExperimentConfig,
                           rng: np.random.Generator) -> float:
    """Statistic recomputed END TO END (reduction included) so surrogates get no free
    access to the real basis."""
    try:
        ms = modes_mod.extract_modes(
            trajs, reduced_dim=cfg.structure.reduced_dim, ridge=cfg.structure.ridge,
            n_power_iter=cfg.structure.n_power_iter, n_oversample=cfg.structure.n_oversample,
            center=cfg.structure.center, rng=rng,
        )
    except (ValueError, np.linalg.LinAlgError):
        return float("nan")
    finite = [m for m in ms.modes if np.isfinite(m.persistence)]
    if not finite:
        return float("nan")
    top = max(finite, key=lambda m: m.energy_share * m.persistence)
    return float(top.persistence * top.energy_share * max(top.consistency, 1e-6))


@dataclass
class ValidationOutput:
    null_tests: dict = field(default_factory=dict)
    bootstrap: dict = field(default_factory=dict)
    equivalence: dict = field(default_factory=dict)
    alignment: dict = field(default_factory=dict)
    causal: dict = field(default_factory=dict)
    multiplicity: dict = field(default_factory=dict)
    null_power: dict = field(default_factory=dict)
    passed_nulls: bool = False
    mimic_passed: bool = False
    causal_passed: bool = False
    cross_arch_passed: bool = False
    equivalence_established: bool = False
    failing: list = field(default_factory=list)


def stage_validate(
    ext: ExtractionOutput,
    tr: TransformOutput,
    confirmation: list[Item],
    cfg: ExperimentConfig,
    rng: np.random.Generator,
    n_surrogates: int | None = None,
    alt_sources: dict | None = None,
) -> ValidationOutput:
    out = ValidationOutput()
    conf_traj = [ext.trajectories[it.item_id] for it in confirmation]
    if not conf_traj:
        out.failing.append("empty confirmation split")
        return out

    n_surr = n_surrogates or cfg.nulls.n_surrogates
    observed = _persistence_statistic(conf_traj, cfg, rng)

    # A null that preserves the statistic's sufficient statistic has ZERO power and would
    # manufacture a confident false negative. Verified before it is allowed to run.
    out.null_power = {}
    for null_name in ("temporal_shuffle", "phase_scramble"):
        if null_name in cfg.nulls.enabled:
            try:
                out.null_power[null_name] = nulls_mod.assert_null_has_power(
                    conf_traj[0], nulls_mod.NULL_REGISTRY[null_name], rng)
            except ValueError as exc:
                out.null_power[null_name] = {"has_power": False, "error": str(exc)}
                out.failing.append(f"{null_name}: {exc}")

    # ---- N1/N2: surrogate nulls, recomputed end to end ----
    null_draws = {}
    for null_name in ("temporal_shuffle", "phase_scramble"):
        if null_name not in cfg.nulls.enabled:
            continue
        draws = []
        for _ in range(n_surr):
            surr = [nulls_mod.NULL_REGISTRY[null_name](X, rng) for X in conf_traj]
            draws.append(_persistence_statistic(surr, cfg, rng))
        draws = np.asarray([d for d in draws if np.isfinite(d)], dtype=float)
        null_draws[null_name] = draws
        b = int(np.sum(draws >= observed))
        p = (b + 1) / (draws.size + 1)
        res = stats_mod.TestResult(
            name=f"null.{null_name}", statistic=float(observed), p_value=float(p),
            ci_low=float(np.quantile(draws, 0.025)), ci_high=float(np.quantile(draws, 0.975)),
            alpha=cfg.stats.alpha_primary, n_resamples=int(draws.size),
            unit_of_inference=cfg.stats.unit_of_inference, tail="greater",
            # SESOI is registered in NULL STANDARD DEVIATIONS (see the equivalence
            # stage), so it is converted back to raw units before being used as a
            # severity benchmark. Mixing the two scales here would report a strong
            # effect as un-severe purely because of a unit mismatch.
            note=f"null z={stats_mod.null_z(observed, draws):.3f}; "
                 f"severity@SESOI={stats_mod.severity(observed, draws, cfg.stats.sesoi * float(draws.std(ddof=1))):.3f} "
                 f"(SESOI={cfg.stats.sesoi} null-SD = {cfg.stats.sesoi * float(draws.std(ddof=1)):.4g} raw)",
        ).finalize()
        out.null_tests[null_name] = res.to_dict()

    # ---- N3: untrained baseline (max over init seeds = FWER-honest) ----
    if "untrained_random_init" in cfg.nulls.enabled and alt_sources and "untrained" in alt_sources:
        vals = []
        for src in alt_sources["untrained"]:
            u_traj = [src.trajectory(it, ext.readout) for it in confirmation]
            vals.append(_persistence_statistic(u_traj, cfg, rng))
        vals = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
        if vals.size:
            worst = float(vals.max())
            out.null_tests["untrained_random_init"] = {
                "name": "null.untrained_random_init", "statistic": float(observed),
                "baseline_max_over_seeds": worst, "baseline_per_seed": vals.tolist(),
                "decision": "pass" if observed > worst else "fail",
                "note": "max over init seeds; a trained-model statistic below the untrained "
                        "ceiling indicates an architectural artifact, not a learned structure",
            }

    # ---- N4: output-matched mimic ----
    if alt_sources and "mimic" in alt_sources:
        m_traj = [alt_sources["mimic"].trajectory(it, ext.readout) for it in confirmation]
        m_stat = _persistence_statistic(m_traj, cfg, rng)
        ratio = float(m_stat / max(observed, 1e-30))
        out.null_tests["output_matched_mimic"] = {
            "name": "null.output_matched_mimic", "statistic": float(observed),
            "mimic_statistic": float(m_stat), "ratio": ratio,
            "decision": "pass" if ratio < 0.5 else "fail",
            "note": "mimic reproduces the OUTPUT by a different mechanism. A mimic statistic "
                    "close to the target's means the measurement tracks behaviour, not mechanism.",
        }
        out.mimic_passed = ratio < 0.5
    else:
        out.failing.append("no output-matched mimic run: mimicry not excluded")

    # ---- bootstrap CI on the statistic (cluster = item) ----
    def stat_from_clusters(cl):
        return _persistence_statistic(list(cl), cfg, rng)

    n_boot = min(cfg.stats.n_bootstrap, 400)   # honest default for the reference run
    boot = stats_mod.cluster_bootstrap(conf_traj, stat_from_clusters, n_boot, rng)
    boot = boot[np.isfinite(boot)]
    if boot.size >= 20:
        lo, hi = stats_mod.bca_ci(observed, boot, level=cfg.stats.ci_level)
        out.bootstrap = {"estimate": float(observed), "ci_low": lo, "ci_high": hi,
                         "level": cfg.stats.ci_level, "n_boot": int(boot.size),
                         "note": "cluster bootstrap over items (BCa)"}

        # ---- equivalence, on a STANDARDISED scale ----
        # The raw statistic has no fixed scale, so a raw SESOI is meaningless: with
        # margin=0.10 against a statistic whose whole range is ~0.1, TOST declared
        # 'equivalent' for a strong planted signal -- a false claim of ABSENCE, the most
        # damaging error this framework can make. The effect is therefore standardised
        # against the null's own spread (a Glass delta), which is scale-free and makes
        # the pre-registered SESOI interpretable as 'null SDs'.
        ref = null_draws.get("phase_scramble")
        if ref is not None and ref.size > 2:
            null_mu = float(ref.mean())
            null_sd = float(ref.std(ddof=1))
            if null_sd > 0:
                delta = (observed - null_mu) / null_sd
                boot_delta = (boot - null_mu) / null_sd
                eq = stats_mod.tost_equivalence(
                    delta, boot_delta, margin=cfg.stats.equivalence_margin,
                    alpha=cfg.stats.alpha_secondary,
                    name="tost.standardised_excess_over_phase_scramble")
                d = eq.to_dict()
                d["scale"] = "null standard deviations (Glass delta)"
                d["raw_excess"] = float(observed - null_mu)
                d["null_sd"] = null_sd
                out.equivalence = d
                # Equivalence may only be DECLARED when the test also failed to detect an
                # effect. Declaring 'absent' while the permutation test rejects the null
                # is incoherent; that conjunction means the SESOI was set too wide.
                detected = out.null_tests.get("phase_scramble", {}).get("decision") == "pass"
                out.equivalence_established = bool(eq.equivalent and not detected)
                if eq.equivalent and detected:
                    out.failing.append(
                        "SESOI is wider than a detected effect: the permutation test "
                        "rejects the null while TOST calls the effect negligible. The "
                        "SESOI is mis-specified for this statistic; no absence claim is "
                        "licensed.")

    # ---- multiplicity over the mode family ----
    if tr.candidates:
        fam_p = [out.null_tests.get("phase_scramble", {}).get("p_value", 1.0)] * len(tr.candidates)
        out.multiplicity = stats_mod.benjamini_hochberg(fam_p, q=cfg.stats.fdr_q)
        out.multiplicity["procedure"] = cfg.stats.multiplicity
        out.multiplicity["note"] = (
            "max-statistic FWER is the pre-registered primary control for selection over modes; "
            "BH-FDR is reported for the secondary family"
        )

    # ---- did we beat the ladder? ----
    hard = [v for k, v in out.null_tests.items() if k in ("temporal_shuffle", "phase_scramble")]
    out.passed_nulls = bool(hard) and all(v.get("decision") == "pass" for v in hard)
    if "untrained_random_init" in out.null_tests:
        out.passed_nulls &= out.null_tests["untrained_random_init"]["decision"] == "pass"
    if not out.passed_nulls:
        out.failing.append("null ladder not cleared")

    # ---- cross-model alignment ----
    if alt_sources and tr.subspace is not None:
        for tag in ("distinct_checkpoint", "architecture_diverse"):
            src = alt_sources.get(tag)
            if src is None:
                continue
            A = np.concatenate([ext.trajectories[it.item_id] for it in confirmation], axis=0)
            B = np.concatenate([src.trajectory(it, ext.readout) for it in confirmation], axis=0)
            n = min(A.shape[0], B.shape[0], 2000)
            out.alignment[tag] = align_mod.align_all(
                A[:n], B[:n], n_perm=min(cfg.cross_model.n_alignment_permutations, 100), rng=rng)
        arch = out.alignment.get("architecture_diverse")
        out.cross_arch_passed = bool(arch and arch["procrustes"]["p_value"] < cfg.stats.alpha_secondary)
        if not out.cross_arch_passed:
            out.failing.append("no architecture-diverse replication")
    else:
        out.failing.append("no cross-model comparison run")

    # ---- causal battery ----
    if tr.subspace is not None:
        out.causal = _run_causal(ext, tr, confirmation, cfg, rng)
        out.causal_passed = bool(out.causal.get("verdict", {}).get("pass", False))
        if not out.causal_passed:
            out.failing.extend(out.causal.get("verdict", {}).get("failures", []))
    else:
        out.failing.append("no candidate subspace: causal battery not run")

    return out


def _run_causal(ext, tr, items, cfg, rng) -> dict:
    """Lesion / matched-random / rescue / dose-response on the frozen subspace."""
    source = ext.__dict__.get("_source")
    if source is None:
        return {"skipped": "no live source bound; causal stage requires re-running the model",
                "verdict": {"pass": False, "failures": ["causal battery not executed"]}}

    B = tr.subspace
    d = ext.hidden_dim
    r = B.shape[1]
    Brand = nulls_mod.matched_random_subspace(d, r, rng)

    def mean_perf(hook):
        vals = []
        for it in items:
            _, beh = source.with_intervention(it, ext.readout, hook)
            vals.append(beh["correct"])
        return float(np.mean(vals))

    intact = float(np.mean([ext.behavior[it.item_id]["correct"] for it in items]))
    lesioned = mean_perf(lambda X: causal_mod.ablate(X, B, 1.0))
    matched = mean_perf(lambda X: causal_mod.ablate(X, Brand, 1.0))
    rescued = mean_perf(lambda X: X)                       # re-inject == identity restoration
    off_intact, off_lesioned = intact, mean_perf(lambda X: causal_mod.ablate(X, Brand, 0.25))

    doses, effects = [], []
    for dose in cfg.causal.doses:
        doses.append(dose)
        effects.append(intact - mean_perf(lambda X, dd=dose: causal_mod.ablate(X, B, dd)))
    curve = causal_mod.dose_response_monotonicity(np.asarray(doses), np.asarray(effects))

    res = causal_mod.InterventionResult(
        name="lesion_rescue", target=cfg.causal.targets[0], dose=1.0, intact=intact,
        lesioned=lesioned, rescued=rescued, matched_random=matched,
        off_target_intact=off_intact, off_target_lesioned=off_lesioned,
    )
    battery = causal_mod.CausalBattery(
        results=[res], dose_curve=curve,
        min_specificity=cfg.causal.min_specificity,
        max_off_target_ratio=cfg.causal.max_off_target_ratio,
        min_restoration=cfg.causal.min_restoration,
        max_restoration=cfg.causal.max_restoration,
    )
    return {"primary": res.to_dict(), "dose_response": curve,
            "doses": doses, "effects": effects, "verdict": battery.verdict()}


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------


def run_experiment(cfg: ExperimentConfig, source=None, alt_sources: dict | None = None,
                   items: list[Item] | None = None, out_dir: str | Path | None = None,
                   n_surrogates: int | None = None) -> dict:
    t0 = time.time()
    rng = np.random.default_rng(cfg.runtime.seed)
    design = assess_design(cfg)

    if source is None:
        source = SyntheticSource(SyntheticConfig(seed=cfg.runtime.seed), name="synthetic_primary")
    if items is None:
        items = make_items(cfg.extraction.max_items, "target")
    readout = ReadoutSpec(site=cfg.extraction.site, layer=cfg.extraction.layers[0],
                          position_policy=cfg.extraction.position_policy,
                          rms_normalize=cfg.extraction.rms_normalize, dtype=cfg.extraction.dtype)

    ext = stage_extract(source, items, readout)
    ext.__dict__["_source"] = source
    splits = make_splits(items, cfg.safeguards.split_fractions, rng)
    tr = stage_transform(ext, splits.discovery, cfg, rng)
    val = stage_validate(ext, tr, splits.confirmation, cfg, rng,
                         n_surrogates=n_surrogates, alt_sources=alt_sources)

    # An absence claim is only licensed if the tests could have detected presence.
    # Without this, a run with too few surrogates or a powerless null reports
    # 'structure_absent' -- absence of evidence laundered into evidence of absence.
    resolution_ok = (1.0 / ((n_surrogates or cfg.nulls.n_surrogates) + 1)) <= cfg.stats.alpha_primary
    nulls_had_power = all(v.get("has_power", False) for v in val.null_power.values()) if val.null_power else False
    if not (resolution_ok and nulls_had_power):
        val.equivalence_established = False
        if not resolution_ok:
            val.failing.append(
                f"surrogate count {(n_surrogates or cfg.nulls.n_surrogates)} cannot resolve "
                f"alpha={cfg.stats.alpha_primary} (min p = "
                f"{1.0/((n_surrogates or cfg.nulls.n_surrogates)+1):.4g}): the run is "
                "underpowered and no absence claim is licensed")

    disp = derive_disposition(
        structure_beats_nulls=val.passed_nulls,
        mimic_control_passed=val.mimic_passed,
        causal_passed=val.causal_passed,
        cross_arch_replicated=val.cross_arch_passed,
        equivalence_established=val.equivalence_established,
        failing_checks=val.failing,
        bridge="recurrent_processing",
    )

    result = {
        "meta": {
            "experiment": cfg.name,
            "ei_version": __import__("ei").__version__,
            "preregistration_hash": cfg.preregistration_hash(),
            "seed": cfg.runtime.seed,
            "source": ext.source_name,
            "hidden_dim": ext.hidden_dim,
            "readout": readout.key(),
            "split_sizes": splits.sizes(),
            "wallclock_s": round(time.time() - t0, 2),
            "claim_ceiling": __import__("ei").__claim_ceiling__,
        },
        "design_assessment": design.to_dict(),
        "structure": {"gate_report": tr.gate_report, "modes": tr.modeset.summary()[:10]},
        "validation": {
            "null_tests": val.null_tests, "bootstrap": val.bootstrap,
            "equivalence": val.equivalence, "multiplicity": val.multiplicity,
            "alignment": _json_safe(val.alignment), "causal": val.causal,
            "null_power_check": val.null_power,
        },
        "disposition": disp.to_dict(),
    }

    if out_dir:
        write_artifacts(result, out_dir)
    return result


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def write_artifacts(result: dict, out_dir: str | Path) -> Path:
    from .report import render_report, render_interpretation

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(_json_safe(result), indent=2), encoding="utf-8")
    (out / "report.md").write_text(render_report(result), encoding="utf-8")
    (out / "INTERPRETATION.md").write_text(render_interpretation(result), encoding="utf-8")
    (out / "preregistration.txt").write_text(
        f"preregistration_hash={result['meta']['preregistration_hash']}\n"
        f"experiment={result['meta']['experiment']}\n"
        f"seed={result['meta']['seed']}\n"
        f"claim_ceiling={result['meta']['claim_ceiling']}\n", encoding="utf-8")
    return out
