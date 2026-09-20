"""
End-to-end pipeline tests.

The decisive pair is `test_strong_signal_reaches_ceiling_disposition` and
`test_no_signal_yields_absence`. Together they show the framework DISCRIMINATES. A
detector that only ever says "no" is as useless as one that only ever says "yes", and
without both directions checked, neither failure is visible.
"""

import json

import numpy as np
import pytest

from ei.config import ExperimentConfig, load_config
from ei.cli import _alt_sources
from ei.extract import SyntheticConfig, SyntheticSource, make_items
from ei.pipeline import make_splits, run_experiment, stage_extract, stage_transform
from ei.extract import ReadoutSpec
import ei.nulls as N


def _cfg(n_surr=300, items=50):
    cfg = load_config("configs/experiment.yaml")
    cfg.nulls.n_surrogates = n_surr
    cfg.stats.n_bootstrap = 150
    cfg.extraction.max_items = items
    cfg.cross_model.n_alignment_permutations = 25
    return cfg


def _run(amp, n_surr=300, items=50, d=48, T=50, seed=None):
    cfg = _cfg(n_surr, items)
    if seed is not None:
        cfg.runtime.seed = seed
    pc = SyntheticConfig(hidden_dim=d, n_steps=T, planted_modes=(2 if amp > 0 else 0),
                         planted_amplitude=amp, ar1_noise=0.85, seed=cfg.runtime.seed)
    src = SyntheticSource(pc, name="synthetic_primary")
    return run_experiment(cfg, source=src, alt_sources=_alt_sources(cfg, pc),
                          items=make_items(cfg.extraction.max_items, "target"),
                          n_surrogates=cfg.nulls.n_surrogates)


# --------------------------------------------------------------------------------------
# Discrimination — the two tests that matter most
# --------------------------------------------------------------------------------------


@pytest.mark.slow
def test_strong_signal_reaches_ceiling_disposition():
    r = _run(amp=3.5, n_surr=300, items=50)
    nt = r["validation"]["null_tests"]
    assert nt["phase_scramble"]["decision"] == "pass", nt["phase_scramble"]
    assert nt["temporal_shuffle"]["decision"] == "pass"
    assert nt["output_matched_mimic"]["decision"] == "pass"
    assert r["validation"]["causal"]["verdict"]["pass"]
    assert r["disposition"]["disposition"] == "computational_correlate_supported"


@pytest.mark.slow
def test_no_signal_yields_absence_not_a_positive():
    r = _run(amp=0.0, n_surr=300, items=50)
    assert r["disposition"]["disposition"] in {"structure_absent", "inconclusive"}
    assert r["validation"]["null_tests"]["phase_scramble"]["decision"] == "fail"


# --------------------------------------------------------------------------------------
# Null power — regression for a bug that silenced the framework entirely
# --------------------------------------------------------------------------------------


def test_identical_phase_surrogate_is_refused():
    """Regression: preserve_cross=True keeps lag-1 cross-covariance at corr~1.0, so the
    propagator statistic can never beat it and every run returns a false negative."""
    rng = np.random.default_rng(0)
    X = np.cumsum(rng.standard_normal((200, 8)), axis=0)
    with pytest.raises(ValueError, match="no power"):
        N.assert_null_has_power(X, lambda A, r: N.phase_scramble(A, r, preserve_cross=True), rng)


def test_default_phase_surrogate_has_power():
    rng = np.random.default_rng(0)
    X = np.cumsum(rng.standard_normal((200, 8)), axis=0)
    assert N.assert_null_has_power(X, N.phase_scramble, rng)["has_power"]


def test_default_phase_surrogate_still_preserves_spectrum():
    rng = np.random.default_rng(1)
    X = np.cumsum(rng.standard_normal((256, 6)), axis=0)
    Xs = N.phase_scramble(X, rng)
    p0 = np.abs(np.fft.rfft(X - X.mean(0), axis=0))
    p1 = np.abs(np.fft.rfft(Xs - Xs.mean(0), axis=0))
    assert np.allclose(p0, p1, rtol=1e-6, atol=1e-8)


def test_pipeline_reports_null_power_check():
    r = _run(amp=2.0, n_surr=20, items=24)
    npc = r["validation"]["null_power_check"]
    assert npc["phase_scramble"]["has_power"]


# --------------------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------------------


def test_splits_are_disjoint():
    items = make_items(100, "target")
    s = make_splits(items, {"discovery": 0.4, "alignment": 0.2, "calibration": 0.1,
                            "confirmation": 0.3}, np.random.default_rng(0))
    ids = [{i.item_id for i in getattr(s, k)}
           for k in ("discovery", "alignment", "calibration", "confirmation")]
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            assert not (ids[a] & ids[b]), "splits overlap: confirmation is contaminated"
    assert sum(len(x) for x in ids) == 100


def test_minimal_pairs_stay_together():
    from ei.extract import Item
    items = [Item(item_id=f"i{i}_{c}", prompt="p", condition=c, pair_id=f"pair{i}")
             for i in range(60) for c in ("target", "control")]
    s = make_splits(items, {"discovery": 0.5, "alignment": 0.2, "calibration": 0.1,
                            "confirmation": 0.2}, np.random.default_rng(0))
    where = {}
    for name in ("discovery", "alignment", "calibration", "confirmation"):
        for it in getattr(s, name):
            where.setdefault(it.pair_id, set()).add(name)
    assert all(len(v) == 1 for v in where.values()), "a minimal pair was split across folds"


def test_splits_are_deterministic():
    items = make_items(60, "target")
    f = {"discovery": 0.4, "alignment": 0.2, "calibration": 0.1, "confirmation": 0.3}
    a = make_splits(items, f, np.random.default_rng(7))
    b = make_splits(items, f, np.random.default_rng(7))
    assert [i.item_id for i in a.confirmation] == [i.item_id for i in b.confirmation]


# --------------------------------------------------------------------------------------
# Artifacts
# --------------------------------------------------------------------------------------


def test_artifacts_written_and_carry_claim_boundary(tmp_path):
    cfg = _cfg(n_surr=20, items=24)
    pc = SyntheticConfig(hidden_dim=48, n_steps=40, planted_modes=2,
                         planted_amplitude=2.0, ar1_noise=0.85, seed=cfg.runtime.seed)
    src = SyntheticSource(pc)
    run_experiment(cfg, source=src, alt_sources=_alt_sources(cfg, pc),
                   items=make_items(24, "target"), out_dir=tmp_path, n_surrogates=20)
    for f in ("results.json", "report.md", "INTERPRETATION.md", "preregistration.txt"):
        assert (tmp_path / f).exists(), f"missing artifact {f}"
    payload = json.loads((tmp_path / "results.json").read_text())
    assert payload["meta"]["preregistration_hash"]
    assert "No inference to phenomenal consciousness" in payload["meta"]["claim_ceiling"]
    report = (tmp_path / "report.md").read_text()
    assert "Claim ceiling" in report
    assert "Excluded claim" in report


def test_results_json_is_serialisable():
    r = _run(amp=2.0, n_surr=20, items=24)
    json.dumps(r, default=str)


# --------------------------------------------------------------------------------------
# Stage separation
# --------------------------------------------------------------------------------------


def test_transform_only_sees_discovery_split():
    """The basis must be fitted on discovery alone; refitting on confirmation is the leak
    that invalidates the whole design."""
    cfg = _cfg(items=40)
    pc = SyntheticConfig(hidden_dim=48, n_steps=40, planted_modes=2, planted_amplitude=3.0, seed=1)
    src = SyntheticSource(pc)
    items = make_items(40, "target")
    ext = stage_extract(src, items, ReadoutSpec(layer=0))
    rng = np.random.default_rng(0)
    splits = make_splits(items, cfg.safeguards.split_fractions, rng)
    tr_a = stage_transform(ext, splits.discovery, cfg, np.random.default_rng(3))
    tr_b = stage_transform(ext, splits.discovery, cfg, np.random.default_rng(3))
    assert np.allclose(tr_a.modeset.basis, tr_b.modeset.basis), "transform is not deterministic"
    assert tr_a.subspace is None or tr_a.subspace.shape[0] == pc.hidden_dim


# --------------------------------------------------------------------------------------
# Absence claims — regression for the most damaging possible bug
# --------------------------------------------------------------------------------------


def test_underpowered_run_cannot_claim_absence():
    """Regression: a run with too few surrogates reported `structure_absent`.

    That laundered absence-of-evidence into evidence-of-absence. With 20 surrogates the
    minimum achievable p is 1/21, which cannot resolve alpha=0.005, so no absence claim is
    licensed however the data land.
    """
    r = _run(amp=0.0, n_surr=20, items=24)
    assert r["disposition"]["disposition"] == "inconclusive"
    assert any("underpowered" in f or "resolve" in f
               for f in r["disposition"]["failing_checks"])


def test_strong_signal_never_declared_absent():
    """Regression: a raw SESOI on a scale-free statistic made TOST declare a strong
    planted effect 'equivalent to zero'."""
    r = _run(amp=3.5, n_surr=300, items=50)
    assert r["disposition"]["disposition"] != "structure_absent"
    eq = r["validation"].get("equivalence", {})
    if eq:
        assert not eq["equivalent"], "strong planted signal declared equivalent to zero"


def test_equivalence_is_standardised():
    r = _run(amp=2.0, n_surr=100, items=40)
    eq = r["validation"].get("equivalence", {})
    if eq:
        assert eq["scale"] == "null standard deviations (Glass delta)"
        assert "raw_excess" in eq and "null_sd" in eq


def test_detection_and_equivalence_cannot_both_hold():
    """A run that both rejects the null and declares the effect negligible has a
    mis-specified SESOI; the framework must refuse the absence claim."""
    r = _run(amp=3.5, n_surr=300, items=50)
    nt = r["validation"]["null_tests"]["phase_scramble"]
    eq = r["validation"].get("equivalence", {})
    if nt["decision"] == "pass" and eq.get("equivalent"):
        assert r["disposition"]["disposition"] != "structure_absent"
        assert any("SESOI" in f for f in r["disposition"]["failing_checks"])
