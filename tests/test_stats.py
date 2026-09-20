"""
Statistical machinery tests, including FALSE-POSITIVE RATE calibration.

The critical test here is `test_permutation_false_positive_rate_is_calibrated`: under a
true null, the test must reject at approximately alpha.  A framework that claims p < 0.005
while actually rejecting 30% of the time under the null is worse than no framework, and
that failure mode is invisible without this check.
"""

import numpy as np
import pytest

import ei.stats as S
from ei import nulls as N


# --------------------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------------------


def test_permutation_false_positive_rate_is_calibrated():
    """Under a true null, rejection rate at alpha=0.05 must be near 0.05."""
    rng = np.random.default_rng(0)
    rejects, trials, alpha = 0, 300, 0.05
    for _ in range(trials):
        data = rng.standard_normal(40)
        obs = float(data.mean())
        res = S.permutation_test(
            obs,
            lambda r, d=data: float((d * r.choice([-1.0, 1.0], size=d.size)).mean()),
            n_perm=200, rng=rng, tail="two-sided", alpha=alpha, name="calib",
        )
        rejects += res.p_value < alpha
    rate = rejects / trials
    assert 0.01 < rate < 0.11, f"false-positive rate {rate:.3f} is not near alpha={alpha}"


def test_permutation_detects_real_effect():
    rng = np.random.default_rng(1)
    data = rng.standard_normal(60) + 0.9
    res = S.permutation_test(
        float(data.mean()),
        lambda r, d=data: float((d * r.choice([-1.0, 1.0], size=d.size)).mean()),
        n_perm=500, rng=rng, tail="greater", alpha=0.05,
    )
    assert res.p_value < 0.05 and res.decision == "pass"


def test_p_value_never_zero():
    """(b+1)/(B+1) keeps p strictly positive: an unbounded claim is never licensed."""
    rng = np.random.default_rng(2)
    res = S.permutation_test(1e9, lambda r: float(r.standard_normal()), n_perm=100, rng=rng)
    assert res.p_value == pytest.approx(1 / 101)
    assert res.p_value > 0


def test_permutation_count_bounds_resolution():
    rng = np.random.default_rng(3)
    res = S.permutation_test(50.0, lambda r: float(r.standard_normal()), n_perm=99, rng=rng,
                             alpha=0.001)
    assert res.p_value > 0.001
    assert res.decision == "fail", "a p floored by resolution must not be reported as a pass"


# --------------------------------------------------------------------------------------
# Pseudoreplication guard
# --------------------------------------------------------------------------------------


def test_token_level_unit_is_rejected():
    with pytest.raises(S.PseudoreplicationError):
        S.check_unit_of_inference("token")
    with pytest.raises(S.PseudoreplicationError):
        S.check_unit_of_inference("timestep")


def test_registered_units_accepted():
    for u in ("independent_semantic_items", "model_checkpoints", "interaction_episodes"):
        S.check_unit_of_inference(u)


def test_permutation_refuses_token_unit():
    rng = np.random.default_rng(0)
    with pytest.raises(S.PseudoreplicationError):
        S.permutation_test(1.0, lambda r: 0.0, n_perm=10, rng=rng, unit="token")


# --------------------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------------------


def test_bca_ci_covers_truth():
    rng = np.random.default_rng(4)
    covered = 0
    for _ in range(120):
        data = list(rng.normal(5.0, 1.0, size=50))
        boot = S.cluster_bootstrap(data, lambda c: float(np.mean(c)), 300, rng)
        jack = S.jackknife(data, lambda c: float(np.mean(c)))
        lo, hi = S.bca_ci(float(np.mean(data)), boot, jack, level=0.95)
        covered += lo <= 5.0 <= hi
    rate = covered / 120
    assert 0.86 < rate < 1.0, f"95% CI covered {rate:.2f} of the time"


def test_cluster_bootstrap_requires_enough_clusters():
    rng = np.random.default_rng(5)
    with pytest.raises(ValueError):
        S.cluster_bootstrap([1, 2], lambda c: float(np.mean(c)), 100, rng)


# --------------------------------------------------------------------------------------
# Equivalence
# --------------------------------------------------------------------------------------


def test_tost_declares_equivalence_for_null_effect():
    rng = np.random.default_rng(6)
    boot = rng.normal(0.0, 0.01, size=2000)
    res = S.tost_equivalence(0.0, boot, margin=0.10)
    assert res.equivalent


def test_tost_rejects_equivalence_for_real_effect():
    rng = np.random.default_rng(7)
    boot = rng.normal(0.5, 0.05, size=2000)
    assert not S.tost_equivalence(0.5, boot, margin=0.10).equivalent


def test_tost_rejects_equivalence_when_underpowered():
    """A wide interval must NOT be read as equivalence: that is the classic error."""
    rng = np.random.default_rng(8)
    boot = rng.normal(0.0, 0.5, size=2000)
    assert not S.tost_equivalence(0.0, boot, margin=0.10).equivalent


def test_margin_must_be_positive():
    with pytest.raises(ValueError):
        S.tost_equivalence(0.0, np.zeros(100), margin=0.0)


# --------------------------------------------------------------------------------------
# Multiplicity
# --------------------------------------------------------------------------------------


def test_bh_controls_fdr_under_global_null():
    rng = np.random.default_rng(9)
    false_disc = 0
    for _ in range(200):
        p = rng.uniform(0, 1, size=20)
        false_disc += S.benjamini_hochberg(p, q=0.05)["n_reject"] > 0
    assert false_disc / 200 < 0.12


def test_bh_finds_true_effects():
    p = [1e-6] * 5 + list(np.linspace(0.2, 0.9, 15))
    assert S.benjamini_hochberg(p, q=0.05)["n_reject"] >= 5


def test_max_statistic_threshold_is_conservative():
    rng = np.random.default_rng(10)
    null_max = np.max(rng.standard_normal((2000, 20)), axis=1)
    thr = S.max_statistic_threshold(null_max, alpha=0.05)
    per_mode = np.quantile(rng.standard_normal(2000), 0.95)
    assert thr > per_mode, "family-wise threshold must exceed the per-test threshold"


# --------------------------------------------------------------------------------------
# Surrogate nulls
# --------------------------------------------------------------------------------------


def test_phase_scramble_preserves_power_spectrum():
    rng = np.random.default_rng(11)
    X = np.cumsum(rng.standard_normal((256, 5)), axis=0)
    Xs = N.phase_scramble(X, rng)
    p0 = np.abs(np.fft.rfft(X - X.mean(0), axis=0))
    p1 = np.abs(np.fft.rfft(Xs - Xs.mean(0), axis=0))
    assert np.allclose(p0, p1, rtol=1e-6, atol=1e-8)


def test_phase_scramble_destroys_temporal_arrangement():
    rng = np.random.default_rng(12)
    X = np.cumsum(rng.standard_normal((256, 3)), axis=0)
    Xs = N.phase_scramble(X, rng)
    assert not np.allclose(X, Xs)


def test_iaaft_preserves_value_distribution():
    rng = np.random.default_rng(13)
    X = rng.standard_exponential((128, 3))          # heavy-tailed, like real activations
    Xs = N.iaaft(X, rng, n_iter=50)
    assert np.allclose(np.sort(X, axis=0), np.sort(Xs, axis=0), atol=1e-8)


def test_temporal_shuffle_preserves_marginals():
    rng = np.random.default_rng(14)
    X = rng.standard_normal((100, 4))
    Xs = N.temporal_shuffle(X, rng)
    assert np.allclose(np.sort(X, axis=0), np.sort(Xs, axis=0))


def test_joint_shuffle_preserves_cross_covariance():
    rng = np.random.default_rng(15)
    base = rng.standard_normal((200, 1))
    X = np.hstack([base, base * 0.9 + 0.1 * rng.standard_normal((200, 1))])
    Xs = N.temporal_shuffle(X, rng, jointly=True)
    assert abs(np.corrcoef(X.T)[0, 1] - np.corrcoef(Xs.T)[0, 1]) < 1e-9


def test_independent_shuffle_destroys_cross_covariance():
    rng = np.random.default_rng(16)
    base = rng.standard_normal((400, 1))
    X = np.hstack([base, base * 0.95 + 0.05 * rng.standard_normal((400, 1))])
    Xs = N.temporal_shuffle(X, rng, jointly=False)
    assert abs(np.corrcoef(Xs.T)[0, 1]) < abs(np.corrcoef(X.T)[0, 1]) - 0.5


def test_matched_random_subspace_is_orthonormal():
    rng = np.random.default_rng(17)
    Q = N.matched_random_subspace(64, 5, rng)
    assert np.allclose(Q.T @ Q, np.eye(5), atol=1e-8)


def test_mimic_donor_rejected_when_behaviour_unmatched():
    spec = N.MimicSpec(donor="persona_prompted")
    ok, msg = spec.admissible(observed_agreement=0.40, observed_js=0.05)
    assert not ok and "confounded" in msg


def test_mimic_donor_admitted_when_matched():
    spec = N.MimicSpec(donor="distilled")
    ok, _ = spec.admissible(observed_agreement=0.93, observed_js=0.03)
    assert ok


def test_untrained_spec_requires_matching():
    bad = N.UntrainedBaselineSpec(architecture="x", n_seeds=1, match_tokenizer=False)
    problems = bad.validate()
    assert len(problems) >= 2


# --------------------------------------------------------------------------------------
# Severity
# --------------------------------------------------------------------------------------


def test_severity_penalises_trivial_effects():
    rng = np.random.default_rng(18)
    null = rng.normal(0, 1, 5000)
    assert S.severity(0.05, null, benchmark=1.0) < 0.3
    assert S.severity(4.0, null, benchmark=1.0) > 0.9
