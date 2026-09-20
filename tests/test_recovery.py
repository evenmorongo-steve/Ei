"""
Ground-truth recovery tests — the instrument's calibration check.

These are not smoke tests.  Each asserts a property that, if it silently broke, would
make every downstream conclusion wrong while the pipeline kept emitting confident output.
Several of them encode bugs that were actually present during development and were found
only because the ground truth was known:

  * conjugate eigenvalue pairs were double-counted, halving recovered subspaces
    (overlap 0.996 for top-1 collapsing to 0.500 for top-2)
  * modal energies computed by inverting an ill-conditioned eigenvector matrix got WORSE
    with more data (0.46 -> 0.02 from 40 to 400 items)
  * the synthetic generator planted decaying transients instead of driven modes, so there
    was no stationary structure to find at all
"""

import numpy as np
import pytest

import ei.modes as M
from ei.extract import SyntheticConfig, SyntheticSource, make_items


def build(amp=3.0, n=60, T=100, d=96, ar1=0.85, modes=2, seed=0):
    cfg = SyntheticConfig(hidden_dim=d, n_steps=T, planted_modes=modes, ar1_noise=ar1,
                          planted_amplitude=amp, seed=seed)
    src = SyntheticSource(cfg)
    return src, [src.trajectory(it) for it in make_items(n, "target")]


# --------------------------------------------------------------------------------------
# Positive control
# --------------------------------------------------------------------------------------


def test_recovers_planted_subspace_above_chance():
    src, traj = build(amp=3.0)
    ms = M.extract_modes(traj, reduced_dim=16, rng=np.random.default_rng(1))
    sel = ms.top(2, by="salience")
    B = ms.ambient_subspace([m.index for m in sel])
    ov = M.subspace_overlap(B, src.planted_subspace())
    chance = src.planted_dim / src.hidden_dim
    assert ov > 0.85, f"overlap {ov:.3f} too low (chance {chance:.3f})"
    assert ov > 20 * chance


def test_recovered_dimension_matches_ground_truth():
    """A 1-D real mode plus a 2-D oscillatory plane is 3 real dimensions, not 4."""
    src, traj = build(amp=3.0)
    ms = M.extract_modes(traj, reduced_dim=16, rng=np.random.default_rng(1))
    B = ms.ambient_subspace([m.index for m in ms.top(2, by="salience")])
    assert B.shape[1] == src.planted_dim == 3


def test_conjugate_pairs_deduplicated():
    """Regression: eig() returns lambda and conj(lambda) spanning ONE real plane."""
    src, traj = build(amp=3.0)
    ms = M.extract_modes(traj, reduced_dim=16, rng=np.random.default_rng(1))
    assert len(ms.modes) < 16, "no deduplication happened"
    for m in ms.modes:
        assert m.frequency >= 0, "negative-frequency duplicate survived"
        assert m.real_dims == (2 if m.is_oscillatory else 1)
    # selecting the single top mode must recover a full plane, not half of one
    top = ms.top(1, by="salience")[0]
    B = ms.ambient_subspace([top.index])
    assert B.shape[1] == top.real_dims


def test_energy_estimate_does_not_degrade_with_more_data():
    """Regression: W^{-1}-based modal amplitudes got worse as n grew."""
    ovs = []
    for n in (40, 200):
        src, traj = build(amp=3.0, n=n)
        ms = M.extract_modes(traj, reduced_dim=16, rng=np.random.default_rng(1))
        B = ms.ambient_subspace([m.index for m in ms.top(2, by="salience")])
        ovs.append(M.subspace_overlap(B, src.planted_subspace()))
    assert ovs[1] > 0.80, f"recovery collapsed with more data: {ovs}"
    assert ovs[1] > ovs[0] - 0.15, f"recovery degraded with more data: {ovs}"


def test_recovery_improves_with_amplitude():
    ovs = []
    for amp in (0.5, 1.5, 3.0):
        src, traj = build(amp=amp)
        ms = M.extract_modes(traj, reduced_dim=16, rng=np.random.default_rng(1))
        B = ms.ambient_subspace([m.index for m in ms.top(2, by="salience")])
        ovs.append(M.subspace_overlap(B, src.planted_subspace()))
    assert ovs[0] < ovs[-1], f"recovery not monotone in SNR: {ovs}"
    assert ovs[-1] > 0.85


# --------------------------------------------------------------------------------------
# Documented bias
# --------------------------------------------------------------------------------------


def test_persistence_is_downward_biased_and_documented():
    """The estimator IS biased; the test pins the direction so it cannot silently flip."""
    cfg = SyntheticConfig(hidden_dim=96, n_steps=120, planted_modes=1,
                          planted_persistence=(0.97,), planted_frequency=(0.0,),
                          ar1_noise=0.85, planted_amplitude=3.0, seed=0)
    src = SyntheticSource(cfg)
    traj = [src.trajectory(it) for it in make_items(60, "target")]
    ms = M.extract_modes(traj, reduced_dim=16, rng=np.random.default_rng(1))
    rho = ms.top(1, by="salience")[0].persistence
    assert rho < 0.97, "attenuation bias vanished -- update the documented bias table"
    assert rho > 0.90, f"bias far larger than documented: {rho}"


def test_persistence_unbiased_without_background_noise():
    cfg = SyntheticConfig(hidden_dim=96, n_steps=200, planted_modes=1,
                          planted_persistence=(0.97,), planted_frequency=(0.0,),
                          ar1_noise=0.0, noise_sigma=0.01, planted_amplitude=3.0, seed=0)
    src = SyntheticSource(cfg)
    traj = [src.trajectory(it) for it in make_items(60, "target")]
    ms = M.extract_modes(traj, reduced_dim=16, rng=np.random.default_rng(1))
    rho = ms.top(1, by="salience")[0].persistence
    assert abs(rho - 0.97) < 0.02, f"clean-data estimate {rho} should be near 0.97"


def test_bias_warning_fires_at_low_snr():
    assert M.persistence_bias_warning(1.0) is not None
    assert M.persistence_bias_warning(5.0) is not None
    assert M.persistence_bias_warning(50.0) is None


# --------------------------------------------------------------------------------------
# Negative control — THE important one
# --------------------------------------------------------------------------------------


def test_gates_alone_do_not_reject_coloured_noise():
    """Pure AR(1) with NOTHING planted still passes the descriptive gates.

    This is the single most important test in the suite. It is the empirical justification
    for making the phase-scramble null mandatory: a persistence screen on its own will
    happily report 'persistent cognitive structure' in noise.
    """
    cfg = SyntheticConfig(hidden_dim=96, n_steps=100, planted_modes=0, planted_amplitude=0.0,
                          ar1_noise=0.85, seed=7)
    src = SyntheticSource(cfg)
    traj = [src.trajectory(it) for it in make_items(60, "target")]
    ms = M.extract_modes(traj, reduced_dim=16, rng=np.random.default_rng(2))
    cand = M.gate_modes(ms, min_persistence=0.80, min_energy_share=0.005,
                        min_consistency=0.20, min_snr=1.5)
    assert len(cand) > 0, (
        "if this ever passes, the gates got strict enough that the mandatory-null argument "
        "needs restating -- do not just delete the test"
    )


def test_white_noise_yields_low_persistence():
    cfg = SyntheticConfig(hidden_dim=96, n_steps=100, planted_modes=0, planted_amplitude=0.0,
                          ar1_noise=0.0, seed=3)
    src = SyntheticSource(cfg)
    traj = [src.trajectory(it) for it in make_items(60, "target")]
    ms = M.extract_modes(traj, reduced_dim=16, rng=np.random.default_rng(2))
    assert max(m.persistence for m in ms.modes) < 0.70


# --------------------------------------------------------------------------------------
# Numerical hygiene
# --------------------------------------------------------------------------------------


def test_randomized_svd_matches_exact_svd():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((300, 60)) @ rng.standard_normal((60, 120))
    U, s, Vt = M.randomized_svd(A, k=10, rng=rng, n_power_iter=3)
    s_exact = np.linalg.svd(A, compute_uv=False)[:10]
    assert np.allclose(s, s_exact, rtol=0.05), f"{s[:4]} vs {s_exact[:4]}"


def test_randomized_svd_needs_power_iterations():
    """Without power iterations the sketch is materially worse on a slow spectrum."""
    rng = np.random.default_rng(0)
    U = np.linalg.qr(rng.standard_normal((200, 50)))[0]
    V = np.linalg.qr(rng.standard_normal((80, 50)))[0]
    A = U @ np.diag(1.0 / np.arange(1, 51) ** 0.5) @ V.T
    exact = np.linalg.svd(A, compute_uv=False)[:10]
    e0 = np.abs(M.randomized_svd(A, 10, rng=np.random.default_rng(1), n_power_iter=0)[1] - exact).sum()
    e3 = np.abs(M.randomized_svd(A, 10, rng=np.random.default_rng(1), n_power_iter=3)[1] - exact).sum()
    assert e3 <= e0


def test_subspace_overlap_bounds():
    rng = np.random.default_rng(0)
    Q = np.linalg.qr(rng.standard_normal((50, 5)))[0]
    assert M.subspace_overlap(Q, Q) == pytest.approx(1.0, abs=1e-8)
    full = np.linalg.qr(rng.standard_normal((50, 10)))[0]
    assert M.subspace_overlap(full[:, :5], full[:, 5:]) == pytest.approx(0.0, abs=1e-8)


def test_propagator_recovers_known_matrix():
    rng = np.random.default_rng(0)
    k = 6
    A_true = np.linalg.qr(rng.standard_normal((k, k)))[0] * 0.9
    Z = [rng.standard_normal(k)]
    for _ in range(4000):
        Z.append(A_true @ Z[-1] + 0.01 * rng.standard_normal(k))
    Z = np.array(Z)
    A_hat = M.fit_propagator([(Z[:-1], Z[1:])], ridge=1e-9)
    assert np.abs(A_hat - A_true).max() < 0.05
