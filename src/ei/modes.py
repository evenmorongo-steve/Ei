"""
Structural analysis: persistent-mode extraction from activation trajectories.

THE MATHEMATICAL OBJECT
-----------------------
Let X_i in R^{T x d} be the residual-stream trajectory for item i at a chosen read-out
site (layer L, stream position policy).  We do NOT look for "patterns" in the loose
sense; we estimate the spectrum of a *linear propagator* fitted to the trajectory
inside a reduced subspace, and call its eigenvectors modes.

Stage 1 — reduction.  Stack trajectories, take a rank-k randomized SVD (Halko,
Martinsson & Tropp 2011):  X ~= U S V^T,  Q = V[:, :k] in R^{d x k}.
    Why randomized SVD and not exact PCA?
      * d is 2k-16k and the stacked T*N is 1e5-1e8 rows; exact SVD is O(min(m,n)^2 max)
        and does not stream.  rSVD with oversampling p and q power iterations costs
        O(mnk) with an expected spectral-norm error within a small constant of the
        optimal rank-k truncation, and it works in one or two passes over data held on
        disk or produced on the fly by forward hooks.
      * PCA *is* used, as the centered special case: the propagator is fitted to
        mean-removed coordinates because an uncentered first component in a residual
        stream is dominated by the well-known always-on "rogue"/outlier dimensions,
        which are a tokenizer/normalisation artifact and not a dynamical mode.
      * Nonlinear reducers (UMAP/t-SNE/autoencoders) are rejected for the *primary*
        analysis: they are not isometries, have no adjoint, and give no operator whose
        eigenvalues mean anything.  They are permitted only as exploratory displays.
      * ICA/NMF are registered alternatives when the question is "which sources", not
        "which persistent directions"; they do not yield a propagator spectrum.

Stage 2 — propagator.  In reduced coordinates z_t = Q^T (x_t - mu), fit
    A = argmin_A || Z_{1:T} - A Z_{0:T-1} ||_F   (exact DMD; Tu et al. 2014)
with ridge regularisation and per-item concatenation that never crosses item
boundaries.  Eigendecompose A = W diag(lambda) W^{-1}.

Stage 3 — mode statistics.  For eigenvalue lambda_j:
    persistence   rho_j   = |lambda_j|                  (per-step retention; rho>1 unstable)
    half-life     h_j     = ln(0.5)/ln(rho_j)           (steps to half amplitude)
    frequency     f_j     = arg(lambda_j)/(2*pi)        (cycles/step; 0 = non-oscillatory)
    energy share  e_j     = ||amplitude_j||^2 / sum_k ||amplitude_k||^2
    consistency   c_j     = mean pairwise |<w_j^(i), w_j^(i')>| across items i,i'
    SNR           snr_j   = e_j / (Marchenko-Pastur noise-floor energy)

A mode is a CANDIDATE only if it clears all four registered gates (rho, energy,
consistency, SNR) *and* beats the null ladder in ei.nulls.  Clearing the gates alone
is a descriptive statement about a linear fit, nothing more.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np

# --------------------------------------------------------------------------------------
# Randomized SVD
# --------------------------------------------------------------------------------------


def randomized_svd(
    X: np.ndarray,
    k: int,
    n_oversample: int = 10,
    n_power_iter: int = 2,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rank-k randomized SVD with power iterations and QR re-orthonormalisation.

    Returns (U, s, Vt) with U (m,k), s (k,), Vt (k,n).  Power iterations are
    re-orthonormalised each step; without that the sketch collapses onto the top
    singular direction in float32 and every downstream mode is the same mode.
    """
    rng = np.random.default_rng() if rng is None else rng
    m, n = X.shape
    k = int(min(k, min(m, n)))
    ell = min(min(m, n), k + int(n_oversample))
    Omega = rng.standard_normal((n, ell))
    Y = X @ Omega
    Q, _ = np.linalg.qr(Y, mode="reduced")
    for _ in range(int(n_power_iter)):
        Z, _ = np.linalg.qr(X.T @ Q, mode="reduced")
        Q, _ = np.linalg.qr(X @ Z, mode="reduced")
    B = Q.T @ X
    Ub, s, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ Ub
    return U[:, :k], s[:k], Vt[:k, :]


def explained_variance_ratio(s: np.ndarray, total_var: float) -> np.ndarray:
    s = np.asarray(s, dtype=float)
    return (s**2) / max(total_var, 1e-30)


# --------------------------------------------------------------------------------------
# Random-matrix noise floor
# --------------------------------------------------------------------------------------


def marchenko_pastur_edge(n_samples: int, n_features: int, sigma2: float) -> float:
    """Upper edge of the MP bulk: the largest *eigenvalue of the covariance* explicable
    by pure iid noise at variance sigma2 and aspect ratio gamma = features/samples.

    Caveat, stated because it matters: MP assumes iid entries.  Activation trajectories
    are temporally autocorrelated, so the MP edge is a NECESSARY-not-sufficient screen.
    The autocorrelation-preserving surrogate null (ei.nulls.phase_scramble) is the
    binding test; MP only removes modes that are not even above white noise.
    """
    gamma = n_features / max(n_samples, 1)
    return float(sigma2 * (1.0 + np.sqrt(gamma)) ** 2)


def noise_floor_energy(X: np.ndarray, k: int) -> float:
    """Energy a rank-k subspace would capture from matched white noise (MP prediction)."""
    T, d = X.shape
    sigma2 = float(np.var(X, axis=0).mean())
    edge = marchenko_pastur_edge(T, d, sigma2)
    total = sigma2 * d
    return float(min(k * edge, total) / max(total, 1e-30))


# --------------------------------------------------------------------------------------
# Propagator / DMD
# --------------------------------------------------------------------------------------


def fit_propagator(Z_pairs: list[tuple[np.ndarray, np.ndarray]], ridge: float = 1e-3) -> np.ndarray:
    """Ridge least-squares propagator A from within-item (z_t, z_{t+1}) pairs.

    Pairs are supplied per item so that no regression pair ever straddles two items;
    concatenating items before differencing is a common and silent bug that manufactures
    spurious slow modes at the seams.

    KNOWN BIAS -- READ BEFORE QUOTING A PERSISTENCE VALUE
    -----------------------------------------------------
    This estimator is DOWNWARD BIASED for |lambda| whenever the reduced coordinates carry
    background noise, which they always do.  The regressor z_t is itself measured with
    error, so the least-squares slope is attenuated toward zero (errors-in-variables).
    Measured on planted ground truth with true rho = 0.97:

        background            T=80      T=640     converges to truth?
        none (sigma -> 0)     0.968     0.970     yes
        AR(1) at 0.85         0.952     0.959     NO -- floor at ~ -0.012

    The bias does NOT vanish with more data; it is set by the signal-to-noise ratio in the
    subspace, not the sample size.  Ridge adds a further small shrinkage on top
    (rho 0.948 -> 0.929 as ridge goes 0 -> 0.1).

    Consequences, which the framework takes seriously rather than papering over:
      1. `min_persistence` gates are CONSERVATIVE: true persistence is >= the estimate, so
         a mode that passes the gate would also pass it with an unbiased estimator.
      2. Absolute persistence values (and the half-lives derived from them) must not be
         quoted as calibrated quantities.  Report them as a lower bound, or de-bias
         against matched surrogates.
      3. Comparisons BETWEEN conditions remain valid only if SNR is matched across them.
         An apparent persistence difference between two conditions with different noise
         levels can be entirely attenuation.  `persistence_bias_warning` flags this.
    """
    Zp = np.concatenate([a for a, _ in Z_pairs], axis=0)
    Zf = np.concatenate([b for _, b in Z_pairs], axis=0)
    k = Zp.shape[1]
    G = Zp.T @ Zp + ridge * np.trace(Zp.T @ Zp) / max(k, 1) * np.eye(k)
    return np.linalg.solve(G, Zp.T @ Zf).T


@dataclass
class Mode:
    index: int
    real_dims: int = 1          # 1 for a real mode; 2 for a conjugate pair's invariant plane
    persistence: float = 0.0  # |lambda|
    half_life: float = 0.0    # steps
    frequency: float = 0.0    # cycles/step
    energy_share: float = 0.0
    consistency: float = 0.0  # cross-item eigenvector agreement in [0,1]
    snr: float = 0.0          # energy share / MP noise-floor share
    is_oscillatory: bool = False
    vector_real: list = field(default_factory=list)   # reduced-space eigenvector, real part
    vector_imag: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ModeSet:
    modes: list[Mode]
    basis: np.ndarray            # Q: d x k projection to reduced space
    mean: np.ndarray             # mu: d
    eigvals: np.ndarray
    reduced_dim: int
    noise_floor: float
    total_energy_captured: float
    eigvec_condition: float = 1.0   # cond(W); > ~1e3 means modal amplitudes are unreliable

    def top(self, n: int = 1, by: str = "salience") -> list[Mode]:
        """Rank modes.  Default is `salience`, NOT energy share.

        The sweep in tests/test_recovery.py shows why: ranking by energy share alone
        recovers a planted subspace WORSE as data increase (overlap 0.46 -> 0.02 from 40
        to 400 items), because a coloured-noise background contributes many moderate-
        energy, low-persistence modes that crowd out the true one.  Persistence is the
        property the construct is defined by, so it must enter the ranking.  `salience`
        is energy share weighted by persistence and cross-item consistency -- all three
        registered properties, combined before selection rather than after.
        """
        keys = {
            "salience": lambda m: -(m.energy_share * max(m.persistence, 0.0) * max(m.consistency, 1e-3)),
            "energy": lambda m: -m.energy_share,
            "persistence": lambda m: -(m.persistence if np.isfinite(m.persistence) else -1),
        }
        if by not in keys:
            raise ValueError(f"unknown ranking {by!r}; use one of {sorted(keys)}")
        return sorted(self.modes, key=keys[by])[:n]

    def salience(self, m: "Mode") -> float:
        return float(m.energy_share * max(m.persistence, 0.0) * max(m.consistency, 1e-3))

    def by_index(self, i: int) -> "Mode":
        """Look up a mode by its EIGENVALUE index.

        After conjugate deduplication `self.modes` is a filtered list, so positional
        indexing into it does not correspond to `Mode.index`.  Everything that selects
        modes speaks in eigenvalue indices, so the lookup is explicit here rather than
        relying on list position.
        """
        for m in self.modes:
            if m.index == i:
                return m
        raise KeyError(f"no mode with eigenvalue index {i}; available: {[m.index for m in self.modes]}")

    def ambient_subspace(self, mode_indices: list[int]) -> np.ndarray:
        """Orthonormal ambient-space basis (d x r) spanned by the selected modes.

        A conjugate-pair mode contributes the real and imaginary parts of its
        eigenvector, which together span its real invariant plane (r = 2).  A real mode
        contributes one column.
        """
        cols = []
        for i in mode_indices:
            m = self.by_index(i)
            v = np.asarray(m.vector_real, dtype=float)
            cols.append(self.basis @ v)
            if m.is_oscillatory:
                w = np.asarray(m.vector_imag, dtype=float)
                if np.linalg.norm(w) > 1e-12:
                    cols.append(self.basis @ w)
        M = np.column_stack(cols)
        Qb, _ = np.linalg.qr(M)
        return Qb

    def summary(self) -> list[dict]:
        out = []
        for m in self.top(len(self.modes), by="salience"):
            d = m.to_dict()
            d["salience"] = self.salience(m)
            d.pop("vector_real", None)
            d.pop("vector_imag", None)
            out.append(d)
        return out


def _dedupe_conjugates(eigvals: np.ndarray, tol: float = 1e-8) -> list[int]:
    """Keep one representative per complex-conjugate pair.

    `np.linalg.eig` on a real matrix returns lambda and conj(lambda) as two separate
    eigenvalues whose eigenvectors span the SAME real two-dimensional invariant plane.
    Treating them as two modes double-counts that plane: selecting the 'top 2' modes then
    returns one plane twice instead of two structures, which silently halves the recovered
    subspace (measured: overlap 0.996 for top-1 collapsing to 0.500 for top-2 on data with
    a known planted answer).  One representative per pair is kept -- the one with
    Im(lambda) >= 0 -- and it carries real_dims=2.
    """
    eigvals = np.asarray(eigvals)
    used = np.zeros(eigvals.size, dtype=bool)
    keep: list[int] = []
    for i, lam in enumerate(eigvals):
        if used[i]:
            continue
        used[i] = True
        if abs(lam.imag) <= tol:
            keep.append(i)
            continue
        if lam.imag < 0:
            partner = None
            for j in range(eigvals.size):
                if not used[j] and abs(eigvals[j] - np.conj(lam)) <= tol * max(1.0, abs(lam)):
                    partner = j
                    break
            if partner is not None:
                used[partner] = True
                keep.append(partner)          # prefer the Im >= 0 representative
                continue
            keep.append(i)
            continue
        for j in range(eigvals.size):
            if not used[j] and abs(eigvals[j] - np.conj(lam)) <= tol * max(1.0, abs(lam)):
                used[j] = True
                break
        keep.append(i)
    return keep


def _projected_energy(Z: np.ndarray, eigvecs: np.ndarray) -> np.ndarray:
    """Energy along each mode's invariant direction, by ORTHOGONAL PROJECTION.

    Why not the textbook DMD amplitude b = W^{-1} z?
    ------------------------------------------------
    A fitted propagator is generally NON-NORMAL: its eigenvectors are not orthogonal and
    W can be severely ill-conditioned.  Then W^{-1} amplifies near-degenerate directions,
    individual modal amplitudes blow up with massive cancellation between them, and the
    'energy share' of a mode becomes an artifact of cond(W) rather than a property of the
    data.  Empirically this gets WORSE with more data (more near-degenerate noise modes
    to cancel against), which is the opposite of what an estimator should do.

    The projection energy  e_j = mean_t |<u_j, z_t>|^2  is not an exact additive
    decomposition (the parts do not sum to the whole when modes are oblique), and that is
    stated rather than hidden.  It is bounded, stable under ill-conditioning, and answers
    the question actually being asked: how much of the trajectory lies along this
    direction.  For a complex-conjugate pair, the real invariant plane spanned by
    Re(w), Im(w) is used and its energy split between the pair.
    """
    k = eigvecs.shape[1]
    out = np.zeros(k)
    for j in range(k):
        w = eigvecs[:, j]
        if np.max(np.abs(np.imag(w))) < 1e-12:
            u = np.real(w)
            nu = np.linalg.norm(u)
            if nu < 1e-30:
                continue
            out[j] = float(np.mean((Z @ (u / nu)) ** 2))
        else:
            B = np.column_stack([np.real(w), np.imag(w)])
            Qb, _ = np.linalg.qr(B)
            # Full plane energy: after conjugate deduplication ONE mode represents the
            # whole 2D invariant plane, so the plane's energy is not halved.
            out[j] = float(np.mean(np.sum((Z @ Qb) ** 2, axis=1)))
    return out


def _eig_consistency(per_item_A: list[np.ndarray], eigvecs: np.ndarray) -> np.ndarray:
    """Cross-item agreement for each global eigenvector: mean over items of the best
    absolute normalised overlap with that item's own eigenvectors.

    Using the *best* per-item match (rather than index-matched) is deliberate: eigenvalue
    ordering is not stable across items, and index matching would report noise as
    inconsistency.
    """
    k = eigvecs.shape[1]
    out = np.zeros(k)
    if not per_item_A:
        return out
    per_item_vecs = []
    for A in per_item_A:
        try:
            _, V = np.linalg.eig(A)
        except np.linalg.LinAlgError:
            continue
        V = V / (np.linalg.norm(V, axis=0, keepdims=True) + 1e-30)
        per_item_vecs.append(V)
    if not per_item_vecs:
        return out
    for j in range(k):
        v = eigvecs[:, j]
        v = v / (np.linalg.norm(v) + 1e-30)
        overlaps = [float(np.max(np.abs(np.conj(v) @ V))) for V in per_item_vecs]
        out[j] = float(np.clip(np.mean(overlaps), 0.0, 1.0))
    return out


def extract_modes(
    trajectories: list[np.ndarray],
    reduced_dim: int = 32,
    ridge: float = 1e-3,
    n_power_iter: int = 2,
    n_oversample: int = 10,
    rng: np.random.Generator | None = None,
    center: bool = True,
) -> ModeSet:
    """Full three-stage extraction.  `trajectories` is a list of (T_i, d) arrays, one per
    item.  Items with T_i < 3 are dropped (a propagator needs at least two pairs)."""
    rng = np.random.default_rng() if rng is None else rng
    trajectories = [np.asarray(X, dtype=np.float64) for X in trajectories if X.shape[0] >= 3]
    if not trajectories:
        raise ValueError("no trajectory has >= 3 timesteps")
    d = trajectories[0].shape[1]
    stacked = np.concatenate(trajectories, axis=0)
    mu = stacked.mean(axis=0) if center else np.zeros(d)
    Xc = stacked - mu

    k = int(min(reduced_dim, min(Xc.shape) - 1))
    _, s, Vt = randomized_svd(Xc, k=k, n_oversample=n_oversample, n_power_iter=n_power_iter, rng=rng)
    Q = Vt.T                                            # d x k
    total_var = float((Xc**2).sum())
    captured = float((s**2).sum() / max(total_var, 1e-30))

    pairs, per_item_A = [], []
    for X in trajectories:
        Z = (X - mu) @ Q
        pairs.append((Z[:-1], Z[1:]))
        if Z.shape[0] >= k + 2:
            try:
                per_item_A.append(fit_propagator([(Z[:-1], Z[1:])], ridge=ridge))
            except np.linalg.LinAlgError:
                pass

    A = fit_propagator(pairs, ridge=ridge)
    eigvals, eigvecs = np.linalg.eig(A)

    Zall = np.concatenate([p[0] for p in pairs], axis=0)
    energy = _projected_energy(Zall, eigvecs)
    energy_share = energy / max(float(energy.sum()), 1e-30)
    kappa = float(np.linalg.cond(eigvecs))

    cons = _eig_consistency(per_item_A, eigvecs)
    floor = noise_floor_energy(Xc, 1)

    keep_idx = _dedupe_conjugates(eigvals)

    modes: list[Mode] = []
    for j in keep_idx:
        lam = eigvals[j]
        rho = float(np.abs(lam))
        if 0.0 < rho < 1.0:
            hl = float(np.log(0.5) / np.log(rho))
        elif rho >= 1.0:
            hl = float("inf")
        else:
            hl = 0.0
        freq = float(np.angle(lam) / (2 * np.pi))
        osc = bool(abs(freq) > 1e-6)
        modes.append(
            Mode(
                index=j,
                real_dims=2 if osc else 1,
                persistence=rho,
                half_life=hl,
                frequency=freq,
                energy_share=float(energy_share[j]),
                consistency=float(cons[j]),
                snr=float(energy_share[j] / max(floor, 1e-30)),
                is_oscillatory=osc,
                vector_real=np.real(eigvecs[:, j]).tolist(),
                vector_imag=np.imag(eigvecs[:, j]).tolist(),
            )
        )

    return ModeSet(
        modes=modes,
        basis=Q,
        mean=mu,
        eigvals=eigvals,
        reduced_dim=k,
        noise_floor=float(floor),
        total_energy_captured=captured,
        eigvec_condition=kappa,
    )


def gate_modes(
    ms: ModeSet,
    min_persistence: float = 0.90,
    min_energy_share: float = 0.01,
    min_consistency: float = 0.50,
    min_snr: float = 3.0,
) -> list[int]:
    """Apply the four pre-registered descriptive gates.  Returns candidate mode indices,
    ordered by energy share.  Gates are a SCREEN, not evidence."""
    keep = [
        m
        for m in ms.modes
        if (m.persistence >= min_persistence)
        and (m.energy_share >= min_energy_share)
        and (m.consistency >= min_consistency)
        and (m.snr >= min_snr)
        and np.isfinite(m.persistence)
        and (m.persistence < 1.5)      # |lambda| >> 1 is a divergent fit artifact, not a mode
    ]
    return [m.index for m in sorted(keep, key=lambda m: -ms.salience(m))]


def persistence_bias_warning(snr: float) -> str | None:
    """Flag persistence estimates whose attenuation bias is large enough to matter.

    Attenuation for an AR(1) observed in additive noise is approximately
    rho_hat ~ rho * SNR/(1+SNR), so an SNR of 3 implies roughly a 25% underestimate of the
    deviation from zero.  Below SNR 10 the absolute value should not be quoted without a
    de-biasing step.
    """
    if not np.isfinite(snr) or snr <= 0:
        return "SNR is undefined; persistence cannot be interpreted"
    if snr < 3:
        return (f"SNR={snr:.2f}: persistence is severely attenuated (order {1/(1+snr):.0%} "
                "downward). Treat the value as a lower bound only.")
    if snr < 10:
        return (f"SNR={snr:.2f}: persistence is attenuated by roughly {1/(1+snr):.0%}. "
                "Do not compare across conditions with differing SNR.")
    return None


def subspace_principal_angles(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Principal angles (radians) between two orthonormal ambient bases."""
    Pq, _ = np.linalg.qr(P)
    Qq, _ = np.linalg.qr(Q)
    s = np.linalg.svd(Pq.T @ Qq, compute_uv=False)
    return np.arccos(np.clip(s, -1.0, 1.0))


def subspace_overlap(P: np.ndarray, Q: np.ndarray) -> float:
    """Mean cos^2 of principal angles in [0,1]: 1 = identical subspace, 0 = orthogonal."""
    th = subspace_principal_angles(P, Q)
    return float(np.mean(np.cos(th) ** 2))
