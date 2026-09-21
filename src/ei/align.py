"""
Cross-model representational alignment.

Three metrics, used for three different questions.  Reporting all three is not
redundancy theatre — they disagree in informative ways, and a claim that survives only
one of them is a claim about that metric.

  linear CKA         question: "is there ANY linear correspondence between these two
                     representation spaces?"  Invariant to orthogonal transforms and
                     isotropic scaling; NOT invariant to invertible linear maps.
                     Weakness: dominated by high-variance directions; two models can
                     score CKA ~ 0.9 while disagreeing completely about a low-variance
                     but causally critical subspace.  Never use CKA alone for a
                     "same mechanism" claim.

  SVCCA              question: "how many shared dimensions survive noise truncation?"
                     Truncates each space by explained variance, then canonical
                     correlation.  Invariant to invertible linear maps within the kept
                     subspace.  Weakness: the variance threshold is a researcher degree
                     of freedom that changes the answer — so it is pre-registered here
                     and a sensitivity sweep is emitted automatically.

  orthogonal         question: "are these the SAME directions, up to rotation?"
  Procrustes         The strictest of the three, and the only one whose residual has
                     units comparable across model pairs after normalisation.  This is
                     the primary metric for structural-correspondence claims.

EVERY alignment score is meaningless without its null.  Two random Gaussian matrices of
realistic shape score CKA ~ 0.2-0.5, not 0.  `alignment_null` produces the matched
random baseline, and `alignment_with_null` reports the score, the null band, and the
excess.  Raw alignment numbers are never reported alone by this package.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


def _center_columns(X: np.ndarray) -> np.ndarray:
    return X - X.mean(axis=0, keepdims=True)


# --------------------------------------------------------------------------------------
# CKA
# --------------------------------------------------------------------------------------


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA via the feature-space (Gram-free) formulation, O(n d^2) not O(n^2 d).

    Requires row correspondence: row i of X and row i of Y must be the same stimulus at
    the same position.  If that is not true the number is meaningless.
    """
    X, Y = _center_columns(np.asarray(X, float)), _center_columns(np.asarray(Y, float))
    if X.shape[0] != Y.shape[0]:
        raise ValueError("CKA requires paired rows (same stimuli, same order)")
    xty = X.T @ Y
    num = float(np.sum(xty**2))
    den = float(np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro"))
    return num / max(den, 1e-30)


# --------------------------------------------------------------------------------------
# SVCCA
# --------------------------------------------------------------------------------------


def svcca(X: np.ndarray, Y: np.ndarray, var_threshold: float = 0.99, eps: float = 1e-8) -> dict:
    """Singular-vector CCA: SVD-truncate each space, then mean canonical correlation."""
    X, Y = _center_columns(np.asarray(X, float)), _center_columns(np.asarray(Y, float))
    if X.shape[0] != Y.shape[0]:
        raise ValueError("SVCCA requires paired rows")

    def trunc(M):
        U, s, _ = np.linalg.svd(M, full_matrices=False)
        if s.sum() <= 0:
            return M
        keep = int(np.searchsorted(np.cumsum(s**2) / np.sum(s**2), var_threshold) + 1)
        keep = max(1, min(keep, s.size))
        return U[:, :keep] * s[:keep]

    Xt, Yt = trunc(X), trunc(Y)
    qx, _ = np.linalg.qr(Xt)
    qy, _ = np.linalg.qr(Yt)
    corrs = np.linalg.svd(qx.T @ qy, compute_uv=False)
    corrs = np.clip(corrs, 0.0, 1.0)
    return {
        "mean_cca": float(corrs.mean()),
        "top_cca": float(corrs[0]) if corrs.size else float("nan"),
        "n_components": int(corrs.size),
        "kept_dims": [int(Xt.shape[1]), int(Yt.shape[1])],
        "var_threshold": var_threshold,
        "correlations": corrs.tolist(),
    }


def svcca_sensitivity(X: np.ndarray, Y: np.ndarray, thresholds=(0.90, 0.95, 0.99)) -> dict:
    """Emitted automatically: if the conclusion flips across thresholds, say so."""
    out = {str(t): svcca(X, Y, var_threshold=t)["mean_cca"] for t in thresholds}
    vals = list(out.values())
    out["range"] = float(max(vals) - min(vals))
    out["stable"] = bool(out["range"] < 0.10)
    return out


# --------------------------------------------------------------------------------------
# Procrustes
# --------------------------------------------------------------------------------------


def orthogonal_procrustes(X: np.ndarray, Y: np.ndarray, scale: bool = True) -> dict:
    """Solve min_R ||XR - Y||_F over orthogonal R (Schönemann 1966).

    `disparity` is normalised to [0,1] (0 = perfect after rotation, 1 = no better than
    predicting the mean), so it is comparable across model pairs of different dimension
    and scale.  Dimension mismatch is handled by zero-padding the narrower space, which
    is the identity-preserving embedding; truncating the wider one instead would silently
    discard the very directions under test.
    """
    X, Y = _center_columns(np.asarray(X, float)), _center_columns(np.asarray(Y, float))
    if X.shape[0] != Y.shape[0]:
        raise ValueError("Procrustes requires paired rows")
    d = max(X.shape[1], Y.shape[1])
    Xp = np.pad(X, ((0, 0), (0, d - X.shape[1])))
    Yp = np.pad(Y, ((0, 0), (0, d - Y.shape[1])))
    if scale:
        nx, ny = np.linalg.norm(Xp), np.linalg.norm(Yp)
        Xp = Xp / max(nx, 1e-30)
        Yp = Yp / max(ny, 1e-30)
    U, s, Vt = np.linalg.svd(Xp.T @ Yp)
    R = U @ Vt
    resid = float(np.linalg.norm(Xp @ R - Yp) ** 2)
    denom = float(np.linalg.norm(Yp) ** 2)
    return {
        "disparity": resid / max(denom, 1e-30),
        "rotation": R,
        "n_dims": int(d),
        "sum_singular_values": float(s.sum()),
    }


# --------------------------------------------------------------------------------------
# Nulls for alignment
# --------------------------------------------------------------------------------------


def alignment_null(
    X: np.ndarray, Y: np.ndarray, metric: str, n_perm: int, rng: np.random.Generator
) -> np.ndarray:
    """Null by breaking the stimulus pairing (row permutation of Y).

    This is the correct null: it keeps both marginal geometries exactly and destroys only
    the correspondence.  Comparing against Gaussian noise matrices instead would test a
    much weaker hypothesis.
    """
    fn = {
        "cka": linear_cka,
        "svcca": lambda a, b: svcca(a, b)["mean_cca"],
        "procrustes": lambda a, b: 1.0 - orthogonal_procrustes(a, b)["disparity"],
    }[metric]
    n = X.shape[0]
    return np.asarray([float(fn(X, Y[rng.permutation(n)])) for _ in range(n_perm)])


@dataclass
class AlignmentResult:
    metric: str
    score: float
    null_mean: float
    null_sd: float
    null_p95: float
    excess: float
    z: float
    p_value: float
    interpretation: str

    def to_dict(self) -> dict:
        return asdict(self)


def alignment_with_null(
    X: np.ndarray, Y: np.ndarray, metric: str = "procrustes", n_perm: int = 200,
    rng: np.random.Generator | None = None
) -> AlignmentResult:
    rng = np.random.default_rng() if rng is None else rng
    fn = {
        "cka": linear_cka,
        "svcca": lambda a, b: svcca(a, b)["mean_cca"],
        "procrustes": lambda a, b: 1.0 - orthogonal_procrustes(a, b)["disparity"],
    }[metric]
    score = float(fn(X, Y))
    null = alignment_null(X, Y, metric, n_perm, rng)
    mu, sd = float(null.mean()), float(null.std(ddof=1))
    p = (int(np.sum(null >= score)) + 1) / (null.size + 1)
    z = (score - mu) / sd if sd > 0 else float("nan")
    excess = score - mu
    if p >= 0.05:
        interp = "not distinguishable from broken-pairing null: no evidence of correspondence"
    elif excess < 0.05:
        interp = "statistically above null but the excess is small: correspondence is weak"
    else:
        interp = "correspondence exceeds the broken-pairing null in both significance and magnitude"
    return AlignmentResult(metric, score, mu, sd, float(np.quantile(null, 0.95)), excess, z, float(p), interp)


def align_all(X: np.ndarray, Y: np.ndarray, n_perm: int = 200,
              rng: np.random.Generator | None = None) -> dict:
    """Run all three metrics with nulls and flag disagreement between them explicitly."""
    rng = np.random.default_rng() if rng is None else rng
    res = {m: alignment_with_null(X, Y, m, n_perm, rng).to_dict()
           for m in ("cka", "svcca", "procrustes")}
    verdicts = {m: (r["p_value"] < 0.05) for m, r in res.items()}
    res["_concordant"] = len(set(verdicts.values())) == 1
    res["_note"] = (
        "all three metrics agree" if res["_concordant"] else
        "METRICS DISAGREE: "
        + ", ".join(f"{m}={'sig' if v else 'ns'}" for m, v in verdicts.items())
        + ". Report the disagreement; do not select the favourable metric post hoc."
    )
    res["_svcca_sensitivity"] = svcca_sensitivity(X, Y)
    return res
