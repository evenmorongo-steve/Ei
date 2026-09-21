"""
Statistical validation primitives.

Design commitments
------------------
1. The unit of inference is the *item* (or the seed, or the checkpoint) — never the
   token.  Token-level resampling is pseudoreplication and inflates significance by
   orders of magnitude; `cluster_bootstrap` and `permutation_test` therefore operate
   on cluster indices, and `check_unit_of_inference` refuses token-level units.
2. Every point estimate ships with an interval.  Bare p-values are not decision inputs.
3. Negative evidence requires an equivalence test (TOST) against a pre-registered
   SESOI.  "p > alpha" is never evidence of absence.
4. Multiplicity is handled by the procedure appropriate to the question:
   - selection over modes            -> max-statistic FWER (permutation)
   - families of secondary contrasts -> Benjamini-Hochberg FDR
   - one pre-registered primary      -> alpha_primary, no correction
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Callable, Sequence

import numpy as np

# --------------------------------------------------------------------------------------
# Result containers
# --------------------------------------------------------------------------------------


@dataclass
class TestResult:
    """A single statistical decision, fully self-describing for audit."""

    name: str
    statistic: float
    p_value: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    ci_level: float = 0.95
    alpha: float = 0.005
    n_resamples: int = 0
    unit_of_inference: str = "item"
    tail: str = "greater"
    decision: str = "undetermined"  # pass | fail | equivalent | undetermined
    note: str = ""

    def finalize(self) -> "TestResult":
        if self.p_value is not None:
            self.decision = "pass" if self.p_value < self.alpha else "fail"
        return self

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EquivalenceResult:
    name: str
    estimate: float
    margin: float
    ci_low: float
    ci_high: float
    ci_level: float
    equivalent: bool
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------------------


class PseudoreplicationError(ValueError):
    """Raised when resampling would treat non-independent observations as independent."""


_VALID_UNITS = {
    "independent_task_series",
    "independent_semantic_items",
    "model_checkpoints",
    "architecture_families",
    "interaction_episodes",
    "item",
    "seed",
}


def check_unit_of_inference(unit: str) -> None:
    if unit in {"token", "timestep", "position", "activation_vector"}:
        raise PseudoreplicationError(
            f"unit_of_inference={unit!r} is pseudoreplication: tokens within a sequence "
            "are strongly autocorrelated and are not exchangeable.  Resample items, "
            "episodes, seeds, or checkpoints instead."
        )
    if unit not in _VALID_UNITS:
        raise PseudoreplicationError(
            f"unit_of_inference={unit!r} is not a registered unit. Registered: "
            f"{sorted(_VALID_UNITS)}"
        )


# --------------------------------------------------------------------------------------
# Permutation testing
# --------------------------------------------------------------------------------------


def permutation_test(
    observed: float,
    null_sampler: Callable[[np.random.Generator], float],
    n_perm: int,
    rng: np.random.Generator,
    tail: str = "greater",
    alpha: float = 0.005,
    name: str = "permutation_test",
    unit: str = "independent_semantic_items",
) -> TestResult:
    """Monte-Carlo permutation test with the (b+1)/(B+1) correction (Phipson & Smyth 2010).

    `null_sampler` must draw ONE statistic from the null by re-randomising the
    exchangeable structure (label permutation or surrogate generation).  It must not
    re-use the observed data ordering.
    """
    check_unit_of_inference(unit)
    null = np.asarray([float(null_sampler(rng)) for _ in range(n_perm)], dtype=float)
    null = null[np.isfinite(null)]
    b_eff = null.size
    if b_eff == 0:
        raise ValueError("null sampler produced no finite statistics")
    if tail == "greater":
        b = int(np.sum(null >= observed))
    elif tail == "less":
        b = int(np.sum(null <= observed))
    elif tail == "two-sided":
        centre = float(np.median(null))
        b = int(np.sum(np.abs(null - centre) >= abs(observed - centre)))
    else:
        raise ValueError(f"unknown tail {tail!r}")
    p = (b + 1.0) / (b_eff + 1.0)
    lo, hi = np.quantile(null, [0.025, 0.975])
    res = TestResult(
        name=name,
        statistic=float(observed),
        p_value=float(p),
        ci_low=float(lo),
        ci_high=float(hi),
        alpha=alpha,
        n_resamples=b_eff,
        unit_of_inference=unit,
        tail=tail,
        note=(
            f"null mean={null.mean():.6g} sd={null.std(ddof=1):.6g}; "
            f"CI shown is the 95% interval of the NULL distribution, not of the estimate; "
            f"p is resolution-limited at {1.0 / (b_eff + 1.0):.2g}"
        ),
    )
    return res.finalize()


def null_z(observed: float, null_samples: np.ndarray) -> float:
    """Standardised distance from the null centre.  Reported alongside p, never instead."""
    null_samples = np.asarray(null_samples, dtype=float)
    sd = null_samples.std(ddof=1)
    if sd <= 0 or not np.isfinite(sd):
        return float("nan")
    return float((observed - null_samples.mean()) / sd)


# --------------------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------------------


def cluster_bootstrap(
    clusters: Sequence,
    stat_fn: Callable[[Sequence], float],
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Nonparametric cluster (block) bootstrap: resample whole clusters with replacement.

    `clusters` is a sequence of per-item payloads; `stat_fn` maps a resampled list of
    payloads to a scalar.  This preserves within-item dependence, which the naive
    bootstrap destroys.
    """
    n = len(clusters)
    if n < 4:
        raise ValueError(f"cluster bootstrap needs >= 4 clusters, got {n}")
    out = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        out[b] = float(stat_fn([clusters[i] for i in idx]))
    return out


def jackknife(clusters: Sequence, stat_fn: Callable[[Sequence], float]) -> np.ndarray:
    n = len(clusters)
    out = np.empty(n, dtype=float)
    for i in range(n):
        out[i] = float(stat_fn([clusters[j] for j in range(n) if j != i]))
    return out


def _norm_cdf(x: float) -> float:
    from math import erf, sqrt

    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    # Acklam's rational approximation; adequate for CI endpoints.
    if not 0.0 < p < 1.0:
        return float("nan")
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > phigh:
        q = np.sqrt(-2 * np.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
    )


def bca_ci(
    theta_hat: float,
    boot: np.ndarray,
    jack: np.ndarray | None = None,
    level: float = 0.95,
) -> tuple[float, float]:
    """Bias-corrected and accelerated bootstrap interval (Efron 1987).

    Falls back to the percentile interval when the acceleration cannot be estimated.
    """
    boot = np.asarray(boot, dtype=float)
    boot = boot[np.isfinite(boot)]
    if boot.size < 20:
        raise ValueError("too few finite bootstrap replicates for an interval")
    alpha = 1.0 - level
    prop = float(np.mean(boot < theta_hat))
    prop = min(max(prop, 1.0 / (boot.size + 1)), 1.0 - 1.0 / (boot.size + 1))
    z0 = _norm_ppf(prop)
    a = 0.0
    if jack is not None and np.isfinite(jack).all() and jack.size > 2:
        jm = jack.mean()
        num = float(np.sum((jm - jack) ** 3))
        den = 6.0 * (float(np.sum((jm - jack) ** 2)) ** 1.5)
        a = num / den if den > 0 else 0.0
    zl, zu = _norm_ppf(alpha / 2), _norm_ppf(1 - alpha / 2)

    def adj(z):
        denom = 1 - a * (z0 + z)
        if abs(denom) < 1e-12:
            return _norm_cdf(z0 + z)
        return _norm_cdf(z0 + (z0 + z) / denom)

    q_lo, q_hi = adj(zl), adj(zu)
    if not (np.isfinite(q_lo) and np.isfinite(q_hi)):
        q_lo, q_hi = alpha / 2, 1 - alpha / 2
    lo, hi = np.quantile(boot, [np.clip(q_lo, 0, 1), np.clip(q_hi, 0, 1)])
    return float(lo), float(hi)


# --------------------------------------------------------------------------------------
# Equivalence testing (licenses NEGATIVE evidence)
# --------------------------------------------------------------------------------------


def tost_equivalence(
    estimate: float,
    boot: np.ndarray,
    margin: float,
    alpha: float = 0.05,
    name: str = "tost",
) -> EquivalenceResult:
    """Two one-sided tests via the bootstrap (1-2*alpha) interval.

    Equivalence is declared iff the whole (1-2*alpha) interval lies inside
    (-margin, +margin).  This is the ONLY route by which this framework reports
    "the effect is absent"; a non-significant permutation test never is.
    """
    if margin <= 0:
        raise ValueError("equivalence margin (SESOI) must be > 0 and pre-registered")
    lo, hi = np.quantile(np.asarray(boot, dtype=float), [alpha, 1 - alpha])
    equivalent = bool(lo > -margin and hi < margin)
    return EquivalenceResult(
        name=name,
        estimate=float(estimate),
        margin=float(margin),
        ci_low=float(lo),
        ci_high=float(hi),
        ci_level=1 - 2 * alpha,
        equivalent=equivalent,
        note="equivalence <=> (1-2a) interval inside +/- SESOI (Lakens 2017)",
    )


# --------------------------------------------------------------------------------------
# Multiplicity
# --------------------------------------------------------------------------------------


def benjamini_hochberg(pvals: Sequence[float], q: float = 0.05) -> dict:
    p = np.asarray(list(pvals), dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    thresh = q * (np.arange(1, n + 1) / n)
    passed = ranked <= thresh
    k = int(np.max(np.where(passed)[0]) + 1) if passed.any() else 0
    reject = np.zeros(n, dtype=bool)
    if k > 0:
        reject[order[:k]] = True
    # BH-adjusted p-values (step-up monotone)
    adj = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    adj_full = np.empty(n)
    adj_full[order] = np.clip(adj, 0, 1)
    return {"reject": reject.tolist(), "adjusted_p": adj_full.tolist(), "q": q, "n_reject": int(reject.sum())}


def max_statistic_threshold(null_max: np.ndarray, alpha: float = 0.005) -> float:
    """FWER-controlling critical value for selection over a family (Westfall-Young).

    `null_max` must contain, per surrogate draw, the MAXIMUM statistic over the whole
    family that was searched.  Using per-mode nulls here would not control FWER.
    """
    null_max = np.asarray(null_max, dtype=float)
    return float(np.quantile(null_max[np.isfinite(null_max)], 1.0 - alpha))


# --------------------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------------------


def minimum_detectable_effect(n_clusters: int, sd: float, alpha: float = 0.005, power: float = 0.9) -> float:
    """Two-sided MDE for a cluster-level mean contrast; for pre-registration only."""
    z_a = _norm_ppf(1 - alpha / 2)
    z_b = _norm_ppf(power)
    return float((z_a + z_b) * sd / np.sqrt(max(n_clusters, 1)))


def severity(observed: float, null_samples: np.ndarray, benchmark: float) -> float:
    """Mayo-style severity: P(statistic < observed | effect = benchmark), estimated by
    shifting the null distribution to the benchmark.  Reported so that a 'pass' with a
    huge N but a trivial effect is visibly unimpressive.

    `benchmark` must be in the SAME RAW UNITS as `observed`.  The registered SESOI is in
    null standard deviations, so callers must multiply by the null SD first; passing the
    standardised value directly makes a large effect look un-severe purely through a unit
    mismatch.
    """
    shifted = np.asarray(null_samples, dtype=float) + benchmark
    return float(np.mean(shifted < observed))
