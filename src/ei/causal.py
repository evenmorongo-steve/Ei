"""
Causal interventions on candidate mode subspaces.

WHY CAUSAL WORK IS NON-OPTIONAL
-------------------------------
Everything in ei.modes and ei.align is correlational.  A persistent, cross-model-aligned,
null-beating mode can still be an epiphenomenal byproduct of the computation — a shadow
cast on the residual stream by machinery that does not read it.  The only way to show a
structure is load-bearing is to perturb it and measure what breaks.

The design is LESION + RESCUE, because lesion alone is weak evidence:

  lesion            project activations onto the orthogonal complement of the mode
                    subspace.  Expect: target task degrades.
                    Confound: ANY rank-r ablation degrades things.
  matched control   ablate a matched random rank-r subspace (ei.nulls).
                    The effect of interest is the DIFFERENCE.
  off-target probe  measure an unrelated task under the same lesion.
                    A lesion that breaks everything has localised nothing.
  rescue            re-inject the removed component (from a donor run, or the item's own
                    stored component) and measure restoration.
                    Rescue is what separates "this subspace carries the information" from
                    "perturbing here breaks the model".
  dose-response     sweep intervention strength.  A genuine functional dependence is
                    monotone in dose; an artifact usually is not.

RESTORATION FRACTION is the primary causal endpoint:

    RF = (perf_lesioned+rescued - perf_lesioned) / (perf_intact - perf_lesioned)

    RF ~ 1  : the removed component was sufficient to restore function
    RF ~ 0  : removing it broke something the component does not carry
    RF > 1  : over-restoration -- usually means the rescue injected task-relevant
              information that was not there originally.  Treat as a FAILED control,
              not as a strong positive.

All interventions here operate on ACTIVATION TENSORS supplied by the caller.  This module
never loads or runs a model; see ei.extract for the hook interface.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np


# --------------------------------------------------------------------------------------
# Projection utilities
# --------------------------------------------------------------------------------------


def projector(basis: np.ndarray) -> np.ndarray:
    """Orthogonal projector P = Q Q^T onto the column space of `basis`."""
    Q, _ = np.linalg.qr(np.asarray(basis, dtype=float))
    return Q @ Q.T


def ablate(X: np.ndarray, basis: np.ndarray, dose: float = 1.0) -> np.ndarray:
    """Remove `dose` of the component of X inside span(basis).  dose=1 is full ablation."""
    P = projector(basis)
    X = np.asarray(X, dtype=float)
    return X - dose * (X @ P)


def steer(X: np.ndarray, direction: np.ndarray, dose: float) -> np.ndarray:
    """Add a signed multiple of a unit direction.

    Dose is in units of the ambient per-token activation norm so that it is comparable
    across layers and models; raw additive doses are not comparable and produce the
    familiar "steering worked at layer 12 but not layer 30" artifact that is really a
    scale artifact.
    """
    v = np.asarray(direction, dtype=float)
    v = v / (np.linalg.norm(v) + 1e-30)
    X = np.asarray(X, dtype=float)
    scale = float(np.linalg.norm(X, axis=-1).mean())
    return X + dose * scale * v


def patch(X_dest: np.ndarray, X_src: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Activation patching restricted to a subspace: replace the in-subspace component of
    the destination run with that of the source run, leaving the complement untouched.

    Subspace-restricted patching (rather than whole-vector patching) is what makes the
    result attributable to the mode rather than to the whole residual stream.
    """
    P = projector(basis)
    Xd, Xs = np.asarray(X_dest, float), np.asarray(X_src, float)
    return Xd - (Xd @ P) + (Xs @ P)


def scrub_pathway(X: np.ndarray, basis: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Pathway scrubbing: replace the in-subspace component with a value resampled from
    the same distribution (another item's component), preserving marginal statistics
    while destroying the item-specific content.

    This distinguishes "the pathway must carry THIS item's information" from "the pathway
    must be in a plausible activation range" — the distinction plain zero-ablation cannot
    make, because zeroing moves activations off-distribution and breaks things for reasons
    unrelated to information content.
    """
    X = np.asarray(X, dtype=float)
    P = projector(basis)
    donor = X[rng.permutation(X.shape[0])]
    return X - (X @ P) + (donor @ P)


def inject_noise(X: np.ndarray, basis: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Isotropic Gaussian noise confined to the subspace, scaled to a fraction `sigma` of
    the in-subspace RMS amplitude."""
    X = np.asarray(X, dtype=float)
    Q, _ = np.linalg.qr(np.asarray(basis, float))
    coords = X @ Q
    amp = float(np.sqrt(np.mean(coords**2)) + 1e-30)
    return X + (rng.standard_normal(coords.shape) * sigma * amp) @ Q.T


# --------------------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------------------


def restoration_fraction(intact: float, lesioned: float, rescued: float) -> float:
    denom = intact - lesioned
    if abs(denom) < 1e-12:
        return float("nan")   # lesion had no effect; RF is undefined, not 0
    return float((rescued - lesioned) / denom)


def selective_risk(correct: np.ndarray, confidence: np.ndarray, coverage: float) -> float:
    """Risk on the most-confident `coverage` fraction.  The endpoint for uncertainty
    monitoring: a system with a working confidence signal should have LOWER error when it
    is allowed to abstain.  Accuracy alone cannot show this."""
    correct = np.asarray(correct, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    n_keep = max(1, int(round(coverage * correct.size)))
    keep = np.argsort(-confidence)[:n_keep]
    return float(1.0 - correct[keep].mean())


def expected_calibration_error(correct: np.ndarray, confidence: np.ndarray, n_bins: int = 10) -> float:
    correct = np.asarray(correct, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (confidence > edges[i]) & (confidence <= edges[i + 1])
        if m.sum() == 0:
            continue
        ece += (m.mean()) * abs(correct[m].mean() - confidence[m].mean())
    return float(ece)


def brier_score(correct: np.ndarray, confidence: np.ndarray) -> float:
    return float(np.mean((np.asarray(confidence, float) - np.asarray(correct, float)) ** 2))


# --------------------------------------------------------------------------------------
# Intervention bookkeeping
# --------------------------------------------------------------------------------------


@dataclass
class InterventionResult:
    name: str
    target: str
    dose: float
    intact: float
    lesioned: float
    rescued: float | None = None
    matched_random: float | None = None
    off_target_intact: float | None = None
    off_target_lesioned: float | None = None

    def effect(self) -> float:
        return self.intact - self.lesioned

    def specificity(self) -> float | None:
        """Targeted effect minus matched-random effect.  This, not `effect`, is the number
        that supports a localisation claim."""
        if self.matched_random is None:
            return None
        return (self.intact - self.lesioned) - (self.intact - self.matched_random)

    def off_target_effect(self) -> float | None:
        if self.off_target_intact is None or self.off_target_lesioned is None:
            return None
        return self.off_target_intact - self.off_target_lesioned

    def rf(self) -> float | None:
        return None if self.rescued is None else restoration_fraction(self.intact, self.lesioned, self.rescued)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(
            effect=self.effect(),
            specificity=self.specificity(),
            off_target_effect=self.off_target_effect(),
            restoration_fraction=self.rf(),
        )
        return d


def dose_response_monotonicity(doses: np.ndarray, effects: np.ndarray) -> dict:
    """Spearman rank correlation between dose and effect, with the sign convention that
    positive = larger dose gives larger effect."""
    doses, effects = np.asarray(doses, float), np.asarray(effects, float)
    if doses.size < 3:
        return {"rho": float("nan"), "monotone": False, "note": "need >= 3 dose levels"}
    rd = np.argsort(np.argsort(doses)).astype(float)
    re = np.argsort(np.argsort(effects)).astype(float)
    rd -= rd.mean()
    re -= re.mean()
    denom = np.sqrt((rd**2).sum() * (re**2).sum())
    rho = float((rd * re).sum() / denom) if denom > 0 else float("nan")
    return {
        "rho": rho,
        "monotone": bool(np.isfinite(rho) and rho > 0.8),
        "note": "monotone dose-response is a necessary condition for a functional dependence",
    }


@dataclass
class CausalBattery:
    """The full battery for one candidate subspace.  `verdict` refuses to return a
    positive unless specificity, off-target containment, rescue, and dose-response all
    hold — the conjunction, not any single test."""

    results: list[InterventionResult] = field(default_factory=list)
    dose_curve: dict = field(default_factory=dict)
    min_specificity: float = 0.05
    max_off_target_ratio: float = 0.30
    min_restoration: float = 0.60
    max_restoration: float = 1.25

    def verdict(self) -> dict:
        checks, failures = {}, []
        primary = self.results[0] if self.results else None
        if primary is None:
            return {"pass": False, "checks": {}, "failures": ["no interventions run"]}

        spec = primary.specificity()
        checks["specificity_vs_matched_random"] = spec is not None and spec >= self.min_specificity
        if not checks["specificity_vs_matched_random"]:
            failures.append(
                "targeted ablation is not distinguishable from ablating a matched random "
                "subspace of the same rank: no localisation is demonstrated"
            )

        ote, te = primary.off_target_effect(), primary.effect()
        if ote is None:
            checks["off_target_contained"] = False
            failures.append("no off-target probe was run: cannot exclude a global damage effect")
        else:
            ratio = abs(ote) / max(abs(te), 1e-12)
            checks["off_target_contained"] = ratio <= self.max_off_target_ratio
            if not checks["off_target_contained"]:
                failures.append(
                    f"off-target/target effect ratio {ratio:.2f} exceeds {self.max_off_target_ratio}: "
                    "the lesion degrades unrelated capability, so the effect is not specific"
                )

        rf = primary.rf()
        if rf is None:
            checks["rescue"] = False
            failures.append("no rescue arm: lesion-only evidence cannot show the subspace carries the content")
        else:
            checks["rescue"] = self.min_restoration <= rf <= self.max_restoration
            if rf > self.max_restoration:
                failures.append(
                    f"restoration fraction {rf:.2f} exceeds {self.max_restoration}: over-restoration "
                    "indicates the rescue injected information not originally present -- FAILED control"
                )
            elif rf < self.min_restoration:
                failures.append(f"restoration fraction {rf:.2f} below {self.min_restoration}")

        checks["dose_response_monotone"] = bool(self.dose_curve.get("monotone", False))
        if not checks["dose_response_monotone"]:
            failures.append("dose-response is not monotone: functional dependence is not established")

        return {"pass": all(checks.values()), "checks": checks, "failures": failures}
