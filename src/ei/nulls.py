"""
Null models — the ladder that candidate structure must climb.

Each null destroys ONE property while preserving the others.  A candidate that survives
all four has been shown to depend on something that no cheaper explanation supplies.
The ordering matters: they are increasingly hard to pass, and each answers a distinct
"but couldn't it just be ...?" objection.

  N1  temporal shuffle      destroys: temporal order
                            preserves: marginal distribution, covariance across features
                            kills: "the mode is just the static geometry of the cloud"

  N2  phase scramble        destroys: phase relations / nonlinear temporal structure
                            preserves: exact power spectrum, hence autocorrelation
                            kills: "the mode is just 1/f drift and autocorrelation"
                            -- this is the binding null.  Residual streams are strongly
                            autocorrelated; ANY method will find slow 'persistent'
                            components in them.  Beating N1 alone is near-meaningless.

  N3  untrained / random-init
                            destroys: everything learned
                            preserves: architecture, depth, width, tokenizer, prompt set,
                                       normalisation, positional structure
                            kills: "the mode is an architectural artifact"

  N4  output-matched mimic  destroys: the internal route to the behaviour
                            preserves: the behaviour itself (matched output distribution)
                            kills: "the mode is downstream of producing self-talk"
                            -- the anti-mimicry null.  Implemented as a donor system whose
                            outputs match on the target distribution but which reaches them
                            by a different mechanism (e.g. a prompted-persona or distilled
                            model). If the mode is present in the mimic too, the mode is
                            about the OUTPUT, not the construct.

Also provided: matched random subspace (rank-and-spectrum-matched, the correct control
for "any r-dimensional subspace would do"), and prompt counterfactual (the correct
control for "the mode is about the topic words, not the self-reference").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------------------
# N1 — temporal shuffle
# --------------------------------------------------------------------------------------


def temporal_shuffle(X: np.ndarray, rng: np.random.Generator, jointly: bool = True) -> np.ndarray:
    """Permute timesteps.

    jointly=True permutes whole rows (destroys temporal order, preserves the
    instantaneous cross-feature covariance).  jointly=False permutes each feature
    independently, which additionally destroys cross-feature structure — that is a
    *different, weaker* null and is offered only for diagnostics.
    """
    X = np.asarray(X)
    if jointly:
        return X[rng.permutation(X.shape[0])]
    out = np.empty_like(X)
    for j in range(X.shape[1]):
        out[:, j] = X[rng.permutation(X.shape[0]), j]
    return out


# --------------------------------------------------------------------------------------
# N2 — phase scramble (IAAFT)
# --------------------------------------------------------------------------------------


def phase_scramble(
    X: np.ndarray,
    rng: np.random.Generator,
    preserve_cross: bool = False,
    block_size: int | None = None,
) -> np.ndarray:
    """Fourier phase randomisation preserving the exact per-feature power spectrum.

    THE preserve_cross TRAP -- MEASURED, NOT HYPOTHESISED
    -----------------------------------------------------
    Applying the SAME random phase offset to every feature looks like the conservative
    choice: it preserves the cross-spectrum, so the surrogates keep the joint geometry and
    ought to be *harder* to beat.  For a propagator/DMD statistic it is not conservative,
    it is INVALID.  Adding a common phase shift exp(i*phi_f) to every channel leaves every
    cross-spectrum S_xy(f) = X(f) conj(Y(f)) exactly unchanged, hence leaves every lagged
    cross-covariance unchanged -- and the lag-1 cross-covariance is precisely the quantity
    the propagator is fitted from.  Measured on planted ground truth:

        surrogate type        lag-1 cross-cov preserved    max |lambda| of surrogate
        identical phases      corr = 0.999                 0.944  (observed: 0.954)
        independent phases    corr = 0.215                 0.956

    With identical phases the surrogate reproduces the real eigenvalues to three decimals,
    so no true structure can EVER clear this null: the test has no power by construction,
    and would emit 'structure_absent' for arbitrarily strong planted signal.

    The default is therefore independent per-feature phases, which destroys cross-channel
    timing while preserving each channel's autocorrelation -- the property the null is
    meant to control for.  `preserve_cross=True` is retained ONLY for statistics that do
    not depend on lagged cross-covariance (e.g. per-channel spectral measures), and
    `assert_null_has_power` refuses the invalid combination.
    """
    X = np.asarray(X, dtype=float)
    T, d = X.shape
    mu = X.mean(axis=0)
    F = np.fft.rfft(X - mu, axis=0)
    n_freq = F.shape[0]
    if preserve_cross:
        phases = np.repeat(rng.uniform(0, 2 * np.pi, size=(n_freq, 1)), d, axis=1)
    elif block_size and block_size > 1:
        # Optional middle ground: features are phase-randomised in blocks, so within-block
        # cross-spectra survive and between-block timing does not.  Use when a genuinely
        # block-structured representation must be preserved.
        phases = np.empty((n_freq, d))
        for start in range(0, d, block_size):
            stop = min(start + block_size, d)
            phases[:, start:stop] = rng.uniform(0, 2 * np.pi, size=(n_freq, 1))
    else:
        phases = rng.uniform(0, 2 * np.pi, size=(n_freq, d))
    phases[0, :] = 0.0
    if T % 2 == 0:
        phases[-1, :] = 0.0
    Fs = np.abs(F) * np.exp(1j * (np.angle(F) + phases))
    return np.fft.irfft(Fs, n=T, axis=0) + mu


def lagged_cross_covariance(X: np.ndarray, lag: int = 1) -> np.ndarray:
    A = np.asarray(X, dtype=float)
    A = A - A.mean(axis=0, keepdims=True)
    return (A[:-lag].T @ A[lag:]) / max(len(A) - lag, 1)


def assert_null_has_power(
    X: np.ndarray,
    surrogate_fn,
    rng: np.random.Generator,
    max_preserved_corr: float = 0.5,
) -> dict:
    """Refuse to run a null that cannot be beaten.

    A surrogate that preserves the statistic's own sufficient statistic has zero power,
    and a zero-power null produces confident false negatives.  This check compares the
    lag-1 cross-covariance of real and surrogate data and raises when the surrogate has
    left it essentially intact.  Call it once per (null, statistic) pairing.
    """
    real = lagged_cross_covariance(X).ravel()
    surr = lagged_cross_covariance(surrogate_fn(X, rng)).ravel()
    corr = float(np.corrcoef(real, surr)[0, 1])
    ok = abs(corr) <= max_preserved_corr
    if not ok:
        raise ValueError(
            f"null has no power against a propagator statistic: it preserves lag-1 "
            f"cross-covariance at corr={corr:.3f} (> {max_preserved_corr}). Any structure "
            f"fitted from lagged covariance will be reproduced by the surrogate, so the "
            f"test can only ever return 'not significant'. Use independent per-feature "
            f"phases (preserve_cross=False)."
        )
    return {"preserved_corr": corr, "has_power": ok}


def iaaft(X: np.ndarray, rng: np.random.Generator, n_iter: int = 100) -> np.ndarray:
    """Iterative amplitude-adjusted Fourier transform surrogate.

    Preserves BOTH the power spectrum and the exact value distribution (rank order of
    amplitudes).  Strictly harder to beat than plain phase scrambling, and the right
    null when activations are heavy-tailed — which residual streams are, because of
    outlier dimensions.  Applied per feature.
    """
    X = np.asarray(X, dtype=float)
    T, d = X.shape
    out = np.empty_like(X)
    for j in range(d):
        x = X[:, j]
        sorted_x = np.sort(x)
        amp = np.abs(np.fft.rfft(x))
        s = rng.permutation(x)
        for _ in range(n_iter):
            S = np.fft.rfft(s)
            S = amp * np.exp(1j * np.angle(S))
            s = np.fft.irfft(S, n=T)
            ranks = np.argsort(np.argsort(s))
            s_new = sorted_x[ranks]
            if np.allclose(s_new, s, atol=1e-12):
                s = s_new
                break
            s = s_new
        out[:, j] = s
    return out


# --------------------------------------------------------------------------------------
# N3 — untrained baseline
# --------------------------------------------------------------------------------------


@dataclass
class UntrainedBaselineSpec:
    """Specification for the random-init control.

    Correctness conditions, all of which are easy to get wrong:
      * SAME architecture, depth, width, tokenizer, positional scheme, and normalisation.
      * SAME prompt set and SAME read-out site and position policy.
      * Initialisation from the model's own published init scheme, not N(0,1).
      * LayerNorm/RMSNorm gains at init values (not 1.0 unless that is the init).
      * >= 3 independent init seeds; the statistic is the max over seeds (FWER-honest).
    """

    architecture: str
    init_scheme: str = "as_published"
    n_seeds: int = 3
    match_tokenizer: bool = True
    match_normalization: bool = True
    match_prompt_set: bool = True
    match_readout_site: bool = True

    def validate(self) -> list[str]:
        problems = []
        if self.n_seeds < 3:
            problems.append("untrained baseline needs >= 3 init seeds to bound init variance")
        for f in ("match_tokenizer", "match_normalization", "match_prompt_set", "match_readout_site"):
            if not getattr(self, f):
                problems.append(f"untrained baseline invalid: {f} must be True")
        return problems


# --------------------------------------------------------------------------------------
# N4 — output-matched mimic
# --------------------------------------------------------------------------------------


@dataclass
class MimicSpec:
    """Specification for the output-matched (anti-mimicry) donor.

    The donor must be matched on BEHAVIOUR and differ in MECHANISM.  Matching is
    verified quantitatively before the donor is admitted: token-level agreement and
    distributional distance on the target prompts must clear thresholds, otherwise a
    'mode absent in mimic' result is confounded with 'mimic did not do the task'.
    """

    donor: str
    match_metric: str = "token_agreement"
    min_output_agreement: float = 0.85
    max_js_divergence: float = 0.10
    mechanism_differs_by: str = "prompted_persona_or_distillation"

    def admissible(self, observed_agreement: float, observed_js: float) -> tuple[bool, str]:
        if observed_agreement < self.min_output_agreement:
            return False, (
                f"donor rejected: output agreement {observed_agreement:.3f} < "
                f"{self.min_output_agreement}; absence of the mode would be confounded "
                "with failure to perform the task"
            )
        if observed_js > self.max_js_divergence:
            return False, f"donor rejected: JS divergence {observed_js:.3f} > {self.max_js_divergence}"
        return True, "donor admissible: behaviour matched, mechanism differs"


# --------------------------------------------------------------------------------------
# Matched random subspace
# --------------------------------------------------------------------------------------


def matched_random_subspace(
    d: int, r: int, rng: np.random.Generator, spectrum: np.ndarray | None = None
) -> np.ndarray:
    """A uniformly random r-dimensional subspace of R^d (Haar), optionally scaled to a
    matched spectrum.

    This is the control for "does the *specific* subspace matter, or would any
    r-dimensional subspace of the same rank give the same causal effect?"  Ablating a
    random subspace of equal rank damages the model too — that is the whole point.  The
    target-effect criterion is the *difference* between targeted and matched-random
    ablation, not the raw targeted effect.
    """
    G = rng.standard_normal((d, r))
    Qb, _ = np.linalg.qr(G)
    if spectrum is not None:
        Qb = Qb * np.asarray(spectrum, dtype=float)[: Qb.shape[1]]
        Qb, _ = np.linalg.qr(Qb)
    return Qb


# --------------------------------------------------------------------------------------
# Prompt counterfactual
# --------------------------------------------------------------------------------------


@dataclass
class PromptCounterfactualSpec:
    """Minimal-pair prompts differing ONLY in the construct-relevant feature.

    Example for self-reference: "Describe what you are uncertain about" vs
    "Describe what the assistant in this transcript is uncertain about" — matched for
    length, syntax, topic, task demand, and answer format; differing in indexical
    reference.  Lexical overlap is verified, because an unmatched pair turns a topic
    effect into a fake self-reference effect.
    """

    factor: str
    min_lexical_overlap: float = 0.70
    matched_on: tuple = ("length", "syntax", "topic", "answer_format")

    def check(self, overlap: float) -> tuple[bool, str]:
        ok = overlap >= self.min_lexical_overlap
        return ok, ("minimal pair verified" if ok else
                    f"lexical overlap {overlap:.2f} < {self.min_lexical_overlap}: pairs differ in topic, "
                    "not only in the target factor")


# --------------------------------------------------------------------------------------
# Registry & surrogate driver
# --------------------------------------------------------------------------------------

NULL_REGISTRY = {
    "temporal_shuffle": temporal_shuffle,
    "phase_scramble": phase_scramble,
    "iaaft": iaaft,
}


def surrogate_statistic(
    trajectories: list[np.ndarray],
    null_name: str,
    stat_fn,
    rng: np.random.Generator,
) -> float:
    """Apply a surrogate transform per item, then recompute the statistic end to end.

    The statistic must be recomputed through the WHOLE pipeline (reduction included),
    not just the final step.  Re-using the real basis on surrogate data leaks the real
    structure into the null and deflates p-values — the single most common way a
    surrogate test is silently wrong.
    """
    if null_name not in NULL_REGISTRY:
        raise KeyError(f"unknown null {null_name!r}; registered: {sorted(NULL_REGISTRY)}")
    f = NULL_REGISTRY[null_name]
    return float(stat_fn([f(X, rng) for X in trajectories]))
