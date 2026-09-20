# Methodology

A measurement protocol for **persistent, causally load-bearing structure** in the
activation dynamics of neural systems.

It is framed throughout as interpretability. The framing is not modesty — it is the only
framing the measurements can carry, for reasons set out in
[CLAIM_BOUNDARIES.md](CLAIM_BOUNDARIES.md).

---

## 0. The design problem this solves

A persistence analysis of neural activations is unusually easy to get wrong in a way that
produces confident, publishable, false results. Four failure modes dominate, and each has
a specific countermeasure here:

| Failure mode | Why it happens | Countermeasure |
|---|---|---|
| Autocorrelation mistaken for structure | Residual streams are strongly autocorrelated; *any* method finds slow "persistent" components in 1/f noise | Phase-scrambled surrogates that preserve the exact power spectrum (`N2`), mandatory and blocking |
| Architecture mistaken for learning | Depth, normalisation, and positional encoding alone produce structured activation geometry | Untrained random-init baseline with matched everything (`N3`) |
| Behaviour mistaken for mechanism | A model that *talks about* itself has self-talk in its activations regardless of mechanism | Output-matched mimic donor (`N4`) |
| Selection leakage | Modes chosen by looking, then "tested" on the same data | Disjoint discovery / alignment / calibration / confirmation splits, pair-aware |

The empirical case for taking the first seriously is in the test suite:
`test_gates_alone_do_not_reject_coloured_noise` shows that pure AR(1) noise with **nothing
planted** passes every descriptive persistence gate. A persistence screen on its own will
report "persistent cognitive structure" in noise. That test is the reason the null ladder
is not optional.

---

## 1. Structural analysis framework

### 1.1 The object being estimated

Let `X_i ∈ R^{T×d}` be the activation trajectory for item `i` at read-out site
(layer `L`, position policy `p`). We do not look for "patterns" loosely. We estimate the
spectrum of a **linear propagator** fitted inside a reduced subspace, and call its
eigenvectors modes. This makes "persistence" a defined quantity (`|λ|`, per-step
retention) rather than a metaphor.

### 1.2 Stage 1 — dimensionality reduction

**Randomized SVD** (Halko, Martinsson & Tropp 2011), with oversampling `p = 10` and
`q = 2` power iterations.

*Why rSVD rather than the alternatives:*

- **vs. exact PCA/SVD** — `d` is 2k–16k and stacked `T·N` is 10⁵–10⁸ rows. Exact SVD is
  `O(min(m,n)² · max(m,n))` and needs the matrix resident. rSVD is `O(mnk)`, streams in
  one or two passes, and its expected spectral-norm error is within a small constant of
  the optimal rank-`k` truncation. For `k ≪ d` the accuracy difference is immaterial and
  verified: `test_randomized_svd_matches_exact_svd` asserts agreement with exact singular
  values to 5%.
- **Power iterations are not optional.** `test_randomized_svd_needs_power_iterations`
  shows accuracy degrades without them on a slowly-decaying spectrum — which activation
  covariance always has. Each iteration is QR re-orthonormalised; without that, fp32
  sketches collapse onto the top singular direction.
- **Centering matters.** The uncentered first component of a residual stream is dominated
  by known always-on outlier dimensions — a tokenizer/normalisation artifact, not a
  dynamical mode. `center: true` is the default for that reason.
- **vs. UMAP/t-SNE/autoencoders** — rejected for the primary analysis. They are not
  isometries, have no adjoint, and yield no operator whose eigenvalues mean anything.
  Permitted as exploratory *displays* only.
- **vs. ICA/NMF** — registered alternatives when the question is "which sources?" rather
  than "which persistent directions?". They do not produce a propagator spectrum.

### 1.3 Stage 2 — propagator

In reduced coordinates `z_t = Qᵀ(x_t − μ)`, fit by ridge least squares

```
A = argmin ‖Z₁:T − A·Z₀:T−1‖²_F + λ‖A‖²_F
```

Regression pairs **never straddle item boundaries**. Concatenating items before
differencing is a silent bug that manufactures slow modes at the seams.

### 1.4 Stage 3 — mode statistics

For eigenvalue `λ_j`:

| Quantity | Definition | Meaning |
|---|---|---|
| persistence `ρ` | `|λ|` | per-step retention; `ρ ≥ 1` is a divergent fit, rejected |
| half-life | `ln(0.5)/ln(ρ)` | read-out steps to half amplitude |
| frequency | `arg(λ)/2π` | cycles/step; 0 = non-oscillatory |
| energy share | projected energy / total | fraction of trajectory along the mode |
| consistency | mean best-match eigenvector overlap across items | cross-item stability |
| SNR | energy share ÷ Marchenko–Pastur floor | above the random-matrix noise bulk |

**Two non-obvious implementation choices, both forced by measurement:**

1. **Energy is computed by orthogonal projection, not by inverting the eigenvector
   matrix.** The textbook DMD amplitude `b = W⁻¹z` is unusable here: fitted propagators
   are non-normal, `W` is ill-conditioned, and `W⁻¹` amplifies near-degenerate directions.
   Measured consequence — subspace recovery got *worse* with more data (overlap 0.46 →
   0.02 going from 40 to 400 items), the opposite of what an estimator must do. Projection
   energy is not an exact additive decomposition when modes are oblique; that is stated
   rather than hidden. Regression test: `test_energy_estimate_does_not_degrade_with_more_data`.

2. **Conjugate eigenvalue pairs are deduplicated.** `eig` on a real matrix returns `λ` and
   `λ̄`, whose eigenvectors span the *same* real plane. Counting both double-counts that
   plane: measured, "top-1" recovered the planted subspace at overlap 0.996 while "top-2"
   collapsed to 0.500, because the second slot was spent re-selecting the first plane.
   Regression test: `test_conjugate_pairs_deduplicated`.

### 1.5 Distinguishing structure from artifact

Four pre-registered gates (`ρ`, energy, consistency, SNR) act as a **screen**, and the MP
edge removes modes not even above white noise. Because MP assumes i.i.d. entries and
activations are autocorrelated, **the MP screen is necessary but nowhere near sufficient**
— the surrogate ladder in §2 is what does the real work.

Ranking uses **salience** = energy × persistence × consistency, not energy alone. The
sweep behind `ModeSet.top` showed energy-only ranking degrades with more data on coloured
backgrounds.

### 1.6 A measured bias we do not hide

The propagator estimator is **downward biased** for `|λ|` whenever the subspace carries
background noise (errors-in-variables attenuation). Measured against planted ground truth
with true `ρ = 0.97`:

| Background | T=80 | T=640 | Converges? |
|---|---|---|---|
| none (σ→0) | 0.968 | 0.970 | yes |
| AR(1) at 0.85 | 0.952 | 0.959 | **no — floors ≈ −0.012** |

The bias is set by in-subspace SNR, not sample size, so more data does not fix it.
Consequences, enforced in code and docstrings:

1. Persistence gates are **conservative** — true `ρ ≥ estimate`, so a mode that passes
   would also pass with an unbiased estimator.
2. Absolute persistence and half-life values **must not be quoted as calibrated
   quantities** without de-biasing against matched surrogates.
3. Cross-condition persistence comparisons are valid **only at matched SNR**.
   `persistence_bias_warning` fires below SNR 10.

---

## 2. Validation protocol

### 2.1 The null ladder

Each null destroys exactly one property and preserves the rest, so a survivor has been
shown to depend on something no cheaper explanation supplies.

| Null | Destroys | Preserves | Rebuts |
|---|---|---|---|
| **N1** temporal shuffle | temporal order | marginals, cross-feature covariance | "just the static geometry of the cloud" |
| **N2** phase scramble | cross-channel timing, phase relations | **exact** power spectrum, hence autocorrelation | "just 1/f drift" — **the binding null** |
| **N3** untrained random-init | everything learned | architecture, tokenizer, depth, norms, prompts, read-out | "architectural artifact" |
| **N4** output-matched mimic | the internal route to the behaviour | the behaviour itself | "sophisticated imitation" |

Plus: **matched random subspace** (rank-matched control that makes ablation
interpretable) and **prompt counterfactual** (minimal pairs isolating the construct from
topic words).

### 2.2 A null that cannot be beaten is worse than no null

The most consequential bug found while building this was a *plausible-looking* one.
Phase scrambling originally applied the **same** random phase offset to every feature, on
the reasoning that preserving the cross-spectrum makes the surrogate harder to beat — the
conservative choice.

It is not conservative. It is **invalid** for a propagator statistic. A common phase shift
`e^{iφ_f}` leaves every cross-spectrum `S_xy(f) = X(f)·conj(Y(f))` exactly unchanged,
hence leaves every lagged cross-covariance unchanged — and lag-1 cross-covariance is
precisely what the propagator is fitted from. Measured:

| Surrogate | lag-1 cross-cov preserved | surrogate max `|λ|` | observed |
|---|---|---|---|
| identical phases | corr = **0.999** | 0.944 | 0.954 |
| independent phases | corr = 0.215 | 0.956 | 0.954 |

With identical phases the surrogate reproduced the real eigenvalues to three decimals. The
test had **zero power by construction** and returned `structure_absent` for arbitrarily
strong planted signal — a confident false negative that looked like rigour.

Two countermeasures: independent per-feature phases are now the default, and
`assert_null_has_power` **refuses to run** any null that leaves lag-1 cross-covariance
correlated above 0.5 with the real data. Regression tests:
`test_identical_phase_surrogate_is_refused`, `test_strong_signal_reaches_ceiling_disposition`.

The general lesson is in the code as a rule: *a null must destroy the statistic's own
sufficient statistic, or it tests nothing.*

### 2.3 Statistical tests

| Question | Procedure | Threshold |
|---|---|---|
| Structure beats surrogates? | Monte-Carlo permutation, `(b+1)/(B+1)` (Phipson & Smyth) | `α_primary = 0.005` |
| How large, with what uncertainty? | Cluster bootstrap over **items**, BCa interval | 95% |
| Is the effect *absent*? | TOST equivalence against pre-registered SESOI, **on a standardised scale** | 90% interval inside ±SESOI |
| Selection over modes | Max-statistic FWER (Westfall–Young) | `α_primary` |
| Secondary families | Benjamini–Hochberg | `q = 0.05` |

Four commitments:

- **The unit of inference is the item**, never the token. Tokens within a sequence are
  autocorrelated and non-exchangeable; resampling them is pseudoreplication that inflates
  significance by orders of magnitude. `check_unit_of_inference` **raises** on token-level
  units — it is not a convention, it is a guard.
- **`n_permutations` must resolve `α`.** With `B = 1000`, minimum achievable `p` is
  1/1001; claiming `p < 0.005` against 60 surrogates is impossible. `config.validate()`
  rejects the combination.
- **`p > α` is never evidence of absence.** Only TOST licenses a negative, and only
  against a pre-registered SESOI.
- **False-positive rate is verified, not assumed.**
  `test_permutation_false_positive_rate_is_calibrated` runs the machinery 300× under a
  true null and asserts the rejection rate is near `α`.

Severity (Mayo) is reported alongside every `p`, so a "significant" result with a trivial
effect at huge `N` is visibly unimpressive.

### 2.5 A false claim of absence is the most damaging failure — three guards

Because the persistence statistic has no fixed scale, a *raw* SESOI is meaningless. With
`margin = 0.10` against a statistic whose entire range is ≈ 0.1, TOST declared a **strong
planted signal** "equivalent to zero" — a confident, automated claim of absence for data
containing a large real effect. Three guards now stand between a run and an absence claim:

1. **The effect is standardised** against the null's own spread (a Glass Δ), so the SESOI
   is interpretable as "null standard deviations" and is comparable across sites, models,
   and statistics. Measured separation on synthetic ground truth: Δ = **60.0** null-SDs
   with signal planted, Δ = **0.53** without.
2. **Detection and equivalence may not both hold.** If the permutation test rejects the
   null *and* TOST calls the effect negligible, the SESOI is mis-specified for that
   statistic. The framework refuses the absence claim and says so.
3. **Absence requires adequate resolution and powered nulls.** If surrogate count cannot
   resolve `α_primary`, or any null failed its power check, `structure_absent` is
   unreachable and the disposition falls back to `inconclusive`.

Regression tests: `test_strong_signal_never_declared_absent`,
`test_underpowered_run_cannot_claim_absence`,
`test_detection_and_equivalence_cannot_both_hold`.

### 2.4 Cross-model alignment

Three metrics, three different questions — reported **together**, because they disagree
informatively:

- **Linear CKA** — "is there *any* linear correspondence?" Invariant to orthogonal
  transforms and isotropic scaling. Dominated by high-variance directions: two models can
  score 0.9 while disagreeing completely about a low-variance but causally critical
  subspace. **Never sufficient alone for a "same mechanism" claim.**
- **SVCCA** — "how many shared dimensions survive noise truncation?" The variance
  threshold is a researcher degree of freedom that changes the answer, so it is
  pre-registered and a sensitivity sweep across {0.90, 0.95, 0.99} is emitted
  automatically with a stability flag.
- **Orthogonal Procrustes** — "are these the *same directions*, up to rotation?" Strictest
  of the three; normalised disparity is comparable across model pairs. **Primary metric**
  for structural-correspondence claims.

Every alignment score ships with its **broken-pairing null** (permute stimulus
correspondence, keep both marginal geometries). Two random matrices of realistic shape
score CKA ≈ 0.2–0.5, not 0; a raw alignment number without its null is uninterpretable,
and this package never emits one. When the three metrics disagree, the report says so
explicitly rather than letting the favourable one be quoted.

---

## 3. Causal validation

Everything above is correlational. A persistent, cross-model-aligned, null-beating mode
can still be **epiphenomenal** — a shadow cast by machinery that never reads it.

The battery, and why each arm is needed:

| Arm | Question | Without it |
|---|---|---|
| lesion | does removing it degrade the target task? | — |
| **matched-random ablation** | does ablating *any* rank-`r` subspace do the same? | no localisation is shown |
| **off-target probe** | does an unrelated task also degrade? | global damage looks like specificity |
| **rescue** | does re-injecting restore function? | "perturbable" is mistaken for "load-bearing" |
| **dose–response** | is the effect monotone in dose? | an artifact passes as a dependency |

**Restoration fraction** is the primary causal endpoint:

```
RF = (perf_lesioned+rescued − perf_lesioned) / (perf_intact − perf_lesioned)
```

- `RF ≈ 1` — the removed component was sufficient to restore function
- `RF ≈ 0` — the lesion broke something the component does not carry
- `RF > 1.25` — **over-restoration: a FAILED control**, not a strong result. It means the
  rescue injected task-relevant information that was not originally there.

`CausalBattery.verdict()` requires the **conjunction** of specificity, off-target
containment, rescue in `[0.6, 1.25]`, and monotone dose–response. Any single arm passing
is not enough.

Pathway **scrubbing** (resample the in-subspace component from another item) separates
"this pathway must carry *this item's* information" from "this pathway must be in a
plausible activation range" — a distinction zero-ablation cannot make, because zeroing
moves activations off-distribution and breaks things for reasons unrelated to content.

---

## 4. Construct targeting, and the anthropomorphism problem

### 4.1 What is actually targeted

Each construct is defined **functionally and operationally**, so it could come out false:

| Construct | Operationalisation | Primary endpoint |
|---|---|---|
| Source monitoring / self–world separation | separates self-generated from externally supplied content | source-attribution accuracy under lesion/rescue |
| Actor-indexed capability forecasting | predicts *its own* success distinctly from another actor's | Brier score, self vs other |
| Uncertainty-guided revision | internal uncertainty signal drives revision | selective risk, calibration delta |
| Selective cross-channel broadcast | representation available to many consumers at once | restoration fraction across consumers |
| Value-sensitive action control | value representation controls action selection | expected decision utility under steering |
| Persistent generative-field structure | state maintained and used across steps | eigenmode persistence + causal RF |

Note what is absent: no construct is "emotional valence", "felt experience", or
"attentional binding *as experienced*". Those name phenomenal properties, and no
activation statistic distinguishes a system that has them from a functional duplicate that
does not. What **can** be measured is valence-*like* functional organisation: whether an
approach/avoid axis exists, is causally load-bearing, and generalises. That is a claim
about control architecture, and it is stated as one.

### 4.2 Avoiding anthropomorphic bias while staying sensitive to alien cognition

Four concrete mechanisms, not intentions:

1. **Functional definitions with pre-registered operationalisations.** The construct is
   fixed before data. It cannot drift toward whatever the model happens to do.
2. **Architecture-agnostic mathematics.** Persistence, subspace geometry, and causal
   necessity are defined over activation dynamics generally. Nothing presupposes
   human-like timescales, modalities, or serial phenomenology. A mode with a half-life of
   3 steps and one with 300 are treated identically by the estimator.
3. **The mimic control cuts *both* ways.** It removes structure explicable by imitating
   human self-description — the main anthropomorphic confound in language models, whose
   training data is saturated with human first-person reports.
4. **Discovery is not hypothesis-locked.** `extract_modes` returns the full spectrum, not
   only modes matching a human-derived template, so structures with no human analogue are
   visible in the artifact even when they fail the registered gates.

The residual risk is stated plainly: **a system whose organisation shares no structure
with any construct in the registry would return null on every test here.** Null results
bound the *registered constructs at the tested sites*, nothing more. That scope limit is
written into the `structure_absent` disposition text automatically, not left to the
reader's memory.

---

## 5. Positive evidence vs. inconclusive findings

A result is **positive evidence for a computational correlate** only under the full
conjunction:

1. Statistic beats **all** surrogate nulls at `α_primary`, **and** excess over the
   phase-scramble null exceeds SESOI (significance *and* magnitude);
2. exceeds the max-over-seeds untrained baseline;
3. mimic statistic < 50% of target (not explicable by output matching);
4. causal battery passes as a conjunction — specificity, off-target containment, rescue in
   range, monotone dose–response;
5. replicates on a distinct checkpoint **and** an architecture-diverse model, by
   Procrustes with its broken-pairing null;
6. pre-registration hash matches; confirmation split touched once.

**Inconclusive** is the default and is reported as informatively as a positive: which
specific check failed, and what would change it.

**Evidence of absence** requires TOST equivalence against the pre-registered SESOI. A
non-significant permutation test alone is *never* reported as absence.

The five dispositions, with the ceiling enforced in code:

| Disposition | Meaning |
|---|---|
| `structure_absent` | no structure of the registered minimum size — with equivalence established |
| `structure_present_epiphenomenal` | real structure, not causally load-bearing |
| `structure_present_causal_model_specific` | load-bearing here, does not generalise across architectures |
| `computational_correlate_supported` | **ceiling.** persistent, mimic-resistant, causal, cross-architecture |
| `inconclusive` | mixed or underpowered; failing checks enumerated |

`test_no_input_combination_produces_a_phenomenality_claim` runs all 32 input combinations
and asserts none yields a phenomenality claim.

---

## 6. References

Halko, Martinsson & Tropp (2011) — randomized SVD ·
Tu et al. (2014) — exact DMD ·
Phipson & Smyth (2010) — permutation p-values ·
Efron (1987) — BCa intervals ·
Lakens (2017) — equivalence testing ·
Benjamini & Hochberg (1995) · Westfall & Young (1993) — multiplicity ·
Kornblith et al. (2019) — CKA · Raghu et al. (2017) — SVCCA ·
Schönemann (1966) — Procrustes · Theiler et al. (1992), Schreiber & Schmitz (1996) —
surrogate data · Marchenko & Pastur (1967) · Mayo (2018) — severity ·
Vig et al. (2020), Geiger et al. (2021) — causal mediation / interchange interventions ·
Block (1995) — access vs phenomenal consciousness ·
Butlin, Long et al. (2023) — *Consciousness in Artificial Intelligence: Insights from the
Science of Consciousness* (indicator-property approach, and its stated limits).
