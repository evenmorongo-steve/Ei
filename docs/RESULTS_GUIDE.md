# Results guide — expected output structure and interpretation

Every run writes to `runs/<UTC-timestamp>-<prereg-hash-prefix>/`.

```
runs/20260920T134500Z-18bb0214/
├── config.snapshot.yaml    exact config used (written BEFORE the run)
├── provenance.txt          git commit, dirty flag, host, python, GPUs, prereg hash
├── uncommitted.patch       present ONLY if the tree was dirty -> run is not reproducible
├── results.json            authoritative machine-readable record
├── report.md               human-readable rendering
├── INTERPRETATION.md       generated reading guide, run-specific
└── preregistration.txt     hash + claim ceiling
```

---

## Reading order that avoids over-claiming

1. **`provenance.txt` — is `git_dirty=true`?** If yes, the run is not reproducible from any
   commit. Treat everything as exploratory regardless of the p-values.
2. **Does the prereg hash match the registered one?** Mismatch means the design changed
   after registration. The run is exploratory. This is checkable, not a matter of trust.
3. **`report.md` §1 — design coverage and blocking gaps.** A design missing the mimic
   control or the rescue arm cannot support a mechanistic claim *however the numbers
   landed*. Design adequacy is logically prior to results.
4. **§3.1 — the phase-scramble null specifically.** Beating temporal shuffle alone is close
   to meaningless; autocorrelated signals always look persistent. Also check
   `null_power_check`: if a null had no power, its "pass" is vacuous.
5. **§3.5 — specificity, not effect size.** `effect` compares to nothing. `specificity`
   (targeted minus matched-random ablation) is the number that supports localisation.
6. **§4 — read the excluded claim before quoting the licensed one.**

---

## Interpreting each artifact field

### `structure.gate_report`

| Field | Reading |
|---|---|
| `n_candidates` | modes passing the descriptive screen. **Not** a count of findings — AR(1) noise passes too |
| `variance_captured_by_reduction` | below ~0.3 means the reduced dim is too small; the propagator is fitted to a fragment |
| `noise_floor_energy_share` | MP prediction for a rank-1 subspace of matched white noise |
| `eigvec_condition` | cond(W). Above ~10³ means modal quantities are numerically unreliable |

### `structure.modes[]`

| Field | Reading |
|---|---|
| `persistence` | `\|λ\|`. **Downward biased** — a lower bound, not a calibrated value (see below) |
| `half_life` | derived from persistence; inherits the same bias |
| `frequency` | cycles per read-out step; 0 = non-oscillatory |
| `energy_share` | projected energy fraction; not exactly additive for oblique modes |
| `consistency` | cross-item eigenvector agreement. Below ~0.3 means the "mode" is item-specific |
| `snr` | energy relative to the MP noise floor |
| `salience` | energy × persistence × consistency — the ranking key |
| `real_dims` | 1 (real mode) or 2 (conjugate-pair invariant plane) |

> **Persistence is attenuated.** With true `ρ = 0.97` on an AR(1) background the estimator
> returns ≈ 0.952 and does **not** converge with more data — the bias is set by in-subspace
> SNR. Quote persistence as a lower bound. Do not compare across conditions with differing
> SNR; `persistence_bias_warning` fires below SNR 10.

### `validation.null_tests`

| Field | Reading |
|---|---|
| `p_value` | floored at `1/(B+1)`. If `p == 1/(B+1)`, the true p may be smaller — increase `B` |
| `ci_low/ci_high` | 95% interval of the **null distribution**, not of the estimate |
| `note` | includes null z and severity at SESOI |
| `decision` | `pass` iff `p < alpha_primary` |

For `output_matched_mimic`, `ratio` is mimic ÷ target. Near 1.0 means the measurement is
tracking output, not mechanism — the most important single number in the file.

### `validation.equivalence`

`estimate` is a **standardised** effect (Glass Δ, in null standard deviations), so
`margin` (the SESOI) is read in the same units. `raw_excess` and `null_sd` are given for
back-conversion. `equivalent: true` licenses an absence claim **only** when the
permutation test did not also reject the null, and only when surrogate resolution was
adequate — otherwise the disposition falls back to `inconclusive`.

### `validation.null_power_check`

`has_power: false` invalidates that null. A null preserving the statistic's sufficient
statistic can only ever return "not significant", producing a confident false negative.

### `validation.causal`

| Field | Reading |
|---|---|
| `effect` | intact − lesioned. **Uninterpretable alone** — any rank-`r` ablation degrades things |
| `specificity` | targeted minus matched-random effect. The localisation number |
| `off_target_effect` | degradation on an unrelated task. Large ⇒ global damage, not specificity |
| `restoration_fraction` | see table below |
| `dose_response.rho` | Spearman ρ; monotone requires > 0.8 |

### `validation.alignment`

Each metric reports `score`, `null_mean`, `excess`, `z`, `p_value`, `interpretation`.
**Always read `excess`, never `score` alone** — CKA between unrelated matrices of realistic
shape is 0.2–0.5, not 0. `_concordant: false` means the three metrics disagree; report the
disagreement rather than selecting the favourable one. `_svcca_sensitivity.stable: false`
means the SVCCA conclusion depends on the variance threshold.

---

## Decision thresholds

| Statistic | Positive evidence | Inconclusive | Evidence of absence |
|---|---|---|---|
| Persistence vs phase-scramble | `p < α_primary` **and** excess > SESOI | `p < α` but excess < SESOI | TOST equivalent within ±SESOI |
| Untrained baseline | trained ≫ max over init seeds | overlapping | trained ≤ baseline ⇒ architectural artifact |
| Output-matched mimic | ratio < 0.50 | 0.50–0.90 | ratio ≈ 1 ⇒ measure tracks output |
| Procrustes cross-arch | `p < 0.05` **and** excess > 0.05 | `p < 0.05`, excess small | indistinguishable from broken-pairing null |
| Specificity | > 0.05 with CI excluding 0 | CI includes 0 | TOST equivalent to 0 |
| Restoration fraction | 0.60 ≤ RF ≤ 1.25 | 0 < RF < 0.60 | RF ≈ 0 with tight CI |

**RF > 1.25 is a failed control, not a strong result.** Over-restoration means the rescue
injected task-relevant information that was not originally present.

---

## The five dispositions

| Disposition | What it licenses | What it does not |
|---|---|---|
| `structure_absent` | no structure of the registered minimum size at the tested site (equivalence established) | absence at other sites, layers, scales, or framings |
| `structure_present_epiphenomenal` | a real, null-beating structure exists | that it plays any functional role |
| `structure_present_causal_model_specific` | load-bearing in **this** model at **this** site | generalisation to other architectures |
| `computational_correlate_supported` | **ceiling** — persistent, mimic-resistant, causal, cross-architecture correlate of a functionally defined construct | anything about phenomenal experience |
| `inconclusive` | nothing; failing checks are enumerated | both presence and absence |

`inconclusive` is a legitimate, informative outcome and is reported as fully as a positive.
It is also the default: the framework does not resolve toward a verdict under pressure.

---

## Common misreadings

| Misreading | Why it is wrong |
|---|---|
| "p < 0.001, so the structure is real" | Only against the nulls that were actually run, and only if each had power |
| "Persistence 0.97 means a 23-step memory" | Persistence is downward biased and uncalibrated; it is a lower bound |
| "CKA = 0.85, so the models agree" | Without the broken-pairing null, 0.85 may be near chance for those shapes |
| "Ablation dropped accuracy 15 points, so the subspace is critical" | Matched-random ablation of equal rank may also drop it 14 |
| "Not significant, so the structure is absent" | Requires TOST against a pre-registered SESOI |
| "It passed everything, so it might be conscious" | The ceiling is a computational claim; the bridge is assumed, not tested |
