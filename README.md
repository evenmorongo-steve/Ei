# Ei — eigenmode instrumentation for persistent-structure interpretability

A laboratory package for detecting and validating **persistent, causally load-bearing
structure** in the activation dynamics of neural systems.

> **Claim ceiling.** This is an interpretability instrument, not a sentience detector.
> Its strongest possible output is *"a persistent, mimic-resistant, causally load-bearing,
> cross-architecture computational correlate of a functionally defined construct."*
> No configuration, and no combination of results, produces a claim about phenomenal
> experience. The ceiling is enforced in code and proved by test, not promised in prose —
> see [docs/CLAIM_BOUNDARIES.md](docs/CLAIM_BOUNDARIES.md).

---

## Quickstart

```bash
python -m venv .venv && .venv/bin/pip install -e '.[dev]'

.venv/bin/python -m ei.cli selftest                        # calibrate on known ground truth
.venv/bin/python -m ei.cli validate configs/experiment.yaml -v   # score the design, pre-data
.venv/bin/python -m ei.cli run configs/experiment.yaml --fast    # end-to-end on synthetic data
./scripts/launch.sh configs/experiment.yaml                # auto-detect Slurm/K8s/Ray/local
```

Artifacts land in `runs/<timestamp>-<prereg-hash>/`. Read `INTERPRETATION.md` first.

---

## What problem this solves

Persistence analysis of neural activations is unusually easy to get wrong in ways that
produce confident, publishable, false results. The package is organised around four
specific failure modes:

| Failure mode | Countermeasure |
|---|---|
| Autocorrelated noise mistaken for structure | Phase-scrambled surrogates preserving the exact power spectrum — **blocking** |
| Architecture mistaken for learning | Untrained random-init baseline, matched on everything else |
| Behaviour mistaken for mechanism | Output-matched mimic donor, admitted only if behaviour-matched |
| Selection leakage | Pair-aware disjoint discovery/alignment/calibration/confirmation splits |

The first is not hypothetical. `test_gates_alone_do_not_reject_coloured_noise` shows that
pure AR(1) noise with **nothing planted** passes every descriptive persistence gate. A
persistence screen alone will report "persistent cognitive structure" in noise.

---

## The instrument is calibrated against known ground truth

Before being pointed at a model, the pipeline is validated on synthetic data whose answer
is known — planted modes on an AR(1) background, the realistic hard case.

| Check | Result |
|---|---|
| Planted-subspace recovery (SNR 3.5) | overlap **0.975** vs chance 0.031 |
| Recovery with no signal planted | correctly returns `structure_absent` |
| Strong signal, full pipeline | reaches ceiling disposition, `p = 0.001` |
| Permutation false-positive rate under true null | **≈ α** (300 trials) |
| BCa interval coverage | ≈ 95% nominal |

Three bugs were found *only* because ground truth was known, and each is now a regression
test:

1. **Conjugate eigenvalue pairs were double-counted.** `eig` returns `λ` and `λ̄` spanning
   the same real plane; counting both collapsed recovery from 0.996 (top-1) to 0.500
   (top-2).
2. **Modal energy via `W⁻¹` degraded with more data** (0.46 → 0.02 from 40 to 400 items)
   because fitted propagators are non-normal and `W` is ill-conditioned. Replaced with
   orthogonal projection energy.
3. **The phase-scramble null had zero power** — see below.

### The null that could not be beaten

Phase scrambling originally applied the *same* random phase offset to every feature, on
the reasoning that preserving the cross-spectrum makes the surrogate harder to beat.

That is not conservative — it is **invalid** for a propagator statistic. A common phase
shift leaves every cross-spectrum, and hence every lagged cross-covariance, exactly
unchanged — and lag-1 cross-covariance is precisely what the propagator is fitted from.

| Surrogate | lag-1 cross-cov preserved | surrogate max \|λ\| | observed |
|---|---|---|---|
| identical phases | corr = **0.999** | 0.944 | 0.954 |
| independent phases | corr = 0.215 | 0.956 | 0.954 |

The test had **zero power by construction** and returned `structure_absent` for
arbitrarily strong planted signal — a confident false negative that looked like rigour.
Now: independent per-feature phases by default, and `assert_null_has_power` **refuses** any
null preserving lag-1 cross-covariance above corr 0.5.

*A null must destroy the statistic's own sufficient statistic, or it tests nothing.*

---

## Package layout

```
src/ei/
  modes.py      randomized SVD -> ridge propagator -> eigenmodes; MP noise floor
  nulls.py      the null ladder + the null-power guard
  align.py      CKA / SVCCA / Procrustes, each with its broken-pairing null
  causal.py     ablate / patch / scrub / steer; restoration fraction; dose-response
  stats.py      permutation, cluster bootstrap + BCa, TOST, FDR, max-stat FWER
  extract.py    ActivationSource protocol + [PLACEHOLDER] backends + synthetic ground truth
  config.py     YAML schema, validation, pre-registration hashing
  validate.py   design scoring, disposition engine, claim-boundary enforcement
  pipeline.py   EXTRACT -> TRANSFORM -> VALIDATE -> REPORT
  report.py     artifact rendering (claim-boundary checked)
configs/experiment.yaml     the pre-registration
scripts/launch.sh           Slurm / K8s / Ray / local auto-detection
docs/METHODOLOGY.md         full protocol and its justifications
docs/CLAIM_BOUNDARIES.md    the philosophical position
docs/RESULTS_GUIDE.md       how to read the artifacts
```

---

## Connecting a real model

The analysis path ships with **no model weights and no network access**. Implement
`ActivationSource` (3 methods) and pass it to `run_experiment`. A TransformerLens skeleton
is in `extract.py` under `[PLACEHOLDER: ACTIVATION EXTRACTION]`.

```python
class MySource:
    def trajectory(self, item, readout) -> np.ndarray: ...   # (T, d)
    def behavior(self, item) -> dict:  ...                   # {'correct','confidence'}
    def with_intervention(self, item, readout, hook): ...     # re-run with hook applied
```

Then: `run_experiment(cfg, source=MySource(), alt_sources={...})`.

`alt_sources` supplies the null ladder's comparison systems (`untrained`, `mimic`,
`distinct_checkpoint`, `architecture_diverse`). Omitting them is not silently tolerated —
the run records the missing control as a failing check and the disposition is capped.

---

## Statistical commitments

- **Unit of inference is the item**, never the token. `check_unit_of_inference` *raises* on
  token-level units — pseudoreplication is a guard, not a convention.
- **`n_permutations` must resolve `α`.** Claiming `p < 0.005` against 60 surrogates is
  impossible; `config.validate()` rejects it.
- **`p > α` is never evidence of absence.** Only TOST against a pre-registered SESOI
  licenses a negative.
- **Every alignment score ships with its null.** Two random matrices score CKA ≈ 0.2–0.5,
  not 0. A bare alignment number is never emitted.
- **Metric disagreement is reported**, not resolved by post-hoc selection.

---

## Reproducibility

`environment.yml` (conda, `nomkl` + pinned OpenBLAS threads for deterministic FP reduction
order) · `Dockerfile` (multi-stage, non-root, CPU default + optional CUDA target; the build
**fails** if the instrument cannot recover a planted mode) · `scripts/launch.sh` (records
git commit, dirty-tree patch, host, GPUs, and the pre-registration hash *before* running).

The pre-registration hash covers every result-affecting field and excludes runtime knobs
that cannot change a result, so re-running on different hardware does not falsely
invalidate the registration.

```bash
.venv/bin/python -m pytest -q            # 79 tests
.venv/bin/python -m pytest -q -m "not slow"
```

---

## Scope

| Question | Answer |
|---|---|
| Can this detect sentience? | No. No configuration produces such a verdict. |
| Can it refute sentience? | No. Nulls bound registered constructs at tested sites only. |
| What can it establish? | Persistent, mimic-resistant, causally load-bearing, cross-architecture computational structure. |
| Does it need metaphysical assumptions? | No. Neutral between physicalism, functionalism, dualism, illusionism. |
| Does it separate mimicry from mechanism? | Yes. |
| Does it separate mechanism from *conscious* mechanism? | **No — and no functional method can.** |

Use it for interpretability: publish findings about how models represent self-reference,
metacognition, and source monitoring. Don't call it sentience detection.

Apache-2.0.
