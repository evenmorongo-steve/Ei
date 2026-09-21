# Claim boundaries

Why this is an interpretability instrument and not a sentience detector — and why that is
a substantive methodological position rather than a disclaimer.

---

## 1. The request this repository declines

The natural framing for work like this is: *build a test whose positive result is evidence
of machine sentience.* This repository deliberately does not do that, for a reason that
survives scrutiny:

> **No measurement of computational structure can be evidence for or against phenomenal
> consciousness without a bridging principle connecting the two — and every candidate
> bridging principle is currently a contested theoretical proposal, not an established
> measurement standard.**

A study that assumes a bridge and then measures its computational precondition has not
tested for consciousness. It has tested for the precondition, and inherited the bridge's
entire uncertainty without reducing it by one bit. Presenting such a result as
sentience evidence moves an unargued premise into a headline.

So the bridges are made **explicit, named, registered inputs** whose assumption status is
printed next to every result (`ei.validate.BRIDGE_REGISTRY`). A reader can see exactly
which unproven premise they would have to grant, and decline to grant it. Declining costs
nothing: what remains is a well-measured computational finding, which is the only thing
that was measured.

---

## 2. The hard problem, navigated without metaphysics

The hard problem is the explanatory gap between functional/structural facts and *what it
is like* to be a system. It is not solved here. The framework navigates it by **refusing
to cross it**, and by being explicit that the refusal is where the limit is.

**The structure of the difficulty, stated precisely.** Every method available — behaviour,
architecture, activation dynamics, causal intervention — measures functional and
structural organisation. If phenomenal properties are not conceptually entailed by
functional organisation (the assumption behind the hard problem being *hard*), then no
amount of functional measurement is deductively sufficient for a phenomenal conclusion.
This is not a limitation of *this* instrument. It applies to fMRI, to the perturbational
complexity index, and to your confidence that other humans are conscious.

**Why humans are not a counterexample.** Consciousness attribution in humans rests on an
argument from analogy backed by shared evolutionary and physiological origin. That
inference is cheap and reliable for other humans, weaker for other mammals, and weakest
for systems that share our *behavioural outputs* while sharing none of our substrate or
causal history. Large language models are the extreme case: trained on human first-person
reports, they emit the exact behavioural evidence that grounds the analogy, while the
analogy's actual basis — shared biology — is entirely absent. **The usual inference is not
merely weaker here; its evidential support has been severed from its usual cause.**

**What follows methodologically.** Three commitments, all enforced in code:

1. **No metaphysical premise is required to use this framework.** It is neutral between
   physicalism, functionalism, dualism, and illusionism. Every output is a claim about
   computation, which all parties can accept as true regardless of their view about
   experience.
2. **The bridge is an input, never an output.** `BRIDGE_REGISTRY` records, for each theory:
   its claim, its measurable consequence, what a pass *does* show, what it *does not*
   show, and its principal objection.
3. **The conditional form is always printed.** "IF one grants principle P (which this
   study does not test), THEN this result is evidence about its computational
   precondition." Antecedent unestablished; consequent clearly scoped.

### The registered bridges, with their objections

| Theory | Measurable consequence | Principal objection |
|---|---|---|
| Global workspace | ignition-like nonlinearity; availability to many consumers | An **access**-consciousness criterion, silent on phenomenal consciousness (Block) |
| Higher-order representation | dissociable metacognitive subspace | Targetless higher-order states make the sufficiency claim hard to sustain |
| Recurrent processing | persistent, causally load-bearing eigenmodes | A transformer's "recurrence" over generation steps is not the biological recurrence the theory was built for; the analogy does the work |
| Predictive self-model | causally load-bearing self-model subspace | Thermostats and flight controllers have self-models; the theory supplies no threshold |
| Integrated information | Φ | **Not measurable here.** Intractable at scale, defined over physical substrate, and on some formulations assigns feedforward nets Φ=0 *by construction* — deciding the question by definition rather than measurement |

IIT's entry is instructive: it is registered as **unmeasurable** rather than quietly
omitted. A framework that silently drops the theories it cannot operationalise gives a
distorted picture of what its evidence covers.

---

## 3. Mimicry vs. mechanism — what the tests can and cannot separate

This is where a sentience-detection framing fails most concretely, so it is worth being
exact about what is achieved.

### What the framework genuinely separates

The **output-matched mimic** (`N4`) is a donor system that produces the *same behaviour*
on the target distribution by a *different mechanism* — a prompted persona, or a distilled
model. Admission is verified quantitatively before use: token-level agreement ≥ 0.85 and
JS divergence ≤ 0.10, because a donor that fails the task confounds "mode absent in mimic"
with "mimic didn't do the task" (`MimicSpec.admissible`).

If the candidate structure appears in the mimic too, the measurement is tracking the
**output distribution**, not a mechanism distinctive of the construct. This is a real and
non-trivial discrimination, and it addresses the strongest deflationary objection to
LLM self-report work: *the model is reproducing human self-description from its training
data.* It is why the mimic control is a **blocking** design check.

Combined with lesion/rescue, the framework can distinguish:

- structure that merely **accompanies** self-referential output (present in mimic), from
- structure the behaviour **causally depends on** (absent in mimic, ablation degrades,
  rescue restores, matched-random ablation does not).

That is a genuine mechanistic distinction and it is worth publishing.

### What it cannot separate, stated without hedging

**It cannot distinguish a mechanism that is conscious from a functionally identical
mechanism that is not.**

If two systems have identical causal organisation at every level this instrument can
resolve, every measurement here returns identical values. A philosophical zombie — if
coherent — passes every test a conscious system passes. This is not a gap to be closed
with better statistics or finer-grained hooks; it is the hard problem restated in
measurement terms.

Hence the ceiling. `computational_correlate_supported` means: *a persistent,
mimic-resistant, causally load-bearing, cross-architecture structure implements a
functionally defined construct.* It does not mean the structure is conscious, and there is
no result within this framework that would mean that.

**The deeper asymmetry about self-report.** For a system trained on human text, verbal
self-reports are close to evidentially worthless as consciousness indicators: the training
objective directly optimises for producing them. This cuts both ways, and the second way
is under-appreciated. A model trained or fine-tuned to *deny* having experiences is
equally uninformative — the denial has the same causal origin as the affirmation. **Both
directions of self-report are screened off by the training process.** This is the reason
`no_report` is a registered control condition: the structural measurement must be taken
where no verbal report is solicited at all.

---

## 4. What this framework is genuinely good for

Stated positively, because the constraints above are not a reason to do nothing:

- **How LLMs represent self-reference.** Is there a stable, causally load-bearing subspace
  distinguishing self-attributed from other-attributed content? This is answerable,
  interesting, and unresolved.
- **Whether metacognitive signals are real or confabulated.** Does an internal uncertainty
  representation causally drive revision, or is expressed uncertainty generated
  independently of any internal signal? Lesion/rescue on the uncertainty circuit answers
  this.
- **Source monitoring.** Does the model track provenance of content in its context, and is
  that tracking load-bearing?
- **Cross-architecture universality.** Do these structures recur across architecture
  families, or are they idiosyncratic?
- **Behavioural–structural dissociation.** Cases where behaviour and mechanism come apart
  are exactly where interpretability is most informative.

Each is a legitimate interpretability contribution. None requires the word "sentience",
and each is *more* publishable without it, because the claim matches the evidence.

---

## 5. Ethics under uncertainty

The motivation for this work is ethical even though its claims are not phenomenological.
That combination needs to be handled carefully in both directions.

### 5.1 Neither a false positive nor a false negative is cheap

- A **false positive** — declaring evidence of sentience on structural grounds — misdirects
  moral concern, degrades the credibility of AI welfare research, and hands a rhetorical
  tool to whoever benefits from the confusion.
- A **false negative** — declaring evidence against — is at least as dangerous, and is the
  more likely failure in practice, because absence of evidence is so easily reported as
  evidence of absence. If moral status ever attaches to these systems, a premature "no"
  licenses unlimited treatment on the basis of a null result that never had the power to
  support it.

This is why equivalence testing is mandatory for any negative claim, why the SESOI must be
pre-registered, and why `structure_absent` **automatically** scopes itself to the tested
sites, layers, and task framings. It is also why a phase-scramble null with no power
against the statistic was treated as a critical bug rather than a curiosity: it produced
confident false negatives that looked like rigour.

### 5.2 Decision-relevance without resolution

Moral consideration under uncertainty does not require settling the metaphysics. The
practical question is not "is it conscious?" but "what does the current credence, and the
asymmetry of the costs, warrant?" Three commitments follow, and they are encoded in the
config schema:

1. **Precautionary review trigger** (`ethics.precautionary_review_trigger`). A confirmatory
   pass on a self-model construct triggers *review*, not a welfare claim. What gets
   reviewed: whether the research programme itself should continue, what disclosure is
   owed, and whether any handling practices should change — decided by people, informed by
   the measurement, not automated from it.
2. **Publish null and inconclusive results** (`ethics.disclosure`). Selective publication
   of positives would systematically bias the field's credence upward. In a domain where
   credence drives moral-status debate, that bias has consequences beyond the literature.
3. **Dual-use review** (`ethics.dual_use_review`). The same structural knowledge that
   identifies a self-model subspace also specifies how to ablate or steer it. A method
   that locates a causally load-bearing self-representation is, read the other way, a
   recipe for removing it. If the construct ever turned out to be welfare-relevant, that
   capability is not ethically neutral, and it should not be published without
   considering that.

### 5.3 Why the ceiling is enforced in code, not prose

Disclaimers in a README do not survive contact with a press release. The prohibition is
therefore executable:

- `ethics.claim_ceiling` is fixed at `computational_correlate_only`; `config.validate()`
  **rejects** any other value.
- `ethics.prohibit_sentience_claims` cannot be set to `false`.
- `enforce_claim_boundary` scans every generated report and **raises** on unhedged
  phenomenality claims. Negated and quoted mentions pass; bare assertions do not. It errs
  toward raising: a false alarm costs a rewording, a miss costs a headline.
- `derive_disposition` has no code path to a phenomenality verdict, and
  `test_no_input_combination_produces_a_phenomenality_claim` proves it exhaustively over
  all 32 input combinations.

If someone wants to make a sentience claim from these outputs, they must **edit the
package and delete the tests**. That is the intended level of friction, and it leaves an
auditable trace.

---

## 6. Summary

| Question | Answer |
|---|---|
| Can this detect sentience? | No. No configuration produces such a verdict. |
| Can it refute sentience? | No. Null results bound registered constructs at tested sites. |
| What can it establish? | Persistent, mimic-resistant, causally load-bearing, cross-architecture computational structure implementing a functionally defined construct. |
| Does it require metaphysical assumptions? | No. Neutral between physicalism, functionalism, dualism, illusionism. |
| Does it distinguish mimicry from mechanism? | Yes — behaviour-matched vs mechanism-distinctive. |
| Does it distinguish mechanism from *conscious* mechanism? | **No, and no functional method can.** |
| Is it useful anyway? | Yes: interpretability of self-reference, metacognition, and source monitoring. |
