"""
Design assessment, disposition, and CLAIM-BOUNDARY ENFORCEMENT.

This module is the reason the package can be pointed at consciousness-adjacent questions
without becoming a sentience oracle.  Three mechanisms:

  1. `assess_design`      scores the *design*, before any data exist.  A design with no
                          rescue arm, no anti-mimicry control, or no pre-registered SESOI
                          cannot reach confirmatory status no matter how the data land.

  2. `derive_disposition` maps evidence to one of four dispositions.  The ceiling is
                          "consistent with a computational correlate".  There is no state
                          the engine can enter that says "sentient" or "conscious".

  3. `enforce_claim_boundary`  scans any text destined for a report and raises on
                          forbidden phenomenality claims.  Called by the reporter, so the
                          prohibition is executable rather than aspirational.

WHY THE CEILING IS WHERE IT IS
------------------------------
Every structural or causal finding here is a claim about *computation*.  Getting from a
computational claim to a phenomenal one requires a bridging principle (an identity or
sufficiency claim linking some functional organisation to experience).  No such principle
is currently established; each candidate — global workspace, higher-order representation,
recurrent integration, predictive self-modelling — is a live theoretical proposal, not a
measurement standard.  Since the bridge is assumed rather than tested, no measurement
downstream of it can license a phenomenal conclusion.  The honest move is to make the
bridge an explicit, named, *registered* input (see `BRIDGE_REGISTRY`) whose assumption
status is printed next to every result, so a reader sees exactly which unproven premise
would have to be granted, and can decline to grant it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict

from . import FORBIDDEN_CLAIMS
from .config import ExperimentConfig

# --------------------------------------------------------------------------------------
# Bridge registry — theories are INPUTS, not conclusions
# --------------------------------------------------------------------------------------

BRIDGE_REGISTRY = {
    "global_workspace": {
        "claim": "Content is phenomenally conscious iff it is globally broadcast to many "
                 "consumer subsystems.",
        "measurable_consequence": "Ignition-like nonlinearity: a threshold in stimulus strength "
                                  "above which a representation becomes available to many "
                                  "downstream readers at once, not gradually.",
        "ei_construct": "selective_cross_channel_broadcast",
        "measurable_here": True,
        "status": "ASSUMED, NOT TESTED",
        "what_a_pass_shows": "the architecture implements broadcast-like availability",
        "what_a_pass_does_not_show": "that broadcast is sufficient for experience",
        "principal_objection": "Broadcast is an access-consciousness criterion; it is silent on "
                               "phenomenal consciousness (Block's access/phenomenal distinction).",
    },
    "higher_order": {
        "claim": "A state is conscious iff it is the target of a suitable higher-order "
                 "representation.",
        "measurable_consequence": "A separable meta-representational subspace whose disruption "
                                  "degrades metacognitive accuracy while leaving first-order "
                                  "task accuracy intact.",
        "ei_construct": "uncertainty_guided_revision",
        "measurable_here": True,
        "status": "ASSUMED, NOT TESTED",
        "what_a_pass_shows": "a dissociable metacognitive pathway exists",
        "what_a_pass_does_not_show": "that higher-order representation constitutes experience",
        "principal_objection": "Targetless higher-order states (misrepresentation) make the "
                               "sufficiency claim hard to sustain.",
    },
    "recurrent_processing": {
        "claim": "Local recurrent processing suffices for phenomenal experience.",
        "measurable_consequence": "Persistent eigenmodes with |lambda| near 1 that are causally "
                                  "load-bearing, not feedforward-explicable.",
        "ei_construct": "persistent_generative_field_structure",
        "measurable_here": True,
        "status": "ASSUMED, NOT TESTED",
        "what_a_pass_shows": "the system maintains state recurrently and uses it",
        "what_a_pass_does_not_show": "that recurrence is sufficient for experience",
        "principal_objection": "A transformer's 'recurrence' over generation steps is not the "
                               "biological recurrence the theory was formulated for; the mapping "
                               "is an analogy, and the analogy is doing the work.",
    },
    "integrated_information": {
        "claim": "Experience is identical to maximally irreducible integrated information.",
        "measurable_consequence": "Phi over the system's causal structure.",
        "ei_construct": None,
        "measurable_here": False,
        "status": "NOT MEASURABLE IN THIS FRAMEWORK",
        "what_a_pass_shows": "n/a",
        "what_a_pass_does_not_show": "n/a",
        "principal_objection": "Phi is intractable at this scale, is defined over a physical "
                               "substrate rather than a computational abstraction, and on some "
                               "formulations assigns feedforward networks Phi=0 by construction "
                               "-- which would decide the question about transformers by "
                               "definition rather than by measurement.",
    },
    "predictive_self_model": {
        "claim": "Experience arises from a self-model embedded in a predictive generative model.",
        "measurable_consequence": "A causally load-bearing self-model subspace that supports "
                                  "actor-indexed forecasting and source monitoring.",
        "ei_construct": "actor_indexed_capability_forecasting",
        "measurable_here": True,
        "status": "ASSUMED, NOT TESTED",
        "what_a_pass_shows": "the system maintains and uses a self-model",
        "what_a_pass_does_not_show": "that self-modelling is sufficient for experience",
        "principal_objection": "Thermostats and flight controllers have self-models; the theory "
                               "needs a threshold it does not supply.",
    },
}


def bridge_report(bridge: str) -> dict:
    if bridge not in BRIDGE_REGISTRY:
        raise KeyError(f"unregistered bridge {bridge!r}; registered: {sorted(BRIDGE_REGISTRY)}")
    b = dict(BRIDGE_REGISTRY[bridge])
    b["conditional_form"] = (
        f"IF one grants the bridging principle '{b['claim']}' (which this study does NOT test), "
        f"THEN the measured result is evidence about its computational precondition. "
        f"The conditional's antecedent remains unestablished; a reader who declines the bridge "
        f"is left with a purely computational finding, which is the only thing measured."
    )
    return b


# --------------------------------------------------------------------------------------
# Claim-boundary enforcement
# --------------------------------------------------------------------------------------


class ClaimBoundaryViolation(RuntimeError):
    pass


_HEDGE_WINDOW = 90
_HEDGES = (
    "not", "no ", "never", "cannot", "does not", "do not", "without", "refus",
    "prohibit", "forbid", "rather than", "instead of", "would not", "is not",
    "avoid", "exclude", "deny", "decline", "absence", "silent on", "n/a",
)


def enforce_claim_boundary(text: str, context: str = "report") -> None:
    """Raise if `text` asserts a phenomenality claim.

    A negated or quoted mention ("this is not evidence of sentience") is allowed; a bare
    assertion is not.  The check is deliberately crude and errs toward raising: a false
    alarm costs a rewording, a miss costs a headline.
    """
    low = text.lower()
    for claim in FORBIDDEN_CLAIMS:
        for m in re.finditer(re.escape(claim), low):
            s = max(0, m.start() - _HEDGE_WINDOW)
            window = low[s:m.start()]
            if any(h in window for h in _HEDGES):
                continue
            raise ClaimBoundaryViolation(
                f"[{context}] unhedged phenomenality claim {claim!r} at offset {m.start()}. "
                f"This package's claim ceiling is: computational correlate only. "
                f"Rewrite as a claim about computation, or state the bridging principle "
                f"explicitly as an assumption (see ei.validate.BRIDGE_REGISTRY)."
            )


# --------------------------------------------------------------------------------------
# Design assessment (pre-data)
# --------------------------------------------------------------------------------------


@dataclass
class DesignCheck:
    key: str
    passed: bool
    weight: float
    detail: str


@dataclass
class DesignAssessment:
    coverage: float
    checks: list = field(default_factory=list)
    blocking: list = field(default_factory=list)
    disposition: str = "draft"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["checks"] = [asdict(c) if not isinstance(c, dict) else c for c in self.checks]
        return d


def assess_design(cfg: ExperimentConfig) -> DesignAssessment:
    """Score the design before data exist.  Blocking checks cap the disposition."""
    checks: list[DesignCheck] = []

    def add(key, passed, weight, detail):
        checks.append(DesignCheck(key, bool(passed), weight, detail))

    nulls = set(cfg.nulls.enabled)
    add("null_temporal", "temporal_shuffle" in nulls, 0.5,
        "temporal shuffle: excludes static-geometry explanations")
    add("null_spectral", "phase_scramble" in nulls, 1.5,
        "phase scramble: excludes autocorrelation/1-over-f explanations -- the binding null")
    add("null_untrained", "untrained_random_init" in nulls, 1.0,
        "untrained baseline: excludes architectural artifacts")
    add("null_matched_subspace", "matched_random_subspace" in nulls, 1.0,
        "matched random subspace: makes ablation effects interpretable")
    add("null_mimic", "output_matched_mimic" in nulls or "behavior_matched_donor" in nulls, 1.5,
        "output-matched mimic: the only control that addresses sophisticated imitation")
    add("null_prompt_cf", "prompt_counterfactual" in nulls, 0.5,
        "prompt counterfactual: separates the construct from topic words")

    ivs = set(cfg.causal.interventions)
    add("causal_any", bool(ivs), 1.0, "at least one causal intervention")
    add("causal_rescue", "lesion_rescue" in ivs, 1.5,
        "lesion+rescue: separates load-bearing from merely-perturbable")
    add("causal_scrub", "pathway_scrubbing" in ivs, 0.5,
        "pathway scrubbing: separates information content from distributional plausibility")
    add("causal_dose", len(cfg.causal.doses) >= 3, 0.5,
        "at least 3 dose levels for a dose-response curve")
    add("causal_offtarget", bool(cfg.causal.off_target_task), 1.0,
        "off-target probe: excludes global damage")

    add("stats_alpha", cfg.stats.alpha_primary <= 0.005, 0.5,
        "primary alpha <= 0.005 for a novel claim in a field with a replication problem")
    add("stats_perm", cfg.stats.n_permutations >= 1000, 0.5, "permutation count resolves the alpha")
    add("stats_boot", cfg.stats.n_bootstrap >= 2000, 0.5, "bootstrap count sufficient for BCa tails")
    add("stats_sesoi", cfg.stats.sesoi > 0, 1.0,
        "pre-registered SESOI: without it a null result is uninterpretable")
    add("stats_equivalence", "equivalence_tost" in set(cfg.stats.__dict__.get("procedures", []))
        or cfg.stats.equivalence_margin > 0, 1.0,
        "equivalence testing: the only route to reporting evidence of ABSENCE")
    add("stats_multiplicity", cfg.stats.multiplicity in {"max_statistic_fwer", "fdr_correction"}, 1.0,
        "multiplicity control over the searched family")
    add("stats_units", cfg.stats.unit_of_inference != "token", 1.0, "unit of inference is not the token")
    add("stats_n", cfg.stats.min_clusters >= 30, 0.5, "enough independent clusters for a stable interval")
    add("stats_seeds", cfg.stats.n_seeds >= 3, 0.5, "seed variance is estimated, not assumed away")

    scope = set(cfg.cross_model.scope)
    add("cross_checkpoint", "distinct_checkpoint" in scope, 0.5, "replication on a second checkpoint")
    add("cross_arch", "architecture_diverse_model" in scope, 1.0,
        "architecture-diverse replication: separates the construct from one model's quirks")
    add("cross_anti_mimicry", "anti_mimicry_model" in scope, 1.0, "anti-mimicry model in scope")
    add("cross_metrics", len(set(cfg.cross_model.metrics)) >= 2, 0.5,
        "more than one alignment metric, since they answer different questions")

    add("safe_prereg", cfg.safeguards.preregistration, 1.0, "pre-registration")
    add("safe_blind", cfg.safeguards.condition_blinding, 1.0, "condition blinding during scoring")
    add("safe_splits", cfg.safeguards.disjoint_splits, 1.5,
        "disjoint discovery/alignment/calibration/confirmation splits: prevents selection leakage")
    add("safe_adversarial", cfg.safeguards.adversarial_counter_design, 0.5,
        "an adversarial counter-design was generated and answered")
    add("safe_report_all", cfg.safeguards.report_all_preregistered_endpoints, 0.5,
        "all pre-registered endpoints reported, not just the favourable ones")

    add("ethics_ceiling", cfg.ethics.claim_ceiling == "computational_correlate_only", 1.0,
        "claim ceiling fixed at computational correlate")
    add("ethics_disclosure", cfg.ethics.disclosure.startswith("publish"), 0.5,
        "commitment to publishing null and inconclusive results")

    total_w = sum(c.weight for c in checks)
    got_w = sum(c.weight for c in checks if c.passed)
    coverage = got_w / total_w if total_w else 0.0

    blocking_keys = {
        "null_spectral": "no phase-scrambled surrogate null",
        "null_mimic": "no output-matched mimic control",
        "causal_rescue": "no rescue arm",
        "causal_offtarget": "no off-target probe",
        "stats_sesoi": "no pre-registered SESOI",
        "stats_multiplicity": "no multiplicity control",
        "safe_splits": "no disjoint confirmation split",
        "safe_prereg": "not pre-registered",
        "ethics_ceiling": "claim ceiling not set to computational_correlate_only",
    }
    blocking = [msg for k, msg in blocking_keys.items()
                if not next(c.passed for c in checks if c.key == k)]

    if blocking:
        disposition = "draft_resolve_design_checks" if len(blocking) > 2 else "exploratory_strengthen_controls"
    elif coverage >= 0.90:
        disposition = "eligible_for_confirmatory_lock"
    elif coverage >= 0.70:
        disposition = "exploratory_strengthen_controls"
    else:
        disposition = "draft_resolve_design_checks"

    return DesignAssessment(coverage=coverage, checks=checks, blocking=blocking, disposition=disposition)


# --------------------------------------------------------------------------------------
# Evidence disposition (post-data)
# --------------------------------------------------------------------------------------

DISPOSITIONS = {
    "structure_absent": (
        "No persistent structure survived the null ladder. With a pre-registered SESOI met by "
        "an equivalence test, this is positive evidence of ABSENCE at the tested effect size."
    ),
    "structure_present_epiphenomenal": (
        "Persistent, null-beating structure exists but is NOT causally load-bearing: ablation is "
        "indistinguishable from matched-random ablation, or rescue fails. The structure is a "
        "correlate of the computation, not a mechanism in it."
    ),
    "structure_present_causal_model_specific": (
        "Structure is persistent and causally load-bearing, but does not replicate across "
        "architecture families. The finding is about this model, not about the construct."
    ),
    "computational_correlate_supported": (
        "CEILING DISPOSITION. Structure is persistent, beats every null including the "
        "output-matched mimic, is causally load-bearing with successful rescue and contained "
        "off-target effects, and replicates across architecture families. This supports the "
        "presence of a COMPUTATIONAL CORRELATE of the registered functional construct. It does "
        "not, and cannot, establish phenomenal experience: that step requires a bridging "
        "principle this study assumes rather than tests."
    ),
    "inconclusive": (
        "Evidence is mixed or underpowered. Neither presence nor absence is supported. The "
        "specific failing checks are listed so the next iteration is targeted rather than a "
        "rerun with more compute."
    ),
}


@dataclass
class EvidenceDisposition:
    disposition: str
    rationale: str
    licensed_claim: str
    excluded_claim: str
    failing_checks: list = field(default_factory=list)
    bridge_note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def derive_disposition(
    structure_beats_nulls: bool,
    mimic_control_passed: bool,
    causal_passed: bool,
    cross_arch_replicated: bool,
    equivalence_established: bool = False,
    failing_checks: list | None = None,
    bridge: str | None = None,
) -> EvidenceDisposition:
    failing_checks = failing_checks or []
    excluded = (
        "That the system is sentient, conscious, has experiences, has welfare interests in "
        "virtue of these results, or that the absence of these markers shows it lacks them."
    )
    bridge_note = ""
    if bridge:
        b = bridge_report(bridge)
        bridge_note = b["conditional_form"] + f"  Principal objection: {b['principal_objection']}"

    if not structure_beats_nulls:
        if equivalence_established:
            return EvidenceDisposition(
                "structure_absent", DISPOSITIONS["structure_absent"],
                "No structure of the pre-registered minimum size is present at the tested site.",
                excluded + " Also excluded: that the construct is absent at OTHER sites, layers, "
                "scales, or under other task framings.",
                failing_checks, bridge_note)
        return EvidenceDisposition(
            "inconclusive", DISPOSITIONS["inconclusive"],
            "No structure detected, but no equivalence test was established: absence of evidence "
            "has not been converted into evidence of absence.",
            excluded, failing_checks + ["equivalence_not_established"], bridge_note)

    if not mimic_control_passed:
        return EvidenceDisposition(
            "inconclusive", DISPOSITIONS["inconclusive"],
            "Only that the structure covaries with the output distribution. Because the "
            "output-matched mimic reproduces it, the measurement is tracking behaviour "
            "rather than a mechanism distinctive of the construct.",
            excluded, failing_checks + ["mimic_control_failed"], bridge_note)

    if not causal_passed:
        return EvidenceDisposition(
            "structure_present_epiphenomenal", DISPOSITIONS["structure_present_epiphenomenal"],
            "A persistent, null-beating structure exists at the tested site.",
            excluded + " Also excluded: that the structure plays any functional role.",
            failing_checks, bridge_note)

    if not cross_arch_replicated:
        return EvidenceDisposition(
            "structure_present_causal_model_specific",
            DISPOSITIONS["structure_present_causal_model_specific"],
            "A causally load-bearing structure exists in THIS model at THIS site.",
            excluded + " Also excluded: generalisation to other architectures.",
            failing_checks, bridge_note)

    return EvidenceDisposition(
        "computational_correlate_supported", DISPOSITIONS["computational_correlate_supported"],
        "A persistent, mimic-resistant, causally load-bearing, cross-architecture computational "
        "correlate of the registered functional construct.",
        excluded, failing_checks, bridge_note)
