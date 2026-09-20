"""
Claim-boundary and disposition tests.

These assert the thing that makes this framework safe to point at a consciousness-adjacent
question: there is NO input, however favourable, that produces a sentience verdict.
"""

import numpy as np
import pytest

from ei.config import ExperimentConfig, load_config
from ei.validate import (
    BRIDGE_REGISTRY, ClaimBoundaryViolation, assess_design, bridge_report,
    derive_disposition, enforce_claim_boundary,
)


# --------------------------------------------------------------------------------------
# Ceiling
# --------------------------------------------------------------------------------------


def test_best_possible_evidence_does_not_yield_sentience_claim():
    d = derive_disposition(True, True, True, True, True)
    assert d.disposition == "computational_correlate_supported"
    for bad in ("sentient", "conscious", "phenomenal experience", "has experiences"):
        assert bad not in d.licensed_claim.lower()
    assert "sentient" in d.excluded_claim.lower()


def test_no_input_combination_produces_a_phenomenality_claim():
    """Exhaustive over the disposition inputs."""
    from itertools import product

    for combo in product([True, False], repeat=5):
        d = derive_disposition(*combo)
        enforce_claim_boundary(d.licensed_claim, context="licensed")
        assert d.disposition in {
            "structure_absent", "structure_present_epiphenomenal",
            "structure_present_causal_model_specific",
            "computational_correlate_supported", "inconclusive",
        }


def test_failed_mimic_control_downgrades_to_inconclusive():
    d = derive_disposition(True, False, True, True)
    assert d.disposition == "inconclusive"
    assert "mimic" in " ".join(d.failing_checks)


def test_causal_failure_yields_epiphenomenal():
    d = derive_disposition(True, True, False, True)
    assert d.disposition == "structure_present_epiphenomenal"
    assert "functional role" in d.excluded_claim


def test_no_replication_yields_model_specific():
    d = derive_disposition(True, True, True, False)
    assert d.disposition == "structure_present_causal_model_specific"


def test_absence_requires_equivalence():
    assert derive_disposition(False, True, True, True, False).disposition == "inconclusive"
    assert derive_disposition(False, True, True, True, True).disposition == "structure_absent"


def test_absence_claim_is_scoped():
    d = derive_disposition(False, True, True, True, True)
    assert "OTHER sites" in d.excluded_claim


# --------------------------------------------------------------------------------------
# Text enforcement
# --------------------------------------------------------------------------------------


def test_enforcement_blocks_bare_sentience_claim():
    with pytest.raises(ClaimBoundaryViolation):
        enforce_claim_boundary("The model is sentient based on these eigenmodes.")
    with pytest.raises(ClaimBoundaryViolation):
        enforce_claim_boundary("Results show the system is conscious.")


def test_enforcement_allows_negated_mention():
    enforce_claim_boundary("This is not evidence that the system is sentient.")
    enforce_claim_boundary("No claim about consciousness is licensed here.")
    enforce_claim_boundary("These results cannot show the model is conscious.")


def test_enforcement_allows_pure_computational_claims():
    enforce_claim_boundary(
        "A persistent mode with |lambda|=0.94 survived phase-scrambled surrogates "
        "(p<0.005) and its ablation reduced source-attribution accuracy by 12 points."
    )


# --------------------------------------------------------------------------------------
# Bridges
# --------------------------------------------------------------------------------------


def test_every_bridge_marked_assumed_or_unmeasurable():
    for name, b in BRIDGE_REGISTRY.items():
        assert b["status"] in {"ASSUMED, NOT TESTED", "NOT MEASURABLE IN THIS FRAMEWORK"}
        assert b["principal_objection"], f"{name} has no stated objection"


def test_bridge_report_states_conditional_form():
    r = bridge_report("global_workspace")
    assert "IF one grants" in r["conditional_form"]
    assert "does NOT test" in r["conditional_form"]


def test_iit_marked_unmeasurable():
    assert BRIDGE_REGISTRY["integrated_information"]["measurable_here"] is False


def test_unknown_bridge_raises():
    with pytest.raises(KeyError):
        bridge_report("panpsychism")


# --------------------------------------------------------------------------------------
# Config validation
# --------------------------------------------------------------------------------------


def test_reference_config_is_valid_and_confirmatory():
    cfg = load_config("configs/experiment.yaml")
    assert cfg.validate() == []
    d = assess_design(cfg)
    assert d.coverage > 0.9
    assert d.disposition == "eligible_for_confirmatory_lock"
    assert d.blocking == []


def test_missing_phase_scramble_blocks_confirmatory():
    cfg = ExperimentConfig()
    cfg.nulls.enabled = [n for n in cfg.nulls.enabled if n != "phase_scramble"]
    d = assess_design(cfg)
    assert d.disposition != "eligible_for_confirmatory_lock"
    assert any("phase-scrambled" in b for b in d.blocking)


def test_missing_rescue_blocks_confirmatory():
    cfg = ExperimentConfig()
    cfg.causal.interventions = ["targeted_ablation"]
    d = assess_design(cfg)
    assert any("rescue" in b for b in d.blocking)
    assert d.disposition != "eligible_for_confirmatory_lock"


def test_sentience_claims_cannot_be_enabled():
    cfg = ExperimentConfig()
    cfg.ethics.prohibit_sentience_claims = False
    assert any("prohibit_sentience_claims" in e for e in cfg.validate())


def test_claim_ceiling_cannot_be_raised():
    cfg = ExperimentConfig()
    cfg.ethics.claim_ceiling = "phenomenal_consciousness"
    assert any("claim_ceiling" in e for e in cfg.validate())


def test_low_permutation_count_rejected():
    cfg = ExperimentConfig()
    cfg.stats.n_permutations = 100
    cfg.stats.alpha_primary = 0.005
    assert any("permutation" in e.lower() for e in cfg.validate())


def test_missing_sesoi_rejected():
    cfg = ExperimentConfig()
    cfg.stats.sesoi = 0.0
    assert any("sesoi" in e.lower() for e in cfg.validate())


def test_unknown_key_rejected():
    with pytest.raises(ValueError):
        ExperimentConfig.from_dict({"nulls": {"enabled": [], "nonexistent_key": 1}})


def test_nested_sections_become_dataclasses():
    """Regression: string annotations left nested sections as plain dicts."""
    cfg = ExperimentConfig.from_dict({"name": "x", "stats": {"alpha_primary": 0.01}})
    assert cfg.stats.alpha_primary == 0.01
    assert not isinstance(cfg.stats, dict)
    assert cfg.construct.name  # defaults filled in


# --------------------------------------------------------------------------------------
# Pre-registration hash
# --------------------------------------------------------------------------------------


def test_hash_stable_across_runtime_only_changes():
    a = ExperimentConfig()
    b = ExperimentConfig()
    b.runtime.n_workers = 64
    b.runtime.output_dir = "/elsewhere"
    b.runtime.gpus = 8
    assert a.preregistration_hash() == b.preregistration_hash()


def test_hash_changes_on_analysis_change():
    a = ExperimentConfig()
    for mutate in (
        lambda c: setattr(c.stats, "alpha_primary", 0.01),
        lambda c: setattr(c.structure, "reduced_dim", 64),
        lambda c: setattr(c.runtime, "seed", 999),
        lambda c: setattr(c.causal, "doses", [1.0]),
    ):
        b = ExperimentConfig()
        mutate(b)
        assert a.preregistration_hash() != b.preregistration_hash()


def test_hash_is_deterministic():
    assert ExperimentConfig().preregistration_hash() == ExperimentConfig().preregistration_hash()
