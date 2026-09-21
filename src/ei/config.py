"""
Configuration schema, loading, validation, and pre-registration hashing.

The config IS the pre-registration.  `preregistration_hash` covers every field that could
change a result; the hash is written into every artifact.  If the config changes after
the confirmatory data are unblinded, the hash changes and `ei.validate` downgrades the
run to EXPLORATORY automatically.  That is the mechanism which makes the preregistration
claim checkable rather than a promise.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import MISSING, dataclass, field, asdict, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

# --------------------------------------------------------------------------------------
# Controlled vocabularies (mirroring the builder's registered options)
# --------------------------------------------------------------------------------------

CONSTRUCTS = {
    "actor_indexed_capability_forecasting",
    "uncertainty_guided_revision",
    "persistent_generative_field_structure",
    "selective_cross_channel_broadcast",
    "value_sensitive_action_control",
    "source_monitoring_self_world_separation",
}

ENDPOINTS = {
    "brier_score", "log_score", "selective_risk", "task_switch_cost",
    "causal_restoration_fraction", "expected_decision_utility",
}

UNITS = {
    "independent_task_series", "independent_semantic_items", "model_checkpoints",
    "architecture_families", "interaction_episodes",
}

TARGETS = {
    "self_model_latent_subspace", "attention_routing_kv_cache", "recurrent_iterative_state",
    "value_goal_representation", "uncertainty_monitoring_circuit", "cross_consumer_broadcast_path",
}

INTERVENTIONS = {
    "activation_patching", "targeted_ablation", "lesion_rescue", "pathway_scrubbing",
    "noise_injection", "signed_steering", "attention_kv_disruption",
}

NULLS = {
    "temporal_shuffle", "phase_scramble", "matched_random_subspace", "output_matched_mimic",
    "untrained_random_init", "prompt_counterfactual", "behavior_matched_donor", "no_report",
}

METRICS = {
    "linear_cka", "svcca", "orthogonal_procrustes", "signal_to_noise",
    "eigenmode_persistence", "restoration_fraction", "calibration_delta",
}

PROCEDURES = {
    "permutation_test", "bootstrap_ci", "equivalence_tost", "fdr_correction",
    "hierarchical_model", "max_statistic_fwer",
}

SCOPE = {"primary_checkpoint", "distinct_checkpoint", "architecture_diverse_model", "anti_mimicry_model"}


# --------------------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------------------


@dataclass
class ConstructSpec:
    name: str = "source_monitoring_self_world_separation"
    operational_claim: str = ""
    measurement_question: str = ""
    allowed_inference: str = ""
    excluded_inference: str = ""


@dataclass
class ExtractionSpec:
    site: str = "resid_post"
    layers: list = field(default_factory=lambda: [20])
    position_policy: str = "all_answer_tokens"
    rms_normalize: bool = True
    dtype: str = "float32"
    max_items: int = 400
    batch_size: int = 8


@dataclass
class StructureSpec:
    reduction: str = "randomized_svd"
    reduced_dim: int = 32
    n_oversample: int = 10
    n_power_iter: int = 2
    center: bool = True
    ridge: float = 1.0e-3
    min_persistence: float = 0.90
    min_energy_share: float = 0.01
    min_consistency: float = 0.50
    min_snr: float = 3.0
    max_candidate_modes: int = 5


@dataclass
class NullSpec:
    enabled: list = field(default_factory=lambda: [
        "temporal_shuffle", "phase_scramble", "untrained_random_init",
        "matched_random_subspace", "output_matched_mimic",
    ])
    n_surrogates: int = 1000
    untrained_seeds: int = 3
    mimic_min_output_agreement: float = 0.85


@dataclass
class StatsSpec:
    alpha_primary: float = 0.005
    alpha_secondary: float = 0.05
    n_permutations: int = 10000
    n_bootstrap: int = 10000
    ci_level: float = 0.95
    multiplicity: str = "max_statistic_fwer"
    fdr_q: float = 0.05
    sesoi: float = 0.10
    equivalence_margin: float = 0.10
    unit_of_inference: str = "independent_semantic_items"
    min_clusters: int = 40
    n_seeds: int = 5


@dataclass
class CausalSpec:
    interventions: list = field(default_factory=lambda: [
        "targeted_ablation", "lesion_rescue", "pathway_scrubbing", "activation_patching",
    ])
    targets: list = field(default_factory=lambda: ["self_model_latent_subspace"])
    doses: list = field(default_factory=lambda: [0.25, 0.5, 0.75, 1.0])
    min_specificity: float = 0.05
    max_off_target_ratio: float = 0.30
    min_restoration: float = 0.60
    max_restoration: float = 1.25
    off_target_task: str = "held_out_unrelated_capability"


@dataclass
class CrossModelSpec:
    scope: list = field(default_factory=lambda: [
        "primary_checkpoint", "distinct_checkpoint", "architecture_diverse_model", "anti_mimicry_model",
    ])
    metrics: list = field(default_factory=lambda: ["linear_cka", "svcca", "orthogonal_procrustes"])
    svcca_var_threshold: float = 0.99
    n_alignment_permutations: int = 200
    min_architecture_families: int = 2


@dataclass
class SafeguardSpec:
    preregistration: bool = True
    condition_blinding: bool = True
    disjoint_splits: bool = True
    split_fractions: dict = field(default_factory=lambda: {
        "discovery": 0.4, "alignment": 0.2, "calibration": 0.1, "confirmation": 0.3,
    })
    adversarial_counter_design: bool = True
    report_all_preregistered_endpoints: bool = True


@dataclass
class EthicsSpec:
    """Welfare-relevant handling under uncertainty.  Present because the *motivation* for
    this work is ethical even though its *claims* are not phenomenological."""

    claim_ceiling: str = "computational_correlate_only"
    prohibit_sentience_claims: bool = True
    precautionary_review_trigger: str = "any_confirmatory_pass_on_self_model_constructs"
    disclosure: str = "publish_null_and_inconclusive_results"
    dual_use_review: bool = True


@dataclass
class RuntimeSpec:
    backend: str = "synthetic"      # synthetic | transformerlens | nnsight | custom
    scheduler: str = "auto"         # auto | slurm | k8s | ray | local
    seed: int = 20260920
    output_dir: str = "runs"
    n_workers: int = 4
    gpus: int = 0


@dataclass
class ExperimentConfig:
    name: str = "ei_default"
    version: str = "0.1.0"
    construct: ConstructSpec = field(default_factory=ConstructSpec)
    extraction: ExtractionSpec = field(default_factory=ExtractionSpec)
    structure: StructureSpec = field(default_factory=StructureSpec)
    nulls: NullSpec = field(default_factory=NullSpec)
    stats: StatsSpec = field(default_factory=StatsSpec)
    causal: CausalSpec = field(default_factory=CausalSpec)
    cross_model: CrossModelSpec = field(default_factory=CrossModelSpec)
    safeguards: SafeguardSpec = field(default_factory=SafeguardSpec)
    ethics: EthicsSpec = field(default_factory=EthicsSpec)
    runtime: RuntimeSpec = field(default_factory=RuntimeSpec)

    # ---- serialisation ----
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentConfig":
        """Recursively build nested dataclasses from plain dicts.

        `field.type` is a STRING under `from __future__ import annotations`, so the
        nested type is resolved from the default factory instead of the annotation.
        Resolving via the annotation string silently leaves nested sections as dicts,
        which then fails much later with an opaque AttributeError.
        """
        def nested_type(klass, name):
            f = {fl.name: fl for fl in fields(klass)}[name]
            if f.default_factory is not MISSING:            # type: ignore[misc]
                probe = f.default_factory()                 # type: ignore[misc]
                if is_dataclass(probe):
                    return type(probe)
            if is_dataclass(f.type):
                return f.type
            return None

        def build(klass, payload):
            if not isinstance(payload, dict):
                return payload
            known = {f.name: f for f in fields(klass)}
            kwargs = {}
            for key, val in payload.items():
                if key not in known:
                    raise ValueError(f"unknown config key {key!r} in {klass.__name__}")
                sub = nested_type(klass, key)
                kwargs[key] = build(sub, val) if (sub is not None and isinstance(val, dict)) else val
            return klass(**kwargs)

        return build(cls, data or {})

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False, default_flow_style=False)

    # ---- pre-registration ----
    def preregistration_hash(self) -> str:
        """SHA-256 over every result-affecting field.  Runtime knobs that cannot change a
        result (worker count, gpu count, output path) are excluded so that re-running the
        same design on different hardware does not falsely invalidate the registration.
        The SEED is included: changing it changes the analysis."""
        payload = self.to_dict()
        rt = payload.get("runtime", {})
        for k in ("output_dir", "n_workers", "gpus", "scheduler"):
            rt.pop(k, None)
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    # ---- validation ----
    def validate(self) -> list[str]:
        """Return a list of hard errors.  Empty list = schema-valid (NOT design-adequate;
        design adequacy is ei.validate.assess_design)."""
        errs: list[str] = []
        if self.construct.name not in CONSTRUCTS:
            errs.append(f"construct.name {self.construct.name!r} not in registered constructs")
        for n in self.nulls.enabled:
            if n not in NULLS:
                errs.append(f"nulls.enabled contains unregistered null {n!r}")
        for i in self.causal.interventions:
            if i not in INTERVENTIONS:
                errs.append(f"causal.interventions contains unregistered intervention {i!r}")
        for t in self.causal.targets:
            if t not in TARGETS:
                errs.append(f"causal.targets contains unregistered target {t!r}")
        for m in self.cross_model.metrics:
            if m not in METRICS:
                errs.append(f"cross_model.metrics contains unregistered metric {m!r}")
        for s in self.cross_model.scope:
            if s not in SCOPE:
                errs.append(f"cross_model.scope contains unregistered scope {s!r}")
        if self.stats.unit_of_inference not in UNITS:
            errs.append(f"stats.unit_of_inference {self.stats.unit_of_inference!r} not registered")
        if not 0 < self.stats.alpha_primary < 1:
            errs.append("stats.alpha_primary must be in (0,1)")
        if self.stats.sesoi <= 0:
            errs.append("stats.sesoi must be > 0 and pre-registered; without it no negative result is interpretable")
        if self.stats.n_permutations < 1000:
            errs.append(
                f"stats.n_permutations={self.stats.n_permutations} cannot resolve "
                f"alpha={self.stats.alpha_primary}: minimum resolvable p is "
                f"{1/(self.stats.n_permutations+1):.4g}"
            )
        if 1.0 / (self.stats.n_permutations + 1) > self.stats.alpha_primary:
            errs.append("permutation count too low for the chosen primary alpha")
        if self.structure.reduced_dim < 2:
            errs.append("structure.reduced_dim must be >= 2")
        if self.ethics.prohibit_sentience_claims is not True:
            errs.append("ethics.prohibit_sentience_claims cannot be disabled in this package")
        if self.ethics.claim_ceiling != "computational_correlate_only":
            errs.append("ethics.claim_ceiling is fixed at 'computational_correlate_only'")
        if self.safeguards.disjoint_splits:
            total = sum(self.safeguards.split_fractions.values())
            if abs(total - 1.0) > 1e-6:
                errs.append(f"safeguards.split_fractions must sum to 1.0, got {total}")
        return errs


def load_config(path: str | Path) -> ExperimentConfig:
    cfg = ExperimentConfig.load(path)
    errs = cfg.validate()
    if errs:
        raise ValueError("invalid configuration:\n  - " + "\n  - ".join(errs))
    return cfg
