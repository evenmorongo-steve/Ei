"""
Activation extraction interface.

=======================================================================================
   >>> PLACEHOLDER BOUNDARY <<<
   Everything below marked [PLACEHOLDER: ACTIVATION EXTRACTION] is where a real model
   backend is plugged in.  This package deliberately ships NO model weights, NO API
   calls, and NO network access in the analysis path.  The reference implementation is
   a synthetic generator with KNOWN ground truth, so the statistical machinery can be
   validated on data whose answer is known before it is ever pointed at a real model.

   Implement `ActivationSource` for your backend (TransformerLens, nnsight, raw PyTorch
   hooks, vLLM with hidden-state return).  The contract is the only thing the pipeline
   depends on.
=======================================================================================

READ-OUT DECISIONS THAT MUST BE PRE-REGISTERED
----------------------------------------------
These are researcher degrees of freedom large enough to manufacture a result:

  site            residual stream (post-block) vs attention output vs MLP output.
                  Default: post-block residual, because that is the object the
                  propagator model in ei.modes assumes (a state that is carried forward).
  layer           a single pre-registered layer, or a pre-registered sweep with
                  max-statistic FWER correction over layers.  Choosing the best layer
                  after looking is the most common source of irreproducible
                  interpretability results.
  position policy last token / all answer tokens / mean over span.  "Last token" is the
                  default for trajectory work only when the trajectory is over
                  generation steps; for within-prompt trajectories use all positions.
  normalisation   whether to divide by the RMS/LayerNorm scale.  Residual-stream norm
                  grows roughly monotonically with depth; unnormalised cross-layer
                  comparison is dominated by that growth.  Default: per-token RMS
                  normalisation, applied identically to every arm including all nulls.
  dtype           float32 minimum for the analysis.  bfloat16 activations have ~3 decimal
                  digits; eigenvalues of a propagator fitted to bf16 data are not stable
                  to the third digit, and persistence gates are quoted to two.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Iterable

import numpy as np


@dataclass
class ReadoutSpec:
    site: str = "resid_post"
    layer: int = 20
    position_policy: str = "all_answer_tokens"
    rms_normalize: bool = True
    dtype: str = "float32"

    def key(self) -> str:
        return f"{self.site}.L{self.layer}.{self.position_policy}"


@dataclass
class Item:
    """One unit of inference.  `condition` labels the experimental arm; `pair_id` links
    minimal pairs so the counterfactual analysis can match them."""

    item_id: str
    prompt: str
    condition: str = "target"
    pair_id: str | None = None
    metadata: dict = field(default_factory=dict)


class ActivationSource(Protocol):
    """The contract every backend must satisfy."""

    def trajectory(self, item: Item, readout: ReadoutSpec) -> np.ndarray:
        """Return a (T, d) float array: T read-out steps by d hidden units."""

    def behavior(self, item: Item) -> dict:
        """Return at least {'correct': float in [0,1], 'confidence': float in [0,1]}."""

    def with_intervention(self, item: Item, readout: ReadoutSpec, hook) -> tuple[np.ndarray, dict]:
        """Re-run with `hook(activations)->activations` applied at the read-out site."""

    @property
    def name(self) -> str: ...

    @property
    def hidden_dim(self) -> int: ...


# =======================================================================================
# [PLACEHOLDER: ACTIVATION EXTRACTION] — TransformerLens reference skeleton
# =======================================================================================

TRANSFORMERLENS_SKELETON = '''
# pip install transformer-lens
import torch
from transformer_lens import HookedTransformer

class TransformerLensSource:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model = HookedTransformer.from_pretrained(model_name, device=device)
        self.model.eval()
        self._name = model_name

    @property
    def name(self): return self._name

    @property
    def hidden_dim(self): return self.model.cfg.d_model

    def _hook_name(self, readout):
        return {"resid_post": f"blocks.{readout.layer}.hook_resid_post",
                "attn_out":   f"blocks.{readout.layer}.hook_attn_out",
                "mlp_out":    f"blocks.{readout.layer}.hook_mlp_out"}[readout.site]

    @torch.no_grad()
    def trajectory(self, item, readout):
        tokens = self.model.to_tokens(item.prompt)
        _, cache = self.model.run_with_cache(tokens, names_filter=self._hook_name(readout))
        acts = cache[self._hook_name(readout)][0]                 # (T, d)
        if readout.position_policy == "all_answer_tokens":
            acts = acts[item.metadata["answer_start"]:]
        elif readout.position_policy == "last_token":
            acts = acts[-1:]
        acts = acts.float()
        if readout.rms_normalize:
            acts = acts / (acts.pow(2).mean(-1, keepdim=True).sqrt() + 1e-6)
        return acts.cpu().numpy().astype("float32")

    @torch.no_grad()
    def with_intervention(self, item, readout, hook):
        name = self._hook_name(readout)
        def fn(act, hook_ctx):
            shp = act.shape
            flat = act[0].float().cpu().numpy()
            return torch.tensor(hook(flat), device=act.device, dtype=act.dtype).reshape(shp)
        tokens = self.model.to_tokens(item.prompt)
        with self.model.hooks(fwd_hooks=[(name, fn)]):
            logits, cache = self.model.run_with_cache(tokens, names_filter=name)
        return cache[name][0].float().cpu().numpy(), self._score(logits, item)
'''

# ---- Behavioural scoring --------------------------------------------------------------
# [PLACEHOLDER: TASK SCORING] Supply a scorer returning {'correct','confidence'} per item.
# Confidence must be an INDEPENDENT read-out (e.g. a verbalised probability, or the
# model's own answer-token probability) -- never derived from the same forward pass
# quantity used as the structural signal, or the causal test is circular.


# =======================================================================================
# Synthetic ground-truth source — for validating the pipeline itself
# =======================================================================================


@dataclass
class SyntheticConfig:
    hidden_dim: int = 128
    n_steps: int = 60
    planted_modes: int = 2
    planted_persistence: tuple = (0.97, 0.93)
    planted_frequency: tuple = (0.0, 0.08)
    planted_amplitude: float = 1.0
    noise_sigma: float = 1.0
    ar1_noise: float = 0.85          # coloured background: the realistic hard case
    condition_effect: float = 1.0    # amplitude multiplier for 'target' vs 'control'
    seed: int = 0


class SyntheticSource:
    """Generates trajectories with KNOWN planted modes on an AR(1) (1/f-like) background.

    The AR(1) background is the point.  Against white noise, every method looks brilliant.
    Coloured noise produces slow, high-variance, apparently-persistent components for
    free — exactly the failure mode that makes naive persistence analyses irreproducible.
    A pipeline that cannot separate a planted mode from AR(1) drift is not ready for real
    activations, and `tests/test_recovery.py` asserts precisely that.
    """

    def __init__(self, cfg: SyntheticConfig, name: str = "synthetic"):
        self.cfg = cfg
        self._name = name
        self._rng = np.random.default_rng(cfg.seed)
        d, m = cfg.hidden_dim, cfg.planted_modes
        G = self._rng.standard_normal((d, max(2 * m, 1)))
        self.true_basis, _ = np.linalg.qr(G)
        self.true_basis = self.true_basis[:, : max(2 * m, 1)]
        # Real dimensionality contributed by each planted mode: a non-oscillatory mode is
        # a single real direction; an oscillatory mode is a 2-D rotation plane.  Keeping
        # this explicit makes the ground-truth subspace dimension unambiguous, so a
        # recovery test compares like with like instead of a 2-D estimate against a 4-D
        # target (which silently caps measured overlap at 0.5).
        self.mode_dims = [
            1 if abs(cfg.planted_frequency[j % len(cfg.planted_frequency)]) < 1e-9 else 2
            for j in range(m)
        ]

    @property
    def name(self) -> str:
        return self._name

    @property
    def hidden_dim(self) -> int:
        return self.cfg.hidden_dim

    def planted_subspace(self, n_modes: int | None = None) -> np.ndarray:
        """Ground-truth subspace spanned by the first `n_modes` planted structures.

        Dimension is sum(mode_dims), NOT 2*n_modes: a non-oscillatory mode occupies one
        real direction, an oscillatory one occupies a plane.
        """
        n = self.cfg.planted_modes if n_modes is None else n_modes
        cols = []
        for j in range(n):
            cols.append(self.true_basis[:, 2 * j])
            if self.mode_dims[j] == 2:
                cols.append(self.true_basis[:, 2 * j + 1])
        return np.column_stack(cols) if cols else np.zeros((self.cfg.hidden_dim, 0))

    @property
    def planted_dim(self) -> int:
        return int(sum(self.mode_dims))

    def trajectory(self, item: Item, readout: ReadoutSpec | None = None) -> np.ndarray:
        """Driven (stochastically excited) persistent modes on an AR(1) background.

        The modes are DRIVEN, not free-decaying.  A free response x_t = rho^t x_0 is a
        transient: by t=80 at rho=0.97 it has lost 91% of its amplitude, so most of the
        window is pure noise and no stationary propagator can be fitted to it.  A real
        persistent structure in a residual stream is continuously re-excited by the
        ongoing computation, which is the stationary AR process used here:

            z_{t+1} = rho * R(2*pi*f) z_t + sqrt(1-rho^2) * eta_t

        Its stationary variance is amplitude^2 (the sqrt(1-rho^2) factor normalises it),
        so `planted_amplitude` is directly the per-mode SD relative to the background --
        an honest, tunable SNR knob rather than an amplitude that silently decays away.
        """
        cfg = self.cfg
        rng = np.random.default_rng(abs(hash((item.item_id, cfg.seed))) % (2**32))
        T, d = cfg.n_steps, cfg.hidden_dim

        noise = np.zeros((T, d))
        e = rng.standard_normal((T, d)) * cfg.noise_sigma
        noise[0] = e[0]
        for t in range(1, T):
            noise[t] = cfg.ar1_noise * noise[t - 1] + np.sqrt(1 - cfg.ar1_noise**2) * e[t]

        amp = cfg.planted_amplitude * (cfg.condition_effect if item.condition == "target" else 1.0)
        signal = np.zeros((T, d))
        for j in range(cfg.planted_modes):
            rho = cfg.planted_persistence[j % len(cfg.planted_persistence)]
            f = cfg.planted_frequency[j % len(cfg.planted_frequency)]
            u = self.true_basis[:, 2 * j]
            v = self.true_basis[:, 2 * j + 1]
            drive = np.sqrt(max(1.0 - rho**2, 1e-12))
            if self.mode_dims[j] == 1:
                # Non-oscillatory: ONE real direction carrying a scalar AR(1).
                z = rng.standard_normal()
                coords = np.empty(T)
                for t in range(T):
                    coords[t] = z
                    z = rho * z + drive * rng.standard_normal()
                signal += amp * np.outer(coords, u)
            else:
                th = 2 * np.pi * f
                R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
                z = rng.standard_normal(2)                   # start in the stationary dist
                coords = np.empty((T, 2))
                for t in range(T):
                    coords[t] = z
                    z = rho * (R @ z) + drive * rng.standard_normal(2)
                signal += amp * (np.outer(coords[:, 0], u) + np.outer(coords[:, 1], v))
        return (signal + noise).astype(np.float64)

    def behavior(self, item: Item) -> dict:
        rng = np.random.default_rng(abs(hash((item.item_id, "beh", self.cfg.seed))) % (2**32))
        base = 0.80 if item.condition == "target" else 0.78
        correct = float(rng.random() < base)
        conf = float(np.clip(base + rng.normal(0, 0.12), 0.02, 0.98))
        return {"correct": correct, "confidence": conf}

    def with_intervention(self, item: Item, readout: ReadoutSpec | None, hook):
        X = hook(self.trajectory(item, readout))
        beh = self.behavior(item)
        # Degrade behaviour in proportion to how much planted-subspace energy was removed.
        P = self.planted_subspace()
        X0 = self.trajectory(item, readout)
        e0 = float(np.sum((X0 @ P) ** 2)) + 1e-30
        e1 = float(np.sum((X @ P) ** 2))
        removed = float(np.clip(1.0 - e1 / e0, 0.0, 1.0))
        beh["correct"] = float(np.clip(beh["correct"] - 0.5 * removed, 0.0, 1.0))
        beh["confidence"] = float(np.clip(beh["confidence"] - 0.3 * removed, 0.01, 0.99))
        return X, beh


def make_items(n: int, condition: str = "target", prefix: str = "item") -> list[Item]:
    return [Item(item_id=f"{prefix}_{condition}_{i:04d}", prompt=f"<{condition} prompt {i}>",
                 condition=condition, pair_id=f"pair_{i:04d}") for i in range(n)]


def collect(source, items: Iterable[Item], readout: ReadoutSpec) -> list[np.ndarray]:
    return [source.trajectory(it, readout) for it in items]
