"""
controls.py — Control Models for the mHC + Sequential MTP Experiment
=====================================================================

This file defines three control conditions that bracket the main experiment
in experiment.py:

    controls.py                         experiment.py
    ───────────────────────────────     ─────────────────────────────────
    Control 0: standard trunk, no MTP
    Control A: mHC trunk, no MTP        Condition A: mHC + MTP (sum)
    Control B: standard trunk + MTP     Condition B: mHC + MTP (mix)

The full 2×2 design:

    ┌────────────────────┬─────────────────┬───────────────────────┐
    │                    │   No MTP        │   Sequential MTP      │
    ├────────────────────┼─────────────────┼───────────────────────┤
    │ Standard residual  │  Control 0 ←    │  Control B  ← here    │
    │ mHC residual       │  Control A ←    │  Experiment A/B       │
    └────────────────────┴─────────────────┴───────────────────────┘

Control 0 is the plain residual LM baseline with no mHC and no MTP.
Control A isolates the contribution of mHC residuals alone, without MTP.
Control B isolates the contribution of sequential MTP alone, using a plain
Pre-Norm transformer trunk instead of mHC.  Together they let you attribute
any performance difference between the experiment conditions to the
interaction of mHC and MTP rather than to either component alone.

Both controls share the MTPStack from mtp.py and present the same interface
as mHCWithMTP from experiment.py, so they plug directly into train.py.

Usage with train.py
-------------------
train.py selects the model via --model:

    python train.py --model residual_only                # Control 0
    python train.py --model mhc_only   --reduction sum   # Control A
    python train.py --model residual_mtp                  # Control B
    python train.py --model mhc_mtp    --reduction sum   # Experiment A (from experiment.py)
    python train.py --model mhc_mtp    --reduction mix   # Experiment B (from experiment.py)

Dependencies
------------
    mhc.py       — mHCTransformerBlock, mHCResidual, sinkhorn
    mtp.py       — MTPStack
    experiment.py — ExperimentConfig  (shared config dataclass)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mhc import (
    CausalSelfAttention,
    SwiGLUFFN,
    init_gpt_style_weights,
    mHCTransformerBlock,
)
from mtp import MTPStack
from experiment import ExperimentConfig


# ---------------------------------------------------------------------------
# Control A: mHC trunk, no MTP
# ---------------------------------------------------------------------------

class mHCOnlyModel(nn.Module):
    """
    Control A — mHC trunk with standard next-token prediction only.

    This is exactly the mHCModel from mhc.py, rewritten here to:
      1. Return hidden states from forward() rather than logits, matching
         the interface convention used throughout this experiment suite.
      2. Expose .embed, .lm_head, .final_norm as top-level attributes so
         train.py can treat all four models identically.

    No MTP modules are attached.  The training loss is purely the
    standard next-token cross-entropy from the trunk's lm_head.

    Use this to answer: how much of any performance gain in the experiment
    conditions comes from mHC alone, before MTP is added?

    Args:
        cfg : ExperimentConfig  (n_mtp and reduction fields are ignored)
    """

    def __init__(self, cfg: ExperimentConfig):
        super().__init__()
        self.n_streams = cfg.n_streams
        self.d_model   = cfg.d_model
        self.n_heads   = cfg.n_heads

        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.d_model)

        self.blocks = nn.ModuleList([
            mHCTransformerBlock(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                n_streams=cfg.n_streams,
                layer_idx=i,
                ffn_mult=cfg.ffn_mult,
                max_seq_len=cfg.max_seq_len,
                sinkhorn_iters=cfg.sinkhorn_iters,
                identity_epsilon=cfg.mhc_identity_epsilon,
                gate_init=cfg.mhc_gate_init,
            )
            for i in range(cfg.n_layers)
        ])

        self.final_norm = nn.RMSNorm(cfg.d_model)
        self.lm_head    = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying and aliases for MTPStack compatibility
        self.lm_head.weight = self.tok_embed.weight
        self.embed = self.tok_embed   # alias: MTPStack looks for .embed

        self._init_weights()

    def _init_weights(self):
        init_gpt_style_weights(self, len(self.blocks))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids : (B, T)

        Returns:
            hidden : (B, T, d_model)   pre-norm hidden states
                     Caller applies final_norm + lm_head for logits.
        """
        B, T   = input_ids.shape
        device = input_ids.device

        pos = torch.arange(T, device=device)
        x   = self.tok_embed(input_ids) + self.pos_embed(pos)   # (B, T, d)

        streams = x.unsqueeze(2).expand(
            B, T, self.n_streams, self.d_model
        ).clone()                                                # (B, T, n, d)

        for block in self.blocks:
            streams = block(streams)

        hidden = streams.mean(dim=2)  # (B, T, d)
        return hidden                 # caller applies final_norm + lm_head

    def mhc_diagnostics(self) -> dict[str, float]:
        stats: dict[str, list[float]] = {
            "row_err": [],
            "col_err": [],
            "diag_mass": [],
            "entropy": [],
            "sigma_1": [],
            "sigma_2": [],
        }
        for block in self.blocks:
            for residual in (block.attn_mhc, block.ffn_mhc):
                diag = residual.diagnostics()
                for key, value in diag.items():
                    stats[key].append(value)
        return {
            f"mhc_{key}": sum(values) / len(values)
            for key, values in stats.items()
            if values
        }


class mHCOnly(nn.Module):
    """
    Top-level wrapper for Control A.

    Presents the same forward() -> (total_loss, metrics) interface as
    mHCWithMTP in experiment.py, so train.py needs no special-casing.

    The metrics dict uses the same keys but with empty per_depth_losses
    and mtp_loss=0, making CSV logging and comparisons uniform.
    """

    def __init__(self, cfg: ExperimentConfig):
        super().__init__()
        self.cfg   = cfg
        self.trunk = mHCOnlyModel(cfg)
        # No MTP stack.

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            input_ids : (B, T)

        Returns:
            total_loss : scalar  (main LM loss only)
            metrics    : dict matching mHCWithMTP.forward() output schema
        """
        hidden  = self.trunk(input_ids)
        logits  = self.trunk.lm_head(self.trunk.final_norm(hidden))

        main_loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            input_ids[:, 1:].reshape(-1),
        )

        return main_loss, {
            "main_loss":        main_loss,
            "mtp_loss":         torch.zeros((), device=input_ids.device),
            "per_depth_losses": [],
            "mix_weights":      None,
            "mhc_diagnostics":  self.trunk.mhc_diagnostics(),
        }

    def infer(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.trunk(input_ids)
        return self.trunk.lm_head(self.trunk.final_norm(hidden))

    def parameter_count(self) -> dict:
        n = sum(p.numel() for p in self.parameters())
        return {"trunk_total": n, "mtp_modules": 0, "mix_matrix": 0, "grand_total": n}


# ---------------------------------------------------------------------------
# Control B: standard Pre-Norm residual trunk + sequential MTP
# ---------------------------------------------------------------------------

class PreNormTransformerBlock(nn.Module):
    """
    Standard Pre-Norm transformer block: the conventional residual baseline.

    LayerNorm is applied before each sublayer; the residual connection adds
    the sublayer output back to the un-normalised input.  This is the
    architecture used in GPT-2 and most subsequent decoder-only models.

    Uses the same CausalSelfAttention and SwiGLUFFN sublayers as the mHC
    blocks in mhc.py, so the per-layer parameter count and computation are
    identical to one mHC block reading from a single stream.  This makes
    the residual trunk directly comparable to the mHC trunk in terms of
    capacity and FLOPs.

    Args:
        d_model     : model dimension
        n_heads     : attention heads
        ffn_mult    : FFN hidden-dim multiplier
        max_seq_len : maximum sequence length for causal mask
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_mult: int = 4,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.attn  = CausalSelfAttention(d_model, n_heads, max_seq_len)
        self.norm2 = nn.RMSNorm(d_model)
        self.ffn   = SwiGLUFFN(d_model, ffn_mult)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class ResidualTrunk(nn.Module):
    """
    Standard single-stream Pre-Norm residual transformer trunk.

    Identical in depth, width, and sublayer design to mHCOnlyModel, but
    with conventional residual connections rather than mHC multi-stream
    connections.  n_streams is not used; the trunk maintains a single
    hidden state vector throughout.

    Exposes .embed, .lm_head, .final_norm at the top level so MTPStack
    can reference them without any special-casing.

    Args:
        cfg : ExperimentConfig  (n_streams is ignored)
    """

    def __init__(self, cfg: ExperimentConfig):
        super().__init__()
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads

        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.d_model)

        self.blocks = nn.ModuleList([
            PreNormTransformerBlock(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                ffn_mult=cfg.ffn_mult,
                max_seq_len=cfg.max_seq_len,
            )
            for _ in range(cfg.n_layers)
        ])

        self.final_norm = nn.RMSNorm(cfg.d_model)
        self.lm_head    = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying and alias
        self.lm_head.weight = self.tok_embed.weight
        self.embed = self.tok_embed

        self._init_weights()

    def _init_weights(self):
        # Standard GPT-style initialisation: N(0, 0.02), output projections
        # scaled down by 1/sqrt(2 * n_layers) to prevent residual variance growth.
        std = 0.02
        rescale_std = std / math.sqrt(2 * sum(1 for _ in self.blocks))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)

        # Rescale output projections of attention and FFN
        for block in self.blocks:
            nn.init.normal_(block.attn.out_proj.weight, std=rescale_std)
            nn.init.normal_(block.ffn.down.weight,      std=rescale_std)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids : (B, T)

        Returns:
            hidden : (B, T, d_model)   pre-norm hidden states
        """
        B, T   = input_ids.shape
        device = input_ids.device

        pos = torch.arange(T, device=device)
        x   = self.tok_embed(input_ids) + self.pos_embed(pos)   # (B, T, d)

        for block in self.blocks:
            x = block(x)

        return x   # caller applies final_norm + lm_head


class ResidualOnly(nn.Module):
    """
    Control 0 — standard Pre-Norm residual trunk with no MTP.

    This is the plain language-model baseline for the full 2x2 comparison.
    It uses the same ResidualTrunk as ResidualWithMTP but trains only with
    the standard next-token cross-entropy objective.
    """

    def __init__(self, cfg: ExperimentConfig):
        super().__init__()
        self.cfg = cfg
        self.trunk = ResidualTrunk(cfg)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, dict]:
        hidden = self.trunk(input_ids)
        logits = self.trunk.lm_head(self.trunk.final_norm(hidden))

        main_loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            input_ids[:, 1:].reshape(-1),
        )

        return main_loss, {
            "main_loss": main_loss,
            "mtp_loss": torch.zeros((), device=input_ids.device),
            "per_depth_losses": [],
            "mix_weights": None,
            "mhc_diagnostics": {},
        }

    def infer(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.trunk(input_ids)
        return self.trunk.lm_head(self.trunk.final_norm(hidden))

    def parameter_count(self) -> dict:
        n = sum(p.numel() for p in self.parameters())
        return {"trunk_total": n, "mtp_modules": 0, "mix_matrix": 0, "grand_total": n}


class ResidualWithMTP(nn.Module):
    """
    Control B — standard Pre-Norm residual trunk + sequential MTP stack.

    This isolates the contribution of sequential MTP alone.  The trunk is
    conventional; only the MTP chain on top is the same as in the experiment
    conditions.  Comparing this against mHCOnly and mHCWithMTP lets you
    answer:
        - Does MTP help a plain residual trunk?  (residual_mtp vs residual baseline)
        - Does mHC help beyond what MTP alone provides?  (mhc_mtp vs residual_mtp)
        - Does mHC help without MTP?  (mhc_only vs residual baseline)

    The MTP stack is constructed with references to trunk.embed and
    trunk.lm_head, so parameters are truly shared (not copied), exactly
    as in DeepSeek-V3 and in the main experiment conditions.

    Args:
        cfg : ExperimentConfig  (n_streams and reduction are ignored)
    """

    def __init__(self, cfg: ExperimentConfig):
        super().__init__()
        self.cfg   = cfg
        self.trunk = ResidualTrunk(cfg)

        self.mtp = MTPStack(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_modules=cfg.n_mtp,
            embed_table=self.trunk.embed,
            lm_head=self.trunk.lm_head,
            loss_scale=cfg.mtp_loss_scale,
            ffn_mult=cfg.ffn_mult,
            max_seq_len=cfg.max_seq_len,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            input_ids : (B, T)

        Returns:
            total_loss : scalar
            metrics    : dict matching mHCWithMTP.forward() output schema
        """
        hidden  = self.trunk(input_ids)
        logits  = self.trunk.lm_head(self.trunk.final_norm(hidden))

        main_loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            input_ids[:, 1:].reshape(-1),
        )

        mtp_loss, per_depth = self.mtp(hidden, input_ids)
        total = main_loss + mtp_loss

        return total, {
            "main_loss":        main_loss,
            "mtp_loss":         mtp_loss,
            "per_depth_losses": per_depth,
            "mix_weights":      None,   # no mixing matrix in this control
            "mhc_diagnostics":  {},
        }

    def infer(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.trunk(input_ids)
        return self.trunk.lm_head(self.trunk.final_norm(hidden))

    def parameter_count(self) -> dict:
        trunk = sum(p.numel() for p in self.trunk.parameters())
        mtp   = sum(p.numel() for p in self.mtp.modules_list.parameters())
        return {
            "trunk_total": trunk,
            "mtp_modules": mtp,
            "mix_matrix":  0,
            "grand_total": trunk + mtp,
        }


# ---------------------------------------------------------------------------
# Unified factory used by train.py
# ---------------------------------------------------------------------------

def build_control(model_name: str, cfg: ExperimentConfig) -> nn.Module:
    """
    Construct a control model by name.

    Args:
        model_name : one of "residual_only", "mhc_only", "residual_mtp"
        cfg        : ExperimentConfig

    Returns:
        model : nn.Module with forward() -> (loss, metrics) and .infer()
    """
    torch.manual_seed(cfg.seed)
    if model_name == "residual_only":
        return ResidualOnly(cfg)
    elif model_name == "mhc_only":
        return mHCOnly(cfg)
    elif model_name == "residual_mtp":
        return ResidualWithMTP(cfg)
    else:
        raise ValueError(
            f"Unknown model name: {model_name!r}. "
            f"Choose from: 'residual_only', 'mhc_only', 'residual_mtp'. "
            f"For experiment conditions use experiment.build_model() with "
            f"reduction='sum' or reduction='mix'."
        )


# ---------------------------------------------------------------------------
# Parameter count comparison utility
# ---------------------------------------------------------------------------

def compare_parameter_counts(cfg: ExperimentConfig) -> None:
    """
    Print a side-by-side parameter count for all four model variants.

    Useful for verifying that the four conditions are fair comparisons:
    the residual trunk should have approximately the same parameter count
    as the mHC trunk (n_streams adds only a small number of HC parameters),
    and the MTP modules add the same count on top of both.
    """
    from experiment import build_model
    import dataclasses

    models = {
        "residual_only": ResidualOnly(cfg),
        "mhc_only":     mHCOnly(cfg),
        "residual_mtp": ResidualWithMTP(cfg),
        "mhc_mtp_sum":  build_model(dataclasses.replace(cfg, reduction="sum")),
        "mhc_mtp_mix":  build_model(dataclasses.replace(cfg, reduction="mix")),
    }

    print(f"\n{'Model':<20} {'Trunk':>12} {'MTP':>10} {'Mix':>8} {'Total':>12}")
    print("─" * 66)
    for name, model in models.items():
        c = model.parameter_count()
        print(
            f"{name:<20} "
            f"{c['trunk_total']:>12,} "
            f"{c['mtp_modules']:>10,} "
            f"{c['mix_matrix']:>8,} "
            f"{c['grand_total']:>12,}"
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    cfg = ExperimentConfig(
        vocab_size=512,
        d_model=64,
        n_layers=4,
        n_heads=4,
        n_streams=4,
        n_mtp=2,
        mtp_loss_scale=0.3,
        max_seq_len=64,
        seq_len=32,
    )

    B, T = 2, 32
    ids  = torch.randint(0, cfg.vocab_size, (B, T))

    print("=" * 50)
    print("Control A: mHC trunk, no MTP")
    print("=" * 50)
    model_a = mHCOnly(cfg)
    loss_a, metrics_a = model_a(ids)
    print(f"  loss       : {loss_a.item():.4f}")
    print(f"  main_loss  : {metrics_a['main_loss'].item():.4f}")
    print(f"  mtp_loss   : {metrics_a['mtp_loss'].item():.4f}  (always 0)")
    loss_a.backward()
    print(f"  backward   : OK")

    print()
    print("=" * 50)
    print("Control B: standard residual trunk + sequential MTP")
    print("=" * 50)
    model_b = ResidualWithMTP(cfg)
    loss_b, metrics_b = model_b(ids)
    print(f"  loss       : {loss_b.item():.4f}")
    print(f"  main_loss  : {metrics_b['main_loss'].item():.4f}")
    print(f"  mtp_loss   : {metrics_b['mtp_loss'].item():.4f}")
    for k, dl in enumerate(metrics_b["per_depth_losses"]):
        print(f"  depth {k}     : {dl.item():.4f}")
    loss_b.backward()
    print(f"  backward   : OK")

    print()
    compare_parameter_counts(cfg)
