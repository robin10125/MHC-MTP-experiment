"""
Sequential Multi-Token Prediction (MTP) Modules
================================================
Architecture from:
  DeepSeek-V3 Technical Report  arxiv.org/abs/2412.19437
  Megatron-Core MTP docs        docs.nvidia.com/megatron-core (Nemotron implementation)
  "Better & Faster LLMs via MTP" (Gloeckle et al., Meta 2024)

Design
------
This is the *sequential* MTP design used in DeepSeek-V3 and Nemotron 3
Super/Ultra.  It differs from the simpler "parallel independent heads" design
in two key ways:

  1. Each depth-k module receives the *processed* representation from depth k-1,
     not a raw copy of the trunk hidden state.  Deeper modules therefore develop
     specialised representations for lookahead planning.

  2. Each module is conditioned on the embedding of the corresponding future
     token (teacher-forced during training), which stabilises learning at deeper
     depths by ensuring the context is always correct.

Per-module computation for depth k (0-indexed):
    fused  = Proj_k( concat( RMSNorm_h( h^{k-1}_i ),
                              RMSNorm_e( embed(t_{i+k+1}) ) ) )
    h^k_i  = TransformerBlock_k( fused )
    logits = lm_head( FinalNorm_k( h^k_i ) )   # predicts t_{i+k+2}

Shared (provided by the trunk, not re-instantiated here):
    embed_table : nn.Embedding
    lm_head     : nn.Linear

Owned (one independent copy per depth):
    enorm, hnorm, proj, block, final_norm

Indexing
--------
    trunk       hidden state h^0_i predicts t_{i+1}   (standard next-token)
    MTP depth 0 hidden state h^1_i predicts t_{i+2}
    MTP depth 1 hidden state h^2_i predicts t_{i+3}
    ...
    MTP depth k hidden state h^{k+1}_i predicts t_{i+k+2}

Valid positions at depth k: i in [0, T - k - 3]
(requires t_{i+k+2} to exist in the sequence of length T)

The hidden state window shrinks by 1 at each depth.  Depth k operates on
a sequence of length T - k - 1.

Teacher-forcing
---------------
During training, embed(t_{i+k+1}) is the ground-truth token embedding.
At inference for speculative decoding, the sampled token from the previous
depth is used instead.

Integration
-----------
See WithSequentialMTP for a one-liner wrapper around any trunk.
The trunk must:
  - return (B, T, d_model) hidden states from forward()
  - expose .embed (nn.Embedding), .lm_head (nn.Linear), .final_norm
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sublayer utilities (self-contained; duplicated from mhc.py intentionally)
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 2048):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.qkv      = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        def split(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q, k, v = split(q), split(k), split(v)
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=0.0,
            is_causal=True,
        )
        return self.out_proj(out.transpose(1, 2).reshape(B, T, d))


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, ffn_mult: int = 4):
        super().__init__()
        h = d_model * ffn_mult
        self.gate = nn.Linear(d_model, h, bias=False)
        self.up   = nn.Linear(d_model, h, bias=False)
        self.down = nn.Linear(h, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MiniTransformerBlock(nn.Module):
    """Single pre-norm attention + FFN block used inside each MTP module."""

    def __init__(self, d_model: int, n_heads: int, ffn_mult: int = 4,
                 max_seq_len: int = 2048):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.attn  = CausalSelfAttention(d_model, n_heads, max_seq_len)
        self.norm2 = nn.RMSNorm(d_model)
        self.ffn   = SwiGLUFFN(d_model, ffn_mult)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Single MTP module at one depth
# ---------------------------------------------------------------------------

class SequentialMTPModule(nn.Module):
    """
    One sequential MTP module at prediction depth k (0-indexed).

    Owned parameters:
        enorm      : RMSNorm — applied to future token embedding
        hnorm      : RMSNorm — applied to incoming hidden state
        proj       : Linear(2*d -> d) — fuses the two normed vectors
        block      : MiniTransformerBlock — processes the fused representation
        final_norm : RMSNorm — applied before the shared lm_head

    The norms before proj are essential: the embedding and hidden state live
    in different magnitude regimes; normalising before concatenation ensures
    proj receives unit-variance inputs at init.

    Args:
        d_model     : model dimension
        n_heads     : attention heads in the transformer block
        depth       : 0-indexed depth (module 0 predicts offset +2, etc.)
        ffn_mult    : FFN hidden-dim multiplier
        max_seq_len : maximum sequence length
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        depth: int,
        ffn_mult: int = 4,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.depth = depth

        # Named enorm / hnorm as in DeepSeek-V3 weight files
        self.enorm = nn.RMSNorm(d_model)
        self.hnorm = nn.RMSNorm(d_model)

        # concat([hnorm(h), enorm(e)]) -> d  (2d -> d)
        self.proj  = nn.Linear(2 * d_model, d_model, bias=False)

        # Per-module transformer block (NOT shared across depths)
        self.block = MiniTransformerBlock(d_model, n_heads, ffn_mult, max_seq_len)

        # Final norm before the shared lm_head
        self.final_norm = nn.RMSNorm(d_model)

        self._init_proj()

    def _init_proj(self):
        # Two d-dim inputs concatenated doubles variance; scale init by 1/sqrt(2)
        nn.init.normal_(self.proj.weight, std=0.02 / math.sqrt(2.0))

    def forward(
        self,
        h_prev: torch.Tensor,
        future_embed: torch.Tensor,
        lm_head: nn.Linear,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h_prev       : (B, T', d)  hidden states from previous depth
            future_embed : (B, T', d)  embeddings of ground-truth future tokens
                           future_embed[:, i, :] = embed(token at position i+depth+1)
            lm_head      : shared Linear(d -> vocab_size) from the trunk

        Returns:
            h_out  : (B, T', d)         output representations for next depth
            logits : (B, T', vocab_size) predictions for token at offset (depth+2)
        """
        # Normalise each input stream independently before fusing
        h_n = self.hnorm(h_prev)       # (B, T', d)
        e_n = self.enorm(future_embed) # (B, T', d)

        # Concatenate along feature dimension and project back to d
        fused = torch.cat([h_n, e_n], dim=-1)  # (B, T', 2d)
        h = self.proj(fused)                    # (B, T', d)

        # Per-module transformer block
        h_out = self.block(h)                   # (B, T', d)

        # Output via shared head
        logits = lm_head(self.final_norm(h_out))  # (B, T', V)

        return h_out, logits


# ---------------------------------------------------------------------------
# Full sequential MTP stack
# ---------------------------------------------------------------------------

class MTPStack(nn.Module):
    """
    Stack of D sequential MTP modules, chained depth-by-depth.

    Prediction offsets:
        trunk output  -> offset +1  (standard next-token, not handled here)
        module 0      -> offset +2
        module 1      -> offset +3
        ...
        module D-1    -> offset +D+1

    Shared (not owned here, just referenced):
        embed_table : nn.Embedding  from trunk
        lm_head     : nn.Linear     from trunk

    The MTP loss is the average cross-entropy over all depths, multiplied by
    loss_scale.  Add it to the main model loss during training.

    Args:
        d_model     : model dimension
        n_heads     : attention heads per MTP transformer block
        n_modules   : number of sequential MTP depths D
        embed_table : shared token embedding table from the trunk
        lm_head     : shared output projection from the trunk
        loss_scale  : scalar weight applied to the averaged MTP loss
        ffn_mult    : FFN multiplier
        max_seq_len : maximum sequence length
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_modules: int,
        embed_table: nn.Embedding,
        lm_head: nn.Linear,
        loss_scale: float = 0.1,
        ffn_mult: int = 4,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.n_modules  = n_modules
        self.loss_scale = loss_scale

        # Shared references — just pointers, no parameter duplication
        self.embed_table = embed_table
        self.lm_head     = lm_head

        self.modules_list = nn.ModuleList([
            SequentialMTPModule(
                d_model=d_model,
                n_heads=n_heads,
                depth=k,
                ffn_mult=ffn_mult,
                max_seq_len=max_seq_len,
            )
            for k in range(n_modules)
        ])

    # ------------------------------------------------------------------
    # Training (teacher-forced)
    # ------------------------------------------------------------------

    def forward(
        self,
        trunk_hidden: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Teacher-forced training pass through all MTP depths.

        Indexing reference (i = position, k = 0-indexed depth):
            h_prev[:, i, :]           representation at depth k for position i
            future_embed[:, i, :]     embed(token at i + k + 1)   <- teacher
            target[:, i]              token at i + k + 2           <- label

        At depth k, valid_len = T - k - 2 positions have both a teacher token
        and a label.  The hidden state window shrinks by 1 at each depth.

        Args:
            trunk_hidden : (B, T, d_model)  trunk output (before lm_head)
            input_ids    : (B, T)           token ids

        Returns:
            mtp_loss       : scalar  (avg loss across depths * loss_scale)
            per_depth_loss : list[D] of scalar losses for logging/monitoring
        """
        B, T, d  = trunk_hidden.shape
        h        = trunk_hidden          # shrinks by 1 each depth
        per_depth_loss: list[torch.Tensor] = []

        for k, mod in enumerate(self.modules_list):
            # valid_len: positions with a label at offset k+2
            # need i + k + 2 <= T - 1  =>  i <= T - k - 3  =>  valid_len = T - k - 2
            valid_len = T - k - 2
            if valid_len <= 0:
                per_depth_loss.append(trunk_hidden.new_zeros(()))
                continue

            # Slice h to valid positions (it already shrank in previous iteration)
            # On iteration 0: h is (B, T, d),   we take [:, :T-2, :]
            # On iteration 1: h is (B, T-2, d), we take [:, :T-3, :] etc.
            h_prev = h[:, :valid_len, :]                            # (B, valid_len, d)

            # Teacher token: the ground-truth token at i+k+1 for each position i
            future_ids   = input_ids[:, k + 1 : k + 1 + valid_len] # (B, valid_len)
            future_embed = self.embed_table(future_ids)             # (B, valid_len, d)

            # Forward through this module
            h_out, logits = mod(h_prev, future_embed, self.lm_head) # (B, vl, d/V)

            # Label: token at i+k+2
            targets = input_ids[:, k + 2 : k + 2 + valid_len]      # (B, valid_len)

            loss = F.cross_entropy(
                logits.reshape(B * valid_len, -1),
                targets.reshape(B * valid_len),
            )
            per_depth_loss.append(loss)

            # h_out is (B, valid_len, d).  Pass to next depth; it will slice
            # to valid_len-1 = T-(k+1)-2 positions, consistent with the window.
            h = h_out

        # Average valid losses and scale
        valid_losses = [l for l in per_depth_loss if l.item() != 0.0 or l.requires_grad]
        if valid_losses:
            mtp_loss = torch.stack(valid_losses).mean() * self.loss_scale
        else:
            mtp_loss = trunk_hidden.new_zeros(())

        return mtp_loss, per_depth_loss

    # ------------------------------------------------------------------
    # Inference: speculative decoding draft
    # ------------------------------------------------------------------

    def draft_tokens(
        self,
        trunk_hidden: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressively draft D tokens beyond the last input position.

        At each depth k, the module is conditioned on the token sampled at
        depth k-1 (or on the trunk's top prediction for depth 0).  This
        produces a draft sequence of length D in a single forward pass through
        the MTP chain.

        Args:
            trunk_hidden : (B, T, d_model)  full trunk hidden states
            temperature  : sampling temperature
            top_k        : top-k filter before sampling

        Returns:
            draft : (B, D)  one draft token per depth
        """
        B = trunk_hidden.size(0)

        # Use only the last-position hidden state as the starting point
        h = trunk_hidden[:, -1:, :]   # (B, 1, d)
        draft_tokens: list[torch.Tensor] = []

        for k, mod in enumerate(self.modules_list):
            if k == 0:
                # Condition on the trunk's most-likely next token
                trunk_logits = self.lm_head(mod.final_norm(h))   # (B, 1, V)
                ctx_token    = self._sample(trunk_logits, temperature, top_k)  # (B, 1)
            else:
                ctx_token = draft_tokens[-1].unsqueeze(1)         # (B, 1)

            future_embed = self.embed_table(ctx_token)             # (B, 1, d)

            h_out, logits = mod(h, future_embed, self.lm_head)    # (B, 1, d/V)

            # Sample this depth's draft token
            sampled = self._sample(logits, temperature, top_k)     # (B, 1)
            draft_tokens.append(sampled.squeeze(1))                # (B,)

            h = h_out

        return torch.stack(draft_tokens, dim=1)   # (B, D)

    @staticmethod
    def _sample(
        logits: torch.Tensor,
        temperature: float,
        top_k: Optional[int],
    ) -> torch.Tensor:
        """Sample from logits (B, 1, V) -> (B, 1)."""
        scaled = logits[:, 0, :] / max(temperature, 1e-8)    # (B, V)
        if top_k is not None:
            thresh = scaled.topk(top_k, dim=-1).values[:, -1:]
            scaled = scaled.masked_fill(scaled < thresh, float("-inf"))
        probs = F.softmax(scaled, dim=-1)
        return torch.multinomial(probs, 1)   # (B, 1)

    # ------------------------------------------------------------------

    def parameter_count(self) -> dict:
        owned = sum(p.numel() for p in self.modules_list.parameters())
        return {
            "mtp_modules_owned":  owned,
            "per_module":         owned // max(self.n_modules, 1),
            "shared_embed":       self.embed_table.weight.numel(),
            "shared_lm_head":     self.lm_head.weight.numel(),
            "note": "shared parameters are owned by the trunk, not re-instantiated",
        }


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

class WithSequentialMTP(nn.Module):
    """
    Attaches MTPStack to any trunk with a compatible interface.

    Trunk requirements:
        trunk(input_ids) -> (B, T, d_model)   hidden states (not logits)
        trunk.embed      : nn.Embedding
        trunk.lm_head    : nn.Linear(d_model, vocab_size)
        trunk.final_norm : nn.RMSNorm (or any norm module)

    Usage::

        # Wrap an existing trunk
        model = WithSequentialMTP(trunk, n_mtp=2, loss_scale=0.3)

        # Training
        total_loss, info = model(input_ids)
        total_loss.backward()

        # Standard inference
        logits = model.infer(input_ids)

        # Speculative decoding draft
        draft = model.draft(input_ids)   # (B, n_mtp)

    Args:
        trunk       : language model trunk
        n_mtp       : number of sequential MTP modules
        loss_scale  : weight on MTP loss (DeepSeek used 0.3 -> 0.1 schedule)
        n_heads     : attention heads for MTP blocks; inferred from trunk if None
        ffn_mult    : FFN multiplier for MTP blocks
        max_seq_len : maximum sequence length
    """

    def __init__(
        self,
        trunk: nn.Module,
        n_mtp: int = 1,
        loss_scale: float = 0.1,
        n_heads: Optional[int] = None,
        ffn_mult: int = 4,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.trunk = trunk

        vocab_size, d_model = trunk.lm_head.weight.shape
        if n_heads is None:
            n_heads = getattr(trunk, "n_heads", max(1, d_model // 64))

        self.mtp = MTPStack(
            d_model=d_model,
            n_heads=n_heads,
            n_modules=n_mtp,
            embed_table=trunk.embed,
            lm_head=trunk.lm_head,
            loss_scale=loss_scale,
            ffn_mult=ffn_mult,
            max_seq_len=max_seq_len,
        )

    def get_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.trunk(input_ids)

    def get_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.trunk.lm_head(self.trunk.final_norm(hidden))

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, dict]:
        hidden = self.get_hidden(input_ids)

        # Main LM loss
        logits    = self.get_logits(hidden)
        main_loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            input_ids[:, 1:].reshape(-1),
        )

        # MTP losses
        mtp_loss, per_depth = self.mtp(hidden, input_ids)
        total = main_loss + mtp_loss

        return total, {
            "main_loss":        main_loss,
            "mtp_loss":         mtp_loss,
            "per_depth_losses": per_depth,
        }

    def infer(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Standard next-token logits; MTP chain bypassed entirely."""
        return self.get_logits(self.get_hidden(input_ids))

    def draft(
        self,
        input_ids: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Return n_mtp draft tokens via the sequential MTP chain."""
        with torch.no_grad():
            hidden = self.get_hidden(input_ids)
            return self.mtp.draft_tokens(hidden, temperature, top_k)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    B, T, d, V = 2, 32, 64, 512
    n_heads, n_mtp = 4, 3

    # Fake trunk outputs
    embed_table = nn.Embedding(V, d)
    lm_head     = nn.Linear(d, V, bias=False)
    lm_head.weight = embed_table.weight   # tied weights

    stack = MTPStack(
        d_model=d, n_heads=n_heads, n_modules=n_mtp,
        embed_table=embed_table, lm_head=lm_head, loss_scale=0.3,
    )

    trunk_hidden = torch.randn(B, T, d)
    input_ids    = torch.randint(0, V, (B, T))

    # Training pass
    mtp_loss, per_depth = stack(trunk_hidden, input_ids)
    print(f"MTP loss (scaled): {mtp_loss.item():.4f}")
    for k, l in enumerate(per_depth):
        print(f"  depth {k} loss: {l.item():.4f}")

    mtp_loss.backward()
    print("Backward: OK")

    # Speculative draft
    draft = stack.draft_tokens(trunk_hidden.detach())
    print(f"\nDraft tokens shape:  {draft.shape}")   # (B, n_mtp)
    print(f"Draft token values:  {draft.tolist()}")

    # Parameter breakdown
    print("\nParameter breakdown:")
    for k, v in stack.parameter_count().items():
        if isinstance(v, int):
            print(f"  {k:30s}: {v:,}")
        else:
            print(f"  {k:30s}: {v}")

    # Verify window shrinkage: depth k should have T-k-2 valid positions
    print("\nValid position counts per depth:")
    for k in range(n_mtp):
        print(f"  depth {k}: {T - k - 2} positions")
