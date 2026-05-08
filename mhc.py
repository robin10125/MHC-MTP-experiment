"""
Manifold-Constrained Hyper-Connections (mHC)
============================================
Papers:
  Hyper-Connections (ByteDance, ICLR 2025)  arxiv.org/abs/2409.19606
  mHC stability fix (DeepSeek)              arxiv.org/abs/2512.24880

Overview
--------
The standard residual connection maintains a single stream of hidden states
that every layer reads from and writes back to.  Hyper-Connections (HC)
generalise this to n parallel streams.  Each layer reads a weighted
combination of the n streams, applies its sublayer, then writes the result
back into all n streams via learned weights, while also mixing the streams
laterally via a learned (n x n) matrix H_res.

The instability problem:  stacking these mixing matrices multiplies their
spectral norms.  For a depth-64 network with unconstrained H_res the
composite gain can reach ~3000x, causing loss spikes and gradient explosions.

The mHC fix:  constrain H_res to the Birkhoff polytope (doubly stochastic
matrices) via Sinkhorn-Knopp normalisation.  Doubly stochastic matrices have
spectral norm <= 1 and are closed under multiplication, so the composite gain
stays bounded at ~1.6 regardless of depth.

Notation
--------
  B  : batch size
  T  : sequence length
  n  : number of parallel streams  (n_streams)
  d  : model dimension              (d_model)

Stream tensor shape throughout the trunk:  (B, T, n, d)

Initialisation (from the HC paper)
-----------------------------------
  H_res  = identity          streams pass through unchanged at init
  H_pre  = e_{k mod n}       rotating one-hot: layer k reads from stream k%n
                             (symmetry breaking so streams differentiate)
  H_post = ones              sublayer output broadcast equally into all streams

With this init every stream holds identical content and the model is
equivalent to a standard Pre-Norm residual network.  Streams differentiate
during training as the parameters are learned.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sinkhorn-Knopp projection onto the Birkhoff polytope
# ---------------------------------------------------------------------------

def sinkhorn(W: torch.Tensor, n_iters: int = 5, eps: float = 1e-8) -> torch.Tensor:
    """
    Project matrix W onto the Birkhoff polytope (doubly stochastic matrices).

    A doubly stochastic matrix S satisfies:
        all entries >= 0
        every row sums to 1
        every column sums to 1

    This is achieved by alternating row and column normalisation, starting
    from softplus(W) to enforce non-negativity.

    Args:
        W       : (..., n, n)  raw (unconstrained) parameter matrix
        n_iters : number of alternating normalisation steps
        eps     : small constant for numerical stability

    Returns:
        S : (..., n, n)  doubly stochastic matrix
    """
    S = F.softplus(W)
    for _ in range(n_iters):
        S = S / (S.sum(dim=-1, keepdim=True) + eps)   # row normalise
        S = S / (S.sum(dim=-2, keepdim=True) + eps)   # column normalise
    return S


# ---------------------------------------------------------------------------
# mHC residual wrapper
# ---------------------------------------------------------------------------

class mHCResidual(nn.Module):
    """
    Wraps any sublayer with the mHC residual mechanism.

    Forward pass for streams X of shape (B, T, n, d):

        1. Read    x  = sum_i  softmax(H_pre)_i * X[:,:,i,:]     -> (B,T,d)
        2. Compute y  = sublayer( norm(x) )                       -> (B,T,d)
        3. Mix     X' = matmul(H_res, X) across stream dimension  lateral mix
        4. Write   X' = X' + H_post[:,None] * y                   broadcast write

    H_res is projected to the Birkhoff polytope on every forward pass so its
    spectral norm is bounded to <= 1.

    Args:
        d_model        : model/embedding dimension
        n_streams      : number of parallel residual streams
        layer_idx      : index of this layer in the network (0-based);
                         sets the rotating one-hot in H_pre
        sublayer       : the wrapped nn.Module (attention, FFN, etc.)
        norm_cls       : normalisation class; default nn.RMSNorm
        sinkhorn_iters : Sinkhorn-Knopp iterations per forward pass
    """

    def __init__(
        self,
        d_model: int,
        n_streams: int,
        layer_idx: int,
        sublayer: nn.Module,
        norm_cls=None,
        sinkhorn_iters: int = 5,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_streams = n_streams
        self.layer_idx = layer_idx
        self.sublayer = sublayer
        self.sinkhorn_iters = sinkhorn_iters

        if norm_cls is None:
            norm_cls = nn.RMSNorm
        self.norm = norm_cls(d_model)

        # Raw (unconstrained) lateral mixing matrix.
        # Projected to Birkhoff polytope in forward() via sinkhorn().
        self.H_res_raw = nn.Parameter(torch.empty(n_streams, n_streams))

        # Read weights: (n,) -> softmax -> convex combination of streams.
        self.H_pre = nn.Parameter(torch.empty(n_streams))

        # Write weights: (n,) scalar multiplier per stream for the write-back.
        self.H_post = nn.Parameter(torch.empty(n_streams))

        self._init_parameters()

    # ------------------------------------------------------------------

    def _init_parameters(self):
        n = self.n_streams

        # H_res_raw: set large diagonal so that after softplus + Sinkhorn
        # the result is approximately the identity matrix.
        with torch.no_grad():
            self.H_res_raw.fill_(0.0)
            self.H_res_raw.diagonal().fill_(10.0)

        # H_pre: rotating one-hot  (symmetry breaking across streams)
        read_stream = self.layer_idx % n
        with torch.no_grad():
            self.H_pre.zero_()
            # Set a large value so softmax concentrates on this stream at init.
            # The other entries are 0, giving softmax ≈ e_{read_stream}.
            self.H_pre[read_stream] = 10.0

        # H_post: uniform write-back into all streams
        with torch.no_grad():
            self.H_post.fill_(1.0)

    # ------------------------------------------------------------------

    def forward(self, streams: torch.Tensor) -> torch.Tensor:
        """
        Args:
            streams : (B, T, n_streams, d_model)

        Returns:
            streams : (B, T, n_streams, d_model)  updated
        """
        B, T, n, d = streams.shape
        assert n == self.n_streams, f"Expected {self.n_streams} streams, got {n}"

        # ---- Step 1: Read -----------------------------------------------
        # Weighted combination of streams -> single input vector for sublayer
        read_w = F.softmax(self.H_pre, dim=0)                     # (n,)
        x = torch.matmul(streams.transpose(-1, -2), read_w)       # (B, T, d)

        # ---- Step 2: Sublayer -------------------------------------------
        y = self.sublayer(self.norm(x))                            # (B, T, d)

        # ---- Step 3: Lateral mix ----------------------------------------
        H_res = sinkhorn(self.H_res_raw, n_iters=self.sinkhorn_iters)  # (n, n)
        streams = torch.matmul(H_res, streams.reshape(B * T, n, d))
        streams = streams.reshape(B, T, n, d)

        # ---- Step 4: Write-back -----------------------------------------
        streams = streams + self.H_post.view(1, 1, n, 1) * y.unsqueeze(2)

        return streams


# ---------------------------------------------------------------------------
# Standard sublayers
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """Grouped-query-compatible causal multi-head attention."""

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 2048):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
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
        out = out.transpose(1, 2).reshape(B, T, d)
        return self.out_proj(out)


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, d_model: int, ffn_mult: int = 4):
        super().__init__()
        hidden = d_model * ffn_mult
        self.gate  = nn.Linear(d_model, hidden, bias=False)
        self.up    = nn.Linear(d_model, hidden, bias=False)
        self.down  = nn.Linear(hidden,  d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ---------------------------------------------------------------------------
# mHC Transformer block (attention + FFN, each wrapped in mHC)
# ---------------------------------------------------------------------------

class mHCTransformerBlock(nn.Module):
    """
    One transformer block: mHC-wrapped attention followed by mHC-wrapped FFN.

    Each sub-layer gets its own H_res, H_pre, H_post.  Layer indices are
    interleaved (2i for attention, 2i+1 for FFN) so the rotating one-hot
    in H_pre stays distinct across the full stack.

    Args:
        d_model     : model dimension
        n_heads     : attention heads
        n_streams   : parallel residual streams
        layer_idx   : block index in the stack (0-based)
        ffn_mult    : FFN hidden-dim multiplier
        max_seq_len : maximum sequence length (for causal mask)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_streams: int,
        layer_idx: int,
        ffn_mult: int = 4,
        max_seq_len: int = 2048,
        sinkhorn_iters: int = 5,
    ):
        super().__init__()

        self.attn_mhc = mHCResidual(
            d_model=d_model,
            n_streams=n_streams,
            layer_idx=layer_idx * 2,
            sublayer=CausalSelfAttention(d_model, n_heads, max_seq_len),
            sinkhorn_iters=sinkhorn_iters,
        )

        self.ffn_mhc = mHCResidual(
            d_model=d_model,
            n_streams=n_streams,
            layer_idx=layer_idx * 2 + 1,
            sublayer=SwiGLUFFN(d_model, ffn_mult),
            sinkhorn_iters=sinkhorn_iters,
        )

    def forward(self, streams: torch.Tensor) -> torch.Tensor:
        streams = self.attn_mhc(streams)
        streams = self.ffn_mhc(streams)
        return streams


# ---------------------------------------------------------------------------
# Full mHC language model
# ---------------------------------------------------------------------------

class mHCModel(nn.Module):
    """
    Decoder-only language model with mHC residual connections.

    Embedding -> stream expansion -> n mHC transformer blocks
    -> stream reduction (sum) -> final norm -> LM head.

    The LM head weights are scaled by 1/sqrt(n_streams) at init to
    compensate for the n-fold magnitude increase from summing n streams.
    This preserves variance-equivalence with a standard residual network
    at initialisation.

    Embedding weights are tied to the LM head (weight tying).

    Args:
        vocab_size  : vocabulary size
        d_model     : model dimension
        n_layers    : number of transformer blocks
        n_heads     : attention heads
        n_streams   : parallel residual streams  (default: 4)
        max_seq_len : maximum sequence length
        ffn_mult    : FFN hidden-dim multiplier
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        n_streams: int = 4,
        max_seq_len: int = 2048,
        ffn_mult: int = 4,
    ):
        super().__init__()
        self.n_streams = n_streams
        self.d_model = d_model

        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList([
            mHCTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                n_streams=n_streams,
                layer_idx=i,
                ffn_mult=ffn_mult,
                max_seq_len=max_seq_len,
            )
            for i in range(n_layers)
        ])

        self.final_norm = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.tok_embed.weight

        self._init_weights()

    def _init_weights(self):
        # Scale output projection down by 1/sqrt(n_streams) to compensate
        # for the sum reduction producing n-fold larger magnitude at init.
        with torch.no_grad():
            self.lm_head.weight.mul_(1.0 / math.sqrt(self.n_streams))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids : (B, T)  integer token ids

        Returns:
            logits    : (B, T, vocab_size)
        """
        B, T = input_ids.shape
        device = input_ids.device

        pos = torch.arange(T, device=device)
        x = self.tok_embed(input_ids) + self.pos_embed(pos)   # (B, T, d)

        # Expand single stream into n identical copies
        # clone() is needed so each stream gets its own gradient path
        streams = x.unsqueeze(2).expand(B, T, self.n_streams, self.d_model).clone()

        for block in self.blocks:
            streams = block(streams)

        # Sum reduction + 1/sqrt(n) is implicit in the LM head weight scaling
        x = streams.sum(dim=2)          # (B, T, d)
        x = self.final_norm(x)
        return self.lm_head(x)          # (B, T, vocab_size)


# ---------------------------------------------------------------------------
# Quick verification (run this file directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    cfg = dict(vocab_size=512, d_model=64, n_layers=4,
               n_heads=4, n_streams=4, max_seq_len=32)
    model = mHCModel(**cfg)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    ids = torch.randint(0, cfg["vocab_size"], (2, 16))
    logits = model(ids)
    print(f"Input shape:  {ids.shape}")
    print(f"Output shape: {logits.shape}")

    loss = logits.mean()
    loss.backward()
    print("Backward pass: OK")

    # Verify doubly stochastic constraint on all H_res matrices
    print("\nBirkhoff polytope verification:")
    for name, mod in model.named_modules():
        if isinstance(mod, mHCResidual):
            H = sinkhorn(mod.H_res_raw, n_iters=20)
            n = cfg["n_streams"]
            row_ok = H.sum(dim=1).allclose(torch.ones(n), atol=1e-5)
            col_ok = H.sum(dim=0).allclose(torch.ones(n), atol=1e-5)
            neg_ok = bool((H >= 0).all())
            print(f"  {name:40s}  rows={row_ok}  cols={col_ok}  nonneg={neg_ok}")
