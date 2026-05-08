"""
Experiment: mHC Trunk + Sequential MTP
Stream Reduction Strategy: Baseline Sum vs. Sinkhorn Mixing Matrix
==================================================================

Research question
-----------------
Does a learned, doubly-stochastic mixing matrix at the mHC trunk-to-MTP
interface outperform plain summation, and if so, does the benefit manifest
as better final loss, faster convergence, or both?

Hypothesis (sum condition)
    All mHC streams are equally useful as a starting point for multi-step
    lookahead prediction.  The MTP transformer blocks are expressive enough
    to compensate for any stream weighting, so the simple sum is sufficient.

Hypothesis (mixing matrix condition)
    After training, the mHC streams encode meaningfully different information.
    A learned, constrained mixing at the trunk-to-MTP boundary selectively
    emphasises streams that are more predictive at multi-step lookahead offsets,
    giving the MTP chain a better starting point and improving training speed
    or final performance.

Architecture overview
---------------------

    input_ids
        │
        ▼
    tok_embed + pos_embed          (shared embedding table)
        │
        ▼  (B, T, n_streams, d)
    mHC transformer blocks          n parallel residual streams
        │
        ▼  (B, T, n_streams, d)
    StreamReduction                 ← the thing being ablated
        │  sum:   streams.sum(dim=2)                     plain, no parameters
        │  mix:   sinkhorn(W) @ streams, W learned       doubly stochastic
        ▼  (B, T, d)
    final_norm
        │
        ├──► lm_head  ──► main LM loss     (trunk next-token prediction)
        │
        └──► MTPStack
                depth 0: proj + block + lm_head  ──► offset +2 loss
                depth 1: proj + block + lm_head  ──► offset +3 loss
                ...

The embedding table and lm_head are shared across the trunk and all MTP
depths, exactly as in the DeepSeek-V3 / Nemotron design.

How to use
----------
    from experiment import build_model, ExperimentConfig, train_step

    # Condition A: baseline sum
    cfg_sum = ExperimentConfig(reduction="sum")
    model_sum = build_model(cfg_sum)

    # Condition B: Sinkhorn mixing matrix
    cfg_mix = ExperimentConfig(reduction="mix")
    model_mix = build_model(cfg_mix)

    # Training loop (pseudocode)
    for batch in dataloader:
        loss, metrics = train_step(model, batch, optimizer)

    # Compare: metrics["main_loss"], metrics["mtp_loss"], metrics["per_depth_losses"]
    # Key diagnostic: metrics["mix_weights"] if reduction == "mix"

File dependencies
-----------------
    mhc.py   — mHCResidual, mHCTransformerBlock, sinkhorn
    mtp.py   — MTPStack, SequentialMTPModule
"""

import math
import time
import dataclasses
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Local modules — mhc.py and mtp.py must be in the same directory
from mhc import mHCResidual, mHCTransformerBlock, sinkhorn
from mtp import MTPStack


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ExperimentConfig:
    """
    All hyperparameters for one experimental condition.

    Keeping everything in a single dataclass makes it easy to serialise,
    diff two configs, or sweep over values programmatically.
    """

    # --- Model architecture ---
    vocab_size:  int = 512
    d_model:     int = 128
    n_layers:    int = 4       # trunk transformer depth
    n_heads:     int = 4       # attention heads (trunk and MTP blocks)
    n_streams:   int = 4       # mHC parallel residual streams
    ffn_mult:    int = 4       # FFN hidden-dim multiplier
    max_seq_len: int = 128

    # --- MTP ---
    n_mtp:       int   = 2     # number of sequential MTP depths
    mtp_loss_scale: float = 0.3

    # --- Reduction strategy (the experimental variable) ---
    reduction: Literal["sum", "mix"] = "sum"
    sinkhorn_iters: int = 5    # for both H_res in trunk and mixing matrix

    # --- Training ---
    lr:          float = 3e-4
    batch_size:  int   = 8
    seq_len:     int   = 64
    n_steps:     int   = 200

    # --- Logging ---
    log_every:   int   = 20
    seed:        int   = 42


# ---------------------------------------------------------------------------
# Stream reduction modules
# ---------------------------------------------------------------------------

class SumReduction(nn.Module):
    """
    Baseline: sum the n mHC streams along the stream dimension.

    No learned parameters.  At initialisation (when all streams are identical
    copies of the embedding), this produces a representation with n× larger
    magnitude than a single stream.  The LM head weight is scaled by
    1/sqrt(n_streams) at build time to compensate, as in the original mHC paper.

    This is the null hypothesis condition: the MTP chain receives an
    unweighted aggregate of all trunk streams.
    """

    def __init__(self, n_streams: int, d_model: int):
        super().__init__()
        self.n_streams = n_streams
        # No parameters.

    def forward(self, streams: torch.Tensor) -> torch.Tensor:
        """
        Args:
            streams : (B, T, n_streams, d_model)
        Returns:
            h       : (B, T, d_model)
        """
        return streams.sum(dim=2)

    def mixing_weights(self) -> Optional[torch.Tensor]:
        """Returns None — no learned weights to inspect."""
        return None


class SinkhornMixReduction(nn.Module):
    """
    Experimental: reduce n mHC streams to a single vector via a learned
    (n x n) doubly-stochastic mixing matrix, then sum the reweighted streams.

    The mixing matrix W is constrained to the Birkhoff polytope (same
    constraint used for H_res inside the mHC trunk) by projecting a raw
    parameter matrix through Sinkhorn-Knopp normalisation before each
    forward pass.

    Why doubly stochastic here?
        The column constraint ensures every input stream contributes equally
        in aggregate across all output positions — no stream is silenced,
        which would discard the diversity that mHC worked to create.
        The row constraint ensures the output is a properly normalised
        convex combination at each position, preventing magnitude blow-up.

    Initialisation:
        The raw parameter is set so that after softplus + Sinkhorn the
        result is close to the identity matrix.  With identity mixing and
        n streams of identical content (at initialisation), the sum of
        reweighted streams equals the sum of original streams — perfectly
        equivalent to SumReduction at step 0.  The mixing matrix is free
        to diverge from identity during training as streams differentiate.

    Output scaling:
        As in SumReduction, the downstream LM head weight is scaled by
        1/sqrt(n_streams) at build time.  Both conditions are therefore
        variance-equivalent at initialisation, making their training
        curves directly comparable.

    Args:
        n_streams      : number of mHC parallel streams
        d_model        : model dimension (not used in mixing, but stored for clarity)
        sinkhorn_iters : Sinkhorn-Knopp iterations per forward pass
    """

    def __init__(self, n_streams: int, d_model: int, sinkhorn_iters: int = 5):
        super().__init__()
        self.n_streams = n_streams
        self.sinkhorn_iters = sinkhorn_iters

        # Raw parameter: initialised so softplus(W_raw) after Sinkhorn ≈ identity.
        # Large diagonal, small off-diagonal → near-identity after normalisation.
        self.W_raw = nn.Parameter(torch.zeros(n_streams, n_streams))
        with torch.no_grad():
            self.W_raw.diagonal().fill_(10.0)

    def forward(self, streams: torch.Tensor) -> torch.Tensor:
        """
        Args:
            streams : (B, T, n_streams, d_model)
        Returns:
            h       : (B, T, d_model)
        """
        # Project to Birkhoff polytope
        W = sinkhorn(self.W_raw, n_iters=self.sinkhorn_iters)   # (n, n)

        # Reweight streams: output_stream_i = sum_j  W[i,j] * input_stream_j
        # Then sum over the output stream dimension to produce a single vector.
        # This is equivalent to a learned weighted sum with doubly stochastic weights.
        B, T, n, d = streams.shape
        mixed = torch.matmul(W, streams.reshape(B * T, n, d))
        return mixed.reshape(B, T, n, d).sum(dim=2)              # (B, T, d)

    def mixing_weights(self) -> torch.Tensor:
        """
        Return the current doubly-stochastic mixing matrix for inspection.

        Useful for monitoring how the mixing matrix evolves during training:
        a matrix that stays near-identity means the condition is learning the
        same thing as the sum baseline; divergence from identity is evidence
        that the learned mixing is doing something the sum cannot.
        """
        with torch.no_grad():
            return sinkhorn(self.W_raw, n_iters=self.sinkhorn_iters)


# ---------------------------------------------------------------------------
# mHC trunk: produces hidden states (not logits)
# ---------------------------------------------------------------------------

class mHCTrunk(nn.Module):
    """
    The mHC language model trunk, factored to return hidden states rather
    than logits.  This is the key adaptation from mHCModel in mhc.py needed
    to integrate with the MTP stack via WithSequentialMTP.

    The trunk owns:
        tok_embed   : token embedding table (shared with MTP via lm_head weight tying)
        pos_embed   : positional embedding
        blocks      : stack of mHCTransformerBlocks
        reduction   : SumReduction or SinkhornMixReduction
        final_norm  : applied after reduction, before lm_head
        lm_head     : output projection (weight-tied to tok_embed)

    The MTP wrapper will call trunk(input_ids) -> (B, T, d), then apply
    trunk.lm_head and trunk.final_norm itself for the main loss, and pass
    the hidden states into the MTP chain.

    Attributes used by MTPStack:
        self.embed      : alias for tok_embed  (MTPStack looks for .embed)
        self.lm_head    : nn.Linear
        self.final_norm : nn.RMSNorm
        self.n_heads    : int  (MTPStack infers MTP block heads from this)
    """

    def __init__(self, cfg: ExperimentConfig):
        super().__init__()
        self.n_streams = cfg.n_streams
        self.d_model   = cfg.d_model
        self.n_heads   = cfg.n_heads

        # Embeddings
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.d_model)

        # mHC transformer blocks
        self.blocks = nn.ModuleList([
            mHCTransformerBlock(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                n_streams=cfg.n_streams,
                layer_idx=i,
                ffn_mult=cfg.ffn_mult,
                max_seq_len=cfg.max_seq_len,
                sinkhorn_iters=cfg.sinkhorn_iters,
            )
            for i in range(cfg.n_layers)
        ])

        # Stream reduction: the variable being ablated
        if cfg.reduction == "sum":
            self.reduction = SumReduction(cfg.n_streams, cfg.d_model)
        elif cfg.reduction == "mix":
            self.reduction = SinkhornMixReduction(
                cfg.n_streams, cfg.d_model, cfg.sinkhorn_iters
            )
        else:
            raise ValueError(f"Unknown reduction: {cfg.reduction!r}")

        self.final_norm = nn.RMSNorm(cfg.d_model)

        # Output projection, weight-tied to the embedding table
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_embed.weight

        # Alias so MTPStack can find the embedding table as trunk.embed
        self.embed = self.tok_embed

        self._init_weights()

    def _init_weights(self):
        # Scale LM head down by 1/sqrt(n_streams) to compensate for the
        # n-fold magnitude increase from summing n identical streams at init.
        # Both reduction conditions apply the same correction so their output
        # distributions are variance-equivalent at step 0.
        with torch.no_grad():
            self.lm_head.weight.mul_(1.0 / math.sqrt(self.n_streams))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Run the trunk and return hidden states (NOT logits).

        The MTP wrapper calls this, then applies final_norm + lm_head itself
        for the main loss, and passes the raw hidden states to the MTP chain.

        Args:
            input_ids : (B, T)

        Returns:
            hidden    : (B, T, d_model)   pre-norm hidden states
        """
        B, T = input_ids.shape
        device = input_ids.device

        pos = torch.arange(T, device=device)
        x = self.tok_embed(input_ids) + self.pos_embed(pos)    # (B, T, d)

        # Expand to n parallel streams
        streams = x.unsqueeze(2).expand(
            B, T, self.n_streams, self.d_model
        ).clone()                                               # (B, T, n, d)

        # mHC transformer blocks
        for block in self.blocks:
            streams = block(streams)

        # Reduce n streams -> single vector
        hidden = self.reduction(streams)                        # (B, T, d)

        return hidden   # caller applies final_norm + lm_head


# ---------------------------------------------------------------------------
# Full model: trunk + sequential MTP chain
# ---------------------------------------------------------------------------

class mHCWithMTP(nn.Module):
    """
    Combines the mHC trunk with the sequential MTP stack.

    This is the top-level model class for both experimental conditions.
    The only difference between the two conditions is the trunk's reduction
    module (SumReduction vs SinkhornMixReduction); everything else is identical.

    The MTPStack is initialised with references to trunk.embed and trunk.lm_head
    so the embedding table and vocabulary projection are truly shared (not copied)
    across the trunk and all MTP depths, as in DeepSeek-V3 and Nemotron.

    Args:
        cfg : ExperimentConfig; cfg.reduction controls which condition is built.
    """

    def __init__(self, cfg: ExperimentConfig):
        super().__init__()
        self.cfg = cfg

        self.trunk = mHCTrunk(cfg)

        self.mtp = MTPStack(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_modules=cfg.n_mtp,
            embed_table=self.trunk.embed,      # shared
            lm_head=self.trunk.lm_head,        # shared
            loss_scale=cfg.mtp_loss_scale,
            ffn_mult=cfg.ffn_mult,
            max_seq_len=cfg.max_seq_len,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Full training forward pass.

        Args:
            input_ids : (B, T)

        Returns:
            total_loss : scalar
            metrics    : dict with keys:
                           main_loss        — trunk next-token cross-entropy
                           mtp_loss         — scaled average MTP loss
                           per_depth_losses — list of D scalars
                           mix_weights      — (n, n) tensor if reduction=="mix", else None
        """
        # Run trunk: returns hidden states (B, T, d)
        hidden = self.trunk(input_ids)

        # Main LM loss: apply final_norm + lm_head, predict next token
        logits = self.trunk.lm_head(self.trunk.final_norm(hidden))  # (B, T, V)
        main_loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            input_ids[:, 1:].reshape(-1),
        )

        # MTP losses: teacher-forced, sequential, shrinking window
        mtp_loss, per_depth = self.mtp(hidden, input_ids)

        total = main_loss + mtp_loss

        # Collect the mixing matrix if this is the mix condition.
        # Tracking its evolution during training is the primary diagnostic for
        # whether the mixing matrix is doing something beyond what sum does.
        mix_weights = None
        if isinstance(self.trunk.reduction, SinkhornMixReduction):
            mix_weights = self.trunk.reduction.mixing_weights()

        return total, {
            "main_loss":        main_loss,
            "mtp_loss":         mtp_loss,
            "per_depth_losses": per_depth,
            "mix_weights":      mix_weights,
        }

    def infer(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Standard next-token logits for evaluation; MTP bypassed."""
        hidden = self.trunk(input_ids)
        return self.trunk.lm_head(self.trunk.final_norm(hidden))

    def parameter_count(self) -> dict:
        trunk_params = sum(p.numel() for p in self.trunk.parameters())
        mtp_params   = sum(p.numel() for p in self.mtp.modules_list.parameters())
        mix_params   = (
            sum(p.numel() for p in self.trunk.reduction.parameters())
            if isinstance(self.trunk.reduction, SinkhornMixReduction)
            else 0
        )
        return {
            "trunk_total":   trunk_params,
            "mtp_modules":   mtp_params,
            "mix_matrix":    mix_params,
            "grand_total":   trunk_params + mtp_params,
        }


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def build_model(cfg: ExperimentConfig) -> mHCWithMTP:
    """Construct and return a model for the given config."""
    torch.manual_seed(cfg.seed)
    return mHCWithMTP(cfg)


def make_batch(cfg: ExperimentConfig, device: torch.device) -> torch.Tensor:
    """
    Generate a random integer token batch.

    In a real experiment, replace this with your DataLoader.
    The sequence length is cfg.seq_len; the model needs at least
    n_mtp + 2 tokens to have any valid MTP positions.
    """
    return torch.randint(
        0, cfg.vocab_size,
        (cfg.batch_size, cfg.seq_len),
        device=device,
    )


def train_step(
    model: mHCWithMTP,
    batch: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> dict:
    """
    Single training step: forward, backward, optimizer update.

    Returns the metrics dict from model.forward() with scalar values
    detached from the graph for logging.
    """
    model.train()
    optimizer.zero_grad()

    total_loss, metrics = model(batch)
    total_loss.backward()

    # Gradient clipping helps stabilise early training, especially with mHC
    # where Sinkhorn projection adds a backward-pass overhead.
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

    # Detach all tensors in metrics for safe logging
    return {
        k: (v.item() if isinstance(v, torch.Tensor) and v.numel() == 1
            else ([x.item() for x in v] if isinstance(v, list)
            else v))
        for k, v in metrics.items()
    }


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_condition(cfg: ExperimentConfig, device: torch.device) -> dict:
    """
    Train one experimental condition and return a log of training metrics.

    Args:
        cfg    : full configuration for this condition
        device : torch device to run on

    Returns:
        log : dict with keys "steps", "main_loss", "mtp_loss",
              "per_depth_losses", "mix_weight_diag" (mix condition only),
              "wall_time_per_step"
    """
    print(f"\n{'='*60}")
    print(f"  Condition: reduction='{cfg.reduction}'")
    print(f"  d_model={cfg.d_model}, n_layers={cfg.n_layers}, "
          f"n_streams={cfg.n_streams}, n_mtp={cfg.n_mtp}")
    print(f"{'='*60}")

    torch.manual_seed(cfg.seed)
    model = build_model(cfg).to(device)

    # Parameter count report
    counts = model.parameter_count()
    print(f"  Parameters: trunk={counts['trunk_total']:,}  "
          f"mtp={counts['mtp_modules']:,}  "
          f"mix={counts['mix_matrix']:,}  "
          f"total={counts['grand_total']:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    # Log storage
    log = {
        "steps":             [],
        "main_loss":         [],
        "mtp_loss":          [],
        "per_depth_losses":  [],  # list of lists
        "mix_weight_diag":   [],  # diagonal of mixing matrix (mix condition only)
        "wall_time_per_step": [],
    }

    for step in range(1, cfg.n_steps + 1):
        batch = make_batch(cfg, device)

        t0 = time.perf_counter()
        metrics = train_step(model, batch, optimizer)
        t1 = time.perf_counter()

        log["steps"].append(step)
        log["main_loss"].append(metrics["main_loss"])
        log["mtp_loss"].append(metrics["mtp_loss"])
        log["per_depth_losses"].append(metrics["per_depth_losses"])
        log["wall_time_per_step"].append(t1 - t0)

        # For the mix condition, track the diagonal of the mixing matrix.
        # A diagonal that stays near 1/n means the matrix is near-uniform
        # (equivalent to sum); divergence signals genuine learning.
        if metrics["mix_weights"] is not None:
            diag = metrics["mix_weights"].diagonal().tolist()
            log["mix_weight_diag"].append(diag)

        if step % cfg.log_every == 0 or step == 1:
            depth_str = "  ".join(
                f"d{k}={l:.4f}"
                for k, l in enumerate(metrics["per_depth_losses"])
            )
            print(
                f"  step {step:4d} | "
                f"main={metrics['main_loss']:.4f}  "
                f"mtp={metrics['mtp_loss']:.4f}  "
                f"[{depth_str}]  "
                f"({(t1-t0)*1000:.1f}ms)"
            )

    return log


def compare_conditions(
    cfg_base: ExperimentConfig,
    device: torch.device,
) -> dict:
    """
    Run both conditions with identical configs except for reduction strategy,
    then print a side-by-side comparison of final metrics.

    Args:
        cfg_base : base config (reduction field is ignored; both are run)
        device   : torch device

    Returns:
        results : dict with keys "sum" and "mix", each containing a log dict
    """
    import dataclasses

    cfg_sum = dataclasses.replace(cfg_base, reduction="sum")
    cfg_mix = dataclasses.replace(cfg_base, reduction="mix")

    log_sum = run_condition(cfg_sum, device)
    log_mix = run_condition(cfg_mix, device)

    # Summary comparison
    def final_avg(log, key, last_n=20):
        """Average of the last last_n values for stability."""
        vals = log[key][-last_n:]
        return sum(vals) / len(vals)

    print(f"\n{'='*60}")
    print(f"  COMPARISON (avg of last {20} steps)")
    print(f"{'='*60}")
    print(f"  {'Metric':<25} {'sum':>12} {'mix':>12}  {'diff':>10}")
    print(f"  {'-'*60}")

    for key in ("main_loss", "mtp_loss"):
        s = final_avg(log_sum, key)
        m = final_avg(log_mix, key)
        diff = m - s
        sign = "+" if diff > 0 else ""
        print(f"  {key:<25} {s:>12.4f} {m:>12.4f}  {sign}{diff:>9.4f}")

    # Per-depth comparison
    n_mtp = cfg_base.n_mtp
    for k in range(n_mtp):
        def depth_avg(log, depth, last_n=20):
            return sum(log["per_depth_losses"][-last_n:][i][depth]
                       for i in range(min(last_n, len(log["per_depth_losses"])))
                       ) / min(last_n, len(log["per_depth_losses"]))
        s = depth_avg(log_sum, k)
        m = depth_avg(log_mix, k)
        diff = m - s
        sign = "+" if diff > 0 else ""
        print(f"  {'mtp_depth_' + str(k):<25} {s:>12.4f} {m:>12.4f}  {sign}{diff:>9.4f}")

    # Wall time
    s_t = sum(log_sum["wall_time_per_step"]) / len(log_sum["wall_time_per_step"]) * 1000
    m_t = sum(log_mix["wall_time_per_step"]) / len(log_mix["wall_time_per_step"]) * 1000
    print(f"  {'ms_per_step':<25} {s_t:>11.1f}ms {m_t:>11.1f}ms")

    # Mixing matrix diagonal report (mix condition only)
    if log_mix["mix_weight_diag"]:
        init_diag = log_mix["mix_weight_diag"][0]
        final_diag = log_mix["mix_weight_diag"][-1]
        print(f"\n  Mix matrix diagonal:")
        print(f"    step 1   : {[f'{x:.3f}' for x in init_diag]}")
        print(f"    step {cfg_base.n_steps}: {[f'{x:.3f}' for x in final_diag]}")
        uniform = 1.0 / cfg_base.n_streams
        max_dev = max(abs(x - uniform) for x in final_diag)
        print(f"    uniform reference: {uniform:.3f}")
        print(f"    max deviation from uniform at final step: {max_dev:.4f}")
        print(f"    Interpretation: {'matrix diverged from uniform (learning something)' if max_dev > 0.05 else 'matrix stayed near-uniform (equivalent to sum)'}")

    return {"sum": log_sum, "mix": log_mix}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Experiment configuration — small enough to run on CPU for validation,
    # scale up d_model/n_layers/n_steps for a real comparison.
    cfg = ExperimentConfig(
        vocab_size=512,
        d_model=128,
        n_layers=4,
        n_heads=4,
        n_streams=4,
        n_mtp=2,
        mtp_loss_scale=0.3,
        ffn_mult=4,
        max_seq_len=128,
        batch_size=8,
        seq_len=64,
        n_steps=200,
        log_every=40,
        lr=3e-4,
        seed=42,
    )

    results = compare_conditions(cfg, device)

    # The results dict contains the full training logs for both conditions.
    # To do more detailed analysis (learning curves, per-depth breakdown,
    # mixing matrix evolution), iterate over results["sum"] and results["mix"].
    #
    # Key questions to answer from the logs:
    #   1. Does mix reach lower main_loss than sum at step N?
    #      -> Evidence that stream selection at the interface helps the trunk.
    #   2. Does mix reach the same loss faster (fewer steps)?
    #      -> Evidence for training speed benefit.
    #   3. Do the per-depth MTP losses differ between conditions?
    #      -> Deeper depths benefit more if lookahead-specialised weighting helps.
    #   4. Does the mixing matrix diagonal diverge from 1/n_streams?
    #      -> If not, mix is learning the identity, which is equivalent to sum
    #         divided by n_streams — and the conditions are identical.
    #   5. Does mix cost meaningfully more wall time per step?
    #      -> The Sinkhorn projection on W_raw adds overhead; measure it.
