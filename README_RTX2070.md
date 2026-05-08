# RTX 2070 Setup

This repo is now prepared for an RTX 2070-class GPU.

## Install

```bash
./setup_rtx2070.sh
source .venv/bin/activate
```

The setup script installs PyTorch CUDA wheels from the official PyTorch wheel
index, then installs the rest of `requirements.txt`.

## Smoke Test

Use a tiny run first:

```bash
python train.py --profile rtx2070 --model mhc_mtp --reduction sum \
  --dataset wikitext2 --max_train_tokens 200000 --max_val_tokens 20000 \
  --n_steps 2 --val_every 2 --save_every 2
```

## Training

Conservative 8 GB starting point:

```bash
python train.py --profile rtx2070 --model mhc_mtp --reduction sum
python train.py --profile rtx2070 --model mhc_mtp --reduction mix
python train.py --profile rtx2070 --model residual_only
python train.py --profile rtx2070 --model mhc_only
python train.py --profile rtx2070 --model residual_mtp
```

The `rtx2070` profile changes the default training shape to:

- `d_model=384`
- `n_layers=6`
- `n_heads=6`
- `seq_len=256`
- `batch_size=2`
- `grad_accum=16`
- `dtype=float16`
- `sinkhorn_iters=5`

Override any of those flags on the command line if you have VRAM headroom.
If you hit CUDA out-of-memory, reduce `--batch_size` first, then `--seq_len`.
If host RAM is the bottleneck while tokenizing WikiText-103, use
`--max_train_tokens` / `--max_val_tokens` or start with `--dataset wikitext2`.

## Notes

The RTX 2070 does not have native BF16 tensor cores, so `train.py` now uses
FP16 autocast automatically on pre-Ampere CUDA GPUs. The old `bfloat16` default
was appropriate for newer cards such as RTX 30/40-series, but not for Turing.
