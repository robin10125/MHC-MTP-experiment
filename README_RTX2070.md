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
- `sinkhorn_iters=20`

Override any of those flags on the command line if you have VRAM headroom.
If you hit CUDA out-of-memory, reduce `--batch_size` first, then `--seq_len`.
If host RAM is the bottleneck while tokenizing WikiText-103, use
`--max_train_tokens` / `--max_val_tokens` or start with `--dataset wikitext2`.

## Larger 8 GB Targets

The default profile is intentionally conservative. A synthetic worst-case
`mhc_mtp --reduction mix` train step on this RTX 2070 fit much larger configs:

| config | params | peak reserved VRAM |
| --- | ---: | ---: |
| `d_model=384 n_layers=6 n_heads=6 batch_size=2` | 38.9M | 1.0 GB |
| `d_model=768 n_layers=12 n_heads=12 batch_size=2` | 173.4M | 3.6 GB |
| `d_model=1024 n_layers=12 n_heads=16 batch_size=2` | 290.9M | 5.6 GB |
| `d_model=1024 n_layers=14 n_heads=16 batch_size=2` | 324.5M | 6.2 GB |
| `d_model=1024 n_layers=16 n_heads=16 batch_size=2` | 358.0M | 6.9 GB |

Recommended practical target:

```bash
python train.py --profile rtx2070 --dataset wikitext103 \
  --d_model 1024 --n_layers 14 --n_heads 16 \
  --batch_size 2 --grad_accum 16 --seq_len 256
```

Aggressive target:

```bash
python train.py --profile rtx2070 --dataset wikitext103 \
  --d_model 1024 --n_layers 16 --n_heads 16 \
  --batch_size 2 --grad_accum 16 --seq_len 256
```

The aggressive target leaves little room for allocator fragmentation or other
desktop GPU use. `d_model=1152 n_layers=16 n_heads=18 batch_size=2` OOMed in
the worst-case arm.

You can rerun the measurement with:

```bash
python profile_rtx2070_model_size.py --model mhc_mtp --reduction mix \
  --d_model 1024 --n_layers 14 --n_heads 16 --batch_size 2 --seq_len 256
```

## Notes

The RTX 2070 does not have native BF16 tensor cores, so `train.py` now uses
FP16 autocast automatically on pre-Ampere CUDA GPUs. The old `bfloat16` default
was appropriate for newer cards such as RTX 30/40-series, but not for Turing.
