#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip

# RTX 2070 is a CUDA-capable Turing GPU. The cu118 wheels are a conservative
# choice for older NVIDIA drivers and include the CUDA runtime used by PyTorch.
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt

python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", ".".join(map(str, torch.cuda.get_device_capability(0))))
    print("vram_gb:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2))
PY
