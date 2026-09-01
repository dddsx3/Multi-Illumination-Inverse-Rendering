#!/usr/bin/env bash
# ==============================================================================
# R5-B′ · A10 环境准备（一次性，启动后立即跑）
# 适用: 1× NVIDIA A10 24GB / 20 vCPU / 116 GB RAM / Ubuntu 22.04
# 入口: bash scripts/launcher/01_a10_env_setup.sh
# 期望 wall-clock: ~5-10 min
# 产物: .venv_t4_a10/ (Python + torch cu128), 完整 LFS data pulled
# ==============================================================================
set -euo pipefail
set -x

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"

echo "=== [01 env] sanity check ==="
uname -a
nproc
free -m
df -h "$REPO_ROOT" | tail -1
date

echo "=== [01 env] git status (must be clean or all R5-B' changes committed) ==="
git status --short

echo "=== [01 env] git LFS pull ==="
if command -v git-lfs >/dev/null 2>&1; then
  git lfs pull
else
  echo "WARN: git-lfs not installed. Install via 'apt-get install -y git-lfs' first."
fi

echo "=== [01 env] verify data layout ==="
DATA_ROOT="$REPO_ROOT/p1/calibration_set/data_sun_confirmatory"
ls -la "$DATA_ROOT" | head -10
N_SCENES=$(ls -d "$DATA_ROOT"/conf_* | wc -l)
echo "Found $N_SCENES scene directories"
if [ "$N_SCENES" -lt 19 ]; then
  echo "ERROR: expected ≥19 scenes, got $N_SCENES. LFS pull failed."
  exit 1
fi

echo "=== [01 env] Python venv ==="
PYTHON="${PYTHON:-python3}"
$PYTHON --version
which $PYTHON

VENV_DIR="$REPO_ROOT/.venv_a10"
if [ ! -d "$VENV_DIR" ]; then
  echo "=== creating venv at $VENV_DIR ==="
  $PYTHON -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --quiet --upgrade pip

echo "=== [01 env] PyTorch CUDA 12.8 (A10 SM86 兼容) ==="
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu128
fi
pip install --quiet numpy scipy pandas

echo "=== [01 env] verify CUDA torch ==="
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
dev = torch.cuda.get_device_name(0)
mem = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'Device: {dev}, total memory: {mem:.1f} GB')
x = torch.randn(100, 100, device='cuda')
y = x @ x
print('Sanity matmul OK, sum:', float(y.sum()))
print('torch version:', torch.__version__)
print('CUDA version:', torch.version.cuda)
"

echo "=== [01 env] OpenBLAS thread config ==="
# A10 20 vCPU: 10 threads for eigh, 10 threads free for solver/OS
export OMP_NUM_THREADS=10
export OPENBLAS_NUM_THREADS=10
export MKL_NUM_THREADS=10
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"

echo ""
echo "=========================================="
echo "[01 env] DONE — ready for 02_a10_run_all.sh"
echo "=========================================="
echo "venv: $VENV_DIR"
echo "data: $DATA_ROOT ($N_SCENES scenes)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "next: bash scripts/launcher/02_a10_run_all.sh"