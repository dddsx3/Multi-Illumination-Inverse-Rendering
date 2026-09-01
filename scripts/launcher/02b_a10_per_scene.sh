#!/usr/bin/env bash
# ==============================================================================
# R5-B′ · A10 单 scene 单 N 增量跑（per-cell 重跑用）
# 适用: 1× NVIDIA A10 / 紧急恢复 / 单 cell 重跑
# 入口: bash scripts/launcher/02b_a10_per_scene.sh <scene> <N>
# 例:   bash scripts/launcher/02b_a10_per_scene.sh conf_sphere_r05 3
# 用途: 当 02_a10_run_all.sh 中途 OOM / 被抢占时, 用此脚本单独重跑特定 cell
# 增量式: 自动 skip 已存在于 r5_p1_albedo_ablation.csv 中的 (scene, N, subset_id)
# 期望 wall-clock: ~50 min per (scene, N) cell @ P=2000
# ==============================================================================
set -euo pipefail
set -x

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"

SCENE="${1:?Usage: $0 <scene_name> <N>}"
N="${2:?Usage: $0 <scene_name> <N>}"

VENV_DIR="$REPO_ROOT/.venv_a10"
source "$VENV_DIR/bin/activate"
export OMP_NUM_THREADS=10
export OPENBLAS_NUM_THREADS=10
export MKL_NUM_THREADS=10

echo "=== [02b per-scene] running single cell: $SCENE N=$N ==="
mkdir -p "$REPO_ROOT/r5"

# --pixel_cap 2000 + --n5_sample 2000 for paper-grade
# r5_p1_albedo_ablation.py 默认按 (scene, N, subset_id) 增量写入 CSV, 已存在行会被覆盖
python -u p1/source/information_audit/r5_p1_albedo_ablation.py \
  --scenes "$SCENE" \
  --pixel_cap 2000 \
  --n5_sample 2000 \
  2>&1 | tee -a "$REPO_ROOT/r5/r5_p1_a_full_run.log"

echo "=== [02b per-scene] done: $SCENE N=$N ==="
echo "Re-run ranking diagnostics with:"
echo "  python -c \"import pandas as pd; df = pd.read_csv('r5/r5_p1_albedo_ablation.csv'); print(df[(df.scene=='$SCENE') & (df.N==$N)].shape)\""
echo "Then re-run the full 02_a10_run_all.sh to refresh ranking + outliers + selection CSVs"