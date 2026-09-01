#!/usr/bin/env bash
# ==============================================================================
# R5-B′ · A10 全任务跑（P1-A full + Task G + solver arm）
# 适用: 1× NVIDIA A10 24GB / 20 vCPU / 116 GB RAM
# 入口: bash scripts/launcher/02_a10_run_all.sh
# 期望 wall-clock: ~6.5 h (P1-A GSIQ ~5.8 h CPU-bound + Task G ~17 min GPU + solver ~0.3 h GPU)
# 抵扣机时: ~21.5 机时 (wall-clock × 3.3)
# 产物:
#   r5/r5_p1_albedo_ablation.csv          (P1-A full, 6 scenes × 13920 rows)
#   r5/r5_p1_albedo_ablation_ranking.csv  (12 cell ranking)
#   r5/r5_p1_albedo_ablation_outliers.csv (boundary outliers)
#   r5/r5_p1_albedo_ablation_gate.md      (PASS-A verdict)
#   r5/r5_p1_albedo_ablation_selection.csv (solver arm)
#   r4pp/07_local_vs_global_init.csv      (Task G, 240 run)
#   r5/r5_p1_a_full_run.log               (完整 stdout 日志)
# ==============================================================================
set -euo pipefail
set -x

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"

# ---------- 0. sanity ----------
echo "=== [02 run] sanity ==="
uname -a
nproc
free -m
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>&1 | head -3
date
SECONDS=0

# ---------- 1. venv ----------
VENV_DIR="$REPO_ROOT/.venv_a10"
if [ ! -d "$VENV_DIR" ]; then
  echo "ERROR: $VENV_DIR not found. Run 01_a10_env_setup.sh first."
  exit 1
fi
source "$VENV_DIR/bin/activate"

# BLAS thread tuning: 10 for eigh, 10 for OS/solver
export OMP_NUM_THREADS=10
export OPENBLAS_NUM_THREADS=10
export MKL_NUM_THREADS=10

# ---------- 2. P1-A full: GSIQ + solver arm ----------
# 6 dev scenes × N{3, 5} × pixel_cap=2000 × 2 score variants (O, A)
# 同时启用 --solver: 在 GSIQ 完成后自动跑 solver arm (top-10 O + top-10 A + 10 random per cell)
# 30 solver runs per cell × 12 cells = 360 runs total
echo "=== [02 run] P1-A full (GSIQ + solver arm) ==="
mkdir -p "$REPO_ROOT/r5"
SCENES_CSV="conf_sphere_r05 conf_cube_axis conf_prism8 conf_egg conf_cylinder_r06_d06 conf_ellipsoid_z06"

# Background launch Task G (GPU-only, 240 runs × 2.5s ≈ 10 min) so it can run while GSIQ is computing
# But careful: GSIQ doesn't use GPU, so they can fully overlap.
TASKG_PID=""
if [ ! -f "$REPO_ROOT/r4pp/07_local_vs_global_init.csv" ] || [ "$(wc -l < "$REPO_ROOT/r4pp/07_local_vs_global_init.csv")" -lt 240 ]; then
  echo "=== [02 run] launching Task G in background ==="
  ( python -u p1/source/information_audit/r4pp_local_vs_global.py 2>&1 | tee "$REPO_ROOT/r4pp/task_g_run.log" ) &
  TASKG_PID=$!
fi

python -u p1/source/information_audit/r5_p1_albedo_ablation.py \
  --scenes $SCENES_CSV \
  --pixel_cap 2000 \
  --n5_sample 2000 \
  --n3_limit 4960 \
  --solver \
  --n_top 10 --n_random 10 \
  2>&1 | tee "$REPO_ROOT/r5/r5_p1_a_full_run.log"

ELAPSED=$SECONDS
echo "=== [02 run] P1-A full done in $(($ELAPSED/3600))h $(($ELAPSED%3600/60))m ==="

# ---------- 3. wait for Task G ----------
if [ -n "$TASKG_PID" ]; then
  echo "=== [02 run] waiting for Task G (PID=$TASKG_PID) ==="
  wait "$TASKG_PID"
  ELAPSED2=$SECONDS
  echo "=== [02 run] Task G done; total $(($ELAPSED2/3600))h $(($ELAPSED2%3600/60))m ==="
fi

# ---------- 4. verify outputs ----------
echo "=== [02 run] verifying outputs ==="
ALL_OK=1
for f in \
  "$REPO_ROOT/r5/r5_p1_albedo_ablation.csv" \
  "$REPO_ROOT/r5/r5_p1_albedo_ablation_ranking.csv" \
  "$REPO_ROOT/r5/r5_p1_albedo_ablation_outliers.csv" \
  "$REPO_ROOT/r5/r5_p1_albedo_ablation_gate.md" \
  "$REPO_ROOT/r5/r5_p1_albedo_ablation_selection.csv" \
  "$REPO_ROOT/r4pp/07_local_vs_global_init.csv" ; do
  if [ -f "$f" ]; then
    SIZE=$(stat -c %s "$f")
    echo "  [OK] $f ($SIZE bytes)"
  else
    echo "  [MISS] $f"
    ALL_OK=0
  fi
done

if [ "$ALL_OK" -eq 0 ]; then
  echo "ERROR: some outputs missing. Check logs."
  exit 1
fi

# ---------- 5. parse gate verdict ----------
echo "=== [02 run] parse P1-A gate verdict ==="
grep -E "Gate verdict|median rho" "$REPO_ROOT/r5/r5_p1_albedo_ablation_gate.md" | head -3

# ---------- 6. commit + push ----------
echo "=== [02 run] git commit + push ==="
cd "$REPO_ROOT"
git add r5/r5_p1_albedo_ablation*.csv r5/r5_p1_albedo_ablation_gate.md \
        r5/r5_p1_a_full_run.log \
        r4pp/07_local_vs_global_init.csv r4pp/task_g_run.log 2>/dev/null || true
git status --short

read -p "Commit + push results? (y/N) " ans
if [ "$ans" = "y" ] || [ "$ans" = "Y" ]; then
  MSG="feat(r5-p1): A10 P1-A full + Task G + solver arm ($(date -u +%FT%TZ))"
  git commit -m "$MSG"
  git push origin HEAD 2>&1 | tail -10
  echo "Pushed. Any teammate can now git pull."
fi

ELAPSED=$SECONDS
TOTAL_H=$((ELAPSED/3600))
TOTAL_M=$((ELAPSED%3600/60))
echo ""
echo "============================================"
echo "[02 run] ALL DONE in ${TOTAL_H}h ${TOTAL_M}m"
# Pure shell arithmetic; avoid bc (may not be installed)
BILLED_M=$((ELAPSED * 11 / 1200))   # ELAPSED * 3.3 / 3600 = ELAPSED * 33 / 36000 ≈ ELAPSED * 11 / 12000
echo "抵扣机时 ~$((BILLED_M / 60)).$((BILLED_M % 60 / 6)) (×3.3 抵扣)"
echo "============================================"
echo ""
echo "Next step: open r5/r5_p1_a_full_run.log + r5/r5_p1_albedo_ablation_gate.md"
echo "and r4pp/task_g_run.log for analysis."