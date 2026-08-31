#!/bin/bash
# R4'-C 确认集批量渲染驱动（每场景独立 blenderproc 进程——踩坑 #2：
# enable_* 进程级一次性； INC-001 修复后的 render_multilight 自动做帧级校验）。
# 用法: bash p1/source/generation/run_confirmatory_batch.sh <start_idx> [end_idx]
#   对 confirmatory_list.txt 的 [start, end) 行逐场景渲染。
# 日志: p1/logs/confirmatory_batch/<scene>.log（含 [INC-001] 重试行）
set -u
REPO="/d/MIR_Archive_20260829/Multi-Illumination-Inverse-Rendering"
BP="/c/Users/35702/AppData/Local/Programs/Python/Python310/Scripts/blenderproc"
LIST="$REPO/p1/calibration_set/meshes_confirmatory/confirmatory_list.txt"
OUT="$REPO/p1/calibration_set/data_sun_confirmatory"
LOGDIR="$REPO/p1/logs/confirmatory_batch"
SEED=20260901
START=${1:-0}
END=${2:-999}
mkdir -p "$LOGDIR"

n_total=$(wc -l < "$LIST")
i=0
ok=0; fail=0
while IFS= read -r obj; do
  i=$((i+1))
  [ "$i" -lt "$((START+1))" ] && continue
  [ "$i" -gt "$END" ] && break
  scene=$(basename "$obj" .obj)
  # 已完成的场景（有全部 32 帧）跳过——断点续跑
  if [ -f "$OUT/$scene/light_032_lin.npy" ]; then
    echo "[$i/$n_total] $scene SKIP (already done)"
    ok=$((ok+1)); continue
  fi
  echo "[$i/$n_total] $scene rendering..."
  echo "$obj" > /tmp/_conf_one_obj.txt
  if "$BP" run "$REPO/p1/source/generation/render_multilight.py" \
      --obj_list /tmp/_conf_one_obj.txt \
      --out_dir "$OUT" --num_lights 32 --gpu --size 128 --samples 32 \
      --light_type sun --light_energy 3.0 --seed "$SEED" \
      > "$LOGDIR/$scene.log" 2>&1; then
    echo "[$i/$n_total] $scene OK"
    ok=$((ok+1))
  else
    echo "[$i/$n_total] $scene FAIL (see $LOGDIR/$scene.log)"
    fail=$((fail+1))
  fi
done < "$LIST"
echo "batch done: ok=$ok fail=$fail"
