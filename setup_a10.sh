#!/usr/bin/env bash
# A10 机器一键引导：装依赖 → 落数据 → 预检出计划 → 正式开跑
#
#   bash setup_a10.sh /path/to/phase2_cloud_package.zip [n5rgb_best.pth]
#   bash setup_a10.sh --data-dir /data/synthetic_v3     [n5rgb_ckpt.pth]
#
# 约定目录（与 run_arms.py 默认一致）：
#   <workdir>/repo/          本仓库
#   <workdir>/data/synthetic_v3/
#   <workdir>/checkpoints/   训练产物
#   <workdir>/logs/  <workdir>/visualizations/
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="$(dirname "$REPO")"
DATA_DIR=""
PKG=""
RESUME_CKPT="${2:-}"

if [[ "${1:-}" == "--data-dir" ]]; then
  echo "用法：bash setup_a10.sh --data-dir <目录> [ckpt]" >&2
  exit 1
elif [[ "${1:-}" == --data-dir=* ]]; then
  DATA_DIR="${1#--data-dir=}"
elif [[ -n "${1:-}" ]]; then
  PKG="$1"
else
  echo "用法：bash setup_a10.sh <phase2_cloud_package.zip | --data-dir=/path> [n5rgb_ckpt.pth]" >&2
  exit 1
fi

echo "== [1/5] 环境检查 =="
python3 -c "import sys; print('python', sys.version.split()[0])"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || {
  echo "!! nvidia-smi 不可用，确认这台机器有 GPU 且驱动正常" >&2; exit 2; }

echo "== [2/5] 安装依赖 =="
python3 -m pip install --upgrade pip -q
# 若镜像内已带匹配 CUDA 的 torch，此处不会重装；否则按官方源装 cu124 轮子
python3 -c "import torch" 2>/dev/null || \
  python3 -m pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu124
python3 -m pip install -q -r "$REPO/requirements.txt"
python3 - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda,
      "| device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE",
      "| bf16", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY

echo "== [3/5] 准备数据 =="
if [[ -n "$PKG" ]]; then
  mkdir -p "$WORK/_pkg"
  echo "解包 $PKG ..."
  python3 - "$PKG" "$WORK/_pkg" <<'PY'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as z:
    z.extractall(sys.argv[2])
print("解包完成")
PY
  mkdir -p "$WORK/data"
  if [[ -d "$WORK/_pkg/data/synthetic_v3" ]]; then
    rm -rf "$WORK/data/synthetic_v3"
    mv "$WORK/_pkg/data/synthetic_v3" "$WORK/data/synthetic_v3"
  fi
  DATA_DIR="$WORK/data/synthetic_v3"
fi
test -d "$DATA_DIR" || { echo "!! 数据目录不存在：$DATA_DIR" >&2; exit 2; }
echo "数据目录：$DATA_DIR（场景数 $(find "$DATA_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)）"

if [[ -n "$RESUME_CKPT" && -f "$RESUME_CKPT" ]]; then
  # n5rgb 臂本地已到 epoch 63：放到 epoch_0063 位置即可被 run_arms.py 识别续跑
  mkdir -p "$WORK/checkpoints/p2_t22_f_n5rgb"
  cp "$RESUME_CKPT" "$WORK/checkpoints/p2_t22_f_n5rgb/checkpoint_epoch_0063.pth"
  cp "$RESUME_CKPT" "$WORK/checkpoints/p2_t22_f_n5rgb/best_model.pth"
  echo "已植入 n5rgb 续跑点（epoch 63）"
fi

echo "== [4/5] 预检与预算规划（不占 GPU 长时）=="
cd "$REPO"
python3 run_arms.py --data_root "$DATA_DIR" --plan-only --budget-hours "${BUDGET_HOURS:-6}"

echo
echo "== [5/5] 正式开跑 =="
echo "确认上面的计划无误后执行（建议放 nohup/tmux 里，断连不影响）："
echo
echo "  cd $REPO && nohup python3 run_arms.py \\"
echo "      --data_root $DATA_DIR --budget-hours ${BUDGET_HOURS:-6} \\"
echo "      > _arms_main_log.txt 2>&1 &"
echo
echo "  # 看进度：  tail -f $REPO/_arms_main_log.txt"
echo "  #           python3 run_arms.py --status"
echo
if [[ "${AUTO_START:-0}" == "1" ]]; then
  echo "AUTO_START=1 → 直接开跑"
  nohup python3 run_arms.py --data_root "$DATA_DIR" \
        --budget-hours "${BUDGET_HOURS:-6}" > _arms_main_log.txt 2>&1 &
  echo "已后台启动，PID $!"
fi
