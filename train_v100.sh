#!/usr/bin/env bash
# ============================================================================
# V100 六小时窗口 · 一键启动
#
#   bash train_v100.sh <phase2_cloud_package.zip>
#   bash train_v100.sh --data-dir=/data/synthetic_v3
#
# 干什么：装依赖 → 落数据 → 体检 → 标定 → 按预算排队 → 训练 → 冻结 test
#         评估 → 打回传包。能跑完几个臂就交付几个，跑不完的臂停在整 epoch
#         边界保持可续跑（回本机继续）。
#
# 可调环境变量：
#   BUDGET_HOURS=6      预算小时数
#   AMP_DTYPE=fp16      精度（V100 无原生 BF16，默认 fp16；32GB 卡可用 fp32）
#   ONLY=armA,armB      只跑指定臂（顺序即执行顺序）
#   NO_REF_ARM=1        不插入同精度参照臂
#   AUTO_START=0        只出计划不开跑
# ============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="$(dirname "$REPO")"
BUDGET_HOURS="${BUDGET_HOURS:-6}"
AMP_DTYPE="${AMP_DTYPE:-fp16}"
AUTO_START="${AUTO_START:-1}"
PY="${PY:-python3}"
DATA_DIR=""

case "${1:-}" in
  --data-dir=*) DATA_DIR="${1#--data-dir=}" ;;
  "")   echo "用法: bash train_v100.sh <phase2_cloud_package.zip | --data-dir=/path>" >&2; exit 1 ;;
  *)    PKG="$1" ;;
esac

say() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }

say "1/6 机器体检"
command -v nvidia-smi >/dev/null || { echo "!! 无 nvidia-smi，这台机器没有可用 GPU 驱动" >&2; exit 2; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
$PY -c "import sys; print('python', sys.version.split()[0])"
NPROC="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)"
echo "CPU 核心: $NPROC ; 磁盘可用: $(df -h "$WORK" | awk 'NR==2{print $4}')"

say "2/6 安装依赖（确保 wheel 覆盖 Volta sm_70）"
$PY -m pip install --upgrade pip -q
# V100 = sm_70。近期部分 CUDA 12.8 轮子已不含 sm_70，装错了要到第一个
# kernel 启动才报错，白烧租用时段——所以这里装完立刻验证 arch_list。
install_torch() {
  local idx="$1"
  echo "-> pip install torch torchvision --index-url $idx"
  $PY -m pip install -q --force-reinstall torch torchvision --index-url "$idx"
}
arch_ok() {
  $PY - <<'PY'
import sys
try:
    import torch
except Exception as e:
    print("torch import 失败:", e); sys.exit(1)
al = torch.cuda.get_arch_list()
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| arch_list", al)
if not torch.cuda.is_available():
    print("CUDA 不可用"); sys.exit(1)
cc = torch.cuda.get_device_capability(0)
tag = f"sm_{cc[0]}{cc[1]}"
if tag not in al:
    print(f"本卡 {tag} 不在 wheel 的 arch_list 中"); sys.exit(1)
# 真正跑一次 kernel，确认不是"能 import 但一算就崩"
x = torch.randn(512, 512, device="cuda", dtype=torch.float16)
y = (x @ x).float().sum().item()
assert y == y, "fp16 matmul 结果非有限"
print(f"kernel 自检通过（{tag}, fp16 matmul ok）")
PY
}
if ! $PY -c "import torch" 2>/dev/null || ! arch_ok; then
  install_torch "https://download.pytorch.org/whl/cu124" || true
  if ! arch_ok; then
    echo "-> cu124 不含本卡架构，回退 cu121"
    install_torch "https://download.pytorch.org/whl/cu121"
    arch_ok || { echo "!! 仍不可用，请手动安装匹配 sm_70 的 torch" >&2; exit 2; }
  fi
fi
$PY -m pip install -q -r "$REPO/requirements.txt"

say "3/6 准备数据"
if [[ -n "${PKG:-}" ]]; then
  mkdir -p "$WORK/_pkg" "$WORK/data"
  echo "解包 $PKG ..."
  $PY - "$PKG" "$WORK/_pkg" <<'PY'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as z:
    z.extractall(sys.argv[2])
print("解包完成")
PY
  if [[ -d "$WORK/_pkg/data/synthetic_v3" ]]; then
    rm -rf "$WORK/data/synthetic_v3"
    mv "$WORK/_pkg/data/synthetic_v3" "$WORK/data/synthetic_v3"
  fi
  DATA_DIR="$WORK/data/synthetic_v3"
fi
test -d "$DATA_DIR" || { echo "!! 数据目录不存在: $DATA_DIR" >&2; exit 2; }
echo "数据目录: $DATA_DIR （场景数 $(find "$DATA_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)）"

say "4/6 精度口径声明"
cat <<EOF
本轮精度: $AMP_DTYPE
  · V100 是 Volta(sm_70)，**没有原生 BF16**，而既有 R0 / F-N5-gray 基线是 bf16。
  · 实测 fp16 与 bf16 不等效（INC-0007：同种子 epoch1 验证损失差 2.04 倍），
    因此非 bf16 精度下会自动把「同精度参照臂 p2_t22_f_n5gray_$AMP_DTYPE」排在最前——
    它既给后续消融臂提供同口径基线，又直接量出精度本身的影响量。
  · 所有产物 run_id 自动带 _$AMP_DTYPE 后缀，不会覆盖既有 bf16 产物。
  · 32GB V100 可用 AMP_DTYPE=fp32（数值更接近 bf16 基线，但无张量核心、更慢）；
    16GB V100 装不下 fp32（bs8 实测峰值分配 15.35GB / 保留 17.76GB），预检会拦下。
EOF

say "5/6 预算规划"
EXTRA=()
[[ -n "${ONLY:-}" ]] && EXTRA+=(--only "$ONLY")
[[ "${NO_REF_ARM:-0}" == "1" ]] && EXTRA+=(--no-reference-arm)
cd "$REPO"
$PY run_arms.py --data_root "$DATA_DIR" --amp-dtype "$AMP_DTYPE" \
    --budget-hours "$BUDGET_HOURS" --plan-only "${EXTRA[@]}"

say "6/6 开跑"
if [[ "$AUTO_START" != "1" ]]; then
  echo "AUTO_START=0，仅出计划。手动开跑："
  echo "  cd $REPO && nohup $PY run_arms.py --data_root $DATA_DIR \\"
  echo "      --amp-dtype $AMP_DTYPE --budget-hours $BUDGET_HOURS > _v100_main_log.txt 2>&1 &"
  exit 0
fi
nohup $PY run_arms.py --data_root "$DATA_DIR" --amp-dtype "$AMP_DTYPE" \
      --budget-hours "$BUDGET_HOURS" "${EXTRA[@]}" > "$REPO/_v100_main_log.txt" 2>&1 &
PID=$!
echo "已后台启动，PID $PID"
cat <<EOF

看进度：
  tail -f $REPO/_v100_main_log.txt
  $PY $REPO/run_arms.py --status

结束后回传：
  $WORK/arms_return_package.zip（可用 --package-dir 改名）   （每臂 best_model + 评估 json/csv + 日志 + TB）

被抢占/断连后恢复：重跑同一条命令即可，训练从最新 epoch 续上。
EOF
