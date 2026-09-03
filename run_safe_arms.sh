#!/usr/bin/env bash
# run_safe_arms.sh · 安全启动包装器（INC-0014 复盘后强制流程）
#
# 用法（在仓库根目录）:
#   ./run_safe_arms.sh --only A3-0_f_n5gray_seed42 --budget-hours 6
#
# 强制流程（每次训练启动都必须走这里，禁止裸跑 run_arms.py / main.py）:
#   1) source _env.sh（热墙/确定性环境）
#   2) 单实例 + 主机资源自检（runtime_safety.py）
#   3) 默认 bs4（FIX-06）；仅当用户显式 RUN_ARMS_BATCH=8 且被 BLOCK 时提示"需 ≥16GB 显存机器"；
#      bs4 仍 BLOCK → 终止
#   4) 后台起独立资源监控哨兵（monitor_host.py）
#   5) 前台执行 run_arms.py（其内部自带温度墙分段续跑 + rc42 恢复）
set -u
REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO" || exit 2

source _env.sh 2>/dev/null || echo "[safety] 未找到 _env.sh，跳过（热墙环境变量缺失请人工核对）"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PY="${SAFE_PY:-C:/Python314/python.exe}"
# FIX-06：默认 batch 4（本机 12GB 实测不可行 bs8，INC-0014）；仅外部 ≥16GB 机器显式覆盖。
BATCH="${RUN_ARMS_BATCH:-4}"

echo "===== [safety] 运行前自检（batch=${BATCH}） ====="
"$PY" runtime_safety.py --batch "$BATCH"
rc=$?
if [ "$rc" -eq 2 ]; then
    if [ "$BATCH" -ge 8 ]; then
        echo "===== [safety] bs${BATCH} 在本机被 BLOCK：bs8 需 ≥16GB 显存机器（INC-0014） ====="
    fi
    echo "===== [safety] 尝试以默认 bs4 启动 ====="
    "$PY" runtime_safety.py --batch 4
    if [ $? -eq 2 ]; then
        echo "===== [safety] bs4 仍 BLOCK：终止启动，请先关闭占用资源的程序 ====="
        exit 2
    fi
    export RUN_ARMS_BATCH=4
    echo "===== [safety] 已回退 RUN_ARMS_BATCH=4 ====="
else
    export RUN_ARMS_BATCH="$BATCH"
fi

LOG="docs/incidents/INC-0014_host_monitor.log"
: > "$LOG"
"$PY" monitor_host.py --interval 3 --log "$LOG" &
MONPID=$!
trap 'kill $MONPID 2>/dev/null' EXIT

echo "===== [safety] 启动 run_arms.py（监控哨兵日志 $LOG） ====="
"$PY" -u run_arms.py "$@"
R=$?
kill $MONPID 2>/dev/null
echo "===== run_arms 退出码 $R ====="
exit "$R"
