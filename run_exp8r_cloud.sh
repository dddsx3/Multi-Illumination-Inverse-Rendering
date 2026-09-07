#!/usr/bin/env bash
# ==============================================================================
# exp8R v3 · 仓库根引导脚本(clone 后在仓库根直接跑这一条)
#
# 用法(在仓库根目录):
#   bash run_exp8r_cloud.sh
#
# 说明: 真正的执行脚本在 cloud_migration/ 子目录, 本引导脚本自动转发
#       所有环境变量与参数。
# ==============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$HERE/cloud_migration/run_exp8r_cloud.sh" ]]; then
    exec bash "$HERE/cloud_migration/run_exp8r_cloud.sh" "$@"
else
    echo "[引导脚本][FATAL] 未找到 cloud_migration/run_exp8r_cloud.sh" >&2
    echo "  请确认你 clone 的是 main 分支且工作树完整(当前目录: $HERE)" >&2
    exit 1
fi
