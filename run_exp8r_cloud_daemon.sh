#!/usr/bin/env bash
# ==============================================================================
# run_exp8r_cloud_daemon.sh · 会话无关启动(防 SSH/会话掉线杀死启动器)
#
# 用法:
#   bash run_exp8r_cloud_daemon.sh
# 返回前会打印启动器 PID; 主日志: cloud_migration/launcher_daemon.log
# (断线/关窗都不影响; 物体级/子集级断点续跑照常)
# ==============================================================================
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f "$HERE/cloud_migration/run_exp8r_cloud.sh" ]]; then
    echo "[daemon][FATAL] 未找到 cloud_migration/run_exp8r_cloud.sh" >&2
    exit 1
fi
LOG="$HERE/cloud_migration/launcher_daemon.log"
nohup bash "$HERE/cloud_migration/run_exp8r_cloud.sh" > "$LOG" 2>&1 < /dev/null &
LPID=$!
disown 2>/dev/null || true
echo "[daemon] 启动器已后台化 PID=$LPID (会话掉线不影响)"
echo "[daemon] 主日志: cloud_migration/launcher_daemon.log"
echo "[daemon] 进度: watch -n 15 bash $HERE/cloud_migration/show_progress.sh"