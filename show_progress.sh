#!/usr/bin/env bash
# 仓库根引导: 转发到 cloud_migration/show_progress.sh
# 用法(运行期间另开终端): watch -n 15 bash show_progress.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$HERE/cloud_migration/show_progress.sh" ]]; then
    exec bash "$HERE/cloud_migration/show_progress.sh"
else
    echo "[show_progress] 未找到 cloud_migration/show_progress.sh" >&2
    exit 1
fi