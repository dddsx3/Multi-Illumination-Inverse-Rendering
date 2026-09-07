#!/usr/bin/env bash
# reproduce_paper.sh · 一键复现（卡 C04 占位；C22 复现审计卡收口为完整链）。
# 目标：clean checkout + 本脚本 = 全部主 Figure/Table 重建（宪法 Gate E）。
set -euo pipefail
cd "$(dirname "$0")/.."
echo "[reproduce] TODO(卡C22): run_ci ci01..ci05 + make_figures 1-9 + make_tables 1-7"
echo "[reproduce] 当前可用：python scripts/run_ci.py --experiment ci01 --config configs/ci01/pilot.yaml"
