#!/usr/bin/env bash
# reproduce_paper.sh · C22 复现审计：clean checkout + 本脚本 = 主 Figure/Table 重建
# 依赖: python>=3.10, numpy, scipy, pyyaml, matplotlib, pillow; 数据见 STATE.md 环境表
set -euo pipefail
cd "$(dirname "$0")/.."
echo "[reproduce] 1/4 安装包"
pip install -e . --quiet
echo "[reproduce] 2/4 单元测试（V1-V6 红队回归 + 迁移绑定 + 诊断稠密对照）"
python -m pytest -q
echo "[reproduce] 3/4 CI01-CI05 正式 run（合成侧 CPU 全量；真实侧需 D:/data 见下）"
python scripts/run_ci.py --experiment ci01 --config configs/ci01/ci01_formal.yaml
python scripts/run_ci.py --experiment ci02 --config configs/ci02/ci02_formal.yaml
python scripts/run_ci.py --experiment ci03 --config configs/ci03/ci03_formal.yaml
python scripts/run_ci.py --experiment ci03nl --config configs/ci03/ci03nl_formal.yaml
if [ -d "D:/data/OpenIllumination" ]; then
  python scripts/run_ci.py --experiment ci04 --config configs/ci04/ci04_formal.yaml
else
  echo "[reproduce] skip ci04/ci05 (no local data; data contract = artifacts/frozen/*_manifest.json)"
fi
if [ -d "D:/data/DiLiGenT/pmsData" ]; then
  python scripts/run_ci.py --experiment ci05 --config configs/ci05/ci05_formal.yaml
  python scripts/run_ci.py --experiment ci05abl --config configs/ci05/ci05_ablation.yaml
fi
echo "[reproduce] 4/4 Figure 1-9 一键重建（provenance 随图入 paper/provenance/）"
for i in 1 2 3 4 5 6 7 8 9; do python scripts/make_figures.py --figure $i; done
echo "[reproduce] DONE — Gate E 素材齐"
