#!/usr/bin/env python3
"""run_ci · 统一实验入口（卡 C04 骨架）。

用法：python scripts/run_ci.py --experiment ci01 --config configs/ci01/pilot.yaml
流程：config 加载 → manifest（git/config/seed/dataset checksum）→ 实验 runner
（experiments/ci01_algebra.py 等）→ results/raw/ 落盘 → artifacts/frozen/ 摘要。
图表由 make_figures.py 从 artifacts/frozen/ 重建（宪法 §11）。
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from calibinfo.io.manifest import write_manifest  # noqa: E402

RUNNERS = {
    "ci01": "experiments.ci01_algebra",
    "ci02": "experiments.ci02_gauge",
    "ci03": "experiments.ci03_mc_validity",
    "ci04": "experiments.ci04_real_corruption",
    "ci05": "experiments.ci05_sanity_robustness",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, choices=sorted(RUNNERS))
    ap.add_argument("--config", required=True, help="configs/ci0X/*.yaml")
    ap.add_argument("--output-root", default=str(ROOT / "results" / "raw"))
    args = ap.parse_args()

    cfg_path = Path(args.config)
    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    run_dir = Path(args.output_root) / f"{args.experiment}_{config.get('run_name', 'run')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    write_manifest(run_dir / "manifest.json", config=config,
                   experiment=args.experiment, seed=config.get("seed"),
                   repo_root=str(ROOT))

    mod = importlib.import_module(RUNNERS[args.experiment])
    result = mod.run(config, run_dir)

    frozen = ROOT / "artifacts" / "frozen"
    frozen.mkdir(parents=True, exist_ok=True)
    summary_path = frozen / f"{args.experiment}_{config.get('run_name', 'run')}_summary.json"
    summary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[run_ci] {args.experiment} done -> {run_dir}")
    print(f"[run_ci] frozen summary -> {summary_path}")


if __name__ == "__main__":
    main()
