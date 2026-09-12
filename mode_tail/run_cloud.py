"""run_cloud · 云端一键运行器（P-ALLOC2 v1.1 三臂 + P-RADIUS + P-CONC + P-CERT）。

用法（32 核 / 64 GB 默认机器）：
    python run_cloud.py --stage all         # 依序跑全部四个实验
    python run_cloud.py --stage alloc2      # 仅 P-ALLOC2 三臂
    python run_cloud.py --stage radius      # 仅 P-RADIUS 线性化半径
    python run_cloud.py --stage conc        # 仅 P-CONC 证书集中度
    python run_cloud.py --stage cert        # 仅 P-CERT 认证间隙表

断点续传：alloc2 逐对象无耦合，中断后原命令重跑即续。
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "app" / "src"
EXPERIMENTS = HERE / "app" / "experiments"
CONFIGS = HERE / "app" / "configs"
RESULTS = HERE / "results"


def _cores() -> int:
    return mp.cpu_count()


def _check_deps() -> None:
    missing = []
    for mod in ("numpy", "scipy", "yaml", "PIL"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print(f"[run_cloud] missing deps {missing}; run:\n"
              f"  pip install numpy scipy pyyaml pillow")
        sys.exit(2)


def _patch_config(cfg_name: str) -> Path:
    """将 config 中的 data_root/data_meta 改写为包内相对路径。"""
    import yaml
    src = CONFIGS / cfg_name
    cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
    cfg["data_root"] = str(HERE / "data" / "OpenIllumination")
    cfg["data_meta"] = str(HERE / "data" / "OpenIllumination_meta")
    patched = CONFIGS / f"_{cfg_name}"
    patched.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
                       encoding="utf-8")
    return patched


def _run_experiment(name: str, script: str, cfg_name: str, out_rel: str,
                    extra_args: list[str] | None = None):
    print(f"\n{'='*60}")
    print(f"  RUNNING: {name}")
    print(f"{'='*60}\n")
    patched_cfg = _patch_config(cfg_name)
    out = RESULTS / out_rel
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(EXPERIMENTS / script),
           "--config", str(patched_cfg), "--out", str(out)]
    if extra_args:
        cmd.extend(extra_args)
    env = os.environ.copy()
    result = subprocess.run(cmd, env=env, cwd=str(HERE))
    if result.returncode != 0:
        print(f"[run_cloud] WARNING: {name} exited with {result.returncode}")
    else:
        print(f"[run_cloud] {name} DONE -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "alloc2", "radius", "conc", "cert"])
    ap.add_argument("--workers", type=int, default=0,
                    help="0 = auto (min(11, cores//2)) for alloc2")
    args = ap.parse_args()

    _check_deps()
    cores = _cores()
    workers = args.workers if args.workers > 0 else min(11, max(1, cores // 2))
    omp = max(1, cores // max(workers, 1))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(omp)
    print(f"[run_cloud] cores={cores} workers={workers} omp={omp}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(SRC))
    sys.path.insert(0, str(EXPERIMENTS))
    os.environ["PYTHONPATH"] = f"{SRC};{EXPERIMENTS};{HERE}"

    stages = {
        "alloc2": ("allocation_mode_tail.py", "allocation_mode_tail.yaml",
                   "mode_tail/allocation_mode_tail.json"),
        "radius": ("linearization_radius.py", "linearization_radius.yaml",
                   "magnitude/linearization_radius.json"),
        "conc": ("certificate_concentration.py", "certificate_concentration.yaml",
                 "certification/certificate_concentration.json"),
        "cert": ("lowrank_fullres.py", "lowrank_fullres_cloud.yaml",
                 "certification/lowrank_fullres.json"),
    }

    if args.stage in ("all", "alloc2"):
        _run_experiment("P-ALLOC2 v1.1 three-arm", *stages["alloc2"][:2],
                        out_rel=stages["alloc2"][2],
                        extra_args=["--workers", str(workers)])
    if args.stage in ("all", "radius"):
        _run_experiment("P-RADIUS", *stages["radius"][:2],
                        out_rel=stages["radius"][2])
    if args.stage in ("all", "conc"):
        _run_experiment("P-CONC", *stages["conc"][:2],
                        out_rel=stages["conc"][2])
    if args.stage in ("all", "cert"):
        _run_experiment("P-CERT full-res", *stages["cert"][:2],
                        out_rel=stages["cert"][2])

    print(f"\n[run_cloud] ALL DONE. Results under {RESULTS}/")
    print(f"[run_cloud] Push back: results/ + checkpoints/ to the main repo.")


if __name__ == "__main__":
    main()
