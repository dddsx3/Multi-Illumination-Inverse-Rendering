"""run_cloud · 云端启动器（P-LOWRANK-FULLRES 全分辨率认证表）。

用法（32 核 / 64 GB 默认机器）：
    python run_cloud.py                # 自动：workers = max(1, cores // 6)
    python run_cloud.py --workers 11   # 每对象一进程（BLAS 线程自动压到 2-3）

职责：
  1. 预设 OMP/MKL/OPENBLAS 线程数（大 P 下 >8 线程已 Amdahl 饱和；
     workers×OMP ≈ 物理核数为最优）；
  2. 断点续传：results/checkpoints_fullres/partial_<obj>.json 存在的对象
     直接跳过——中断后原命令重跑即续；
  3. 数据路径解析到本包内（./data/OpenIllumination）；
  4. 依赖自检（numpy/scipy/pyyaml/Pillow，缺失时提示 pip 安装）；
  5. 产出 results/lowrank_fullres.json + checkpoints/（完成后推回仓库）。

必须在 `python run_cloud.py` 下运行（Windows spawn 要求 main 守卫；
Linux 亦建议）。
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _cores() -> int:
    try:
        return mp.cpu_count()
    except Exception:
        return os.cpu_count() or 8


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=0,
                    help="0 = auto (cores // 6, at least 1)")
    ap.add_argument("--objects", default="",
                    help="comma list of object names; default = full cohort")
    ap.add_argument("--config", default=str(HERE / "configs" / "lowrank_fullres_cloud.yaml"))
    ap.add_argument("--out", default=str(HERE / "results" / "lowrank_fullres.json"))
    args = ap.parse_args()

    _check_deps()

    cores = _cores()
    workers = args.workers if args.workers > 0 else max(1, cores // 6)
    omp = max(1, cores // workers)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(omp)

    objects = [o.strip() for o in args.objects.split(",") if o.strip()]
    print(f"[run_cloud] cores={cores} workers={workers} omp_threads={omp} "
          f"objects={objects or 'full cohort'}")

    # 数据路径改写为包内相对路径（config 里的绝对路径只对本地有意义）
    import yaml
    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    data_root = HERE / "data" / "OpenIllumination"
    data_meta = HERE / "data" / "OpenIllumination_meta"
    if not (data_root / "OLAT").is_dir() or not (data_meta / "light_pos.npy").exists() \
            and not (data_root / "light_pos.npy").exists():
        print(f"[run_cloud] data missing under {data_root} — see README_CLOUD.md "
              f"(expected: OLAT/<obj>/Lights/NNN/com_masked_thumbnail/A1.png "
              f"for the 11 cohort objects + light_pos.npy)")
        sys.exit(3)
    cfg["data_root"] = str(data_root)
    cfg["data_meta"] = str(data_meta)
    cfg_path_patched = HERE / "configs" / "lowrank_fullres_cloud_patched.yaml"
    cfg_path_patched.write_text(yaml.safe_dump(cfg, sort_keys=False,
                                               allow_unicode=True),
                                encoding="utf-8")

    ckpt = HERE / "results" / "checkpoints_fullres"
    ckpt.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(HERE / "src"))
    from experiments.lowrank_fullres import run as fullres_run

    fullres_run(config_path=str(cfg_path_patched), out_path=args.out,
                objects=objects or None, workers=workers,
                checkpoint_dir=str(ckpt))
    print(f"[run_cloud] DONE. Push back: results/lowrank_fullres.json "
          f"+ results/checkpoints_fullres/")


if __name__ == "__main__":
    main()
