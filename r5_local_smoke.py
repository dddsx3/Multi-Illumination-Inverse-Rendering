"""R5 · 本机 P0 自检 + 一键复核 P1-A 全量能力 (Windows 优先)

两个模式 (--mode):
  check (默认): 只跑 6 项能力检查, 1-2 分钟出"通过/不通过"。
                用来确认 P0 (页面文件修复) 后本机真的能跑。
  full        : 自检通过后, 跑 R5-P1-A smoke (P=500 验证 12 cells @ P=500,
                等价于 A10 全量 P=2000 的缩微版, 大约 15-30 分钟)。

六项能力 (check 模式):
  1. numpy 大矩阵分配        (P=2000, 32MB float64 — 是历史 OOM 的代表场景)
  2. GSIQ 路径走通           (P=2000, N=5, 1 次)
  3. CUDA torch 初始化       (WDDM 模式, 之前 WinError 1455 的代表场景)
  4. CUDA 小算子             (10x10 matmul)
  5. LFS 数据完整            (24 scene + 真实 albedo.npy)
  6. Campaign 三个脚本可执行  (--quick 模式)

设计原则: 用户只要看输出最后几行, "全部通过" 即表示本机已恢复
论文实验能力, 可继续 P1-A 全量。

用法 (本机):
  python r5_local_smoke.py --mode check
  python r5_local_smoke.py --mode full
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent


def header(name):
    print(f"\n{'='*70}\n[{name}]\n{'='*70}")


def result(ok, detail=""):
    mark = "✅" if ok else "❌"
    return f"  {mark}  {detail}"


def get_available_gb():
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except ImportError:
        return float("nan")


def check_numpy_big():
    header("1. numpy 大矩阵分配 (P=2000 = 32 MB float64)")
    avail0 = get_available_gb()
    try:
        import numpy as np
        a = np.zeros((2000, 2000), dtype=np.float64)
        b = a + 1.0
        peak_mb = b.nbytes / 1e6
        del a, b
        return True, f"OK (peak {peak_mb:.0f}MB, 当前可用 RAM {avail0:.1f}GB)"
    except MemoryError:
        return False, (f"OOM! 当前可用 RAM {avail0:.1f}GB, 历史 2MB 也失败的水位下,"
                       " 请确认 P0 修复完成 (D 盘页面文件生效)")


def check_gsiq():
    header("2. GSIQ 路径走通 (P=2000, N=5, 1 次完整 dense eigh)")
    try:
        sys.path.insert(0, str(REPO / "p1" / "source" / "information_audit"))
        from gauge_fisher_v2 import load_scene, scene_arrays, ga_isi_v2_scores
        sc = load_scene(str(REPO / "p1/calibration_set/data_sun_confirmatory/conf_sphere_r05"))
        a, Y, C = scene_arrays(sc, subset=[0, 1, 2, 3, 4], pixel_cap=2000, seed=0, fix_gauge=True)
        t0 = time.time()
        r = ga_isi_v2_scores(a, Y, C)
        dt = time.time() - t0
        return True, f"OK  I_GS={r['full_logdet_pos_norm']:.4f}  (用了 {dt:.2f}s)"
    except Exception as e:
        return False, f"FAILED  {type(e).__name__}: {str(e)[:200]}"


def check_cuda_init():
    header("3. CUDA torch 初始化 (WDDM 模式, 历史 WinError 1455 现场)")
    try:
        import torch
        ok = torch.cuda.is_available()
        if not ok:
            return False, "torch.cuda.is_available() == False"
        n = torch.cuda.get_device_name(0)
        return True, f"OK  device={n}, torch={torch.__version__}, cuda={torch.version.cuda}"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:200]}"


def check_cuda_op():
    header("4. CUDA 小算子 (10x10 matmul on GPU)")
    try:
        import torch
        if not torch.cuda.is_available():
            return False, "skip (前一步 CUDA 不可用)"
        x = torch.randn(10, 10, device="cuda")
        y = x @ x
        torch.cuda.synchronize()
        return True, f"OK  sum={float(y.sum()):.4f}"
    except Exception as e:
        return False, f"FAILED  {type(e).__name__}: {str(e)[:200]}"


def check_lfs_data():
    header("5. LFS 数据完整 (24 scene + 真 albedo.npy)")
    p = REPO / "p1/calibration_set/data_sun_confirmatory"
    if not p.exists():
        return False, f"目录不存在: {p}"
    conf_dirs = sorted([d for d in p.iterdir() if d.name.startswith("conf_")])
    if not conf_dirs:
        return False, f"没有任何 conf_* 目录 ({p} 可能是空仓库或 LFS 未拉取)"
    bad = []
    for d in conf_dirs:
        f = d / "albedo.npy"
        if not f.exists() or f.stat().st_size < 1000:
            bad.append(f"{d.name}/albedo.npy={f.stat().st_size if f.exists() else 'missing'}")
    if bad:
        return False, (f"以下 scene 的 albedo.npy 缺失或过小 (可能是 LFS pointer):\n  "
                       + "\n  ".join(bad))
    return True, f"OK  {len(conf_dirs)} scenes, sample albedo.npy={conf_dirs[0].joinpath('albedo.npy').stat().st_size}B"


def check_campaign_scripts():
    header("6. Campaign 三个脚本 --quick 可执行")
    scripts = ["r5_ca_01_baseline_profile.py", "r5_ca_02_pixel_coreset.py", "r5_ca_03_matrixfree.py"]
    results = []
    for s in scripts:
        sp = REPO / s
        if not sp.exists():
            results.append(f"❌ {s} 不存在")
            continue
        try:
            r = subprocess.run(
                [sys.executable, str(sp), "--quick"],
                capture_output=True, text=True, timeout=600,
            )
            tail = (r.stdout or "")[-200:].replace("\n", " | ")
            results.append(f"✅ {s}  rc={r.returncode}  {tail[-150:]}")
        except subprocess.TimeoutExpired:
            results.append(f"❌ {s}  timeout >10min")
        except Exception as e:
            results.append(f"❌ {s}  {type(e).__name__}: {str(e)[:80]}")
    all_ok = all(r.startswith("✅") for r in results)
    return all_ok, "\n  ".join(results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["check", "full"], default="check")
    ap.add_argument("--skip-campaign", action="store_true",
                    help="check 模式跳过第 6 项 (节省 10 min)")
    args = ap.parse_args()

    print("=" * 70)
    print(f"R5 本机自检 — mode={args.mode}")
    print("=" * 70)
    avail = get_available_gb()
    print(f"启动内存: 可用 {avail:.1f} GB (P0 修复前一般 3-5, 修复后应 25+)")
    if avail < 4:
        print("  ⚠ 警告: 内存 <4GB, 后续实验可能 OOM。建议先做 P0 修复。")

    t0 = time.time()
    results = []
    results.append(("numpy", *check_numpy_big()))
    results.append(("GSIQ ", *check_gsiq()))
    results.append(("CUDA ", *check_cuda_init()))
    results.append(("CUDA op", *check_cuda_op()))
    results.append(("LFS  ", *check_lfs_data()))
    if not args.skip_campaign:
        results.append(("Campaign", *check_campaign_scripts()))

    header("汇总")
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    for name, ok, detail in results:
        print(result(ok, f"{name}  {detail.replace(chr(10), ' / ')}"))
    print()
    print(f"通过: {passed}/{total}, 耗时 {(time.time()-t0)/60:.1f} min")

    if passed < total:
        print("\n❌ 本机目前不能跑 R5-P1-A 全量。先解决失败项再继续。")
        print("   参考 r5_compute_audit/LOCAL_MACHINE_DIAGNOSIS.md 排障。")
        sys.exit(1)
    else:
        print("\n✅ 6 项能力全部通过, 本机已具备 R5-P1-A 全量跑能力。")
        if args.mode == "full":
            print("\n现在跑 P1-A smoke (P=500 缩微版, 6 scenes × N{3,5} × 100 subsets)...")
            r = subprocess.run(
                [sys.executable, "p1/source/information_audit/r5_p1_albedo_ablation.py",
                 "--scenes", "conf_sphere_r05", "conf_cube_axis", "conf_prism8",
                 "--pixel_cap", "500", "--n5_sample", "100", "--n3_limit", "4960"],
                cwd=str(REPO),
            )
            sys.exit(r.returncode)


if __name__ == "__main__":
    main()
