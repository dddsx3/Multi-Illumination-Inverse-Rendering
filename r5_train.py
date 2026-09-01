"""R5-B' 统一入口 — Cloud Studio IDE 点击 Run 即可启动

用法:
  1. 在 Cloud Studio web IDE 里打开本文件 (r5_train.py)
  2. 点击右上角的 Run (▶) 按钮
  3. 下方 TERMINAL 窗口显示实时进度
  4. 不用手动操作, 等6-7 小时跑完即可

也支持命令行:
  python r5_train.py                # 跑全量 (P1-A full + Task G)
  python r5_train.py --mode cell    # 跑1 个 cell (验证环境用, ~50 min)
  python r5_train.py --mode resume  # 接着跑 (跳过已完成)

要求: Ubuntu 22.04 + GPU (NVIDIA A10 24GB 推荐)
      Cloud Studio / 任何 Linux + GPU 实例皆可

本脚本自动处理:
  - venv 创建 + torch+cu128 安装
  - git-lfs 数据校验 (失败时直接给出修复命令)
  - 屏幕显示实时进度 (CSV 写入行数 + ETA)
  - 完成时自动 git commit + push
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import threading
from pathlib import Path

# ---- constants ----
REPO_ROOT = Path(__file__).resolve().parent
VENV_DIR = REPO_ROOT / ".venv_a10"
SCENES = [
    "conf_sphere_r05",
    "conf_cube_axis",
    "conf_prism8",
    "conf_egg",
    "conf_cylinder_r06_d06",
    "conf_ellipsoid_z06",
]
NS = [3, 5]
PIXEL_CAP = 2000
N3_LIMIT = 4960
N5_SAMPLE = 2000
DONE_MARKER = VENV_DIR / "DONE"
RAW_CSV = REPO_ROOT / "r5" / "r5_p1_albedo_ablation.csv"
RANK_CSV = REPO_ROOT / "r5" / "r5_p1_albedo_ablation_ranking.csv"
OUTLIER_CSV = REPO_ROOT / "r5" / "r5_p1_albedo_ablation_outliers.csv"
GATE_MD = REPO_ROOT / "r5" / "r5_p1_albedo_ablation_gate.md"
SELECTION_CSV = REPO_ROOT / "r5" / "r5_p1_albedo_ablation_selection.csv"
P1A_LOG = REPO_ROOT / "r5" / "r5_p1_a_full_run.log"
TASKG_CSV = REPO_ROOT / "r4pp" / "07_local_vs_global_init.csv"
TASKG_LOG = REPO_ROOT / "r4pp" / "task_g_run.log"


# ---- helpers ----
def log(msg: str) -> None:
    print(f"[r5_train] {msg}", flush=True)


def run(cmd: str | list, cwd: str | None = None, check: bool = True, env: dict | None = None) -> subprocess.CompletedProcess:
    """Run a command, streaming output."""
    if isinstance(cmd, str):
        cmd_str = cmd
    else:
        cmd_str = " ".join(cmd)
    log(f"$ {cmd_str}")
    result = subprocess.run(
        cmd,
        shell=isinstance(cmd, str),
        cwd=cwd,
        env=env or os.environ.copy(),
        check=check,
        capture_output=False,
        text=True,
    )
    return result


def get_venv_python() -> str:
    return str(VENV_DIR / "bin" / "python")


def in_venv() -> bool:
    return sys.prefix == str(VENV_DIR)


def venv_run(args: list, cwd: str | None = None) -> subprocess.CompletedProcess:
    """Run python inside the venv."""
    if in_venv():
        return run([sys.executable] + args, cwd=cwd)
    else:
        return run([get_venv_python()] + args, cwd=cwd)


# ---- step 1: env setup ----
def ensure_venv_and_torch() -> None:
    """Create venv if missing, install torch+cu128+numpy if missing."""
    log(f"venv dir = {VENV_DIR}")
    if not VENV_DIR.exists():
        log("creating venv...")
        run([sys.executable, "-m", "venv", str(VENV_DIR)])
    if in_venv():
        py = sys.executable
    else:
        py = get_venv_python()
    # check torch
    rc = subprocess.run(
        [py, "-c", "import torch; assert torch.cuda.is_available()"],
        capture_output=True, text=True,
    )
    if rc.returncode == 0:
        log("venv + torch + CUDA already OK")
        return
    log("installing torch + numpy + scipy + pandas...")
    if not in_venv():
        # need to pip install inside venv
        pip = str(VENV_DIR / "bin" / "pip")
    else:
        pip = "-m pip"
    run([pip, "install", "--quiet", "--upgrade", "pip"])
    run([pip, "install", "--quiet",
         "torch", "torchvision",
         "--index-url", "https://download.pytorch.org/whl/cu128"])
    run([pip, "install", "--quiet", "numpy", "scipy", "pandas"])
    # verify
    rc = subprocess.run(
        [py, "-c", "import torch; assert torch.cuda.is_available()"],
        capture_output=True, text=True,
    )
    if rc.returncode != 0:
        log("ERROR: torch install failed; rerun manually inside the venv.")
        sys.exit(1)


# ---- step 2: data sanity ----
def verify_lfs_data() -> None:
    """Verify that LFS data is real (not pointer files)."""
    test_file = REPO_ROOT / "p1/calibration_set/data_sun_confirmatory/conf_sphere_r05/albedo.npy"
    if not test_file.exists():
        log("ERROR: albedo.npy missing. LFS data not pulled.")
        log("  Fix:")
        log("    sudo apt-get install -y git-lfs   # package is 'git-lfs' (not 'git-lf')")
        log("    git lfs install")
        log("    git lfs pull")
        sys.exit(1)
    size = test_file.stat().st_size
    if size < 1000:
        log(f"ERROR: albedo.npy is {size} bytes (likely LFS pointer).")
        log("  Real data is ~52 KB. Run:")
        log("    git lfs pull")
        sys.exit(1)
    n_scenes = sum(
        1 for p in (REPO_ROOT / "p1/calibration_set/data_sun_confirmatory").iterdir()
        if p.name.startswith("conf_")
    )
    log(f"LFS data OK: {n_scenes} scenes, sample albedo.npy = {size} bytes")


# ---- step 3: git sanity ----
def ensure_git_clean_and_pulled() -> None:
    """git pull if behind, fail if dirty."""
    rc = subprocess.run(["git", "status", "--short"], cwd=str(REPO_ROOT), capture_output=True, text=True)
    if rc.stdout.strip():
        log(f"ERROR: working tree has uncommitted changes:\n{rc.stdout}")
        log("  Fix: commit or stash them, then re-run.")
        sys.exit(1)
    log("git status clean, pulling latest...")
    rc = subprocess.run(["git", "pull", "--rebase", "origin", "main"], cwd=str(REPO_ROOT))
    if rc.returncode != 0:
        log("WARNING: git pull failed (non-fatal; continuing with local)")


# ---- step 4: launch P1-A full in background ----
def launch_p1a_full() -> subprocess.Popen:
    """Launch P1-A full (GSIQ + solver arm) in background.

    Returns the Popen handle so caller can stream stdout/stderr.
    """
    scenes_csv = " ".join(SCENES)
    cmd = [
        get_venv_python(), "-u",
        str(REPO_ROOT / "p1/source/information_audit/r5_p1_albedo_ablation.py"),
        "--scenes", *SCENES,
        "--pixel_cap", str(PIXEL_CAP),
        "--n5_sample", str(N5_SAMPLE),
        "--n3_limit", str(N3_LIMIT),
        "--solver",
        "--n_top", "10",
        "--n_random", "10",
    ]
    log(f"launching P1-A full: {' '.join(cmd)}")
    log(f"  log: {P1A_LOG}")
    (REPO_ROOT / "r5").mkdir(parents=True, exist_ok=True)
    log_f = open(P1A_LOG, "ab", buffering=0)
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=log_f,
        stderr=subprocess.STDOUT,
        env={**os.environ,
             "OMP_NUM_THREADS": "10",
             "OPENBLAS_NUM_THREADS": "10",
             "MKL_NUM_THREADS": "10"},
    )
    return proc


# ---- step 5: launch Task G in background ----
def launch_taskg() -> subprocess.Popen:
    """Launch Task G (r4pp_local_vs_global.py) in background."""
    cmd = [
        get_venv_python(), "-u",
        str(REPO_ROOT / "p1/source/information_audit/r4pp_local_vs_global.py"),
    ]
    log(f"launching Task G: {' '.join(cmd)}")
    log(f"  log: {TASKG_LOG}")
    (REPO_ROOT / "r4pp").mkdir(parents=True, exist_ok=True)
    log_f = open(TASKG_LOG, "ab", buffering=0)
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=log_f,
        stderr=subprocess.STDOUT,
    )
    return proc


# ---- step 6: progress monitor ----
def monitor_progress(p1a: subprocess.Popen, taskg: subprocess.Popen | None,
                     stop_event: threading.Event) -> None:
    """Print progress every 30 s. Exit when stop_event is set."""
    log("entering monitor loop (30 s interval)...")
    start = time.time()
    p1a_total = len(SCENES) * len(NS) * 4960 * 2  # 6 scenes × 2 N × 4960 subsets × 2 score
    last_csv_lines = 0
    while not stop_event.is_set():
        elapsed = time.time() - start
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = int(elapsed % 60)
        # P1-A progress
        if RAW_CSV.exists():
            n_lines = sum(1 for _ in open(RAW_CSV, "rb")) - 1  # minus header
            pct = 100.0 * n_lines / max(p1a_total, 1)
            delta = n_lines - last_csv_lines
            last_csv_lines = n_lines
            rate = delta / 30.0 if delta > 0 else 0
            eta_min = (p1a_total - n_lines) / max(rate, 1e-3) / 60 if rate > 0 else float("nan")
            p1a_msg = (
                f"P1-A: {n_lines}/{p1a_total} rows ({pct:.1f}%) "
                f"rate={rate:.0f}/s ETA={eta_min:.0f}min"
            )
        else:
            p1a_msg = "P1-A: not started"
        # Task G progress
        if TASKG_CSV.exists():
            tg_rows = sum(1 for _ in open(TASKG_CSV, "rb")) - 1
            p1a_msg += f"  |  Task G: {tg_rows}/240"
        else:
            p1a_msg += "  |  Task G: not started"
        # process status
        p1a_alive = p1a.poll() is None
        taskg_alive = taskg.poll() is None if taskg else False
        status = []
        status.append("P1-A:alive" if p1a_alive else "P1-A:done")
        if taskg:
            status.append("TaskG:alive" if taskg_alive else "TaskG:done")
        p1a_msg += "  |  " + " ".join(status)
        log(f"[{h:02d}:{m:02d}:{s:02d}] {p1a_msg}")
        # exit if both done
        if not p1a_alive and (not taskg or not taskg_alive):
            log("both processes exited")
            break
        stop_event.wait(30.0)
    log("monitor loop exited")


# ---- step 7: verify + commit + push ----
def verify_outputs() -> bool:
    files = [RAW_CSV, RANK_CSV, OUTLIER_CSV, GATE_MD, SELECTION_CSV, TASKG_CSV]
    ok = True
    for f in files:
        if f.exists() and f.stat().st_size > 0:
            log(f"  [OK] {f.relative_to(REPO_ROOT)}  ({f.stat().st_size} bytes)")
        else:
            log(f"  [MISS] {f.relative_to(REPO_ROOT)}")
            ok = False
    return ok


def commit_and_push() -> None:
    """git add + commit + push the results."""
    log("git add results...")
    subprocess.run(
        ["git", "add",
         "r5/r5_p1_albedo_ablation.csv",
         "r5/r5_p1_albedo_ablation_ranking.csv",
         "r5/r5_p1_albedo_ablation_outliers.csv",
         "r5/r5_p1_albedo_ablation_gate.md",
         "r5/r5_p1_albedo_ablation_selection.csv",
         "r5/r5_p1_a_full_run.log",
         "r4pp/07_local_vs_global_init.csv",
         "r4pp/task_g_run.log"],
        cwd=str(REPO_ROOT), check=False,
    )
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
    )
    if not status.stdout.strip():
        log("nothing to commit (everything already pushed)")
        return
    log(status.stdout)
    log("git commit...")
    msg = "feat(r5-p1): A10 P1-A full + Task G results"
    rc = subprocess.run(["git", "commit", "-m", msg], cwd=str(REPO_ROOT))
    if rc.returncode != 0:
        log("git commit failed")
        return
    log("git push...")
    rc = subprocess.run(["git", "push", "origin", "HEAD"], cwd=str(REPO_ROOT))
    if rc.returncode != 0:
        log("git push failed (run 'git push origin HEAD' manually)")


# ---- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "cell", "resume"], default="full",
                    help="full = P1-A full + Task G (6.5h); "
                         "cell = 1 cell (sphere_r05 N=3, 50 min, sanity check); "
                         "resume = skip what's done, run rest")
    args = ap.parse_args()

    log(f"REPO_ROOT = {REPO_ROOT}")
    log(f"mode = {args.mode}")

    # 1. ensure venv + torch
    ensure_venv_and_torch()

    # 2. verify LFS data
    verify_lfs_data()

    # 3. git sanity (only for full / resume)
    if args.mode in ("full", "resume"):
        ensure_git_clean_and_pulled()

    # 4. decide what to run
    if args.mode == "cell":
        log("CELL mode: running only (conf_sphere_r05, N=3) for sanity")
        scenes = ["conf_sphere_r05"]
        ns = [3]
        cmd = [
            get_venv_python(), "-u",
            str(REPO_ROOT / "p1/source/information_audit/r5_p1_albedo_ablation.py"),
            "--scenes", *scenes,
            "--pixel_cap", str(PIXEL_CAP),
            "--n5_sample", "100",
            "--n3_limit", "4960",
        ]
        log(f"$ {' '.join(cmd)}")
        rc = subprocess.run(cmd, cwd=str(REPO_ROOT))
        log(f"cell done with rc={rc.returncode}")
        return

    # 5. launch both processes
    p1a = launch_p1a_full()
    time.sleep(2)  # let P1-A start first
    # Task G only if csv doesn't exist or has < 240 rows
    need_taskg = True
    if TASKG_CSV.exists():
        n = sum(1 for _ in open(TASKG_CSV, "rb")) - 1
        if n >= 240:
            log(f"Task G already done ({n} rows), skipping")
            need_taskg = False
    taskg = launch_taskg() if need_taskg else None

    # 6. monitor
    stop_event = threading.Event()
    monitor_progress(p1a, taskg, stop_event)

    # wait for processes
    if p1a.poll() is None:
        log("waiting for P1-A to finish...")
        p1a.wait()
    if taskg and taskg.poll() is None:
        log("waiting for Task G to finish...")
        taskg.wait()
    stop_event.set()

    # 7. verify outputs
    log("=== verifying outputs ===")
    if not verify_outputs():
        log("WARNING: some outputs missing")
    else:
        log("all outputs present ✓")

    # 8. parse gate verdict
    if GATE_MD.exists():
        log("=== P1-A gate verdict ===")
        for line in GATE_MD.read_text().splitlines():
            if "Gate verdict" in line or "median rho" in line:
                log(f"  {line.strip()}")

    # 9. commit + push
    log("=== git commit + push ===")
    commit_and_push()
    DONE_MARKER.parent.mkdir(parents=True, exist_ok=True)
    DONE_MARKER.write_text(f"DONE at {time.ctime()}")
    log(f"=== ALL DONE. Marker at {DONE_MARKER} ===")


if __name__ == "__main__":
    main()