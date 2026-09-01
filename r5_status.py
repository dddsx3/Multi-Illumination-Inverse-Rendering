"""R5-B' 进度检查 — Cloud Studio IDE 点击 Run 看现状

用法:
  在 Cloud Studio 里打开 r5_status.py, 点击 ▶ Run
  显示 P1-A / Task G 当前进度 + 是否完成
"""
from __future__ import annotations
import os
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
RAW_CSV = REPO_ROOT / "r5" / "r5_p1_albedo_ablation.csv"
TASKG_CSV = REPO_ROOT / "r4pp" / "07_local_vs_global_init.csv"
P1A_LOG = REPO_ROOT / "r5" / "r5_p1_a_full_run.log"
TASKG_LOG = REPO_ROOT / "r4pp" / "task_g_run.log"
DONE_MARKER = REPO_ROOT / ".venv_a10" / "DONE"
GATE_MD = REPO_ROOT / "r5" / "r5_p1_albedo_ablation_gate.md"
SCENES = ["conf_sphere_r05","conf_cube_axis","conf_prism8","conf_egg",
          "conf_cylinder_r06_d06","conf_ellipsoid_z06"]
NS = [3, 5]
EXPECTED = len(SCENES) * len(NS) * 4960 * 2  # ~83520

def count_lines(p: Path) -> int:
    if not p.exists(): return 0
    return max(0, sum(1 for _ in open(p, "rb")) - 1)  # minus header

print(f"=== r5_status @ {time.ctime()} ===")
print()
# python procs
procs = subprocess.run(
    ["ps", "aux"], capture_output=True, text=True,
).stdout
p1a_procs = [l for l in procs.splitlines() if "r5_p1_albedo_ablation" in l and "grep" not in l]
tg_procs = [l for l in procs.splitlines() if "r4pp_local_vs_global" in l and "grep" not in l]
print(f"P1-A 进程: {'len' if p1a_procs else '死'}  ({len(p1a_procs)} 个)")
for l in p1a_procs[:3]: print(f"  {l.split()[-4:]}")  # CPU%, MEM%, etc
print(f"Task G 进程: {'alive' if tg_procs else '死'}  ({len(tg_procs)} 个)")
print()

# CSV counts
p1a_n = count_lines(RAW_CSV)
tg_n = count_lines(TASKG_CSV)
print(f"P1-A CSV: {p1a_n}/{EXPECTED} rows ({100*p1a_n/EXPECTED:.1f}%)")
if p1a_n > 0:
    rate_estimate = "~0.25 s/call (P=2000 + 28 vCPU)"
    eta_calls = EXPECTED - p1a_n
    eta_h = eta_calls * 0.25 / 3600
    print(f"  ETA: ~{eta_h:.1f} h (assuming {rate_estimate})")
print(f"Task G CSV: {tg_n}/240 rows")
print()

# log tails
if P1A_LOG.exists():
    print(f"P1-A log 末 3 行 ({P1A_LOG}):")
    with open(P1A_LOG, "rb") as f:
        lines = f.readlines()
    for l in lines[-3:]:
        try:
            print(f"  {l.decode(errors='replace').rstrip()}")
        except Exception:
            pass
print()
if TASKG_LOG.exists():
    print(f"Task G log 末 3 行 ({TASKG_LOG}):")
    with open(TASKG_LOG, "rb") as f:
        lines = f.readlines()
    for l in lines[-3:]:
        try:
            print(f"  {l.decode(errors='replace').rstrip()}")
        except Exception:
            pass
print()

# done marker
if DONE_MARKER.exists():
    print(f"✓ DONE marker exists: {DONE_MARKER.read_text().strip()}")
else:
    print("✗ no DONE marker yet (run not finished)")
print()

# gate verdict (if available)
if GATE_MD.exists():
    print("=== P1-A Gate Verdict ===")
    for line in GATE_MD.read_text().splitlines():
        if "Gate verdict" in line or "median rho" in line:
            print(f"  {line.strip()}")

print("=== next step ===")
if not p1a_procs and not tg_procs and not DONE_MARKER.exists():
    print("Both processes died; check logs and re-run r5_train.py")
elif DONE_MARKER.exists():
    print("Run complete. Read r5/r5_p1_albedo_ablation_gate.md and r4pp/task_g_run.log.")
else:
    print("In progress. Re-run this script to check again.")