#!/usr/bin/env python3
"""剩余 5 臂的 1-epoch 冒烟（D9：先冒烟后放量）。

逐臂验证：数据链路（含 rgb luma）、前向反向、阶段调度、checkpoint 落盘。
全量训练由 run_phase2_all.py 承担；本脚本只做门禁，产物入 p2_smoke_* 目录。
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATE = "20260825"
DATA_ROOT = r"D:/data/synthetic_v3"
MANIFEST = str(HERE / "splits" / "synthetic_v3.json")

SMOKES = [
    ("n5rgb",   ["--model", "fusion", "--modality", "rgb"]),
    ("physcon", ["--model", "fusion", "--modality", "gray",
                 "--sh_constraint", "softplus"]),
    ("resA",    ["--model", "fusion", "--modality", "gray",
                 "--residual_off"]),
    ("resC",    ["--model", "fusion", "--modality", "gray",
                 "--res_hidden", "32"]),
    ("albOff",  ["--model", "fusion", "--modality", "gray",
                 "--no_per_light_albedo"]),
]

results = {}
for name, extra in SMOKES:
    rid = f"p2_smoke_{name}_{DATE}"
    cmd = [sys.executable, "-u", "main.py", "--mode", "train",
           "--data_root", DATA_ROOT,
           "--total_epochs", "1", "--stage1_epochs", "1", "--stage2_epochs", "0",
           "--batch_size", "8", "--image_size", "256", "256",
           "--num_lights", "5", "--device", "cuda",
           "--use_amp", "--amp_dtype", "bf16",
           "--split_manifest", MANIFEST,
           "--run_id", rid,
           "--checkpoint_dir", f"../checkpoints/{rid}",
           "--log_dir", f"../logs/{rid}",
           "--viz_dir", f"../visualizations/{rid}",
           ] + extra
    print(f"\n===== SMOKE {name}: {' '.join(extra)}", flush=True)
    with open(f"_smoke_{name}_log.txt", "w", encoding="utf-8") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    ck_ok = any((Path("../checkpoints") / rid).glob("*.pth")) if rc == 0 else False
    txt = Path(f"_smoke_{name}_log.txt").read_text(encoding="utf-8", errors="replace")
    nan_hit = "nan" in txt.lower().replace("nan守卫", "")
    results[name] = {"rc": rc, "ckpt": ck_ok, "nan_suspect": nan_hit}
    verdict = "PASS" if (rc == 0 and ck_ok and not nan_hit) else "FAIL"
    print(f"===== SMOKE {name}: {verdict} (rc={rc}, ckpt={ck_ok})", flush=True)

print("\n===== SUMMARY =====")
for k, v in results.items():
    ok = v["rc"] == 0 and v["ckpt"] and not v["nan_suspect"]
    print(f"{k:8s} {'PASS' if ok else 'FAIL'} {v}")
sys.exit(0 if all(v["rc"] == 0 and v["ckpt"] and not v["nan_suspect"]
                  for v in results.values()) else 1)
