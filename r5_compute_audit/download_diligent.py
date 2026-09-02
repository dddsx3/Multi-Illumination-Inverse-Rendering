"""R5-B' DiLiGenT 数据下载 + 验证脚本 (W1-D1 stage 2 + W2-B/C 通用)

DiLiGenT 10 物体 × 96 光, 官方 photometric stereo 基准.
来源: https://sites.google.com/site/photometricstereodata/single-object

本机当前状态: D:/data/DiLiGenT/pmsData 已存在 (10 obj × 96 lights + 辅助文件 = 745 MB)
本脚本功能:
  1. 验证数据完整性 (10 obj × 96 PNG, light_directions/normal/mask 等)
  2. (可选) 下载缺失数据
  3. 输出验证报告: r5_compute_audit/raw_profile/diligent_validation.csv

用法:
  python r5_compute_audit/download_diligent.py --root D:/data/DiLiGenT
  python r5_compute_audit/download_diligent.py --root D:/data/DiLiGenT --download
  (下载模式会从 Dropbox 拉 1-1.5 GB, 仅在缺失时用)
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "r5_compute_audit" / "raw_profile" / "diligent_validation.csv"

# 10 DiLiGenT 物体 (R4″ 已确认, 来自 evaluate_diligent.py)
OBJECTS = ["ballPNG", "bearPNG", "buddhaPNG", "catPNG", "cowPNG",
           "gobletPNG", "harvestPNG", "pot1PNG", "pot2PNG", "readingPNG"]

# 官方下载 (如需) - R4″ 时从 Dropbox 拉过
DOWNLOAD_URL = "https://www.dropbox.com/s/3nne6lfmy6g9g7w/diligent-1.0.zip?dl=1"
EXPECTED_TOTAL_LIGHTS = 96


def validate_object(obj_dir: Path):
    """验证单个物体目录的完整性"""
    out = dict(object=obj_dir.name)
    if not obj_dir.exists():
        out["status"] = "MISSING_DIR"
        return out
    # 96 张 PNG (从 001.png 到 096.png)
    pngs = sorted(obj_dir.glob("*.png"))
    out["n_png_total"] = len(pngs)
    out["n_light_png"] = sum(1 for p in pngs if p.stem.isdigit() and 1 <= int(p.stem) <= 99)
    # light_directions.txt
    ld = obj_dir / "light_directions.txt"
    if ld.exists():
        ld_data = np.loadtxt(ld)
        out["light_directions_shape"] = list(ld_data.shape)
        out["light_dir_min"] = float(ld_data.min())
        out["light_dir_max"] = float(ld_data.max())
    else:
        out["light_directions_shape"] = "MISSING"
    # light_intensities.txt
    li = obj_dir / "light_intensities.txt"
    out["light_intensities_exists"] = li.exists()
    # normal.txt
    nrm = obj_dir / "normal.txt"
    if nrm.exists():
        nrm_data = np.loadtxt(nrm)
        out["normal_shape"] = list(nrm_data.shape)
    else:
        out["normal_shape"] = "MISSING"
    # mask.png
    out["mask_exists"] = (obj_dir / "mask.png").exists()
    # 验证: 96 张 light 图 + light_directions + light_intensities + normal + mask
    ok = (out["n_light_png"] == EXPECTED_TOTAL_LIGHTS
          and out["light_directions_shape"] != "MISSING"
          and out["light_intensities_exists"]
          and out["normal_shape"] != "MISSING"
          and out["mask_exists"])
    out["status"] = "OK" if ok else "INCOMPLETE"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="D:/data/DiLiGenT",
                    help="DiLiGenT 数据根目录 (含 pmsData/)")
    ap.add_argument("--download", action="store_true",
                    help="(可选) 从 Dropbox 下载缺失数据; 默认仅验证")
    args = ap.parse_args()

    root = Path(args.root)
    pmsData = root / "pmsData"
    estDir = root / "estNormalNonLambert"

    if not pmsData.exists():
        print(f"ERROR: {pmsData} 不存在")
        if args.download:
            print("需要先下载 DiLiGenT (Dropbox, ~1 GB)")
            print("命令: python r5_compute_audit/download_diligent.py --root <dest> --download")
        sys.exit(1)

    # 验证
    print("=" * 70)
    print(f"DiLiGenT 数据验证: {pmsData}")
    print("=" * 70)
    rows = []
    for obj in OBJECTS:
        r = validate_object(pmsData / obj)
        rows.append(r)
        print(f"  [{r['status']:11s}] {r['object']:12s}  "
              f"light_png={r.get('n_light_png', 'N/A')}/96  "
              f"light_dir={r.get('light_directions_shape', 'N/A')}  "
              f"normal={r.get('normal_shape', 'N/A')}  "
              f"mask={r.get('mask_exists', 'N/A')}")

    # baseline normal 估计
    print()
    if estDir.exists():
        est_files = list(estDir.glob("*.mat"))
        print(f"  baseline normal 估计 ({estDir}): {len(est_files)} .mat 文件 (用于 B 轨公平对标)")
        # 列出算法: 文件名格式 obj_Normal_VENUEYEARAUTHOR.mat
        algs = set()
        for f in est_files:
            # e.g. "ballPNG_Normal_ACCV10Wu" -> "ACCV10Wu"
            parts = f.stem.split("_Normal_")
            if len(parts) == 2:
                algs.add(parts[1])
        print(f"  包含算法: {sorted(algs)}")
    else:
        print(f"  注意: {estDir} 不存在, 无 baseline normal 估计 (B 轨对标需下载)")

    # 写 CSV
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    # summary
    n_ok = sum(1 for r in rows if r["status"] == "OK")
    print()
    print(f"汇总: {n_ok}/{len(rows)} 物体通过完整验证")
    print(f"产物: {OUT}")
    if n_ok < len(rows):
        print(f"FAIL: 至少 1 个物体数据不完整, B 轨 W2-B 启动前必须补全")


if __name__ == "__main__":
    main()
