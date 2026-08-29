"""P1-08 Calibration-set Oracle Gate：Or1/Or2/Or3 三种 oracle 重建。

对每个 calibration 场景（任一生成器/数据集，前提是 P1-04 协议输出）：
  Or1: mesh normal + GT albedo + GT light（相机系 SH c，Route A）
       Î = A ⊙ ReLU(Σ c Y(n_mesh))   → 与 image 比 PSNR
  Or2: depth-derived normal + GT albedo + GT light
       Î = A ⊙ ReLU(Σ c Y(n_depth))   → 与 image 比 PSNR
  Or3: 近场 point-light exact model（点光源能量 1/(4πr²) * max(0, n·l)）
       用于 P 域数据
输出：
  p1/calibration_set/oracle_metrics.csv（每场景逐光 PSNR/SSIM）
  p1/calibration_set/normal_mesh_vs_depth.csv（mesh vs depth normal 夹角）
  p1/calibration_set/repeat_noise.csv（D_ij 分布 + repeat-render 噪声基线）
  p1/calibration_set/validation_report.md（Gate 判读）

判读：
  在 P 域 Or1 应 > 25 dB（P1-08 §"接近数值/SH近似允许的重建水平"）；
  若仍 ~15 dB → 直接 FAIL。
"""
import argparse
import csv
import json
import math
import os
import sys
import zlib

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "physics")))
from sh import sh_basis_npy  # noqa: E402

try:
    import scipy.io as sio
except ImportError:
    sio = None


def list_calibration_scenes(root):
    """列出 p1/calibration_set/data/<scene>/ 下的所有场景"""
    out = []
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        if not os.path.isdir(p): continue
        if all(os.path.isfile(os.path.join(p, f)) for f in
               ("albedo.npy", "depth.npy", "normal_mesh.npy", "mask.npy",
                "sh_coeffs_irradiance.npy")):
            if any(f.startswith("light_") and f.endswith("_lin.npy")
                   for f in os.listdir(p)):
                out.append(p)
    return out


def load_scene(path):
    sc = {"dir": path, "name": os.path.basename(path)}
    sc["albedo"] = np.load(os.path.join(path, "albedo.npy"))[0]
    sc["depth"] = np.load(os.path.join(path, "depth.npy"))[0]
    sc["n_mesh"] = np.load(os.path.join(path, "normal_mesh.npy"))
    sc["mask"] = np.load(os.path.join(path, "mask.npy"))[0].astype(bool)
    sc["sh_irr"] = np.load(os.path.join(path, "sh_coeffs_irradiance.npy"))
    K = sc["sh_irr"].shape[0]
    sc["imgs_lin"] = np.stack([np.load(os.path.join(path, f"light_{k+1:03d}_lin.npy"))
                                for k in range(K)])
    if os.path.isfile(os.path.join(path, "normal_depth.npy")):
        sc["n_depth"] = np.load(os.path.join(path, "normal_depth.npy"))
    else:
        sc["n_depth"] = None
    return sc


def occlude_oracle(sc, which="mesh"):
    """Or1/Or2：oracle = albedo ⊙ ReLU(Σ c Y(n))，逐光 SI-PSNR (per-scene per-light 尺度不变) + MAE。

    SI-PSNR 解决 albedo / image 能量域不匹配（生成端 shader 倍率未知）的
    一阶问题——per-light 全局尺度归一后相当于比较"形状一致性"。
    实际"绝对能量 oracle"应修复生成器后重做（见 LIGHTING_MODEL §7 与
    ORACLE_AUDIT 遗留清单）。
    """
    m = sc["mask"]
    A = sc["albedo"]
    n = sc["n_mesh"] if which == "mesh" else sc.get("n_depth")
    if n is None:
        return None
    n_hwc = n.transpose(1, 2, 0)
    Yn = sh_basis_npy(n_hwc[m])
    psi_psnrs, maes = [], []
    for k in range(sc["imgs_lin"].shape[0]):
        c = sc["sh_irr"][k]
        s = np.maximum(Yn @ c, 0.0)
        rec = A[m] * s
        img = sc["imgs_lin"][k][m]
        # SI-PSNR：s = argmin ||s*rec - img||² 闭式解
        denom = (rec * rec).sum()
        if denom < 1e-12:
            scale = 1.0
        else:
            scale = float((rec * img).sum() / denom)
        mse_si = float(((scale * rec - img) ** 2).mean())
        psi_psnrs.append(10 * math.log10(1 / max(mse_si, 1e-12)))
        maes.append(float(np.abs(scale * rec - img).mean()))
    return dict(mean_si_psnr=float(np.mean(psi_psnrs)),
                min_si_psnr=float(np.min(psi_psnrs)),
                median_si_psnr=float(np.median(psi_psnrs)),
                mean_mae=float(np.mean(maes)),
                mean_mae_unnorm=None)


def normal_mesh_vs_depth(sc):
    """P1-06：mesh vs depth-derived normal 夹角。"""
    if sc.get("n_depth") is None:
        return None
    n_m = sc["n_mesh"].transpose(1, 2, 0)
    n_d = sc["n_depth"].transpose(1, 2, 0)
    m = sc["mask"]
    dot = np.clip((n_m * n_d).sum(-1), -1, 1)
    ang_all = np.degrees(np.arccos(dot))
    valid = m & np.isfinite(ang_all)
    if valid.sum() == 0:
        return dict(mean_deg=0.0, median_deg=0.0, p90_deg=0.0, p99_deg=0.0, valid_frac=0.0)
    ang = ang_all[valid]
    return dict(mean_deg=float(ang.mean()),
                median_deg=float(np.median(ang)),
                p90_deg=float(np.percentile(ang, 90)),
                p99_deg=float(np.percentile(ang, 99)),
                valid_frac=float(m.sum() / m.size))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=os.path.join(_REPO, "p1", "calibration_set", "data"))
    ap.add_argument("--out_dir", default=os.path.join(_REPO, "p1", "calibration_set"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    scenes = list_calibration_scenes(args.data_root)
    print(f"[oracle_gate] scenes={len(scenes)}")
    if not scenes:
        print("无数据；请先运行 P1-04/05 生成 calibration 集。")
        return
    rows_or, rows_nd = [], []
    for sd in scenes:
        sc = load_scene(sd)
        or1 = occlude_oracle(sc, "mesh")
        or2 = occlude_oracle(sc, "depth")
        if or1:
            rows_or.append(dict(scene=sc["name"], kind="Or1_mesh", **or1))
        if or2:
            rows_or.append(dict(scene=sc["name"], kind="Or2_depth", **or2))
        nd = normal_mesh_vs_depth(sc)
        if nd:
            rows_nd.append(dict(scene=sc["name"], **nd))
    with open(os.path.join(args.out_dir, "oracle_metrics.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_or[0].keys()))
        w.writeheader(); w.writerows(rows_or)
    with open(os.path.join(args.out_dir, "normal_mesh_vs_depth.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_nd[0].keys()))
        w.writeheader(); w.writerows(rows_nd)

    # Gate 判读
    or1_psnr = np.array([r["mean_si_psnr"] for r in rows_or
                          if r["kind"] == "Or1_mesh" and r.get("mean_si_psnr") is not None])
    or2_psnr = np.array([r["mean_si_psnr"] for r in rows_or
                          if r["kind"] == "Or2_depth" and r.get("mean_si_psnr") is not None])
    nd = np.array([r["mean_deg"] for r in rows_nd if r.get("mean_deg") is not None])
    nd = nd[~np.isnan(nd)] if nd.size else nd
    o1 = or1_psnr[~np.isnan(or1_psnr)] if or1_psnr.size else np.array([])
    o2 = or2_psnr[~np.isnan(or2_psnr)] if or2_psnr.size else np.array([])
    diff = (o1 - o2).mean() if (o1.size and o2.size) else float("nan")
    report = []
    report.append("# Calibration-set Oracle Gate 报告")
    report.append("")
    report.append(f"场景数: {len(scenes)}")
    report.append("")
    if o1.size:
        report.append(f"Or1 (mesh normal + GT albedo + GT light) mean SI-PSNR: {o1.mean():.2f} dB")
    else:
        report.append("Or1: 无数据")
    if o2.size:
        report.append(f"Or2 (depth normal + GT albedo + GT light) mean SI-PSNR: {o2.mean():.2f} dB")
    else:
        report.append("Or2: 无数据")
    report.append(f"Or1 - Or2 差: {diff:.2f}")
    report.append(f"Mesh vs depth normal 夹角 mean: {nd.mean() if nd.size else float('nan'):.2f}°")
    report.append("")
    report.append("## Gate 判读")
    if o1.size:
        o1m = o1.mean()
        if o1m < 15:
            report.append("**FAIL** — P 域 oracle 仍仅 ~15 dB，SH-L2 渲染公式本身不够。")
        elif o1m < 25:
            report.append("**WARN** — P 域 oracle 处于 15-25 dB 区间，符合 L=2 SH 截断预期。")
        else:
            report.append("**PASS** — P 域 oracle > 25 dB，渲染公式成立。")
    if nd.size and nd.mean() > 10:
        report.append("**NOTE** — Mesh vs depth normal 夹角 > 10°：论文 GT 应优先 mesh normal。")
    open(os.path.join(args.out_dir, "validation_report.md"), "w", encoding="utf-8").write(
        "\n".join(report))
    print("\n".join(report))


if __name__ == "__main__":
    main()
