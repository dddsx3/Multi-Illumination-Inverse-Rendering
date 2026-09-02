"""W1-D1: 合成图低层统计画像 (不需 DiLiGenT)

W1-D1 = "domain gap 假设生死" 检验的第一阶段
(任务书路线 B §B.2):
  量化域差的直接证据: 合成图 vs (未来下载的) DiLiGenT 图的
  梯度直方图 KL 散度、噪声功率谱、高光像素占比
  若三项 KL 都 < 0.1, 域差假设直接死亡

本脚本 (W1-D1 stage 1, 仅本机合成图) 算:
  - 梯度直方图 (梯度幅值 + 角度)
  - 高光像素占比 (RGB 接近 (1,1,1) 的像素)
  - 噪声功率谱 (2D FFT 径向平均)
  - 全图平均亮度 / 对比度

产物:
  r5_compute_audit/raw_profile/synth_low_level_stats.csv (per-scene)
  r5_compute_audit/decision_reports/W1D1_synth_stats.md
"""
from __future__ import annotations
import argparse, csv, os, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO / "p1" / "calibration_set" / "data_sun_confirmatory"
OUT = REPO / "r5_compute_audit" / "raw_profile" / "synth_low_level_stats.csv"
OUT_MD = REPO / "r5_compute_audit" / "decision_reports" / "W1D1_synth_stats.md"

SCENES = ["conf_sphere_r05", "conf_cube_axis", "conf_prism8", "conf_egg",
          "conf_cylinder_r06_d06", "conf_ellipsoid_z06"]


def gradient_hist(img, bins=64):
    """2D 梯度幅值 + 方向的直方图 (1D 联合)"""
    gx = np.diff(img.astype(np.float32), axis=1)
    gy = np.diff(img.astype(np.float32), axis=0)
    gx = gx[:-1, :]
    gy = gy[:, :-1]
    mag = np.sqrt(gx**2 + gy**2)
    ang = np.arctan2(gy, gx + 1e-12)
    H, _, _ = np.histogram2d(mag.ravel(), ang.ravel(),
                              bins=[bins, bins],
                              range=[[0, 1.0], [-np.pi, np.pi]])
    # 归一化为概率 + 熵
    p = H.flatten() / max(H.sum(), 1)
    p_pos = p[p > 0]
    ent = float(-np.sum(p_pos * np.log(p_pos + 1e-12)))
    return p, ent, float(mag.mean()), float(mag.std())


def highlight_fraction(img):
    """接近 (1,1,1) 的高光像素占比 (R,G,B 都 > 0.95)"""
    if img.ndim == 2:
        return 0.0
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    return float(((r > 0.95) & (g > 0.95) & (b > 0.95)).mean())


def noise_power_spectrum(img):
    """2D FFT 径向平均功率谱 (低频均值 / 高频均值 比 = 平滑度)"""
    gray = img.mean(axis=-1) if img.ndim == 3 else img
    F = np.fft.fft2(gray.astype(np.float32))
    P = np.abs(F) ** 2
    P = np.fft.fftshift(P)
    h, w = P.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    R = np.sqrt((Y - cy)**2 + (X - cx)**2)
    R_int = R.astype(int)
    max_r = min(cy, cx)
    radial = np.bincount(R_int.ravel(), weights=P.ravel(), minlength=max_r)
    counts = np.bincount(R_int.ravel(), minlength=max_r)
    radial = radial / np.maximum(counts, 1)
    low = float(radial[:max_r // 4].mean())
    high = float(radial[max_r // 4:].mean())
    return low, high, low / max(high, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=SCENES)
    ap.add_argument("--n_lights", type=int, default=5,
                    help="每个 scene 取前 N 盏灯合成图")
    args = ap.parse_args()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"W1-D1 stage 1: 合成图低层统计画像 ({len(args.scenes)} scene × {args.n_lights} light)")
    for scene in args.scenes:
        scene_dir = DATA_ROOT / scene
        sh = np.load(scene_dir / "sh_coeffs_irradiance.npy")[:args.n_lights]
        albedo = np.load(scene_dir / "albedo.npy")[0]  # [H,W,3] or [H,W]
        normal = np.load(scene_dir / "normal_mesh.npy").transpose(1, 2, 0)  # [H,W,3]
        for k in range(args.n_lights):
            c = sh[k]  # [9]
            Y = sph_harm_basis(normal)  # [H,W,9]
            L = (Y @ c[:9] * 4 * np.pi)  # Lambertian radiance
            img = L[..., None] * albedo  # [H,W,3]
            img = np.clip(img, 0, 1)
            # 统计
            p, ent, mag_mean, mag_std = gradient_hist(img.mean(-1) if img.ndim == 3 else img)
            hi = highlight_fraction(img)
            low, high, ratio = noise_power_spectrum(img)
            mean_lum = float(img.mean())
            std_lum = float(img.std())
            rows.append(dict(scene=scene, light_idx=k,
                              mean_lum=round(mean_lum, 4),
                              std_lum=round(std_lum, 4),
                              grad_mag_mean=round(mag_mean, 4),
                              grad_mag_std=round(mag_std, 4),
                              grad_entropy=round(ent, 3),
                              highlight_frac=round(hi, 4),
                              spec_low=round(low, 3),
                              spec_high=round(high, 3),
                              spec_smoothness=round(ratio, 3)))
            print(f"  [{scene} L{k}] mean_lum={mean_lum:.3f}  hi={hi:.3f}  "
                  f"grad_ent={ent:.2f}  spec_ratio={ratio:.2f}")

    # write csv
    with open(OUT, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    # summary
    avg_hi = np.mean([r["highlight_frac"] for r in rows])
    avg_ent = np.mean([r["grad_entropy"] for r in rows])
    avg_ratio = np.mean([r["spec_smoothness"] for r in rows])
    avg_mean_lum = np.mean([r["mean_lum"] for r in rows])

    md = []
    md.append("# W1-D1 Stage 1 · 合成图低层统计画像\n\n")
    md.append(f"## 数据规模\n- scene: {len(args.scenes)} × light: {args.n_lights} = {len(rows)} 张图\n\n")
    md.append("## 聚合 (跨 scene × light)\n\n")
    md.append(f"| 指标 | 均值 | 备注 |\n|---|---:|---|\n")
    md.append(f"| 平均亮度 | {avg_mean_lum:.3f} | 0=black, 1=white |\n")
    md.append(f"| 高光像素占比 (R,G,B 都 >0.95) | **{avg_hi:.4f}** | Lambertian 0; 含高光 0.001-0.01 |\n")
    md.append(f"| 梯度直方图熵 | {avg_ent:.2f} | 越高=越复杂 |\n")
    md.append(f"| 频谱平滑度 (低频/高频) | {avg_ratio:.2f} | 越高=越平滑 |\n")
    md.append("\n## 解读 (B 轨任务书 §B.2)\n\n")
    md.append("**KL 检验 (与 DiLiGenT 对比) 尚未做**: 本仓库无 DiLiGenT 原始图。\n")
    md.append("**stage 1 自身画像**:\n")
    if avg_hi < 1e-4:
        md.append(f"- 平均高光占比 {avg_hi:.4f} → 合成图**完全没有高光** (因为 Lambertian)\n")
        md.append("- 这意味着 A 轨 A-P1 的 albedo box constraint (ρ ≤ 1) **自动满足**\n")
        md.append("- **B 轨结论预期**: 加噪合成 vs DiLiGenT (含真实高光) 的 KL 散度会 > 0.1\n")
        md.append("  → **域差假设反而获支持** (合成太干净, 真实图有高光 = 域差来源)\n")
    elif avg_hi > 0.01:
        md.append(f"- 平均高光占比 {avg_hi:.4f} → 合成图有非平凡高光\n")
        md.append("- 需要查 albedo.npy 数据, 确认是否泄漏了 specular 分量\n")
    md.append("\n## 下一步\n\n")
    md.append("- **W1-D1 stage 2**: 下载 DiLiGenT 基准 (Calibrated photometric stereo, 10 objects, 96 lights each)\n")
    md.append("  URL: https://sites.google.com/site/photometricstereodata/single-object\n")
    md.append("  路径: `pre0/raw_data/diligent/`\n")
    md.append("- 跑同 4 个指标, 计算 KL(合成 || DiLiGenT)\n")
    md.append("- 三项 KL 全 < 0.1 → 域差假设**死**, 转查架构容量 (B→C)\n")
    md.append("- 任何 KL > 0.5 → 域差假设**获强支持**, 推进 B0 协议\n")
    OUT_MD.write_text("".join(md), encoding="utf-8")
    print(f"\n产出: {OUT}")
    print(f"      {OUT_MD}")


def sph_harm_basis(n):
    """简化 SH 基 (L=2, 9 维) - 复用 gauge_fisher_v2 的实现"""
    sys.path.insert(0, str(REPO / "p1" / "source" / "information_audit"))
    from gauge_fisher_v2 import sh_basis_npy
    H, W, _ = n.shape
    return sh_basis_npy(n.reshape(-1, 3)).reshape(H, W, 9)


if __name__ == "__main__":
    main()
