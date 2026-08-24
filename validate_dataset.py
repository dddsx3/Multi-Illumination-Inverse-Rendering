"""
Phase 1 合成数据集协议校验脚本（验收门禁执行器）

用法：
  python validate_dataset.py --root <data_root> [--sample 50] [--out_dir report]
  （在仓库目录下运行；需要 torch、numpy、PIL、matplotlib）

校验项与门禁（Phase 1 验收标准）：
  C1 完整性   每场景 10 个文件齐、尺寸匹配（light_001..K.png + 5 个 npy）
  C2 数值范围 depth>0 且非无穷；albedo∈[0,1]；normal 单位范数(误差<1e-3)；
              mask∈{0,1}
  C3 掩码覆盖 前景占比 ∈ [5%, 95%]（过小/过大说明相机距离或归一化问题）
  C4 法线自洽 normal 与深度导数法线（复刻 physics_renderer 约定）平均夹角 < 10°，
              且方向一致性（同侧占比 > 90%）——抓解码/方向错误
  C5 光照自洽 用 GT depth/albedo/sh_coeffs 走 PhysicsRenderer（torch CPU）
              重渲染，与 PNG 解码图（1/2.2 反 gamma）对比 PSNR > 12dB
              ——二阶 SH 对点光源的固有重建误差上限；若 < 5dB 说明
              gamma 链或 SH 量级/方向错误（Phase 1 最重要的一道门禁）
  C6 场景统计 各场景文件数、渲染统计（depth 范围、掩码覆盖、法线夹角），
              输出汇总 JSON + CSV + 抽样拼图（normal 编码可视化、depth
              jet 可视化），供人眼抽查与论文素材
  C7 判定     全部门禁通过 -> PASS；否则 FAIL 并列出违规场景与原因

抽样拼图输出：report/montage_<n>.png（每行一个场景：light_001 | albedo |
normal_rgb | depth_jet | mask），默认随机 50 场景。
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from physics_renderer import PhysicsRenderer  # noqa: E402
from evaluate import recon_metrics  # noqa: E402

FILES = ["light_001.png", "light_002.png", "light_003.png",
         "light_004.png", "light_005.png",
         "depth.npy", "albedo.npy", "normal.npy", "mask.npy", "sh_coeffs.npy"]


def sobel_normal(depth):
    """numpy 复刻 physics_renderer.DepthToNormal（use_edge_aware=False）"""
    h, w = depth.shape
    pad = np.pad(depth, 1, mode="edge")
    gx = (-pad[:-2, :-2] + pad[2:, :-2] - 2 * pad[:-2, 1:-1] + 2 * pad[2:, 1:-1]
          - pad[:-2, 2:] + pad[2:, 2:]) / 4.0
    gy = (pad[:-2, :-2] + 2 * pad[1:-1, :-2] + pad[2:, :-2]
          - pad[:-2, 2:] - 2 * pad[1:-1, 2:] - pad[2:, 2:]) / 4.0
    n = np.stack([-gx, -gy, np.ones_like(gx)], axis=-1)
    return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-8)


def depth_jet(depth, mask):
    """深度 jet 可视化（min-max 归一化到前景）"""
    d = depth.copy()
    fg = mask > 0
    lo, hi = np.percentile(d[fg], 1), np.percentile(d[fg], 99)
    d = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    import matplotlib.cm as cm
    rgba = (cm.jet(d)[:, :, :3] * 255).astype(np.uint8)
    rgba[~fg] = [20, 20, 20]
    return rgba


def validate_scene(scene_dir):
    """校验单场景，返回 (问题列表, 统计 dict)"""
    issues, stat = [], {}
    for f in FILES:
        if not os.path.isfile(os.path.join(scene_dir, f)):
            issues.append(f"缺少文件 {f}")

    try:
        depth = np.load(os.path.join(scene_dir, "depth.npy"))
        albedo = np.load(os.path.join(scene_dir, "albedo.npy"))
        normal = np.load(os.path.join(scene_dir, "normal.npy"))
        mask = np.load(os.path.join(scene_dir, "mask.npy"))
        sh = np.load(os.path.join(scene_dir, "sh_coeffs.npy"))
    except Exception as e:
        return [f"npy 加载失败: {e}"], stat

    if depth.ndim != 3 or depth.shape[0] != 1:
        issues.append(f"depth 形状 {depth.shape}（期望 [1,H,W]）")
    if albedo.ndim != 3 or albedo.shape[0] != 1:
        issues.append(f"albedo 形状 {albedo.shape}（期望 [1,H,W]）")
    if normal.ndim != 3 or normal.shape[0] != 3:
        issues.append(f"normal 形状 {normal.shape}（期望 [3,H,W]）")
    if sh.shape[0] != 5 or sh.shape[1] != 9:
        issues.append(f"sh_coeffs 形状 {sh.shape}（期望 [5,9]）")

    H = W = None
    for s in (depth, albedo, mask):
        if s.ndim == 3 and s.shape[0] == 1:
            H, W = s.shape[1], s.shape[2]
    if H is None:
        return issues, stat

    m = mask[0].astype(bool)
    if H * W == 0 or not m.any():
        issues.append("mask 无前景像素")
        return issues, stat

    # C2 数值范围
    if not np.isfinite(depth).all() or depth[0][m].min() <= 0:
        issues.append(f"depth 非法值（min={depth[0][m].min():.4f}）")
    if albedo[0][m].min() < -1e-3 or albedo[0][m].max() > 1 + 1e-3:
        issues.append(f"albedo 越界 [{albedo[0][m].min():.4f}, {albedo[0][m].max():.4f}]")
    n = normal[:, m]
    n_norm = np.linalg.norm(n, axis=0)
    if np.abs(n_norm - 1).max() > 1e-3:
        issues.append(f"normal 非单位范数（max偏差 {np.abs(n_norm - 1).max():.4f}）")
    if not set(np.unique(mask)).issubset({0, 1}):
        issues.append("mask 含非 0/1 值")

    # C3 掩码覆盖
    coverage = m.mean()
    stat["coverage"] = float(coverage)
    if not (0.05 <= coverage <= 0.95):
        issues.append(f"掩码覆盖 {coverage:.3f} 不在 [0.05, 0.95]")

    # C4 法线自洽
    n_sobel = sobel_normal(depth[0])
    dot = np.clip((normal[:, m].T * n_sobel[m]).sum(axis=1), -1, 1)
    angle = np.degrees(np.arccos(dot))
    mean_angle = float(angle.mean())
    same_side = float((angle < 90).mean())
    stat["normal_sobel_angle"] = mean_angle
    if mean_angle > 10.0:
        issues.append(f"法线与深度导数平均夹角 {mean_angle:.1f}° > 10°")
    if same_side < 0.9:
        issues.append(f"法线方向不一致（同侧占比 {same_side:.3f}）")

    # C5 光照自洽：GT 重渲染 vs PNG 解码图
    try:
        renderer = PhysicsRenderer()
        with torch.no_grad():
            pred, _, _ = renderer(torch.from_numpy(depth[None]), torch.from_numpy(albedo[None]),
                                  torch.from_numpy(sh[None]))
            pred = pred.numpy()[0]  # [K,H,W]
        psnrs = []
        for k in range(5):
            img = np.asarray(Image.open(os.path.join(scene_dir, f"light_{k + 1:03d}.png")),
                             dtype=np.float32) / 255.0
            dec = np.power(np.clip(img, 0, 1), 2.2)          # 与 data_loader 相同解码
            m2 = recon_metrics(torch.from_numpy(dec[None, None]), torch.from_numpy(pred[k:k + 1]))
            psnrs.append(m2["psnr"])
        stat["rerender_psnr"] = float(np.mean(psnrs))
        if np.mean(psnrs) < 5.0:
            issues.append(f"重渲染 PSNR {np.mean(psnrs):.1f}dB 过低（gamma/SH 错误）")
        elif np.mean(psnrs) < 12.0:
            issues.append(f"重渲染 PSNR {np.mean(psnrs):.1f}dB 低于 12dB（SH 量级/方向需检查）")
    except Exception as e:
        issues.append(f"C5 重渲染失败: {e}")

    # 统计（供报告）
    stat["depth_min"], stat["depth_max"] = float(depth[0][m].min()), float(depth[0][m].max())
    stat["albedo_mean"] = float(albedo[0][m].mean())
    stat["sh_absmax"] = float(np.abs(sh).max())
    stat["size"] = [H, W]
    return issues, stat


def make_montage(scene_dirs, out_path, thumb=128):
    """抽样拼图：每场景一行 5 格（light_001 | albedo | normal_rgb | depth_jet | mask）

    文件缺失/损坏的场景跳过——FAIL 数据集同样要能产出报告与拼图。
    """
    rows = []
    for sd in scene_dirs:
        try:
            cells = []
            img = Image.open(os.path.join(sd, "light_001.png")).resize((thumb, thumb))
            cells.append(np.asarray(img))
            albedo = np.load(os.path.join(sd, "albedo.npy"))[0]
            a8 = np.clip(albedo * 255, 0, 255).astype(np.uint8)
            cells.append(np.stack([a8] * 3, axis=-1))
            normal = np.load(os.path.join(sd, "normal.npy")).transpose(1, 2, 0)
            nrgb = ((normal * 0.5 + 0.5) * 255).astype(np.uint8)
            cells.append(np.clip(nrgb, 0, 255))
            depth = np.load(os.path.join(sd, "depth.npy"))[0]
            mask = np.load(os.path.join(sd, "mask.npy"))[0]
            cells.append(depth_jet(depth, mask))
            cells.append((np.stack([mask] * 3, axis=-1) * 255).astype(np.uint8))
            for i, c in enumerate(cells):
                if c.ndim == 2:
                    cells[i] = np.stack([c] * 3, axis=-1)
                if c.shape[:2] != (thumb, thumb):
                    cells[i] = np.asarray(Image.fromarray(c).resize((thumb, thumb)))
            rows.append(np.concatenate(cells, axis=1))
        except Exception as e:
            print(f"  [montage] 跳过 {os.path.basename(sd)}: {e}")
    if not rows:
        return 0
    canvas = np.concatenate(rows, axis=0)
    Image.fromarray(canvas).save(out_path)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Phase 1 数据集协议校验")
    parser.add_argument("--root", required=True, help="数据集根目录（含场景子目录）")
    parser.add_argument("--sample", type=int, default=50, help="抽样拼图场景数（默认 50）")
    parser.add_argument("--out_dir", default=None, help="报告输出目录（默认 root/_validation）")
    args = parser.parse_args()

    scene_dirs = sorted(
        d for d in os.listdir(args.root)
        if os.path.isdir(os.path.join(args.root, d))
        and not d.startswith("_")
    )
    if not scene_dirs:
        print(f"[FAIL] {args.root} 下无场景目录"); sys.exit(1)

    all_issues, all_stats = {}, {}
    for s in scene_dirs:
        issues, stat = validate_scene(os.path.join(args.root, s))
        if issues:
            all_issues[s] = issues
        all_stats[s] = stat

    out_dir = args.out_dir or os.path.join(args.root, "_validation")
    os.makedirs(out_dir, exist_ok=True)

    # 汇总
    n_fail = len(all_issues)
    report = {
        "root": args.root,
        "scenes": len(scene_dirs),
        "failed_scenes": n_fail,
        "issues": all_issues,
        "stats": all_stats,
    }
    with open(os.path.join(out_dir, "validation.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)

    # 统计表（CSV）
    import csv
    keys = ["coverage", "normal_sobel_angle", "rerender_psnr", "depth_min", "depth_max",
            "albedo_mean", "sh_absmax", "size"]
    with open(os.path.join(out_dir, "stats.csv", ), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["scene"] + keys)
        w.writeheader()
        for s in scene_dirs:
            row = {"scene": s}
            row.update({k: all_stats[s].get(k, "") for k in keys})
            w.writerow(row)

    # 抽样拼图（随机 seed 固定，可复现）
    random.seed(0)
    sample = random.sample(scene_dirs, min(args.sample, len(scene_dirs)))
    sample_abs = [os.path.join(args.root, s) for s in sample]
    n_shot = make_montage(sample_abs, os.path.join(out_dir, "montage_sample.png"))
    print(f"拼图已生成: {n_shot} 场景 -> {os.path.join(out_dir, 'montage_sample.png')}")

    # 汇总统计（供门禁判定）
    if all_stats:
        cov = [s["coverage"] for s in all_stats.values() if "coverage" in s]
        ang = [s["normal_sobel_angle"] for s in all_stats.values() if "normal_sobel_angle" in s]
        psnr = [s["rerender_psnr"] for s in all_stats.values() if "rerender_psnr" in s]
        print(f"\n场景数: {len(scene_dirs)} | 失败场景: {n_fail}")
        if cov:
            print(f"掩码覆盖: mean={np.mean(cov):.3f}  range=[{np.min(cov):.3f}, {np.max(cov):.3f}]")
        if ang:
            print(f"法线-导数夹角: mean={np.mean(ang):.2f}°  max={np.max(ang):.2f}°")
        if psnr:
            print(f"重渲染 PSNR: mean={np.mean(psnr):.1f}dB  min={np.min(psnr):.1f}dB")
    print(f"\n结果: {'PASS' if n_fail == 0 else 'FAIL'}（详情: {os.path.join(out_dir, 'validation.json')}）")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
