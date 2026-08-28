"""PRE-01 · GT Oracle 物理一致性审计（无网络）。

实验 A  GT Oracle Reconstruction:
    Î_k = A^GT ⊙ ReLU(Σ c_ki Y_i(n^GT))，与真实图像在三个域分别对比
    （linear 物理域 / train 域 / train 域+域变换 oracle），逐场景记录
    MAE/MSE/PSNR/SSIM/residual 统计，保存失败案例与残差空间图。
实验 B  Normal protocol audit:
    normal.npy 理论上 = sobel_normal(depth)（按构造恒等）——本实验实测验证，
    并报告"mesh normal 对照不可行"的协议事实。
实验 C  Lighting convention sanity:
    从 sh_coeffs 反解每盏光的隐含方向 d̂（球面网格上最大化 <c, Y(d)>），
    与相机几何推导的期望相机系方向 d_exp 对比；再对三个规范法线做
    SH shading 数值自检。

用法（repo 根目录）:
    python pre0/source/renderer/oracle.py --split test [--limit 0]
输出:
    pre0/oracle_renderer/{oracle_metrics.csv,oracle_summary.json,
    normal_protocol.csv,residual_*.png,lighting_convention.csv,ORACLE_AUDIT.md}
"""
import argparse
import csv
import json
import math
import os
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset")))

from scene_loader import load_scene, list_scenes, scenes_with_files  # noqa: E402

# ---- SH 常量（与 physics_renderer 完全一致）----
C0, C1 = 0.282095, 0.488603
C2 = [1.092548, 1.092548, 0.315392, 1.092548, 0.546274]


def sh_basis(n_xyz: np.ndarray) -> np.ndarray:
    """n_xyz: [...,3] -> [...,9]，顺序与 physics_renderer.compute_sh_basis 一致"""
    x, y, z = n_xyz[..., 0], n_xyz[..., 1], n_xyz[..., 2]
    return np.stack([
        np.full_like(x, C0),
        C1 * y, C1 * z, C1 * x,
        C2[0] * x * y, C2[1] * y * z, C2[2] * (3 * z * z - 1),
        C2[3] * x * z, C2[4] * (x * x - y * y),
    ], axis=-1)


def shading_from_sh(normal_hwc: np.ndarray, sh_k9: np.ndarray) -> np.ndarray:
    """s = ReLU(Σ c_i Y_i(n))；normal [H,W,3]，sh [K,9] -> [K,H,W]"""
    B = sh_basis(normal_hwc)                      # [H,W,9]
    return np.maximum(B @ sh_k9.T, 0.0).T         # [K,H,W]


def sobel_normal_np(depth_hw: np.ndarray) -> np.ndarray:
    """复刻 render_dataset.sobel_normal（4 阶中心差分近似）"""
    d = depth_hw
    p = np.pad(d, 1, mode="edge")
    gx = (-p[:-2, :-2] + p[2:, :-2] - 2 * p[:-2, 1:-1] + 2 * p[2:, 1:-1]
          - p[:-2, 2:] + p[2:, 2:]) / 4.0
    gy = (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2]
          - p[:-2, 2:] - 2 * p[1:-1, 2:] - p[2:, 2:]) / 4.0
    n = np.stack([-gx, -gy, np.ones_like(gx)], axis=-1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def masked_stats(err: np.ndarray, mask: np.ndarray) -> dict:
    e = err[mask]
    mse = float((e ** 2).mean())
    m = dict(mae=float(np.abs(e).mean()), mse=mse,
             psnr=float(10 * math.log10(1.0 / max(mse, 1e-12))),
             res_mean=float(e.mean()), res_std=float(e.std()))
    return m


def ssim_torch(x, y):
    """标准 SSIM（11×15 高斯窗），输入 [1,1,H,W] torch"""
    import torch
    from evaluate import _gaussian_window, _ssim_torch as _ssim
    win = _gaussian_window().to(x.device)
    return float(_ssim(x, y, win).mean())


# ---------------------------------------------------------------------------
# 实验对象
# ---------------------------------------------------------------------------
def expected_camera_light_dirs() -> np.ndarray:
    """由相机几何推导 5 盏光在相机系的方向（render_dataset.light_dirs 的刚体变换）。

    相机：pos=2.6*(cos30°,0,sin30°) look-at 原点，up=世界 +Z（look_at 实现）。
    世界→相机：d_c = [d·x_axis, d·y_axis, d·z_axis]。
    """
    el, R = math.radians(50.0), 2.99
    cam_pos = np.array([2.6 * math.cos(math.radians(30.0)), 0.0,
                        2.6 * math.sin(math.radians(30.0))])
    z_ax = cam_pos / np.linalg.norm(cam_pos)          # look_at: z = (pos-target)/|..|
    up = np.array([0.0, 0.0, 1.0])
    x_ax = np.cross(up, z_ax); x_ax /= np.linalg.norm(x_ax)
    y_ax = np.cross(z_ax, x_ax)
    dirs_w = []
    for k in range(5):
        az = k * (360.0 / 5)
        lw = np.array([R * math.cos(el) * math.cos(math.radians(az)),
                       R * math.cos(el) * math.sin(math.radians(az)),
                       R * math.sin(el)])
        dirs_w.append(lw / np.linalg.norm(lw))
    Rot = np.stack([x_ax, y_ax, z_ax])                # 行向量 = 相机基
    return np.stack([Rot @ d for d in dirs_w])        # [5,3] 相机系


def world_light_dirs() -> np.ndarray:
    """5 盏光的世界系单位方向（render_dataset.light_dirs 语义）"""
    el, R = math.radians(50.0), 2.99
    out = []
    for k in range(5):
        az = k * (360.0 / 5)
        lw = np.array([R * math.cos(el) * math.cos(math.radians(az)),
                       R * math.cos(el) * math.sin(math.radians(az)),
                       R * math.sin(el)])
        out.append(lw / np.linalg.norm(lw))
    return np.stack(out)


def camera_frame_matrix() -> np.ndarray:
    """法线系(相机系) <- 世界系 的旋转 M：d_cam = M @ d_world。

    法线系基：x=图像右(世界 (0,1,0))，y=图像下，z=朝相机(世界 (0.866,0,0.5))。
    注意 y 与 look_at 的 y_axis 相反（图像 y 向下）。
    """
    cam_pos = np.array([2.6 * math.cos(math.radians(30.0)), 0.0,
                        2.6 * math.sin(math.radians(30.0))])
    z_c = cam_pos / np.linalg.norm(cam_pos)
    x_c = np.array([0.0, 1.0, 0.0])
    y_c = np.cross(z_c, x_c); y_c /= np.linalg.norm(y_c)
    return np.stack([x_c, y_c, z_c])


def pixel_world_positions(depth_hw: np.ndarray, fov_deg: float = 50.0) -> np.ndarray:
    """由视空间 z 深度恢复像素世界坐标 [H,W,3]（针孔模型，z 沿视线轴）"""
    H, W = depth_hw.shape
    f = (H / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    cam_pos = np.array([2.6 * math.cos(math.radians(30.0)), 0.0,
                        2.6 * math.sin(math.radians(30.0))])
    fwd = -cam_pos / np.linalg.norm(cam_pos)
    x_img = np.array([0.0, 1.0, 0.0])
    y_up = np.cross(x_img, -fwd); y_up /= np.linalg.norm(y_up)
    u = np.arange(W)[None, :].repeat(H, 0)
    v = np.arange(H)[:, None].repeat(W, 1)
    du = (u - (W - 1) / 2.0) / f
    dv = ((H - 1) / 2.0 - v) / f
    return (cam_pos[None, None, :]
            + depth_hw[..., None] * (fwd[None, None, :] + du[..., None] * x_img[None, None, :]
                                     + dv[..., None] * y_up[None, None, :]))


def point_light_direct_image(albedo_hw, normal_hwc, depth_hw, light_w, energy=100.0):
    """近场点光直接漫反射物理模型（无间接光/无阴影截断）。

    I(p) = A(p) * energy/(4π|L-p|²) * max(0, n_world·(L-p)/|L-p|)
    """
    M = camera_frame_matrix()
    n_w = normal_hwc @ M.T
    p_w = pixel_world_positions(depth_hw)
    d = light_w[None, None, :] - p_w
    dist2 = (d ** 2).sum(-1)
    dist = np.sqrt(dist2)
    cosv = np.maximum((n_w * d).sum(-1) / np.maximum(dist, 1e-9), 0.0)
    return albedo_hw * energy / (4 * math.pi * dist2) * cosv


def recover_light_dir(c9: np.ndarray, grid: np.ndarray) -> tuple:
    """从 9 维 SH 系数反解隐含方向：d̂ = argmax_d <c, Y(d)>，返回 (d̂, <c,Y(d̂)>)"""
    Y = sh_basis(grid)                                # [G,9]
    proj = Y @ c9                                     # [G]
    i = int(np.argmax(proj))
    d = grid[i]
    d = d / np.linalg.norm(d)
    return d, float(proj[i])


def make_sphere_grid(n_phi=180, n_theta=90):
    phis = np.linspace(0, 2 * math.pi, n_phi, endpoint=False)
    thetas = np.linspace(0.02, math.pi - 0.02, n_theta)
    P, T = np.meshgrid(phis, thetas)
    g = np.stack([np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)], axis=-1)
    return g.reshape(-1, 3)


def save_residual_map(residual, mask, path, vmin=-0.25, vmax=0.25):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = np.where(mask, residual, np.nan)
    fig, ax = plt.subplots(figsize=(3, 3), dpi=110)
    im = ax.imshow(r, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax.axis("off"); fig.colorbar(im, fraction=0.046)
    fig.tight_layout(pad=0.2); fig.savefig(path); plt.close(fig)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个场景（0=全部）")
    ap.add_argument("--data_root", default="D:/data/synthetic_v3")
    args = ap.parse_args()

    out_dir = os.path.join(_REPO, "pre0", "oracle_renderer")
    os.makedirs(out_dir, exist_ok=True)

    manifest = os.path.join(_REPO, "pre0", "protocol", "split_manifest.json")
    ids = list_scenes(manifest, args.split)
    scene_dirs = scenes_with_files(args.data_root, ids)
    if args.limit:
        scene_dirs = scene_dirs[: args.limit]
    print(f"[oracle] split={args.split} 场景数={len(scene_dirs)}")

    grid = make_sphere_grid()
    d_exp_cam = expected_camera_light_dirs()          # [5,3] 相机系（若协议把 c 当相机系）
    d_world = world_light_dirs()                      # [5,3] 世界系（生成端真实语义）
    M_cam = camera_frame_matrix()
    d_cam_from_world = np.stack([M_cam @ d for d in d_world])

    rows, norm_rows, light_rows = [], [], []
    summaries = []
    per_scene_err = {}                                # 用于选失败案例

    for si, sd in enumerate(scene_dirs):
        sc = load_scene(sd)
        mask = sc["mask_bool"]
        n_hwc = sc["normal"].transpose(1, 2, 0)       # [H,W,3]
        A = sc["albedo"][0]                            # [H,W] 线性
        sh = sc["sh"]                                  # [5,9]
        shade = shading_from_sh(n_hwc, sh)             # [5,H,W]
        Ihat_lin = A[None] * shade                     # 物理域 oracle

        mets = []
        # 近场点光直接漫反射物理模型（误差归因用）
        Ihat_point = np.stack([
            point_light_direct_image(A, n_hwc, sc["depth"][0], d_world[k] * 2.99)
            for k in range(5)])                        # [5,H,W]
        for k in range(5):
            m_lin = masked_stats(Ihat_lin[k] - sc["img_lin"][k], mask)
            m_pt = masked_stats(Ihat_point[k] - sc["img_lin"][k], mask)
            m_tr = masked_stats(Ihat_lin[k] - sc["img_train"][k], mask)
            Ihat_tr = np.power(np.clip(Ihat_lin[k], 0, 1), 1.0 / 2.2)
            m_trdom = masked_stats(Ihat_tr - sc["img_train"][k], mask)
            rows.append(dict(scene=sc["scene"], light=k + 1,
                             mae_lin=m_lin["mae"], mse_lin=m_lin["mse"],
                             psnr_lin=m_lin["psnr"], res_mean_lin=m_lin["res_mean"],
                             res_std_lin=m_lin["res_std"],
                             mae_train_target=m_tr["mae"],
                             psnr_train_target=m_tr["psnr"],
                             mae_train_domain=m_trdom["mae"],
                             psnr_train_domain=m_trdom["psnr"],
                             mae_point_direct=m_pt["mae"],
                             psnr_point_direct=m_pt["psnr"]))
            mets.append((m_lin, m_trdom))

        # 场景级聚合（linear 域）
        mae = float(np.mean([m["mae"] for m, _ in mets]))
        psnr_lin = float(np.mean([m["psnr"] for m, _ in mets]))
        psnr_trdom = float(np.mean([m["psnr"] for _, m in mets]))
        # SSIM（linear 域，逐光平均，torch）
        try:
            import torch
            from evaluate import _gaussian_window, _ssim_torch as _ssim
            win = _gaussian_window()
            ssims = []
            for k in range(5):
                x = torch.from_numpy(np.clip(Ihat_lin[k], 0, 1))[None, None]
                y = torch.from_numpy(sc["img_lin"][k])[None, None]
                ssims.append(float(_ssim(x, y, win).mean()))
            ssim_mean = float(np.mean(ssims))
        except Exception:
            ssim_mean = float("nan")
        summaries.append(dict(scene=sc["scene"], mae_lin=mae,
                              psnr_lin=psnr_lin, psnr_train_domain=psnr_trdom,
                              ssim_lin=ssim_mean))
        per_scene_err[sc["scene"]] = mae

        # 实验 B：normal.npy vs 重算 sobel normal
        n_re = sobel_normal_np(sc["depth"][0])
        dot = np.clip((n_hwc[mask] * n_re[mask]).sum(-1), -1, 1)
        ang = np.degrees(np.arccos(dot))
        norm_rows.append(dict(scene=sc["scene"],
                              ang_mae=float(ang.mean()),
                              ang_median=float(np.median(ang)),
                              ang_p90=float(np.percentile(ang, 90))))

        # 实验 C：sh_coeffs 反解方向 vs 世界系期望（生成端语义）与相机系期望
        for k in range(5):
            d_hat, proj = recover_light_dir(sh[k], grid)
            err_world = math.degrees(math.acos(float(np.clip(
                np.dot(d_hat, d_world[k]), -1, 1))))
            err_cam = math.degrees(math.acos(float(np.clip(
                np.dot(d_hat, d_cam_from_world[k]), -1, 1))))
            light_rows.append(dict(scene=sc["scene"], light=k + 1,
                                   err_vs_world_deg=err_world,
                                   err_vs_camframe_deg=err_cam,
                                   implied_z=float(d_hat[2]),
                                   i_eff_proj=float(proj)))

        if (si + 1) % 20 == 0:
            print(f"  {si+1}/{len(scene_dirs)}")

    # ---- 失败案例可视化：mae 最差 8 + 中位 4 ----
    order = sorted(per_scene_err, key=per_scene_err.get, reverse=True)
    worst = order[:8]
    mid = order[len(order) // 2 - 2: len(order) // 2 + 2]
    viz_scenes = list(dict.fromkeys(worst + mid))
    for scene in viz_scenes:
        sd = os.path.join(args.data_root, scene)
        sc = load_scene(sd)
        mask = sc["mask_bool"]
        n_hwc = sc["normal"].transpose(1, 2, 0)
        A = sc["albedo"][0]
        Ihat = A[None] * shading_from_sh(n_hwc, sc["sh"])
        for k in [0, 2]:
            save_residual_map(Ihat[k] - sc["img_lin"][k], mask,
                              os.path.join(out_dir, f"residual_{scene}_L{k+1}.png"))
        # 拼一张四联图（图像|oracle|残差|mask）便于人工查看
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        k = 0
        fig, axes = plt.subplots(1, 4, figsize=(11, 3), dpi=110)
        for ax, (img, ttl) in zip(axes, [
                (sc["img_lin"][k], "image (linear)"),
                (Ihat[k], "oracle"),
                (Ihat[k] - sc["img_lin"][k], "residual"),
                (sc["mask"][0], "mask")]):
            if ttl == "residual":
                im = ax.imshow(np.where(mask, img, np.nan), cmap="RdBu_r",
                               vmin=-0.25, vmax=0.25)
                fig.colorbar(im, ax=ax, fraction=0.046)
            else:
                ax.imshow(np.clip(img, 0, 1), cmap="gray" if img.ndim == 2 else None)
            ax.set_title(ttl, fontsize=8); ax.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"panel_{scene}.png"))
        plt.close(fig)

    # ---- 落盘 ----
    def write_csv(path, rows):
        if not rows:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    write_csv(os.path.join(out_dir, "oracle_metrics.csv"), rows)
    write_csv(os.path.join(out_dir, "normal_protocol.csv"), norm_rows)
    write_csv(os.path.join(out_dir, "lighting_convention.csv"), light_rows)

    agg = dict(
        n_scenes=len(summaries),
        mae_lin_mean=float(np.mean([s["mae_lin"] for s in summaries])),
        mae_lin_std=float(np.std([s["mae_lin"] for s in summaries])),
        psnr_lin_mean=float(np.mean([s["psnr_lin"] for s in summaries])),
        psnr_lin_p10=float(np.percentile([s["psnr_lin"] for s in summaries], 10)),
        psnr_train_domain_mean=float(np.mean([s["psnr_train_domain"] for s in summaries])),
        psnr_point_direct_mean=float(np.mean(
            [r["psnr_point_direct"] for r in rows])),
        psnr_point_direct_p10=float(np.percentile(
            [r["psnr_point_direct"] for r in rows], 10)),
        ssim_lin_mean=float(np.nanmean([s["ssim_lin"] for s in summaries])),
        normal_ang_mae_mean=float(np.mean([r["ang_mae"] for r in norm_rows])),
        normal_ang_p90_max=float(np.max([r["ang_p90"] for r in norm_rows])),
        light_err_vs_world_mean=float(np.mean([r["err_vs_world_deg"] for r in light_rows])),
        light_err_vs_camframe_mean=float(np.mean([r["err_vs_camframe_deg"] for r in light_rows])),
        worst_scenes=order[:10],
    )
    json.dump(agg, open(os.path.join(out_dir, "oracle_summary.json"), "w",
                        encoding="utf-8"), indent=2, ensure_ascii=False)
    print(json.dumps(agg, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
