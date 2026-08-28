"""PRE-02 · 不依赖神经网络的 N 信息量实验（只用 GT + 优化/求解）。

实验 1  GT geometry 条件下恢复 material/light：
    固定 n=n^GT，交替最小二乘联合求解 A 与 {L_k}（signed SH，无 ReLU），
    N∈{1,2,3,5}（真实 5 光 PNG）与 N∈{1,2,3,5,7,10,15}（解析补光）。
    报告 SI-MAE(A)、光照方向误差、lighting shape 误差。
实验 2  GT albedo 条件下恢复 geometry/light：
    固定 A=A^GT，Adam 联合优化 n（每像素）与 {L_k}，GPU，
    N∈{1,2,3,5} 真实图像，32 个 test 场景，128×128。
    报告 normal angular error 随 N 变化。
实验 3  illumination diversity：
    每个子集记录 N、angular spread、方向协方差谱、lighting 设计矩阵条件数
    （κ(Σ_p A²YYᵀ)，9×9），画 error-vs-N 与 error-vs-diversity。
实验 4  新证据 vs 重复证据：
    S={1,2,3} 上分别加入新光（4 或 5）与重复光（1 的拷贝），
    报告 Δ_new 与 Δ_dup（bootstrap 95% CI）。真实图（含路径追踪噪声）与
    解析图（加 σ=0.005 高斯噪声模拟）各做一组。

固定子集（所有场景/模型共用，禁止各自随机抽图）：
  real5:    N1=[1] N2=[1,3] N3=[1,3,5] N5=[1..5]
  analytic: 在 15 光索引上等距取 {1,8,15} 等（见 SUBSETS_ANALYTIC）

用法（repo 根目录）:
  python pre0/source/information_audit/pre02.py --exp 1 3 4          # CPU 快
  python pre0/source/information_audit/pre02.py --exp 2              # GPU
输出: pre0/information_audit/{subset_results.csv,diversity_results.csv,
  novel_vs_duplicate.csv, exp2_normal_recovery.csv, *.png, INFORMATION_AUDIT.md}
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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "renderer")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset")))

from scene_loader import load_scene, list_scenes, scenes_with_files  # noqa: E402
from oracle import sh_basis, camera_frame_matrix, recover_light_dir, make_sphere_grid  # noqa: E402
from relight import analytic_relight, fib_dirs, sh_basis_np  # noqa: E402

SEED = 20260829
SUBSETS_REAL = {1: [0], 2: [0, 2], 3: [0, 2, 4], 5: [0, 1, 2, 3, 4]}
SUBSETS_ANALYTIC = {1: [0], 2: [0, 7], 3: [0, 7, 14], 5: [0, 3, 6, 10, 14],
                    7: [0, 2, 4, 7, 9, 12, 14],
                    10: [0, 1, 3, 4, 6, 7, 9, 10, 13, 14],
                    15: list(range(15))}
GRID = make_sphere_grid()


# ---------------- 指标 ----------------
def si_mae(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    """逐场景全局标量尺度不变 MAE（与 evaluate.albedo_metrics 同定义）"""
    p, g = pred[mask], gt[mask]
    denom = (p * p).sum()
    if denom < 1e-12:
        return float("nan")
    s = (p * g).sum() / denom
    return float(np.abs(s * p - g).mean())


def ang_err(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip(a @ b, -1, 1))))


# ---------------- 实验 1：固定 n，解 A 与 {L} ----------------
def _torch_sh_basis(n):
    """n: [...,3] -> [...,9]（与 physics_renderer 同阶同序）"""
    import torch
    x, y, z = n[..., 0], n[..., 1], n[..., 2]
    C0, C1 = 0.282095, 0.488603
    C2 = [1.092548, 1.092548, 0.315392, 1.092548, 0.546274]
    return torch.stack([
        torch.full_like(x, C0),
        C1 * y, C1 * z, C1 * x,
        C2[0] * x * y, C2[1] * y * z, C2[2] * (3 * z * z - 1),
        C2[3] * x * z, C2[4] * (x * x - y * y),
    ], dim=-1)


def exp1_scene(scene: dict, imgs_lin: np.ndarray, light_dirs_w: np.ndarray,
               subset: list, iters=500, lr=1e-2, lam_tv=0.03, device="cuda",
               model: str = "sh", restarts: int = 1) -> dict:
    """固定 n=n^GT，用真实协议模型（I = A⊙ReLU(Y c)，含 ReLU）联合优化 A 与 {c_k}。

    病态性与处理（写入 INFORMATION_AUDIT）：
    逐像素自由 A + 每光 9 维 SH 的双线性恢复在失配数据上存在"伪解吸引子"
    （A 收敛到 首图/常数，与 GT 反照率相关性低）。为打破病态性，对 A 施加
    全变差正则（物理动机：反照率分片平滑，shading 随几何高频变化）；
    lam_tv 对所有 N / 两个域固定，保证 N 间与域间可比。A 以 softplus 参数化
    保持非负；光照 signed；初始化不使用 GT 光照/反照率信息。

    model="sh"  : 每光 9 维自由 SH（协议光照族）——真实域存在病态吸引子
                  （A←首图/DC），用于呈现病态性证据。
    model="dir" : 每光 方向+强度 4 dof（经典 uncalibrated PS 方向光族）——
                  可辨识，真实域用于 N 趋势分析。
    """
    import torch
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    H, W = scene["mask_bool"].shape
    mask = scene["mask_bool"]
    n_cam = scene["normal"].transpose(1, 2, 0)
    M = camera_frame_matrix()
    n_world = n_cam @ M.T
    Y_full = torch.from_numpy(sh_basis(n_world)).float().to(dev)     # [H,W,9]
    n_world_t = torch.from_numpy(n_world).float().to(dev)            # [H,W,3]
    m = torch.from_numpy(mask.astype(np.float32)).to(dev)            # [H,W]
    I = torch.from_numpy(imgs_lin[subset]).float().to(dev)           # [N,H,W]
    N = len(subset)
    A_gt = scene["albedo"][0]

    # 初始化（无 GT 泄露）：A=常数(均值图)，c=DC
    I_mean = float((I * m).sum() / m.sum() / N)
    a_raw = torch.full((1, 1, H, W), math.log(math.expm1(max(I_mean / 0.3, 1e-3))),
                       device=dev, requires_grad=True)
    c = torch.zeros(N, 9, device=dev, requires_grad=True)
    with torch.no_grad():
        c[:, 0] = 0.3
    if model == "dir":
        # 方向光族：c_k = α_k · d_k（3+1 参数/光），s_k = α_k·ReLU(Y·d̂_k)
        d_raw = torch.zeros(N, 3, device=dev, requires_grad=True)
        alpha = torch.zeros(N, device=dev, requires_grad=True)
        with torch.no_grad():
            fib0 = fib_dirs(8)
            for k in range(N):
                d_raw[k] = torch.from_numpy(fib0[k % 8]).float().to(dev)
                alpha[k] = max(I_mean, 1e-3)
        params = [{"params": [a_raw], "lr": lr},
                  {"params": [d_raw, alpha], "lr": lr}]
    else:
        params = [{"params": [a_raw], "lr": lr},
                  {"params": [c], "lr": lr}]
    def _init_params():
        a = torch.full((1, 1, H, W), math.log(math.expm1(max(I_mean / 0.3, 1e-3))),
                       device=dev, requires_grad=True)
        if model == "dir":
            d0 = torch.zeros(N, 3, device=dev)
            al0 = torch.zeros(N, device=dev)
            with torch.no_grad():
                fib0 = fib_dirs(8)
                for k in range(N):
                    d0[k] = torch.from_numpy(fib0[k % 8]).float().to(dev)
                    al0[k] = max(I_mean, 1e-3)
            d0.requires_grad_(True); al0.requires_grad_(True)
            return a, d0, al0, None
        c0 = torch.zeros(N, 9, device=dev)
        with torch.no_grad():
            c0[:, 0] = 0.3
        c0.requires_grad_(True)
        return a, None, None, c0

    best = None
    for _rs in range(restarts):
        a_raw, d_raw, alpha, c = _init_params()
        if model == "dir":
            params = [{"params": [a_raw], "lr": lr},
                      {"params": [d_raw, alpha], "lr": lr}]
        else:
            params = [{"params": [a_raw], "lr": lr},
                      {"params": [c], "lr": lr}]
        opt = torch.optim.Adam(params, betas=(0.9, 0.99))
        for _ in range(iters):
            opt.zero_grad()
            A = torch.nn.functional.softplus(a_raw)[0, 0]            # [H,W]
            if model == "dir":
                d = d_raw / (d_raw.norm(dim=1, keepdim=True) + 1e-9)  # [N,3]
                s_k = torch.nn.functional.relu(n_world_t @ d.T)      # [H,W,N]
                s = (s_k * alpha).permute(2, 0, 1)                   # [N,H,W]
            else:
                s = torch.nn.functional.relu(Y_full @ c.T).permute(2, 0, 1)
            recon = (((A[None] * s - I) * m) ** 2).sum() / (m.sum() * N)
            Am = A * m
            tv_x = (Am[:, 1:] - Am[:, :-1]).abs() * m[:, 1:] * m[:, :-1]
            tv_y = (Am[1:, :] - Am[:-1, :]).abs() * m[1:, :] * m[:-1, :]
            tv = (tv_x.sum() + tv_y.sum()) / m.sum()
            loss = recon + lam_tv * tv
            loss.backward()
            opt.step()
        with torch.no_grad():
            if best is None or float(loss) < best[0]:
                A_hat = torch.nn.functional.softplus(a_raw)[0, 0].cpu().numpy() * mask
                if model == "dir":
                    d_fit = (d_raw / (d_raw.norm(dim=1, keepdim=True) + 1e-9)).cpu().numpy()
                    a_fit = alpha.cpu().numpy()
                    best = (float(loss), A_hat, d_fit, a_fit)
                else:
                    best = (float(loss), A_hat, None, c.cpu().numpy())
    e_A = si_mae(best[1], A_gt, mask)
    A_hat = best[1]
    if model == "dir":
        d_fit, a_fit = best[2], best[3]
    else:
        c_np = best[3]

    dir_errs, shape_errs = [], []
    if model == "dir":
        for k, idx in enumerate(subset):
            d_gt_vec = light_dirs_w[idx]
            dir_errs.append(ang_err(d_fit[k], d_gt_vec))
            # 强度相对误差（对齐全局尺度：SI 意义下用 alpha 比值）
            shape_errs.append(float(a_fit[k]))
        se = float(np.std(shape_errs) / max(np.mean(shape_errs), 1e-9))
    else:
        for k, idx in enumerate(subset):
            d_hat, _ = recover_light_dir(c_np[k], GRID)
            d_gt_vec = light_dirs_w[idx]
            dir_errs.append(ang_err(d_hat, d_gt_vec))
            cg = sh_basis_np(d_gt_vec[None])[0]
            shape_errs.append(float(np.linalg.norm(
                c_np[k] / max(np.linalg.norm(c_np[k]), 1e-9)
                - cg / np.linalg.norm(cg))))
        se = float(np.mean(shape_errs))
    return dict(si_mae_A=e_A, light_dir_err=float(np.mean(dir_errs)),
                light_shape_err=se)


# ---------------- 实验 2：固定 A，Adam 优化 n 与 {L} ----------------
def _sh_basis_torch(n):
    """n: [B,3] -> [B,9]，与 numpy 版同阶同序"""
    import torch
    x, y, z = n[:, 0], n[:, 1], n[:, 2]
    C0, C1 = 0.282095, 0.488603
    C2 = [1.092548, 1.092548, 0.315392, 1.092548, 0.546274]
    return torch.stack([
        torch.full_like(x, C0),
        C1 * y, C1 * z, C1 * x,
        C2[0] * x * y, C2[1] * y * z, C2[2] * (3 * z * z - 1),
        C2[3] * x * z, C2[4] * (x * x - y * y),
    ], dim=-1)


def exp2_run(scene_dirs, subset, imgs_provider, iters=2000, lr_n=3e-3, lr_c=3e-3,
             device="cuda", domain="real"):
    """固定 A=A^GT，联合优化每像素世界系法线 n 与每光 SH 系数 c（ReLU 协议模型）。

    存在全局旋转规范自由度（旋转所有法线+反向旋转所有光照 → 图像不变），
    评估前用 Kabsch 将拟合法线对齐到 GT 法线，再报告角误差。
    imgs_provider(scene) -> [K,H,W] 线性域图像（real=磁盘PNG；analytic=解析补光）。
    """
    import torch
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    N = len(subset)
    results = []
    sl = (slice(0, 256, 2), slice(0, 256, 2))       # 256 -> 128 隔点采样
    M = camera_frame_matrix()                       # d_cam = M @ d_world
    for sd in scene_dirs:
        sc = load_scene(sd)
        imgs = imgs_provider(sc)
        mask_s = sc["mask_bool"][sl]
        if not mask_s.any():
            continue
        A_s = torch.from_numpy(sc["albedo"][0][sl]).float().to(dev)
        I = torch.from_numpy(imgs[subset][:, sl[0], sl[1]]).float().to(dev)
        n_gt_cam = sc["normal"].transpose(1, 2, 0)[sl]
        n_gt_world = n_gt_cam @ M.T
        n_gt_t = torch.from_numpy(n_gt_world).float().to(dev)

        # 对称性破缺初始化：全 (0,0,1) 极点是鞍点（SH 的 x/y 项梯度恒 0，
        # c1/c7 永远无法激活），需小随机扰动 + c 小噪声才能启动联合优化
        g = torch.Generator(device="cpu").manual_seed(SEED)
        eps = torch.randn(*mask_s.shape, 2, generator=g) * 0.05
        n_init = torch.zeros(*mask_s.shape, 3)
        n_init[..., :2] = eps
        n_init[..., 2] = 1.0
        n_param = n_init.to(dev).requires_grad_(True)
        c_param = (torch.randn(N, 9, generator=g) * 0.01).to(dev)
        with torch.no_grad():
            c_param[:, 0] += 0.5
        c_param.requires_grad_(True)
        opt = torch.optim.Adam([{"params": [n_param], "lr": lr_n},
                                {"params": [c_param], "lr": lr_c}])
        mflat = mask_s
        m_t = torch.from_numpy(mask_s.astype(np.float32)).to(dev)
        for _ in range(iters):
            opt.zero_grad()
            n = n_param / (n_param.norm(dim=-1, keepdim=True) + 1e-9)
            Y = _sh_basis_torch(n.reshape(-1, 3)).reshape(*mask_s.shape, 9)
            s = torch.nn.functional.relu(
                torch.einsum("hwc,nc->nhw", Y, c_param))            # [N,h,w]
            pred = A_s[None] * s
            loss = (((pred - I) ** 2) * m_t[None]).sum() / m_t.sum() / N
            loss.backward()
            opt.step()
        with torch.no_grad():
            n_hat = (n_param / n_param.norm(dim=-1, keepdim=True)).cpu().numpy()
            P_hat = n_hat[mflat]
            P_gt = n_gt_world[mflat]
            Hm = P_hat.T @ P_gt
            U, _, Vt = np.linalg.svd(Hm)
            dd = np.sign(np.linalg.det(Vt.T @ U.T))
            R = Vt.T @ np.diag([1, 1, dd]) @ U.T
            P_hat_al = P_hat @ R.T
            dot = np.clip((P_hat_al * P_gt).sum(-1), -1, 1)
            ang = np.degrees(np.arccos(dot))
            # observable 像素：mask 内、至少一图有信号(>0.02)、非低反照率
            # （全光阴影/极暗像素无法约束法线，随机漂移造成重尾，须单列）
            obs = mask_s & (imgs[:, sl[0], sl[1]].max(0) > 0.02)                        & (sc["albedo"][0][sl] > 0.05)
            obs_flat = obs[mask_s]
            results.append(dict(domain=domain, scene=sc["scene"], N=N,
                                normal_ang_mae=float(ang.mean()),
                                normal_ang_median=float(np.median(ang)),
                                normal_ang_p90=float(np.percentile(ang, 90)),
                                normal_ang_mae_obs=float(ang[obs_flat].mean()),
                                normal_ang_median_obs=float(np.median(ang[obs_flat])),
                                obs_frac=float(obs_flat.mean())))
    return results


# ---------------- 实验 3：diversity ----------------
def diversity_metrics(dirs_w: np.ndarray) -> dict:
    """dirs_w: [N,3] 世界系单位方向。定义全部写明，不发明阈值。"""
    N = len(dirs_w)
    if N >= 2:
        ang = [math.degrees(math.acos(np.clip(dirs_w[i] @ dirs_w[j], -1, 1)))
               for i in range(N) for j in range(i + 1, N)]
        spread = float(np.mean(ang))
    else:
        spread = 0.0
    C = dirs_w.T @ dirs_w / N                       # [3,3]
    ev = np.linalg.eigvalsh(C)
    eig_ratio = float(ev[-1] / max(ev[0], 1e-9))
    return dict(N=N, angular_spread_deg=spread, cov_eig_min=float(ev[0]),
                cov_eig_max=float(ev[-1]), cov_eig_ratio=eig_ratio)


def kappa_lighting(scene: dict, dirs_w: np.ndarray, subset: list) -> float:
    """光照可辨识性条件数：κ(Σ_{p∈mask} A(p)² Y(n_p) Y(n_p)ᵀ)（9×9，逐光平均）"""
    mask = scene["mask_bool"]
    n_cam = scene["normal"].transpose(1, 2, 0)
    M = camera_frame_matrix()
    n_world = n_cam @ M.T
    Y = sh_basis(n_world[mask])                     # [P,9]
    A = scene["albedo"][0]
    kappas = []
    for idx in subset:
        G = (Y * A[mask][:, None] ** 2).T @ Y
        ev = np.linalg.eigvalsh(G)
        kappas.append(float(ev[-1] / max(ev[0], 1e-12)))
    return float(np.mean(kappas))


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", type=int, nargs="+", default=[1, 3, 4])
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--exp2_scenes", type=int, default=32)
    ap.add_argument("--data_root", default="D:/data/synthetic_v3")
    args = ap.parse_args()

    out = os.path.join(_REPO, "pre0", "information_audit")
    os.makedirs(out, exist_ok=True)
    manifest = os.path.join(_REPO, "pre0", "protocol", "split_manifest.json")
    ids = list_scenes(manifest, args.split)
    scene_dirs = scenes_with_files(args.data_root, ids)
    if args.limit:
        scene_dirs = scene_dirs[: args.limit]
    print(f"[pre02] scenes={len(scene_dirs)} exps={args.exp}")

    rows_subset, rows_div, rows_nvd = [], [], []
    if 1 in args.exp or 3 in args.exp or 4 in args.exp:
        # 预加载解析补光（逐场景，缓存目录避免重复计算）
        cache = os.path.join(out, "_relight_cache")
        os.makedirs(cache, exist_ok=True)

    # ---- 实验 1 + 3（真实 5 光 + 解析 15 光）----
    if 1 in args.exp or 3 in args.exp:
        dirs_real5 = None
        for si, sd in enumerate(scene_dirs):
            sc = load_scene(sd)
            # 真实 5 光方向（世界系，全数据集恒定，取自 sh_coeffs 反解一次即可）
            if dirs_real5 is None:
                dirs_real5 = np.stack([
                    recover_light_dir(sc["sh"][k], GRID)[0] for k in range(5)])
            n_cam = sc["normal"].transpose(1, 2, 0)
            M = camera_frame_matrix()
            # 解析 15 光
            cf = os.path.join(cache, sc["scene"] + ".npy")
            if os.path.isfile(cf):
                imgs15 = np.load(cf)
            else:
                imgs15 = analytic_relight(sc)
                np.save(cf, imgs15)
            doms = [("real5_sh", sc["img_lin"], SUBSETS_REAL, dirs_real5, "sh"),
                    ("real5_dir", sc["img_lin"], SUBSETS_REAL, dirs_real5, "dir"),
                    ("analytic15_sh", imgs15, SUBSETS_ANALYTIC, fib_dirs(), "sh")]
            for dom_name, imgs, subsets, dirs_w, mdl in doms:
                for N, sub in subsets.items():
                    r = exp1_scene(sc, imgs, dirs_w, sub, model=mdl)
                    div = diversity_metrics(dirs_w[sub])
                    kap = kappa_lighting(sc, dirs_w, sub)
                    rows_subset.append(dict(domain=dom_name, model=mdl,
                                            scene=sc["scene"], N=N, **r))
                    rows_div.append(dict(domain=dom_name, model=mdl,
                                         scene=sc["scene"], si_mae_A=r["si_mae_A"],
                                         kappa_lighting=kap, **div))
            if (si + 1) % 20 == 0:
                print(f"  exp1/3 {si+1}/{len(scene_dirs)}")

        with open(os.path.join(out, "subset_results.csv"), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_subset[0].keys()))
            w.writeheader(); w.writerows(rows_subset)
        with open(os.path.join(out, "diversity_results.csv"), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_div[0].keys()))
            w.writeheader(); w.writerows(rows_div)

        # bootstrap CI + 图
        plot_n_curves(rows_subset, rows_div, out)

    # ---- 实验 4：novel vs duplicate ----
    if 4 in args.exp:
        rng = np.random.default_rng(SEED)
        _dirs_cache = {}
        for si, sd in enumerate(scene_dirs):
            sc = load_scene(sd)
            if si == 0:
                dirs_real5_cache = np.stack([
                    recover_light_dir(sc["sh"][k], GRID)[0] for k in range(5)])
            else:
                dirs_real5_cache = _dirs_cache.get("d") or dirs_real5_cache
            mask = sc["mask_bool"]
            A_gt = sc["albedo"][0]
            imgs = sc["img_lin"]
            # 真实域
            S = [0, 2, 4]
            sets = {"S3": S, "S3+new": S + [1], "S3+dup": S + [0]}
            rec = {}
            for name, sub in sets.items():
                r = exp1_scene(sc, imgs, dirs_real5_cache, sub, model="dir", restarts=1, iters=800)
                rec[name] = r["si_mae_A"]
            rows_nvd.append(dict(domain="real_dir", scene=sc["scene"],
                                 E_S3=rec["S3"], E_S3_new=rec["S3+new"],
                                 E_S3_dup=rec["S3+dup"],
                                 d_new=rec["S3"] - rec["S3+new"],
                                 d_dup=rec["S3"] - rec["S3+dup"]))
            # 解析域 + 噪声
            cf = os.path.join(cache, sc["scene"] + ".npy")
            imgs15 = np.load(cf) if os.path.isfile(cf) else analytic_relight(sc)
            noisy = imgs15 + rng.normal(0, 0.005, imgs15.shape).astype(np.float32)
            sub3 = SUBSETS_ANALYTIC[3]
            r3 = exp1_scene(sc, noisy, fib_dirs(), sub3, model="sh", restarts=1, iters=800)["si_mae_A"]
            r4n = exp1_scene(sc, noisy, fib_dirs(), sub3 + [8], model="sh", restarts=1, iters=800)["si_mae_A"]
            r4d = exp1_scene(sc, noisy, fib_dirs(), sub3 + [sub3[0]], model="sh", restarts=1, iters=800)["si_mae_A"]
            rows_nvd.append(dict(domain="analytic_noisy_sh", scene=sc["scene"],
                                 E_S3=r3, E_S3_new=r4n, E_S3_dup=r4d,
                                 d_new=r3 - r4n, d_dup=r3 - r4d))
            if (si + 1) % 20 == 0:
                print(f"  exp4 {si+1}/{len(scene_dirs)}")
        with open(os.path.join(out, "novel_vs_duplicate.csv"), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_nvd[0].keys()))
            w.writeheader(); w.writerows(rows_nvd)
        plot_nvd(rows_nvd, out)

    # ---- 实验 2：法线恢复（真实域 + 解析域）----
    if 2 in args.exp:
        sub_dirs = scene_dirs[: args.exp2_scenes]
        all_res = []
        for N in (1, 2, 3, 5):
            res = exp2_run(sub_dirs, SUBSETS_REAL[N],
                           lambda sc: sc["img_lin"], domain="real")
            all_res.extend(res)
            print(f"  exp2 real N={N}: mean ang "
                  f"{np.mean([r['normal_ang_mae'] for r in res]):.2f}°")
        cache = os.path.join(out, "_relight_cache")
        os.makedirs(cache, exist_ok=True)

        def _analytic(sc):
            cf = os.path.join(cache, sc["scene"] + ".npy")
            if os.path.isfile(cf):
                return np.load(cf)
            im = analytic_relight(sc)
            np.save(cf, im)
            return im

        for N in (1, 2, 3, 5, 7, 10, 15):
            res = exp2_run(sub_dirs, SUBSETS_ANALYTIC[N], _analytic,
                           domain="analytic")
            all_res.extend(res)
            print(f"  exp2 analytic N={N}: mean ang "
                  f"{np.mean([r['normal_ang_mae'] for r in res]):.2f}°")
        with open(os.path.join(out, "exp2_normal_recovery.csv"), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_res[0].keys()))
            w.writeheader(); w.writerows(all_res)
        plot_exp2(all_res, out)

    print("[pre02] done ->", out)


# ---------------- 绘图 ----------------
def _bootstrap_ci(vals, n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    vals = np.asarray(vals)
    means = [rng.choice(vals, len(vals), replace=True).mean() for _ in range(n)]
    return float(vals.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def plot_n_curves(rows, rows_div, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=120)
    for ax, (metric, label) in zip(axes, [
            ("si_mae_A", "SI-MAE(A)"),
            ("light_dir_err", "light dir err (deg)"),
            ("light_shape_err", "light shape err (L2)")]):
        for dom in sorted(set(r["domain"] for r in rows)):
            xs = sorted(set(r["N"] for r in rows if r["domain"] == dom))
            mv, lo, hi = [], [], []
            for N in xs:
                v = [r[metric] for r in rows if r["domain"] == dom and r["N"] == N]
                m, l, h = _bootstrap_ci(v)
                mv.append(m); lo.append(m - l); hi.append(h - m)
            ax.errorbar(xs, mv, yerr=[lo, hi], marker="o", capsize=3, label=dom)
        ax.set_xlabel("N"); ax.set_ylabel(label); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "n_error_curves.png"))
    plt.close(fig)

    # error vs diversity（散点）
    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=120)
    for dom in sorted(set(r["domain"] for r in rows_div)):
        xs = [r["angular_spread_deg"] for r in rows_div if r["domain"] == dom]
        ys = [r["si_mae_A"] for r in rows_div if r["domain"] == dom]
        ax.scatter(xs, ys, s=8, alpha=0.4, label=dom)
    ax.set_xlabel("angular spread (deg)"); ax.set_ylabel("SI-MAE(A)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "diversity_error_scatter.png"))
    plt.close(fig)


def plot_nvd(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=120)
    for ax, dom in zip(axes, ["real", "analytic_noisy"]):
        rs = [r for r in rows if r["domain"] == dom]
        m_new, l_new, h_new = _bootstrap_ci([r["d_new"] for r in rs])
        m_dup, l_dup, h_dup = _bootstrap_ci([r["d_dup"] for r in rs])
        ax.bar(["Δ_new", "Δ_dup"], [m_new, m_dup],
               yerr=[[m_new - l_new, m_dup - l_dup], [h_new - m_new, h_dup - m_dup]],
               capsize=4, color=["#2a7de1", "#999"])
        ax.set_title(f"{dom}: add 4th light vs duplicate")
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "novel_vs_duplicate.png"))
    plt.close(fig)


def plot_exp2(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 4), dpi=120)
    for dom in ("real", "analytic"):
        xs = sorted(set(r["N"] for r in rows if r["domain"] == dom))
        mv, lo, hi = [], [], []
        for N in xs:
            v = [r["normal_ang_median_obs"] for r in rows
                 if r["domain"] == dom and r["N"] == N]
            m, l, h = _bootstrap_ci(v)
            mv.append(m); lo.append(m - l); hi.append(h - m)
        ax.errorbar(xs, mv, yerr=[lo, hi], marker="o", capsize=3, label=dom)
    ax.set_xlabel("N"); ax.set_ylabel("normal angular median err, observable px (deg)")
    ax.set_title("normal recovery vs N (GT albedo fixed)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "exp2_normal_vs_N.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
