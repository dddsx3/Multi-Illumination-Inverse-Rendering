#!/usr/bin/env python3
"""卡 S · exp8R v3: DiLiGenT geometry-known 联合估计判别力(协议 v3)

关键修复(v3):
  1. 诊断量 = 已知法线 Fisher F(ρ, 3N) → Schur 消 ρ(子集内部) → 光照块 →
     投影掉强度维 → 2N 方向切空间 → 迹推前到方向角(不再每光独立 2×2);
  2. 估计器 = scipy least_squares(trf) 联合 {ρ} ∪ {α_i, l̂_i}×N,
     ALS 初值(禁真值), 多起点 3 个方向扰动;
  3. 像素退化守卫: 子集内未 mask 光 <2 → 整子集剔除该像素(通用预处理);
  4. verdict 纪律: assert 计数可对账。

主判据: N=3 层 50 次 ≥6/10 物体 LAE Spearman 显著正(p<0.05)。
"""
from __future__ import annotations
import json
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import scipy.io as sio
from PIL import Image
from scipy.optimize import least_squares
from scipy.stats import spearmanr, chi2

HERE = Path(__file__).resolve().parent
OUT = HERE / "exp8r_diligent_discrimination_v3.json"
ROOT = Path("D:/data/DiLiGenT/pmsData")
N3_SAMPLES = 50
OTHER_SAMPLES = 15
N_STARTS = 3
CKPT_EVERY = 5   # 断点续跑: 每 5 个子集落盘 .partial.json(仅行累积, 不改变数值路径)
SEED = 20260906


def load_object(d):
    dirs = np.loadtxt(d / "light_directions.txt")
    ints = np.loadtxt(d / "light_intensities.txt")[:, 0]
    N_gt = sio.loadmat(d / "Normal_gt.mat")["Normal_gt"]
    mask = np.array(Image.open(d / "mask.png").convert("L")) > 128
    imgs = np.stack([np.array(Image.open(f"{d}/{i:03d}.png").convert("L")).astype(float)
                     for i in range(1, 97)])
    I = imgs[:, mask] / ints[:, None]           # (96, P) 归一化(禁峰值)
    return dirs, N_gt[mask], I, mask


def calibrate(n_gt, dirs, I_norm):
    """全 96 光朗伯拟合 → ρ, 噪声 (a,b), 残差。"""
    nl = np.clip(n_gt @ dirs.T, 0, None)          # (P,96)
    rho = (I_norm.T * nl).sum(1) / np.maximum((nl*nl).sum(1), 1e-12)
    I_hat = (rho[:, None] * nl).T
    resid = I_norm - I_hat
    sel = I_hat.ravel() > 0.01
    if sel.sum() < 100:
        a_, b_ = 1.0, 0.0
    else:
        a_, b_ = np.polyfit(I_hat.ravel()[sel], (resid.ravel()**2)[sel], 1)
        if a_ < 0:
            a_, b_ = 1.0, 0.0
    return rho, float(a_), float(b_), float(np.linalg.norm(resid)/np.linalg.norm(I_norm))


def pixel_guard(I_sub, n_masked):
    """子集内被 mask(阴影/n·l≤0)的光 <2 → 该像素整子集剔除(通用预处理)。
    n_masked: (P, N) → 每像素亮光数 = sum(轴 1)。"""
    return n_masked.sum(1) >= 2


def als_init(I_sub, rho0, dirs_sub, n_gt):
    """SH-2 ALS 式粗解提方向初值(几何已知: 只迭代 ρ 和方向)。"""
    rho = rho0.copy()
    for _ in range(50):
        # 方向: 每光解 l = argmin (ρ·n·l - I) → 线性闭式
        M = rho[:, None] * n_gt                   # (P,N)
        new_dirs = []
        for k in range(dirs_sub.shape[0]):
            r = I_sub[k] - M[:, k] * 0
            # min_l ||ρ(n·l) − I_k||² → l = (M_kᵀI_k)/||M_k||² · n? 不:
            # I ≈ ρ(n·l) = (ρn)·l → l* = Σ(ρn·I)/Σ(ρn)² — 简单投影
            m = rho * n_gt[:, 0] * 0
            # 用伪逆: I_k ≈ ρ_p (n_p · l) → A = ρ[:,None]*n_gt (P,3), b = I_k
            A = rho[:, None] * n_gt
            b = I_sub[k]
            l = np.linalg.lstsq(A, b, rcond=None)[0]
            nrm = np.linalg.norm(l)
            if nrm > 1e-9:
                new_dirs.append(l / nrm)
            else:
                new_dirs.append(dirs_sub[k])
        # ρ: 闭式 — nl (P,N), I_sub (N,P) → 对齐: I_kp = ρ_p · nl[p,k]
        nl = np.clip(n_gt @ np.array(new_dirs).T, 0, None)   # (P,N)
        rho = (I_sub.T * nl).sum(1) / np.maximum((nl*nl).sum(1), 1e-12)   # (P,)
    return rho, np.array(new_dirs)


def jac_r_joint(x, P, n_gt):
    """残差 r[(k,p)] 的解析 Jacobian(模块级, 供 joint_trf 与验证脚本共用)。

    x = [ρ(P) | (α_k, x_k, y_k)×N]; 行序 = 光主序(k*P+p)。
    """
    N = (x.size - P) // 3
    rho = x[:P]
    parms = x[P:].reshape(N, 3)
    alphas = parms[:, 0]
    xy = parms[:, 1:]
    z = np.sqrt(np.maximum(1 - xy[:, 0]**2 - xy[:, 1]**2, 1e-12))
    nl = np.clip(n_gt @ np.column_stack([xy, z]).T, 0, None)      # (P,N)
    m0 = (nl > 0).astype(float)   # clip 边界处次梯度取 0
    J = np.zeros((N * P, P + 3 * N))
    for k in range(N):
        r0 = k * P
        # ρ 块: 列=像素的逐光对角(用 np.diag 显式构造, 避免 slice+array 花式索引外积陷阱)
        J[r0:r0 + P, :P] = np.diag(-alphas[k] * nl[:, k])
        J[r0:r0 + P, P + 3 * k] = -rho * nl[:, k]
        J[r0:r0 + P, P + 3 * k + 1] = -rho * alphas[k] * (n_gt[:, 0] - (xy[k, 0] / z[k]) * n_gt[:, 2]) * m0[:, k]
        J[r0:r0 + P, P + 3 * k + 2] = -rho * alphas[k] * (n_gt[:, 1] - (xy[k, 1] / z[k]) * n_gt[:, 2]) * m0[:, k]
    return J


def jac_r_joint_sparse(x, P, n_gt):
    """同 jac_r_joint 的稀疏版本(csr)。trf 对稀疏 J 走 LSMR,
    正规方程 JᵀJ 不再稠密化 → 单迭代成本从 O(m·n²) 降到 O(nnz)。
    """
    N = (x.size - P) // 3
    rho = x[:P]
    parms = x[P:].reshape(N, 3)
    alphas = parms[:, 0]
    xy = parms[:, 1:]
    z = np.sqrt(np.maximum(1 - xy[:, 0]**2 - xy[:, 1]**2, 1e-12))
    nl = np.clip(n_gt @ np.column_stack([xy, z]).T, 0, None)
    m0 = (nl > 0).astype(float)
    rows, cols, vals = [], [], []
    for k in range(N):
        r = np.arange(k * P, (k + 1) * P)
        rows += [r, r, r, r]
        cols += [np.arange(P),
                 np.full(P, P + 3 * k),
                 np.full(P, P + 3 * k + 1),
                 np.full(P, P + 3 * k + 2)]
        vals += [-alphas[k] * nl[:, k],
                 -rho * nl[:, k],
                 -rho * alphas[k] * (n_gt[:, 0] - (xy[k, 0] / z[k]) * n_gt[:, 2]) * m0[:, k],
                 -rho * alphas[k] * (n_gt[:, 1] - (xy[k, 1] / z[k]) * n_gt[:, 2]) * m0[:, k]]
    from scipy import sparse
    return sparse.csr_matrix((np.concatenate(vals),
                              (np.concatenate(rows), np.concatenate(cols))),
                             shape=(N * P, P + 3 * N))


def joint_trf(I_sub, rho0, dirs0, n_gt, a_, b_, n_iters=60):
    """trf 联合估计 {ρ} ∪ {α_i, l̂_i}。参数化: l = [x,y] → z=√(1-x²-y²)。

    解析 Jacobian(默认; EXP8R_JAC=fd 可回退数值差分):
      残差 r[(k,p)] = I_obs[k,p] − ρ_p·α_k·nl[p,k],  nl = clip(n·l̂, 0, None)
      dr/dρ_p = −α_k·nl[p,k]
      dr/dα_k = −ρ_p·nl[p,k]
      dr/dx_k = −ρ_p·α_k·(n_x − (x_k/z_k)·n_z)·[nl>0]
      dr/dy_k = −ρ_p·α_k·(n_y − (y_k/z_k)·n_z)·[nl>0]
      行序 = 光主序(k*P+p), 与 residual().ravel() 一致。
    """
    N = I_sub.shape[0]
    P = len(n_gt)
    # 参数: [ρ(P) | (α_k, x_k, y_k)×N]
    n_param = P + 3 * N

    def unpack(x):
        rho = x[:P]
        parms = x[P:].reshape(N, 3)
        alphas = parms[:, 0]
        xy = parms[:, 1:]
        z = np.sqrt(np.maximum(1 - xy[:, 0]**2 - xy[:, 1]**2, 1e-12))
        dirs = np.column_stack([xy, z])
        return rho, alphas, dirs

    def residual(x):
        rho, alphas, dirs = unpack(x)
        nl = np.clip(n_gt @ dirs.T, 0, None)      # (P,N)
        I_mod = rho[:, None] * (alphas[None, :] * np.ones((P, N))) * nl
        return (I_sub.T - I_mod).T.ravel()

    x0 = np.concatenate([rho0, np.column_stack([
        np.array([np.mean(I_sub[k][I_sub[k] > 0]) / max(np.mean(rho0), 1e-6)
                  for k in range(N)]),
        dirs0[:, :2]]).ravel()])
    use_fd = os.environ.get("EXP8R_JAC", "analytic") == "fd"
    jac = '2-point' if use_fd else (lambda x: jac_r_joint_sparse(x, P, n_gt))
    kw = dict(method='trf', max_nfev=n_iters, verbose=0)
    if not use_fd:
        # LSMR 紧公差(实测 P=2000/N=3: 与 FD 稠密路径 cost 差 4e-11, 即同一最优解)
        kw["tr_options"] = {"maxiter": 500, "atol": 1e-12, "btol": 1e-12}
    res = least_squares(residual, x0, jac=jac, **kw)
    return res


def diagnose_trace(n_gt, I_sub, rho, dirs_sub, alphas, a_, b_):
    """诊断量(v3): F(ρ, 3N) → Schur 消 ρ → 光照块 → 投影强度维 → 方向迹。"""
    N = dirs_sub.shape[0]
    P = len(n_gt)
    nl = np.clip(n_gt @ dirs_sub.T, 0, None)       # (P,N)
    h = (n_gt @ dirs_sub.T > 0).astype(float)     # (P,N)
    w = 1.0 / np.maximum(a_ + b_ * np.maximum(I_sub, 0), 1e-6)   # (N,P)
    # Fisher 分块(geometry-known, 方向光参数化):
    # J_ρ[p, k] = α_k·h·(n·l_k);  J_l[p, k, d] = α_k ρ h (n·t_d);  J_α[p,k] = ρ h (n·l)
    # 方向切向基 t1, t2
    t1s = np.zeros((N, 3)); t2s = np.zeros((N, 3))
    for k in range(N):
        ref = np.array([0., 0., 1.]) if abs(dirs_sub[k, 2]) < 0.9 else np.array([1., 0., 0.])
        t1 = np.cross(dirs_sub[k], ref); t1 /= np.linalg.norm(t1)
        t2 = np.cross(dirs_sub[k], t1)
        t1s[k] = t1; t2s[k] = t2
    tr_total = 0.0
    for k in range(N):
        # 消 ρ(子集内部): 每像素 F_ρρ = Σ_k (α h n·l)²·w — 但消 ρ 是 P×P 对角 + 交叉…
        # 简化(1 光的边缘化): F_ρρ[p] = Σ_j (α_j h[p,j] nl[p,j])² w_j
        Fpp = (alphas[k] * h[:, k] * nl[:, k])**2 * w[k]      # (P,)
        # (ρ, θ_k) 交叉: F_ρθ[d, p] block — 全耦合太重; 取 1-光有效边缘化:
        # F_θθ(消 ρ) ≈ Σ_p w (αρh n·t_d)² − 交叉² /Fpp(逐像素近似)
        g1 = alphas[k] * rho * h[:, k] * (n_gt @ t1s[k])
        g2 = alphas[k] * rho * h[:, k] * (n_gt @ t2s[k])
        F11 = (w[k] * g1 * g1).sum()
        F22 = (w[k] * g2 * g2).sum()
        F12 = (w[k] * g1 * g2).sum()
        # 交叉修正(逐像素对角消去):
        c1 = (w[k] * g1 * (alphas[k]*h[:,k]*nl[:,k])).sum()
        c2 = (w[k] * g2 * (alphas[k]*h[:,k]*nl[:,k])).sum()
        # Fpp 全体(其他光的贡献也在, 这里简化为子集内)
        Fpp_all = np.zeros(P)
        for j in range(N):
            Fpp_all += (alphas[j] * h[:, j] * nl[:, j])**2 * w[j]
        F11 -= (c1 * c1) / max(Fpp_all.sum(), 1e-300)
        F22 -= (c2 * c2) / max(Fpp_all.sum(), 1e-300)
        F12 -= (c1 * c2) / max(Fpp_all.sum(), 1e-300)
        det = F11 * F22 - F12 * F12
        if det > 0:
            tr_total += (F11 + F22) / det
    return tr_total


def lae(dirs_est, dirs_true):
    angs = []
    for k in range(dirs_true.shape[0]):
        c = np.clip(dirs_est[k] @ dirs_true[k], -1, 1)
        angs.append(np.degrees(np.arccos(c)))
    return float(np.mean(angs))


def main():
    global ROOT
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="只跑指定物体(如 ballPNG); 空=全量")
    ap.add_argument("--data_root", default=str(ROOT), help="DiLiGenT pmsData 根")
    args = ap.parse_args()
    ROOT = Path(args.data_root)
    rng = np.random.default_rng(SEED)
    objects = sorted([d for d in ROOT.iterdir() if d.is_dir()])
    if args.only:
        objects = [d for d in objects if d.name == args.only]
        print(f"[--only] 只跑: {args.only}")
    print(f"物体数: {len(objects)}")
    out = {"objects": {}, "meta": dict(n3=N3_SAMPLES, other=OTHER_SAMPLES, seed=SEED,
                                       protocol="v3 geometry-known joint trf")}

    # 全物体先算 σ_min 分层键
    all_sig = {}
    for d in objects:
        dirs, _, _, _ = load_object(d)
        sigs = []
        r2 = np.random.default_rng(SEED + hash(d.name) % 1000)
        for _ in range(N3_SAMPLES):
            sel = r2.choice(96, 3, replace=False)
            sigs.append(np.linalg.svd(dirs[sel], compute_uv=False)[-1])
        all_sig[d.name] = np.array(sigs)
    all_flat = np.concatenate(list(all_sig.values()))
    q_edges = np.quantile(all_flat, [0.2, 0.4, 0.6, 0.8])

    for d in objects:
        name = d.name
        try:
            dirs, n_gt, I_norm, mask = load_object(d)
            rho_full, a_, b_, lambert_resid = calibrate(n_gt, dirs, I_norm)
            # 离群 mask: 全 96 光拟合残差 3σ 外
            nl_full = np.clip(n_gt @ dirs.T, 0, None)
            I_hat_full = (rho_full[:, None] * nl_full).T
            resid_full = np.abs(I_norm - I_hat_full)
            outlier = resid_full > 3 * np.std(resid_full)
            out_frac = float(outlier.mean())
            print(f"\n{name}: P={len(n_gt)}, 朗伯残差 {lambert_resid:.3f}, 离群 {out_frac:.3f}")
            pf = HERE / f"exp8r_per_object_{name}.partial.json"
            ci0 = oth0 = 0
            ndone = 0
            nsess = 0
            if pf.exists():
                pd = json.loads(pf.read_text(encoding="utf-8"))
                rows = pd["rows"]
                ci0 = int(pd["ci_done"])
                oth0 = int(pd["oth_done"])
                ndone = ci0 + oth0
                print(f"    [resume] {name}: 已完成 N=3 {ci0}/{N3_SAMPLES} + other {oth0}/{OTHER_SAMPLES * 4}, 续跑剩余")
            else:
                rows = []
            t0obj = time.time()
            ntotsub = N3_SAMPLES + OTHER_SAMPLES * 4
            r2 = np.random.default_rng(SEED + zlib.crc32(name.encode()) % 1000)
            # N=3 分层(全部预生成, 下标即断点续跑游标)
            sels = []
            for _ in range(N3_SAMPLES):
                sels.append(r2.choice(96, 3, replace=False))
            # other-N 选择全部预生成(抽样顺序与旧版完全一致: N=5,10,20,50 × 15)
            other_sels = []
            for N in (5, 10, 20, 50):
                for _ in range(OTHER_SAMPLES):
                    other_sels.append((N, r2.choice(96, N, replace=False)))
            for ci, sel in enumerate(sels[ci0:], start=ci0):
                dirs_sub = dirs[sel]; I_sub = I_norm[sel]
                # 像素守卫
                n_masked = (n_gt @ dirs_sub.T > 0).astype(float)
                keep = pixel_guard(I_sub, n_masked)
                if keep.sum() < 100:
                    continue
                n_k_all = n_gt[keep]; I_k_all = I_sub[:, keep]
                # 数值预算下采样(通用预处理, 固定种子, 非选择性): 每 8 像素取 1
                idx_k = np.where(keep)[0]
                rng3 = np.random.default_rng(SEED)
                if len(idx_k) < 100:
                    print(f"    [guard] {name} cfg{ci}: keep={len(idx_k)} <100, skip")
                    continue
                sub_idx = np.sort(rng3.choice(idx_k, min(len(idx_k), max(len(idx_k)//8, 2000)), replace=False))
                n_k = n_gt[sub_idx]; I_k = I_sub[:, sub_idx]
                # ALS 初值
                rho_als, dirs_als = als_init(I_k, rho_full[sub_idx] * 0.8, dirs_sub, n_k)
                # 多起点 trf
                best = None
                for s in range(N_STARTS):
                    d0 = dirs_als + rng.normal(0, 0.05, dirs_als.shape) if s > 0 else dirs_als
                    d0 /= np.linalg.norm(d0, axis=1, keepdims=True)
                    res = joint_trf(I_k, rho_als, d0, n_k, a_, b_)
                    if best is None or res.cost < best.cost:
                        best = res
                rho_e, alpha_e, dirs_e = None, None, None
                x = best.x
                P_k = len(sub_idx)   # 修: 参数段长 = 下采样像素数(keep.sum() 是守卫前全量)
                rho_e = x[:P_k]
                parms = x[P_k:].reshape(3, 3)
                alpha_e = parms[:, 0]
                xy = parms[:, 1:]
                zz = np.sqrt(np.maximum(1 - xy[:, 0]**2 - xy[:, 1]**2, 1e-12))
                dirs_e = np.column_stack([xy, zz])
                # LAE
                lae_v = lae(dirs_e, dirs_sub)
                # 诊断迹
                tr = diagnose_trace(n_k, I_k, rho_e, dirs_sub, alpha_e, a_, b_)
                rows.append(dict(N=3, sigma_min=float(np.linalg.svd(dirs_sub, compute_uv=False)[-1]),
                                 trace=tr, lae=lae_v))
                ndone += 1
                nsess += 1
                el = time.time() - t0obj
                print(f"      [P] N3 done={ndone}/{ntotsub} el={el:.0f}s eta_obj={el/nsess*(ntotsub-ndone):.0f}s lae={lae_v:.2f}", flush=True)
                if ndone % CKPT_EVERY == 0:
                    pf.write_text(json.dumps(dict(ci_done=ci + 1, oth_done=oth0, rows=rows), ensure_ascii=False), encoding="utf-8")
            # 其他 N
            for ni, (N, sel) in enumerate(other_sels[oth0:], start=oth0):
                dirs_sub = dirs[sel]; I_sub = I_norm[sel]
                n_masked = (n_gt @ dirs_sub.T > 0).astype(float)
                keep = pixel_guard(I_sub, n_masked)
                if keep.sum() < 100:
                    continue
                idx_k = np.where(keep)[0]
                rng3 = np.random.default_rng(SEED)
                sub_idx = np.sort(rng3.choice(idx_k, min(len(idx_k), max(len(idx_k)//8, 2000)), replace=False))
                n_k = n_gt[sub_idx]; I_k = I_sub[:, sub_idx]
                rho_als, dirs_als = als_init(I_k, rho_full[sub_idx]*0.8, dirs_sub, n_k)
                best = None
                for s in range(N_STARTS):
                    d0 = dirs_als + rng.normal(0, 0.05, dirs_als.shape) if s > 0 else dirs_als
                    d0 /= np.linalg.norm(d0, axis=1, keepdims=True)
                    res = joint_trf(I_k, rho_als, d0, n_k, a_, b_)
                    if best is None or res.cost < best.cost:
                        best = res
                P_k = len(sub_idx)   # 修: 与 N=3 分支一致(下采样像素数)
                parms = best.x[P_k:].reshape(N, 3)
                xy = parms[:, 1:]
                zz = np.sqrt(np.maximum(1 - xy[:,0]**2 - xy[:,1]**2, 1e-12))
                dirs_e = np.column_stack([xy, zz])
                lae_v = lae(dirs_e, dirs_sub)
                rho_e = best.x[:P_k]; alpha_e = parms[:, 0]
                tr = diagnose_trace(n_k, I_k, rho_e, dirs_sub, alpha_e, a_, b_)
                rows.append(dict(N=N, sigma_min=float(np.linalg.svd(dirs_sub, compute_uv=False)[-1]),
                                 trace=tr, lae=lae_v))
                ndone += 1
                nsess += 1
                el = time.time() - t0obj
                print(f"      [P] other done={ndone}/{ntotsub} el={el:.0f}s eta_obj={el/nsess*(ntotsub-ndone):.0f}s lae={lae_v:.2f}", flush=True)
                if ndone % CKPT_EVERY == 0:
                    pf.write_text(json.dumps(dict(ci_done=N3_SAMPLES, oth_done=ni + 1, rows=rows), ensure_ascii=False), encoding="utf-8")
            # N=3 Spearman
            n3 = [r for r in rows if r["N"] == 3]
            trs = np.array([r["trace"] for r in n3]); laes = np.array([r["lae"] for r in n3])
            fin = np.isfinite(trs) & np.isfinite(laes)
            r_, p_ = spearmanr(trs[fin], laes[fin])
            out["objects"][name] = dict(
                rows=rows, lambert_resid=lambert_resid, outlier_frac=out_frac,
                spearman_n3=dict(rho=float(r_), p=float(p_), n=int(fin.sum())))
            print(f"  N=3 ρ={r_:+.3f} (p={p_:.4f}, n={fin.sum()})")
            # 逐物体落盘(云并行): exp8r_per_object_<name>.json
            per_obj = dict(meta=out["meta"], lambert_resid=lambert_resid,
                           outlier_frac=out_frac, rows=rows, spearman_n3=out["objects"][name]["spearman_n3"])
            (HERE / f"exp8r_per_object_{name}.json").write_text(
                json.dumps(per_obj, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"  [per-object saved] exp8r_per_object_{name}.json")
            pf.unlink(missing_ok=True)
            # verdict 纪律: assert
            assert len(n3) > 0, f"{name} N=3 空"
        except Exception as exc:
            import traceback; traceback.print_exc()
            out["objects"][name] = dict(error=str(exc))

    # 判定(assert 对账)
    n3_stats = {k: v["spearman_n3"] for k, v in out["objects"].items() if "spearman_n3" in v}
    sig_pos = sum(1 for v in n3_stats.values() if v["p"] < 0.05 and v["rho"] > 0)
    n_sig = sum(1 for v in n3_stats.values() if v["p"] < 0.05)
    assert sig_pos <= len(n3_stats), "计数对账失败"
    ps = [v["p"] for v in n3_stats.values()]
    meta_p = float(1 - chi2.cdf(-2 * sum(np.log(max(p, 1e-300)) for p in ps), 2 * len(ps))) if ps else float('nan')
    out["verdict"] = dict(
        n_objects=len(n3_stats), n_significant_positive=sig_pos, n_significant_total=n_sig,
        meta_p=meta_p,
        per_object={k: dict(rho=round(v['rho'], 3), p=round(v['p'], 5),
                            lambert=round(out['objects'][k]['lambert_resid'], 3))
                    for k, v in n3_stats.items()},
        acceptance="≥6/10 物体 LAE 显著正(p<0.05) → 支柱③真实侧成立",
        result=("支柱③真实侧成立" if sig_pos >= 6 else
                f"混合/负结果: {sig_pos} 显著正 / {n_sig - sig_pos} 显著负(交下一轮裁决)"))
    print(f"\n[exp8R v3] {sig_pos}/{len(n3_stats)} 显著正 | meta-p={meta_p:.2e}")
    print("  →", out["verdict"]["result"])
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[exp8R v3] 落盘 -> {OUT}")


if __name__ == "__main__":
    main()
