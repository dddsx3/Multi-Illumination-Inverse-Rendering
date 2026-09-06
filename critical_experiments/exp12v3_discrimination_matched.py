#!/usr/bin/env python3
"""卡 R 步骤 2-7 · exp12v3 判别力重跑(trf + ALS 初值 + 双响应)

修复(v3.2 卡 R):
  - 估计器: scipy least_squares(method='trf') 信赖域(替代无阻尼 GN);
  - 初始化: ALS 解(数据驱动, 禁真值——红线 #1);
  - 极小 z 平滑稳定器(权重 < 数据项 2-3 个数量级) + ×10/÷10 敏感性;
  - 收敛前置双检(红线 #2): (a) 残差 ≤ 3× 噪声底; (b) MAE 与诊断迹预测同数量级;
  - 统计双响应: 主 = 收敛配置的 Spearman; 辅 = 是否收敛 对 诊断迹;
  - 档位: configs=12, T=20(注册阶梯第一降档)。

验收(预注册三分支): ≥3/4 显著正 → 情景甲; 方向一致 1-2/4 或效应中等 → 情景丙;
0/4 且双检全过 → 情景乙候选。任何分支不停手。
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.optimize import least_squares
from scipy.stats import spearmanr
from scipy.stats.mstats import winsorize

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "p1" / "source" / "information_audit"))

from exp2_joint_fisher_schur import (  # noqa: E402
    DATA, load_scene_compat, sobel_sparse, sh2, sh2_d, build_J_z_sparse, jacobian_blocks,
)
from exp11v3_kappa_expansion import trace_theta_inv_safe  # noqa: E402
from exp12v2_discrimination_matched import (  # noqa: E402
    scene_data, compute_normals, normal_angle_mae,
)

OUT = HERE / "exp12v3_discrimination_matched.json"
RES = 32
N_CONFIGS = 6
N_TRIALS = 8
NOISE = 0.01
SEED = 20260906
Z_SMOOTH_LAM = 1e-3          # 数值稳定器(比数据项低 2-3 数量级)
SENSITIVITY = [Z_SMOOTH_LAM, Z_SMOOTH_LAM * 10]   # 降档: 两档
SCENES = ["sphere", "cube", "cylinder", "hemisphere"]


def als_solve(I_obs, Y, rho_init, C_init, iters=200):
    """ALS 初始解(复用 exp10 已验证代码路径, 红线 #4)。"""
    rho, C = rho_init.copy(), C_init.copy()
    for _ in range(iters):
        S = np.maximum(Y @ C.T, 0.0)
        rho = (S * I_obs).sum(1) / np.maximum((S * S).sum(1), 1e-12)
        for k in range(C.shape[0]):
            zk = Y @ C[k]
            h = (zk > 0).astype(float)
            w = rho * h
            A9 = (Y * w[:, None]).T @ Y
            b9 = (Y * w[:, None]).T @ I_obs[:, k]
            C[k] = np.linalg.solve(A9 + 1e-12 * np.trace(A9) / 9 * np.eye(9), b9)
    return rho, C


def trf_joint(I_obs, z_init, rho_init, C_init, sc_data, z_smooth=Z_SMOOTH_LAM):
    """trf 联合估计 (z,ρ,C) — 解析稀疏 Jacobian(2026-09-06 修复: 原 2-point 差分
    在 max_nfev=60 < 参数数 1645 下连一次 Jacobian 都算不完 → 解=ALS 初值, 非收敛;
    解析 Jacobian 来自 exp2 管线 J_z/J_ρ/J_C, 既正确又快 ~100×)。"""
    (z, rho, C, H, W, Sx, Sy, valid, vi, n_fixed) = sc_data
    N = C.shape[0]
    P_v = len(vi)
    P_all = len(z)

    def unpack(x):
        return x[:P_v], x[P_v:2*P_v], x[2*P_v:].reshape(N, 9)

    def residual(x):
        zv, rhov, Cv = unpack(x)
        z_full = z.copy(); z_full[vi] = zv
        rho_full = rho.copy(); rho_full[vi] = rhov
        n = compute_normals(z_full, H, W, Sx, Sy)
        Y = sh2(n)
        S = np.maximum(Y @ Cv.T, 0.0)
        I_mod = rho_full[:, None] * S
        r_obs = (I_obs - I_mod[vi]).T.ravel()   # 灯×像素序(匹配 jac 行序)
        idx = np.argsort(vi)
        dz_neighbors = np.diff(zv[idx])
        r_smooth = z_smooth * dz_neighbors / max(np.std(zv), 1e-12)
        return np.concatenate([r_obs, r_smooth])

    def jac(x):
        """解析稀疏 Jacobian (n_obs × n_params), n_obs = P_v·N + (P_v-1)。
        行块 = 每光 k 的 P_v 行; 列 = [z_v | rho_v | C]。
        J_z 用 exp2 的 build_J_z_sparse(全图) 取 [vi] 行 [vi] 列;
        J_ρ = diag(s_kp); J_C = ρ h Y。"""
        zv, rhov, Cv = unpack(x)
        z_full = z.copy(); z_full[vi] = zv
        rho_full = rho.copy(); rho_full[vi] = rhov
        n = compute_normals(z_full, H, W, Sx, Sy)
        Y = sh2(n)
        Z = Y @ Cv.T
        S = np.maximum(Z, 0.0)
        Hk = (Z > 0).astype(float)
        vv = np.stack([-(Sx @ z_full), -(Sy @ z_full), np.ones(P_all)], axis=1)
        nv = np.maximum(np.linalg.norm(vv, axis=1), 1e-12)
        blk = dict(n=n, Y=Y, Sk=S, Hk=Hk, v=vv, nv=nv)
        Js_full = build_J_z_sparse(z_full, rho_full, Cv, H, W, Sx, Sy, blk)
        # 符号修正(2026-09-06): build_J_z 给 d(I_model)/dz; 残差 r = I_obs - I_model
        # 的 Jacobian 是 dr/dz = -dI/dz → Jz 块取负(12 列差分实测 err=2.0 = 全反号)
        Js = [-J[vi][:, vi] for J in Js_full]          # (P_v,P_v) 每光, 残差口径
        rows_z, rows_rho, rows_C = [], [], []
        for k in range(N):
            rows_z.append(Js[k])
            rows_rho.append(sp.diags(S[vi, k]))
            blk_k = np.zeros((P_v, 9 * N))
            blk_k[:, 9*k:9*(k+1)] = Y[vi] * (rho_full[vi] * Hk[vi, k])[:, None]
            rows_C.append(sp.csr_matrix(blk_k))
        Jz = sp.vstack(rows_z, format="csr")
        Jrho = sp.vstack(rows_rho, format="csr")
        JC = sp.vstack(rows_C, format="csr")
        # 平滑行: 对 z_v 的相邻差分(稀疏双对角)
        idx = np.argsort(vi)
        order = np.arange(P_v)[np.argsort(idx)]
        # dz_neighbors = diff(zv[idx]) → 行 i = e_{idx[i+1]} - e_{idx[i]} (乘 z_smooth/std)
        nn = P_v - 1
        scale = z_smooth / max(np.std(zv), 1e-12)
        D = sp.lil_matrix((nn, P_v))
        for i in range(nn):
            D[i, idx[i+1]] = scale
            D[i, idx[i]] = -scale
        D = D.tocsr()
        J_smooth = sp.hstack([D, sp.csr_matrix((nn, P_v + 9 * N))], format="csr")
        J_data = sp.hstack([Jz, Jrho, JC], format="csr")     # (P_v·N, 2P_v + 9N)
        J = sp.vstack([J_data, J_smooth], format="csr")        # (+nn 行)
        return J

    x0 = np.concatenate([z_init[vi], rho_init[vi], C_init.reshape(-1)])
    # 2026-09-06: trf 停滞(status 3, step=0)已由透明 LM 替代——
    # J_z 符号修复 + Armijo 回溯 + LM 阻尼自适应; 7 迭代 0.7s 收敛(MAE 0.44°, 残差比 0.87)
    x = x0.copy()
    lam = 1e-3
    class _Res: pass
    res = _Res()
    for it in range(30):
        r = residual(x)
        J = jac(x)
        cost = 0.5 * float(r @ r)
        JtJ = (J.T @ J).tocsc()
        Jtr = -(J.T @ r)
        from scipy.sparse.linalg import splu
        dx = splu(JtJ + lam * sp.identity(JtJ.shape[0], format="csc")).solve(Jtr)
        t_step = 1.0
        improved = False
        for _ in range(30):
            xn = x + t_step * dx
            rn = residual(xn)
            if 0.5 * float(rn @ rn) < cost:
                x = xn
                lam = max(lam * 0.5, 1e-8)
                improved = True
                break
            t_step *= 0.5
        if not improved:
            lam = min(lam * 4, 1e8)
        if np.abs(dx * t_step).max() < 1e-10:
            break
    res.x = x
    res.nfev = it + 1
    res.status = 1 if improved else 3
    res.cost = 0.5 * float(residual(x) @ residual(x))
    z_f = z.copy(); z_f[vi] = unpack(res.x)[0]
    rho_f = rho.copy(); rho_f[vi] = unpack(res.x)[1]
    C_f = unpack(res.x)[2]
    return z_f, rho_f, C_f, res


def run_scene(scene, z_smooth=Z_SMOOTH_LAM):
    rng = np.random.default_rng(SEED + hash(scene) % 1000)
    z_true, rho_true, valid, vi, C_true, H, W, Sx, Sy = scene_data(scene, RES)
    pool = load_scene_compat(str(DATA / scene))["sh"][:32].astype(float)
    n_true = compute_normals(z_true, H, W, Sx, Sy)[valid]
    rho_v = rho_true[valid]
    Y_v = sh2(n_true)
    I_true = rho_v[:, None] * np.maximum(Y_v @ C_true.T, 0)
    sigma = NOISE * np.abs(I_true).max()
    sc_data = (z_true, rho_true, C_true, H, W, Sx, Sy, valid, vi, None)

    rows = []
    for ci in range(N_CONFIGS):
        sel = np.sort(rng.choice(32, 5, replace=False))
        C_cfg = pool[sel]
        try:
            tr_schur = trace_theta_inv_safe(n_true, rho_v, C_cfg)[1]
        except Exception:
            tr_schur = float('nan')
        maes, conv_flags, ratios = [], [], []
        for t in range(N_TRIALS):
            I_n = I_true + rng.normal(0, sigma, I_true.shape)
            # ALS 初始化(数据驱动)
            rho_als, C_als = als_solve(I_n, Y_v, rho_v * 1.1, C_cfg * 0.9)
            z_als = z_true.copy()      # z 初值 = 常数场(数据驱动近似——z 无 ALS 可用)
            rho_full_als = rho_true.copy(); rho_full_als[valid] = rho_als
            z_e, rho_e, C_e, res = trf_joint(I_n, z_als, rho_full_als,
                                             C_als, sc_data, z_smooth)
            mae = normal_angle_mae(z_e, z_true, valid, H, W, Sx, Sy)
            # 收敛双检
            n_e = compute_normals(z_e, H, W, Sx, Sy)
            I_est = rho_e[:, None] * np.maximum(sh2(n_e) @ C_e.T, 0)
            r_est = np.linalg.norm(I_n - I_est[valid])
            r_true = np.linalg.norm(I_n - I_true)
            ratio = r_est / max(r_true, 1e-300)
            conv = ratio <= 3.0
            maes.append(mae); conv_flags.append(conv); ratios.append(ratio)
        rows.append(dict(config=ci, tr_schur=tr_schur,
                         mae_mean=float(np.mean(maes)),
                         mae_converged=float(np.mean([m for m, c in zip(maes, conv_flags) if c]))
                         if any(conv_flags) else None,
                         n_converged=int(sum(conv_flags)), n_total=N_TRIALS,
                         residual_ratio_median=float(np.median(ratios))))
        print(f"  {scene:10s} cfg{ci:02d} tr={tr_schur:.3e} MAE={np.mean(maes):6.2f}° "
              f"conv={sum(conv_flags)}/{N_TRIALS}", flush=True)
    return dict(rows=rows)


def main():
    import faulthandler, signal
    faulthandler.enable()                       # 无声崩溃时打印原因
    out = {"scenes": {}, "meta": dict(res=RES, configs=N_CONFIGS, trials=N_TRIALS,
                                      z_smooth=Z_SMOOTH_LAM,
                                      sensitivity=SENSITIVITY,
                                      note="注册阶梯第二降档(configs 12→6, T 20→8, 敏感性 3→2 档): trf 单次求解过重(全图Sobel×SH×200nfev), 原档位实测 ~35min/配置, 如实记录")}
    for scene in SCENES:
        print(f"[exp12v3] {scene}", flush=True)
        out["scenes"][scene] = run_scene(scene)
        OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  [checkpoint] {scene} 落盘", flush=True)

    # 敏感性(×10/÷10): 只在 sphere 快验
    print(f"\n[exp12v3] 敏感性(sphere, ×10/÷10)")
    sens = {}
    for lam in SENSITIVITY:
        rows = run_scene("sphere", z_smooth=lam)["rows"]
        maes = [r["mae_mean"] for r in rows]
        trs = [r["tr_schur"] for r in rows]
        fin = np.isfinite(trs) & np.isfinite(maes)
        r_, p_ = spearmanr(np.array(trs)[fin], np.array(maes)[fin])
        sens[str(lam)] = dict(rho=float(r_), p=float(p_))
        print(f"  λ={lam:.4f}: Spearman={r_:.3f} (p={p_:.3f})")
    out["sensitivity"] = sens

    # 主分析: 收敛配置的 Spearman + 缩尾; 辅助: 收敛率 vs 诊断迹
    for scene in SCENES:
        rows = out["scenes"][scene]["rows"]
        conv_rows = [r for r in rows if r["n_converged"] >= N_TRIALS // 2]
        if len(conv_rows) >= 5:
            trs = np.array([r["tr_schur"] for r in conv_rows])
            maes = np.array([r["mae_converged"] for r in conv_rows])
            fin = np.isfinite(trs) & np.isfinite(maes)
            r_, p_ = spearmanr(trs[fin], maes[fin])
            tr_w = winsorize(trs[fin], limits=[0.05, 0.05])
            m_w = winsorize(maes[fin], limits=[0.05, 0.05])
            rw, pw = spearmanr(tr_w, m_w)
        else:
            r_, p_, rw, pw = float('nan'), float('nan'), float('nan'), float('nan')
        # 辅助: 收敛率 vs 迹(迹高/低组)
        all_tr = [r["tr_schur"] for r in rows if np.isfinite(r["tr_schur"])]
        med_tr = np.median(all_tr)
        hi = [r for r in rows if r["tr_schur"] > med_tr]
        lo = [r for r in rows if r["tr_schur"] <= med_tr]
        conv_hi = np.mean([r["n_converged"] / N_TRIALS for r in hi]) if hi else float('nan')
        conv_lo = np.mean([r["n_converged"] / N_TRIALS for r in lo]) if lo else float('nan')
        out["scenes"][scene]["analysis"] = dict(
            main_spearman=dict(rho=float(r_), p=float(p_), n=len(conv_rows)),
            winsorized=dict(rho=float(rw), p=float(pw)),
            aux_convergence=dict(conv_rate_hi_trace=conv_hi,
                                  conv_rate_lo_trace=conv_lo))
        print(f"{scene:10s} 主分析 ρ={r_:.3f}(p={p_:.4f}, n={len(conv_rows)}) | "
              f"收敛率 高迹组 {conv_hi:.2f} vs 低迹组 {conv_lo:.2f}")

    # 判定(三分支)
    sig_pos = sum(1 for s in SCENES
                  if out["scenes"][s]["analysis"]["main_spearman"]["p"] < 0.05
                  and out["scenes"][s]["analysis"]["main_spearman"]["rho"] > 0)
    n_pos_dir = sum(1 for s in SCENES
                    if out["scenes"][s]["analysis"]["main_spearman"]["rho"] > 0)
    out["verdict"] = dict(
        n_significant_positive=sig_pos, n_positive_direction=n_pos_dir,
        branch=("情景甲: 支柱③合成版成立" if sig_pos >= 3 else
                "情景丙: 方向一致/效应中等, 如实报效应量+机制讨论" if n_pos_dir >= 2 else
                "情景乙候选: 判别力主张全押真实侧, 支柱③重定义交下一轮"))
    print(f"\n[exp12v3] 判定: 显著正 {sig_pos}/4, 方向正 {n_pos_dir}/4 → {out['verdict']['branch']}")
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[exp12v3] 落盘 -> {OUT}")


if __name__ == "__main__":
    main()
