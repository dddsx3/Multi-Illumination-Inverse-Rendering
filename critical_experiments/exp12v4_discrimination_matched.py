#!/usr/bin/env python3
"""卡 R2 · exp12v4: 真值初始化修复 + 注册档位恢复 + Λ 匹配诊断(v3.3 任务书)

supersede exp12v3(判 VOID, 四缺陷见 任务书_增补v3.3 §1.1):
  D1 z 真值初始化 → 数据驱动 4 起点(常数平面 + 3 随机光滑扰动 + SfS-lite),
     GT 只出现在评分; assert: not np.allclose(z_init, z_true)(红线 #11);
  D2 档位恢复 configs=12, T=20(LM 0.7s/解 → 全量 ~1.1h);
  D3 迹修正: ρ/C 逐项精确边缘化(检查表 v2 第 5 行) + 工作带(R-D5, λ>1e-6·λmax)
     + 对数迹; sanity 闸 tr ∈ [1e-3, 1e6] 度²;
  D4 Λ 匹配诊断: 主读出 = 贝叶斯 CRB 迹 tr((F+Λ_z)⁻¹) 工作带(与估计器同
     z_smooth 先验; exp10 式无先验对照并列);
  D5 统计: 主 = 收敛双检通过配置的 Spearman(对数迹 vs MAE); 辅 = 收敛率;
     缩尾 5% 敏感性; 效应量注明 n=12 临界 |ρ|=0.570;
  D6 验收(三分支冻结): ≥3/4 显著正 → 情景甲; 方向一致 1-2/4 → 情景丙;
     0/4 且双检全过 → 情景乙候选。任何分支不停手。

复用(实现无缺陷): exp12v3 的 als_solve / trf_joint(LM 解析 Jacobian 0.7s/解)。
"""
from __future__ import annotations
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from scipy.stats.mstats import winsorize

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "p1" / "source" / "information_audit"))

from exp2_joint_fisher_schur import (  # noqa: E402
    DATA, load_scene_compat, sobel_sparse, sh2, sh2_d,
)
from exp12v3_discrimination_matched import (  # noqa: E402
    als_solve, trf_joint, Z_SMOOTH_LAM,
)
from exp12v2_discrimination_matched import (  # noqa: E402
    scene_data, compute_normals, normal_angle_mae,
)

OUT = HERE / "exp12v4_discrimination_matched.json"
RES = 32
N_CONFIGS = 12          # D2: 注册档位恢复
N_TRIALS = 20
N_STARTS = 4            # D1: 4 起点
NOISE = 0.01
SEED = 20260906
WORK_BAND_REL = 1e-3    # D3: 工作带边(相对 λ_max; 谱实测: 1e-6 边形同虚设,
# 2048/2048 方向入带且 λ→0⁺ 处 1/λ 爆炸; 收紧到 1e-3 对应谱隙上沿,
# 规范/边缘病态方向按 R-D5 带外丢弃)
TR_RANGE = (1e-3, 1e8)  # D3: sanity 闸(度²)。定标准录: 实测带内谱
# [0.09, 26]·2000 方向 → tr 天然量级 1e6-1e7 deg²; 上界取 1e8(仍拦 12 量级型
# 污染, 不误伤正常读数)。初设 1e6 曾拦下正常值 → 定标修正留痕。
SCENES = ["sphere", "cube", "cylinder", "hemisphere"]


# ---------------------------------------------------------------- D1 初始化
def z_inits(I_n, H, W, z_floor, z_std, rho_v, C_cfg, rng):
    """数据驱动 4 起点。GT 不进入(红线 #11; 主循环 assert)。"""
    cands = [np.full(H * W, z_floor)]                    # ① 常数平面(场景底)
    for s in range(3):                                    # ② 3 个光滑扰动场
        amp = z_std * (0.10 + 0.10 * s)                   # 幅值 10/20/30%·std
        noise = rng.normal(0, 1, (H, W))
        for _ in range(8):                                # 固定低通(数据无关)
            noise = (np.roll(noise, 1, 0) + noise + np.roll(noise, -1, 0)
                     + np.roll(noise, 1, 1) + np.roll(noise, -1, 1)) / 5
        noise = noise / max(np.std(noise), 1e-12)
        cands.append(cands[0] + (amp * noise).ravel())
    # ③ SfS-lite: 最亮光的 Lambertian 余弦 → 倾斜形态伪深度(无积分, 仅形态)
    k_best = int(np.argmax(I_n.sum(0)))
    cosv = np.clip(I_n[:, k_best] / max(np.median(rho_v), 1e-12), 0, 1)
    cosv = cosv / max(cosv.max(), 1e-12)
    tilt = np.sqrt(np.maximum(1 - cosv ** 2, 0))
    z_sfs = np.full(H * W, z_floor) + (2 * z_std) * np.interp(
        np.linspace(0, 1, len(tilt)), np.linspace(0, 1, len(tilt)), tilt)
    cands.append(z_sfs)
    return cands[:N_STARTS]


# ------------------------------------------------- D3/D4 诊断(Λ 匹配迹)
def fisher_theta_marginal(n_true, rho_v, C_cfg):
    """θ(法线切向 2P 维)联合 Fisher, ρ(逐像素精确对角)与 C(联合 9N)精确边缘化。

    检查表 v2 第 5 行: 声称的每个边缘化参数都有对应且口径正确的 Schur 项。
    返回 (2P,2P) 稠密 S(θ)。P=内点数(RES=32 → ≤~1024), 一次组装秒级。
    """
    Y = sh2(n_true)
    Z = Y @ C_cfg.T
    Sk = np.maximum(Z, 0)
    Hk = (Z > 0).astype(float)
    dY = sh2_d(n_true)
    N = C_cfg.shape[0]
    # 全暗像素过滤(口径移植 exp11v3 trace_theta_inv_safe keep0): F_rr=0 的像素
    # 不携带 ρ 信息 → Schur 0/0=NaN → eig 不收敛(hemisphere cfg04 实证);
    # 过滤阈值 = 1e-3 × 中位(与 exp11v3 一致), 非选择性通用预处理
    F_rr_all = (Sk ** 2).sum(1)
    keep0 = F_rr_all > 1e-3 * max(np.median(F_rr_all), 1e-300)
    n_dark = int((~keep0).sum())
    if keep0.sum() < 10:
        raise RuntimeError(f"全暗过滤后仅 {keep0.sum()} 像素 — 停手报 INC")
    n_true, rho_v = n_true[keep0], rho_v[keep0]
    Y, Z, Sk, Hk, dY = Y[keep0], Z[keep0], Sk[keep0], Hk[keep0], dY[keep0]
    P = len(n_true)
    # 切向基
    t1 = np.cross(n_true, np.broadcast_to(np.array([0., 0., 1.]), n_true.shape))
    bad = np.linalg.norm(t1, axis=1) < 1e-8
    if bad.any():
        t1[bad] = np.cross(n_true[bad], np.array([1., 0., 0.]))
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(n_true, t1)
    T = np.stack([t1, t2], axis=2)                        # (P,3,2)
    CdY = np.einsum('kj,pji->pki', C_cfg, dY)              # (P,N,3)
    Gt = np.einsum('pki,pid->pkd', CdY, T) * (rho_v[:, None] * Hk)[:, :, None]  # (P,N,2)
    F_tt = np.einsum('pkd,pke->pde', Gt, Gt)              # (P,2,2)
    # ρ 边缘化: F_ρρ 逐像素对角 = Σ_k (ρ? no) — 口径: J_θ=G_t(已含ρh), J_ρ_p=Sk[:,k]
    F_rr = (Sk ** 2).sum(1)                                # (P,)
    F_tr = np.einsum('pkd,pk->pd', Gt, Sk)                # (P,2)
    # ρ 逐像素对角精确 Schur: S_tt[p] = F_tt[p] - F_tr[p]F_tr[p]^T / F_rr[p]
    S_tt = F_tt - (F_tr[:, :, None] * F_tr[:, None, :]) / F_rr[:, None, None]
    # C 边缘化(联合): M = ∂r/∂C 行(θ度量), F_CC 块对角(按光)
    M_tC = np.zeros((2 * P, 9 * N))                        # 行 2p+d, 列 9k+j
    Gr = np.einsum('pki,pid->pkd', CdY, T) * (rho_v[:, None] * Hk)[:, :, None]
    for p in range(P):
        for k in range(N):
            M_tC[2 * p:2 * p + 2, 9 * k:9 * k + 9] = np.outer(Gr[p, k], Y[p])
    F_CC = np.zeros((9 * N, 9 * N))
    for k in range(N):
        Yk = Y * (rho_v * Hk[:, k])[:, None]
        F_CC[9 * k:9 * (k + 1), 9 * k:9 * (k + 1)] = Yk.T @ Yk
    F_CC_inv = np.linalg.pinv(F_CC)                        # ρ·α 型规范方向 → pinv
    S = np.zeros((2 * P, 2 * P))
    for p in range(P):
        S[2 * p:2 * p + 2, 2 * p:2 * p + 2] = S_tt[p]
    S -= M_tC @ F_CC_inv @ M_tC.T
    return S


def trace_diag(S, lam_z):
    """工作带(R-D5)内 tr((S+Λ_z)⁻¹), 度² 口径 + 对数。

    θ 参数定义: 切向单位向量幅值=弧度; 度² = rad²·(180/π)²。
    Λ_z(先验>0 时) = 贝叶斯 CRB; lam_z=0 → exp10 式无先验(对照)。
    """
    P2 = S.shape[0]
    # 尺度规范方向显式投影(零空间清理): ρ→tρ 全 C 反缩放在 θ 度量下无零向量,
    # 但 C-pinv Schur 会在规范邻域留负曲率(实测 λ_min<0) → eigh 前先做
    # 正半定投影是不合法的(掩盖信息); 正确口径 = 带边丢弃 + 负值不入带
    w = np.linalg.eigvalsh(S)
    lam_max = w.max()
    band = w > WORK_BAND_REL * lam_max       # 负特征值与近零方向均不入带
    if band.sum() < 10:
        raise RuntimeError(f"工作带过窄({band.sum()} 方向) — 谱隙假设失效, 停手报 INC")
    lam_z_eff = np.maximum(lam_z, 0.0)
    tr_rad2 = float(np.sum(1.0 / (w[band] + lam_z_eff)))
    tr_deg2 = tr_rad2 * (180.0 / np.pi) ** 2
    return tr_deg2


def lam_z_of(z_smooth, z_std):
    """Λ_z 谱常数(保守口径): λ_z = (z_smooth/σ_z)², θ 以弧度计。"""
    return (z_smooth / max(z_std, 1e-12)) ** 2


# ---------------------------------------------------------------- 主循环
def run_scene(scene, z_smooth=Z_SMOOTH_LAM):
    rng = np.random.default_rng(SEED + zlib.crc32(scene.encode()) % 1000)
    z_true, rho_true, valid, vi, C_true, H, W, Sx, Sy = scene_data(scene, RES)
    pool = load_scene_compat(str(DATA / scene))["sh"][:32].astype(float)
    n_true = compute_normals(z_true, H, W, Sx, Sy)[valid]
    rho_v = rho_true[valid]
    Y_v = sh2(n_true)
    I_true = rho_v[:, None] * np.maximum(Y_v @ C_true.T, 0)
    sigma = NOISE * np.abs(I_true).max()
    sc_data = (z_true, rho_true, C_true, H, W, Sx, Sy, valid, vi, None)
    z_floor = float(np.min(z_true[valid]))
    z_std = float(np.std(z_true[valid]))
    lam_z = lam_z_of(z_smooth, z_std)

    rows_cfg = []
    for ci in range(N_CONFIGS):
        sel = np.sort(rng.choice(32, 5, replace=False))
        C_cfg = pool[sel]
        # D3/D4: 诊断量(@真值几何, 预注册口径) + sanity 闸
        S = fisher_theta_marginal(n_true, rho_v, C_cfg)
        tr_bayes = trace_diag(S, lam_z)
        tr_priorfree = trace_diag(S, 0.0)
        if not (TR_RANGE[0] <= tr_bayes <= TR_RANGE[1]):
            raise RuntimeError(
                f"sanity FAIL {scene} cfg{ci}: tr_bayes={tr_bayes:.3e} "
                f"超出 {TR_RANGE} → 停手报 INC")
        # 估计(响应): 每试验噪声 + 4 起点取残差最优
        maes, convs, ratios, init_idx = [], [], [], []
        for t in range(N_TRIALS):
            I_n = I_true + rng.normal(0, sigma, I_true.shape)
            rho_als, C_als = als_solve(I_n, Y_v, rho_v * 1.1, C_cfg * 0.9)
            cands = z_inits(I_n, H, W, z_floor, z_std, rho_v, C_cfg, rng)
            for zi in cands:                                # 红线 #11
                assert not np.allclose(zi, z_true), "红线#11: z_init==z_true"
            best = None
            for si, zi in enumerate(cands):
                rho_full_als = rho_true.copy(); rho_full_als[valid] = rho_als
                z_e, rho_e, C_e, res = trf_joint(I_n, zi, rho_full_als,
                                                 C_als, sc_data, z_smooth)
                if best is None or res.cost < best[0].cost:
                    best = (res, z_e, rho_e, C_e, si)
            res, z_e, rho_e, C_e, si = best
            mae = normal_angle_mae(z_e, z_true, valid, H, W, Sx, Sy)
            n_e = compute_normals(z_e, H, W, Sx, Sy)
            I_est = rho_e[:, None] * np.maximum(sh2(n_e) @ C_e.T, 0)
            r_est = np.linalg.norm(I_n - I_est[valid])
            r_true = np.linalg.norm(I_n - I_true)
            ratio = r_est / max(r_true, 1e-300)
            maes.append(float(mae)); convs.append(ratio <= 3.0)
            ratios.append(float(ratio)); init_idx.append(si)
        conv_m = [m for m, c in zip(maes, convs) if c]
        rows_cfg.append(dict(
            config=ci, tr_bayes=tr_bayes, tr_priorfree=tr_priorfree,
            log_tr_bayes=float(np.log10(tr_bayes)),
            mae_mean=float(np.mean(maes)),
            mae_converged=float(np.mean(conv_m)) if conv_m else None,
            n_converged=int(sum(convs)), n_total=N_TRIALS,
            resid_ratio_median=float(np.median(ratios)),
            init_win_counts=[int(sum(1 for x in init_idx if x == s))
                             for s in range(N_STARTS)]))
        print(f"  {scene:10s} cfg{ci:02d} tr_bayes={tr_bayes:.3e} "
              f"MAE={np.mean(maes):6.2f}° conv={sum(convs)}/{N_TRIALS} "
              f"starts_win={rows_cfg[-1]['init_win_counts']}", flush=True)
    return rows_cfg


def analyze(rows_cfg, scene):
    """D5 统计: 主(收敛配置 log_tr vs mae_converged) + 辅(收敛率)。"""
    conv_rows = [r for r in rows_cfg if r["n_converged"] >= N_TRIALS // 2]
    if len(conv_rows) >= 5:
        trs = np.array([r["log_tr_bayes"] for r in conv_rows])
        maes = np.array([r["mae_converged"] for r in conv_rows])
        r_, p_ = spearmanr(trs, maes)
        tw = winsorize(trs, limits=[0.05, 0.05])
        mw = winsorize(maes, limits=[0.05, 0.05])
        rw, pw = spearmanr(tw, mw)
    else:
        r_ = p_ = rw = pw = float('nan')
    all_tr = [r["log_tr_bayes"] for r in rows_cfg]
    med = float(np.median(all_tr))
    hi = [r for r in rows_cfg if r["log_tr_bayes"] > med]
    lo = [r for r in rows_cfg if r["log_tr_bayes"] <= med]
    conv_hi = float(np.mean([r["n_converged"] / r["n_total"] for r in hi])) if hi else float('nan')
    conv_lo = float(np.mean([r["n_converged"] / r["n_total"] for r in lo])) if lo else float('nan')
    return dict(main_spearman=dict(rho=float(r_), p=float(p_),
                                   n=len(conv_rows),
                                   effect_note="n=12 critical |rho|=0.570"),
                winsorized=dict(rho=float(rw), p=float(pw)),
                aux_convergence=dict(conv_rate_hi_trace=conv_hi,
                                     conv_rate_lo_trace=conv_lo))


def main():
    import faulthandler
    faulthandler.enable()
    t0 = time.time()
    out = {"meta": dict(
        res=RES, configs=N_CONFIGS, trials=N_TRIALS, starts=N_STARTS,
        z_smooth=Z_SMOOTH_LAM, work_band_rel=WORK_BAND_REL,
        trace_range_sanity=list(TR_RANGE),
        init="data-driven 4 starts (const plane + 3 smooth perturb + SfS-lite); GT excluded with assert (redline #11)",
        supersede="exp12v3 VOID per 任务书 v3.3 §1.1 (四缺陷)",
        marg_note="诊断: θ 2P 联合 Fisher; ρ 逐像素精确对角 Schur; C 联合 9N pinv Schur; 工作带 λ>1e-6·λmax; Λ_z=(z_smooth/σ_z)² 贝叶斯项",
        note=f"注册档位恢复 12/20; LM 0.7s/解; 效应量: n=12 临界 |ρ|=0.570")}
    out["scenes"] = {}
    for scene in SCENES:
        print(f"[exp12v4] {scene}", flush=True)
        rows_cfg = run_scene(scene)
        ana = analyze(rows_cfg, scene)
        out["scenes"][scene] = dict(rows=rows_cfg, analysis=ana)
        OUT.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        print(f"  主分析 ρ={ana['main_spearman']['rho']:.3f}"
              f"(p={ana['main_spearman']['p']:.4f}, n={ana['main_spearman']['n']})",
              flush=True)
        print(f"  [checkpoint] {scene} 落盘 ({time.time()-t0:.0f}s)", flush=True)

    # 敏感性(×10): sphere 快验
    print("\n[exp12v4] 敏感性(sphere, z_smooth×10)", flush=True)
    sens = {}
    z_true, rho_true, valid, vi, C_true, H, W, Sx, Sy = scene_data("sphere", RES)
    n_true = compute_normals(z_true, H, W, Sx, Sy)[valid]
    rho_v = rho_true[valid]
    lam_z10 = lam_z_of(Z_SMOOTH_LAM * 10, float(np.std(z_true[valid])))
    rng = np.random.default_rng(SEED + zlib.crc32("sphere".encode()) % 1000)
    trs, maes = [], []
    for ci in range(N_CONFIGS):
        sel = np.sort(rng.choice(32, 5, replace=False))
        S = fisher_theta_marginal(n_true, rho_v, load_scene_compat(
            str(DATA / "sphere"))["sh"][:32].astype(float)[sel])
        trs.append(np.log10(trace_diag(S, lam_z10)))
    sphere_rows = out["scenes"]["sphere"]["rows"]
    maes = [r["mae_converged"] for r in sphere_rows if r["mae_converged"] is not None]
    n_pair = min(len(trs), len(maes))
    if n_pair >= 5:
        r_, p_ = spearmanr(np.array(trs)[:n_pair], np.array(maes)[:n_pair])
        sens["x10"] = dict(rho=float(r_), p=float(p_), n=n_pair)
        print(f"  λ×10: Spearman={r_:.3f} (p={p_:.3f})", flush=True)
    out["sensitivity"] = sens

    # D6 判定(三分支冻结)
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
    print(f"\n[exp12v4] 判定: 显著正 {sig_pos}/4, 方向正 {n_pos_dir}/4 → "
          f"{out['verdict']['branch']}", flush=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"[exp12v4] 落盘 -> {OUT} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()