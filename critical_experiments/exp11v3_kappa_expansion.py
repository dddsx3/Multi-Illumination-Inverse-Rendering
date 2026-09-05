#!/usr/bin/env python3
"""exp11v3 · κ_weighted 预测力扩展验证(主智能体裁决执行)

背景:exp11v2 发现 cube 场景 κ_weighted(最坏法线切向覆盖的 ρ² 加权版)在双块
一致显著正相关(ρ=0.516/0.518), 但仅 1/4 场景——主智能体裁决: 性能允许时扩展到
更多场景以解决"限定口径"问题。

扩展: data_sun_confirmatory 的 17 个 conf_ 场景(校准协议场景, 32 灯池,
GT 法线/ρ 齐备), 每场景 30 配置 × {κ_weighted, σ_min(C_1)²} 双预测子 ×
双块(条件/Schur)响应量 → 场景内 Spearman → 跨场景分布。

预注册判据(扩展版, 现在写死): 17 场景中 ≥60% 显示 κ_weighted 至少一块
显著正相关(p<0.05)→ "几何退化场景限定口径"升级为"多数场景成立";
30-59% → 维持"限定于几何退化场景"; <30% → cube 信号降级为孤例。
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "p1" / "source" / "information_audit"))

from exp2_joint_fisher_schur import sh2, sh2_d, sobel_sparse  # noqa: E402
from exp11_dispersion_v2 import (  # noqa: E402
    N_LIGHTS, N_CONFIGS, SEED, fisher_theta_blocks, trace_theta_inv,
)

DATA_CONF = REPO / "p1" / "calibration_set" / "data_sun_confirmatory"
OUT = HERE / "exp11v3_kappa_expansion.json"
N_CONFIGS = 20          # 17 场景 × 20 配置(总预算与 exp11v2 同量级)


def scene_conf(scene_dir):
    """conf_ 场景加载(校准确认集)。mask 前置应用(2026-09-05 修: conf_ 数据
    mask 外法线为 NaN, NaN 传播致 SVD 失败——已由重试 1 定位)。"""
    import os
    d = scene_dir
    n_mesh = np.load(os.path.join(d, "normal_mesh.npy")).transpose(1, 2, 0)
    H, W = n_mesh.shape[:2]
    nm = n_mesh.reshape(-1, 3)
    a = np.load(os.path.join(d, "albedo.npy"))[0].ravel()
    m = np.load(os.path.join(d, "mask.npy"))[0].ravel() > 0
    nm = nm[m]; a = a[m]
    nm = nm / np.maximum(np.linalg.norm(nm, axis=1, keepdims=True), 1e-9)
    nm = np.nan_to_num(nm, nan=0.0, posinf=0.0, neginf=0.0)
    ok = np.linalg.norm(nm, axis=1) > 1e-9
    nm, a = nm[ok], a[ok]
    pool = np.load(os.path.join(d, "sh_coeffs_irradiance.npy")).astype(float)
    return nm, a, np.ones(len(nm), bool), pool, H, W


def predictors_full(C, nrm, rho):
    """exp11v2 的三预测子(κ 加权版为主)。"""
    sv = np.linalg.svd(C, compute_uv=False)
    C1 = C[:, 1:4]
    sv1 = np.linalg.svd(C1, compute_uv=False)
    M = C1.T @ C1
    Mn = nrm @ M.T
    nMn = (nrm * Mn).sum(1)
    nn = nrm[:, :, None] * nrm[:, None, :]
    A = M[None, :, :] - (Mn[:, :, None] * nrm[:, None, :]) \
        - (nrm[:, :, None] * Mn[:, None, :]) + (nMn[:, None, None]) * nn
    w = np.linalg.eigvalsh(A)
    return dict(sigma_min_C_sq=float(sv[-1] ** 2),
                sigma_min_C1_sq=float(sv1[-1] ** 2),
                kappa=float(w[:, 0].min()),
                kappa_weighted=float(np.linalg.eigvalsh(
                    np.einsum('pij,p->ij', A, rho ** 2) / max((rho ** 2).sum(), 1e-300))[0]))


def trace_pair(nrm, rho, C):
    """双块响应。修正(2026-09-05): trace_theta_inv_safe 一次调用即返回
    (tr_cond, tr_schur) 二元组——初版二次包装导致 tr_c 变为元组(np.isfinite 歧义)。"""
    return trace_theta_inv_safe(nrm, rho, C)


def trace_theta_inv_safe(nrm, rho, C, cond_cut=1e6):
    """exp11v2 终版的解析迹(含全暗过滤 + 条件数过滤), 独立副本防循环。"""
    Y = sh2(nrm)
    Z = Y @ C.T
    Sk = np.maximum(Z, 0)
    Hk = (Z > 0).astype(float)
    dY = sh2_d(nrm)
    CdY = np.einsum('kj,pji->pki', C, dY)
    P_all = len(nrm)
    N = C.shape[0]
    F_rr_all = (Sk ** 2).sum(1)
    keep0 = F_rr_all > 1e-3 * max(np.median(F_rr_all), 1e-300)
    F_rr = F_rr_all[keep0]
    Sk_k, Hk_k, Y_k, rho_k, nrm_k = Sk[keep0], Hk[keep0], Y[keep0], rho[keep0], nrm[keep0]
    dY_k = dY[keep0]
    P = int(keep0.sum())
    if P < 10:
        return float('nan'), float('nan')
    # 切向基
    ref = np.array([0.0, 0.0, 1.0])
    t1 = np.cross(nrm_k, np.broadcast_to(ref, nrm_k.shape))
    bad = np.linalg.norm(t1, axis=1) < 1e-8
    if bad.any():
        t1[bad] = np.cross(nrm_k[bad], np.array([1.0, 0.0, 0.0]))
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(nrm_k, t1)
    T = np.stack([t1, t2], axis=2)
    CdY_k = np.einsum('kj,pji->pki', C, dY_k)
    G_t = np.einsum('pki,pid->pkd', CdY_k, T) * (rho_k[:, None] * Hk_k)[:, :, None]
    F_tt = np.einsum('pkd,pke->pde', G_t, G_t)
    F_tr = np.einsum('pkd,pk->pd', G_t, Sk_k)
    inv_rr = 1.0 / np.maximum(F_rr, 1e-300)
    S_tt = F_tt - F_tr[:, :, None] * F_tr[:, None, :] * inv_rr[:, None, None]
    # F_tC / F_rC / F_CC
    FtC = (G_t[:, :, :, None] * 0)  # G_t 已含 ρh 权重? 本口径: J_θ 用 ρh·G_t_raw
    Gt_raw = np.einsum('pki,pid->pkd', CdY_k, T)
    F_tC = (Gt_raw[:, :, :, None] * Y_k[:, None, None, :] *
            (rho_k[:, None] * Hk_k)[:, :, None, None]).transpose(0, 2, 3, 1)  # (P,2,9,N)
    F_rC = np.concatenate([(Sk_k[:, k] * rho_k * Hk_k[:, k])[:, None] * Y_k
                           for k in range(N)], axis=1)
    F_rC4 = F_rC.reshape(P, N, 9)
    F_tC_corr = F_tC - F_tr[:, :, None, None] * inv_rr[:, None, None, None] * \
        F_rC4.transpose(0, 2, 1)[:, None, :, :]
    F_CC = np.zeros((9 * N, 9 * N))
    for k in range(N):
        Yk = Y_k * (rho_k * Hk_k[:, k])[:, None]
        F_CC[9*k:9*(k+1), 9*k:9*(k+1)] = Yk.T @ Yk
    F_CC_corr = F_CC - F_rC.T @ (F_rC * inv_rr[:, None])
    # 逐像素 2×2 逆 + 条件数过滤
    det = S_tt[:, 0, 0] * S_tt[:, 1, 1] - S_tt[:, 0, 1] ** 2
    tr2 = S_tt[:, 0, 0] + S_tt[:, 1, 1]
    ok = (det > 1e-12 * np.maximum(tr2 ** 2, 1e-300)) & (tr2 > 0) & np.isfinite(det)
    cond = np.full(P, np.inf)
    cond[ok] = tr2[ok] ** 2 / np.maximum(det[ok], 1e-300)
    keep = ok & (cond < 1e6)
    S_inv = np.zeros((P, 2, 2))
    if keep.any():
        S_inv[keep] = np.linalg.pinv(S_tt[keep])
    tr_cond = float(S_inv.sum())
    # Schur(Woodbury)
    Mt = F_tC_corr.transpose(0, 1, 3, 2).reshape(2 * P, 9 * N)
    keep_full = np.repeat(keep, 2)
    Minv = (S_inv @ Mt.reshape(P, 2, 9 * N)).reshape(2 * P, 9 * N)
    Minv[~keep_full] = 0.0
    Mt_z = Mt.copy(); Mt_z[~keep_full] = 0.0
    W_mid = F_CC_corr - Mt.T @ Minv
    W_mid = (W_mid + W_mid.T) / 2
    Minv2 = (S_inv @ (S_inv @ Mt.reshape(P, 2, 9 * N))).reshape(2 * P, 9 * N)
    Minv2[~keep_full] = 0.0
    B2 = Minv2.T @ Mt_z
    Wm_inv = np.linalg.pinv(W_mid, rcond=1e-10)
    tr_schur = tr_cond + float(np.trace(Wm_inv @ B2))
    return tr_cond, tr_schur


def main():
    scenes = sorted([d for d in DATA_CONF.iterdir() if d.is_dir()
                     and (d / "sh_coeffs_irradiance.npy").exists()])
    rng = np.random.default_rng(SEED)
    out = {"scenes": {}, "meta": dict(n_configs=N_CONFIGS, n_lights=N_LIGHTS,
                                      seed=SEED, data="data_sun_confirmatory",
                                      purpose="κ_weighted 预测力扩展(主智能体裁决执行)")}
    for sd in scenes:
        name = sd.name
        try:
            nrm, rho, m, pool, H, W = scene_conf(sd)
            if len(nrm) < 100:
                print(f"  [skip] {name}: 有效像素不足")
                continue
            # 常亮交集像素集(R-D2)
            Z_all = sh2(nrm) @ pool.T
            lit_all = (np.maximum(Z_all, 0) > 1e-10 * np.abs(Z_all).max()).all(axis=1)
            nrm_u, rho_u = nrm[lit_all], rho[lit_all]
            if int(lit_all.sum()) < 100:
                print(f"  [skip] {name}: 常亮像素不足 ({int(lit_all.sum())})")
                continue
            rows = []
            for ci in range(N_CONFIGS):
                sel = np.sort(rng.choice(len(pool), N_LIGHTS, replace=False))
                C = pool[sel]
                pred = predictors_full(C, nrm_u, rho_u)
                tr_c, tr_s = trace_pair(nrm_u, rho_u, C)
                if not (np.isfinite(tr_c) and np.isfinite(tr_s)):
                    continue
                rows.append(dict(config=ci, **pred, tr_conditional=tr_c, tr_schur=tr_s))
            stats = {}
            for blk, key in (("conditional", "tr_conditional"), ("schur", "tr_schur")):
                resp = [r[key] for r in rows]
                for pname in ("kappa_weighted", "sigma_min_C1_sq", "sigma_min_C_sq"):
                    xs = [r[pname] for r in rows]
                    if np.std(xs) > 0 and np.std(resp) > 0:
                        r_, p_ = spearmanr(xs, resp)
                        stats[f"{blk}:{pname}"] = dict(rho=float(r_), p=float(p_))
            kw_sig = any(v["p"] < 0.05 and v["rho"] > 0
                         for k, v in stats.items() if "kappa_weighted" in k)
            out["scenes"][name] = dict(n_rows=len(rows), stats=stats,
                                       kappa_weighted_positive=bool(kw_sig))
            sigs = {k: f"ρ={v['rho']:.2f}(p={v['p']:.3f})" for k, v in stats.items()
                    if v["p"] < 0.05}
            print(f"{name:24s} n={len(rows):2d} κw显著: {kw_sig} | {sigs}")
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"{name:24s} FAIL: {exc}")
            out["scenes"][name] = dict(error=str(exc))

    # 汇总判定(预注册)
    kw_pos = [s for s, v in out["scenes"].items()
              if v.get("kappa_weighted_positive")]
    n_total = len([v for v in out["scenes"].values() if "kappa_weighted_positive" in v])
    frac = len(kw_pos) / max(n_total, 1)
    out["verdict"] = dict(
        n_scenes=n_total, n_kappa_weighted_positive=len(kw_pos), frac=frac,
        classification=("多数场景成立(≥60%)" if frac >= 0.6 else
                        "限定于几何退化场景(30-59%)" if frac >= 0.3 else
                        "cube 信号降级为孤例(<30%)"),
        kappa_positive_scenes=kw_pos)
    print(f"\n[exp11v3] κ_weighted 正相关场景: {len(kw_pos)}/{n_total} = {frac:.0%}")
    print("  判定:", out["verdict"]["classification"])
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[exp11v3] 落盘 -> {OUT}")


if __name__ == "__main__":
    main()
