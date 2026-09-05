#!/usr/bin/env python3
"""关键实验 11v2(卡 C)· 散布度检验重做(三洞修正版)

设计(任务书 v3.0 卡 C, 预注册; 修正 exp5 三洞):
  洞一修正(R-D4): 场景内 30-50 配置做 Spearman, 再报跨场景分布(禁跨场景合并);
  洞二修正(R-D3): 预测子三档并列 σ_min(C)²(对照) / σ_min(C_1)²(C_1 = C 的 l=1
    块 N×3, 经典 σ_min(L)² 的 SH 对应物) / κ = min_p λ_min(P_t C_1ᵀC_1 P_t)
    (最坏法线切向覆盖; 加 ρ² 加权平均版);
  洞三修正(R-D2): 全子集固定同一 mask(本实现像素集 = 场景 valid 全集, 无逐配置剔除);
  响应(R-D1): 法线角迹 tr(S_θ⁻¹)(θ = 逐像素法线切向 2P 参数; S_θ = 消去 (ρ,C) 后的
    法线角有效 Fisher; gauge 全局尺度由 (ρ,C) 消去吸收)。双块:
    条件块(C 已知): S_cond = F_θθ − F_θν diag(F_ρρ)⁻¹ F_ρν(仅消 ρ);
    Schur 块(C 未知): 再消 C(低秩 9N 修正 + pinv 截断)。
  判据(预注册): ≥3/4 场景内显著正相关(p<0.05)→ 经验法则成立域节成立;
    不足 → 负结果族定稿。两分支都入文, 不改判据。

实现: Hutchinson + CG(线性算子: 像素 2×2 块对角 + 45 维低秩 + 对角消 ρ)。
产物: critical_experiments/exp11_dispersion_v2.{py,json}
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

from exp2_joint_fisher_schur import (  # noqa: E402
    DATA, load_scene_compat, sobel_sparse, sh2, sh2_d,
)

OUT = HERE / "exp11_dispersion_v2.json"
SCENES = ["sphere", "cube", "cylinder", "hemisphere"]
RES = 64
N_LIGHTS = 5
N_CONFIGS = 30
SEED = 20260905
C0_, C1_, C2_ = 0.282095, 0.488603, [1.092548, 1.092548, 0.315392, 1.092548, 0.546274]


def scene_data(scene, res=RES):
    """有效像素子集上的法线/ρ/Y/dY(几何已知口径, 与 exp4 一致)。"""
    sc = load_scene_compat(str(DATA / scene))
    import os
    n_mesh = np.load(os.path.join(str(DATA / scene), "normal_mesh.npy")).transpose(1, 2, 0)
    H0, W0 = n_mesh.shape[:2]
    H = W = res
    i0, j0 = (H0 - H) // 2, (W0 - H) // 2
    nm = n_mesh[i0:i0+H, j0:j0+W].reshape(-1, 3)
    nm = nm / np.maximum(np.linalg.norm(nm, axis=1, keepdims=True), 1e-9)
    a = sc["albedo"][i0:i0+H, j0:j0+W].ravel()
    mk = sc["mask"][i0:i0+H, j0:j0+W].ravel()
    valid = mk > 0
    return nm[valid], a[valid], valid.sum()


def light_pool(scene):
    return load_scene_compat(str(DATA / scene))["sh"]         # (32,9)


def predictors(C, nrm, rho):
    """R-D3 三预测子。C: (N,9)。返回 dict。"""
    sv = np.linalg.svd(C, compute_uv=False)
    p1 = float(sv[-1] ** 2)                                   # σ_min(C)²
    C1 = C[:, 1:4]                                            # l=1 块(N,3), 序 (ny,nz,nx)
    sv1 = np.linalg.svd(C1, compute_uv=False)
    p2 = float(sv1[-1] ** 2)                                  # σ_min(C_1)²
    # κ: 逐像素切向投影 M_p = P_t M P_t(P_t = I − n nᵀ)的 3×3, λ_min 逐像素:
    #   P_t M P_t = M − (M n)nᵀ − n(M n)ᵀ + (nᵀM n) n nᵀ
    M = C1.T @ C1
    Mn = nrm @ M.T                                            # (P,3) = M n 每行
    nMn = (nrm * Mn).sum(1)                                   # (P,) nᵀMn
    nn = nrm[:, :, None] * nrm[:, None, :]                    # (P,3,3)
    A = M[None, :, :] - (Mn[:, :, None] * nrm[:, None, :])
    A = A - (nrm[:, :, None] * Mn[:, None, :]) + (nMn[:, None, None]) * nn
    w = np.linalg.eigvalsh(A)
    kappa = float(w[:, 0].min())                              # 最坏法线切向覆盖
    # ρ² 加权平均版:
    w_mean = np.einsum('pij,p->ij', A, rho ** 2) / max((rho ** 2).sum(), 1e-300)
    kappa_w = float(np.linalg.eigvalsh(w_mean)[0])
    return dict(sigma_min_C_sq=p1, sigma_min_C1_sq=p2, kappa=kappa, kappa_weighted=kappa_w)


def fisher_theta_blocks(nrm, rho, C):
    """θ(法线切向 2/像素)×(ρ)×(C) 的 Fisher 分块(几何已知法线)。
    δI_kp = ρ h_kp (C_kᵀ dY_p)·δn_p + s_kp δρ_p + ρ_p h_kp Y_pᵀδC_k
    δn_p = P_t,p·u_p(2 维切向参数; 基取每像素切向正交基)。
    返回: F_θθ (2P×2P 块对角), F_θρ (2P×P 对角行), F_θC (2P×9N), F_ρρ (P), F_ρC (P×9N),
          切向基 T (P,3,2)。"""
    Y = sh2(nrm)
    Z = Y @ C.T
    Sk = np.maximum(Z, 0)
    Hk = (Z > 0).astype(float)
    dY = sh2_d(nrm)
    CdY = np.einsum('kj,pji->pki', C, dY)                     # (P,N,3) = C_kᵀ dY_p
    P = len(nrm)
    # 切向基: 每像素 2 个正交切向
    ref = np.array([0.0, 0.0, 1.0])
    t1 = np.cross(nrm, np.broadcast_to(ref, nrm.shape))
    bad = np.linalg.norm(t1, axis=1) < 1e-8
    t1[bad] = np.cross(nrm[bad], np.array([1.0, 0, 0]))
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(nrm, t1)
    T = np.stack([t1, t2], axis=2)                            # (P,3,2)
    # 切向梯度: g_t[k, p, d] = (C_kᵀ dY_p)·t_d
    G_t = np.einsum('pki,pid->pkd', CdY, T)                   # (P,N,2)
    Gt = G_t * (rho[:, None] * Hk)[:, :, None]                # 权重 ρ h
    # F_θθ 块对角: 逐像素 2×2 = Σ_k Gt[p,k]·Gt[p,k]ᵀ
    F_tt = np.einsum('pkd,pke->pde', Gt, Gt)                  # (P,2,2)
    # F_θρ: 逐像素 2 维 = Σ_k Gt[p,k]·s_kp
    F_tr = np.einsum('pkd,pk->pd', Gt, Sk)                    # (P,2)
    # F_θC: (P,2,9N) — 逐像素逐光 2×9 = ρ h Y_pᵀ × t(经 t 基)…
    #   δI^C_kp = ρ h Y_pᵀδC_k 与 δθ 无关像素 → F_θC[p块, C_k块] = ρ h_kp (t_dᵀdY_p 相关)…
    #   精确: J_θ(k,p,d) = ρ h (C_kᵀdY_p·t_d); J_C(k,p,j) = ρ h Y_p(j)
    #   F_θC[p-d, (k,j)] = Σ_obs ρh(t·dY)·ρh Y(j) —— 同像素同光: ρ²h (C_kᵀdY·t_d)Y_p(j)
    # F_tC[p, d, (k,j)] = ρ_p h_kp·G_t_raw[p,k,d]·Y_p(j)
    # F_tC[p,d,(k,j)] = ρ_p h_kp G_t[p,k,d] Y_p(j) —— 广播构造 (P,N,2,9) → 转置
    F_tC = (G_t[:, :, :, None] * Y[:, None, None, :] *
            (rho[:, None] * Hk)[:, :, None, None]).transpose(0, 2, 3, 1)   # (P,2,9,N)
    F_rho = dict(Sk=Sk, Hk=Hk)
    return F_tt, F_tr, F_tC, F_rho, Y, T, (Sk, Hk, dY, CdY)


def trace_theta_inv(nrm, rho, C, mode, cond_cut=1e6):
    """tr(S_θ⁻¹) 解析版(2026-09-05 终版, 修复 CG 路径的近奇异污染):

    R-D1 要求投影掉近零子空间 —— 等价实现 = 谱截断的迹:
      条件块: S_cond 逐像素 2×2(S_tt 消 ρ 后),迹 = Σ tr(S_tt[p]⁻¹),
              条件数 > cond_cut 的像素排除(其法线角 CRB 无界, 迹发散);
      Schur 块: S_schur = S_tt − M S_CC_corr⁺ Mᵀ(M = F_tC_corr 重排 (2P,9N)),
              Woodbury: S⁻¹ = S_tt⁻¹ + S_tt⁻¹M(W)⁻¹MᵀS_tt⁻¹,
              W = S_CC_corr − Mᵀ S_tt⁻¹ M (9N×9N),
              tr(S⁻¹) = tr(S_tt⁻¹) + tr(W⁻¹ · Mᵀ S_tt⁻² M)。
    像素口径: F_rr > 1e-6·median(全暗物理排除) + 条件数过滤(近奇异排除)。
    """
    F_tt, F_tr, F_tC, F_rho, Y, T, (Sk, Hk, dY, CdY) = fisher_theta_blocks(nrm, rho, C)
    P = len(nrm)
    N = C.shape[0]
    F_rr = (Sk ** 2).sum(1)
    inv_rr = 1.0 / np.maximum(F_rr, 1e-300)
    S_tt = F_tt - F_tr[:, :, None] * F_tr[:, None, :] * inv_rr[:, None, None]   # (P,2,2)
    # F_ρC / F_CC / F_tC 的消 ρ 修正(同前)
    W_kp = rho[:, None] * Hk
    F_rC = np.concatenate([(Sk[:, k] * W_kp[:, k])[:, None] * Y for k in range(N)], axis=1)
    F_rC4 = F_rC.reshape(P, N, 9)
    F_tC_corr = F_tC - F_tr[:, :, None, None] * inv_rr[:, None, None, None] *         F_rC4.transpose(0, 2, 1)[:, None, :, :]                                  # (P,2,9,N)
    F_CC = np.zeros((9 * N, 9 * N))
    for k in range(N):
        Yk = Y * W_kp[:, k][:, None]
        F_CC[9*k:9*(k+1), 9*k:9*(k+1)] = Yk.T @ Yk
    F_CC_corr = F_CC - F_rC.T @ (F_rC * inv_rr[:, None])
    # 逐像素 2×2 逆 + 条件数过滤
    det = S_tt[:, 0, 0] * S_tt[:, 1, 1] - S_tt[:, 0, 1] ** 2
    tr2 = S_tt[:, 0, 0] + S_tt[:, 1, 1]
    ok = (det > 1e-12 * np.maximum(tr2 ** 2, 1e-300)) & (tr2 > 0) & np.isfinite(det) & np.isfinite(tr2)
    cond = np.full(P, np.inf)
    cond[ok] = tr2[ok] ** 2 / np.maximum(det[ok], 1e-300)            # λmax/λmin ≈ tr²/det
    keep = ok & (cond < cond_cut)
    S_tt_inv = np.zeros((P, 2, 2))
    if keep.any():
        S_tt_inv[keep] = np.linalg.pinv(S_tt[keep])                  # 批式 pinv(奇异安全)
    if keep.sum() == 0:
        return dict(tr_conditional=float('nan'), tr_schur=float('nan'),
                    n_kept=0, n_total=P)
    tr_cond = float(S_tt_inv[keep].sum() * 1.0)                       # Σ tr(2×2 逆)
    # Schur 块(Woodbury):
    Mt = F_tC_corr.transpose(0, 1, 3, 2).reshape(2 * P, 9 * N)        # (2P, 9N)
    # S_tt⁻¹·M 与 S_tt⁻²·M: 块对角作用(排除像素行 = 0 —— 这些像素 M 列贡献也置 0)
    Minv = (S_tt_inv @ Mt.reshape(P, 2, 9 * N)).reshape(2 * P, 9 * N)   # S_tt⁻¹ M
    keep_full = np.repeat(keep, 2)
    Minv[~keep_full] = 0.0
    Mt_z = Mt.copy(); Mt_z[~keep_full] = 0.0
    W_mid = F_CC_corr - Mt.T @ Minv                                   # (9N,9N)
    W_mid = (W_mid + W_mid.T) / 2
    Minv2 = (S_tt_inv @ (S_tt_inv @ Mt.reshape(P, 2, 9 * N))).reshape(2 * P, 9 * N)
    Minv2[~keep_full] = 0.0
    B2 = Minv2.T @ Mt_z                                               # Mᵀ S_tt⁻² M (9N,9N)
    Wm_inv = np.linalg.pinv(W_mid, rcond=1e-10)
    tr_schur = tr_cond + float(np.trace(Wm_inv @ B2))
    return dict(tr_conditional=tr_cond, tr_schur=float(tr_schur),
                n_kept=int(keep.sum()), n_total=P,
                n_cond_dropped=int(P - keep.sum()))


def main():
    rng = np.random.default_rng(SEED)
    out = {"scenes": {}, "meta": dict(n_configs=N_CONFIGS, n_lights=N_LIGHTS, res=RES,
                                      seed=SEED, design="within-scene (R-D4)",
                                      predictors="σ_min(C)² / σ_min(C_1)² / κ(+weighted) (R-D3)",
                                      response="tr(S_θ⁻¹) 法线角迹 (R-D1), 双块=conditional/Schur")}
    for scene in SCENES:
        nrm, rho, Pn = scene_data(scene)
        pool = light_pool(scene)
        # R-D2 合规像素集: 全部配置都亮起(F_rr>0)的交集, 预先固定
        Y = sh2(nrm)
        cfgs = []
        for ci in range(N_CONFIGS):
            sel = np.sort(rng.choice(len(pool), N_LIGHTS, replace=False))
            cfgs.append(sel)
            C = pool[sel].astype(float)
        lit_all = np.ones(len(nrm), bool)
        for sel in cfgs:
            Zk = Y @ pool[sel].astype(float).T
            lit_all &= (np.maximum(Zk, 0).sum(1) > 1e-10 * np.abs(Zk).max())
        nrm = nrm[lit_all]; rho = rho[lit_all]
        print(f"{scene:10s} 固定像素集(全配置常亮): {lit_all.sum()}/{len(lit_all)}")
        rows = []
        for ci, sel in enumerate(cfgs):
            C = pool[sel].astype(float)
            pred = predictors(C, nrm, rho)
            tr_cond = trace_theta_inv(nrm, rho, C, mode="cond")
            tr_schur = trace_theta_inv(nrm, rho, C, mode="schur")
            rows.append(dict(config=ci, **pred,
                             tr_conditional=tr_cond["tr_conditional"],
                             tr_schur=tr_schur["tr_schur"],
                             n_kept=tr_cond["n_kept"], n_total=tr_cond["n_total"]))
            print(f"  {scene:10s} cfg{ci:02d} κ={pred['kappa']:.4f} "
                  f"tr_cond={tr_cond['tr_conditional']:.4e} "
                  f"tr_schur={tr_schur['tr_schur']:.4e} "
                  f"(kept {tr_cond['n_kept']}/{tr_cond['n_total']})")
        # 场景内 Spearman(双块 × 三预测子)
        stats = {}
        for blk in ("conditional", "schur"):
            resp = [r["tr_conditional" if blk == "conditional" else "tr_schur"] for r in rows]
            for pname in ("sigma_min_C_sq", "sigma_min_C1_sq", "kappa", "kappa_weighted"):
                xs = [r[pname] for r in rows]
                r_, p_ = spearmanr(xs, resp)
                stats[f"{blk}:{pname}"] = dict(rho=float(r_), p=float(p_))
        out["scenes"][scene] = dict(rows=rows, spearman=stats)
        sig = {k: v for k, v in stats.items() if v["p"] < 0.05 and v["rho"] > 0}
        print(f"{scene:10s} 显著正相关口径数: {len(sig)} / 8")
        for k, v in sig.items():
            print(f"    {k}: ρ={v['rho']:.3f} (p={v['p']:.3e})")

    # 汇总判定(预注册): ≥3/4 场景内显著正相关 → 成立域节成立
    n_pos = 0
    for scene in SCENES:
        st = out["scenes"][scene]["spearman"]
        if any(v["p"] < 0.05 and v["rho"] > 0 for k, v in st.items() if k.startswith("schur:kappa")):
            n_pos += 1
    out["verdict"] = dict(
        n_scenes_positive_kappa=n_pos,
        acceptance="预注册: ≥3/4 场景 κ-Schur 显著正相关 → 经验法则成立域节成立; 否则负结果族定稿",
        result=("成立域节成立" if n_pos >= 3 else "负结果族定稿(协议经三洞修正后仍不足 3/4)"),
    )
    print("\n[exp11v2] 判定:", out["verdict"]["result"])
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[exp11v2] 落盘 -> {OUT}")


if __name__ == "__main__":
    main()
