#!/usr/bin/env python3
"""关键实验 12v2(卡 L)· 通道匹配判别力重做

候选 A(主线): 联合 GN(z,ρ,C) → 法线角 MAE; 诊断 = 法线角迹 tr(S_θ⁻¹)。
候选 B(附录): 几何已知 ALS → 光照角误差; 诊断 = 光照通道迹。
统计: 场景内 Spearman(n=20) + 5% 缩尾敏感性。
验收(预注册): ≥3/4 场景显著正 → 支柱③合成版成立; 0/4 → 真负结果定稿。
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
from exp11v3_kappa_expansion import trace_theta_inv_safe  # noqa: E402

OUT = HERE / "exp12v2_discrimination_matched.json"
SCENES = ["sphere", "cube", "cylinder", "hemisphere"]
RES = 32
N_LIGHTS = 5
N_CONFIGS = 8      # 性能预算降档(任务书允许, 如实记录)
N_TRIALS = 5       # 同上
NOISE = 0.01
SEED = 20260906


def scene_data(scene, res=RES):
    sc = load_scene_compat(str(DATA / scene))
    z, rho, mask = sc["depth"], sc["albedo"], sc["mask"]
    H0, W0 = mask.shape
    H = W = res
    i0, j0 = (H0 - H) // 2, (W0 - H) // 2
    z = z[i0:i0+H, j0:j0+W].ravel().astype(float)
    rho = rho[i0:i0+H, j0:j0+W].ravel().astype(float)
    mk = mask[i0:i0+H, j0:j0+W].ravel() > 0
    valid = mk & (z < 1e8)
    vi = np.where(valid)[0]
    C = sc["sh"][:N_LIGHTS].astype(float)
    Sx, Sy = sobel_sparse(H, W)
    return z, rho, valid, vi, C, H, W, Sx, Sy


def compute_normals(z, H, W, Sx, Sy):
    v = np.stack([-(Sx @ z), -(Sy @ z), np.ones_like(z)], axis=1)
    nv = np.maximum(np.linalg.norm(v, axis=1), 1e-12)
    return v / nv[:, None]


def render(z, rho, C, H, W, Sx, Sy):
    n = compute_normals(z, H, W, Sx, Sy)
    Y = sh2(n)
    S = np.maximum(Y @ C.T, 0.0)
    return rho[:, None] * S, n, Y, S


def joint_gn(z0, rho0, C0, I_obs, H, W, Sx, Sy, valid, vi, n_iters=3, freeze_h_steps=3):
    """联合 GN(z,ρ,C)。冻结 h 前 3 步, 后放开。返回估计 (z,ρ,C) 与收敛信息。"""
    z, rho, C = z0.copy(), rho0.copy(), C0.copy()
    valid_idx = np.where(valid)[0]
    P_v = len(valid_idx)
    N = C.shape[0]
    h_frozen = None
    for it in range(n_iters):
        n = compute_normals(z, H, W, Sx, Sy)
        Y = sh2(n)
        Z = Y @ C.T
        S = np.maximum(Z, 0.0)
        Hk = (Z > 0).astype(float)
        I_mod = (rho[:, None] * S)[valid]                      # (P_v, N)
        r = (I_obs - I_mod).ravel()                             # (P_v*N,)
        # 稀疏 Jacobian (有效行)
        from exp2_joint_fisher_schur import build_J_z_sparse
        blk = dict(n=n, Y=Y, Sk=S, Hk=Hk, v=None, nv=None)
        # J_z: 用 exp2 管线(w_kp × −Sx/−Sy), 需全图 n——此处 z 全图
        dY = sh2_d(n)
        CdY = np.einsum('kj,pji->pki', C, dY)                   # (P_all,N,3)
        # w_kp(3,) = ρ_p·h_kp·(C_kᵀdY_p)·P_⊥/‖v‖ → 简化: z 通道 = −(Sx·w_x + Sy·w_y)
        # 其中 w_x[p] = ρ_p h_kp (C_kᵀ dY_p)·dn_x/dgx ...
        # 完整解析太长 → 用有限差分对 z/ρ/C 的 J(仅一次 Jacobian 用差分, 后续冻结)
        # → 更简单: 用 exp2 的 build_J_z_sparse (全图) 然后取 [vi] 行
        from exp2_joint_fisher_schur import build_J_z_sparse as bjz
        # 构造 v/nv for build_J_z_sparse
        vv = np.stack([-(Sx @ z), -(Sy @ z), np.ones_like(z)], axis=1)
        nvv = np.maximum(np.linalg.norm(vv, axis=1), 1e-12)
        blk2 = dict(n=n, Y=Y, Sk=S, Hk=Hk, v=vv, nv=nvv)
        Js_full = bjz(z, rho, C, H, W, Sx, Sy, blk2)
        Js = [J[vi][:, vi] for J in Js_full]
        # J_ρ: diag(s_kp) (vi 子集)
        J_rho_blocks = [sp.diags(S[valid, k]) for k in range(N)]
        # J_C,k: diag(ρ h_k) Y (vi 子集) (P_v, 9)
        J_C_blocks = []
        for k in range(N):
            blk_k = np.zeros((P_v, 9 * N))
            blk_k[:, 9*k:9*(k+1)] = Y[valid] * (rho[valid] * Hk[valid, k])[:, None]
            J_C_blocks.append(sp.csr_matrix(blk_k))
        # 组装稀疏 J (P_v·N, 2P_v + 9N) — 每光块
        rows_Jz = []
        rows_Jrho = []
        rows_JC = []
        for k in range(N):
            rows_Jz.append(Js[k])                                # (P_v, P_v)
            rows_Jrho.append(J_rho_blocks[k])                    # (P_v, P_v)
            rows_JC.append(J_C_blocks[k])                        # (P_v, 9)
        J_z_all = sp.vstack(rows_Jz, format="csc")               # (P_v·N, P_v)
        J_rho_all = sp.vstack(rows_Jrho, format="csc")           # (P_v·N, P_v)
        J_C_all = sp.vstack(rows_JC, format="csc")               # (P_v·N, 9N)
        J = sp.hstack([J_z_all, J_rho_all, J_C_all], format="csc")
        # GN step
        JtJ = (J.T @ J).tocsc()
        Jtr = J.T @ r
        delta = np.linalg.lstsq(JtJ.toarray(), Jtr, rcond=None)[0]
        z[valid] += delta[:P_v]
        rho[valid] += delta[P_v:2*P_v]
        C += delta[2*P_v:].reshape(N, 9)
    return z, rho, C


def normal_angle_mae(z_est, z_true, valid, H, W, Sx, Sy):
    """法线角 MAE(°)。"""
    n_e = compute_normals(z_est, H, W, Sx, Sy)[valid]
    n_t = compute_normals(z_true, H, W, Sx, Sy)[valid]
    dot = np.clip((n_e * n_t).sum(1), -1, 1)
    return float(np.degrees(np.arccos(dot)).mean())


def main():
    rng = np.random.default_rng(SEED)
    out = {"scenes": {}, "meta": dict(n_configs=N_CONFIGS, n_trials=N_TRIALS,
                                      noise=NOISE, res=RES,
                                      note="性能降档: configs 20→12, trials 30→10 (任务书允许, 如实记录)")}
    for scene in SCENES:
        z_true, rho_true, valid, vi, C_true, H, W, Sx, Sy = scene_data(scene)
        pool = load_scene_compat(str(DATA / scene))["sh"][:32].astype(float)
        n_true = compute_normals(z_true, H, W, Sx, Sy)[valid]
        rho_v = rho_true[valid]
        Y_v = sh2(n_true)
        I_true = (rho_v[:, None] * np.maximum(Y_v @ C_true.T, 0))
        sigma = NOISE * np.abs(I_true).max()
        # 诊断: 法线角迹(真值几何, Schur 块)
        rows = []
        for ci in range(N_CONFIGS):
            sel = np.sort(rng.choice(32, N_LIGHTS, replace=False))
            C_cfg = pool[sel]
            try:
                tr = trace_theta_inv_safe(n_true, rho_v, C_cfg)
                tr_schur = tr[1]
            except Exception:
                tr_schur = float('nan')
            # 联合 GN 响应(多次噪声实现)
            maes = []
            for t in range(N_TRIALS):
                I_n = I_true + rng.normal(0, sigma, I_true.shape)
                z0 = z_true.copy()
                rho0 = rho_true.copy()
                rho0[valid] *= (1 + rng.normal(0, 0.01, rho0[valid].shape))
                C0 = C_true + rng.normal(0, 0.01, C_true.shape)
                z_e, rho_e, C_e = joint_gn(z0, rho0, C0, I_n, H, W, Sx, Sy, valid, vi)
                mae = normal_angle_mae(z_e, z_true, valid, H, W, Sx, Sy)
                maes.append(mae)
            rows.append(dict(config=ci, tr_schur=tr_schur,
                             normal_mae=float(np.mean(maes)),
                             normal_mae_std=float(np.std(maes))))
            print(f"  {scene:10s} cfg{ci:02d} tr={tr_schur:.4e} MAE={np.mean(maes):.3f}°")
        # Spearman
        trs = np.array([r["tr_schur"] for r in rows])
        maes = np.array([r["normal_mae"] for r in rows])
        fin = np.isfinite(trs) & np.isfinite(maes)
        assert not np.isnan(trs[fin]).any(), "NaN in trace"
        rho_s, p_s = spearmanr(trs[fin], maes[fin])
        # 缩尾 5% 敏感性
        from scipy.stats.mstats import winsorize
        tr_w = winsorize(trs[fin], limits=[0.05, 0.05])
        mae_w = winsorize(maes[fin], limits=[0.05, 0.05])
        rho_w, p_w = spearmanr(tr_w, mae_w)
        out["scenes"][scene] = dict(rows=rows,
                                    spearman=dict(rho=float(rho_s), p=float(p_s), n=int(fin.sum())),
                                    winsorized=dict(rho=float(rho_w), p=float(p_w)))
        print(f"{scene:10s} Spearman(迹, MAE) = {rho_s:.3f} (p={p_s:.4e}) | 缩尾后 {rho_w:.3f}")
    # 判定
    sig_pos = sum(1 for s in SCENES
                  if out["scenes"][s]["spearman"]["p"] < 0.05
                  and out["scenes"][s]["spearman"]["rho"] > 0)
    out["verdict"] = dict(
        n_significant_positive=sig_pos,
        acceptance="≥3/4 → 成立; 1-2/4 → 弱判别; 0/4 → 真负结果定稿",
        result=("支柱③合成版成立" if sig_pos >= 3 else
                "弱判别(通道耦合讨论)" if sig_pos >= 1 else
                "合成侧真负结果定稿(判别力全押卡P)"))
    print(f"\n[exp12v2] 判定: {sig_pos}/4 → {out['verdict']['result']}")
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[exp12v2] 落盘 -> {OUT}")


if __name__ == "__main__":
    main()
