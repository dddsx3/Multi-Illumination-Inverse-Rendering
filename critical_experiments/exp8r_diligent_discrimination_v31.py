#!/usr/bin/env python3
"""卡 S2 · exp8R v3.1: 诊断量代数修正(任务书 v3.3 §3 卡 S2)

supersede exp8R v3 主诊断两处代数错误(§1.2):
  D1 ρ 边缘化: 全局单标量口径 → 逐像素对角精确 Schur
     (Σ_p x_p²/d_p, 一行可写; 注释"全耦合太重"不成立);
  D2 α 强度维: 完全未边缘化 → 联合 3×3 α 块 pinv 边缘化(ρ·α 乘积规范方向);
  D3 求值点: 拟合值(维持预注册口径)——估计器逻辑零改动, 响应 lae 不变,
     种子确定性 → 与云端 rows 可逐子集对账(σ_min ≤1e-16 自检保留);
  D4 每子集落盘拟合参数(npz: rho/alphas/dirs_est)——一劳永逸;
  D5 oracle 臂并列: 变体 A@真值(已有 exp8S)与修正@拟合值差值 = 拟合点污染量化。

红线 #8(已知答案单测): 随机小规模 P=50/N=3 稠密 Fisher → 稠密 pinv Schur
全流程对照移植版, rel err <1e-10 才准跑批。

主判据(冻结不改): N=3 层 50 次下 ≥6/10 物体显著正(p<0.05)+
Fisher 合并 meta-p(前提 = 方向一致性检查通过)。

用法(与 v3 相同, 云/本机均可):
  python exp8r_diligent_discrimination_v31.py --only ballPNG --data_root <pmsData>
"""
from __future__ import annotations
import json
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# 估计器链零改动复用(v3.3 §1.2 判定链: 估计器侧合格, LAE 响应数据保留有效)
import exp8r_diligent_discrimination_v3 as v3  # noqa: E402

from exp8r_diligent_discrimination_v3 import (  # noqa: E402
    SEED, N3_SAMPLES, OTHER_SAMPLES, N_STARTS, CKPT_EVERY,
    load_object, calibrate, pixel_guard, als_init, joint_trf, lae,
)

OUT = HERE / "exp8r_diligent_discrimination_v31.json"


# ------------------------------------------------ 修正版诊断量(卡 S2 核心)
def diagnose_trace_v31(n_gt, I_sub, rho, dirs_sub, alphas, a_, b_):
    """v3.1 修正诊断: 变体 A 代数(严格全耦合 Schur)@ 拟合值。

    参数: n_gt (P,3) 已知法线; I_sub (N,P); rho (P,) 拟合 ρ;
          dirs_sub (N,3) 光方向(估计); alphas (N,) 拟合强度; a_,b_ 噪声模型。
    返回: float tr(方向块 2×2 逐光 (F11+F22)/det 求和)。

    代数(与 exp8S 变体 A 逐行一致, 求值点换拟合值):
      参数序 [α_k, t1_k, t2_k]×N (3N 维) + ρ_p(P 维, 逐像素对角);
      F_ll 3N×3N 全耦合; F_lr 3N×P; Fpp (P,) 逐像素;
      S = F_ll − F_lr diag(1/Fpp) F_lrᵀ        ← ρ 精确逐像素对角 Schur (D1);
      α 块联合边缘化: S_dir −= S_da pinv(S_aa) S_ad  (D2, pinv 处理 ρ·α 规范);
      每光 2×2 方向块 tr(F⁻¹) 口径 (D1/D2 之外的口径与 v3 一致)。
    """
    N = dirs_sub.shape[0]
    P = len(n_gt)
    nl = np.clip(n_gt @ dirs_sub.T, 0, None)       # (P,N)
    h = (nl > 0).astype(float)
    w = 1.0 / np.maximum(a_ + b_ * np.maximum(I_sub, 0), 1e-6)   # (N,P)
    t1s = np.zeros((N, 3)); t2s = np.zeros((N, 3))
    for k in range(N):
        ref = np.array([0., 0., 1.]) if abs(dirs_sub[k, 2]) < 0.9 else np.array([1., 0., 0.])
        t1 = np.cross(dirs_sub[k], ref); t1 /= np.linalg.norm(t1)
        t1s[k] = t1; t2s[k] = np.cross(dirs_sub[k], t1)
    # ρ 信息(逐像素对角, D1): Fpp[p] = Σ_k w_kp (α_k h nl)²
    Fpp = (((alphas[None, :] * h * nl) ** 2) * w.T).sum(1)     # (P,)
    # 全暗像素过滤(修复: 拟合 rho_e 可为精确 0 → ALS 闭式 0/0 → NaN → SVD 不收敛,
    # ball cfg21 实证; Fpp=0 像素不携带方向信息, 通用预处理非选择性)
    keep = Fpp > 1e-12 * max(np.median(Fpp), 1e-300)
    if keep.sum() < 10:
        return float('nan')
    n_gt, I_sub, rho, alphas = n_gt[keep], I_sub[:, keep], rho[keep], alphas
    nl, h = nl[keep], h[keep]
    w = w[:, keep]
    Fpp = Fpp[keep]
    P = int(keep.sum())
    Fpp = np.maximum(Fpp, 1e-300)
    n_l = 3 * N
    F_ll = np.zeros((n_l, n_l))
    F_lr = np.zeros((n_l, P))
    for k in range(N):
        wt = w[k] * h[:, k]                       # (P,)
        g_a = rho * alphas[k] * nl[:, k]          # dI/dα
        g_1 = rho * alphas[k] * (n_gt @ t1s[k])   # dI/dt1
        g_2 = rho * alphas[k] * (n_gt @ t2s[k])   # dI/dt2
        i0 = 3 * k
        F_ll[i0 + 0, i0 + 0] = (wt * g_a * g_a).sum()
        F_ll[i0 + 1, i0 + 1] = (wt * g_1 * g_1).sum()
        F_ll[i0 + 2, i0 + 2] = (wt * g_2 * g_2).sum()
        F_ll[i0 + 0, i0 + 1] = F_ll[i0 + 1, i0 + 0] = (wt * g_a * g_1).sum()
        F_ll[i0 + 0, i0 + 2] = F_ll[i0 + 2, i0 + 0] = (wt * g_a * g_2).sum()
        F_ll[i0 + 1, i0 + 2] = F_ll[i0 + 2, i0 + 1] = (wt * g_1 * g_2).sum()
        # d r/dρ_p (对第 k 光行): −α_k h_pk nl_pk (与 Jacobian 推导一致, 不经 rho)
        F_lr[i0 + 0] = wt * alphas[k] * nl[:, k] * g_a
        F_lr[i0 + 1] = wt * alphas[k] * nl[:, k] * g_1
        F_lr[i0 + 2] = wt * alphas[k] * nl[:, k] * g_2
    # D1: ρ 逐像素对角精确 Schur
    S = F_ll - F_lr @ (F_lr.T / Fpp[:, None])
    # D2: α 联合边缘化(pinv 处理 ρ·α 乘积规范方向的奇异 S_aa)
    alpha_idx = [3 * k for k in range(N)]
    dir_idx = [i for i in range(n_l) if i % 3 != 0]
    S_aa = S[np.ix_(alpha_idx, alpha_idx)]
    S_dir = S[np.ix_(dir_idx, dir_idx)] - \
        S[np.ix_(dir_idx, alpha_idx)] @ np.linalg.pinv(S_aa) @ S[np.ix_(alpha_idx, dir_idx)]
    tr_total = 0.0
    for k in range(N):
        blk = S_dir[2 * k:2 * k + 2, 2 * k:2 * k + 2]
        det = blk[0, 0] * blk[1, 1] - blk[0, 1] * blk[1, 0]
        if det > 0:
            tr_total += (blk[0, 0] + blk[1, 1]) / det
    return tr_total


# ------------------------------------------------ 红线 #8: 已知答案单测
def known_answer_test():
    """随机小规模 P=50/N=3: 移植版 vs 稠密全 Fisher pinv-Schur 全流程对照。

    稠密参考: 参数 [ρ(50) | α,t1,t2 ×3] 全 Fisher 2009×... → 显式组装
    F(ρ ∪ light) → 消 ρ(取光照块 Schur via block pinv)→ 消 α(pinv)→ 方向块迹。
    注意: ρ 与光照块间的 Schur 在稠密口径下用 F 的块分解精确实现;
    移植版应与之一致(唯一差别 = 移植版把 ρ 的 off-diagonal 当 0——本模型下
    ρ_p 只进第 p 像素行 → F_ρρ 确为对角, 两口径数学恒等, rel err 应 <1e-10)。
    """
    rng = np.random.default_rng(20260907)
    P, N = 50, 3
    n = rng.normal(size=(P, 3)); n[:, 2] += 1.2
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    dirs = rng.normal(size=(N, 3)); dirs[:, 2] += 1.2
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    rho = rng.uniform(0.05, 0.5, P)
    alphas = rng.uniform(0.5, 2.0, N)
    a_, b_ = 0.02, 0.01
    I = rho[:, None] * alphas[None, :] * np.clip(n @ dirs.T, 0, None)   # (P,N)
    I_sub = I.T + rng.normal(0, 0.01, I.T.shape)                        # (N,P) 加噪
    nl = np.clip(n @ dirs.T, 0, None)
    h = (nl > 0).astype(float)
    w = 1.0 / np.maximum(a_ + b_ * np.maximum(I_sub, 0), 1e-6)          # (N,P)
    # 切向基(与移植版同定义)
    t1s = np.zeros((N, 3)); t2s = np.zeros((N, 3))
    for k in range(N):
        ref = np.array([0., 0., 1.]) if abs(dirs[k, 2]) < 0.9 else np.array([1., 0., 0.])
        t1 = np.cross(dirs[k], ref); t1 /= np.linalg.norm(t1)
        t1s[k] = t1; t2s[k] = np.cross(dirs[k], t1)
    # ---- 稠密参考: 参数序 [ρ(50) | (α_k,t1_k,t2_k)×3] = 59 维 ----
    n_par = P + 3 * N
    F = np.zeros((n_par, n_par))
    # 残差行序: 光主序 (k,p) → r[(k*P+p)] = I_obs - ρ_p α_k h nl
    for k in range(N):
        rows = np.arange(k * P, (k + 1) * P)
        Jk = np.zeros((P, n_par))
        Jk[rows - k * P, :P] = np.diag(-alphas[k] * nl[:, k] * h[:, k])   # d r/dρ
        Jk[:, P + 3 * k] = -rho * h[:, k] * nl[:, k]                      # d r/dα
        Jk[:, P + 3 * k + 1] = -rho * alphas[k] * h[:, k] * (n @ t1s[k])  # d r/dt1
        Jk[:, P + 3 * k + 2] = -rho * alphas[k] * h[:, k] * (n @ t2s[k])
        Wk = np.diag(w[k])
        F += Jk.T @ Wk @ Jk
    # 稠密 Schur: 消 ρ(50) → 9 维光照块 → 消 α(3, pinv)→ 方向迹
    F_rr = F[:P, :P]
    F_rl = F[:P, P:]
    F_ll = F[P:, P:]
    S_light = F_ll - F_rl.T @ np.linalg.pinv(F_rr) @ F_rl
    # 注意: F_rr 对角可逆(pinv=逆); 消后 9 维
    ai = [0, 3, 6]
    di = [i for i in range(9) if i not in ai]
    S_aa = S_light[np.ix_(ai, ai)]
    S_dir = S_light[np.ix_(di, di)] - \
        S_light[np.ix_(di, ai)] @ np.linalg.pinv(S_aa) @ S_light[np.ix_(ai, di)]
    tr_ref = 0.0
    for k in range(N):
        blk = S_dir[2 * k:2 * k + 2, 2 * k:2 * k + 2]
        det = blk[0, 0] * blk[1, 1] - blk[0, 1] * blk[1, 0]
        if det > 0:
            tr_ref += (blk[0, 0] + blk[1, 1]) / det
    # ---- 移植版 ----
    tr_new = diagnose_trace_v31(n, I_sub, rho, dirs, alphas, a_, b_)
    rel = abs(tr_new - tr_ref) / max(abs(tr_ref), 1e-300)
    print(f"[单测红线#8] P=50,N=3: 稠密参考 tr={tr_ref:.6e} | 移植版 tr={tr_new:.6e} "
          f"| rel err={rel:.2e} {'PASS' if rel < 1e-10 else 'FAIL'}")
    return rel < 1e-10


# ------------------------------------------------ 主循环(v3 骨架 + v3.1 诊断)
def main():
    import argparse
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="只跑指定物体(如 ballPNG)")
    ap.add_argument("--data_root", default=str(v3.ROOT))
    args = ap.parse_args()
    ROOT = Path(args.data_root)

    if not known_answer_test():
        print("[FATAL] 红线#8 单测未过 — 不准跑批(停手)")
        sys.exit(1)

    rng = np.random.default_rng(SEED)
    objects = sorted([d for d in ROOT.iterdir() if d.is_dir()])
    if args.only:
        objects = [d for d in objects if d.name == args.only]
        print(f"[--only] {args.only}")
    print(f"物体数: {len(objects)}")
    out = {"objects": {}, "meta": dict(n3=N3_SAMPLES, other=OTHER_SAMPLES, seed=SEED,
                                       diag="v3.1 变体A代数@拟合值(逐像素ρ-Schur+联合α-pinv边缘化)",
                                       estimator="v3 估计器链零改动(trf+解析稀疏J)",
                                       redline8="PASS")}
    for d in objects:
        name = d.name
        try:
            dirs, n_gt, I_norm, _mask = load_object(d)
            rho_full, a_, b_, lambert_resid = calibrate(n_gt, dirs, I_norm)
            resid_full = None
            import numpy as _np
            nl_full = _np.clip(n_gt @ dirs.T, 0, None)
            I_hat_full = (rho_full[:, None] * nl_full).T
            resid_full = _np.abs(I_norm - I_hat_full)
            out_frac = float((resid_full > 3 * resid_full.std()).mean())
            print(f"\n{name}: P={len(n_gt)}, 朗伯残差 {lambert_resid:.3f}, 离群 {out_frac:.3f}")
            pf = HERE / f"exp8r_v31_partial_{name}.json"
            params_npz = HERE / f"exp8r_v31_params_{name}.npz"
            ci0 = oth0 = ndone = nsess = 0
            rows = []
            if pf.exists():
                pd = json.loads(pf.read_text(encoding="utf-8"))
                rows = pd["rows"]; ci0 = pd["ci_done"]; oth0 = pd["oth_done"]
                ndone = ci0 + oth0
                print(f"    [resume] {name}: N=3 {ci0}/50 + other {oth0}/60")
            t0obj = time.time()
            ntotsub = N3_SAMPLES + OTHER_SAMPLES * 4
            r2 = np.random.default_rng(SEED + zlib.crc32(name.encode()) % 1000)
            sels = [r2.choice(96, 3, replace=False) for _ in range(N3_SAMPLES)]
            other_sels = []
            for N_ in (5, 10, 20, 50):
                for _ in range(OTHER_SAMPLES):
                    other_sels.append((N_, r2.choice(96, N_, replace=False)))
            # 拟合参数累积落盘(D4)
            fit_params = []
            for ci, sel in enumerate(sels[ci0:], start=ci0):
                dirs_sub = dirs[sel]; I_sub = I_norm[sel]
                n_masked = (n_gt @ dirs_sub.T > 0).astype(float)
                keep = pixel_guard(I_sub, n_masked)
                idx_k = np.where(keep)[0]
                if len(idx_k) < 100:
                    continue
                rng3 = np.random.default_rng(SEED)
                sub_idx = np.sort(rng3.choice(idx_k, min(len(idx_k), max(len(idx_k)//8, 2000)), replace=False))
                n_k = n_gt[sub_idx]; I_k = I_sub[:, sub_idx]
                rho_als, dirs_als = als_init(I_k, rho_full[sub_idx]*0.8, dirs_sub, n_k)
                rng_m = np.random.default_rng(SEED + ci)
                best = None
                for s in range(N_STARTS):
                    d0 = dirs_als + rng_m.normal(0, 0.05, dirs_als.shape) if s > 0 else dirs_als
                    d0 /= np.linalg.norm(d0, axis=1, keepdims=True)
                    res = joint_trf(I_k, rho_als, d0, n_k, a_, b_)
                    if best is None or res.cost < best.cost:
                        best = res
                P_k = len(sub_idx)
                parms = best.x[P_k:].reshape(3, 3)
                alpha_e = parms[:, 0]
                xy = parms[:, 1:]
                zz = np.sqrt(np.maximum(1 - xy[:, 0]**2 - xy[:, 1]**2, 1e-12))
                dirs_e = np.column_stack([xy, zz])
                rho_e = best.x[:P_k]
                lae_v = lae(dirs_e, dirs_sub)
                # v3.1 修正诊断 @ 拟合值(D3)
                tr = diagnose_trace_v31(n_k, I_k, rho_e, dirs_sub, alpha_e, a_, b_)
                rows.append(dict(N=3, sigma_min=float(np.linalg.svd(dirs_sub, compute_uv=False)[-1]),
                                 trace=tr, lae=lae_v))
                fit_params.append((rho_e, alpha_e, dirs_e))
                ndone += 1; nsess += 1
                el = time.time() - t0obj
                print(f"      [P] N3 {ndone}/{ntotsub} el={el:.0f}s eta={el/nsess*(ntotsub-ndone):.0f}s lae={lae_v:.2f}", flush=True)
                if ndone % CKPT_EVERY == 0:
                    pf.write_text(json.dumps(dict(ci_done=ci+1, oth_done=oth0, rows=rows)), encoding="utf-8")
            for ni, (N_, sel) in enumerate(other_sels[oth0:], start=oth0):
                dirs_sub = dirs[sel]; I_sub = I_norm[sel]
                n_masked = (n_gt @ dirs_sub.T > 0).astype(float)
                keep = pixel_guard(I_sub, n_masked)
                idx_k = np.where(keep)[0]
                if len(idx_k) < 100:
                    continue
                rng3 = np.random.default_rng(SEED)
                sub_idx = np.sort(rng3.choice(idx_k, min(len(idx_k), max(len(idx_k)//8, 2000)), replace=False))
                n_k = n_gt[sub_idx]; I_k = I_sub[:, sub_idx]
                rho_als, dirs_als = als_init(I_k, rho_full[sub_idx]*0.8, dirs_sub, n_k)
                rng_m = np.random.default_rng(SEED + 10000 + ci0 + ni)
                best = None
                for s in range(N_STARTS):
                    d0 = dirs_als + rng_m.normal(0, 0.05, dirs_als.shape) if s > 0 else dirs_als
                    d0 /= np.linalg.norm(d0, axis=1, keepdims=True)
                    res = joint_trf(I_k, rho_als, d0, n_k, a_, b_)
                    if best is None or res.cost < best.cost:
                        best = res
                P_k = len(sub_idx)
                parms = best.x[P_k:].reshape(N_, 3)
                xy = parms[:, 1:]
                zz = np.sqrt(np.maximum(1 - xy[:, 0]**2 - xy[:, 1]**2, 1e-12))
                dirs_e = np.column_stack([xy, zz])
                lae_v = lae(dirs_e, dirs_sub)
                tr = diagnose_trace_v31(n_k, I_k, best.x[:P_k], dirs_sub, parms[:, 0], a_, b_)
                rows.append(dict(N=N_, sigma_min=float(np.linalg.svd(dirs_sub, compute_uv=False)[-1]),
                                 trace=tr, lae=lae_v))
                fit_params.append((best.x[:P_k], parms[:, 0], dirs_e))
                ndone += 1; nsess += 1
                el = time.time() - t0obj
                print(f"      [P] oth {ndone}/{ntotsub} el={el:.0f}s eta={el/nsess*(ntotsub-ndone):.0f}s lae={lae_v:.2f}", flush=True)
                if ndone % CKPT_EVERY == 0:
                    pf.write_text(json.dumps(dict(ci_done=N3_SAMPLES, oth_done=ni+1, rows=rows)), encoding="utf-8")
            n3 = [r for r in rows if r["N"] == 3]
            from scipy.stats import spearmanr
            trs = np.array([r["trace"] for r in n3]); laes = np.array([r["lae"] for r in n3])
            fin = np.isfinite(trs) & np.isfinite(laes)
            r_, p_ = spearmanr(trs[fin], laes[fin])
            out["objects"][name] = dict(rows=rows, lambert_resid=lambert_resid,
                                         outlier_frac=out_frac,
                                         spearman_n3=dict(rho=float(r_), p=float(p_), n=int(fin.sum())))
            print(f"  N=3 ρ={r_:+.3f} (p={p_:.4f}, n={fin.sum()})")
            per_obj = dict(meta=out["meta"], lambert_resid=lambert_resid,
                           outlier_frac=out_frac, rows=rows,
                           spearman_n3=out["objects"][name]["spearman_n3"])
            (HERE / f"exp8r_v31_per_object_{name}.json").write_text(
                json.dumps(per_obj, ensure_ascii=False, indent=1), encoding="utf-8")
            # D4: 拟合参数 npz(变长 P_k → object 数组)
            np.savez(params_npz,
                     n_rows=len(fit_params),
                     rho=np.stack([f[0] for f in fit_params]) if fit_params else np.zeros(0),
                     alphas=np.stack([f[1] for f in fit_params]) if fit_params else np.zeros(0),
                     dirs=np.stack([f[2] for f in fit_params]) if fit_params else np.zeros(0))
            pf.unlink(missing_ok=True)
            print(f"  [saved] {name} (+params npz)")
        except Exception as exc:
            import traceback; traceback.print_exc()
            out["objects"][name] = dict(error=str(exc))
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[v3.1] 落盘 -> {OUT}")


if __name__ == "__main__":
    main()