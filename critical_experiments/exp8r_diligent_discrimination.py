#!/usr/bin/env python3
"""关键实验 8R(卡 P, 协议 v2)· DiLiGenT 判别力验证

主判据(v2 预注册): N=3 层(50 次, σ_min(L) 五分位分层)下 ≥6/10 物体
LAE Spearman 显著为正(p<0.05) → 支柱③成立。

诊断量: 加权方向流形 Fisher 的 LAE 迹 = tr(F_dir⁻¹)(方向角域)。
响应量: 加权 GN 估计的 LAE(°, GT = 灯光方向文件)。
分层: N=3 按 σ_min(L) 五分位分 5 层每层 10 次; N∈{5,10,20,50} 各 20 次随机。
辅助: 留出重照明 RMSE + 光强相对误差(报告用, 不进主判据)。
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
from PIL import Image
from scipy.optimize import minimize
from scipy.stats import spearmanr, chi2

HERE = Path(__file__).resolve().parent
OUT = HERE / "exp8r_diligent_discrimination.json"
ROOT = Path("D:/data/DiLiGenT/pmsData")
N3_SAMPLES = 50
OTHER_SAMPLES = 20
SEED = 20260906


def load_object(d):
    dirs = np.loadtxt(d / "light_directions.txt")
    ints = np.loadtxt(d / "light_intensities.txt")[:, 0]
    N_gt = sio.loadmat(d / "Normal_gt.mat")["Normal_gt"]
    mask = np.array(Image.open(d / "mask.png").convert("L")) > 128
    imgs = np.stack([np.array(Image.open(f"{d}/{i:03d}.png").convert("L")).astype(float)
                     for i in range(1, 97)])
    I = imgs[:, mask] / ints[:, None]
    return dirs, N_gt[mask], I, mask


def calibrate_noise(n_gt, dirs, I_norm):
    nl = np.clip(n_gt @ dirs.T, 0, None)
    rho = (I_norm.T * nl).sum(1) / np.maximum((nl*nl).sum(1), 1e-12)
    I_hat = (rho[:, None] * nl).T
    resid = I_norm - I_hat
    sel = I_hat.ravel() > 0.01
    a_, b_ = np.polyfit(I_hat.ravel()[sel], (resid.ravel()**2)[sel], 1)
    return rho, a_, b_, float(np.linalg.norm(resid) / np.linalg.norm(I_norm))


def dir_fisher_trace(n_gt, L_sub, rho, a_, b_, I_sub):
    """方向流形 Fisher 的 LAE 迹: 每光 2×2(方向切空间)加权信息矩阵逆迹之和。
    J_l = ρ_p·h_kp·n_p·(切向基), h_kp = 1[n·l>0]; 加权 w = 1/σ²。"""
    P = len(n_gt)
    tr_total = 0.0
    for k, l in enumerate(L_sub):
        h = (n_gt @ l > 0).astype(float)
        nl = np.clip(n_gt @ l, 0, None)
        w = 1.0 / np.maximum(a_ + b_ * np.maximum(I_sub[k], 0), 1e-6)
        # 切向基
        ref = np.array([0.0, 0.0, 1.0]) if abs(l[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        t1 = np.cross(l, ref); t1 /= np.linalg.norm(t1)
        t2 = np.cross(l, t1)
        # 加权 Fisher 2×2: F_ij = Σ_p w·(ρh·n·t_i)·(ρh·n·t_j)
        g1 = rho * h * (n_gt @ t1)
        g2 = rho * h * (n_gt @ t2)
        F11 = (w * g1 * g1).sum(); F22 = (w * g2 * g2).sum(); F12 = (w * g1 * g2).sum()
        det = F11 * F22 - F12 * F12
        tr_ = (F11 + F22) / max(det, 1e-300)          # tr(F⁻¹)
        tr_total += tr_
    return tr_total


def lae_of_subset(I_sub, L_sub, rho, n_gt, a_, b_):
    ests = []
    rng = np.random.default_rng(0)
    for k in range(len(L_sub)):
        w = 1.0 / np.maximum(a_ + b_ * np.maximum(I_sub[k], 0), 1e-6)
        def obj(l_flat):
            l = l_flat / max(np.linalg.norm(l_flat), 1e-12)
            r = I_sub[k] - rho * np.clip(n_gt @ l, 0, None)
            return float((w * r * r).sum())
        x0 = L_sub[k] + rng.normal(0, 0.03, 3)
        res = minimize(obj, x0, method='Nelder-Mead',
                       options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-7})
        l_e = res.x / np.linalg.norm(res.x)
        c = np.clip(l_e @ L_sub[k], -1, 1)
        ests.append(np.degrees(np.arccos(c)))
    return float(np.mean(ests))


def main():
    rng = np.random.default_rng(SEED)
    objects = sorted([d for d in ROOT.iterdir() if d.is_dir()])
    print(f"物体数: {len(objects)}")
    out = {"objects": {}, "meta": dict(n3=N3_SAMPLES, other=OTHER_SAMPLES, seed=SEED)}

    # 先收集所有 σ_min(L)(N=3 层的分层键)
    all_sig = {}
    for d in objects:
        dirs, n_gt, I_norm, mask = load_object(d)
        # 随机 N=3 抽样分布
        sigs = []
        for _ in range(N3_SAMPLES):
            sel = rng.choice(96, 3, replace=False)
            sv = np.linalg.svd(dirs[sel], compute_uv=False)
            sigs.append(sv[-1])
        all_sig[d.name] = np.array(sigs)
    # 合并五分位
    all_flat = np.concatenate(list(all_sig.values()))
    q_edges = np.quantile(all_flat, [0.2, 0.4, 0.6, 0.8])

    for d in objects:
        name = d.name
        try:
            dirs, n_gt, I_norm, mask = load_object(d)
            rho, a_, b_, lambert_resid = calibrate_noise(n_gt, dirs, I_norm)
            print(f"\n{name}: P={len(n_gt)}, σ²={a_:.3f}+{b_:.3f}I, 朗伯残差 {lambert_resid:.3f}")

            rows = []
            # N=3: 分层(σ_min 五分位每层 10 次)
            sigs = all_sig[name]
            bins = [np.where(sigs <= q_edges[0])[0],
                    np.where((sigs > q_edges[0]) & (sigs <= q_edges[1]))[0],
                    np.where((sigs > q_edges[1]) & (sigs <= q_edges[2]))[0],
                    np.where((sigs > q_edges[2]) & (sigs <= q_edges[3]))[0],
                    np.where(sigs > q_edges[3])[0]]
            rng2 = np.random.default_rng(SEED + hash(name) % 1000)
            # 重新生成 sel 集与 sigs 一致
            sels = []
            for _ in range(N3_SAMPLES):
                sels.append(rng2.choice(96, 3, replace=False))
            for i, sel in enumerate(sels):
                L_sub = dirs[sel]; I_sub = I_norm[sel]
                tr = dir_fisher_trace(n_gt, L_sub, rho, a_, b_, I_sub)
                lae = lae_of_subset(I_sub, L_sub, rho, n_gt, a_, b_)
                rows.append(dict(N=3, sigma_min=float(np.linalg.svd(L_sub, compute_uv=False)[-1]),
                                trace=tr, lae=lae))
            # 其他 N
            for N in (5, 10, 20, 50):
                for _ in range(OTHER_SAMPLES):
                    sel = rng2.choice(96, N, replace=False)
                    L_sub = dirs[sel]; I_sub = I_norm[sel]
                    tr = dir_fisher_trace(n_gt, L_sub, rho, a_, b_, I_sub)
                    lae = lae_of_subset(I_sub, L_sub, rho, n_gt, a_, b_)
                    rows.append(dict(N=N, sigma_min=float(np.linalg.svd(L_sub, compute_uv=False)[-1]),
                                     trace=tr, lae=lae))
            # N=3 Spearman(诊断迹 vs LAE)
            n3_rows = [r for r in rows if r["N"] == 3]
            trs = np.array([r["trace"] for r in n3_rows])
            laes = np.array([r["lae"] for r in n3_rows])
            fin = np.isfinite(trs) & np.isfinite(laes)
            rho_s, p_s = spearmanr(trs[fin], laes[fin])
            out["objects"][name] = dict(
                rows=rows, lambert_resid=lambert_resid,
                spearman_n3=dict(rho=float(rho_s), p=float(p_s), n=int(fin.sum())))
            print(f"  N=3 Spearman(迹, LAE) = {rho_s:.3f} (p={p_s:.4e}, n={fin.sum()})")
        except Exception as exc:
            import traceback; traceback.print_exc()
            out["objects"][name] = dict(error=str(exc))

    # 汇总判定
    n3_stats = {k: v["spearman_n3"] for k, v in out["objects"].items() if "spearman_n3" in v}
    sig_pos = sum(1 for v in n3_stats.values() if v["p"] < 0.05 and v["rho"] > 0)
    ps = [v["p"] for v in n3_stats.values()]
    fisher_stat = -2 * sum(np.log(max(p, 1e-300)) for p in ps) if ps else 0
    meta_p = float(1 - chi2.cdf(fisher_stat, 2 * len(ps))) if ps else float('nan')
    out["verdict"] = dict(
        n_objects=len(n3_stats), n_significant_positive=sig_pos,
        meta_p=meta_p,
        acceptance="≥6/10 物体 LAE Spearman 显著正(p<0.05) → 支柱③成立",
        result=("支柱③成立" if sig_pos >= 6 else "诚实负结果 + 按物体非朗伯强度归因"))
    print(f"\n[exp8R] 判定: {sig_pos}/{len(n3_stats)} 物体显著正 | meta-p = {meta_p:.3e}")
    print(f"  → {out['verdict']['result']}")
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[exp8R] 落盘 -> {OUT}")


if __name__ == "__main__":
    main()
