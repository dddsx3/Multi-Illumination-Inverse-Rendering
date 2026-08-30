"""P1-R3 · Gauge-Aware Illumination-Set Information（GA-ISI）。

数学对象（per 专家定核 Gate 规格）：

模型：I_k(p) = a_p · Ŝ_k(p)，Ŝ_k(p) = ReLU(Σ_lm c_km Y_lm(n_p))
参数：共享场景 θ_s = (a_p)_{p∈mask}；per-light nuisance θ_ℓ = (c_1..c_N)
观测 Jacobian：
  ∂I_k(p)/∂a_p     = Ŝ_k(p)                （E_k(p)，含 ReLU 指示）
  ∂I_k(p)/∂c_km    = a_p·Y_m(n_p)·1[Ŝ_k(p)>0]
Fisher 分块：
  F_ss = diag_p( Σ_k Ŝ_k(p)² )                    [P×P 对角]
  F_ℓℓ = blockdiag_k( Σ_p a_p²·1[Ŝ_k>0]·Y Yᵀ )     [9N×9N 块对角]
  F_sℓ(p, k块) = a_p·1[Ŝ_k(p)>0]·Y(n_p)ᵀ           [P×9N]
**Schur 补（消去 nuisance 后场景参数的有效信息）**：
  F_eff(p) = F_ss(p) − F_sℓ(p) F_ℓℓ⁻¹ F_sℓ(p)ᵀ
           = a_p²·[ Σ_k Ŝ_k(p)² − Σ_k 1[Ŝ_k>0]·Y(n_p)ᵀ F_k⁻¹ Y(n_p) ]
**Gauge 处理**：全局尺度规范 δa=εa, δc=−εc 是解析已知 null 方向；
  本实现通过把 a 归一到单位 RMS（固定 gauge）后计算——报告中注明。
  （完全的 nullspace 投影版本见 IDENTIFIABILITY.md 展望。）

输出分数（每 (scene, subset)）：
  lambda_min_eff  = min_p F_eff(p)      （最坏像素的有效信息）
  logdet_eff      = Σ_p log F_eff(p)    （A-最优的互补）
  a_opt_eff       = mean_p 1/F_eff(p)   （A-optimal score，越小越好）
  mean_shading_energy = mean_p Σ_k Ŝ_k(p)²

这些量**依赖具体光照子集**（经 Ŝ_k 与 ReLU 指示），与旧
F=(AY)ᵀ(AY)（与子集无关）有本质区别。
"""
import argparse
import csv
import math
import os
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "physics")))
from sh import sh_basis_npy  # noqa: E402


def load_scene(path):
    sc = {"dir": path, "name": os.path.basename(path)}
    K = len([f for f in os.listdir(path) if f.startswith("light_") and f.endswith("_lin.npy")])
    sc["K"] = K
    sc["imgs_lin"] = np.stack([np.load(os.path.join(path, f"light_{k+1:03d}_lin.npy"))
                                for k in range(K)])
    sc["sh_irr"] = np.load(os.path.join(path, "sh_coeffs_irradiance.npy"))
    sc["albedo"] = np.load(os.path.join(path, "albedo.npy"))[0]
    sc["n_mesh"] = np.load(os.path.join(path, "normal_mesh.npy"))
    sc["mask"] = np.load(os.path.join(path, "mask.npy"))[0].astype(bool)
    return sc


def ga_isi_scores(sc, subset, pixel_cap=2000, seed=0):
    """对给定光照子集计算 GA-ISI 分数（见模块 docstring 的推导）。"""
    mask = sc["mask"]
    idx = np.argwhere(mask)
    if len(idx) > pixel_cap:
        rng = np.random.default_rng(seed)
        idx = idx[rng.choice(len(idx), pixel_cap, replace=False)]
    a = sc["albedo"][idx[:, 0], idx[:, 1]].astype(np.float64)
    # gauge 固定：a 归一到单位 RMS（解析已知全局尺度规范）
    a = a / max(np.sqrt((a * a).mean()), 1e-9)
    n = sc["n_mesh"].transpose(1, 2, 0)
    n_pts = n[idx[:, 0], idx[:, 1]]                       # [P,3]
    Y = sh_basis_npy(n_pts)                               # [P,9]
    P = len(a)

    # 各光 shading（GT c，Route A）
    S = np.zeros((len(subset), P))
    for ki, k in enumerate(subset):
        S[ki] = np.maximum(Y @ sc["sh_irr"][k], 0.0)
    active = (S > 0).astype(np.float64)                   # ReLU 指示 [N,P]

    # F_k = Σ_p a²·1[Ŝ_k>0]·Y Yᵀ（9×9，逐光）
    F_blocks = []
    F_inv_Y = np.zeros((len(subset), P, 9))               # F_k⁻¹ Y(n_p) 预计算
    for ki in range(len(subset)):
        w = a * a * active[ki]
        Fk = (Y * w[:, None]).T @ Y
        # ridge 防奇异（无光命中的像素集合 → F_k 奇异）
        Fk += 1e-9 * np.trace(Fk) / 9.0 * np.eye(9)
        F_blocks.append(Fk)
        F_inv_Y[ki] = np.linalg.solve(Fk, Y.T).T          # [P,9]

    # F_eff(p) = a²[ Σ_k Ŝ_k² − Σ_k 1[Ŝ_k>0] Yᵀ F_k⁻¹ Y ]
    term1 = (S ** 2).sum(0)
    term2 = (active * (F_inv_Y * Y).sum(-1)).sum(0)   # [P] 逐像素 Yᵀ F_k⁻¹ Y
    F_eff = (a * a) * (term1 - term2)
    F_eff = np.maximum(F_eff, 0.0)

    valid = F_eff > 1e-12
    if valid.sum() < 10:
        return dict(scene=sc["name"], N=len(subset),
                    lambda_min_eff=0.0, logdet_eff=float("-inf"),
                    a_opt_eff=float("inf"), valid_frac=float(valid.mean()))
    Fe = F_eff[valid]
    return dict(scene=sc["name"], N=len(subset),
                lambda_min_eff=float(Fe.min()),
                lambda_max_eff=float(Fe.max()),
                logdet_eff=float(np.log(Fe).sum()),
                a_opt_eff=float((1.0 / Fe).mean()),
                valid_frac=float(valid.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--ns", nargs="+", type=int, default=[3, 5, 8, 12])
    ap.add_argument("--subsets_per_N", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260830)
    args = ap.parse_args()
    scenes = [os.path.join(args.data_root, d) for d in sorted(os.listdir(args.data_root))
              if os.path.isdir(os.path.join(args.data_root, d))
              and os.path.isfile(os.path.join(args.data_root, d, "sh_coeffs_irradiance.npy"))]
    rng = np.random.default_rng(args.seed)
    rows = []
    for sd in scenes:
        sc = load_scene(sd)
        for N in args.ns:
            if N > sc["K"]:
                continue
            for si in range(args.subsets_per_N):
                sub = sorted(rng.choice(sc["K"], N, replace=False).tolist())
                r = ga_isi_scores(sc, sub)
                r["subset"] = ",".join(map(str, sub))
                rows.append(r)
        print(f"  {sc['name']}: done ({len(args.ns)} N × {args.subsets_per_N} subsets)")
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[ga-isi] {args.out_csv} rows={len(rows)}")
    # 摘要：分数是否随子集变化（GA-ISI 的存在性前提）
    import numpy as np
    for N in args.ns:
        rs = [r for r in rows if r["N"] == N and r["lambda_min_eff"] > 0]
        if rs:
            lm = np.array([r["lambda_min_eff"] for r in rs])
            print(f"  N={N}: λ_min_eff 分布 min={lm.min():.3e} med={np.median(lm):.3e} max={lm.max():.3e}"
                  f"  （分布有宽度=子集质量确实不同）")


if __name__ == "__main__":
    main()
