"""P1-11 · Fisher / Conditioning 信息分析（无神经网络）。

对每个 (scene, light subset)：
  固定 A=A_GT、n=n_GT（mesh normal），把
    I_k = A ⊙ E(n, L_k) + ε   视为对 L_k 的非线性观测
  在 GT L=0 附近做线性化（c_lm → Y(n)·c_lm = E(n)），
  得到 9 维光照参数 Fisher 信息矩阵：
    J = dI/dL ∈ R^{(K*H*W_mask) × 9}
    F ≈ J^T J  ∈ R^{9×9}
  报告：
    - 最小非零特征值 λ_min
    - 条件数 κ = λ_max / λ_min
    - effective rank（λ_i > ε·λ_max 的个数）
    - log det(F)
随 N 与 illumination diversity 增大，预期 F 的 κ 与 effective rank 改善。

支持 P 域 calibration set 与 R 域主数据集（直接读 P1-04 输出格式）。
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
    sc["albedo"] = np.load(os.path.join(path, "albedo.npy"))[0]
    sc["n_mesh"] = np.load(os.path.join(path, "normal_mesh.npy"))
    sc["mask"] = np.load(os.path.join(path, "mask.npy"))[0].astype(bool)
    sc["sh_irr"] = np.load(os.path.join(path, "sh_coeffs_irradiance.npy"))
    return sc


def fisher_for_subset(sc, subset, eps=1e-3, mask_subsample=4):
    """对 scene 的 light subset 计算 9×9 Fisher 信息矩阵（F = J^T J，A·J_w·A）。

    J: 每像素对每个 c_lm 的导数。
    dI/dc_lm = A · ∂E/∂c_lm = A · Y_lm(n)
    """
    m = sc["mask"]
    # 简单子采样（mask 内的随机像素）以限制 J 的大小
    idx = np.argwhere(m)
    if len(idx) > 4000:
        rng = np.random.default_rng(0)
        sel = rng.choice(len(idx), 4000, replace=False)
        idx = idx[sel]
    A = sc["albedo"][idx[:, 0], idx[:, 1]]            # [P]
    n = sc["n_mesh"].transpose(1, 2, 0)
    n_pts = n[idx[:, 0], idx[:, 1]]

    Y = sh_basis_npy(n_pts)                              # [P,9]
    # J = A * Y（每像素对 c_lm 的导数）
    J = Y * A[:, None]                                  # [P,9]
    F = J.T @ J                                         # [9,9]
    ev = np.linalg.eigvalsh(F)
    ev = np.sort(ev)[::-1]
    eps_lam = max(eps, eps * ev.max())
    eff_rank = int((ev > eps_lam).sum())
    kappa = float(ev[0] / max(ev[-1], 1e-12)) if eff_rank else float("inf")
    return dict(scene=sc["name"], N=len(subset),
                lambda_min=float(ev[-1]) if eff_rank else 0.0,
                lambda_max=float(ev[0]),
                condition_number=kappa,
                effective_rank=eff_rank,
                log_det=float(np.log(max(np.linalg.det(F), 1e-30))),
                ev_top3=ev[:3].tolist(),
                ev_bot3=ev[-3:].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--max_scenes", type=int, default=0)
    ap.add_argument("--ns", nargs="+", type=int, default=[1, 2, 3, 5, 8, 15])
    args = ap.parse_args()

    scenes = [os.path.join(args.data_root, d) for d in sorted(os.listdir(args.data_root))
              if os.path.isdir(os.path.join(args.data_root, d))
              and os.path.isfile(os.path.join(args.data_root, d, "sh_coeffs_irradiance.npy"))]
    if args.max_scenes:
        scenes = scenes[: args.max_scenes]
    print(f"[conditioning] scenes={len(scenes)} Ns={args.ns}")

    rows = []
    for sd in scenes:
        sc = load_scene(sd)
        for N in args.ns:
            if N > sc["sh_irr"].shape[0]:
                continue
            sub = list(range(N))
            rows.append(fisher_for_subset(sc, sub))
    if not rows:
        print("无数据；请先生成 calibration/data。")
        return
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    # 简要报告
    import numpy as np
    for N in args.ns:
        rs = [r for r in rows if r["N"] == N]
        if not rs: continue
        k = np.array([r["condition_number"] for r in rs])
        eff = np.array([r["effective_rank"] for r in rs])
        print(f"  N={N:2d}  κ mean={k.mean():.2e} p95={np.percentile(k,95):.2e}  "
              f"eff_rank mean={eff.mean():.1f}/{9}")


if __name__ == "__main__":
    main()
