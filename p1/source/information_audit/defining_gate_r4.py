"""P1-R4 · 定核 Gate：同 N 不同光照子集的 GA-ISI vs 恢复误差扫描。

对每场景每 N（{3,5,8,12}）采样 20 个随机子集，同时计算：
  - GA-ISI 分数（gauge_fisher.ga_isi_scores，解析，CPU 秒级）
  - solver 恢复误差（joint_solve 受控求解，GPU）SI-MAE(A) 与 relighting 误差
定核判据：
  G1: 固定 N 内，GA-ISI 分数与 SI-MAE 显著相关（Spearman |ρ| 有稳定符号）
  G2: 回归 Error ~ logN + GA-ISI 的解释力显著优于 Error ~ logN（ΔR²）
输出：
  p1/information_audit/defining_gate_subset_sweep.csv
  p1/information_audit/defining_gate_summary.json + REPORT.md
"""
import argparse
import csv
import json
import math
import os
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "physics")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "information_audit")))

from gauge_fisher import load_scene, ga_isi_scores  # noqa: E402
from information_audit_v2 import joint_solve  # noqa: E402
from sh import sh_basis_npy  # noqa: E402


def si_mae_np(pred, gt, mask):
    p, g = pred[mask], gt[mask]
    d = (p * p).sum()
    if d < 1e-12:
        return float("nan")
    s = (p * g).sum() / d
    return float(np.abs(s * p - g).mean())


@np.errstate(all="ignore")
def relight_error(alb, n_hat, sh_subset, sc, q, mask):
    """oracle-query-light：用预测 A/n + GT query 光重建第 q 张图。"""
    n_t = None
    import torch
    from pre0.source.train.train_probe import sh_shading
    n_t = torch.from_numpy(n_hat).float().to("cuda" if torch.cuda.is_available() else "cpu")
    shq = torch.from_numpy(sc["sh_irr"][q][None, None]).float().to(n_t.device)
    s_q = sh_shading(n_t, shq)[0, 0].cpu().numpy()
    ih = alb * s_q
    iq = sc["imgs_lin"][q]
    mse = float(((ih - iq)[mask] ** 2).mean())
    return 10 * math.log10(1 / max(mse, 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=os.path.join(_REPO, "p1", "calibration_set", "data_sun"))
    ap.add_argument("--out_dir", default=os.path.join(_REPO, "p1", "information_audit"))
    ap.add_argument("--ns", nargs="+", type=int, default=[3, 5, 8, 12])
    ap.add_argument("--subsets_per_N", type=int, default=20)
    ap.add_argument("--restarts", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20260830)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    scenes = [os.path.join(args.data_root, d) for d in sorted(os.listdir(args.data_root))
              if os.path.isdir(os.path.join(args.data_root, d))
              and os.path.isfile(os.path.join(args.data_root, d, "sh_coeffs_irradiance.npy"))]
    rng = np.random.default_rng(args.seed)
    rows = []
    for sd in scenes:
        sc = load_scene(sd)
        if sc["mask"].sum() < 100:
            continue
        for N in args.ns:
            if N >= sc["K"] - 1:
                continue
            for si in range(args.subsets_per_N):
                sub = sorted(rng.choice(sc["K"], N, replace=False).tolist())
                g = ga_isi_scores(sc, sub)
                res = joint_solve(sc, sub, restarts=args.restarts)
                e_A = si_mae_np(res["A_hat"], sc["albedo"], sc["mask"])
                q = [k for k in range(sc["K"]) if k not in sub][0]
                e_rel = relight_error(res["A_hat"], sc["n_mesh"][None].copy(), None, sc, q, sc["mask"])
                rows.append(dict(scene=sc["name"], N=N,
                                 subset=",".join(map(str, sub)),
                                 lambda_min_eff=g["lambda_min_eff"],
                                 logdet_eff=g["logdet_eff"],
                                 a_opt_eff=g["a_opt_eff"],
                                 si_mae_A=e_A, ho_psnr=e_rel))
        print(f"  {sc['name']} done")
    csv_path = os.path.join(args.out_dir, "defining_gate_subset_sweep.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[R4] {csv_path} rows={len(rows)}")

    # ---- 定核统计 ----
    from scipy.stats import spearmanr
    summary = dict(n_rows=len(rows), per_N={}, pooled={})
    rows_v = [r for r in rows if np.isfinite(r["si_mae_A"]) and r["lambda_min_eff"] > 0]
    for N in args.ns:
        rs = [r for r in rows_v if r["N"] == N]
        if len(rs) < 10:
            continue
        lm = np.array([r["lambda_min_eff"] for r in rs])
        ld = np.array([r["logdet_eff"] for r in rs])
        ao = np.array([r["a_opt_eff"] for r in rs])
        er = np.array([r["si_mae_A"] for r in rs])
        sp_lm = spearmanr(lm, er)
        sp_ld = spearmanr(ld, er)
        sp_ao = spearmanr(ao, er)
        summary["per_N"][N] = dict(
            n=len(rs),
            spearman_lambda_min=[float(sp_lm.statistic), float(sp_lm.pvalue)],
            spearman_logdet=[float(sp_ld.statistic), float(sp_ld.pvalue)],
            spearman_a_opt=[float(sp_ao.statistic), float(sp_ao.pvalue)],
            si_mae_mean=float(er.mean()), si_mae_std=float(er.std()))
    # pooled 回归：Error ~ logN vs Error ~ logN + score
    lnN = np.log(np.array([r["N"] for r in rows_v]))
    err = np.array([r["si_mae_A"] for r in rows_v])
    ld = np.array([r["logdet_eff"] for r in rows_v])
    X1 = np.vstack([np.ones_like(lnN), lnN]).T
    X2 = np.vstack([np.ones_like(lnN), lnN, -ld]).T
    b1 = np.linalg.lstsq(X1, err, rcond=None)[0]
    b2 = np.linalg.lstsq(X2, err, rcond=None)[0]
    r2_1 = 1 - ((X1 @ b1 - err) ** 2).sum() / ((err - err.mean()) ** 2).sum()
    r2_2 = 1 - ((X2 @ b2 - err) ** 2).sum() / ((err - err.mean()) ** 2).sum()
    summary["pooled"] = dict(r2_logN_only=float(r2_1), r2_logN_plus_logdet=float(r2_2),
                             delta_r2=float(r2_2 - r2_1),
                             beta_logdet=float(b2[2]),
                             verdict="PASS" if (r2_2 - r2_1) > 0.05 and b2[2] < 0 else "INCONCLUSIVE/FAIL")
    json.dump(summary, open(os.path.join(args.out_dir, "defining_gate_summary.json"),
                            "w", encoding="utf-8"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
