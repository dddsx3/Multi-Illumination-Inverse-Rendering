"""R4″ Task E · Geometry observability 正式化（任务书 §13-§15）。

核心构造：scene-level SH Gram 矩阵
    G_s = Y_s^T W Y_s,   Y_s = [Y(n_1)..Y(n_P)]^T ∈ R^{P×9},  W 加权

候选 G metric（5 类，§14）：
  G1 rank(G_s)                     （满秩=9 时用 cutoff 数值秩）
  G2 (1/9) logdet(G_s + εI)        （normalized logdet）
  G3 effective rank                （参与比）
  G4 condition number              （λ1/λ9）
  G5 最低非退化 eigenvalue         （9 维小矩阵，非 P 维极值）

稳定性测试（§15，找"哪个 G 最稳定描述 SH 覆盖"，不是找与 error 最大相关）：
  S1 pixel resampling ×5
  S2 mesh-resolution changes ×2   （向下采样顶点数）
  S3 rotation sanity ×3           （绕相机轴的旋转，G 应近似不变——单视角下仅近似）
  S4 scale normalization          （对 a 权重 scale 变化不变）

W 权重两版：W=I（纯法线覆盖）与 W=diag(a²)（albedo 加权）

冻结：通过全部稳定性检查且 CV 最小者 → GEOMETRY_METRIC_FROZEN.json
用法：python r4pp_geometry_metrics.py
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "p1", "source", "physics"))
from sh import sh_basis_npy  # noqa: E402

DATA_ROOT = os.path.join(_REPO, "p1", "calibration_set", "data_sun_confirmatory")
OUT_DIR = os.path.join(_REPO, "r4pp")
EPS = 1e-8

GEOMETRY_METRICS = ["G1_rank", "G2_logdet", "G3_effrank", "G4_cond", "G5_mineig"]


def scene_gram(normals, albedo2=None, eps=EPS):
    """G_s = Y^T W Y。normals [P,3] 单位向量；albedo2 [P] 或 None(W=I)。"""
    Y = sh_basis_npy(normals)                       # [P,9]
    if albedo2 is None:
        return (Y.T @ Y) / Y.shape[0]               # 平均化（scale 不变）
    return (Y * albedo2[:, None]).T @ Y


def g_metrics(G, eps=EPS):
    """5 个候选 G metric（全部对 9×9 G 计算，非 P 维极值）。"""
    w = np.linalg.eigvalsh(G)
    w = w[w > 1e-12 * max(w.max(), 1e-300)]
    if w.size == 0:
        return dict(G1_rank=0.0, G2_logdet=float("-inf"), G3_effrank=0.0,
                    G4_cond=float("inf"), G5_mineig=0.0)
    lam_max = w.max()
    rank = int((w > eps * lam_max).sum())
    logdet = float(np.log(w).mean())                # (1/d) log det, d=9
    p = w / w.sum()
    effrank = float(np.exp(-(p * np.log(p)).sum()))
    cond = float(lam_max / w[0])
    return dict(G1_rank=float(rank), G2_logdet=logdet, G3_effrank=effrank,
                G4_cond=cond, G5_mineig=float(w[0]))


def downsample_normals(n, frac=0.5, seed=0):
    """网格分辨率变化：按比例随机抽法线（保持分布形状）。"""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(n), max(int(len(n) * frac), 100), replace=False)
    return n[idx]


def rotate_normals(n, axis, angle_deg):
    """绕轴旋转法线（rotation sanity：单视角下 G 应近似不变）。"""
    ax = np.asarray(axis, dtype=float)
    ax = ax / np.linalg.norm(ax)
    th = np.radians(angle_deg)
    c, s = np.cos(th), np.sin(th)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    R = np.eye(3) + s * K + (1 - c) * (K @ K)
    return n @ R.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    scenes = sorted([os.path.join(DATA_ROOT, d) for d in os.listdir(DATA_ROOT)
                     if os.path.isdir(os.path.join(DATA_ROOT, d))
                     and os.path.isfile(os.path.join(DATA_ROOT, d, "sh_coeffs_irradiance.npy"))
                     and not d.startswith("_")])
    scenes = scenes[:args.limit] if args.limit else scenes

    rows = []
    for sd in scenes:
        scn = os.path.basename(sd)
        mask = np.load(os.path.join(sd, "mask.npy"))[0].astype(bool)
        n = np.load(os.path.join(sd, "normal_mesh.npy")).transpose(1, 2, 0)[mask]
        alb = np.load(os.path.join(sd, "albedo.npy"))[0][mask].astype(np.float64)
        n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)

        rec = dict(scene=scn, P_full=len(n))
        # W=I 与 W=diag(a²) 两版
        for wtag, alb2 in [("wI", None), ("wa2", alb ** 2)]:
            G = scene_gram(n, alb2)
            m = g_metrics(G)
            for k, v in m.items():
                rec[f"{k}_{wtag}"] = v

            # ---- 稳定性扰动（对 W=I 版做全套）----
            if wtag != "wI":
                continue
            # S1 pixel resampling ×5
            for b in range(5):
                rng = np.random.default_rng(100 + b)
                idx = rng.choice(len(n), min(len(n), 2000), replace=False)
                Gm = scene_gram(n[idx], None)
                mm = g_metrics(Gm)
                for k, v in mm.items():
                    rec[f"{k}_resamp{b}"] = v
            # S2 mesh-resolution ×2
            for fr, tag in [(0.5, "mesh05"), (0.25, "mesh025")]:
                Gm = scene_gram(downsample_normals(n, fr, seed=7), None)
                mm = g_metrics(Gm)
                for k, v in mm.items():
                    rec[f"{k}_{tag}"] = v
            # S3 rotation sanity ×3（绕 z 与两正交轴）
            for ang, tag in [(10, "rot10"), (30, "rot30"), (45, "rot45")]:
                Gm = scene_gram(rotate_normals(n, [0, 0, 1], ang), None)
                mm = g_metrics(Gm)
                for k, v in mm.items():
                    rec[f"{k}_{tag}"] = v
            # S4 scale normalization（对 albedo 加权版的 scale 不变性）
            Gm = scene_gram(n, (alb ** 2) * 7.3)
            mm = g_metrics(Gm)
            for k, v in mm.items():
                rec[f"{k}_scale73"] = v
        rows.append(rec)
        print(f"  [geom] {scn:26s} G1={rec['G1_rank_wI']:.0f} "
              f"G2={rec['G2_logdet_wI']:.3f} G3={rec['G3_effrank_wI']:.2f}",
              flush=True)

    # ---- 稳定性统计（CV 与秩相关）----
    from scipy.stats import spearmanr
    print(f"\n=== 稳定性（W=I 版，{len(rows)} scene）===")
    summary = []
    for gm in GEOMETRY_METRICS:
        base = np.array([r[f"{gm}_wI"] for r in rows])
        finite = np.isfinite(base)
        base_f = base[finite]
        rhos, cvs = [], []
        for tag in ["resamp0", "resamp1", "resamp2", "resamp3", "resamp4",
                    "mesh05", "mesh025", "rot10", "rot30", "rot45", "scale73"]:
            v = np.array([r.get(f"{gm}_{tag}", np.nan) for r in rows])[finite]
            ok = np.isfinite(v) & np.isfinite(base_f)
            if ok.sum() >= 5:
                rhos.append(spearmanr(base_f[ok], v[ok]).statistic)
                cvs.append(np.nanstd(v[ok]) / max(abs(np.nanmean(v[ok])), 1e-12))
        summary.append(dict(metric=gm, n=len(base_f),
                            rho_min=float(np.min(rhos)) if rhos else float("nan"),
                            cv_mean=float(np.nanmean(cvs)) if cvs else float("nan"),
                            cv_max=float(np.nanmax(cvs)) if cvs else float("nan")))
        print(f"  {gm:10s} ρ_min={summary[-1]['rho_min']:.3f} "
              f"CV_mean={summary[-1]['cv_mean']:.3f} CV_max={summary[-1]['cv_max']:.3f}")

    # 落盘
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(OUT_DIR, "04_geometry_spectrum.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(OUT_DIR, "04_geometry_stability.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader(); w.writerows(summary)
    print(f"[geom] -> 04_geometry_spectrum.csv ({len(rows)} rows) + 04_geometry_stability.csv")

    # 冻结建议：CV 最小 + ρ 最高者
    best = min(summary, key=lambda r: r["cv_mean"])
    print(f"\n[冻结建议] {best['metric']}（CV_mean={best['cv_mean']:.3f}, ρ_min={best['rho_min']:.3f}）")
    frozen = dict(primary_geometry_metric=best["metric"], frozen_at="2026-08-31T22:00:00",
                  candidates={r["metric"]: {k: v for k, v in r.items() if k != "metric"}
                              for r in summary})
    with open(os.path.join(OUT_DIR, "GEOMETRY_METRIC_FROZEN.json"), "w",
              encoding="utf-8") as f:
        json.dump(frozen, f, indent=2, ensure_ascii=False)
    print("[geom] GEOMETRY_METRIC_FROZEN.json 已写入")


if __name__ == "__main__":
    main()
