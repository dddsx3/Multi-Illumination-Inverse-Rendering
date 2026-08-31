"""R4″ Task F · Controlled geometry pilot（任务书 §16-§20，48h 最关键的 mechanism test）。

对每个 geometry level 估计 β_G = ∂logError/∂I，画 β_G vs G 图。
真正想看到：随 geometry observability 增大，β_G 从接近 0 系统性变得更负。

流程（§19）：
  1. 每 (geometry, N) 生成 1000 candidate subsets，只算 cheap-tier information（P=300）
  2. 按 information quintile 分 5 层，每层等量抽 4 个 → 20 subset/cell
  3. 被抽中的 subset 用 full-tier（P=1000）重算 information + solver（restarts=3）
  4. 每 geometry level 拟合 log Error ~ z(I) → β_G, se, bootstrap CI

约束：
  - N ∈ {3, 5}（§18，不浪费预算跑 N=8）
  - 相机/材质/尺度/renderer/光池全固定，只有 normal coverage 变化
  - 收敛判据用 CONV_CRITERIA_FROZEN（不删 failure，作 hurdle）

用法：python r4pp_controlled_pilot.py --stage candidate   # 分层抽样（CPU 便宜）
      python r4pp_controlled_pilot.py --stage solve       # solver（GPU ~2.5h）
      python r4pp_controlled_pilot.py --stage analyze     # β_G vs G
"""
import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "p1", "source", "information_audit"))
sys.path.insert(0, os.path.join(_REPO, "p1", "source", "physics"))

from gauge_fisher_v2 import (fisher_blocks, schur_full, load_scene,  # noqa
                             scene_arrays)
from sh import sh_basis_npy  # noqa

DATA_ROOT = os.path.join(_REPO, "p1", "calibration_set", "data_sun_controlled")
OUT_DIR = os.path.join(_REPO, "r4pp")
NS = [3, 5]
N_CANDIDATES = 1000
N_STRATA = 5
SUBSETS_PER_STRATUM = 4
CHEAP_P = 300
FULL_P = 1000
RESTARTS = 3
CAND_SEED = 20260910
G_METRIC = "G1_rank"          # 由 Task E 冻结

CFG = os.path.join(OUT_DIR, "config", "controlled_pilot_plan.json")
CAND = os.path.join(OUT_DIR, "controlled_candidates.csv")
SOLVE_CSV = os.path.join(OUT_DIR, "06_controlled_geometry_results.csv")
BETA_CSV = os.path.join(OUT_DIR, "06_beta_per_geometry.csv")


def list_scenes():
    return sorted([d for d in os.listdir(DATA_ROOT)
                   if os.path.isdir(os.path.join(DATA_ROOT, d))
                   and os.path.isfile(os.path.join(DATA_ROOT, d, "sh_coeffs_irradiance.npy"))])


def info_cheap(sc, sub, cap=CHEAP_P, seed=0):
    """cheap-tier information：只用 primary metric（M1 log-pdet）"""
    a, Y, C = scene_arrays(sc, sub, pixel_cap=cap, seed=seed)
    bl = fisher_blocks(a, Y, C)
    F = schur_full(bl)
    w = np.linalg.eigvalsh(F)
    tr = w.sum()
    wn = w / tr if tr > 0 else w * 0
    pos = wn[wn > 1e-12]
    return float(np.log(pos).mean()) if pos.size else float("-inf")


def build_plan():
    """Stage 1-2：候选生成 + cheap 打分 + 分层抽样 → 冻结 plan。"""
    os.makedirs(os.path.join(OUT_DIR, "config"), exist_ok=True)
    if os.path.exists(CFG):
        print(f"[plan] 已存在：{CFG}")
        return json.load(open(CFG, encoding="utf-8"))
    rng = np.random.default_rng(CAND_SEED)
    plan = dict(frozen_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                scenes=[], N_values=NS, n_candidates=N_CANDIDATES,
                n_strata=N_STRATA, subsets_per_stratum=SUBSETS_PER_STRATUM,
                cheap_P=CHEAP_P, full_P=FULL_P, restarts=RESTARTS,
                geometry_metric=G_METRIC, cells=[])
    cand_rows = []
    for scn in list_scenes():
        sc = load_scene(os.path.join(DATA_ROOT, scn))
        K = sc["imgs_lin"].shape[0]
        for N in NS:
            # 生成 1000 候选 + cheap 打分
            cands = []
            for _ in range(N_CANDIDATES):
                sub = sorted(rng.choice(K, N, replace=False).tolist())
                px = int(rng.integers(1 << 31))
                try:
                    s = info_cheap(sc, sub, CHEAP_P, px)
                except Exception as e:  # noqa: BLE001
                    s = float("-inf")
                cands.append((s, sub, px))
                cand_rows.append(dict(scene=scn, N=N, subset=",".join(map(str, sub)),
                                      pixel_seed=px, cheap_info=s))
            # 分层：按 cheap_info 的 quantile
            vals = np.array([c[0] for c in cands])
            finite = np.isfinite(vals)
            if finite.sum() < N_STRATA * SUBSETS_PER_STRATUM:
                print(f"  [plan] {scn} N={N}: finite 候选不足 {finite.sum()}")
                continue
            qs = np.quantile(vals[finite], np.linspace(0, 1, N_STRATA + 1))
            chosen = []
            for si in range(N_STRATA):
                lo, hi = qs[si], qs[si + 1]
                pool = [c for c in cands if lo <= c[0] <= hi]
                if len(pool) < SUBSETS_PER_STRATUM:
                    pool = sorted(cands, key=lambda c: abs(c[0] - (lo + hi) / 2))
                rng.shuffle(pool)
                chosen.extend(pool[:SUBSETS_PER_STRATUM])
            plan["cells"].append(dict(scene=scn, N=N,
                                      subsets=[",".join(map(str, c[1])) for c in chosen],
                                      pixel_seeds=[c[2] for c in chosen],
                                      strata_means=[float(np.mean([cc[0] for cc in
                                                                   cands if qs[i] <= cc[0] <= qs[i+1]]))
                                                    for i in range(N_STRATA)]))
            print(f"  [plan] {scn:14s} N={N}: 20 subsets 分层完成", flush=True)
    json.dump(plan, open(CFG, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    with open(CAND, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["scene", "N", "subset", "pixel_seed", "cheap_info"])
        w.writeheader(); w.writerows(cand_rows)
    print(f"[plan] -> {CFG} ({len(plan['cells'])} cells)")
    return plan


def stage_solve(limit_cells=None):
    from information_audit_v2 import joint_solve, si_mae_np  # lazy torch
    plan = build_plan()
    cells = plan["cells"][:limit_cells] if limit_cells else plan["cells"]
    done = set()
    if os.path.exists(SOLVE_CSV):
        for r in csv.DictReader(open(SOLVE_CSV, encoding="utf-8")):
            if r.get("N", "").lstrip("-").isdigit():
                done.add((r["scene"], int(r["N"]), r["subset"]))
    out_f = open(SOLVE_CSV, "a", newline="", encoding="utf-8")
    wr = None
    cache = {}
    t0 = time.time()
    for cell in cells:
        scn, N = cell["scene"], cell["N"]
        if scn not in cache:
            cache.clear()
            cache[scn] = load_scene(os.path.join(DATA_ROOT, scn))
        sc = cache[scn]
        # full-tier information（P=1000）与 solver 并行算
        for s, px in zip(cell["subsets"], cell["pixel_seeds"]):
            if (scn, N, s) in done:
                continue
            sub = [int(x) for x in s.split(",")]
            info_full = info_cheap(sc, sub, FULL_P, px)     # 用 frozen primary
            r = joint_solve(sc, sub, restarts=RESTARTS, seed=px)
            err = si_mae_np(r["A_hat"], sc["albedo"], sc["mask"])
            rec = dict(scene=scn, N=N, subset=s, pixel_seed=px,
                       information=info_full, reconstruction_error=err,
                       final_objective=r["final_loss"], grad_norm=r["grad_norm"],
                       proj_grad_norm=r["proj_grad_norm"],
                       success=int(r["success"]), converged=int(r.get("converged", 0)),
                       iters=r["iters"])
            if wr is None:
                wr = csv.DictWriter(out_f, fieldnames=list(rec))
                wr.writeheader()
            wr.writerow(rec)
            out_f.flush()
        print(f"  [solve] {scn:14s} N={N}: {len(cell['subsets'])} subsets "
              f"({time.time()-t0:.0f}s total)", flush=True)
    out_f.close()
    print(f"[solve] -> {SOLVE_CSV}")


def stage_analyze():
    import pandas as pd
    from scipy import stats
    df = pd.read_csv(SOLVE_CSV)
    # geometry metric（用 scene Gram 的 rank，从 04_geometry_spectrum 读）
    geo = {}
    for r in csv.DictReader(open(os.path.join(OUT_DIR, "04_geometry_spectrum.csv"))):
        geo[r["scene"]] = float(r[f"{G_METRIC}_wI"])
    rows = []
    for (scn, N), g in df.groupby(["scene", "N"]):
        if len(g) < 10:
            continue
        I = g["information"].values
        E = np.log(g["reconstruction_error"].values)
        # 标准化 I（z-score），log Error ~ z(I)
        Iz = (I - I.mean()) / max(I.std(), 1e-12)
        slope, intercept, r, p, se = stats.linregress(Iz, E)
        # bootstrap CI（scene 内重采样 subset）
        rng = np.random.default_rng(42)
        boots = []
        for _ in range(2000):
            idx = rng.integers(0, len(Iz), len(Iz))
            b = stats.linregress(Iz[idx], E[idx])[0]
            boots.append(b)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append(dict(scene=scn, N=N, G=geo.get(scn, float("nan")),
                         n=len(g), beta_G=float(slope), se=float(se),
                         r=float(r), p=float(p), boot_ci_lo=float(lo),
                         boot_ci_hi=float(hi)))
    out = pd.DataFrame(rows)
    out.to_csv(BETA_CSV, index=False)
    print(f"[analyze] -> {BETA_CSV}")
    # 按 G 排序展示 β_G（Gate 4 的核心输出）
    out = out.sort_values(["N", "G"])
    print("\n=== β_G vs G（Gate 4 判定）===")
    for _, r in out.iterrows():
        print(f"  {r['scene']:12s} N={r['N']} G={r['G']:.0f} n={r['n']:3d} "
              f"β_G={r['beta_G']:+.4f} CI=[{r['boot_ci_lo']:+.3f}, {r['boot_ci_hi']:+.3f}]")
    # 汇总：每 family 的 β_G 是否随 G 单调变负
    for fam, scns in [("A", [f"A_{x}" for x in ["prism4", "prism8", "prism16", "prism32", "cylinder"]]),
                      ("B", ["B_cube", "B_bevel05", "B_bevel15", "B_bevel30", "B_rounded"])]:
        sub = out[out["scene"].isin(scns)]
        if len(sub):
            # G 与 β_G 的 Spearman（跨 N 合并）
            for N in NS:
                sN = sub[sub["N"] == N].sort_values("G")
                if len(sN) >= 4:
                    sp = stats.spearmanr(sN["G"], sN["beta_G"])
                    print(f"  family {fam} N={N}: ρ(G, β_G) = {sp.statistic:+.3f} "
                          f"(p={sp.pvalue:.3f})  {len(sN)} levels")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["plan", "solve", "analyze"])
    ap.add_argument("--limit_cells", type=int, default=None)
    args = ap.parse_args()
    if args.stage == "plan":
        build_plan()
    elif args.stage == "solve":
        stage_solve(args.limit_cells)
    else:
        stage_analyze()


if __name__ == "__main__":
    main()
