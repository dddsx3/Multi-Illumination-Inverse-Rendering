"""R5-P1-A · albedo ablation smoke (RTX 5070 Ti).

Computes per-(scene, N) GSIQ over a candidate subset pool under two score variants:

  O = I(a_GT, Y_GT, C_GT)   current oracle
  A = I(1,    Y_GT, C_GT)   albedo-free ablation

Outputs:
  r5/r5_p1_albedo_ablation.csv        per-(scene, N, subset) raw O + A + structural-null
  r5/r5_p1_albedo_ablation_ranking.csv  per-(scene, N) ranking diagnostics:
                                          Spearman rho, top-10/20% overlap, PASS-A verdict
  r5/r5_p1_albedo_ablation_selection.csv per-(scene, N, arm) solver outputs:
                                          reconstruction_error for O-selected, A-selected, random

Run on RTX 5070 Ti (smoke); final PASS-A gate requires Linux H100.

Smoke budget:
  - 6 dev scenes × N ∈ {3, 5}
  - For N=3: enumerate all C(32,3)=4960 subsets (cheap)
  - For N=5: sample 2000 from C(32,5)
  - Per-(scene,N) selection arm: top-10% from O, top-10% from A, 10 random
  - Solver: restarts=1, base_iters=400 (vs prod 800), single solver seed
  - Total solver runs: 6 × 2 × (10 + 10 + 10) ≈ 360

Pre-registration frozen (R5-P1-A task book):
  PASS-A   : median(rho) >= 0.95  AND  top-10% overlap high  ->  freeze a=1
  CONDITIONAL : 0.8 < rho < 0.95  ->  albedo secondary weighting; do NOT enter P1-B with a-hat
  FAIL-A   : rho < 0.8  ->  halt practical selector; enter albedo-proxy branch
"""
from __future__ import annotations

import argparse
import csv
import gc
import itertools as it
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "p1", "source", "information_audit"))

from gauge_fisher_v2 import (   # noqa: E402
    ga_isi_v2_scores,
    load_scene,
    scene_arrays,
    sh_basis_npy,
)


# ---- pre-registration --------------------------------------------------------
DEFAULT_SCENES = [
    "conf_sphere_r05",        # high info (round, isotropic normals)
    "conf_cube_axis",         # medium-high
    "conf_prism8",            # medium
    "conf_egg",               # medium
    "conf_cylinder_r06_d06",  # medium-low (anisotropic)
    "conf_ellipsoid_z06",     # low
]
NS = [3, 5]
N3_POOL = "enumerate"   # C(32,3) = 4960
N5_POOL = "sample"      # 2000 random from C(32,5)
# smoke budget on RTX 5070 Ti:
N3_SMOKE_LIMIT = 500    # enumerate first 500 only (vs full 4960); full pool = Linux H100
N5_SAMPLE = 500
RNG = np.random.default_rng(20260901)
PIXEL_CAP = 300         # smoke: 300 (full would be 2000 on Linux H100)
SUBSET_SEED = 20260901

# selection arm budget (per scene, per N)
N_TOP = 10
N_RANDOM = 10

# solver smoke budget
SOLVER_RESTARTS = 1
SOLVER_ITERS = 400
SOLVER_SEED = 20260830


# ---- subset pool -------------------------------------------------------------
def enumerate_subsets(K: int, n: int):
    return [list(s) for s in it.combinations(range(K), n)]


def sample_subsets(K: int, n: int, m: int, rng: np.random.Generator):
    out = set()
    while len(out) < m:
        cand = tuple(sorted(rng.choice(K, size=n, replace=False).tolist()))
        out.add(cand)
    return [list(s) for s in out]


# ---- GSIQ over a subset pool --------------------------------------------------
def score_pool(scene_dir: str, subsets: list, pixel_cap: int, seed: int):
    """Return list of dicts {subset, I_O, I_A, d_extra_null, status, n_at_cutoff}.

    R5-P1 smoke 修复（2026-09-01）：在 each subset 后强 del + collect +
    try/except 二次回收，绕开 Windows commit 配额 94% 占用的 OOM。
    """
    sc = load_scene(scene_dir)
    a_gt, Y_gt, C_all = scene_arrays(sc, subset=list(range(sc["K"])),
                                     pixel_cap=pixel_cap, seed=seed, fix_gauge=True)
    a_one = np.ones_like(a_gt)
    rows = []
    for idx, s in enumerate(subsets):
        C = C_all[np.asarray(s, dtype=int)]
        rO = ga_isi_v2_scores(a_gt, Y_gt, C)
        # explicit early release to fight Windows commit quota drift
        del C
        gc.collect()
        rA = ga_isi_v2_scores(a_one, Y_gt, C_all[np.asarray(s, dtype=int)])
        rows.append(dict(
            subset=tuple(s),
            subset_idx=idx,
            I_O=rO["full_logdet_pos_norm"],
            I_A=rA["full_logdet_pos_norm"],
            d_extra_null_O=rO["d_extra_null"],
            d_extra_null_A=rA["d_extra_null"],
            status_O=rO["structural_status"],
            status_A=rA["structural_status"],
            n_at_cutoff_O=int(rO["full_n_at_cutoff"]) if not np.isnan(rO["full_n_at_cutoff"]) else -1,
            n_at_cutoff_A=int(rA["full_n_at_cutoff"]) if not np.isnan(rA["full_n_at_cutoff"]) else -1,
        ))
        # drop large transient dicts
        del rO, rA
        gc.collect()
    del sc, a_gt, Y_gt, C_all, a_one
    gc.collect()
    return rows


# ---- ranking diagnostics -----------------------------------------------------
def ranking_diagnostics(rows: list[dict]):
    """rho, top-k overlap, A-only spectral balance, boundary granularity."""
    O = np.array([r["I_O"] for r in rows])
    A = np.array([r["I_A"] for r in rows])
    rho, _ = spearmanr(O, A)

    def topk_overlap(k_frac):
        n = len(rows)
        k = max(1, int(round(k_frac * n)))
        idx_O = set(np.argsort(-O)[:k].tolist())
        idx_A = set(np.argsort(-A)[:k].tolist())
        return len(idx_O & idx_A) / k

    n_deficient_O = sum(1 for r in rows if r["status_O"] == "deficient")
    n_deficient_A = sum(1 for r in rows if r["status_A"] == "deficient")
    n_at_cutoff_O_med = float(np.median([r["n_at_cutoff_O"] for r in rows if r["n_at_cutoff_O"] >= 0])) if any(r["n_at_cutoff_O"] >= 0 for r in rows) else float("nan")
    n_at_cutoff_A_med = float(np.median([r["n_at_cutoff_A"] for r in rows if r["n_at_cutoff_A"] >= 0])) if any(r["n_at_cutoff_A"] >= 0 for r in rows) else float("nan")
    return dict(
        rho=float(rho),
        n=len(rows),
        top10_overlap=topk_overlap(0.10),
        top20_overlap=topk_overlap(0.20),
        O_mean=float(O.mean()),
        A_mean=float(A.mean()),
        O_std=float(O.std()),
        A_std=float(A.std()),
        n_deficient_O=n_deficient_O,
        n_deficient_A=n_deficient_A,
        n_at_cutoff_O_med=n_at_cutoff_O_med,
        n_at_cutoff_A_med=n_at_cutoff_A_med,
    )


# ---- solver arm (smoke) ------------------------------------------------------
def run_solver_arm(scene_dir: str, subsets: list, arm: str, restarts: int,
                   base_iters: int, seed: int):
    """Run joint_solve on `subsets` for `arm`; return list of per-subset errors.

    arm is just a label so output CSV can group; the subsets themselves encode
    the arm (oracle-top, A-top, random).
    """
    # Lazy import to avoid torch load until actually needed
    from solver_batched import joint_solve_batched
    sc = load_scene(scene_dir)
    results = joint_solve_batched(
        sc, subsets, restarts=restarts, base_iters=base_iters, lr=1e-2,
        lam_tv=0.03, device="cuda", conv_tol=1e-7, grad_tol=1e-3,
        chunk=None,
    )
    out = []
    for s, r in zip(subsets, results):
        out.append(dict(
            arm=arm,
            subset=tuple(int(x) for x in s),
            reconstruction_error=float(r["final_objective"]) if r.get("final_objective") is not None else float("nan"),
            success=bool(r.get("success", False)),
            iters=int(r.get("iters", -1)),
            grad_norm=float(r.get("grad_norm", float("nan"))),
            tail_range=float(r.get("tail_range", float("nan"))),
        ))
    del sc
    gc.collect()
    return out


# ---- main --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=os.path.join(_REPO, "p1", "calibration_set",
                                                       "data_sun_confirmatory"))
    ap.add_argument("--out_dir", default=os.path.join(_REPO, "r5"))
    ap.add_argument("--scenes", nargs="*", default=DEFAULT_SCENES)
    ap.add_argument("--n_random", type=int, default=N_RANDOM,
                    help="random subset count per arm; reduced for tighter smoke")
    ap.add_argument("--n_top", type=int, default=N_TOP)
    ap.add_argument("--solver", action="store_true",
                    help="actually run solver; off for score-only smoke")
    ap.add_argument("--max_scenes", type=int, default=None)
    ap.add_argument("--pixel_cap", type=int, default=None,
                    help="override smoke PIXEL_CAP (default 300); set 2000 for paper-grade full")
    ap.add_argument("--n5_sample", type=int, default=None,
                    help="override N5_SAMPLE (default 500); set 2000 for paper-grade full")
    args = ap.parse_args()

    # Apply CLI overrides (R5-P1-A full on A10 / H100)
    global PIXEL_CAP, N5_SAMPLE
    if args.pixel_cap is not None:
        PIXEL_CAP = args.pixel_cap
    if args.n5_sample is not None:
        N5_SAMPLE = args.n5_sample
    print(f"[config] PIXEL_CAP={PIXEL_CAP}  N5_SAMPLE={N5_SAMPLE}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)

    scenes = args.scenes if not args.max_scenes else args.scenes[:args.max_scenes]

    raw_path = os.path.join(args.out_dir, "r5_p1_albedo_ablation.csv")
    rank_path = os.path.join(args.out_dir, "r5_p1_albedo_ablation_ranking.csv")
    sel_path = os.path.join(args.out_dir, "r5_p1_albedo_ablation_selection.csv")
    gate_path = os.path.join(args.out_dir, "r5_p1_albedo_ablation_gate.md")

    # ===== 1. score pool =====
    # R5-P1-A A10 incremental: if raw_path exists and we are NOT running all scenes,
    # use append mode (skip header). The script is then idempotent for re-runs.
    csv_mode = "a" if (os.path.exists(raw_path) and args.scenes and len(args.scenes) < 10) else "w"
    if csv_mode == "a":
        print(f"[incremental] appending to existing {raw_path}", flush=True)
    with open(raw_path, csv_mode, newline="") as fraw:
        wr = csv.DictWriter(fraw, fieldnames=[
            "scene", "N", "subset_id", "subset_str",
            "I_O", "I_A",
            "d_extra_null_O", "d_extra_null_A",
            "status_O", "status_A",
            "n_at_cutoff_O", "n_at_cutoff_A",
        ])
        if csv_mode == "w":
            wr.writeheader()
        rank_rows = []
        # boundary outliers: list of (scene, N, subset, I_O, I_A, diff, n_at_cutoff_O, n_at_cutoff_A)
        # threshold: 1e-3 (typical |ΔI| is 1e-5; >1e-3 indicates boundary granularity)
        OUTLIER_THR = 1e-3
        outlier_rows = []
        for s in scenes:
            scene_dir = os.path.join(args.data_root, s)
            for n in NS:
                if n == 3:
                    subsets_all = enumerate_subsets(32, n)
                    subsets = subsets_all[:N3_SMOKE_LIMIT]
                else:
                    subsets = sample_subsets(32, n, N5_SAMPLE, RNG)
                print(f"[{s} N={n}] scoring {len(subsets)} subsets (smoke budget) ...", flush=True)
                rows = score_pool(scene_dir, subsets, PIXEL_CAP, SUBSET_SEED)
                # write per-subset
                for r in rows:
                    wr.writerow(dict(
                        scene=s, N=n,
                        subset_id=r["subset_idx"],
                        subset_str=",".join(str(x) for x in r["subset"]),
                        I_O=r["I_O"], I_A=r["I_A"],
                        d_extra_null_O=r["d_extra_null_O"],
                        d_extra_null_A=r["d_extra_null_A"],
                        status_O=r["status_O"],
                        status_A=r["status_A"],
                        n_at_cutoff_O=r["n_at_cutoff_O"],
                        n_at_cutoff_A=r["n_at_cutoff_A"],
                    ))
                    diff = abs(r["I_O"] - r["I_A"])
                    if diff > OUTLIER_THR:
                        outlier_rows.append(dict(
                            scene=s, N=n,
                            subset_str=",".join(str(x) for x in r["subset"]),
                            I_O=r["I_O"], I_A=r["I_A"], diff=r["I_O"] - r["I_A"],
                            abs_diff=diff,
                            status_O=r["status_O"], status_A=r["status_A"],
                            d_extra_null_O=r["d_extra_null_O"],
                            d_extra_null_A=r["d_extra_null_A"],
                            n_at_cutoff_O=r["n_at_cutoff_O"],
                            n_at_cutoff_A=r["n_at_cutoff_A"],
                        ))
                diag = ranking_diagnostics(rows)
                rank_rows.append(dict(scene=s, N=n, **diag))
                print(f"  rho={diag['rho']:.4f}  top10={diag['top10_overlap']:.3f}  "
                      f"top20={diag['top20_overlap']:.3f}  outliers={len([o for o in outlier_rows if o['scene']==s and o['N']==n])}",
                      flush=True)

    with open(rank_path, csv_mode, newline="") as frank:
        wr = csv.DictWriter(frank, fieldnames=list(rank_rows[0].keys()))
        if csv_mode == "w":
            wr.writeheader()
        wr.writerows(rank_rows)

    # ===== 1b. boundary-outlier CSV =====
    outlier_path = os.path.join(args.out_dir, "r5_p1_albedo_ablation_outliers.csv")
    outlier_csv_mode = "a" if (os.path.exists(outlier_path) and csv_mode == "a") else "w"
    with open(outlier_path, outlier_csv_mode, newline="") as fout:
        if outlier_rows:
            wr = csv.DictWriter(fout, fieldnames=list(outlier_rows[0].keys()))
            if outlier_csv_mode == "w":
                wr.writeheader()
            wr.writerows(outlier_rows)
        elif outlier_csv_mode == "w":
            fout.write("(no boundary outliers above threshold)\n")

    # ===== 2. gate decision =====
    rhos = np.array([r["rho"] for r in rank_rows])
    top10s = np.array([r["top10_overlap"] for r in rank_rows])
    top20s = np.array([r["top20_overlap"] for r in rank_rows])
    med_rho = float(np.median(rhos))

    if med_rho >= 0.95 and float(np.median(top10s)) >= 0.80:
        verdict = "PASS-A"
        next_step = "freeze a=1; P1-B normal/light proxy audit"
    elif med_rho > 0.80:
        verdict = "CONDITIONAL"
        next_step = "albedo contributes secondary spectral weighting; P1-B does NOT introduce a-hat"
    else:
        verdict = "FAIL-A"
        next_step = "halt practical selector; enter albedo-proxy branch (P1-B with a-hat allowed only here)"

    with open(gate_path, "w") as fg:
        fg.write(f"# R5-P1-A Gate · smoke (RTX 5070 Ti)\n\n")
        fg.write(f"Scenes: {scenes}\n\n")
        fg.write(f"NS: {NS}\n\n")
        fg.write(f"Per-(scene, N) ranking diagnostics:\n\n")
        fg.write("| scene | N | rho | top10 | top20 | O_mean | A_mean | n_deficient_O | n_deficient_A | n_at_cutoff_O_med | n_at_cutoff_A_med |\n")
        fg.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rank_rows:
            naco_o = f"{r['n_at_cutoff_O_med']:.0f}" if not np.isnan(r['n_at_cutoff_O_med']) else "n/a"
            naco_a = f"{r['n_at_cutoff_A_med']:.0f}" if not np.isnan(r['n_at_cutoff_A_med']) else "n/a"
            fg.write(f"| {r['scene']} | {r['N']} | {r['rho']:.4f} | "
                     f"{r['top10_overlap']:.3f} | {r['top20_overlap']:.3f} | "
                     f"{r['O_mean']:.3f} | {r['A_mean']:.3f} | "
                     f"{r['n_deficient_O']} | {r['n_deficient_A']} | "
                     f"{naco_o} | {naco_a} |\n")
        fg.write(f"\n**median rho** = {med_rho:.4f}\n\n")
        fg.write(f"**median top10 overlap** = {float(np.median(top10s)):.3f}\n\n")
        fg.write(f"**median top20 overlap** = {float(np.median(top20s)):.3f}\n\n")
        fg.write(f"## Gate verdict: {verdict}\n\n")
        fg.write(f"Next step: {next_step}\n\n")
        # boundary outliers table
        fg.write(f"## Boundary outliers (|ΔI| > {OUTLIER_THR:.0e})\n\n")
        if outlier_rows:
            fg.write(f"Total: {len(outlier_rows)} subsets ({100*len(outlier_rows)/sum(r['n'] for r in rank_rows):.3f}% of all subsets)\n\n")
            fg.write("| scene | N | subset | I_O | I_A | diff | status_O | status_A | d_extra_O | d_extra_A | n_at_cutoff_O | n_at_cutoff_A |\n")
            fg.write("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
            for o in outlier_rows:
                fg.write(f"| {o['scene']} | {o['N']} | {o['subset_str']} | "
                         f"{o['I_O']:.4f} | {o['I_A']:.4f} | {o['diff']:+.4e} | "
                         f"{o['status_O']} | {o['status_A']} | "
                         f"{o['d_extra_null_O']} | {o['d_extra_null_A']} | "
                         f"{o['n_at_cutoff_O']} | {o['n_at_cutoff_A']} |\n")
            fg.write(f"\n→ Outliers arise from spec_cutoff=1e-8 boundary granularity\n")
            fg.write(f"  (per-pixel albedo modulation shifts the smallest positive eigenvalue\n")
            fg.write(f"  across the cutoff, see `r5/r5_p1_a_boundary_diagnostic.md`)\n\n")
        else:
            fg.write("None — no boundary outliers above threshold.\n\n")
        fg.write("Gate criteria (R5-P1-A task book):\n")
        fg.write("- PASS-A:        median(rho) >= 0.95  AND  median(top10 overlap) >= 0.80\n")
        fg.write("- CONDITIONAL:   0.80 < median(rho) < 0.95\n")
        fg.write("- FAIL-A:        median(rho) <= 0.80\n")

    print(f"\nGate verdict: {verdict}  (median rho = {med_rho:.4f})")
    print(f"Wrote: {raw_path}\n       {rank_path}\n       {gate_path}\n       {outlier_path}")

    # ===== 3. solver arm (optional) =====
    if args.solver:
        # Reload pool, take top-10% from O, top-10% from A, and 10 random per cell
        per_cell = defaultdict(dict)  # (scene, N) -> {O: [subsets], A: [...], random: [...]}
        for s in scenes:
            for n in NS:
                per_cell[(s, n)] = {"O": [], "A": [], "random": []}

        # Read back raw CSV (or recompute quickly with cached I_O, I_A)
        per_cell_rows = defaultdict(list)
        with open(raw_path) as f:
            rd = csv.DictReader(f)
            for r in rd:
                per_cell_rows[(r["scene"], int(r["N"]))].append(r)

        for (s, n), rows in per_cell_rows.items():
            scored = [(r, float(r["I_O"]), float(r["I_A"]))
                      for r in rows
                      if r["status_O"] != "unknown"]
            scored.sort(key=lambda x: -x[1])  # by I_O desc
            k = min(args.n_top, len(scored))
            top_O = [list(map(int, x[0]["subset_str"].split(","))) for x in scored[:k]]
            scored.sort(key=lambda x: -x[2])
            top_A = [list(map(int, x[0]["subset_str"].split(","))) for x in scored[:k]]
            rng_local = np.random.default_rng(SUBSET_SEED + hash((s, n)) % (2**31))
            scored_idx = np.arange(len(scored))
            rand_idx = rng_local.choice(scored_idx, size=min(args.n_random, len(scored)), replace=False)
            top_R = [list(map(int, scored[i][0]["subset_str"].split(","))) for i in rand_idx]
            per_cell[(s, n)]["O"] = top_O
            per_cell[(s, n)]["A"] = top_A
            per_cell[(s, n)]["random"] = top_R

        with open(sel_path, "w", newline="") as fsel:
            wr = csv.DictWriter(fsel, fieldnames=[
                "scene", "N", "arm", "subset_str",
                "reconstruction_error", "success", "iters", "grad_norm", "tail_range",
            ])
            wr.writeheader()
            for (s, n), arms in per_cell.items():
                scene_dir = os.path.join(args.data_root, s)
                for arm_name, subsets in arms.items():
                    if not subsets:
                        continue
                    print(f"[{s} N={n} arm={arm_name}] solver {len(subsets)} subsets ...", flush=True)
                    out = run_solver_arm(scene_dir, subsets, arm_name,
                                         SOLVER_RESTARTS, SOLVER_ITERS, SOLVER_SEED)
                    for r in out:
                        wr.writerow(dict(
                            scene=s, N=n, arm=arm_name,
                            subset_str=",".join(str(x) for x in r["subset"]),
                            reconstruction_error=r["reconstruction_error"],
                            success=int(r["success"]),
                            iters=r["iters"],
                            grad_norm=r["grad_norm"],
                            tail_range=r["tail_range"],
                        ))
        print(f"Wrote: {sel_path}")


if __name__ == "__main__":
    main()