"""R4″ Task G · Optimization 与 identifiability 解耦实验（任务书 §21）。

目的：回应 reviewer 最可能的攻击——"你的 Fisher metric 只是在预测
optimizer 哪次容易卡住"。若 information 在 GT-near initialization 下仍
预测 recovery quality，则 effect 不只是 global optimization artifact。

设计：
  对 6 个代表性 scene × N{3,5} × 10 subset × 2 init mode：
    global       : 现行初始化（softplus⁻¹(0.3) 常数 + c 0.01 噪声 + DC 0.3）
    oracle_local : θ₀ = θ_GT + δ，δ 固定相对幅度（albedo 5% RMS, c 5% norm）

输出：
  07_local_vs_global_init.csv
  判定：两 mode 下分别估 β(I→logError)；oracle_local 下 β 仍为负 ⇒
        effect 非纯 optimization artifact（Gate 6 PASS）

用法：python r4pp_local_vs_global.py [--limit N]
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

from gauge_fisher_v2 import fisher_blocks, schur_full, load_scene, scene_arrays  # noqa

DATA_ROOT = os.path.join(_REPO, "p1", "calibration_set", "data_sun_confirmatory")
OUT_DIR = os.path.join(_REPO, "r4pp")
SCENES = ["conf_cube_axis", "conf_prism8", "conf_cylinder_r03_d12",
          "conf_cone_r04_d12", "conf_egg", "conf_icosphere_sub3"]
NS = [3, 5]
N_SUBSETS = 10
SEED = 20260920
PERTURB_A = 0.05      # albedo 5% RMS
PERTURB_C = 0.05      # c 5% norm
RESTARTS = 1          # 解耦实验：单 restart 即可（θ₀ 给定）

OUT_CSV = os.path.join(OUT_DIR, "07_local_vs_global_init.csv")


def info_score(sc, sub, px, cap=1000):
    a, Y, C = scene_arrays(sc, sub, pixel_cap=cap, seed=px)
    bl = fisher_blocks(a, Y, C)
    F = schur_full(bl)
    w = np.linalg.eigvalsh(F)
    tr = w.sum()
    wn = w / tr if tr > 0 else w * 0
    pos = wn[wn > 1e-12]
    return float(np.log(pos).mean()) if pos.size else float("-inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from information_audit_v2 import joint_solve, si_mae_np  # lazy torch
    rng = np.random.default_rng(SEED)
    scenes = SCENES[:args.limit] if args.limit else SCENES

    done = set()
    if os.path.exists(OUT_CSV):
        for r in csv.DictReader(open(OUT_CSV, encoding="utf-8")):
            if r.get("N", "").lstrip("-").isdigit():
                done.add((r["scene"], int(r["N"]), r["subset"], r["init_mode"]))
    out_f = open(OUT_CSV, "a", newline="", encoding="utf-8")
    wr = None
    cache = {}
    t0 = time.time()
    n_new = 0
    for scn in scenes:
        if scn not in cache:
            cache.clear()
            cache[scn] = load_scene(os.path.join(DATA_ROOT, scn))
        sc = cache[scn]
        K = sc["imgs_lin"].shape[0]
        a_gt = sc["albedo"].astype(np.float64)
        for N in NS:
            subs = []
            seen = set()
            while len(subs) < N_SUBSETS:
                s = tuple(sorted(rng.choice(K, N, replace=False).tolist()))
                if s in seen:
                    continue
                seen.add(s)
                subs.append(list(s))
            for sub in subs:
                key = ",".join(map(str, sub))
                px = int(rng.integers(1 << 31))
                info = info_score(sc, sub, px)
                c_gt = sc["sh_irr"][np.asarray(sub)].astype(np.float64)
                # oracle-local θ₀ = θ_GT + δ（固定相对幅度）
                d_a = PERTURB_A * np.sqrt((a_gt[sc["mask"]] ** 2).mean())
                d_c = PERTURB_C * np.linalg.norm(c_gt) / math.sqrt(c_gt.size)
                theta0 = (np.clip(a_gt + rng.normal(0, d_a, a_gt.shape), 1e-4, None),
                          c_gt + rng.normal(0, d_c, c_gt.shape))

                for mode, th in [("global", None), ("oracle_local", theta0)]:
                    if (scn, N, key, mode) in done:
                        continue
                    r = joint_solve(sc, sub, restarts=RESTARTS, seed=px, theta0=th)
                    err = si_mae_np(r["A_hat"], sc["albedo"], sc["mask"])
                    rec = dict(scene=scn, N=N, subset=key, pixel_seed=px,
                               init_mode=mode, information=info,
                               reconstruction_error=err,
                               final_objective=r["final_loss"],
                               grad_norm=r["grad_norm"],
                               proj_grad_norm=r["proj_grad_norm"])
                    if wr is None:
                        wr = csv.DictWriter(out_f, fieldnames=list(rec))
                        wr.writeheader()
                    wr.writerow(rec)
                    n_new += 1
                out_f.flush()
        print(f"  [lg] {scn:26s} done ({time.time()-t0:.0f}s)", flush=True)
    out_f.close()
    print(f"[lg] +{n_new} runs -> {OUT_CSV}")


if __name__ == "__main__":
    main()
