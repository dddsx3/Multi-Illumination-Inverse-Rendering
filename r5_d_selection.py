"""R5-D: C3 selection preservation (本机版, 验证 Case 2 wording 下 selection 收益)

任务书 §14 简化版:
  对每个 (scene, N) cell, 取 GSIQ top-10% 子集 (proxy-selected) + random 等量子集
  跑 joint_solve_batched, 对比两种选择的平均 reconstruction_error
  proxy < random → selection 收益成立

本机版预算: P2 12 held-out scene × 1 N (N=3) × (50 top + 50 random) = 1200 runs
  @ 0.6s/run = ~12 min

输出: r5/r5_d_selection.csv
  含 (scene, N, arm, subset_id, reconstruction_error)
"""
from __future__ import annotations
import argparse, csv, gc, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "p1" / "source" / "information_audit"))
from gauge_fisher_v2 import load_scene, scene_arrays  # noqa
from solver_batched import joint_solve_batched  # noqa

DATA_ROOT = REPO / "p1" / "calibration_set" / "data_sun_confirmatory"
SEED = 20260901
N = 3
N_PER_ARM = 50
SOLVER_ITERS = 400
SOLVER_RESTARTS = 1
SOLVER_SEED_BASE = 20260901


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p2_csv", default=str(REPO / "r5" / "r5_p2_heldout.csv"))
    ap.add_argument("--scenes", nargs="*", default=None,
                    help="默认从 p2_csv 取所有 held-out scene")
    ap.add_argument("--n_per_arm", type=int, default=N_PER_ARM)
    ap.add_argument("--solver_iters", type=int, default=SOLVER_ITERS)
    ap.add_argument("--out", default=str(REPO / "r5" / "r5_d_selection.csv"))
    ap.add_argument("--limit_scenes", type=int, default=0,
                    help="限制 scene 数 (0 = 全跑, 适合调试)")
    args = ap.parse_args()

    p2 = pd.read_csv(args.p2_csv)
    p2 = p2[p2["N"] == N]
    scenes = sorted(p2.scene.unique()) if args.scenes is None else args.scenes
    if args.limit_scenes > 0:
        scenes = scenes[:args.limit_scenes]
    print(f"D: selection preservation on {len(scenes)} scene × N={N}")
    print(f"  每 (scene, N) cell: {args.n_per_arm} proxy-top + {args.n_per_arm} random = {2*args.n_per_arm} solver runs")
    print(f"  Total solver runs: {len(scenes) * 2 * args.n_per_arm}")
    print(f"  output: {args.out}")
    print()

    out = open(args.out, "w", newline="")
    wr = csv.DictWriter(out, fieldnames=["scene", "N", "arm", "subset_id", "subset_str",
                                          "reconstruction_error", "success"])
    wr.writeheader()

    t0 = time.time()
    for si, scene in enumerate(scenes):
        # 取 top-N (按 I_O 排名) 和 random
        cell = p2[p2.scene == scene].sort_values("I_O", ascending=False)
        top = cell.head(args.n_per_arm)
        rng = np.random.default_rng(SOLVER_SEED_BASE + hash(scene) % 10000)
        random_idx = rng.choice(cell.index, size=min(args.n_per_arm, len(cell)),
                                replace=False)
        random_arm = cell.loc[random_idx]

        scene_dir = DATA_ROOT / scene
        sc = load_scene(str(scene_dir))
        a_gt, Y, C_all = scene_arrays(sc, subset=list(range(sc["K"])),
                                      pixel_cap=1000, seed=SEED, fix_gauge=True)
        for arm_name, sub_df in [("proxy_top", top), ("random", random_arm)]:
            t_arm = time.time()
            for _, row in sub_df.iterrows():
                subset = [int(x) for x in row.subset_str.split(",")]
                C = C_all[subset]
                res = joint_solve_batched(
                    sc, [subset], restarts=SOLVER_RESTARTS,
                    base_iters=args.solver_iters, lr=1e-2, lam_tv=0.03, device="cuda",
                )
                wr.writerow(dict(
                    scene=scene, N=N, arm=arm_name, subset_id=row.subset_id,
                    subset_str=",".join(str(x) for x in subset),
                    reconstruction_error=float(res[0]["final_loss"]),
                    success=int(bool(res[0].get("success", False))),
                ))
            out.flush()
            elapsed = time.time() - t_arm
            print(f"[{si+1:2d}/{len(scenes)}] {scene:24s} {arm_name}  "
                  f"{len(sub_df):3d} runs  {elapsed:5.0f}s  (overall {time.time()-t0:5.0f}s)",
                  flush=True)
        del sc, a_gt, Y, C_all
        gc.collect()
        torch = None  # placeholder
    out.close()
    print(f"\n[Phase D] done in {time.time()-t0:.0f}s = {(time.time()-t0)/60:.1f} min")
    print(f"output: {args.out}")


if __name__ == "__main__":
    main()
