"""R5-P1-A FULL 本机版

任务书 §8-12 (P1-A 阶段):
  - 6 dev scene × NS={3,5} × pixel_cap=2000 (论文级精度)
  - N=3 enumerate 4960, N=5 sample 2000
  - 2 score variants: O (a_GT) + A (a=1)
  - solver arm: top-10% O + top-10% A + random 10 per cell

本机版预算 (P=2000, 8 vCPU, 1.5s/call):
  - GSIQ: 6 * (4960+2000) * 2 = 83520 calls, ~5-6 h
  - solver: 6 * 2 * 30 = 360 runs @ 0.6s/run = 4 min
  - 总: ~6 h wall-clock, 0 云算力

设计: incremental append (每个 cell 完成后立即 flush; 中断可恢复)
输出: r5/r5_p1_albedo_ablation_full.csv (合并 smoke 6000 + full 77520 rows)
"""
from __future__ import annotations
import argparse, csv, gc, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "p1" / "source" / "information_audit"))
from gauge_fisher_v2 import ga_isi_v2_scores, load_scene, scene_arrays

DATA_ROOT = REPO / "p1" / "calibration_set" / "data_sun_confirmatory"
SCENES = ["conf_sphere_r05", "conf_cube_axis", "conf_prism8",
          "conf_egg", "conf_cylinder_r06_d06", "conf_ellipsoid_z06"]
NS = [3, 5]
SEED = 20260901


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pixel_cap", type=int, default=2000)
    ap.add_argument("--n3_limit", type=int, default=4960)
    ap.add_argument("--n5_sample", type=int, default=2000)
    ap.add_argument("--csv", default=str(REPO / "r5" / "r5_p1_albedo_ablation_full.csv"))
    ap.add_argument("--n_top", type=int, default=10)
    ap.add_argument("--n_random", type=int, default=10)
    ap.add_argument("--solver", action="store_true", default=True)
    ap.add_argument("--no-solver", dest="solver", action="store_false")
    ap.add_argument("--only_scenes", nargs="*", default=None,
                    help="只跑这几个 scene (默认 6 个全跑)")
    args = ap.parse_args()

    import itertools
    K = 32
    n3 = [list(s) for s in itertools.combinations(range(K), 3)][:args.n3_limit]
    rng = np.random.default_rng(SEED)
    seen = set()
    n5 = []
    while len(n5) < args.n5_sample:
        cand = tuple(sorted(rng.choice(K, 5, replace=False).tolist()))
        if cand not in seen:
            seen.add(cand)
            n5.append(list(cand))
    subs = {3: n3, 5: n5}
    for n in NS:
        if n not in subs:
            seen2 = set(); sN = []
            rng2 = np.random.default_rng(SEED + n)
            while len(sN) < 2000:
                c = tuple(sorted(rng2.choice(K, n, replace=False).tolist()))
                if c not in seen2: seen2.add(c); sN.append(list(c))
            subs[n] = sN

    scenes = args.only_scenes or SCENES
    print(f"P1-A full: {len(scenes)} scene × NS={NS} × P={args.pixel_cap}")
    print(f"  N=3 subsets: {len(subs[3])}, N=5: {len(subs[5])}")
    print(f"  2 score variants (O, A)")
    print(f"  Total GSIQ calls: {sum(len(subs[n]) for n in NS) * len(scenes) * 2}")
    print(f"  csv: {args.csv}")
    print(f"  estimated wall-clock: ~{(len(scenes) * (len(subs[3]) + len(subs[5])) * 2 * 0.0015 / 3600):.1f} h (assuming 1.5s/call)")
    print()

    csv_path = Path(args.csv)
    csv_mode = "a" if csv_path.exists() else "w"
    print(f"csv mode: {csv_mode}")
    fout = open(csv_path, csv_mode, newline="")
    wr = csv.DictWriter(fout, fieldnames=["scene","N","subset_id","subset_str","I_O","I_A"])
    if csv_mode == "w":
        wr.writeheader()

    t0 = time.time()
    n_skipped = 0
    for si, scene in enumerate(scenes):
        scene_dir = DATA_ROOT / scene
        for n in NS:
            sublist = subs[n]
            t_scene = time.time()
            sc = load_scene(str(scene_dir))
            a_gt, Y, C_all = scene_arrays(sc, subset=list(range(sc["K"])),
                                          pixel_cap=args.pixel_cap, seed=SEED, fix_gauge=True)
            a_one = np.ones_like(a_gt)
            for idx, s in enumerate(sublist):
                C = C_all[np.asarray(s, dtype=int)]
                rO = ga_isi_v2_scores(a_gt, Y, C)
                rA = ga_isi_v2_scores(a_one, Y, C)
                wr.writerow(dict(scene=scene, N=n, subset_id=idx,
                                 subset_str=",".join(str(x) for x in s),
                                 I_O=rO["full_logdet_pos_norm"],
                                 I_A=rA["full_logdet_pos_norm"]))
                if (idx + 1) % 1000 == 0:
                    fout.flush()
                    elapsed_n = time.time() - t_scene
                    rate = (idx + 1) / elapsed_n
                    eta = (len(sublist) - idx - 1) / max(rate, 1e-3)
                    print(f"  [{si+1:2d}/{len(scenes)}] {scene:24s} N={n}  "
                          f"{idx+1}/{len(sublist)}  {rate:.0f}/s  ETA {eta:.0f}s",
                          flush=True)
            fout.flush()
            del sc, a_gt, Y, C_all, a_one
            gc.collect()
            elapsed = time.time() - t_scene
            overall = time.time() - t0
            print(f"[{si+1:2d}/{len(scenes)}] {scene:24s} N={n}  "
                  f"{len(sublist):4d} subsets  {elapsed:5.0f}s  (overall {overall:5.0f}s)",
                  flush=True)
    fout.close()
    print(f"\n[Phase 1] GSIQ done in {time.time()-t0:.0f}s = {(time.time()-t0)/3600:.1f} h")
    print(f"output: {csv_path}")


if __name__ == "__main__":
    main()
