"""R5-P2 held-out GSIQ (本机版, Phase 1 only)

任务书 §8-10:
  >=12 held-out scene, scene-family 多样性
  N={3,5,8} (本机版: {3,5}, N=8 单独跑)
  N=3 enumerate 4960, N=5 sample 2000

19 dev scene 切 6 in-domain + 13 held-out (满足任务书 >=12 要求)
"""
from __future__ import annotations
import argparse, csv, gc, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "p1" / "source" / "information_audit"))
from gauge_fisher_v2 import ga_isi_v2_scores, load_scene, scene_arrays

DATA_ROOT = REPO / "p1" / "calibration_set" / "data_sun_confirmatory"
ALL_SCENES = sorted([d.name for d in DATA_ROOT.iterdir()
                     if d.name.startswith("conf_") and d.is_dir()])
IN_DOMAIN = ["conf_sphere_r05", "conf_cube_axis", "conf_prism8",
             "conf_egg", "conf_cylinder_r06_d06", "conf_ellipsoid_z06"]
HELD_OUT = [s for s in ALL_SCENES if s not in IN_DOMAIN]  # 18 scene
SEED = 20260901


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=HELD_OUT,
                    help="默认 18 个 held-out scene (任务书 §8 >=12)")
    ap.add_argument("--ns", default="3,5")
    ap.add_argument("--n3_limit", type=int, default=4960)
    ap.add_argument("--n5_sample", type=int, default=2000)
    ap.add_argument("--pixel_cap", type=int, default=2000)
    ap.add_argument("--csv", default=str(REPO / "r5" / "r5_p2_heldout.csv"))
    args = ap.parse_args()

    import itertools
    K = 32
    NS = [int(x) for x in args.ns.split(",")]
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
            rng2 = np.random.default_rng(SEED + n)
            seen2 = set()
            sN = []
            while len(sN) < 2000:
                c = tuple(sorted(rng2.choice(K, n, replace=False).tolist()))
                if c not in seen2:
                    seen2.add(c); sN.append(list(c))
            subs[n] = sN

    REPO.joinpath("r5").mkdir(exist_ok=True)
    csv_path = Path(args.csv)
    csv_mode = "a" if csv_path.exists() else "w"

    print(f"P2 held-out GSIQ: {len(args.scenes)} scene × NS={NS} × P={args.pixel_cap}")
    print(f"  N=3 subsets: {len(subs[3])}, N=5: {len(subs.get(5, []))}")
    print(f"  total GSIQ calls: {sum(len(subs.get(n, [])) for n in NS) * len(args.scenes) * 2}")
    print(f"  csv: {csv_path} (mode={csv_mode})")
    print()

    fout = open(csv_path, csv_mode, newline="")
    wr = csv.DictWriter(fout, fieldnames=["scene","N","subset_id","subset_str","I_O","I_A"])
    if csv_mode == "w":
        wr.writeheader()

    t0 = time.time()
    n_skipped = 0
    for si, scene in enumerate(args.scenes):
        scene_dir = DATA_ROOT / scene
        # 跳过缺数据的 scene (e.g. R4§ 已知的 conf_pyramid6)
        sh_file = scene_dir / "sh_coeffs_irradiance.npy"
        alb_file = scene_dir / "albedo.npy"
        n_file = scene_dir / "normal_mesh.npy"
        if not all(f.exists() for f in (sh_file, alb_file, n_file)):
            print(f"[{si+1:2d}/{len(args.scenes)}] {scene}: 缺数据文件, 跳过",
                  flush=True)
            n_skipped += 1
            continue
        for n in NS:
            sublist = subs.get(n, [])
            t_scene = time.time()
            sc = load_scene(str(scene_dir))
            a_gt, Y, C_all = scene_arrays(sc, subset=list(range(sc["K"])),
                                          pixel_cap=args.pixel_cap, seed=SEED, fix_gauge=True)
            a_one = np.ones_like(a_gt)
            t0_n = time.time()
            for idx, s in enumerate(sublist):
                C = C_all[np.asarray(s, dtype=int)]
                rO = ga_isi_v2_scores(a_gt, Y, C)
                rA = ga_isi_v2_scores(a_one, Y, C)
                wr.writerow(dict(scene=scene, N=n, subset_id=idx,
                                 subset_str=",".join(str(x) for x in s),
                                 I_O=rO["full_logdet_pos_norm"],
                                 I_A=rA["full_logdet_pos_norm"]))
                # 每 1000 行 print 进度 + 强制 flush
                if (idx + 1) % 1000 == 0:
                    fout.flush()
                    elapsed_n = time.time() - t0_n
                    rate = (idx + 1) / elapsed_n
                    eta = (len(sublist) - idx - 1) / max(rate, 1e-3)
                    print(f"  [{si+1:2d}/{len(args.scenes)}] {scene:24s} N={n}  "
                          f"{idx+1}/{len(sublist)}  {rate:.0f}/s  ETA {eta:.0f}s",
                          flush=True)
            fout.flush()
            del sc, a_gt, Y, C_all, a_one
            gc.collect()
            elapsed = time.time() - t_scene
            overall = time.time() - t0
            print(f"[{si+1:2d}/{len(args.scenes)}] {scene:24s} N={n}  "
                  f"{len(sublist):4d} subsets  {elapsed:5.0f}s  (overall {overall:5.0f}s)",
                  flush=True)
    if n_skipped:
        print(f"\n[skip summary] 跳过 {n_skipped} scene (数据不完整)")
    fout.close()
    print(f"\n[Phase 1] done in {time.time()-t0:.0f}s")
    print(f"output: {csv_path}")


if __name__ == "__main__":
    main()
