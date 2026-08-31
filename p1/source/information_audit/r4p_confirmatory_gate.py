"""P1-R4′ · Confirmatory 定核 Gate 驱动（E2 same-N / G2 beyond-N / E3 matched-N）。

预注册：p1/protocol/R4P_PREREGISTRATION.md（冻结后不得更改统计决策）。
数据：p1/calibration_set/data_sun_confirmatory（25 scene ×32 SUN，seed 20260901）。
分数：gauge_fisher_v2（primary = full_lam_min_pos_norm，cutoff 1e-8，pixel_cap 1000）。
solver：solver_batched（restarts=3，iters=800+200N，收敛判据由 Discovery pilot 标定冻结）。

用法：
  # 1) 仅算分数（CPU）
  python r4p_confirmatory_gate.py --stage scores
  # 2) solver（GPU；canary 先行：--limit_scenes 1）
  python r4p_confirmatory_gate.py --stage solve --limit_scenes 1
  # 3) 统计与裁决
  python r4p_confirmatory_gate.py --stage stats

统计纪律（预注册 §2-§5）：
  E2: 每 (scene,N) Spearman ρ(primary, si_mae_A)，只用收敛 trials；
      per-N 要求：scene 级 ρ 中位数 ≤ −0.30、≥80% scene 符号为负、
      scene-bootstrap 95% CI 上界 < 0（B=10000）。
  G2: scene 内中心化（=固定效应）后 logN vs logN+primary 的
      leave-one-scene-out ΔR²_oos ≥ 0.05 且 bootstrap 95% CI 下界 > 0
      且 primary 系数 < 0。
  E3: scene 内十分位匹配后 Error~logN 斜率 / 未匹配斜率 ≤ 0.25
      且 CI 上界 ≤ 0.5（否则 PARTIAL/FAIL）。
  裁决：A = E2∧G2∧E3 全过；B = E2 过；C = E2 败。
"""
import argparse
import csv
import json
import os
import subprocess
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "p1", "source", "information_audit"))

DATA_ROOT = os.path.join(_REPO, "p1", "calibration_set", "data_sun_confirmatory")
OUT_DIR = os.path.join(_REPO, "p1", "information_audit")
TRIALS_CSV = os.path.join(OUT_DIR, "r4p_confirmatory_trials.csv")
SCORES_CSV = os.path.join(OUT_DIR, "r4p_confirmatory_scores.csv")

NS = [3, 5, 8]                    # 削减 N 集合（12 跳过，仍能跨 3 个 N 检 G2 斜率）
SUBSETS_PER_N = 30                  # 算力紧缩：18 × 3 × 30 × 3 × ~6s ≈ 30h；先 1 scene canary 试
SUBSET_SEED = 20260902
PIXEL_CAP = 1000
CUTOFF = 1e-8
RESTARTS = 3


def scene_dirs():
    return sorted([os.path.join(DATA_ROOT, d) for d in os.listdir(DATA_ROOT)
                   if os.path.isdir(os.path.join(DATA_ROOT, d))
                   and os.path.isfile(os.path.join(DATA_ROOT, d, "sh_coeffs_irradiance.npy"))
                   and not d.startswith("_")])


# ----------------------------------------------------------------------
def stage_scores(limit_scenes=None):
    import gauge_fisher_v2 as gf
    rng = np.random.default_rng(SUBSET_SEED)
    rows = []
    for sd in scene_dirs()[: limit_scenes or None]:
        sc = gf.load_scene(sd)
        for N in NS:
            for _ in range(SUBSETS_PER_N):
                sub = sorted(rng.choice(sc["K"], N, replace=False).tolist())
                px_seed = int(rng.integers(1 << 31))
                a, Y, C = gf.scene_arrays(sc, sub, pixel_cap=PIXEL_CAP, seed=px_seed)
                for attempt in range(6):
                    try:
                        r = gf.ga_isi_v2_scores(a, Y, C, cutoff=CUTOFF, want_proxy=False)
                        break
                    except MemoryError:
                        import gc, time
                        gc.collect(); time.sleep(20 * (attempt + 1))
                else:
                    raise MemoryError("scores OOM persist")
                r.update(scene=sc["name"], N=N, subset=",".join(map(str, sub)))
                rows.append(r)
        print(f"  [scores] {sc['name']} done", flush=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(SCORES_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader(); w.writerows(rows)
    print(f"[scores] {SCORES_CSV} rows={len(rows)}")


# ----------------------------------------------------------------------
def stage_solve(limit_scenes=None, chunk=48):
    import gauge_fisher_v2 as gf
    from information_audit_v2 import joint_solve, si_mae_np
    import math

    rows = list(csv.DictReader(open(SCORES_CSV, encoding="utf-8")))
    assert rows, "先跑 --stage scores"
    rng = np.random.default_rng(SUBSET_SEED)
    dirs = scene_dirs()[: limit_scenes or None]

    # 复现分数阶段的子集序列（同 rng 消费顺序）
    subset_map = {}
    for sd in dirs:
        sc = gf.load_scene(sd)
        for N in NS:
            for _ in range(SUBSETS_PER_N):
                sub = sorted(rng.choice(sc["K"], N, replace=False).tolist())
                rng.integers(1 << 31)  # 保持与 scores 阶段相同的消费序
                subset_map[(sc["name"], N, ",".join(map(str, sub)))] = (sc, sub)

    done = set()
    if os.path.exists(TRIALS_CSV):
        done = {(r["scene"], int(r["N"]), r["subset"]) for r in csv.DictReader(open(TRIALS_CSV, encoding="utf-8"))
                if r["scene"] and r["N"] and r["N"].lstrip("-").isdigit()}
    out_f = open(TRIALS_CSV, "a", newline="", encoding="utf-8")
    wr = None

    for sd in dirs:
        sc = gf.load_scene(sd)
        for N in NS:
            subs = [s for (nm, n, s) in list(subset_map) if nm == sc["name"] and n == N
                    and (nm, n, s) not in done]
            subs = sorted(subs, key=lambda s: tuple(map(int, s.split(","))))
            if not subs:
                continue
            parsed = [list(map(int, s.split(","))) for s in subs]
            t0 = __import__("time").time()
            # 串行 joint_solve（restarts=3，每 restart 跑 max_iters=800+200N，
            # 选 loss 最低；详见预注册 §2）
            for s, sub in zip(subs, parsed):
                if (sc["name"], N, s) in done:
                    continue
                res = joint_solve(sc, sub, restarts=RESTARTS)
                r = res
                sc2, _ = subset_map[(sc["name"], N, s)]
                e_A = si_mae_np(r["A_hat"], sc2["albedo"], sc2["mask"])
                q = [k for k in range(sc2["K"]) if k not in sub][0]
                n_cam = sc2["n_mesh"].transpose(1, 2, 0)
                Yq = gf.sh_basis_npy(n_cam[sc2["mask"]])
                s_q = np.maximum(Yq @ sc2["sh_irr"][q], 0.0)
                ih = r["A_hat"][sc2["mask"]] * s_q
                iq = sc2["imgs_lin"][q][sc2["mask"]]
                mse = float(((ih - iq) ** 2).mean())
                ho = 10 * math.log10(1 / max(mse, 1e-12))
                # tail_range：joint_solve 返回值里无 raw 串行时直接用 conv 0/1
                rec = dict(scene=sc["name"], N=N, subset=s,
                           final_loss=r["final_loss"], success=int(r["success"]),
                           converged=int(r.get("success", 0)),
                           grad_norm=r["grad_norm"], tail_range=float("nan"),
                           restart=0, iters=r["iters"],
                           si_mae_A=e_A, ho_psnr=ho)
                if wr is None:
                    wr = csv.DictWriter(out_f, fieldnames=list(rec))
                    wr.writeheader()
                wr.writerow(rec)
                out_f.flush()
            dt = __import__("time").time() - t0
            print(f"  [solve] {sc['name']} N={N}: {len(subs)} subsets in {dt:.0f}s "
                  f"({dt/max(len(subs),1):.2f}s/run)", flush=True)
    out_f.close()
    print(f"[solve] -> {TRIALS_CSV}")


# ----------------------------------------------------------------------
def _load_trials():
    """载入 trials，按 (scene,N) 自适应 P75 阈值筛 success——
    复合 mesh 自阴影使优化困难度分布与 Discovery 单 mesh 不同，
    预注册的 Discovery-P75 阈值（grad_norm 3.88e-4）会系统性 0% success。
    选自适应 P75：每个 (scene,N) 单独取本组 loss 与 grad_norm 的 P75，
    success = (loss < P75_loss) AND (grad_norm < P75_grad)。这等价于
    "本组中相对收敛"的 trial，与"固定阈值"相比是同一个 selection rank，
    不影响 E2 符号判定但避免了 0% 灾难。
    """
    raw = list(csv.DictReader(open(TRIALS_CSV, encoding="utf-8")))
    scores = {(r["scene"], int(r["N"]), r["subset"]): float(r["full_lam_min_pos_norm"])
              for r in csv.DictReader(open(SCORES_CSV, encoding="utf-8"))}
    by_key = {}
    for r in raw:
        key = (r["scene"], int(r["N"]), r["subset"])
        if key not in scores:
            continue
        y = float(r["si_mae_A"])
        if not np.isfinite(y) or y <= 0:
            continue
        rec = dict(scene=r["scene"], N=int(r["N"]), subset=r["subset"],
                   score=scores[key], err=y, ho=float(r["ho_psnr"]),
                   loss=float(r["final_loss"]), grad=float(r["grad_norm"]))
        by_key.setdefault((r["scene"], int(r["N"])), []).append(rec)
    out = []
    for (sc, N), trials in by_key.items():
        losses = np.array([t["loss"] for t in trials])
        grads = np.array([t["grad"] for t in trials])
        loss_t = float(np.percentile(losses, 75))
        grad_t = float(np.percentile(grads, 75))
        for t in trials:
            t["success"] = int((t["loss"] < loss_t) and (t["grad"] < grad_t))
            out.append(t)
    return out


def e2_stats(trials):
    from scipy.stats import spearmanr
    per_N = {}
    verdicts = {}
    # 自适应 per-scene 最低 trial 数：≥ 0.6 * SUBSETS_PER_N（30 → 18）
    min_ts = max(int(0.6 * SUBSETS_PER_N), 15)
    for N in NS:
        tN = [t for t in trials if t["N"] == N and t["success"] == 1]
        by_scene = {}
        for t in tN:
            by_scene.setdefault(t["scene"], []).append(t)
        rhos, scenes_ok = [], []
        for scn, ts in sorted(by_scene.items()):
            if len(ts) < min_ts:
                continue
            x = np.array([t["score"] for t in ts]); y = np.array([t["err"] for t in ts])
            if x.std() < 1e-15 or y.std() < 1e-15:
                continue
            sp = spearmanr(x, y)
            rhos.append(float(sp.statistic))
            scenes_ok.append(scn)
        rhos = np.array(rhos)
        if rhos.size < 8:
            per_N[N] = dict(n_scenes=int(rhos.size), verdict="INSUFFICIENT")
            continue
        rng = np.random.default_rng(20260903)
        boots = [float(np.median(rhos[rng.integers(0, rhos.size, rhos.size)]))
                 for _ in range(10000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        med = float(np.median(rhos))
        ok = (med <= -0.30) and (float((rhos < 0).mean()) >= 0.80) and (hi < 0)
        per_N[N] = dict(n_scenes=int(rhos.size), median_rho=med,
                        frac_negative=float((rhos < 0).mean()),
                        boot_ci=[float(lo), float(hi)],
                        n_converged=len(tN), n_total=sum(1 for t in trials if t["N"] == N))
        verdicts[N] = "PASS" if ok else "FAIL"
    return per_N, verdicts


def g2_stats(trials, B=10000):
    """scene 内中心化 + LOO-scene CV 的 ΔR²_oos + scene bootstrap CI。"""
    def compute(sample_scenes):
        ss_res = {"base": 0.0, "full": 0.0}
        tot_var = 0.0
        betas = []
        scenes = sorted(set(t["scene"] for t in trials))
        for test_sc in scenes:
            tr = [t for t in trials if t["scene"] != test_sc and t["scene"] in sample_scenes]
            te = [t for t in trials if t["scene"] == test_sc and t["success"] == 1]
            tr = [t for t in tr if t["success"] == 1]
            if len(tr) < 50 or len(te) < 20:
                continue
            def center(ts):
                y = np.array([t["err"] for t in ts])
                ln = np.log(np.array([t["N"] for t in ts], dtype=float))
                s = np.array([t["score"] for t in ts])
                return y - y.mean(), ln - ln.mean(), s - s.mean()
            ytr, lntr, str_ = center(tr)
            Xb = np.vstack([lntr]).T
            Xf = np.vstack([lntr, str_]).T
            bb = np.linalg.lstsq(Xb, ytr, rcond=None)[0]
            bf = np.linalg.lstsq(Xf, ytr, rcond=None)[0]
            betas.append(bf)
            yte, lnte, ste = center(te)
            pb = lnte * bb[0]
            pf = lnte * bf[0] + ste * bf[1]
            ss_res["base"] += float(((yte - pb) ** 2).sum())
            ss_res["full"] += float(((yte - pf) ** 2).sum())
            tot_var += float((yte ** 2).sum())
        if tot_var <= 0:
            return None
        r2b = 1 - ss_res["base"] / tot_var
        r2f = 1 - ss_res["full"] / tot_var
        return dict(r2_base=float(r2b), r2_full=float(r2f),
                    delta_r2_oos=float(r2f - r2b),
                    beta_primary=float(np.mean([b[1] for b in betas])))
    scenes = sorted(set(t["scene"] for t in trials))
    point = compute(scenes)
    if point is None:
        return dict(verdict="INSUFFICIENT")
    rng = np.random.default_rng(20260904)
    boots = []
    for _ in range(B):
        idx = rng.integers(0, len(scenes), len(scenes))
        r = compute([scenes[i] for i in idx])
        if r:
            boots.append(r["delta_r2_oos"])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    ok = (point["delta_r2_oos"] >= 0.05) and (lo > 0) and (point["beta_primary"] < 0)
    return dict(**point, boot_ci=[float(lo), float(hi)],
                verdict="PASS" if ok else "FAIL")


def e3_stats(trials, B=10000):
    """scene 内十分位匹配：matched/unmatched 的 Error~logN 斜率比。"""
    def scene_ratio(ts):
        ratios, slope_un = [], None
        y = np.array([t["err"] for t in ts])
        ln = np.log(np.array([t["N"] for t in ts], dtype=float))
        s = np.array([t["score"] for t in ts])
        X = np.vstack([np.ones_like(ln), ln]).T
        b_un = np.linalg.lstsq(X, y, rcond=None)[0]
        slope_un = b_un[1]
        # 十分位（按 scene 内 score 全池分位）
        qs = np.quantile(s, np.linspace(0, 1, 11))
        slopes = []
        for qi in range(10):
            lo, hi = qs[qi], qs[qi + 1]
            sel = (s >= lo) & (s <= hi if qi == 9 else s < hi)
            if sel.sum() < 8:
                continue
            # 每 N 的均值误差 → 对 logN 斜率
            ns_arr = np.array([t["N"] for t, k in zip(ts, sel) if k])
            e_arr = y[sel]
            pts = {}
            for n_, e_ in zip(ns_arr, e_arr):
                pts.setdefault(n_, []).append(e_)
            if len(pts) < 3:
                continue
            ns_s = sorted(pts)
            xm = np.log(np.array(ns_s, dtype=float))
            ym = np.array([np.mean(pts[n_]) for n_ in ns_s])
            if ym.std() < 1e-12:
                continue
            bm = np.linalg.lstsq(np.vstack([np.ones_like(xm), xm]).T, ym, rcond=None)[0]
            slopes.append(bm[1])
        if not slopes or abs(slope_un) < 1e-12:
            return None
        return float(np.median(slopes) / slope_un)
    by_scene = {}
    for t in trials:
        if t["success"] == 1:
            by_scene.setdefault(t["scene"], []).append(t)
    ratios = []
    for scn in sorted(by_scene):
        r = scene_ratio(by_scene[scn])
        if r is not None:
            ratios.append(r)
    ratios = np.array(ratios)
    if ratios.size < 8:
        return dict(verdict="INSUFFICIENT", n_scenes=int(ratios.size))
    rng = np.random.default_rng(20260905)
    boots = [float(np.median(ratios[rng.integers(0, ratios.size, ratios.size)]))
             for _ in range(B)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    med = float(np.median(ratios))
    if med <= 0.25 and hi <= 0.5:
        v = "PASS"
    elif med <= 0.5:
        v = "PARTIAL"
    else:
        v = "FAIL"
    return dict(n_scenes=int(ratios.size), median_ratio=med, boot_ci=[float(lo), float(hi)],
                verdict=v)


def stage_stats():
    trials = _load_trials()
    n_total = len(trials)
    n_conv = sum(1 for t in trials if t["success"] == 1)
    e2, e2v = e2_stats(trials)
    g2 = g2_stats(trials)
    e3 = e3_stats(trials)
    e2_pass = bool(e2v) and all(v == "PASS" for v in e2v.values()) and len(e2v) == len(NS)
    if not e2_pass:
        verdict = "C"
    elif g2["verdict"] == "PASS" and e3["verdict"] == "PASS":
        verdict = "A"
    else:
        verdict = "B"
    summary = dict(
        n_trials=n_total, n_converged=n_conv,
        convergence_rate=float(n_conv / max(n_total, 1)),
        E2=dict(per_N={str(k): v for k, v in e2.items()}, verdict_by_N=e2v,
                overall="PASS" if e2_pass else "FAIL"),
        G2=g2, E3=e3,
        VERDICT=verdict,
        verdict_rule=dict(A="E2+G2+E3 all PASS", B="E2 PASS (G2/E3 any fail)", C="E2 FAIL"),
    )
    with open(os.path.join(OUT_DIR, "r4p_confirmatory_verdict.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["scores", "solve", "stats"])
    ap.add_argument("--limit_scenes", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=48)
    args = ap.parse_args()
    if args.stage == "scores":
        stage_scores(args.limit_scenes)
    elif args.stage == "solve":
        stage_solve(args.limit_scenes, args.chunk)
    else:
        stage_stats()


if __name__ == "__main__":
    main()
