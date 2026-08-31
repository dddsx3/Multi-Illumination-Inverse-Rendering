"""R4′ 诊断交付物生成器（INC-002 取证用；不改任何度量/阈值，纯观测）。

产出四份文件到 p1/information_audit/diagnostics/：
  1. r4p_trial_eigenspectrum.csv   每 (scene,N,subset) 的 F_eff 归一化全谱摘要
  2. r4p_raw_trials_joined.csv     未筛选 raw trial ⊕ scores 全字段 join
  3. r4p_scene_gram_spectrum.csv   每 scene 的 normal 协方差谱 + SH Gram 谱
  4. （rank_Fk_min 定义见 R4P_DIAGNOSTIC_BUNDLE.md，非本脚本产物）

用法：python p1/source/information_audit/r4p_diagnostics.py [--stage 1|2|3|all]
"""
import argparse
import csv
import gc
import os
import sys
import time

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "p1", "source", "information_audit"))
sys.path.insert(0, os.path.join(_REPO, "p1", "source", "physics"))

from gauge_fisher_v2 import (load_scene, scene_arrays, fisher_blocks, schur_full,  # noqa
                             pinv_psd, gauge_unit, DEFAULT_CUTOFF)
from sh import sh_basis_npy  # noqa

DATA_ROOT = os.path.join(_REPO, "p1", "calibration_set", "data_sun_confirmatory")
IA = os.path.join(_REPO, "p1", "information_audit")
OUT = os.path.join(IA, "diagnostics")
SCORES_CSV = os.path.join(IA, "r4p_confirmatory_scores.csv")
TRIALS_CSV = os.path.join(IA, "r4p_confirmatory_trials.csv")

# 与 r4p_confirmatory_gate 冻结值一致（诊断必须复现同一 F_eff）
NS = [3, 5, 8]
SUBSETS_PER_N = 30
SUBSET_SEED = 20260902
PIXEL_CAP = 1000
CUTOFF = 1e-8

QUANTILES = [0, 0.1, 0.5, 1, 2, 5, 10, 25, 50, 75, 90, 95, 99, 100]


def scene_dirs():
    return sorted([os.path.join(DATA_ROOT, d) for d in os.listdir(DATA_ROOT)
                   if os.path.isdir(os.path.join(DATA_ROOT, d))
                   and os.path.isfile(os.path.join(DATA_ROOT, d, "sh_coeffs_irradiance.npy"))
                   and not d.startswith("_")])


def replay_subsets():
    """严格复现 scores/solve 阶段的 (scene, N, subset, pixel_seed) 序列。"""
    rng = np.random.default_rng(SUBSET_SEED)
    plan = []
    for sd in scene_dirs():
        sc = load_scene(sd)
        for N in NS:
            for _ in range(SUBSETS_PER_N):
                sub = sorted(rng.choice(sc["K"], N, replace=False).tolist())
                px_seed = int(rng.integers(1 << 31))
                plan.append((sd, sc["name"], N, sub, px_seed))
    return plan


# ======================================================================
# 交付物 1 · 每 trial 完整 eigen-spectrum 摘要
# ======================================================================
def stage_eigenspectrum():
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "r4p_trial_eigenspectrum.csv")
    plan = replay_subsets()
    done = set()
    if os.path.exists(path):
        for r in csv.DictReader(open(path, encoding="utf-8")):
            if r.get("N", "").lstrip("-").isdigit():
                done.add((r["scene"], int(r["N"]), r["subset"]))
    f = open(path, "a", newline="", encoding="utf-8")
    wr = None
    t0 = time.time()
    cache = {}
    for i, (sd, name, N, sub, px_seed) in enumerate(plan):
        key = (name, N, ",".join(map(str, sub)))
        if key in done:
            continue
        if name not in cache:
            cache.clear()
            gc.collect()
            cache[name] = load_scene(sd)
        sc = cache[name]
        a, Y, C = scene_arrays(sc, sub, pixel_cap=PIXEL_CAP, seed=px_seed)
        for attempt in range(6):
            try:
                bl = fisher_blocks(a, Y, C)
                F = schur_full(bl, CUTOFF)
                w = np.linalg.eigvalsh(F)            # 升序，含负的 fp 噪声
                break
            except MemoryError:
                gc.collect(); time.sleep(15 * (attempt + 1))
        else:
            raise MemoryError(f"{key} OOM persist")

        tr = float(w.sum())
        wn = w / tr if tr > 0 else w * 0.0           # 归一化谱（无量纲）
        pos = wn[wn > 0]
        # 逐光 F_k 的秩与谱（rank_Fk_min 的来源，逐光展开）
        ranks, kmax, kmin_pos = [], [], []
        for k in range(bl["N"]):
            wk = np.linalg.eigvalsh(bl["Fk"][k])
            lm = float(max(wk.max(), 0.0))
            ranks.append(int((wk > CUTOFF * lm).sum()) if lm > 0 else 0)
            kmax.append(lm)
            wkp = wk[wk > CUTOFF * max(lm, 1e-300)]
            kmin_pos.append(float(wkp.min()) if wkp.size else 0.0)

        rec = dict(scene=name, N=N, subset=",".join(map(str, sub)),
                   P=bl["P"], pixel_seed=px_seed, cutoff=CUTOFF, trace=tr)
        # 归一化全谱分位
        for q in QUANTILES:
            rec[f"eig_norm_q{q}"] = float(np.percentile(wn, q))
        # 尾部计数（退化诊断核心）
        for thr in [0.0, 1e-12, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]:
            rec[f"n_above_{thr:.0e}"] = int((wn > thr).sum())
        rec["n_negative"] = int((w < 0).sum())
        rec["eig_raw_min"] = float(w.min())
        rec["eig_raw_max"] = float(w.max())
        rec["neg_mass_over_trace"] = float(-w[w < 0].sum() / tr) if tr > 0 and (w < 0).any() else 0.0
        # bulk 统计量（候选替代 primary）
        rec["logdet_pos_mean_log"] = float(np.log(pos).mean()) if pos.size else float("-inf")
        rec["eig_norm_geomean_pos"] = float(np.exp(np.log(pos).mean())) if pos.size else 0.0
        rec["eff_rank_entropy"] = float(np.exp(-(pos * np.log(pos)).sum())) if pos.size else 0.0
        rec["participation_ratio"] = float((wn.sum() ** 2) / (wn ** 2).sum()) if (wn ** 2).sum() > 0 else 0.0
        rec["cond_p1_p99"] = float(np.percentile(wn, 99) / max(np.percentile(wn, 1), 1e-300))
        # gauge 与逐光诊断
        rec["gauge_residual"] = float(np.linalg.norm(F @ gauge_unit(bl["a"]))
                                      / max(np.linalg.norm(F, 2), 1e-300))
        rec["rank_Fk_min"] = int(min(ranks))
        rec["rank_Fk_max"] = int(max(ranks))
        rec["rank_Fk_mean"] = float(np.mean(ranks))
        rec["Fk_eigmax_min"] = float(min(kmax))
        rec["Fk_minpos_min"] = float(min(kmin_pos))
        rec["active_frac_min"] = float(bl["diag"]["active_frac"].min())
        rec["active_frac_mean"] = float(bl["diag"]["active_frac"].mean())
        rec["boundary_frac_max"] = float(bl["diag"]["boundary_frac"].max())
        rec["F_ss_diag_min"] = float(bl["F_ss_diag"].min())
        rec["F_ss_diag_median"] = float(np.median(bl["F_ss_diag"]))
        rec["n_dead_pixels"] = int((bl["F_ss_diag"] <= 1e-12 * bl["F_ss_diag"].max()).sum())

        if wr is None:
            wr = csv.DictWriter(f, fieldnames=list(rec))
            if not done:
                wr.writeheader()
        wr.writerow(rec)
        del F, bl, a, Y, C
        if (i + 1) % 60 == 0:
            f.flush(); gc.collect()
            print(f"  [eig] {i+1}/{len(plan)} ({time.time()-t0:.0f}s)", flush=True)
    f.close()
    print(f"[eig] -> {path}")


# ======================================================================
# 交付物 2 · 未筛选 raw trial ⊕ scores join
# ======================================================================
def stage_raw_join():
    os.makedirs(OUT, exist_ok=True)
    sc = {}
    for r in csv.DictReader(open(SCORES_CSV, encoding="utf-8")):
        sc[(r["scene"], int(r["N"]), r["subset"])] = r
    trials = [r for r in csv.DictReader(open(TRIALS_CSV, encoding="utf-8"))
              if r.get("N", "").lstrip("-").isdigit()]
    rows = []
    for t in trials:
        key = (t["scene"], int(t["N"]), t["subset"])
        s = sc.get(key, {})
        rec = dict(scene=t["scene"], N=int(t["N"]), subset=t["subset"])
        # solver 侧 raw（未做任何筛选）
        for k in ["final_loss", "grad_norm", "tail_range", "restart", "iters",
                  "si_mae_A", "ho_psnr"]:
            rec["solver_" + k] = t.get(k, "")
        rec["solver_success_asrecorded"] = t.get("success", "")
        rec["solver_converged_asrecorded"] = t.get("converged", "")
        # scores 侧全字段
        for k, v in s.items():
            if k in ("scene", "N", "subset"):
                continue
            rec["score_" + k] = v
        rec["scores_matched"] = int(bool(s))
        rows.append(rec)
    path = os.path.join(OUT, "r4p_raw_trials_joined.csv")
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader(); w.writerows(rows)
    print(f"[raw] -> {path} rows={len(rows)} (未做任何 success/收敛筛选)")
    n_match = sum(r["scores_matched"] for r in rows)
    print(f"  scores 匹配: {n_match}/{len(rows)}")
    return rows


# ======================================================================
# 交付物 3 · 每 scene 的 normal / SH Gram spectrum
# ======================================================================
def stage_scene_gram():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for sd in scene_dirs():
        sc = load_scene(sd)
        mask = sc["mask"]
        n = sc["n_mesh"].transpose(1, 2, 0)[mask]              # [P_full,3]
        a_full = sc["albedo"][mask].astype(np.float64)
        Yf = sh_basis_npy(n)                                    # [P_full,9]
        Pf = len(n)
        rec = dict(scene=os.path.basename(sd), P_full=Pf, K=sc["K"])

        # --- normal 协方差谱（3×3，几何法线覆盖）---
        nc = n - n.mean(axis=0, keepdims=True)
        Cn = (nc.T @ nc) / Pf
        wn = np.linalg.eigvalsh(Cn)[::-1]
        for i, v in enumerate(wn):
            rec[f"normal_cov_eig{i+1}"] = float(v)
        rec["normal_cov_trace"] = float(wn.sum())
        rec["normal_cov_anisotropy"] = float(wn[0] / max(wn[-1], 1e-300))
        rec["normal_cov_eff_rank"] = float(wn.sum() ** 2 / (wn ** 2).sum())
        # 法线方向覆盖：单位球面上的二阶矩（未中心化）
        M2 = (n.T @ n) / Pf
        w2 = np.linalg.eigvalsh(M2)[::-1]
        for i, v in enumerate(w2):
            rec[f"normal_2ndmoment_eig{i+1}"] = float(v)

        # --- SH Gram 谱（9×9，两种归一化）---
        G_raw = (Yf.T @ Yf) / Pf                                # 未加权
        wg = np.linalg.eigvalsh(G_raw)[::-1]
        for i, v in enumerate(wg):
            rec[f"sh_gram_eig{i+1}"] = float(v)
        rec["sh_gram_trace"] = float(wg.sum())
        rec["sh_gram_min_over_max"] = float(wg[-1] / max(wg[0], 1e-300))
        rec["sh_gram_logdet"] = float(np.log(np.clip(wg, 1e-300, None)).sum())
        rec["sh_gram_rank_1em8"] = int((wg > 1e-8 * wg[0]).sum())
        rec["sh_gram_eff_rank"] = float(wg.sum() ** 2 / (wg ** 2).sum())

        # albedo² 加权（= F_ll,k 在 h≡1 时的形态，直接对应 rank_Fk 的上界）
        Gw = (Yf * (a_full ** 2)[:, None]).T @ Yf
        wgw = np.linalg.eigvalsh(Gw)[::-1]
        for i, v in enumerate(wgw):
            rec[f"sh_gram_a2_eig{i+1}"] = float(v)
        rec["sh_gram_a2_min_over_max"] = float(wgw[-1] / max(wgw[0], 1e-300))
        rec["sh_gram_a2_rank_1em8"] = int((wgw > 1e-8 * wgw[0]).sum())

        # --- 实际每盏光的 F_ll,k 秩（h 由 ReLU 决定，全 32 灯）---
        Ca = sc["sh_irr"]                                       # [K,9]
        Z = (Yf @ Ca.T).T                                       # [K,P_full]
        H = (Z > 0).astype(np.float64)
        ranks, afr = [], []
        for k in range(sc["K"]):
            Fk = (Yf * (a_full ** 2 * H[k])[:, None]).T @ Yf
            wk = np.linalg.eigvalsh(Fk)
            lm = float(max(wk.max(), 0.0))
            ranks.append(int((wk > 1e-8 * lm).sum()) if lm > 0 else 0)
            afr.append(float(H[k].mean()))
        ranks = np.array(ranks)
        rec["Fk_rank_min_allK"] = int(ranks.min())
        rec["Fk_rank_median_allK"] = float(np.median(ranks))
        rec["Fk_rank_max_allK"] = int(ranks.max())
        rec["Fk_rank_lt9_frac"] = float((ranks < 9).mean())
        rec["active_frac_min_allK"] = float(min(afr))
        rec["active_frac_median_allK"] = float(np.median(afr))
        rec["albedo_min"] = float(a_full.min())
        rec["albedo_median"] = float(np.median(a_full))
        rows.append(rec)
        print(f"  [gram] {rec['scene']:28s} normal_eff_rank={rec['normal_cov_eff_rank']:.2f} "
              f"sh_gram_rank={rec['sh_gram_rank_1em8']} Fk_rank_min={rec['Fk_rank_min_allK']}",
              flush=True)
    path = os.path.join(OUT, "r4p_scene_gram_spectrum.csv")
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader(); w.writerows(rows)
    print(f"[gram] -> {path} rows={len(rows)}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["1", "2", "3", "all"])
    args = ap.parse_args()
    if args.stage in ("3", "all"):
        stage_scene_gram()
    if args.stage in ("2", "all"):
        stage_raw_join()
    if args.stage in ("1", "all"):
        stage_eigenspectrum()


if __name__ == "__main__":
    main()
