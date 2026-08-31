"""R4″ H1.7 · 重建 trial-level master table（任务书 §5 Task A）。

产出 r4pp/01_master_trial_table.parquet：每行 = 一个 solver trial。

关键设计（对任务书 §5 的事实修正，见 R4PP_EXECUTION_MANUAL §1.1）：
  旧 convergence filter 是**分析时**（r4p_confirmatory_gate._load_trials 的组内
  P75）施加的，**不是采集时**。因此 trials.csv 本身即全量未筛选记录，
  Task A 不需要"恢复被排除的 trials"，只需格式化 + 补字段 + 并列两套 flag。

  `objective_rel_change` 对旧 trial 恒为 NaN：旧 joint_solve 未落盘 loss trace
  （C-1 改动后的新 trial 才有）。这是刻意保留的缺口，不得用近似值填充。

用法：python p1/source/information_audit/r4pp_master_table.py
"""
import argparse
import csv
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
FROZEN = os.path.join(_REPO, "archive", "R4prime_frozen", "data", "p1")
OUT_DIR = os.path.join(_REPO, "r4pp")

TRIALS = os.path.join(FROZEN, "information_audit", "r4p_confirmatory_trials.csv")
SCORES = os.path.join(FROZEN, "information_audit", "r4p_confirmatory_scores.csv")
EIGSPEC = os.path.join(FROZEN, "information_audit", "diagnostics",
                       "r4p_trial_eigenspectrum.csv")
GRAM = os.path.join(FROZEN, "information_audit", "diagnostics",
                    "r4p_scene_gram_spectrum.csv")

# 旧 solver 的种子基（joint_solve 写死 20260830+rs，restarts=3）
OLD_SEED_BASE = 20260830
OLD_RESTARTS = 3
DATASET_TAG = "R4prime_exploratory"

# geometry_family 分类（依 make_confirmatory_meshes.py 的三族设计）
FAMILY = {
    # 平滑法线族
    "conf_sphere_r05": "smooth", "conf_icosphere_sub3": "smooth",
    "conf_ellipsoid_z06": "smooth", "conf_ellipsoid_x13z07": "smooth",
    "conf_torus_R05_r02": "smooth", "conf_torus_R06_r035": "smooth",
    "conf_hemisphere_sq": "smooth", "conf_egg": "smooth",
    # 簇状法线族（sparse-normal）
    "conf_cube_axis": "cluster", "conf_cube_rot30z": "cluster",
    "conf_cube_rot45z30x": "cluster", "conf_pyramid4": "cluster",
    "conf_pyramid4_rot30z": "cluster", "conf_pyramid6": "cluster",
    "conf_prism8": "cluster", "conf_cylinder_r03_d12": "cluster",
    "conf_cylinder_r06_d06": "cluster", "conf_cone_r08_d06": "cluster",
    "conf_cone_r04_d12": "cluster",
    # 复合族（遮挡 + 混合法线）
    "conf_snowman": "composite", "conf_sphere_on_cube": "composite",
    "conf_two_spheres_row": "composite", "conf_cyl_plus_sphere": "composite",
    "conf_cube_plus_cone": "composite", "conf_three_cubes": "composite",
}


def read_csv_num(path, key_cols=("scene", "N", "subset")):
    """读 CSV → dict[(scene,N,subset)] = row(dict)，跳过 header 污染行。"""
    out = {}
    n_skipped = 0
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if not r.get("N", "").lstrip("-").isdigit():
            n_skipped += 1
            continue
        out[(r["scene"], int(r["N"]), r["subset"])] = r
    return out, n_skipped


def f(v, default=np.nan):
    try:
        x = float(v)
        return x
    except (TypeError, ValueError):
        return default


def i(v, default=-1):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "01_master_trial_table.parquet"))
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---------- 读入四张源表 ----------
    trials_rows = [r for r in csv.DictReader(open(TRIALS, encoding="utf-8"))
                   if r.get("N", "").lstrip("-").isdigit()]
    n_trials_raw = len(trials_rows)
    scores, sk_s = read_csv_num(SCORES)
    eig, sk_e = read_csv_num(EIGSPEC)
    gram = {r["scene"]: r for r in csv.DictReader(open(GRAM, encoding="utf-8"))}
    print(f"[in] trials  : {n_trials_raw} 数据行")
    print(f"[in] scores  : {len(scores)}（跳过污染行 {sk_s}）")
    print(f"[in] eigspec : {len(eig)}（跳过污染行 {sk_e}）")
    print(f"[in] gram    : {len(gram)} scene")

    # ---------- 旧 P75 双筛（复现 _load_trials 的分析时筛选）----------
    by_cell = {}
    for r in trials_rows:
        by_cell.setdefault((r["scene"], int(r["N"])), []).append(r)
    p75 = {}
    for cell, rs in by_cell.items():
        losses = np.array([f(x["final_loss"]) for x in rs])
        grads = np.array([f(x["grad_norm"]) for x in rs])
        p75[cell] = (float(np.percentile(losses, 75)), float(np.percentile(grads, 75)))

    # ---------- 逐 trial 组装 ----------
    recs = []
    n_miss_score, n_miss_eig = 0, 0
    for r in trials_rows:
        scn, N, sub = r["scene"], int(r["N"]), r["subset"]
        key = (scn, N, sub)
        s = scores.get(key)
        e = eig.get(key)
        g = gram.get(scn, {})
        if s is None:
            n_miss_score += 1
        if e is None:
            n_miss_eig += 1
        lt, gt = p75[(scn, N)]
        fl, gn = f(r["final_loss"]), f(r["grad_norm"])

        rec = {
            # ---- identity ----
            "scene_id": scn,
            "geometry_family": FAMILY.get(scn, "unknown"),
            "dataset_tag": DATASET_TAG,
            # ---- budget / subset ----
            "N": N,
            "illumination_ids": sub,
            "n_lights": len(sub.split(",")),
            # ---- randomness ----
            # 旧 trial 取 restarts=3 的 best，未记录 winning restart ⇒ 具体 seed 未知
            "solver_seed_base": OLD_SEED_BASE,
            "solver_restarts": OLD_RESTARTS,
            "solver_seed": np.nan,
            "pixel_seed": i(e["pixel_seed"]) if e else -1,
            # ---- optimization ----
            "solver_status": "completed",          # 旧管线无异常退出记录
            "iteration_count": i(r.get("iters")),
            "final_objective": fl,
            "grad_norm": gn,
            "proj_grad_norm": np.nan,              # C-1 后才有
            "objective_rel_change": np.nan,        # 旧 trial 无 loss trace（见 docstring）
            "tail_range_abs": f(r.get("tail_range")),
            # ---- error ----
            "reconstruction_error": f(r["si_mae_A"]),
            "ho_psnr": f(r["ho_psnr"]),
            # ---- 两套 old filtering flag ----
            "old_converged_flag": i(r.get("converged"), -1),
            "old_success_asrecorded": i(r.get("success"), -1),
            "old_p75_success_flag": int((fl < lt) and (gn < gt)),
            "old_p75_loss_thr": lt,
            "old_p75_grad_thr": gt,
        }
        # ---- Fisher：scores 侧 ----
        if s:
            rec.update({
                "score_lam_min_pos_norm": f(s["full_lam_min_pos_norm"]),
                "score_lam_max_norm": f(s["full_lam_max_norm"]),
                "score_logdet_pos_norm": f(s["full_logdet_pos_norm"]),
                "score_a_opt_pos_norm": f(s["full_a_opt_pos_norm"]),
                "score_d_pos": f(s["full_d_pos"]),
                "score_trace": f(s["full_trace"]),
                "score_min_eig": f(s["full_min_eig"]),
                "score_gauge_residual": f(s["full_gauge_residual"]),
                "score_offdiag_max": f(s["full_offdiag_max"]),
                "P": f(s["P"]),
                "cutoff": f(s["cutoff"]),
            })
        # ---- Fisher：完整谱（eigenspectrum 侧）----
        if e:
            for q in [0, 0.1, 0.5, 1, 2, 5, 10, 25, 50, 75, 90, 95, 99, 100]:
                rec[f"eig_norm_q{q}"] = f(e[f"eig_norm_q{q}"])
            for thr in ["0e+00", "1e-12", "1e-10", "1e-09", "1e-08",
                        "1e-07", "1e-06", "1e-05", "1e-04"]:
                rec[f"n_above_{thr}"] = f(e[f"n_above_{thr}"])
            for k in ["n_negative", "eig_raw_min", "eig_raw_max",
                      "neg_mass_over_trace", "logdet_pos_mean_log",
                      "eig_norm_geomean_pos", "eff_rank_entropy",
                      "participation_ratio", "cond_p1_p99",
                      "rank_Fk_min", "rank_Fk_max", "rank_Fk_mean",
                      "Fk_eigmax_min", "Fk_minpos_min",
                      "active_frac_min", "active_frac_mean", "boundary_frac_max",
                      "F_ss_diag_min", "F_ss_diag_median", "n_dead_pixels"]:
                rec[k] = f(e[k])
        # ---- geometry（scene 级）----
        if g:
            rec.update({
                "sh_gram_rank": f(g["sh_gram_rank_1em8"]),
                "sh_gram_logdet": f(g["sh_gram_logdet"]),
                "sh_gram_eff_rank": f(g["sh_gram_eff_rank"]),
                "sh_gram_min_over_max": f(g["sh_gram_min_over_max"]),
                "sh_gram_a2_rank": f(g["sh_gram_a2_rank_1em8"]),
                "sh_gram_a2_min_over_max": f(g["sh_gram_a2_min_over_max"]),
                "normal_cov_eff_rank": f(g["normal_cov_eff_rank"]),
                "normal_cov_anisotropy": f(g["normal_cov_anisotropy"]),
                "Fk_rank_min_allK": f(g["Fk_rank_min_allK"]),
                "Fk_rank_median_allK": f(g["Fk_rank_median_allK"]),
                "Fk_rank_lt9_frac": f(g["Fk_rank_lt9_frac"]),
                "scene_P_full": f(g["P_full"]),
                "scene_K": f(g["K"]),
            })
        recs.append(rec)

    df = pd.DataFrame(recs)

    # ---------- 断言（执行手册 §6.2）----------
    print("\n[assert] 契约检查")
    ok = True
    a1 = len(df) == n_trials_raw
    print(f"  {'PASS' if a1 else 'FAIL'} 行数守恒: {len(df)} == {n_trials_raw}（join 未丢行）")
    ok &= a1
    mean_p75 = float(df["old_p75_success_flag"].mean())
    a2 = abs(mean_p75 - 0.5625) < 0.05
    print(f"  {'PASS' if a2 else 'FAIL'} old_p75_success_flag 均值 = {mean_p75:.4f} "
          f"≈ 0.75×0.75 = 0.5625（endogenous filtering 证据）")
    ok &= a2
    a3 = (n_miss_score == 0)
    print(f"  {'PASS' if a3 else 'WARN'} scores 全部匹配（缺失 {n_miss_score}）")
    a4 = (n_miss_eig == 0)
    print(f"  {'PASS' if a4 else 'WARN'} eigspectrum 全部匹配（缺失 {n_miss_eig}）")
    a5 = bool(df["objective_rel_change"].isna().all())
    print(f"  {'PASS' if a5 else 'FAIL'} objective_rel_change 全 NaN（旧 trial 无 trace，"
          f"刻意留缺口）")
    ok &= a5
    a6 = df["geometry_family"].ne("unknown").all()
    print(f"  {'PASS' if a6 else 'FAIL'} geometry_family 全部已分类"
          f"（unknown: {int(df['geometry_family'].eq('unknown').sum())}）")
    ok &= a6

    # ---------- 落盘 ----------
    df.to_parquet(args.out, index=False, compression="snappy")
    csv_out = args.out.replace(".parquet", ".csv")
    df.to_csv(csv_out, index=False)
    print(f"\n[out] {args.out}  ({os.path.getsize(args.out):,} B)")
    print(f"[out] {csv_out}  ({os.path.getsize(csv_out):,} B)")
    print(f"[out] shape = {df.shape[0]} 行 × {df.shape[1]} 列")

    # ---------- 摘要（供 05 报告引用）----------
    summary = {
        "generated_at": datetime.now().isoformat(),
        "dataset_tag": DATASET_TAG,
        "n_trials": int(len(df)),
        "n_scenes": int(df["scene_id"].nunique()),
        "n_cells": int(df.groupby(["scene_id", "N"]).ngroups),
        "N_values": sorted(df["N"].unique().tolist()),
        "family_counts": df["geometry_family"].value_counts().to_dict(),
        "old_p75_success_rate": mean_p75,
        "old_success_asrecorded_rate": float(df["old_success_asrecorded"].mean()),
        "old_converged_flag_rate": float(df["old_converged_flag"].mean()),
        "objective_rel_change_available": False,
        "assertions_all_pass": bool(ok),
        "note": ("旧 filter 为分析时施加，采集未丢数据；objective_rel_change 需 C-1 "
                 "之后的新 trial 才可计算。"),
    }
    sp = os.path.join(OUT_DIR, "01_master_trial_table_summary.json")
    json.dump(summary, open(sp, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"[out] {sp}")
    print("\n" + json.dumps(summary, indent=2, ensure_ascii=False))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
