"""MF-0.3 · A/B/C/D factorial correctness rerun（数学冻结 v1.0 §57）。

两个 correctness gate 修正后的四臂复算（同一冻结协议：相同对象、pixel subset、
level、seed、统计规则；禁止按结果好看与否挑选——最终论文口径固定为 D）：

    | run | noise fit (M0-1) | mode coordinate (M0-2) |
    |-----|------------------|------------------------|
    | A   | legacy           | legacy                 |  == 冻结基准（机制校验锚点）
    | B   | corrected        | legacy                 |
    | C   | legacy           | dual                   |
    | D   | corrected        | dual                   |  ← 论文最终口径

- noise fit / mode coordinate 的语义见 experiments/openillumination_validation.py
  （noise_fit(convention=...) 与 predicted_degradation(mode_coordinate=...)）。
- 机制校验：A 臂逐 cell 对照冻结 results/openillumination/ci04_formal_summary.json
  （pred/emp 逐元素相对差 + 顶层 Spearman），超容差即报错（重算链失效信号）。
- 统计（v1.0 §38-40 口径）：R_A（object-cluster bootstrap，B=10000、seed 20260908）、
  positive cells/objects、stratified median（P_mode 与 best scalar）、severity 比较
  （paired Δ = mode − best scalar，replicate 内重选 best）。

输出（新目录，不触碰任何冻结产物）：
  results/openillumination/correctness/mf0_factorial_summary.json
  results/openillumination/correctness/mf0_factorial_rows_<variant>.json

用法：
  python experiments/openillumination_factorial.py \
      [--config configs/openillumination.yaml] [--out results/openillumination/correctness]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import spearmanr

import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from calibinfo.datasets.openillumination import load_object             # noqa: E402
from calibinfo.metrics.cluster_bootstrap import cluster_bootstrap       # noqa: E402
from calibinfo.metrics.spectral_criteria import (                        # noqa: E402
    predictors_from_spectrum, retention_spectrum_full)
from calibinfo.models.corruption import CorruptionGenerator              # noqa: E402
from experiments.openillumination_validation import (                    # noqa: E402
    N_MODES, NominalScene)

VARIANTS = ("A", "B", "C", "D")
NOISE_CONV = {"A": "legacy", "B": "corrected", "C": "legacy", "D": "corrected"}
MODE_COORD = {"A": "legacy", "B": "legacy", "C": "dual", "D": "dual"}
SCALARS = ("P_trace", "P_logdet", "P_emin")
PAPER_VARIANT = "D"
MACHINERY_TOL = 1e-7          # A 臂 vs 冻结产物的跨机浮点预算（预期 ~1e-12）


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def spearman_safe(x, y):
    r = spearmanr(x, y).statistic
    return float(r) if np.isfinite(r) else float("nan")


def median_valid(vals, min_valid=4):
    """median 忽略 nan；有效 < min_valid → (nan, 计数)。

    返回 (median, n_finite)：第二返回值是**有限条目计数**（v1.1 修正——
    v1 把布尔值误存进 stratified_valid_levels 字段，导致审计误读为
    "6 层只有 1 层有效"；实际六层全部有限，0.536 是六层中位数）。
    """
    v = [x for x in vals if np.isfinite(x)]
    return (float(np.median(v)) if len(v) >= min_valid else float("nan")), len(v)


# --------------------------------------------------------------- 单对象资产
def _scene_pair(obj, rng):
    """同一 rng 流构造 legacy/corrected 双场景（修正场景复用同一冻结子集）。

    legacy 场景按原 run 消耗 rng（choice(142,48) + choice(P,1200)），
    corrected 场景复用其 (sel, pidx)——四臂共享对象/像素/灯子集与 seed 流。
    """
    scen_leg = NominalScene(obj, rng, noise_fit_convention="legacy")
    scen_cor = NominalScene(obj, rng, noise_fit_convention="corrected",
                            selections=(scen_leg.sel, scen_leg.pidx))
    return scen_leg, scen_cor


def _prediction_side(scen, sig3, cfg):
    """单 convention 的预测侧：pred_deg（与原 run 逐位同序）、弱模式投影算子
    （legacy 与 dual 两种）、全谱 scalar predictors。"""
    Finf_diag = scen.Finf_diag
    DF = scen.delta_f_injected(sig3)
    # 与 predicted_degradation 逐位相同的运算序列（A 臂锚点依赖）
    Finv_h = np.diag(1.0 / np.sqrt(Finf_diag))
    R = Finv_h @ DF @ Finv_h
    rho_R, V = np.linalg.eigh(R)                    # 升序
    weak = list(range(min(N_MODES, len(rho_R))))
    pred_deg = 1.0 / rho_R[weak]
    W = {"legacy": V[:, weak],
         "dual": np.diag(np.sqrt(Finf_diag)) @ V[:, weak]}
    spec = retention_spectrum_full(
        DF, np.diag(Finf_diag),
        rank_relative_tol=cfg["rank_relative_tol"],
        clip_tol=cfg["retention_clip_tol"], log_eps=cfg["log_eps"])
    scalars = predictors_from_spectrum(spec["r"], spec["q"], log_eps=cfg["log_eps"])
    return dict(pred_deg=pred_deg, W=W, scalars=scalars)


# --------------------------------------------------------------- 主流程
def run(config_path=REPO / "configs/openillumination.yaml",
        out_dir=REPO / "results/openillumination/correctness"):
    cfg_path = Path(config_path)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    rng = np.random.default_rng(cfg["seed"])
    objects = cfg["objects"][:cfg.get("max_objects", 11)]
    levels = [float(lv) for lv in cfg["levels"]]
    seeds_per_level = cfg["seeds_per_level"]
    ctype = cfg.get("corruption_type", "joint")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # cells[obj][level] = per-noise-convention 预测侧 + per-variant 经验侧
    cells = {}
    for obj_name in objects:
        obj = load_object(cfg["data_root"], obj_name,
                          data_meta=cfg.get("data_meta"))
        scen_leg, scen_cor = _scene_pair(obj, rng)
        scenes = {"legacy": scen_leg, "corrected": scen_cor}
        cell = {}
        for level in levels:
            gen = CorruptionGenerator(ctype, level)
            sig3 = gen.sigma_phi_diag()              # (3,3) 物理单位
            pred = {conv: _prediction_side(scenes[conv], sig3, cfg)
                    for conv in ("legacy", "corrected")}
            E = {conv: np.empty((seeds_per_level, scenes[conv].I.shape[1]))
                 for conv in ("legacy", "corrected")}
            E0 = {conv: np.empty_like(E[conv]) for conv in E}
            resid_pool = (scen_leg.I - scen_leg.s_hat * scen_leg.rho[None, :]).ravel()
            for s_i in range(seeds_per_level):
                # 与原 run 相同的共享 seed 流（每 level×seed 一次 integers 抽取）
                s_rng = np.random.default_rng(
                    rng.integers(0, 2 ** 63 - 1) + s_i * 7)
                d2, gains = gen.apply(s_rng, scen_leg.dirs)
                for conv in ("legacy", "corrected"):
                    sc = scenes[conv]
                    rho_t = sc.estimate_albedo(d2, gains)
                    E[conv][s_i] = rho_t - sc.rho
                    # 对照臂（与原 run 相同的残差 bootstrap；对两 convention 同分布）
                    I_star = (sc.s_hat * sc.rho[None, :]
                              + s_rng.choice(resid_pool, size=sc.I.shape,
                                             replace=True))
                    sc_b = NominalScene._from_arrays(sc, I_star)
                    rho_c = sc_b.estimate_albedo(sc.dirs, np.ones(len(sc.dirs)))
                    E0[conv][s_i] = rho_c - sc.rho
            # 尺度 gauge 对齐（预注册；ρ 跨 convention 相同 → 对齐系数一致）
            for conv in ("legacy", "corrected"):
                sc = scenes[conv]
                for _E in (E[conv], E0[conv]):
                    s_scale = (_E * sc.rho).sum() / (sc.rho ** 2).sum()
                    _E -= s_scale * sc.rho
            emp = {}
            for variant in VARIANTS:
                conv_v, coord_v = NOISE_CONV[variant], MODE_COORD[variant]
                proj = E[conv_v] @ pred[conv_v]["W"][coord_v]
                proj0 = E0[conv_v] @ pred[conv_v]["W"][coord_v]
                emp[variant] = (proj.var(0, ddof=1)
                                / np.maximum(proj0.var(0, ddof=1), 1e-30))
            cell[float(level)] = dict(pred=pred, emp=emp)
        cells[obj_name] = cell
        print(f"[factorial] scene done: {obj_name}", flush=True)

    # ---- 逐 variant 行 + 统计 ----
    variants_out = {}
    rows_by_variant = {}
    for variant in VARIANTS:
        conv, coord = NOISE_CONV[variant], MODE_COORD[variant]
        rows = []
        for obj_name in objects:
            for level in levels:
                c = cells[obj_name][level]
                rows.append(dict(
                    object=obj_name, level=level, ctype=ctype,
                    pred_deg=[float(x) for x in c["pred"][conv]["pred_deg"]],
                    emp_deg=[float(x) for x in c["emp"][variant]],
                    scalars={k: float(c["pred"][conv]["scalars"][k])
                             for k in SCALARS},
                    n_seeds=seeds_per_level))
        rows_by_variant[variant] = rows
        variants_out[variant] = dict(
            noise_fit_convention=conv, mode_coordinate=coord,
            **_analyze(rows, cfg))
        print(f"[factorial] variant {variant}: "
              f"R_A={variants_out[variant]['RA']:.4f} "
              f"CI95={variants_out[variant]['RA_ci95']}", flush=True)

    # ---- 机制校验：A 臂 vs 冻结产物 ----
    frozen = json.loads((REPO / "results/openillumination/ci04_formal_summary.json")
                        .read_text(encoding="utf-8"))
    frz = {(r["object"], float(r["level"])): r for r in frozen["rows"]}
    pred_rel = emp_rel = 0.0
    for r in rows_by_variant["A"]:
        f = frz[(r["object"], r["level"])]
        pred_rel = max(pred_rel, float(np.max(np.abs(
            np.asarray(r["pred_deg"]) - np.asarray(f["pred_deg"]))
            / np.abs(np.asarray(f["pred_deg"])))))
        emp_rel = max(emp_rel, float(np.max(np.abs(
            np.asarray(r["emp_deg"]) - np.asarray(f["emp_deg"]))
            / np.abs(np.asarray(f["emp_deg"])))))
    # 顶层 Spearman 用与原 run 相同的验证式口径（330 点 pred-vs-emp 配对拼接），
    # 而非 severity 口径的 P_mode-vs-T
    preds_flat = np.concatenate([np.asarray(r["pred_deg"], float)
                                 for r in rows_by_variant["A"]])
    emps_flat = np.concatenate([np.asarray(r["emp_deg"], float)
                                for r in rows_by_variant["A"]])
    sp_abs = abs(spearman_safe(preds_flat, emps_flat) - frozen["spearman"])
    machinery = dict(pred_deg_max_rel=pred_rel, emp_deg_max_rel=emp_rel,
                     pooled_spearman_abs_diff=sp_abs,
                     tolerance=MACHINERY_TOL, pass_=bool(
                         pred_rel < MACHINERY_TOL and emp_rel < MACHINERY_TOL
                         and sp_abs < MACHINERY_TOL))
    if not machinery["pass_"]:
        raise RuntimeError(f"MF-0.3 机制校验失败（A 臂未复现冻结基准）: {machinery}")

    # ---- 落盘 ----
    display = {}
    for variant in VARIANTS:
        v = variants_out[variant]
        (out / f"mf0_factorial_rows_{variant}.json").write_text(
            json.dumps(dict(variant=variant,
                            noise_fit_convention=NOISE_CONV[variant],
                            mode_coordinate=MODE_COORD[variant],
                            rows=rows_by_variant[variant]),
                       ensure_ascii=False, indent=1), encoding="utf-8")
        display[variant] = dict(
            RA=round(v["RA"], 2),
            RA_ci95=[round(v["RA_ci95"][0], 2), round(v["RA_ci95"][1], 2)],
            stratified_median_mode=(round(v["stratified_median_mode"], 3)
                                    if np.isfinite(v["stratified_median_mode"])
                                    else None),
            delta_mode_vs_best_scalar=(
                round(v["delta_mode_vs_best_scalar"], 3)
                if np.isfinite(v["delta_mode_vs_best_scalar"]) else None))

    summary = dict(
        gate="MF-0.3 factorial correctness rerun (math freeze v1.0 section 57)",
        paper_variant=PAPER_VARIANT,
        machinery_check=machinery,
        protocol=dict(
            config_sha256=_sha(cfg_path),
            frozen_reference_sha256=_sha(REPO / "results/openillumination/"
                                         "ci04_formal_summary.json"),
            seed=cfg["seed"], levels=levels,
            seeds_per_level=seeds_per_level, ctype=ctype, objects=objects,
            n_cells=len(objects) * len(levels), n_modes=N_MODES,
            bootstrap=dict(reps=cfg["bootstrap_reps"], seed=cfg["bootstrap_seed"],
                           unit="object"),
            variants={v: dict(noise_fit_convention=NOISE_CONV[v],
                              mode_coordinate=MODE_COORD[v]) for v in VARIANTS}),
        variants=variants_out,
        display=display,
        note="A=legacy/legacy (== frozen benchmark, machinery anchor); "
             "B=corrected/legacy; C=legacy/dual; D=corrected/dual (paper-facing "
             "variant, fixed a priori). R_A = median within-cell Spearman of "
             "S=1-1/rho vs empirical degradation; object-cluster bootstrap "
             "B=10000 seed 20260908; stratified median = median over the six "
             "fixed-level Spearman; delta = mode minus best scalar "
             "(within-replicate re-selection).")
    (out / "mf0_factorial_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[factorial] machinery check: pred_rel={pred_rel:.3e} "
          f"emp_rel={emp_rel:.3e} spearman_diff={sp_abs:.3e}")
    for v in VARIANTS:
        d = display[v]
        print(f"[factorial] {v} ({NOISE_CONV[v]}/{MODE_COORD[v]}): "
              f"R_A={d['RA']} CI={d['RA_ci95']} strat={d['stratified_median_mode']} "
              f"delta={d['delta_mode_vs_best_scalar']}")
    return summary


def _analyze(rows, cfg):
    """v1.0 §38-40 统计口径（与 openillumination_severity 的 R2-A/R2-B 一致）。"""
    # --- R2-A: within-cell mode ranking → R_A ---
    per_cell, per_obj = {}, {}
    undefined_cells = 0
    for r in rows:
        key = (r["object"], float(r["level"]))
        S = 1.0 - 1.0 / np.asarray(r["pred_deg"], float)
        D = np.asarray(r["emp_deg"], float)
        r_ol = spearman_safe(S, D)
        if not np.isfinite(r_ol):
            undefined_cells += 1
        per_cell[key] = r_ol
    obj_ids = sorted({k[0] for k in per_cell})
    for o in obj_ids:
        vals = [v for k, v in per_cell.items() if k[0] == o and np.isfinite(v)]
        per_obj[o] = float(np.median(vals)) if vals else float("nan")
    r_o_list = [per_obj[o] for o in obj_ids]
    R_A = float(np.median(r_o_list))
    n_pos_cells = int(sum(1 for v in per_cell.values() if np.isfinite(v) and v > 0))
    n_pos_obj = int(sum(1 for v in r_o_list if np.isfinite(v) and v > 0))

    def stat_ra(sub):
        vals = [per_obj[o] for o, _ in sub if np.isfinite(per_obj[o])]
        return float(np.median(vals))

    _pt, ci95, _boots = cluster_bootstrap(
        [(o, None) for o in obj_ids], stat_ra,
        cfg["bootstrap_reps"], cfg["bootstrap_seed"])

    # --- R2-B: fixed-level severity（P_mode vs scalar predictors）---
    # per-level 点：(object, P_value, T_ol)
    def level_points(pick):
        pts = {lv: [] for lv in sorted({float(r["level"]) for r in rows})}
        for r in rows:
            pd_ = np.asarray(r["pred_deg"], float)
            P_mode = 1.0 - 1.0 / float(np.max(pd_))
            T = float(np.max(np.asarray(r["emp_deg"], float)))
            pts[float(r["level"])].append(
                (r["object"], P_mode if pick is None else r["scalars"][pick], T))
        return pts

    points = {"P_mode": level_points(None)}
    for s in SCALARS:
        points[s] = level_points(s)

    def strat_of(pts):
        rhos = [spearman_safe([x[1] for x in v], [x[2] for x in v])
                for v in pts.values()]
        m, valid = median_valid(rhos)
        return m, valid

    strat = {}
    for name, pts in points.items():
        m, valid = strat_of(pts)
        strat[name] = dict(median=m, valid_levels=int(valid))
    best_name = max(SCALARS, key=lambda s: strat[s]["median"]
                    if np.isfinite(strat[s]["median"]) else -9e9)

    # paired Δ = mode − best scalar（object-level payload，replicate 内重选 best）
    per_obj_lvl = {}
    for r in rows:
        pd_ = np.asarray(r["pred_deg"], float)
        per_obj_lvl.setdefault(r["object"], {})[float(r["level"])] = dict(
            P_mode=1.0 - 1.0 / float(np.max(pd_)),
            T=float(np.max(np.asarray(r["emp_deg"], float))),
            **{s: r["scalars"][s] for s in SCALARS})
    payload = [(o, per_obj_lvl[o]) for o in sorted(per_obj_lvl)]

    def _strat_of_sub(sub, which):
        by_lv = {}
        for _o, d in sub:
            for lv, x in d.items():
                by_lv.setdefault(lv, ([], []))
                by_lv[lv][0].append(x["P_mode"] if which == "P_mode"
                                    else x[which])
                by_lv[lv][1].append(x["T"])
        rhos = [spearman_safe(ps, ts) for _lv, (ps, ts) in sorted(by_lv.items())]
        m, _valid = median_valid(rhos)
        return m

    def stat_mode(sub):
        return _strat_of_sub(sub, "P_mode")

    def stat_scalar_best(sub):
        vals = [_strat_of_sub(sub, s) for s in SCALARS]
        vals = [v for v in vals if np.isfinite(v)]
        return max(vals) if vals else float("nan")

    d_pt = stat_mode(payload) - stat_scalar_best(payload)
    rng_b = np.random.default_rng(cfg["bootstrap_seed"])
    n = len(payload)
    d_boots = []
    for _ in range(cfg["bootstrap_reps"]):
        idx = rng_b.integers(0, n, n)
        sub = [payload[i] for i in idx]
        d_boots.append(stat_mode(sub) - stat_scalar_best(sub))
    d_ci = (float(np.percentile(d_boots, 2.5)), float(np.percentile(d_boots, 97.5)))

    # pooled descriptive（P_mode vs T；flat，仅描述）
    P_all = [x["P_mode"] for _o, d in payload for x in d.values()]
    T_all = [x["T"] for _o, d in payload for x in d.values()]

    return dict(RA=R_A, RA_ci95=[float(ci95[0]), float(ci95[1])],
                n_pos_cells=n_pos_cells, n_pos_objects=n_pos_obj,
                undefined_cells=undefined_cells,
                n_objects=len(obj_ids),
                pooled_spearman=float(spearman_safe(P_all, T_all)),
                stratified_median_mode=strat["P_mode"]["median"],
                stratified_valid_levels=strat["P_mode"]["valid_levels"],
                stratified_scalar_medians={s: strat[s]["median"] for s in SCALARS},
                best_scalar_name=best_name,
                delta_mode_vs_best_scalar=float(d_pt),
                delta_mode_ci95=[d_ci[0], d_ci[1]])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs/openillumination.yaml"))
    ap.add_argument("--out", default=str(REPO / "results/openillumination/correctness"))
    args = ap.parse_args()
    run(args.config, args.out)
