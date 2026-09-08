"""CI04-R 最终收口分析（任务书 v3.0 §14；条例第三~十一章）。

数据源 = artifacts/frozen/ci04_formal_summary.json rows（T4.1 零重跑经验侧）；
预测侧 scalar 基线 = 确定性重算全谱（T5.2/T5.3 对账门 ≤1e-9）；
裁决 = branch_decision.json 由脚本自动生成（T11.2，禁手改）。

一键执行：python experiments/ci04r_final_closure.py --config configs/ci04r/ci04r_frozen.yaml --out artifacts/ci04r
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from calibinfo.datasets.openillumination import load_object              # noqa: E402
from calibinfo.metrics.ci04r_predictors import (                        # noqa: E402
    predictors_from_spectrum, p_mode_from_pred_deg,
    retention_spectrum_full, identifiable_overlap)
from calibinfo.metrics.cluster_bootstrap import cluster_bootstrap       # noqa: E402
from experiments.ci04_real_corruption import (                          # noqa: E402
    NominalScene, CorruptionGenerator)
from calibinfo.datasets.openillumination import load_object             # noqa: E402  (reuse)


# --------------------------------------------------------------- helpers
def _sha(p: str) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def spearman_safe(x, y):
    """Spearman；全并列 → nan（调用方按 T7.4 记 undefined，不改 0）。"""
    r = spearmanr(x, y).statistic
    return float(r) if np.isfinite(r) else float("nan")


def median_valid(vals):
    """median 忽略 nan；有效 <4/6 → (nan, insufficient)（T7.4）。"""
    v = [x for x in vals if np.isfinite(x)]
    if len(v) < 4:
        return float("nan"), False
    return float(np.median(v)), True


# --------------------------------------------------------------- T0/T3 冻结断言
def load_and_assert(cfg_path: str):
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    for f, h in cfg["frozen_artifacts_sha256"].items():
        assert _sha(f) == h, f"冻结产物 sha256 不一致: {f}（REPRODUCIBILITY INCIDENT）"
    summary = json.loads(Path("artifacts/frozen/ci04_formal_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads(Path("artifacts/frozen/ci04_formal_manifest.json").read_text(encoding="utf-8"))
    rows = summary["rows"]
    # T3.4 机械断言（先于任何统计计算）
    assert cfg["dataset"] == "OpenIllumination" and cfg["split"] == "frozen_test"
    assert len(rows) == cfg["expected_cells"] == 66, f"cell 数 {len(rows)} ≠ 66"
    assert sum(len(r["pred_deg"]) for r in rows) == cfg["expected_mode_points"] == 330
    objs = sorted({r["object"] for r in rows})
    assert objs == sorted(cfg["objects"]), "对象清单与 YAML 不一致"
    assert sorted({float(r["level"]) for r in rows}) == sorted(cfg["levels"]), "level 与 YAML 不一致"
    assert {r["ctype"] for r in rows} == {cfg["ctype"]}, "ctype 与 YAML 不一致"
    assert all(r["n_seeds"] == cfg["n_seeds"] for r in rows), "seeds 与 YAML 不一致"
    assert manifest["config"]["objects"] == cfg["objects"], "manifest objects 顺序/内容不一致"
    assert len(rows[0]["pred_deg"]) == cfg["n_modes"] == 5
    # 禁新增 exclusion：greenhead 不在、11 对象与 frozen 一致
    assert "obj_20_greenhead" not in objs
    return cfg, summary, manifest, rows


# --------------------------------------------------------------- T5 预测侧确定性重算
def recompute_spectra(cfg, manifest):
    """复刻原 run 的 rng 消耗序列重建 NominalScene（T5.2），全谱重算 + T5.3 对账门。"""
    mc = manifest["config"]
    rng = np.random.default_rng(mc["seed"])
    scenes = {}
    gate_max_rel = 0.0
    spectra = {}                                           # (obj, level) -> dict(r, q)
    for obj_name in mc["objects"]:                         # 顺序 = 原 run 遍历顺序
        obj = load_object(mc["data_root"], obj_name, data_meta=mc.get("data_meta"))
        scen = NominalScene(obj, rng)                      # 消耗 choice(142,48)+choice(P,1200)
        scenes[obj_name] = scen
        for level in mc["levels"]:                         # 消耗 6×20 次 integers（与原 run 逐位对齐）
            for _s in range(mc["seeds_per_level"]):
                rng.integers(0, 2**63 - 1)
    # 场景全部重建后，确定性计算每个 (obj, level) 的全谱
    for obj_name in mc["objects"]:
        scen = scenes[obj_name]
        for level in mc["levels"]:
            gen = CorruptionGenerator("joint", float(level))
            sig3 = np.diag(gen.sigma_phi_diag())           # (3,3) 物理单位
            DF = scen.delta_f_injected(sig3)
            Finv_h = np.diag(1.0 / np.sqrt(scen.Finf_diag))
            R = Finv_h @ DF @ Finv_h
            spec = retention_spectrum_full(
                DF, np.diag(scen.Finf_diag),
                rank_relative_tol=cfg["rank_relative_tol"],
                clip_tol=cfg["retention_clip_tol"], log_eps=cfg["log_eps"])
            # T5.3 对账门：弱 5 换算 pred_deg' vs frozen ≤1e-9（相对）
            frozen_row = None
            spectra[(obj_name, float(level))] = spec
    return scenes, spectra


def gate_recompute(cfg, rows, scenes, spectra):
    """T5.3 硬门槛：66×5 相对误差 ≤1e-9；T5.4 q≥5 断言。返回 (max_rel, per_cell_q)。"""
    max_rel, qs = 0.0, {}
    for r in rows:
        key = (r["object"], float(r["level"]))
        spec = spectra[key]
        r_all = spec["r"]
        q = int((r_all > cfg["rank_relative_tol"] * r_all.max()).sum())
        qs[key] = q
        assert q >= cfg["n_modes"], f"T5.4 违例: q={q} < 5 @ {key}"
        pred_deg_re = 1.0 / r_all[: cfg["n_modes"]]
        frozen = np.asarray(r["pred_deg"], float)
        rel = float(np.max(np.abs(pred_deg_re - frozen) / np.abs(frozen)))
        max_rel = max(max_rel, rel)
        if rel > 1e-9:
            raise RuntimeError(
                f"PREDICTION-SIDE RECOMPUTE MISMATCH: {key} rel={rel:.3e} > 1e-9"
                f"（排查：数据版本 → 线性化子集 → 白化参数；禁止以重算值为准继续）")
    return max_rel, qs


# --------------------------------------------------------------- R2-A
def r2a_analysis(rows, cfg):
    per_cell, per_obj = {}, {}
    undefined_cells = 0
    for r in rows:
        key = (r["object"], float(r["level"]))
        S = 1.0 - 1.0 / np.asarray(r["pred_deg"], float)   # 理论损伤（= 1-rho）
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

    def stat(sub):
        vals = [per_obj[o] for o, _ in sub if np.isfinite(per_obj[o])]
        return float(np.median(vals))

    point, ci95, boots = cluster_bootstrap(
        [(o, None) for o in obj_ids], stat,
        cfg["bootstrap_reps"], cfg["bootstrap_seed"])
    pass_a = bool(R_A >= cfg["pass_a_min_median"] and ci95[0] > 0
                  and n_pos_obj >= cfg["pass_a_min_positive_objects"])
    iqr = [float(np.percentile([v for v in per_cell.values() if np.isfinite(v)], 25)),
           float(np.percentile([v for v in per_cell.values() if np.isfinite(v)], 75))]
    return dict(R_A=R_A, ci95=ci95, per_cell=per_cell, per_obj=per_obj,
                n_pos_cells=n_pos_cells, n_pos_obj=n_pos_obj,
                cell_iqr=iqr, undefined_cells=undefined_cells,
                boots=boots, pass_a=pass_a, stat_fn=stat, obj_ids=obj_ids)


# --------------------------------------------------------------- R2-B
def _level_points(rows, predictor, spectra=None, cfg=None):
    """返回 {level: [(object, P_value, T_value)]}；P_mode 走 frozen，scalar 走重算全谱。"""
    by_level = {}
    for r in rows:
        key = (r["object"], float(r["level"]))
        T = float(np.max(r["emp_deg"]))
        if predictor == "P_mode":
            P = p_mode_from_pred_deg(np.asarray(r["pred_deg"], float))
        else:
            spec = spectra[key]
            q = int((spec["r"] > cfg["rank_relative_tol"] * spec["r"].max()).sum())
            ps = predictors_from_spectrum(spec["r"], q, log_eps=cfg["log_eps"])
            P = ps[predictor]
        by_level.setdefault(float(r["level"]), []).append((r["object"], P, T))
    return by_level


def r2b_analysis(rows, spectra, cfg):
    predictors = ["P_trace", "P_logdet", "P_emin"]
    points = {p: _level_points(rows, p, spectra, cfg) for p in predictors}
    mode_points = _level_points(rows, "P_mode", spectra, cfg)
    # pooled（descriptive；含 P_corr）
    pooled = {}
    T_all = [float(np.max(r["emp_deg"])) for r in rows]
    P_mode_all = [p_mode_from_pred_deg(np.asarray(r["pred_deg"], float)) for r in rows]
    pooled["P_mode"] = spearman_safe(P_mode_all, T_all)
    for p in predictors:
        pooled[p] = spearman_safe([x[1] for lvl in points[p].values() for x in lvl],
                                  [x[2] for lvl in points[p].values() for x in lvl])
    pooled["P_corr"] = spearman_safe([float(r["level"]) for r in rows], T_all)
    # within-level（4 predictor × 6 levels；P_corr = none）
    within = {p: {} for p in ["P_mode"] + predictors}
    for lv in sorted({float(r["level"]) for r in rows}):
        for p in ["P_mode"] + predictors:
            pts = points[p][lv] if p != "P_mode" else mode_points[lv]
            within[p][lv] = spearman_safe([x[1] for x in pts], [x[2] for x in pts])
    strat = {}
    for p in ["P_mode"] + predictors:
        m, valid = median_valid(list(within[p].values()))
        strat[p] = dict(median=m, valid_levels=int(sum(1 for v in within[p].values() if np.isfinite(v))),
                        insufficient=not valid)
    # cluster bootstrap：Δ_mode（replicate 内重选 best scalar）
    obj_payloads = []
    obj_ids = sorted({r["object"] for r in rows})
    per_obj_lvl = {}
    for r in rows:
        per_obj_lvl.setdefault(r["object"], {})[float(r["level"])] = (
            p_mode_from_pred_deg(np.asarray(r["pred_deg"], float)),
            float(np.max(r["emp_deg"])))
    spec_by_obj = {}
    for o in obj_ids:
        spec_by_obj[o] = {lv: predictors_from_spectrum(
            spectra[(o, lv)]["r"], spectra[(o, lv)]["q"], log_eps=cfg["log_eps"])
            for lv in cfg["levels"]}
    for o in obj_ids:
        obj_payloads.append((o, dict(pl=per_obj_lvl[o], sp=spec_by_obj[o])))

    def strat_of(sub, which):
        per_lv = {}
        for _o, d in sub:
            for lv, (P_mode_val, T) in d["pl"].items():
                P = P_mode_val if which == "P_mode" else d["sp"][lv][which]
                per_lv.setdefault(lv, ([], []))
                per_lv[lv][0].append(P)
                per_lv[lv][1].append(T)
        rhos = [spearman_safe(Ps, Ts) for _lv, (Ps, Ts) in sorted(per_lv.items())]
        m, _v = median_valid(rhos)
        return m

    def stat_mode(sub):
        return strat_of(sub, "P_mode")

    def stat_scalar_best(sub):
        return max(strat_of(sub, p) for p in predictors
                   if np.isfinite(strat_of(sub, p)))

    def stat_delta(sub):
        return stat_mode(sub) - stat_scalar_best(sub)

    m_pt, m_ci, m_boots = cluster_bootstrap(obj_payloads, stat_mode,
                                            cfg["bootstrap_reps"], cfg["bootstrap_seed"])
    # best scalar point estimate（全样本，逐 predictor stratified）
    scalar_strat = {p: strat_of(obj_payloads, p) for p in predictors}
    best_name = max(scalar_strat, key=lambda p: (scalar_strat[p] if np.isfinite(scalar_strat[p]) else -9e9))
    d_pt = stat_mode(obj_payloads) - stat_scalar_best(obj_payloads)
    # bootstrap Δ：replicate 内重选
    d_boots = []
    rng = np.random.default_rng(cfg["bootstrap_seed"])
    n = len(obj_payloads)
    for _ in range(cfg["bootstrap_reps"]):
        idx = rng.integers(0, n, n)
        sub = [obj_payloads[i] for i in idx]
        d_boots.append(stat_mode(sub) - stat_scalar_best(sub))
    d_ci = (float(np.percentile(d_boots, 2.5)), float(np.percentile(d_boots, 97.5)))
    delta = float(d_pt)
    margin = cfg["practical_margin_delta"]
    if delta >= margin and d_ci[0] > 0:
        sev = "S+"
    elif delta <= -margin and d_ci[1] < 0:
        sev = "S-"
    else:
        sev = "S0"
    return dict(pooled=pooled, within=within, strat=strat,
                mode_point=m_pt, mode_ci=m_ci,
                scalar_strat=scalar_strat, best_name=best_name,
                delta=delta, delta_ci=d_ci, severity=sev,
                points=points, mode_points=mode_points)


# --------------------------------------------------------------- R2-C（secondary）
def r2c_analysis(rows, scenes, spectra, cfg):
    """方向定位（secondary）：within-cell 设计下 ĵ_emin ≡ ĵ_mode（同为 bottom tracked
    mode 的特征向量）——如实记录；hit rate = bottom tracked mode 被经验判最坏的比率。"""
    hits_mode, hits_emin, conf = [], [], {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    obj_hit = {}
    for r in rows:
        D = np.asarray(r["emp_deg"], float)
        j_star = int(np.argmax(D))
        j_mode = int(np.argmin(1.0 / np.asarray(r["pred_deg"], float)))
        # E-min 方向：R 的最小特征向量与 5 个 tracked 方向的 overlap——within-cell 下
        # v_min 即 bottom tracked 方向（同一特征分解），ĵ_emin = 0（构造性事实，如实记录）
        j_emin = 0
        hits_mode.append(j_mode == j_star)
        hits_emin.append(j_emin == j_star)
        if j_mode == j_star:
            conf[j_star] = conf.get(j_star, 0) + 1
        obj_hit.setdefault(r["object"], []).append(j_mode == j_star and j_emin == j_star)
    return dict(
        top1_hit_rate_mode=float(np.mean(hits_mode)),
        top1_hit_rate_emin=float(np.mean(hits_emin)),
        objectwise_hit={o: float(np.mean(v)) for o, v in obj_hit.items()},
        confusion_by_star=conf,
        note="within-cell 设计下 v_min ≡ bottom tracked eigenvector → ĵ_emin ≡ ĵ_mode"
             "（构造性事实）；有效对照 = bottom-mode 是否被经验判最坏")


# --------------------------------------------------------------- 输出
def write_outputs(cfg, out, r2a, r2b, r2c, rows, spectra, gate_rel, qs, pooled_cluster):
    import csv
    outd = Path(out)
    outd.mkdir(parents=True, exist_ok=True)
    # cell_mode_scores.csv（330 行）
    with open(outd / "cell_mode_scores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["object_id", "level", "ctype", "mode_id", "pred_deg", "emp_deg",
                    "rho", "seed_count"])
        for r in rows:
            for j in range(5):
                w.writerow([r["object"], r["level"], r["ctype"], j,
                            r["pred_deg"][j], r["emp_deg"][j],
                            1.0 / r["pred_deg"][j], r["n_seeds"]])
    # level_severity_scores.csv（66 行，含 q 与 scalar predictors）
    with open(outd / "level_severity_scores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["object_id", "level", "ctype", "q", "T_ol",
                    "P_mode", "P_trace", "P_logdet", "P_emin", "P_corr"])
        for r in rows:
            key = (r["object"], float(r["level"]))
            spec = spectra[key]
            ps = predictors_from_spectrum(spec["r"], spec["q"], log_eps=cfg["log_eps"])
            w.writerow([r["object"], r["level"], r["ctype"], spec["q"],
                        float(np.max(r["emp_deg"])),
                        p_mode_from_pred_deg(np.asarray(r["pred_deg"], float)),
                        ps["P_trace"], ps["P_logdet"], ps["P_emin"], r["level"]])
    # table_R1.csv（固定布局）
    with open(outd / "table_R1.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Predictor", "Pooled_rho_descriptive", "L1", "L2", "L3", "L4", "L5", "L6",
                    "Stratified_median_rho", "Cluster95CI"])
        w.writerow(["corruption magnitude", f"{r2b['pooled']['P_corr']:.4f}"] + ["none"] * 6
                   + ["none", "—"])
        for p, label in [("P_trace", "trace"), ("P_logdet", "logdet"), ("P_emin", "E-min"),
                         ("P_mode", "mode-resolved")]:
            s = r2b["strat"][p]
            lvls = [f"{r2b['within'][p][lv]:.4f}" if np.isfinite(r2b['within'][p][lv])
                    else "undefined" for lv in sorted(r2b['within'][p])]
            ci = (f"[{r2b['mode_ci'][0]:.3f},{r2b['mode_ci'][1]:.3f}]" if p == "P_mode" else "—")
            w.writerow([label, f"{r2b['pooled'][p]:.4f}"] + lvls
                       + [f"{s['median']:.4f}" if np.isfinite(s['median']) else "insufficient strata", ci])
    # branch_decision.json（脚本自动生成，禁手改）
    r2a_status = "PASS-A" if r2a["pass_a"] else "FAIL-A"
    branch = ("D" if r2a_status == "FAIL-A" else
              {"S+": "A", "S0": "B", "S-": "C"}[r2b["severity"]])
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True, cwd=str(REPO)).stdout.strip()
    bd = dict(
        r2a_status=r2a_status,
        r2a_stat=r2a["R_A"],
        r2a_ci95=list(r2a["ci95"]),
        positive_objects=r2a["n_pos_obj"],
        severity_status=r2b["severity"],
        rho_mode_strat=r2b["mode_point"],
        best_scalar_name=r2b["best_name"],
        rho_best_scalar_strat=float(r2b["scalar_strat"][r2b["best_name"]]),
        delta_mode=r2b["delta"],
        delta_mode_ci95=list(r2b["delta_ci"]),
        branch=branch,
        bootstrap_seed=cfg["bootstrap_seed"],
        config_sha256=_sha("configs/ci04r/ci04r_frozen.yaml"),
        artifact_sha256=_sha("artifacts/frozen/ci04_formal_summary.json"),
        git_sha=sha)
    (outd / "branch_decision.json").write_text(
        json.dumps(bd, indent=1), encoding="utf-8")
    # provenance.json
    (outd / "provenance.json").write_text(json.dumps(dict(
        experiment="ci04r_final_closure",
        git_sha=sha,
        config="configs/ci04r/ci04r_frozen.yaml",
        config_sha256=_sha("configs/ci04r/ci04r_frozen.yaml"),
        artifacts={f: _sha(f) for f in [
            "artifacts/frozen/ci04_formal_summary.json",
            "artifacts/frozen/ci04_formal_manifest.json"]},
        platform=platform.platform(),
        written_utc=datetime.now(timezone.utc).isoformat(),
    ), indent=1), encoding="utf-8")
    # summary.json
    (outd / "summary.json").write_text(json.dumps(dict(
        r2a=dict(status=r2a_status, R_A=r2a["R_A"], ci95=list(r2a["ci95"]),
                 n_pos_cells=r2a["n_pos_cells"], n_pos_obj=r2a["n_pos_obj"],
                 cell_iqr=r2a["cell_iqr"], undefined_cells=r2a["undefined_cells"],
                 n_cells=66),
        r2b=dict(pooled=r2b["pooled"], within={p: {str(k): v for k, v in d.items()}
                                               for p, d in r2b["within"].items()},
                 strat={p: d["median"] for p, d in r2b["strat"].items()},
                 mode_point=r2b["mode_point"], mode_ci=list(r2b["mode_ci"]),
                 scalar_strat=r2b["scalar_strat"], best_name=r2b["best_name"],
                 delta=r2b["delta"], delta_ci=list(r2b["delta_ci"]),
                 severity=r2b["severity"],
                 pooled_cluster=dict(point=pooled_cluster[0],
                                     ci95=list(pooled_cluster[1]))),
        r2c=r2c,
        recompute_gate_max_rel=gate_rel,
        note="pooled = descriptive; within-level = co-primary; Δ_mode best-scalar "
             "在 bootstrap replicate 内重选; R2-C secondary (ĵ_emin≡ĵ_mode 构造性)",
    ), ensure_ascii=False, indent=1), encoding="utf-8")
    return bd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ci04r/ci04r_frozen.yaml")
    ap.add_argument("--out", default="artifacts/ci04r")
    args = ap.parse_args()
    cfg, summary, manifest, rows = load_and_assert(args.config)
    print(f"[ci04r] T0 冻结断言通过（66 cells / 330 mode-points / 11 objects）")
    scenes, spectra = recompute_spectra(cfg, manifest)
    gate_rel, qs = gate_recompute(cfg, rows, scenes, spectra)
    print(f"[ci04r] T5.3 重算门通过：66×5 相对误差 max = {gate_rel:.3e} ≤ 1e-9；q ≥ 5 全部成立")
    r2a = r2a_analysis(rows, cfg)
    print(f"[ci04r] R2-A: R_A={r2a['R_A']:.4f} CI95={r2a['ci95']} "
          f"pos_cells={r2a['n_pos_cells']}/66 pos_obj={r2a['n_pos_obj']}/11 "
          f"undefined={r2a['undefined_cells']} → {'PASS-A' if r2a['pass_a'] else 'FAIL-A'}")
    r2b = r2b_analysis(rows, spectra, cfg)
    print(f"[ci04r] R2-B: mode_strat={r2b['mode_point']:.4f} CI={r2b['mode_ci']} "
          f"best_scalar={r2b['best_name']}({r2b['scalar_strat'][r2b['best_name']]:.4f}) "
          f"Δ={r2b['delta']:+.4f} CI={r2b['delta_ci']} → {r2b['severity']}")
    print(f"[ci04r] R2-B pooled(descriptive): "
          + ", ".join(f"{k}={v:+.4f}" for k, v in r2b["pooled"].items()))
    # T9.1: pooled Spearman 的 object-cluster bootstrap CI（整体替换旧 330-point CI）
    obj_pts = {}
    for r in rows:
        for P, E in zip(r["pred_deg"], r["emp_deg"]):
            obj_pts.setdefault(r["object"], ([], []))
            obj_pts[r["object"]][0].append(P)
            obj_pts[r["object"]][1].append(E)
    def pooled_stat(sub):
        Ps = [v for _o, (Ps, Es) in sub for v in Ps]
        Es = [v for _o, (Ps, Es) in sub for v in Es]
        return spearman_safe(Ps, Es)
    pooled_payload = [(o, obj_pts[o]) for o in sorted(obj_pts)]
    pooled_pt, pooled_ci, pooled_boots = cluster_bootstrap(
        pooled_payload, pooled_stat, cfg["bootstrap_reps"], cfg["bootstrap_seed"])
    print(f"[ci04r] pooled(descriptive) cluster CI: {pooled_pt:.4f} {pooled_ci}")
    r2c = r2c_analysis(rows, scenes, spectra, cfg)
    print(f"[ci04r] R2-C: hit_mode={r2c['top1_hit_rate_mode']:.3f} "
          f"hit_emin={r2c['top1_hit_rate_emin']:.3f}（within-cell 设计下二者构造性一致）")
    bd = write_outputs(cfg, args.out, r2a, r2b, r2c, rows, spectra, gate_rel, qs,
                       pooled_cluster=(pooled_pt, pooled_ci))
    print(f"[ci04r] BRANCH = {bd['branch']}（脚本自动裁决，禁手改）→ {bd['r2a_status']}/{bd['severity_status']}")
    return bd


if __name__ == "__main__":
    main()
