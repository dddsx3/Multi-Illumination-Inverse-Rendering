"""R4″ Task D · Metric bake-off（任务书 §9-§12）。

候选指标（5 类，全部基于统一预处理后的 F_eff 谱）：
  M1 normalized log pdet     : (1/d) Σ log λ_i           （当前首选）
  M2 regularized usable info : (1/d) Σ log(1 + λ_i/τ)    （τ 需由噪声标定）
  M3 A-optimal               : (1/d) Σ λ_i^{-1}
  M4 lower-spectrum quantile : Q_{0.1}(log λ) 或 Q20
  M5 effective rank          : exp(-Σ p_i log p_i), p_i = λ_i/Σλ

稳定性测试（§11，**不看 error 关联**）：
  M-A cutoff stability    : cutoff ∈ {1e-9, 1e-8, 1e-7}，ρ_rank > 0.95
  M-B pixel-cap stability : P ∈ {500, 1000, 2000}，ρ_rank > 0.90
  M-C pixel bootstrap     : 同 P=1000，5 个重采种子，score CV 小 + 排序不漂
  M-D duplicate-light     : 加 1 盏 <5° 的复制光，指标不得大幅上升
  M-E complementary-light : 加 1 盏正交度最高的光，指标应合理上升
  M-F extreme-mode drop   : 删最低 1/3/5 个 eigenmode 后重算，排序不得彻底变

评估单元：6 scene × N{3,5} × 30 subset = 360 units
  （子集与旧 R4′ 同 seed 序列，保证可复现；不用 6×N×30 全量，省算力）

冻结规则（§12）：通过全部 stability test 且数值稳定者冻结为 primary；
  不允许以 error 相关性为选优依据。

用法：python r4pp_metric_bakeoff.py [--limit N]   # --limit 调试用小规模
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

from gauge_fisher_v2 import (fisher_blocks, schur_full, pinv_psd,  # noqa
                             load_scene, scene_arrays)
from sh import sh_basis_npy  # noqa

DATA_ROOT = os.path.join(_REPO, "p1", "calibration_set", "data_sun_confirmatory")
OUT_DIR = os.path.join(_REPO, "r4pp")
SCENES = ["conf_cube_axis", "conf_prism8", "conf_cylinder_r03_d12",
          "conf_cone_r04_d12", "conf_egg", "conf_icosphere_sub3"]
NS = [3, 5]
SUBSETS_PER_N = 30
SUBSET_SEED = 20260902        # 与旧 R4′ 完全相同的序列
PIXEL_CAP = 1000
CUTOFF_REF = 1e-8
M2_TAU = 1e-5                 # 待由噪声标定更新（占位）

RESULTS_CSV = os.path.join(OUT_DIR, "03_metric_stability.csv")
RAW_CSV = os.path.join(OUT_DIR, "03_metric_raw_scores.csv")


def replay_units(limit=None):
    """复现 (scene, N, subset, pixel_seed) 序列（与 scores 阶段一致）。"""
    rng = np.random.default_rng(SUBSET_SEED)
    units = []
    for scn in SCENES:
        sc = load_scene(os.path.join(DATA_ROOT, scn))
        for N in NS:
            for _ in range(SUBSETS_PER_N):
                sub = sorted(rng.choice(sc["K"], N, replace=False).tolist())
                px = int(rng.integers(1 << 31))
                units.append((scn, N, sub, px))
    return units[:limit] if limit else units


# ---------------- 候选指标（输入：F_eff 的归一化谱 wn = w/trace）----------------
def m1_log_pdet(wn, pos_cut=1e-12):
    pos = wn[wn > pos_cut]
    if pos.size == 0:
        return float("-inf")
    return float(np.log(pos).mean())


def m2_reg_usable(wn, tau=M2_TAU):
    return float(np.log1p(wn / tau).mean())


def m3_a_opt(wn, pos_cut=1e-12):
    pos = wn[wn > pos_cut]
    if pos.size == 0:
        return float("inf")
    return float((1.0 / pos).mean())


def m4_lower_quantile(wn, q=0.1):
    pos = wn[wn > 0]
    if pos.size == 0:
        return float("-inf")
    return float(np.percentile(np.log(pos), q * 100))


def m5_eff_rank(wn):
    s = wn.sum()
    if s <= 0:
        return 0.0
    p = wn[wn > 0] / s
    return float(np.exp(-(p * np.log(p)).sum()))


METRICS = {"M1": m1_log_pdet, "M2": m2_reg_usable, "M3": m3_a_opt,
           "M4": m4_lower_quantile, "M5": m5_eff_rank}


def spectrum_of(a, Y, C, cutoff=CUTOFF_REF, cap=PIXEL_CAP):
    """统一预处理 → 归一化谱（任务书 §D1）。"""
    bl = fisher_blocks(a, Y, C)
    F = schur_full(bl, cutoff)
    w = np.linalg.eigvalsh(F)
    tr = float(w.sum())
    return (w / tr) if tr > 0 else w * 0.0


def load_arrays(scn, sub, px, cap):
    sc = load_scene(os.path.join(DATA_ROOT, scn))
    return scene_arrays(sc, sub, pixel_cap=cap, seed=px)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="调试：只跑前 N 个 unit")
    ap.add_argument("--no_duplicate", action="store_true", help="跳过 M-D/M-E（省时）")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    units = replay_units(args.limit)
    print(f"[bakeoff] {len(units)} units × {len(METRICS)} metrics × 6 tests")

    # 每个 unit 计算所有 metric 的所有 variant 分数 → raw
    raw_rows = []
    t0 = time.time()
    for ui, (scn, N, sub, px) in enumerate(units):
        # 基准：cutoff=1e-8, cap=1000
        a, Y, C = load_arrays(scn, sub, px, PIXEL_CAP)
        base_wn = spectrum_of(a, Y, C)
        base_scores = {m: METRICS[m](base_wn) for m in METRICS}

        raw = dict(scene=scn, N=N, subset=",".join(map(str, sub)), pixel_seed=px,
                   base_cut=CUTOFF_REF, base_cap=PIXEL_CAP)
        for m, v in base_scores.items():
            raw[f"{m}_base"] = v

        # M-A cutoff sweep（1e-9, 1e-7）
        for c in [1e-9, 1e-7]:
            wn = spectrum_of(a, Y, C, cutoff=c)
            for m in METRICS:
                raw[f"{m}_cut{c:.0e}"] = METRICS[m](wn)

        # M-B pixel-cap（500, 2000）
        for cap in [500, 2000]:
            a2, Y2, C2 = load_arrays(scn, sub, px, cap)
            wn = spectrum_of(a2, Y2, C2, cap=cap)
            for m in METRICS:
                raw[f"{m}_cap{cap}"] = METRICS[m](wn)

        # M-C pixel bootstrap（cap=1000，5 个新重采种子）
        for b in range(5):
            a3, Y3, C3 = scene_arrays(load_scene(os.path.join(DATA_ROOT, scn)),
                                      sub, pixel_cap=1000, seed=px + 1000 * (b + 1))
            wn = spectrum_of(a3, Y3, C3)
            for m in METRICS:
                raw[f"{m}_boot{b}"] = METRICS[m](wn)

        # M-D duplicate-light：加一盏与子集内某盏光夹角 < 5° 的复制光
        if not args.no_duplicate:
            sc = load_scene(os.path.join(DATA_ROOT, scn))
            dirs = sc["sh_irr"][np.asarray(sub)]
            # 取子集内第一盏光，加一个 3° 扰动作为"几乎重复"光
            dup = dirs[0] * 0.999 + np.random.default_rng(px).normal(0, 0.02, 9)
            C_dup = np.vstack([sc["sh_irr"][np.asarray(sub)], dup[None]])
            a4, Y4, _ = load_arrays(scn, sub, px, PIXEL_CAP)
            wn = spectrum_of(a4, Y4, C_dup)
            for m in METRICS:
                raw[f"{m}_dup"] = METRICS[m](wn)

            # M-E complementary-light：加一盏与现有 span 正交度最高的光
            cands = np.delete(sc["sh_irr"], np.asarray(sub), axis=0)
            # 与子集 span 的正交度 = 残差范数（用 SVD 投影后残差）
            U, _, _ = np.linalg.svd(sc["sh_irr"][np.asarray(sub)].T, full_matrices=False)
            proj = cands @ U @ U.T
            resid = np.linalg.norm(cands - proj, axis=1)
            comp = cands[np.argmax(resid)]
            C_comp = np.vstack([sc["sh_irr"][np.asarray(sub)], comp[None]])
            wn = spectrum_of(a4, Y4, C_comp)
            for m in METRICS:
                raw[f"{m}_comp"] = METRICS[m](wn)

        # M-F extreme-mode drop：删最低 1/3/5 个 eigenmode 后重算
        for k_drop in [1, 3, 5]:
            w_sorted = np.sort(base_wn)
            w_dropped = w_sorted[k_drop:]
            for m in METRICS:
                raw[f"{m}_drop{k_drop}"] = METRICS[m](w_dropped)

        raw_rows.append(raw)
        if (ui + 1) % 60 == 0:
            print(f"  [{ui+1}/{len(units)}] ({time.time()-t0:.0f}s)", flush=True)

    # 落盘 raw
    keys = list(dict.fromkeys(k for r in raw_rows for k in r))
    with open(RAW_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader(); w.writerows(raw_rows)
    print(f"[bakeoff] raw -> {RAW_CSV} rows={len(raw_rows)}")

    # ---------------- 稳定性统计（§11 判据，不看 error）----------------
    from scipy.stats import spearmanr
    rows = []
    for m in METRICS:
        base = np.array([r[f"{m}_base"] for r in raw_rows])
        finite = np.isfinite(base)
        base_f = base[finite]

        # M-A: cutoff 1e-9/1e-7 vs 1e-8
        rho_a1 = spearmanr(base_f, np.array([r[f"{m}_cut1e-09"] for r in raw_rows])[finite]).statistic
        rho_a2 = spearmanr(base_f, np.array([r[f"{m}_cut1e-07"] for r in raw_rows])[finite]).statistic
        m_a = min(rho_a1, rho_a2)

        # M-B: cap 500/2000 vs 1000
        rho_b1 = spearmanr(base_f, np.array([r[f"{m}_cap500"] for r in raw_rows])[finite]).statistic
        rho_b2 = spearmanr(base_f, np.array([r[f"{m}_cap2000"] for r in raw_rows])[finite]).statistic
        m_b = min(rho_b1, rho_b2)

        # M-C: pixel bootstrap 的 CV + 与 base 的秩相关
        cv_list = []
        rho_c_list = []
        for b in range(5):
            bv = np.array([r[f"{m}_boot{b}"] for r in raw_rows])[finite]
            cv_list.append(np.nanstd(bv) / max(abs(np.nanmean(bv)), 1e-12))
            rho_c_list.append(spearmanr(base_f, bv).statistic)
        m_c_cv = float(np.nanmean(cv_list))
        m_c_rho = float(np.nanmin(rho_c_list))

        # M-D: duplicate 不得大幅上升
        dup = np.array([r.get(f"{m}_dup", np.nan) for r in raw_rows])[finite]
        # 对"越大越好"的指标（M1/M2/M4/M5），dup 应 < base × 1.5
        # 对"越小越好"的 M3（A-opt），dup 应 < base × 1.5（A-opt 变大=信息变差）
        ratio = dup / base_f
        m_d_ratio = float(np.nanmedian(ratio))

        # M-E: complementary 应合理上升（M1/M2/M4/M5 变大；M3 变小）
        comp = np.array([r.get(f"{m}_comp", np.nan) for r in raw_rows])[finite]
        m_e_ratio = float(np.nanmedian(comp / base_f))

        # M-F: 删最低 mode 后排序不得彻底变
        rho_f_list = []
        for k in [1, 3, 5]:
            fv = np.array([r[f"{m}_drop{k}"] for r in raw_rows])[finite]
            rho_f_list.append(spearmanr(base_f, fv).statistic)
        m_f = float(np.nanmin(rho_f_list))

        rows.append(dict(
            metric=m,
            test_MA_cutoff_rho=max(abs(m_a), 0), test_MB_cap_rho=max(abs(m_b), 0),
            test_MC_boot_cv=m_c_cv, test_MC_boot_rho=m_c_rho,
            test_MD_dup_ratio_med=m_d_ratio, test_ME_comp_ratio_med=m_e_ratio,
            test_MF_modedrop_rho_min=m_f,
            MA_PASS=bool(m_a > 0.95), MB_PASS=bool(m_b > 0.90),
            MC_PASS=bool(m_c_rho > 0.85 and m_c_cv < 0.3),
            MD_PASS=bool(m_d_ratio < 1.5), ME_PASS=bool(m_e_ratio > 1.0 or m_e_ratio < 1.0),
            MF_PASS=bool(m_f > 0.8),
        ))
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"[bakeoff] -> {RESULTS_CSV}")
    for r in rows:
        print(f"  {r['metric']}: MA ρ={r['test_MA_cutoff_rho']:.3f} | "
              f"MB ρ={r['test_MB_cap_rho']:.3f} | MC ρ={r['test_MC_boot_rho']:.3f} "
              f"CV={r['test_MC_boot_cv']:.3f} | MD dup={r['test_MD_dup_ratio_med']:.2f} | "
              f"MF ρ={r['test_MF_modedrop_rho_min']:.3f}")

    # 冻结建议（§12：按稳定性，不看 error）
    best = max(rows, key=lambda r: sum([r["MA_PASS"], r["MB_PASS"], r["MC_PASS"],
                                        r["MD_PASS"], r["MF_PASS"]]))
    print(f"\n[冻结建议] {best['metric']}（通过 {sum([best[k] for k in ['MA_PASS','MB_PASS','MC_PASS','MD_PASS','MF_PASS']])}/5 项）")
    print("[提醒] 最终冻结需 6 项全 PASS；M2 的 τ 需由噪声标定（任务书 §10）")


if __name__ == "__main__":
    main()
