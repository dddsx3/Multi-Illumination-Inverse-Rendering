"""P1-R4'-D · Discovery Set 复跑（v2 实现稳定性检查；非确认性证据）。

任务书 §4 T4'.0/T4'.2：
  - 现有 4 SUN calibration scene = Discovery Set，仅做实现稳定性检查；
  - 冻结 v2 指标列（primary = scene-normalized λ_min⁺，即
    full_lam_min_pos_norm），确认无数值病态，为预注册提供输入。

输出（p1/information_audit/）：
  ga_isi_v2_discovery.csv          主扫描（pixel_cap=2000, cutoff=1e-8）
  ga_isi_v2_discovery_cap1000.csv  稳定性变体 A（pixel_cap=1000，同子集序列）
  ga_isi_v2_discovery_cut1e6.csv   稳定性变体 B（cutoff=1e-6，同像素）
  R4P_DISCOVERY_RERUN_REPORT.md    稳定性/病态摘要
"""
import csv
import os
import sys

import numpy as np
from scipy.stats import spearmanr

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "p1", "source", "information_audit"))
from gauge_fisher_v2 import load_scene, scene_arrays, ga_isi_v2_scores  # noqa: E402

OUT_DIR = os.path.join(_REPO, "p1", "information_audit")
DATA_ROOT = os.path.join(_REPO, "p1", "calibration_set", "data_sun")
NS = [3, 5, 8, 12]
SUBSETS = 20
SEED = 20260831


def run(tag, out_name, pixel_cap, cutoff):
    scenes = [os.path.join(DATA_ROOT, d) for d in sorted(os.listdir(DATA_ROOT))
              if os.path.isfile(os.path.join(DATA_ROOT, d, "sh_coeffs_irradiance.npy"))]
    rng = np.random.default_rng(SEED)
    rows = []
    for sd in scenes:
        sc = load_scene(sd)
        for N in NS:
            if N > sc["K"]:
                continue
            for _ in range(SUBSETS):
                sub = sorted(rng.choice(sc["K"], N, replace=False).tolist())
                px_seed = int(rng.integers(1 << 31))
                a, Y, C = scene_arrays(sc, sub, pixel_cap=pixel_cap, seed=px_seed)
                r = ga_isi_v2_scores(a, Y, C, cutoff=cutoff)
                r.update(scene=sc["name"], N=N, subset=",".join(map(str, sub)),
                         variant=tag)
                rows.append(r)
        print(f"  [{tag}] {sc['name']} done", flush=True)
    path = os.path.join(OUT_DIR, out_name)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"  [{tag}] -> {path} rows={len(rows)}", flush=True)
    return rows


def rank_stability(base, variant, key="full_lam_min_pos_norm"):
    """同 (scene,N,subset) 的 primary 秩相关（Spearman）。"""
    def index(rows):
        return {(r["scene"], r["N"], r["subset"]): float(r[key]) for r in rows
                if np.isfinite(float(r[key]))}
    ib, iv = index(base), index(variant)
    common = sorted(set(ib) & set(iv))
    x = [ib[k] for k in common]
    y = [iv[k] for k in common]
    sp = spearmanr(x, y)
    return len(common), float(sp.statistic), float(sp.pvalue)


def main():
    print("== R4'-D discovery rerun (v2) ==", flush=True)
    base = run("cap2000_cut1e-8", "ga_isi_v2_discovery.csv", 2000, 1e-8)
    cap1k = run("cap1000_cut1e-8", "ga_isi_v2_discovery_cap1000.csv", 1000, 1e-8)
    cut6 = run("cap2000_cut1e-6", "ga_isi_v2_discovery_cut1e6.csv", 2000, 1e-6)

    lines = ["# R4′-D · Discovery Set 复跑报告（v2 稳定性检查；非确认性证据）", "",
             f"> seed={SEED} · N∈{NS} × {SUBSETS} subsets × 4 scenes · "
             "primary = full_lam_min_pos_norm（scene/trace 归一 λ_min⁺）", ""]

    # 病态检查（主扫描）
    gr = max(float(r["full_gauge_residual"]) for r in base)
    me = min(float(r["full_min_eig"]) / max(float(r["full_trace"]), 1e-300) for r in base)
    rk = {}
    for r in base:
        rk[r["scene"]] = min(rk.get(r["scene"], 9), int(r["rank_Fk_min"]))
    af = min(float(r["active_frac_min"]) for r in base)
    bf = max(float(r["boundary_frac_max"]) for r in base)
    zero_primary = sum(1 for r in base if float(r["full_lam_min_pos_norm"]) <= 0)
    lines += ["## 病态检查（主扫描 cap2000/cutoff1e-8）", "",
              f"- gauge residual 最大值：{gr:.3e}（应 ≲1e-9）",
              f"- PSD 余量 min(λmin/trace)：{me:.3e}（应 ≥ −1e-10）",
              f"- rank(F_k) 逐场景最小值：{rk}（低秩=法线多样性限制，pinv 正确处理）",
              f"- active_frac 最小：{af:.3f} · ReLU 边界占比最大：{bf:.3e}",
              f"- primary ≤0 的 (scene,subset)：{zero_primary}/{len(base)}", ""]

    # 稳定性：秩相关
    n1, s1, p1 = rank_stability(base, cap1k)
    n2, s2, p2 = rank_stability(base, cut6)
    lines += ["## 稳定性（primary 的 Spearman 秩相关）", "",
              f"- pixel_cap 2000 vs 1000（同子集）：ρ={s1:.4f}（n={n1}, p={p1:.2e}）",
              f"- cutoff 1e-8 vs 1e-6（同像素）：ρ={s2:.4f}（n={n2}, p={p2:.2e}）", ""]

    # 存在性前提：固定 N 内 primary 分布宽度（GA-ISI 非退化的前提）
    lines += ["## 存在性前提（固定 N 内 primary 有宽度）", "",
              "| N | min | median | max | IQR/median |", "|---|---|---|---|---|"]
    for N in NS:
        v = np.array([float(r["full_lam_min_pos_norm"]) for r in base
                      if r["N"] == N and float(r["full_lam_min_pos_norm"]) > 0])
        if v.size:
            iqr = (np.percentile(v, 75) - np.percentile(v, 25)) / max(np.median(v), 1e-300)
            lines.append(f"| {N} | {v.min():.3e} | {np.median(v):.3e} | {v.max():.3e} "
                         f"| {iqr:.2f} |")
        else:
            lines.append(f"| {N} | （无正值） | | | |")

    # v1 视角失真量化：proxy vs full 最坏像素信息
    ratios = [float(r["proxy_lam_min_norm"]) / max(float(r["full_lam_min_pos_norm"]), 1e-300)
              for r in base if float(r["full_lam_min_pos_norm"]) > 0]
    if ratios:
        lines += ["", "## v1(diag-proxy) 视角失真量化", "",
                  f"- proxy_lam_min_norm / full_lam_min_pos_norm：median="
                  f"{np.median(ratios):.1f}×，range=[{min(ratios):.1f}×, {max(ratios):.1f}×]"
                  "（v1 的逐像素视角系统性高估最坏像素信息）", ""]
    lines += ["## 结论", "",
              "- 本复跑**只用于**：指标列冻结、数值病态排查、预注册参数选择；",
              "- 不得作为 confirmatory 证据（T4′.0 防双重使用）；",
              "- 确认性统计只在 R4′-C 新确认集上做。", ""]
    rpt = os.path.join(OUT_DIR, "R4P_DISCOVERY_RERUN_REPORT.md")
    with open(rpt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[R4'-D] report -> {rpt}", flush=True)


if __name__ == "__main__":
    main()
