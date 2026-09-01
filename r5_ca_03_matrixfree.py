"""R5 Compute-Aware Campaign · Route C · Matrix-free GSIQ (C1 + C2)

C1: matvec 正确性 — 无矩阵的 F·v 乘法 vs 显式 F@v, 相对误差 median < 1e-5 才 GO。
C2: SLQ (随机 Lanczos 求积) 估计 logdet — 与精确 GSIQ 的排名一致性:
      最低 GO: Spearman rho >= 0.95 (在 50 步)
      推荐 GO: Kendall tau >= 0.9

理论要点 (不改变 GSIQ 定义, 只换算法):
  GSIQ = (1/d+) Σ_{λ̃>1e-8} log(λ̃),  λ̃ = λ/trace(F_eff)
  SLQ 目标改为 G = (1/P) Σ_i log(λ̃_i + c),  c = 1e-8 (把零空间推到 log(c))
  关系: G ≈ (d+/P)·GSIQ + ((P-d+)/P)·log(c)
  => 修正版 I_GS ≈ (P/d+)·(G - ((P-d+)/P)·log(c)), 其中 d+ 用精确值 (仅评估用)

用法 (本地 Windows 直接跑, 无需 GPU):
  python r5_ca_03_matrixfree.py                       # 默认: C1@P={500,2000} + C2@P=1000
  python r5_ca_03_matrixfree.py --quick               # 快速 smoke (~2 min)
输出:
  r5_compute_audit/runtime/matrixfree.csv
  r5_compute_audit/decision_reports/C_verdict.md      <- 大白话结论
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.stats import spearmanr, kendalltau

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "p1" / "source" / "information_audit"))

from gauge_fisher_v2 import (  # noqa: E402
    fisher_blocks, schur_full, schur_operator, spectrum_metrics,
    ga_isi_v2_scores, load_scene, sh_basis_npy,
)

DATA_ROOT = REPO / "p1" / "calibration_set" / "data_sun_confirmatory"
OUT_DIR = REPO / "r5_compute_audit"
SPEC_CUTOFF = 1e-8
SCENES = ["conf_sphere_r05", "conf_cube_axis"]


def build_arrays(scene, P, N, seed):
    sc = load_scene(str(DATA_ROOT / scene))
    idx_all = np.argwhere(sc["mask"])
    take = min(P, len(idx_all))
    rng = np.random.default_rng(seed)
    idx = idx_all[rng.choice(len(idx_all), take, replace=False)]
    a = sc["albedo"][idx[:, 0], idx[:, 1]].astype(np.float64)
    a /= max(np.sqrt((a * a).mean()), 1e-9)
    n = sc["n_mesh"].transpose(1, 2, 0)
    Y = sh_basis_npy(n[idx[:, 0], idx[:, 1]])
    C = sc["sh_irr"][:N].astype(np.float64)
    del sc
    gc.collect()
    return a, Y, C


# ---------------- C1: matvec 正确性 ----------------
def c1_matvec_check(P_list, n_vec=100, seed=7):
    rows = []
    for P in P_list:
        a, Y, C = build_arrays(SCENES[0], P, 3, seed + P)
        bl = fisher_blocks(a, Y, C)
        F = schur_full(bl)
        apply_f, trace, meta = schur_operator(bl)
        rng = np.random.default_rng(seed)
        errs = []
        for _ in range(n_vec):
            v = rng.standard_normal(P)
            exact = F @ v
            approx = apply_f(v)
            denom = max(np.linalg.norm(exact), 1e-300)
            errs.append(float(np.linalg.norm(exact - approx) / denom))
        errs = np.array(errs)
        rows.append(dict(part="C1", P=P, n_vec=n_vec,
                         median_rel_err=float(np.median(errs)),
                         max_rel_err=float(errs.max()),
                         status="PASS" if np.median(errs) < 1e-5 else "FAIL"))
        print(f"  [C1] P={P}: median rel err = {np.median(errs):.2e}, "
              f"max = {errs.max():.2e} -> {rows[-1]['status']}")
        del F, bl, apply_f
        gc.collect()
    return rows


# ---------------- C2: SLQ logdet ----------------
def slq_logdet_norm(apply_f, trace, P, n_probes, m_steps, seed):
    """估计 G = (1/P) Σ_i log(λ̃_i + c), 其中 λ̃=λ/trace, c=SPEC_CUTOFF。

    内部位移: (F/trace + c·I) v = apply_f(v)/trace + c·v
    """
    c = SPEC_CUTOFF
    rng = np.random.default_rng(seed)

    def mv(v):
        return apply_f(v) / trace + c * v

    total = 0.0
    for j in range(n_probes):
        v = rng.choice([-1.0, 1.0], size=P).astype(np.float64)
        v /= np.linalg.norm(v)
        V = np.empty((m_steps, P))
        al = np.empty(m_steps)
        be = np.empty(m_steps - 1)
        V[0] = v
        w = mv(V[0])
        al[0] = V[0] @ w
        breakdown = False
        for k in range(1, m_steps):
            w = w - al[k - 1] * V[k - 1]
            if k > 1:
                w = w - be[k - 2] * V[k - 2]
            # full reorthogonalization
            w -= V[:k].T @ (V[:k] @ w)
            nb = float(np.linalg.norm(w))
            if nb < 1e-10:
                breakdown = True
                break
            be[k - 1] = nb
            V[k] = w / nb
            w = mv(V[k])
            al[k] = V[k] @ w
        m_eff = k if breakdown else m_steps
        if breakdown:
            # 不变子空间提前出现: 用已有部分
            al_used, be_used = al[:m_eff], be[:m_eff - 1]
        else:
            al_used, be_used = al, be
        theta, U = eigh_tridiagonal(al_used, be_used)
        w2 = U[0, :] ** 2
        theta_safe = np.maximum(theta, 1e-300)
        total += float(np.sum(w2 * np.log(theta_safe)))
    return total / n_probes


def c2_slq_rank(P, n_subsets=40, n_probes=10, step_list=(10, 20, 50, 100), seed=11):
    rows = []
    exact_scores, exact_dpos, exact_trace = [], [], []
    slq_scores = {m: [] for m in step_list}
    t_exact = 0.0
    t_slq = {m: 0.0 for m in step_list}

    # 修复 (campaign 复盘): 每个场景只加载/采像素一次, 循环里只换光照子集。
    # 之前的实现误把像素采样种子当子集变化, 导致所有 "子集" 都是同一批灯, 排名无意义。
    K = 32
    N = 3
    rng_sub = np.random.default_rng(seed)
    subsets = []
    seen = set()
    while len(subsets) < n_subsets:
        cand = tuple(sorted(rng_sub.choice(K, N, replace=False).tolist()))
        if cand not in seen:
            seen.add(cand)
            subsets.append(cand)

    per_scene = n_subsets // len(SCENES)
    si = 0
    for scene in SCENES:
        # 固定像素集 (所有光照子集共用同一批像素)
        a, Y, C_all = build_arrays(scene, P, K, seed + 555)  # C_all = 全部 32 灯 SH
        for sub in subsets[si:si + per_scene]:
            C = C_all[list(sub)]
            bl = fisher_blocks(a, Y, C)
            apply_f, trace, meta = schur_operator(bl)

            t0 = time.perf_counter()
            m_exact = spectrum_metrics(schur_full(bl))
            t_exact += time.perf_counter() - t0
            exact_scores.append(m_exact["logdet_pos_norm"])
            exact_dpos.append(m_exact["d_pos"])
            exact_trace.append(trace)

            for m in step_list:
                t0 = time.perf_counter()
                g = slq_logdet_norm(apply_f, trace, P, n_probes, m, seed + si * 31 + m)
                t_slq[m] += time.perf_counter() - t0
                slq_scores[m].append(g)
            del bl, apply_f
            gc.collect()
            si += 1
            if si % 10 == 0:
                print(f"  [C2] {si}/{n_subsets} subsets done", flush=True)
        del a, Y, C_all
        gc.collect()

    exact_arr = np.array(exact_scores)
    dpos_arr = np.array(exact_dpos)
    P_dpos = np.mean(dpos_arr)  # 平均 d+
    logc = np.log(SPEC_CUTOFF)
    for m in step_list:
        raw = np.array(slq_scores[m])
        # 修正版: I_GS ≈ (P/d+)·(G − ((P−d+)/P)·log c)
        corrected = (P / P_dpos) * (raw - ((P - P_dpos) / P) * logc)
        rho_raw = float(spearmanr(raw, exact_arr)[0])
        rho_cor = float(spearmanr(corrected, exact_arr)[0])
        tau = float(kendalltau(corrected, exact_arr)[0])
        status = "PASS" if (rho_cor >= 0.95 and m >= 50) else ("PASS-low" if rho_cor >= 0.95 else "FAIL")
        rows.append(dict(part="C2", P=P, steps=m, probes=n_probes,
                         n_subsets=n_subsets,
                         spearman_raw=round(rho_raw, 4),
                         spearman_corrected=round(rho_cor, 4),
                         kendall_tau=round(tau, 4),
                         t_exact_per_subset=round(t_exact / n_subsets, 3),
                         t_slq_per_subset=round(t_slq[m] / n_subsets, 3),
                         speedup=round((t_exact) / max(t_slq[m], 1e-9), 1),
                         status=status))
        print(f"  [C2] steps={m:3d}: rho_raw={rho_raw:.4f}  rho_corr={rho_cor:.4f}  "
              f"tau={tau:.4f}  speedup={rows[-1]['speedup']}x -> {status}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="快速 smoke")
    ap.add_argument("--c1_pixels", default=None)
    ap.add_argument("--c2_pixels", type=int, default=1000)
    ap.add_argument("--n_subsets", type=int, default=40)
    ap.add_argument("--n_probes", type=int, default=10)
    args = ap.parse_args()

    if args.quick:
        c1_pixels = [300, 800]
        n_vec = 30
        step_list = (10, 20)
        n_subsets = 12
    else:
        c1_pixels = ([int(x) for x in args.c1_pixels.split(",")]
                     if args.c1_pixels else [500, 2000])
        n_vec = 100
        step_list = (10, 20, 50, 100)
        n_subsets = args.n_subsets

    (OUT_DIR / "runtime").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "decision_reports").mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "runtime" / "matrixfree.csv"

    print("=" * 70)
    print("R5-Campaign · Route C Matrix-free")
    print(f"C1 pixels={c1_pixels} ({n_vec} vectors) | "
          f"C2 P={args.c2_pixels} × {n_subsets} subsets, probes={args.n_probes}, steps={step_list}")
    print("=" * 70)

    rows = []
    print("\n--- C1: matvec 正确性 ---")
    rows += c1_matvec_check(c1_pixels, n_vec=n_vec)

    print(f"\n--- C2: SLQ logdet 排名一致性 (P={args.c2_pixels}) ---")
    rows += c2_slq_rank(args.c2_pixels, n_subsets=n_subsets,
                        n_probes=args.n_probes, step_list=step_list)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        # C1 与 C2 行字段不同: 取所有出现过的字段的并集
        fieldnames = []
        for r in rows:
            for k in r.keys():
                if k not in fieldnames:
                    fieldnames.append(k)
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        wr.writerows(rows)

    # ---- 大白话裁决 ----
    c1_pass = all(r["status"] == "PASS" for r in rows if r["part"] == "C1")
    c2_rows = [r for r in rows if r["part"] == "C2"]
    c2_50 = [r for r in c2_rows if r["steps"] >= 50]
    c2_pass = bool(c2_50) and max(r["spearman_corrected"] for r in c2_50) >= 0.95
    c2_tau = max((r["kendall_tau"] for r in c2_50), default=float("nan"))
    best = max(c2_rows, key=lambda r: r["spearman_corrected"]) if c2_rows else None

    lines = []
    lines.append("# Route C Matrix-free · 大白话结论\n\n")
    lines.append(f"- C1 (无矩阵乘法正确性): **{'通过' if c1_pass else '不通过'}** "
                 f"(median 相对误差 < 1e-5 门槛)\n")
    if best:
        lines.append(f"- C2 (快速近似排名): 最好在 **{best['steps']} 步** 达到 "
                     f"rho={best['spearman_corrected']:.4f}, tau={best['kendall_tau']:.4f}, "
                     f"提速 **{best['speedup']}x**\n\n")
    # Stop Rule 2
    tau_100 = [r["kendall_tau"] for r in c2_rows if r["steps"] >= 100]
    if tau_100 and max(tau_100) < 0.85:
        lines.append("**Stop Rule 2 触发**: 100 步 tau < 0.85 -> Route C/D 停止。\n\n")
    if c1_pass and c2_pass:
        lines.append("## 裁决: **GO** — 不存大矩阵也能算出一样的排名\n")
        lines.append("> 结论: GSIQ 可以在低内存下近似计算, 排名保真。这是论文升级的核心筹码: "
                     "`scalable GSIQ` 成立, 进入 C3 (选择保持) 验证。\n")
        lines.append("> 下一步: C3 需要 solver 误差对比 (GPU 或云), 本地无法完成 — 需要小规模算力。\n")
    elif c1_pass and not c2_pass:
        lines.append("## 裁决: **部分 GO** — 乘法正确, 但近似排名不够准\n")
        lines.append("> 结论: 基础设施可用, 但 Lanczos 步数/探针数需要加大后重测; "
                     "若加大后仍 rho<0.95, Route C 停止。\n")
    else:
        lines.append("## 裁决: **FAIL** — 基础实现有问题, 先修再谈\n")
    (OUT_DIR / "decision_reports" / "C_verdict.md").write_text("".join(lines), encoding="utf-8")
    print("\n" + "".join(lines))

    for r in rows:
        rec = dict(scene=",".join(SCENES), N=3, method=f"{r['part']}_steps{r.get('steps','')}",
                   runtime=r.get("t_slq_per_subset", ""), peak_memory="",
                   score=r.get("spearman_corrected", r.get("median_rel_err", "")),
                   rank_score=r.get("kendall_tau", ""), selection_error="",
                   status=r["status"])
        with open(OUT_DIR / "raw_profile" / "campaign.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    print(f"产物: {csv_path}\n      {OUT_DIR / 'decision_reports' / 'C_verdict.md'}")


if __name__ == "__main__":
    main()
