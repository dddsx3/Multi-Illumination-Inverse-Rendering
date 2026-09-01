"""R5 Compute-Aware Campaign · Phase 1 / Route A0 · 基线瓶颈画像

回答一个问题: GSIQ 计算时间到底花在哪?
  T1 = Fisher construction (fisher_blocks)
  T2 = Schur complement    (schur_full)
  T3 = Eigen decomposition (spectrum_metrics = P×P eigh)
  T4 = 峰值内存 (peak RAM)
  T5 = 显存 (本路径不用 GPU, 记 0)

GO 判定 (任务书 Phase 1):
  Case A: eigh 占 > 70% 总时间  -> 进入 Matrix-free (Route C)
  Case B: P×P 内存带宽操作占 > 70% -> 进入 Pixel Coreset (Route B)
  Case C: 进程内存泄漏 (连续调用 RSS 持续增长 > 20%) -> 工程修复 (Route A)
  三个都不是 -> 不做复杂优化, 直接结论 "调用次数是瓶颈"

用法 (本地 Windows 直接跑, 无需 GPU):
  python r5_ca_01_baseline_profile.py                        # 默认 P={2000,3000,5000}, N={3,5,8}
  python r5_ca_01_baseline_profile.py --pixels 500,1000      # 快速版
输出:
  r5_compute_audit/raw_profile/baseline_profile.csv
  r5_compute_audit/decision_reports/A0_verdict.md            <- 大白话结论
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "p1" / "source" / "information_audit"))

from gauge_fisher_v2 import (  # noqa: E402
    fisher_blocks, schur_full, spectrum_metrics, diag_proxy_metrics,
    structural_null_gate, load_scene, sh_basis_npy,
)

SCENE = "conf_sphere_r05"
DATA_ROOT = REPO / "p1" / "calibration_set" / "data_sun_confirmatory"
OUT_DIR = REPO / "r5_compute_audit"
SEED = 20260901


def peak_rss_mb(stop_event, proc, out):
    peak = 0.0
    while not stop_event.is_set():
        try:
            peak = max(peak, proc.memory_info().rss / 1e6)
        except Exception:
            pass
        time.sleep(0.02)
    out.append(peak)


def build_arrays(P, N, rng):
    """真实 scene 的 albedo/normal + 随机取 P 像素 + 前 N 盏灯的 SH。"""
    sc = load_scene(str(DATA_ROOT / SCENE))
    idx_all = np.argwhere(sc["mask"])
    take = min(P, len(idx_all))
    sel = rng.choice(len(idx_all), take, replace=False)
    idx = idx_all[sel]
    a = sc["albedo"][idx[:, 0], idx[:, 1]].astype(np.float64)
    rms = np.sqrt((a * a).mean())
    a = a / max(rms, 1e-9)
    n = sc["n_mesh"].transpose(1, 2, 0)
    Y = sh_basis_npy(n[idx[:, 0], idx[:, 1]])
    C = sc["sh_irr"][:N].astype(np.float64)
    del sc
    gc.collect()
    return a, Y, C


def stage_timed(a, Y, C, reps):
    """分阶段计时: 返回 dict(T1,T2,T2b,T3, total, per-call 派生)。"""
    proc = psutil.Process()
    res = {}
    # warm-up 1 次 (排除首次分配)
    bl = fisher_blocks(a, Y, C)
    F = schur_full(bl)
    m = spectrum_metrics(F)
    del bl, F, m
    gc.collect()

    base = proc.memory_info().rss / 1e6
    stop = threading.Event()
    peak_out: list[float] = []
    th = threading.Thread(target=peak_rss_mb, args=(stop, proc, peak_out), daemon=True)
    th.start()

    t0 = time.perf_counter()
    for _ in range(reps):
        bl = fisher_blocks(a, Y, C)
    t1 = time.perf_counter()

    for _ in range(reps):
        F = schur_full(bl)
    t2 = time.perf_counter()

    # offdiag 扫描 (与 ga_isi_v2_scores 内部一致的分块 max)
    P = bl["P"]
    for _ in range(reps):
        off = 0.0
        for r0 in range(0, P, 256):
            r1 = min(r0 + 256, P)
            blk = F[r0:r1].copy()
            blk[np.arange(r0, r1) - r0, np.arange(r0, r1)] = 0.0
            if blk.size:
                off = max(off, float(np.abs(blk).max()))
            del blk
    t2b = time.perf_counter()

    for _ in range(reps):
        m = spectrum_metrics(F)
    t3 = time.perf_counter()

    stop.set()
    th.join(timeout=1.0)

    res["T1_fisher"] = (t1 - t0) / reps
    res["T2_schur"] = (t2 - t1) / reps
    res["T2b_offdiag"] = (t2b - t2) / reps
    res["T3_eigh"] = (t3 - t2b) / reps
    res["T_total_stages"] = (t3 - t0) / reps
    res["peak_rss_mb"] = max(peak_out[0] if peak_out else 0.0, proc.memory_info().rss / 1e6) - base
    res["d_pos"] = m["d_pos"]
    res["gsiq"] = m["logdet_pos_norm"]
    del F, bl, m
    gc.collect()
    return res


def main():
    global SCENE
    ap = argparse.ArgumentParser()
    ap.add_argument("--pixels", default="2000,3000,5000",
                    help="逗号分隔的像素数列表; 5000 以上每个配置只跑 1 次")
    ap.add_argument("--ns", default="3,5,8")
    ap.add_argument("--scene", default=SCENE)
    args = ap.parse_args()

    pixels = [int(x) for x in args.pixels.split(",")]
    ns = [int(x) for x in args.ns.split(",")]
    SCENE = args.scene

    (OUT_DIR / "raw_profile").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "decision_reports").mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "raw_profile" / "baseline_profile.csv"
    jsonl_path = OUT_DIR / "raw_profile" / "baseline_profile.jsonl"

    print("=" * 70)
    print("R5-Campaign · A0 基线画像")
    print(f"scene={SCENE}  pixels={pixels}  N={ns}")
    print("=" * 70)

    rows = []
    rng = np.random.default_rng(SEED)
    for P in pixels:
        for N in ns:
            reps = 3 if P <= 2000 else (2 if P <= 3000 else 1)
            try:
                a, Y, C = build_arrays(P, N, np.random.default_rng(SEED + P * 100 + N))
                r = stage_timed(a, Y, C, reps)
            except MemoryError:
                print(f"  P={P} N={N}: 内存不足 (OOM), 跳过")
                r = dict(T1_fisher=np.nan, T2_schur=np.nan, T2b_offdiag=np.nan,
                         T3_eigh=np.nan, T_total_stages=np.nan, peak_rss_mb=np.nan,
                         d_pos=-1, gsiq=np.nan)
            tot = r["T_total_stages"]
            frac_eigh = r["T3_eigh"] / tot if tot and tot > 0 else np.nan
            frac_pp = (r["T2_schur"] + r["T2b_offdiag"]) / tot if tot and tot > 0 else np.nan
            row = dict(scene=SCENE, P=P, N=N, reps=reps,
                       T1_fisher_s=round(r["T1_fisher"], 4),
                       T2_schur_s=round(r["T2_schur"], 4),
                       T2b_offdiag_s=round(r["T2b_offdiag"], 4),
                       T3_eigh_s=round(r["T3_eigh"], 4),
                       T_total_s=round(tot, 4) if tot == tot else "OOM",
                       frac_eigh=round(frac_eigh, 3) if frac_eigh == frac_eigh else "OOM",
                       frac_PxP_ops=round(frac_pp, 3) if frac_pp == frac_pp else "OOM",
                       peak_rss_mb=round(r["peak_rss_mb"], 1),
                       d_pos=r["d_pos"], gsiq=round(r["gsiq"], 4) if r["gsiq"] == r["gsiq"] else "OOM")
            rows.append(row)
            print(f"  P={P:5d} N={N}: total={row['T_total_s']}s  "
                  f"eigh={row['frac_eigh']}  P×P ops={row['frac_PxP_ops']}  "
                  f"peakRAM={row['peak_rss_mb']}MB")
            # 统一审计 JSON (任务书 Phase 0 schema)
            rec = dict(scene=SCENE, N=N, method="dense_gsiq",
                       runtime=tot, peak_memory=row["peak_rss_mb"],
                       score=row["gsiq"], rank_score="", selection_error="",
                       status="ok" if tot == tot else "oom")
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            del a, Y, C
            gc.collect()

    # ---- 写 CSV ----
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        wr.writerows(rows)

    # ---- 泄漏检测 (Case C) ----
    print("\n[泄漏检测] 连续 5 次调用看内存是否持续增长 ...")
    a, Y, C = build_arrays(1500, 5, np.random.default_rng(SEED))
    proc = psutil.Process()
    rss = []
    for i in range(5):
        bl = fisher_blocks(a, Y, C)
        F = schur_full(bl)
        m = spectrum_metrics(F)
        gc.collect()
        rss.append(proc.memory_info().rss / 1e6)
        del bl, F, m
    growth = (rss[-1] - rss[0]) / max(rss[0], 1e-9)
    leak = growth > 0.20
    print(f"  RSS 序列: {[round(x, 1) for x in rss]} MB, 增长 {growth*100:.1f}% -> "
          f"{'有泄漏迹象' if leak else '无泄漏'}")

    # ---- 大白话裁决 ----
    ok_rows = [r for r in rows if r["T_total_s"] != "OOM"]
    med_frac_eigh = float(np.median([r["frac_eigh"] for r in ok_rows])) if ok_rows else float("nan")
    med_frac_pp = float(np.median([r["frac_PxP_ops"] for r in ok_rows])) if ok_rows else float("nan")

    if leak:
        case = "C"
    elif med_frac_eigh > 0.70:
        case = "A"
    elif med_frac_pp > 0.70:
        case = "B"
    else:
        case = "NONE"

    verdict = []
    verdict.append("# A0 基线画像 · 大白话结论\n")
    verdict.append(f"- 场景: {SCENE} · 像素档位: {pixels} · 灯数: {ns}\n")
    verdict.append(f"- 特征分解(eigh)平均占总时间: **{med_frac_eigh*100:.0f}%**\n")
    verdict.append(f"- P×P 矩阵操作(Schur+扫描)平均占总时间: **{med_frac_pp*100:.0f}%**\n")
    verdict.append(f"- 内存泄漏检测: {'**有**' if leak else '无'}\n\n")
    if case == "A":
        verdict.append("## 裁决: Case A -> **进入 Route C (Matrix-free)**\n")
        verdict.append("> 时间主要花在特征分解上, 不存大矩阵、不做完整分解的算法有最大收益。\n")
    elif case == "B":
        verdict.append("## 裁决: Case B -> **进入 Route B (Pixel Coreset)**\n")
        verdict.append("> 时间主要花在大矩阵的读写搬运上, 减少像素数有最大收益。\n")
    elif case == "C":
        verdict.append("## 裁决: Case C -> **先做工程修复 (Route A)**\n")
        verdict.append("> 进程内存持续增长, 先修泄漏再谈优化。\n")
    else:
        verdict.append("## 裁决: 无单一瓶颈 -> **不做复杂优化**\n")
        verdict.append("> 时间分散在多个环节。真正的瓶颈是\"要算很多次\", "
                       "降像素(Route B)依然是最直接的省钱手段, 但不属于本门槛触发的优化。\n")
    # 附加速倍率提示
    if ok_rows:
        r2k = [r for r in ok_rows if r["P"] == 2000 and r["N"] == 5]
        if r2k:
            t = r2k[0]["T_total_s"]
            verdict.append(f"\n参考: P=2000, N=5 单次全流程 ≈ **{t}s**; "
                           f"论文全量 11.9 万次调用 ≈ **{t*119040/3600:.0f} 小时**(单核)\n")
    (OUT_DIR / "decision_reports" / "A0_verdict.md").write_text("".join(verdict), encoding="utf-8")
    print("\n" + "".join(verdict))
    print(f"产物: {csv_path}\n      {OUT_DIR / 'decision_reports' / 'A0_verdict.md'}")


if __name__ == "__main__":
    main()
