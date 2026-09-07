#!/usr/bin/env python3
"""卡 X · exp9d: 破缺第三通道定位 — 附着阴影终止子实验(v3.3 任务书 §3)

背景(SH-4 降幅 <4% 推翻"SH-2 截断主源"归因; 挖轮廓带只解释 λ 的 45%):
  候选第三通道 = 附着阴影终止子非线性(GBR 变换移动 terminator, |n·l|≈0 带
  的一阶阴影状态翻转 h: 0↔1)。

方法(与 exp9c part2 同配置: 球体 + 标准核方向光):
  1. 对 μ/ν 生成元算逐像素一阶图像变化 |∂I/∂μ|、|∂I/∂ν| 的空间分布;
  2. 按"到最近 terminator 带的距离"分桶(存在某光 k 使 |n·l_k| < 阈值带的
     像素集合; 距离 = 有效像素网格上的 BFS 距离);
  3. 对照桶 = 轮廓带、内部远带区。

判据(预写死, 不得事后放宽):
  破缺能量若集中于 terminator 桶(富集 ≥5×: 该桶均值/全局中位 ≥5)→
  附着阴影通道坐实, 判别(c)改写;
  不集中 → 第三通道候选表(离散 Sobel 微分、极点参数化)如实列出, 不强行归因。

产物: critical_experiments/exp9d_terminator_localization.{py,json}
"""
from __future__ import annotations
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from exp9_sphere_gbr_verdict import (  # noqa: E402
    SEED, K_STD, delta_fields, delta_observation, sh2,
)
from exp9c_stratification_closure import (  # noqa: E402
    make_sphere_sc, contour_mask_valid,
)

OUT = HERE / "exp9d_terminator_localization.json"
N_LIGHTS = 5
RES = 48
TERM_EPS = 0.02          # 逐光 |Z_k| < ε·max|Z_k| 窄翻转带(初版 0.05 归一口径 87% 覆盖系 bug, 修正留痕)
ENRICH_THRESHOLD = 5.0   # 预写死: 富集 ≥5× 判坐实
BFS_MAX = 24             # 距离分桶上限(网格步)


def scene_and_generators():
    sc = make_sphere_sc(res=RES, kernel=K_STD)       # 固定 SEED 内部方向
    gens = {}
    for g in ("lambda", "mu", "nu"):
        dfd = delta_fields(sc, g)
        dI, _ = delta_observation(sc, dfd, mode="opt")   # C 最优补偿口径(与 Rayleigh 主口径一致)
        gens[g] = np.abs(dI).max(1)                     # 逐像素: |δI| 跨光最大(一阶图像变化)
    return sc, gens


def terminator_mask_and_dist(sc):
    """terminator 带 = 有效像素中 ∃k: |n·l_k| < TERM_EPS; 距离 = BFS(带内=0)。"""
    n = sc["n"]; C = sc["C"]; valid = sc["valid"]
    H, W = sc["H"], sc["W"]
    # 光方向从 C 反解: 标准 clamped-cosine 核下 C_k = kernel * Y(l_k) → l = 前几项比
    # 直接用 n·C_k 的 SH 内积近似: |n·l| ≈ |Y(n)·Y(l)|/|Y(l)|² = |Z_k|/max(Z_k) 归一
    # 更直接: l_k = (C_k[1:4]/(2π/3·kernel 首项缩放))——标准核 Y(0,0)=π, Y(1,·)=2π/3
    # 简化且无歧义: 用 Z_k = Y(n)·C_k 的符号带边界 = |Z| < eps·max|Z| (逐光归一)
    Y = sh2(n)
    Z = Y @ C.T                                       # (P,N)
    # 修正(留痕): 初版 |Zn|<0.05 归一化"或"语义 → 87% 覆盖(桶无意义, bug);
    # 改窄翻转带 = 逐光 |Z_k| < 0.02·max|Z_k| 的并集(43%, 一阶 h 翻转带)
    maxZ = np.abs(Z).max(0)
    term = np.zeros_like(valid)
    for k in range(Z.shape[1]):
        term |= valid & (np.abs(Z[:, k]) < 0.02 * maxZ[k])
    # BFS 距离(有效像素网格)
    dist = np.full(H * W, np.inf)
    dist[term & valid] = 0
    q = deque(np.where(term & valid)[0].tolist())
    vm = valid
    while q:
        p = q.popleft()
        r, c = divmod(p, W)
        d1 = dist[p] + 1
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            r2, c2 = r + dr, c + dc
            if 0 <= r2 < H and 0 <= c2 < W:
                p2 = r2 * W + c2
                if vm[p2] and dist[p2] > d1:
                    dist[p2] = d1
                    if d1 <= BFS_MAX:
                        q.append(p2)
    return term, dist


def main():
    sc, gens = scene_and_generators()
    term, dist = terminator_mask_and_dist(sc)
    contour = contour_mask_valid(sc)
    valid = sc["valid"]

    out = dict(meta=dict(res=RES, n_lights=N_LIGHTS, term_eps=TERM_EPS,
                         enrich_threshold=ENRICH_THRESHOLD,
                         distance_metric="BFS on valid-pixel grid",
                         diag_mode="C 最优补偿口径(与 Rayleigh 主口径一致)"))
    # 桶定义
    d = dist.copy()
    d[~np.isfinite(d)] = 1e9
    buckets = {
        "terminator_band(0)": term,
        "d=1": valid & (d == 1),
        "d=2-3": valid & (d >= 2) & (d <= 3),
        "d=4-7": valid & (d >= 4) & (d <= 7),
        "d=8-15": valid & (d >= 8) & (d <= 15),
        "far(>15)": valid & (d > 15),
        "contour_band(对照)": contour & ~term,
        "interior_far(对照)": valid & ~contour & (d > 7),
    }
    res_rows = {}
    verdicts = {}
    for g, energy in gens.items():
        med = float(np.median(energy[valid]))          # 全局中位(非选择性)
        rows = {}
        for bname, bm in buckets.items():
            if bm.sum() < 5:
                rows[bname] = dict(n=int(bm.sum()), mean=None)
                continue
            m = float(energy[bm].mean())
            rows[bname] = dict(n=int(bm.sum()), mean=m, ratio_to_median=m / max(med, 1e-300))
        res_rows[g] = dict(global_median=med, buckets=rows)
        # 富集判读: terminator 桶均值/全局中位
        tb = rows["terminator_band(0)"]
        enrich = tb["ratio_to_median"] if tb["mean"] is not None else float('nan')
        verdicts[g] = dict(terminator_enrichment=enrich,
                          passes=bool(enrich >= ENRICH_THRESHOLD))
        print(f"[{g}] 全局中位={med:.3e} | terminator 桶均值={tb['mean']:.3e} "
              f"富集={enrich:.1f}× {'≥5× 坐实' if enrich >= ENRICH_THRESHOLD else '<5× 不集中'}")

    # 对照桶核验: 轮廓带与内部远带的富集(供归因对照)
    for g in gens:
        cb = res_rows[g]["buckets"]["contour_band(对照)"]
        ib = res_rows[g]["buckets"]["interior_far(对照)"]
        print(f"  [{g}] 轮廓带(非term) 富集={cb['ratio_to_median'] if cb['mean'] is not None else float('nan'):.1f}× "
              f"内部远带 富集={ib['ratio_to_median'] if ib['mean'] is not None else float('nan'):.1f}×")

    n_pass = sum(1 for v in verdicts.values() if v["passes"])
    overall = ("附着阴影终止子通道坐实(μ/ν 均富集 ≥5×) → 判别(c) 改写:"
               "'SH 截断贡献 <4%, 主源 = 附着阴影非线性(物理性) + λ 的轮廓带泄漏'"
               if n_pass == len(verdicts) else
               f"部分坐实({n_pass}/{len(verdicts)} 生成元过富集)——按过/未过逐生成元如实分层报告"
               if n_pass > 0 else
               "不集中 → 不强行归因; 第三通道候选表(离散 Sobel 微分、极点参数化)如实列出")
    out.update(generators=res_rows, verdicts=verdicts,
               overall=dict(n_pass=n_pass, n_total=len(verdicts), conclusion=overall))
    print(f"\n[exp9d] 判定: {n_pass}/{len(verdicts)} 生成元富集 ≥{ENRICH_THRESHOLD}×")
    print(f"  → {overall}")
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[exp9d] 落盘 -> {OUT}")


if __name__ == "__main__":
    main()