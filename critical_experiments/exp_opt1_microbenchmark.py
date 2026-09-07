#!/usr/bin/env python3
"""OPT-1 阶段 0 · 微基准: 现行 exp8R v3 joint_trf 单子集计时分解(不修改现行实现)。

目的: 量化现行 scipy-trf 联合估计器在单个 N=3 子集上的时间分解, 为 VarPro
(变量投影)加速路线提供对照基线。

分解段(全部 time.perf_counter):
  1. data_load   : load_object(ballPNG) 96 张 PNG 读入 + 归一化(单次, 主循环外)
  2. calibrate   : 全 96 光朗伯拟合 + 噪声 (a,b) 标定(单次)
  3. subset_prep : 像素守卫 + 下采样(单次)
  4. als_init    : SH-2 ALS 粗解(单次, 三起点共享)
  5. trf_x1      : scipy least_squares(trf, 稀疏 J + LSMR) 单起点 1 次
  6. trf_x3      : 3 起点全跑(现行协议实际成本)
  7. diag        : diagnose_trace 1 次
  8. perturb     : 起点扰动生成(1 次, 可忽略)

采样: ballPNG, N=3, 第 1 个子集(SEED=20260906 完全复刻现行抽样代码路径:
  r2 = default_rng(SEED + crc32(name)%1000), r2.choice(96,3) 首个;
  下采样 rng3 = default_rng(SEED), choice(idx_k, min(len, max(len//8,2000)))。
  —— 该子集与现行成功运行 exp8r_v3_results/ 的 cfg00 一致可对账)。

注记(预声明): 本机有 python 后台进程 exp12v4 全量在跑(单核 ~95%),
  32 逻辑核; 计时受并行负载干扰, 所有数字为【上界】(真实无负载值应更低)。
  不杀后台进程(纪律)。

不修改任何现有文件; 产物: critical_experiments/exp_opt1_microbenchmark.json
"""
from __future__ import annotations

import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import exp8r_diligent_discrimination_v3 as m  # noqa: E402  现行实现, 只读 import

OUT = HERE / "exp_opt1_microbenchmark.json"
SEED = 20260906
OBJ = "ballPNG"
REP = 3  # trf 单起点重复次数(取中位数, 减少负载抖动)


def main():
    t_wall0 = time.perf_counter()

    # ---- 段 1: 数据加载 ----
    t0 = time.perf_counter()
    d = m.ROOT / OBJ
    dirs, n_gt, I_norm, mask = m.load_object(d)
    t_load = time.perf_counter() - t0

    # ---- 段 2: 标定 ----
    t0 = time.perf_counter()
    rho_full, a_, b_, lambert_resid = m.calibrate(n_gt, dirs, I_norm)
    t_calib = time.perf_counter() - t0

    # ---- 段 3: 子集准备(复刻现行抽样路径, cfg00) ----
    t0 = time.perf_counter()
    r2 = np.random.default_rng(SEED + zlib.crc32(OBJ.encode()) % 1000)
    sel = r2.choice(96, 3, replace=False)
    dirs_sub = dirs[sel]; I_sub = I_norm[sel]
    n_masked = (n_gt @ dirs_sub.T > 0).astype(float)
    keep = m.pixel_guard(I_sub, n_masked)
    idx_k = np.where(keep)[0]
    rng3 = np.random.default_rng(SEED)
    sub_idx = np.sort(rng3.choice(idx_k, min(len(idx_k), max(len(idx_k)//8, 2000)),
                                  replace=False))
    n_k = n_gt[sub_idx]; I_k = I_sub[:, sub_idx]
    t_prep = time.perf_counter() - t0
    P_k = len(sub_idx)

    # ---- 段 4: ALS 初值 ----
    t0 = time.perf_counter()
    rho_als, dirs_als = m.als_init(I_k, rho_full[sub_idx] * 0.8, dirs_sub, n_k)
    t_als = time.perf_counter() - t0

    # ---- 段 5: 起点扰动(可忽略, 仍计时) ----
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)   # 与现行一致: 主 rng 顺序不可复刻(主循环内被
    d0s = [dirs_als]                     # 多子集消耗), 这里只量扰动生成本身的成本量级
    for s in range(2):
        d0 = dirs_als + rng.normal(0, 0.05, dirs_als.shape)
        d0 /= np.linalg.norm(d0, axis=1, keepdims=True)
        d0s.append(d0)
    t_perturb = time.perf_counter() - t0

    # ---- 段 6: trf 单起点(REP 次, 中位数) ----
    costs = []
    t_trf = []
    res0 = None
    for rep in range(REP):
        t0 = time.perf_counter()
        res0 = m.joint_trf(I_k, rho_als, d0s[0], n_k, a_, b_)
        t_trf.append(time.perf_counter() - t0)
        costs.append(res0.cost)
    t_trf_x1_med = float(np.median(t_trf))
    nfev = res0.nfev; njev = getattr(res0, "njev", None)

    # ---- 段 7: 3 起点全跑(现行协议单子集总成本, 1 次) ----
    t0 = time.perf_counter()
    best = None
    for d0 in d0s:
        res = m.joint_trf(I_k, rho_als, d0, n_k, a_, b_)
        if best is None or res.cost < best.cost:
            best = res
    t_trf_x3 = time.perf_counter() - t0

    # ---- 段 8: Jacobian 单次组装成本(诊断: LSMR 路径每 njev 次的成本单元) ----
    t_jac = []
    for _ in range(5):
        t0 = time.perf_counter()
        m.jac_r_joint_sparse(best.x, P_k, n_k)
        t_jac.append(time.perf_counter() - t0)
    t_jac_med = float(np.median(t_jac))

    # ---- 段 9: 诊断求值 ----
    P_kk = len(sub_idx)
    parms = best.x[P_kk:].reshape(3, 3)
    alpha_e = parms[:, 0]
    xy = parms[:, 1:]
    zz = np.sqrt(np.maximum(1 - xy[:, 0]**2 - xy[:, 1]**2, 1e-12))
    dirs_e = np.column_stack([xy, zz])
    t0 = time.perf_counter()
    tr = m.diagnose_trace(n_k, I_k, best.x[:P_kk], dirs_sub, alpha_e, a_, b_)
    t_diag = time.perf_counter() - t0

    lae_v = m.lae(dirs_e, dirs_sub)
    t_wall = time.perf_counter() - t_wall0

    segs = [
        ("data_load(ballPNG 96 PNG + 归一化)", t_load),
        ("calibrate(全 96 光拟合+噪声)", t_calib),
        ("subset_prep(守卫+下采样 cfg00)", t_prep),
        ("als_init(50 次 ALS 粗解)", t_als),
        ("perturb(2 个扰动起点)", t_perturb),
        (f"trf_x1(单起点, {REP} 次中位数)", t_trf_x1_med),
        ("trf_x3(现行协议 3 起点实际)", t_trf_x3),
        (f"sparse_jac_assembly(单次, 5 次中位数)", t_jac_med),
        ("diagnose_trace(单次)", t_diag),
    ]
    total_est = sum(v for _, v in segs)
    rows = []
    print(f"{'段':<44}{'耗时(s)':>12}{'占比%':>8}")
    for name, v in segs:
        frac = 100 * v / total_est
        rows.append(dict(segment=name, seconds=round(v, 6), pct=round(frac, 2)))
        print(f"{name:<44}{v:>12.4f}{frac:>8.2f}")
    print(f"{'合计(分段和)':<44}{total_est:>12.4f}{100.0:>8.2f}")

    out = dict(
        meta=dict(
            purpose="OPT-1 阶段 0 微基准: 现行 joint_trf 单子集计时分解",
            object=OBJ, N=3, subset="cfg00(SEED=20260906 复刻现行抽样)",
            P_subsampled=P_k, pixels_total=int(mask.sum()),
            seed=SEED, rep_median=f"trf_x1 取 {REP} 次中位数",
            python="3.14.2", numpy=np.__version__, scipy=__import__("scipy").__version__,
            cores=32,
            load_note=("机器有 python 后台进程 exp12v4 全量在跑(单核 ~95%), "
                       "计时受并行负载干扰, 所有数字为上界(真实无负载值应更低)"),
        ),
        trf_detail=dict(nfev=nfev, njev=njev, cost_x1_median=float(np.median(costs)),
                        cost_3start_best=float(best.cost), lae_deg=lae_v,
                        trace=tr, jac_time_share_est_pct=round(
                            100 * (t_jac_med * (njev or 0)) / t_trf_x1_med, 2)),
        segments=rows,
        total_seconds=round(total_est, 6),
        wall_seconds=round(t_wall, 6),
    )
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[exp_opt1_microbenchmark] 落盘 -> {OUT}")
    print(f"trf nfev={nfev} njev={njev} cost={best.cost:.6e} LAE={lae_v:.3f}°")
    print(f"单起点中位耗时 {t_trf_x1_med:.3f}s | 3 起点 {t_trf_x3:.3f}s | "
          f"Jac 单次 {t_jac_med*1e3:.2f}ms (估 LSMR+J 组装占 trf 单起点的 "
          f"{out['trf_detail']['jac_time_share_est_pct']}%)")


if __name__ == "__main__":
    main()
