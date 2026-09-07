#!/usr/bin/env python3
"""OPT-1 阶段 1 · 产物 2: VarPro 烟测(ballPNG 真实数据, 单子集对照)。

对照协议(与阶段 0 微基准完全同数据同起点, 可对账):
  - 数据: ballPNG, N=3, cfg00(SEED=20260906 复刻现行抽样代码路径);
  - 子集: 像素守卫 + 下采样 P=2000(固定种子, 非选择性);
  - 初值: als_init 粗解(rho_als, dirs_als)——两法共享, 单起点;
  - A = 现行 m.joint_trf(scipy trf, 解析稀疏 J + LSMR);
  - B = varpro_estimate(本阶段新估计器, weighted=False → 目标与 trf 严格一致);
  - 记录: final cost / LAE(m.lae)/ 耗时(各 REP=3 取中位, 减负载抖动)。

计时环境(如实记录): 机器【非】空闲——后台 python PID 8260
(exp8r_diligent_discrimination_v31.py, ~1 核)在跑; 32 逻辑核, 单核负载对
单线程 numpy 影响小, 但所有时间为上界(与阶段 0 同口径)。

等价性附加诊断(为阶段 2 风险清单收集证据, 不设阈值):
  - gauge 不变量: 乘积矩阵 ρ_p·α_k 的中位比值(trf 与 VarPro 的 ρ/α 各自
    不可 raw 对照, 尺度规范 cρ·α/c);
  - 逐光方向夹角(trf vs VarPro 估计方向互夹);
  - 交叉 cost: 用 VarPro 的 x 放回 trf 口径残差算 cost, 与 scipy res.cost 同式
    ½Σr²(两者 cost 语义一致, 可直接比);
  - VarPro 外层迭代数 / 剖面求值数 / 末次 ALS 步数(成本结构)。

不修改任何现有文件; 产物: critical_experiments/exp8r_varpro_smoke.json
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

import exp8r_diligent_discrimination_v3 as m          # noqa: E402  现行实现, 只读
from exp8r_varpro_estimator import varpro_estimate     # noqa: E402  本阶段新估计器

OUT = HERE / "exp8r_varpro_smoke.json"
SEED = 20260906
OBJ = "ballPNG"
REP = 3  # 计时重复次数(取中位; 单起点语义不变, 每次同起点重跑)


def unpack_trf_layout(x, P_k, N):
    """按 joint_trf 的 x 布局解析(与 exp8r v3 main() 完全一致)。"""
    rho_e = x[:P_k]
    parms = x[P_k:].reshape(N, 3)
    alpha_e = parms[:, 0]
    xy = parms[:, 1:]
    zz = np.sqrt(np.maximum(1 - xy[:, 0] ** 2 - xy[:, 1] ** 2, 1e-12))
    dirs_e = np.column_stack([xy, zz])
    return rho_e, alpha_e, dirs_e


def cost_trf_semantics(I_sub, n_gt, rho, alphas, dirs):
    """trf 口径 cost(与 scipy res.cost 同式): ½Σ I_obs−ρ·α·clip(n·l)²。"""
    nl = np.clip(n_gt @ dirs.T, 0, None)
    r = I_sub.T - rho[:, None] * alphas[None, :] * nl
    return 0.5 * float((r ** 2).sum())


def main():
    t_wall0 = time.perf_counter()

    # ---- 数据 + 子集(复刻现行抽样路径 cfg00, 与微基准一致) ----
    d = m.ROOT / OBJ
    dirs, n_gt, I_norm, mask = m.load_object(d)
    rho_full, a_, b_, lambert_resid = m.calibrate(n_gt, dirs, I_norm)

    r2 = np.random.default_rng(SEED + zlib.crc32(OBJ.encode()) % 1000)
    sel = r2.choice(96, 3, replace=False)
    dirs_sub = dirs[sel]; I_sub = I_norm[sel]
    n_masked = (n_gt @ dirs_sub.T > 0).astype(float)
    keep = m.pixel_guard(I_sub, n_masked)
    idx_k = np.where(keep)[0]
    rng3 = np.random.default_rng(SEED)
    sub_idx = np.sort(rng3.choice(idx_k, min(len(idx_k), max(len(idx_k) // 8, 2000)),
                                  replace=False))
    n_k = n_gt[sub_idx]; I_k = I_sub[:, sub_idx]
    P_k = len(sub_idx)

    # 阴影边界统计(h=0 像素-光对占比, 阶段 2 风险点证据)
    nl_chk = n_k @ dirs_sub.T
    frac_shadow = float((nl_chk <= 0).mean())

    # ---- ALS 初值(两法共享, 单起点) ----
    rho_als, dirs_als = m.als_init(I_k, rho_full[sub_idx] * 0.8, dirs_sub, n_k)

    print(f"[smoke] {OBJ} cfg00: sel={sel.tolist()}, P_k={P_k}, a_={a_:.3g}, b_={b_:.3g}")
    print(f"[smoke] 阴影对(h≤0)占比 {frac_shadow:.3f}; 起点 LAE(als_init vs 真值)="
          f"{m.lae(dirs_als, dirs_sub):.3f}°")

    # ---- A: 现行 trf 单起点(REP 次中位) ----
    t_trf, costs_trf = [], []
    res_trf = None
    for _ in range(REP):
        t0 = time.perf_counter()
        res_trf = m.joint_trf(I_k, rho_als, dirs_als, n_k, a_, b_)
        t_trf.append(time.perf_counter() - t0)
        costs_trf.append(res_trf.cost)
    rho_t, al_t, dirs_t = unpack_trf_layout(res_trf.x, P_k, 3)
    lae_trf = m.lae(dirs_t, dirs_sub)
    t_trf_med = float(np.median(t_trf))

    # ---- B: VarPro 单起点(同起点同数据, weighted=False 与 trf 同目标) ----
    t_vp, costs_vp = [], []
    res_vp = None
    for _ in range(REP):
        t0 = time.perf_counter()
        res_vp = varpro_estimate(I_k, rho_als, dirs_als, n_k, a_, b_,
                                 n_iters_outer=50)
        t_vp.append(time.perf_counter() - t0)
        costs_vp.append(res_vp.cost)
    rho_v, al_v, dirs_v = unpack_trf_layout(res_vp.x, P_k, 3)   # 布局一致性检验
    lae_vp = m.lae(dirs_v, dirs_sub)
    t_vp_med = float(np.median(t_vp))

    # ---- 交叉验证 ----
    # 1) VarPro 的 x 放回 trf 口径 cost(应与 res_vp.cost 一致到机器精度)
    cost_vp_x_trf = cost_trf_semantics(I_k, n_k, rho_v, al_v, dirs_v)
    # 2) trf 的解放回 VarPro 剖面 cost
    cost_trf_x_vp = cost_trf_semantics(I_k, n_k, rho_t, al_t, dirs_t)
    # 3) gauge 不变量: 乘积矩阵中位比
    prod_t = rho_t[:, None] * al_t[None, :]
    prod_v = rho_v[:, None] * al_v[None, :]
    rel_prod = float(np.median(prod_v / np.maximum(prod_t, 1e-300)))
    #    ρ 块与 α 块的偏移比(预期近似互倒 —— 规范现象)
    rho_ratio = float(np.median(rho_v / np.maximum(rho_t, 1e-300)))
    alpha_ratio = float(np.median(al_v / np.maximum(al_t, 1e-300)))
    # 4) 逐光方向互夹(trf 解 vs VarPro 解)
    angs_tv = [float(np.degrees(np.arccos(np.clip(dirs_t[k] @ dirs_v[k], -1, 1))))
               for k in range(3)]

    rel_cost_gap = abs(costs_trf[0] - costs_vp[0]) / max(costs_trf[0], 1e-300)

    # ---- 打印对照表 ----
    print()
    print("=" * 76)
    print(f"{'':<24}{'trf(现行)':>18}{'VarPro(新)':>18}")
    print("=" * 76)
    print(f"{'final cost(½Σr²)':<24}{res_trf.cost:>18.6f}{res_vp.cost:>18.6f}")
    print(f"{'LAE(deg)':<24}{lae_trf:>18.4f}{lae_vp:>18.4f}")
    print(f"{'耗时(s, 中位x' + str(REP) + ')':<24}{t_trf_med:>18.4f}{t_vp_med:>18.4f}")
    print(f"{'加速比(VarPro/trf)':<24}{t_trf_med / t_vp_med:>17.2f}x")
    print("-" * 76)
    print(f"trf  nfev={res_trf.nfev} njev={getattr(res_trf, 'njev', None)}")
    print(f"VarPro 外层={res_vp.n_outer} 剖面求值={res_vp.n_fev} "
          f"末次ALS步={res_vp.als_steps_last} 收敛={res_vp.converged} ({res_vp.message})")
    print("-" * 76)
    print(f"交叉: cost(VarPro.x @ trf口径)={cost_vp_x_trf:.6f} vs res_vp.cost={res_vp.cost:.6f}"
          f" (差 {abs(cost_vp_x_trf - res_vp.cost):.2e})")
    print(f"      cost(trf.x @ 同口径)  ={cost_trf_x_vp:.6f} vs res_trf.cost={res_trf.cost:.6f}"
          f" (差 {abs(cost_trf_x_vp - res_trf.cost):.2e})")
    print(f"      两法 final cost 相对差 = {rel_cost_gap:.3e}")
    print(f"gauge: ρ中位比={rho_ratio:.4f} α中位比={alpha_ratio:.4f} "
          f"(预期互倒); 乘积矩阵中位比={rel_prod:.6f}")
    print(f"方向: trf vs VarPro 逐光夹角 = {[f'{a:.4f}°' for a in angs_tv]}")

    out = dict(
        meta=dict(purpose="OPT-1 阶段 1 烟测: VarPro vs trf 单起点对照(ballPNG cfg00)",
                  object=OBJ, N=3, subset="cfg00(SEED=20260906 复刻现行抽样)",
                  P_subsampled=P_k, seed=SEED, rep_median=f"计时取 {REP} 次中位",
                  weighted="False(与 joint_trf 目标严格一致)",
                  python="3.14.2", numpy=np.__version__,
                  scipy=__import__("scipy").__version__,
                  timing_note="机器非空闲: 后台 PID 8260 exp8r_v3.1(约1核)在跑; "
                               "计时为上界(与阶段 0 同口径)",
                  shadow_frac=frac_shadow, a_=a_, b_=b_,
                  lambert_resid=lambert_resid),
        trf=dict(cost=float(res_trf.cost), lae_deg=lae_trf, time_s_med=t_trf_med,
                 nfev=int(res_trf.nfev), njev=getattr(res_trf, "njev", None),
                 times=[round(v, 6) for v in t_trf]),
        varpro=dict(cost=float(res_vp.cost), lae_deg=lae_vp, time_s_med=t_vp_med,
                    n_outer=int(res_vp.n_outer), n_fev=int(res_vp.n_fev),
                    als_steps_last=int(res_vp.als_steps_last),
                    converged=bool(res_vp.converged), message=res_vp.message,
                    times=[round(v, 6) for v in t_vp]),
        cross=dict(cost_varpro_x_trf_semantics=cost_vp_x_trf,
                   cost_trf_x_same_semantics=cost_trf_x_vp,
                   rel_cost_gap=rel_cost_gap,
                   rho_ratio_med=rho_ratio, alpha_ratio_med=alpha_ratio,
                   product_ratio_med=rel_prod,
                   per_light_angle_trf_vs_vp_deg=angs_tv),
        wall_seconds=round(time.perf_counter() - t_wall0, 6),
    )
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[smoke] 落盘 -> {OUT}")


if __name__ == "__main__":
    main()
