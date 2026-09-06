#!/usr/bin/env python3
"""卡 O · exp11c 反向切法: 固定光照、换几何的标量预测子检验

设计(v3.1 卡 O): 固定 K=10 个光照配置, 跨 19 个 conf_ 场景换几何,
看几何感知的标量预测子(κ, κ_weighted)能否跨几何排序可辨识性。
响应 = 每场景法线角迹(Schur 块)。阴性对照 = σ_min(C_1)²(光照固定时应无预测力)。

判据: κ/κ_weighted 跨几何也失败 → "谱即诊断"彻底版成立;
     κ 跨几何有预测力 → 限定结论(几何感知标量的适用域)。两分支都入文。
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "p1" / "source" / "information_audit"))

from exp2_joint_fisher_schur import sh2  # noqa: E402
from exp11v3_kappa_expansion import (  # noqa: E402
    DATA_CONF, scene_conf, predictors_full, trace_theta_inv_safe,
)

OUT = HERE / "exp11c_reverse_cut.json"
K_CONFIGS = 10
N_LIGHTS = 5
SEED = 20260906


def main():
    rng = np.random.default_rng(SEED)
    scenes = sorted([d for d in DATA_CONF.iterdir() if d.is_dir()
                     and (d / "sh_coeffs_irradiance.npy").exists()
                     and d.name != "_canary_run1_degenerate"])
    print(f"场景数: {len(scenes)}")

    # 光照池: 从第一场景取 32 灯方向, 固定 K=10 个配置
    # (跨场景用同一组 C 矩阵——R-D2: 光照固定)
    pool0 = np.load(str(scenes[0] / "sh_coeffs_irradiance.npy")).astype(float)
    cfg_sels = [np.sort(rng.choice(32, N_LIGHTS, replace=False)) for _ in range(K_CONFIGS)]
    C_mats = [pool0[sel] for sel in cfg_sels]

    # 阴性对照验证: σ_min(C_1)² 应恒定(光照固定)
    c1_mins = [np.linalg.svd(C[:, 1:4], compute_uv=False)[-1] ** 2 for C in C_mats]
    print(f"σ_min(C_1)² 范围: [{min(c1_mins):.4f}, {max(c1_mins):.4f}] (应恒定 = 阴性对照)")

    # 每场景 × 每配置: 预测子 + 响应
    all_rows = []      # (scene, config, κ, κ_w, σ_min_C1², tr_schur)
    for sd in scenes:
        name = sd.name
        try:
            nrm, a, _, pool, H, W = scene_conf(sd)
            # 该场景的灯方向可能与 pool0 不同 → 用场景自己的池但同一批方向子集
            # (R-D2 修正: "固定光照"= 固定方向子集索引, 而非固定系数——不同场景的灯方向不同)
            for ci, sel in enumerate(cfg_sels):
                if max(sel) >= len(pool):
                    continue
                C = pool[sel]
                pred = predictors_full(C, nrm, a)
                try:
                    tr = trace_theta_inv_safe(nrm, a, C)[1]   # Schur 迹
                except Exception:
                    tr = float('nan')
                all_rows.append(dict(scene=name, config=ci,
                                     kappa=pred['kappa'],
                                     kappa_w=pred['kappa_weighted'],
                                     sigma_min_C1_sq=pred['sigma_min_C1_sq'],
                                     tr_schur=tr))
        except Exception as exc:
            print(f"  {name}: FAIL {exc}")

    print(f"有效行: {len(all_rows)}")

    # 每配置出一条跨场景 Spearman
    results = {}
    for ci in range(K_CONFIGS):
        rows_ci = [r for r in all_rows if r["config"] == ci and np.isfinite(r["tr_schur"])]
        if len(rows_ci) < 10:
            continue
        trs = [r["tr_schur"] for r in rows_ci]
        stats = {}
        for pname in ("kappa", "kappa_w", "sigma_min_C1_sq"):
            xs = [r[pname] for r in rows_ci]
            if np.std(xs) > 0 and np.std(trs) > 0:
                r_, p_ = spearmanr(xs, trs)
                stats[pname] = dict(rho=float(r_), p=float(p_))
        results[ci] = dict(n=len(rows_ci), **stats)
        ks = "  ".join(f"{k}: ρ={v['rho']:.3f}(p={v['p']:.3f})"
                       for k, v in stats.items())
        print(f"  cfg{ci:02d} (n={len(rows_ci)}): {ks}")

    # 汇总
    summary = {}
    for pname in ("kappa", "kappa_w"):
        rhos = [results[ci][pname]["rho"] for ci in results
                if pname in results[ci]]
        ps = [results[ci][pname]["p"] for ci in results if pname in results[ci]]
        n_sig = sum(1 for p in ps if p < 0.05)
        # Fisher 合并
        from scipy.stats import chi2
        if ps:
            fisher_stat = -2 * sum(np.log(max(p, 1e-300)) for p in ps)
            meta_p = 1 - chi2.cdf(fisher_stat, 2 * len(ps))
        else:
            meta_p = float('nan')
        summary[pname] = dict(n_configs=len(rhos), rho_mean=float(np.mean(rhos)),
                              rho_std=float(np.std(rhos)), n_sig=n_sig,
                              meta_p=float(meta_p))
    out = dict(per_config=results, summary=summary,
               meta=dict(K=K_CONFIGS, n_lights=N_LIGHTS, seed=SEED, n_scenes=len(scenes),
                         n_rows=len(all_rows)))
    print("\n[exp11c] 汇总:")
    for pname, s in summary.items():
        print(f"  {pname}: ρ均值={s['rho_mean']:.3f}±{s['rho_std']:.3f} | "
              f"显著配置 {s['n_sig']}/{s['n_configs']} | meta-p={s['meta_p']:.2e}")

    # 判定
    kw = summary.get("kappa_w", {})
    kp = summary.get("kappa", {})
    out["verdict"] = dict(
        kappa_w=kw, kappa=kp,
        interpretation=("κ/κ_w 跨几何有预测力 → 几何感知标量的适用域" if
                        max(kw.get('n_sig', 0), kp.get('n_sig', 0)) > K_CONFIGS * 0.3 else
                        "κ/κ_w 跨几何也失败 → 任何标量代理都不足, 必须看全谱(谱即诊断彻底版)"))
    print("判定:", out["verdict"]["interpretation"])
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[exp11c] 落盘 -> {OUT}")


if __name__ == "__main__":
    main()
