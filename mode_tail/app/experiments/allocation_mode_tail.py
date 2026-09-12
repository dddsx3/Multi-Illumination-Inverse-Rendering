"""P-ALLOC2 v1.1 · mode-tail targeted calibration intervention（三臂修订版）。

**勘误（2026-09-12，外部审稿发现）**：v1.0 的 random_active48 臂在投影前
**遗漏了 gauge 对齐**（`sg` 已计算但未应用，`e = e − sg·ρ` 缺失），而
targeted 臂有对齐——两臂残差口径不对称。gauge 方向 ρ̂ 在 bottom-5 dual
子空间内的能量占比实测 0.91（obj_09_ball），random 臂能量被系统性抬高
（实测 ~27.8×，逐 seed 配对），v1.0 报告的"靶向干预全 10 单元显著"完全是
该伪影，**撤回**。v1.1 修复：三臂共用同一残差管线函数 `arm_energy`（结构性
杜绝口径不对称），并新增 scalar_targeted 臂（外部评估 action 1：检验
模式分辨诊断是否给出标量给不出的方向性预测）。

预注册判断判据（config，run 前提交；结果无关报告）：
  J1: targeted 是否相对 random_active48 降低 bottom-5 能量；
  J2: scalar_targeted 是否同样降低；
  J3（区分性检验）: targeted 的降幅是否严格大于 scalar 臂——
     是 ⇒ 模式分辨的方向性信号成立；否 ⇒ 模式语言无增量干预价值（诚实负结果）。

沿袭：与冻结 allocation 同构的场景装配/腐蚀机械/预算网格（P=1200、48
active/142、corrected interface）；配对 innovations 种子空间不相交。

输出：results/mode_tail/allocation_mode_tail.json（覆盖 v1.0 无效结果，
勘误记录于 docs/claims.md N-4）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from calibinfo.allocation.corruption import (                          # noqa: E402
    apply_scaled_corruption, raw_innovations)
from calibinfo.allocation.policies import (                            # noqa: E402
    LightBlocks, SelectionState, budget_scales, select_ordering_mode_aware)
from calibinfo.allocation.convex import CertificateProblem             # noqa: E402
from calibinfo.datasets.openillumination import load_object            # noqa: E402
from calibinfo.models.corruption import CorruptionGenerator            # noqa: E402
from experiments.openillumination_validation import NominalScene       # noqa: E402

N_RANDOM_PERMS = 6
N_MODES = 5


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, cwd=str(REPO)).stdout.strip()


def arm_energy(scales48, raw, scen, sig_logI, sig_rad, W_dual):
    """**唯一**的臂残差管线（三臂共用——结构性保证口径对称）：
    腐蚀注入 → 固定 n̂ 白化 GLS 重估 → gauge 对齐 → bottom-5 dual 坐标能量。"""
    d2, g = apply_scaled_corruption(scen.dirs, sig_logI, sig_rad, scales48, raw)
    rho_t = scen.estimate_albedo(d2, g)
    e = rho_t - scen.rho
    sg = (e * scen.rho).sum() / (scen.rho ** 2).sum()
    e = e - sg * scen.rho
    return ((e @ W_dual) ** 2).tolist()


def scalar_greedy_prefix(prob, kappa, k_max):
    """scalar_targeted 臂排序：J_A 最陡下降前缀（精确梯度；48 步）。
    这是标量 OED 口径的靶向（与 P-CERT 的 greedy 同款），预注册声明：
    它优化的是 J_A 本身，不是模式尾端——区分性检验的对照。"""
    t = np.ones(prob.blocks.L)
    chosen, out = [], {}
    for _ in range(k_max):
        _J, g = prob.J_and_grad(t)
        g_masked = np.where(prob.blocks.active, g, np.inf)
        pick = int(np.argmin(g_masked))
        t[pick] = kappa
        chosen.append(pick)
        out[len(chosen)] = list(chosen)
    return out


def run_object(obj_idx: int, obj_name: str, cfg: dict):
    """单对象全量计算（对象间零耦合，可并行/重排）。"""
    levels = [float(lv) for lv in cfg["levels"]]
    regimes = [int(rg) for rg in cfg["regimes"]]
    budgets_k = list(cfg["budgets_k"])
    seeds_per_level = int(cfg["seeds_per_level"])
    k_max = max(budgets_k)
    rows = []
    orderings_meta = []
    obj = load_object(cfg["data_root"], obj_name, data_meta=cfg.get("data_meta"))
    scen = NominalScene(obj, np.random.default_rng([20260910, obj_idx]),
                        noise_fit_convention=cfg["noise_fit_convention"])
    K = 142
    u_act = (scen.w * scen.s_hat)[:, :, None] * scen.B_phi
    M0_act = np.einsum("kpi,kpj->kij", scen.B_phi,
                       scen.w[:, :, None] * scen.B_phi)
    finf = scen.Finf_diag
    u142 = np.zeros((K, finf.shape[0], 3))
    M0142 = np.zeros((K, 3, 3))
    lam0142 = np.stack([np.eye(3)] * K)
    u142[scen.sel] = u_act
    M0142[scen.sel] = M0_act
    active142 = np.zeros(K, bool)
    active142[scen.sel] = True

    for lv_i, level in enumerate(levels):
        gen = CorruptionGenerator("joint", level)
        sig_logI, sig_rad = gen.sig_logI, np.radians(gen.sig_deg)
        lam0142[scen.sel] = np.linalg.inv(gen.sigma_phi_diag())
        for rg_i, regime in enumerate(regimes):
            state = SelectionState(u=u142, M0=M0142, lam0=lam0142.copy(),
                                   finf=finf, active=active142)
            _deg, W_dual = scen.predicted_degradation(
                gen.sigma_phi_diag(), mode_coordinate="dual")
            t_order, _steps = select_ordering_mode_aware(state, float(regime))
            prob = CertificateProblem(
                blocks=LightBlocks(u142, M0142, lam0142.copy(), finf,
                                   active142), kappa=1.0)
            s_order = scalar_greedy_prefix(prob, float(regime), k_max)
            orderings_meta.append(
                dict(object=obj_name, level=level, regime=regime,
                     targeted_first8=[int(x) for x in t_order[:8]],
                     scalar_first8=[int(x) for x in s_order[k_max][:8]]))
            for s_i in range(seeds_per_level):
                rng = np.random.default_rng(
                    [20260916, 7777, obj_idx, lv_i, s_i])
                raw = raw_innovations(rng, len(scen.sel))
                for k in budgets_k:
                    energies = {}
                    scales48 = budget_scales(t_order, k, regime)[scen.sel]
                    energies["targeted"] = arm_energy(
                        scales48, raw, scen, sig_logI, sig_rad, W_dual)
                    refined = set(s_order[k])
                    scales48 = np.where(
                        np.isin(scen.sel, list(refined)),
                        1.0 / np.sqrt(regime), 1.0)
                    energies["scalar_targeted"] = arm_energy(
                        scales48, raw, scen, sig_logI, sig_rad, W_dual)
                    acc = np.zeros(N_MODES)
                    for _p in range(N_RANDOM_PERMS):
                        ordr48 = rng.permutation(48)
                        scales48 = budget_scales(ordr48, k, regime)
                        acc += np.asarray(arm_energy(
                            scales48, raw, scen, sig_logI, sig_rad, W_dual))
                    energies["random_active48"] = (acc / N_RANDOM_PERMS).tolist()
                    rows.append(dict(
                        object=obj_name, level=level, regime=regime, k=k,
                        seed=s_i,
                        targeted=[float(x) for x in energies["targeted"]],
                        scalar_targeted=[float(x)
                                         for x in energies["scalar_targeted"]],
                        random_active48=[float(x)
                                         for x in energies["random_active48"]]))
    print(f"[alloc2] {obj_name} done", flush=True)
    return obj_name, rows, orderings_meta


def _task(payload):
    obj_idx, obj_name, cfg = payload
    t0 = time.time()
    obj_name, rows, ometa = run_object(obj_idx, obj_name, cfg)
    return obj_name, rows, ometa, time.time() - t0


def run(config_path=REPO / "configs/allocation_mode_tail.yaml",
        out_path=REPO / "results/mode_tail/allocation_mode_tail.json",
        workers: int = 1):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    levels = [float(lv) for lv in cfg["levels"]]
    regimes = [int(rg) for rg in cfg["regimes"]]
    budgets_k = list(cfg["budgets_k"])
    seeds_per_level = int(cfg["seeds_per_level"])
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    orderings_meta = []
    t_start = time.time()
    payloads = [(i, o, cfg) for i, o in enumerate(cfg["cohort"])]
    if workers > 1 and len(payloads) > 1:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=min(workers, len(payloads))) as pool:
            for obj_name, rws, ometa, dt in pool.imap_unordered(_task, payloads):
                rows.extend(rws)
                orderings_meta.extend(ometa)
                print(f"[alloc2] {obj_name} done in {dt:.0f}s "
                      f"(elapsed {time.time() - t_start:.0f}s)", flush=True)
    else:
        for pd in payloads:
            obj_name, rws, ometa, dt = _task(pd)
            rows.extend(rws)
            orderings_meta.extend(ometa)
            print(f"[alloc2] {obj_name} done in {dt:.0f}s "
                  f"(elapsed {time.time() - t_start:.0f}s)", flush=True)

    # ---- 聚合：配对 Δ，object 级 bootstrap ----
    from calibinfo.metrics.cluster_bootstrap import cluster_bootstrap
    ARMS = ("targeted", "scalar_targeted")
    agg = {}
    for regime in regimes:
        for k in budgets_k:
            rs = [r for r in rows if r["regime"] == regime and r["k"] == k]
            per_obj = {arm: {} for arm in ARMS + ("random_active48",)}
            for r in rs:
                for arm in ARMS + ("random_active48",):
                    per_obj[arm].setdefault(r["object"], []).append(
                        float(np.sum(r[arm])))
            objs = sorted({r["object"] for r in rs})
            blk = {}
            for arm in ARMS:
                deltas = {o: float(np.mean(per_obj[arm][o])
                                   - np.mean(per_obj["random_active48"][o]))
                          for o in objs}

                def _stat(sub, deltas=deltas):
                    return float(np.median([deltas[o] for o, _ in sub]))

                _pt, ci, _b = cluster_bootstrap([(o, None) for o in objs],
                                                _stat, 10000, 20260916)
                blk[arm] = dict(
                    delta_median=float(np.median(list(deltas.values()))),
                    delta_ci95=[float(ci[0]), float(ci[1])],
                    improved_negative=int(sum(1 for o in objs
                                              if deltas[o] < 0)),
                    per_object={o: deltas[o] for o in objs})
            # 区分性：targeted vs scalar（配对，同 bootstrap 重采样池）
            d_ts = {o: float(np.mean(per_obj["targeted"][o])
                             - np.mean(per_obj["scalar_targeted"][o]))
                    for o in objs}

            def _stat_ts(sub, d_ts=d_ts):
                return float(np.median([d_ts[o] for o, _ in sub]))

            _pt, ci_ts, _b = cluster_bootstrap([(o, None) for o in objs],
                                               _stat_ts, 10000, 20260916)
            blk["distinctiveness"] = dict(
                delta_targeted_minus_scalar_median=float(
                    np.median(list(d_ts.values()))),
                delta_ci95=[float(ci_ts[0]), float(ci_ts[1])],
                per_object={o: d_ts[o] for o in objs})
            # per-mode breakdown（targeted vs random）
            per_mode = np.zeros(N_MODES)
            for r in rs:
                per_mode += np.asarray(r["targeted"]) - np.asarray(
                    r["random_active48"])
            blk["per_mode_delta_mean"] = [float(x) for x in
                                          per_mode / len(rs)]
            agg[f"{regime}/{k}"] = dict(n_rows=len(rs), **blk)

    display = {key: {arm: dict(delta_median=round(blk[arm]["delta_median"], 6),
                               ci95=[round(v, 6) for v in
                                     blk[arm]["delta_ci95"]])
                     for arm in ARMS}
               | dict(distinctive_delta_median=round(
                   blk["distinctiveness"]["delta_targeted_minus_scalar_median"],
                   6))
               for key, blk in agg.items()}

    summary = dict(
        gate="P-ALLOC2 v1.1: mode-tail targeted calibration intervention "
             "(three-arm corrected rerun)",
        analysis_status="allocation_mode_tail_v1.1_gauge_fix",
        erratum=dict(
            v1_0_claim="targeted intervention significant in all 10 cells "
                       "(11/11 objects, CIs exclude 0)",
            retracted=True,
            cause="random_active48 arm missed the gauge alignment "
                  "(sg computed but not applied) while the targeted arm had "
                  "it; the gauge direction holds 0.91 energy fraction inside "
                  "the bottom-5 dual subspace, inflating the random arm ~28x",
            retraction="C4 claim in docs/claims.md and the README research "
                       "section is retracted; this v1.1 rerun replaces it"),
        endpoint=cfg["endpoint"]["primary"],
        noise_fit_convention=cfg["noise_fit_convention"],
        levels=levels, regimes=regimes, budgets_k=budgets_k,
        seeds_per_level=seeds_per_level,
        rows=rows, aggregated=agg, display=display,
        orderings_meta=orderings_meta[:20],
        manifest=dict(config_sha256=_sha(Path(config_path)),
                      git_sha=_git_sha(),
                      bootstrap=dict(B=10000, seed=20260916),
                      elapsed_s=round(time.time() - t_start, 1)),
        note="Three arms share ONE residual pipeline function "
             "(arm_energy: corrupt -> GLS -> gauge-align -> dual-project) - "
             "convention asymmetry structurally impossible. Paired-by-seed; "
             "outcome-independent reporting. J1/J2/J3 preregistered in the "
             "config.")
    out.write_bytes(json.dumps(summary, ensure_ascii=False, indent=1)
                    .encode("utf-8"))
    for key, blk in agg.items():
        for arm in ARMS:
            b = blk[arm]
            print(f"[alloc2] {key} {arm}: d median {b['delta_median']:+.4e} "
                  f"CI [{b['delta_ci95'][0]:+.4e}, {b['delta_ci95'][1]:+.4e}] "
                  f"improved {b['improved_negative']}/11")
        d = blk["distinctiveness"]
        print(f"[alloc2] {key} targeted-vs-scalar: d median "
              f"{d['delta_targeted_minus_scalar_median']:+.4e} "
              f"CI [{d['delta_ci95'][0]:+.4e}, {d['delta_ci95'][1]:+.4e}]")
    print(f"[alloc2] wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs/allocation_mode_tail.yaml"))
    ap.add_argument("--out", default=str(REPO / "results/mode_tail/allocation_mode_tail.json"))
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    run(args.config, args.out, workers=args.workers)
