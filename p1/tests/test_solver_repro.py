"""R4″ H1.5 · C-1 改动的可复现性单测（阻塞 Task B/C/G）。

必须全部 PASS 才允许进入 Task C（noise floor），否则 σ_solver 测的不是种子效应。

T1 向后兼容 : seed=None 时与冻结的 988 trial 语义一致（写死 20260830+rs）
T2 种子生效 : 不同 seed 给出不同结果；同 seed 两次逐位一致
T3 theta0   : 显式初值生效（oracle-local 路径可用）
T4 诊断量   : proj_grad_norm / tail_rel_change / conv_finite / restart_records 齐备且合理
T5 trace    : return_trace 的轨迹长度 = base_iters+200N，末值 == final_loss
T6 gauge    : proj_grad_norm 对 gauge 变换 (a,c)→(λa,c/λ) 不敏感（同一物理点）

运行：python p1/tests/test_solver_repro.py
"""
import math
import os
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "p1", "source", "information_audit"))
sys.path.insert(0, os.path.join(_REPO, "p1", "source", "physics"))

from information_audit_v2 import joint_solve, load_scene, si_mae_np  # noqa: E402

SCENE = os.path.join(_REPO, "p1", "calibration_set", "data_sun_confirmatory",
                     "conf_sphere_r05")
SUBSET = [0, 5, 11]          # N=3，跑得快
FAST = dict(restarts=1, base_iters=120)   # 单测只验证机制，不求收敛

RESULTS = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    RESULTS.append((name, status, detail))
    print(f"  [{status}] {name}  {detail}")
    return cond


def main():
    print("=" * 74)
    print("R4″ C-1 solver 可复现性单测（seed / theta0 / trace / proj_grad_norm）")
    print("=" * 74)
    sc = load_scene(SCENE)
    H, W = sc["mask"].shape

    # ---------------- T1 向后兼容 ----------------
    print("T1 · 向后兼容（seed=None ⇔ 写死 20260830+rs）")
    r_none_a = joint_solve(sc, SUBSET, seed=None, **FAST)
    r_none_b = joint_solve(sc, SUBSET, seed=None, **FAST)
    r_explicit = joint_solve(sc, SUBSET, seed=20260830, **FAST)
    check("T1a seed=None 两次逐位一致",
          r_none_a["final_loss"] == r_none_b["final_loss"]
          and np.array_equal(r_none_a["A_hat"], r_none_b["A_hat"]),
          f"loss={r_none_a['final_loss']:.12e}")
    check("T1b seed=None == seed=20260830（默认基准值）",
          r_none_a["final_loss"] == r_explicit["final_loss"]
          and np.array_equal(r_none_a["A_hat"], r_explicit["A_hat"]),
          f"{r_none_a['final_loss']:.12e} vs {r_explicit['final_loss']:.12e}")
    check("T1c seed_base 记录正确", r_none_a["seed_base"] == 20260830,
          f"seed_base={r_none_a['seed_base']}")

    # ---------------- T2 种子生效 ----------------
    print("T2 · 种子生效（Task C 的 σ_solver 依赖此项）")
    seeds = [101, 202, 303, 404, 505]
    losses, errs = [], []
    for s in seeds:
        r = joint_solve(sc, SUBSET, seed=s, **FAST)
        losses.append(r["final_loss"])
        errs.append(si_mae_np(r["A_hat"], sc["albedo"], sc["mask"]))
    n_uniq = len(set(f"{v:.12e}" for v in losses))
    check("T2a 5 个不同 seed 给出 5 个不同 loss（非逐位相同）", n_uniq == 5,
          f"unique={n_uniq}/5, spread={max(losses)-min(losses):.3e}")
    r_rep_a = joint_solve(sc, SUBSET, seed=777, **FAST)
    r_rep_b = joint_solve(sc, SUBSET, seed=777, **FAST)
    check("T2b 同 seed 重复调用逐位一致",
          r_rep_a["final_loss"] == r_rep_b["final_loss"]
          and np.array_equal(r_rep_a["A_hat"], r_rep_b["A_hat"]),
          f"loss={r_rep_a['final_loss']:.12e}")
    e_arr = np.array(errs)
    check("T2c seed 引起的 error 变异非零（σ_solver 可测）",
          float(e_arr.std()) > 0,
          f"err mean={e_arr.mean():.6f} std={e_arr.std():.3e} "
          f"cv={e_arr.std()/max(e_arr.mean(),1e-12):.3e}")

    # ---------------- T3 theta0 ----------------
    print("T3 · theta0 显式初值（oracle-local 路径）")
    a_gt = sc["albedo"].astype(np.float64)
    c_gt = sc["sh_irr"][np.asarray(SUBSET)].astype(np.float64)
    rng = np.random.default_rng(0)
    d_a = 0.05 * np.sqrt((a_gt[sc["mask"]] ** 2).mean())
    d_c = 0.05 * np.linalg.norm(c_gt) / math.sqrt(c_gt.size)
    theta0 = (np.clip(a_gt + rng.normal(0, d_a, a_gt.shape), 1e-4, None),
              c_gt + rng.normal(0, d_c, c_gt.shape))
    r_loc = joint_solve(sc, SUBSET, seed=101, theta0=theta0, **FAST)
    r_glb = joint_solve(sc, SUBSET, seed=101, theta0=None, **FAST)
    check("T3a theta0 改变结果（初值确实生效）",
          r_loc["final_loss"] != r_glb["final_loss"],
          f"local={r_loc['final_loss']:.6e} global={r_glb['final_loss']:.6e}")
    r_loc_b = joint_solve(sc, SUBSET, seed=101, theta0=theta0, **FAST)
    check("T3b 同 theta0 + 同 seed 逐位一致",
          r_loc["final_loss"] == r_loc_b["final_loss"],
          f"loss={r_loc['final_loss']:.12e}")
    e_loc = si_mae_np(r_loc["A_hat"], sc["albedo"], sc["mask"])
    e_glb = si_mae_np(r_glb["A_hat"], sc["albedo"], sc["mask"])
    check("T3c GT-near 初值在同迭代预算下误差不劣于 global（合理性）",
          e_loc <= e_glb * 1.5,
          f"err local={e_loc:.6f} vs global={e_glb:.6f}")

    # ---------------- T4 诊断量 ----------------
    print("T4 · 绝对收敛判据的原始量")
    r = joint_solve(sc, SUBSET, seed=101, restarts=2, base_iters=120)
    keys = ["proj_grad_norm", "tail_rel_change", "conv_finite", "restart_records",
            "best_restart", "seed_base"]
    check("T4a 新字段齐备", all(k in r for k in keys),
          f"missing={[k for k in keys if k not in r]}")
    check("T4b proj_grad_norm 有限且 >0",
          math.isfinite(r["proj_grad_norm"]) and r["proj_grad_norm"] > 0,
          f"pgn={r['proj_grad_norm']:.4e}")
    check("T4c tail_rel_change 有限且 ≥0",
          math.isfinite(r["tail_rel_change"]) and r["tail_rel_change"] >= 0,
          f"tail_rel={r['tail_rel_change']:.4e}")
    check("T4d conv_finite=True（正常场景应有限）", r["conv_finite"] is True)
    rr = r["restart_records"]
    check("T4e restart_records 覆盖全部 restart 且 seed 递增",
          len(rr) == 2 and rr[0]["seed"] == 101 and rr[1]["seed"] == 102,
          f"seeds={[x['seed'] for x in rr]}")
    check("T4f best_restart 指向 loss 最小者",
          rr[r["best_restart"]]["final_loss"] == min(x["final_loss"] for x in rr),
          f"best={r['best_restart']}, losses={[round(x['final_loss'],9) for x in rr]}")

    # ---------------- T5 trace ----------------
    print("T5 · loss trace")
    rt = joint_solve(sc, SUBSET, seed=101, restarts=2, base_iters=120,
                     return_trace=True)
    exp_len = 120 + 200 * len(SUBSET)
    check("T5a loss_trace 长度 == base_iters + 200N",
          rt["loss_trace"].shape[0] == exp_len,
          f"len={rt['loss_trace'].shape[0]}, expect={exp_len}")
    check("T5b trace 末值 == final_loss",
          float(rt["loss_trace"][-1]) == rt["final_loss"],
          f"{float(rt['loss_trace'][-1]):.12e} vs {rt['final_loss']:.12e}")
    check("T5c all_traces 覆盖全部 restart", len(rt["all_traces"]) == 2)
    check("T5d trace 单调下降为主（末 10% 均值 < 首 10% 均值）",
          rt["loss_trace"][-exp_len // 10:].mean() < rt["loss_trace"][:exp_len // 10].mean(),
          f"first10%={rt['loss_trace'][:exp_len//10].mean():.4e} "
          f"last10%={rt['loss_trace'][-exp_len//10:].mean():.4e}")
    check("T5e return_trace 不改变数值（与不带 trace 的同 seed 一致）",
          rt["final_loss"] == r["final_loss"],
          f"{rt['final_loss']:.12e} vs {r['final_loss']:.12e}")

    # ---------------- T6 gauge 不敏感 ----------------
    print("T6 · proj_grad_norm 的 gauge 不敏感性")
    lam = 4.0
    th_a = (np.clip(a_gt, 1e-4, None), c_gt)
    th_b = (np.clip(a_gt * lam, 1e-4, None), c_gt / lam)
    ra = joint_solve(sc, SUBSET, seed=101, theta0=th_a, restarts=1, base_iters=0)
    rb = joint_solve(sc, SUBSET, seed=101, theta0=th_b, restarts=1, base_iters=0)
    # base_iters=0 → 仍会跑 200N 步；用 loss 是否接近判断是否同一物理点
    rel_loss = abs(ra["final_loss"] - rb["final_loss"]) / max(ra["final_loss"], 1e-12)
    check("T6a gauge 变换后 loss 接近（同一物理点，优化轨迹允许差异）",
          rel_loss < 0.5,
          f"loss {ra['final_loss']:.4e} vs {rb['final_loss']:.4e} (rel {rel_loss:.2e})")
    ratio = rb["proj_grad_norm"] / max(ra["proj_grad_norm"], 1e-300)
    check("T6b proj_grad_norm 未随 gauge 缩放量级漂移（0.05 < ratio < 20）",
          0.05 < ratio < 20,
          f"pgn {ra['proj_grad_norm']:.3e} vs {rb['proj_grad_norm']:.3e} (ratio {ratio:.3f})")

    n_fail = sum(1 for _, s, _ in RESULTS if s == "FAIL")
    print("=" * 74)
    print(f"总计 {len(RESULTS)} 项 · PASS {len(RESULTS)-n_fail} · FAIL {n_fail}")
    if n_fail:
        print("C-1 GATE: FAIL —— 禁止进入 Task C（σ_solver 会测不到种子效应）")
        sys.exit(1)
    print("C-1 GATE: PASS（seed / theta0 / trace / 诊断量 全部生效且向后兼容）")


if __name__ == "__main__":
    main()
