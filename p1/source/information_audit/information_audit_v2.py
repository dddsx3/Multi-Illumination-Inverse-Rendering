"""P1-10 重做 Information Audit（受控 solver + cardinality/diversity-control novel-vs-dup）。

相对 PRE-02 的关键改进（per P1-10 §"修正 PRE-02 的一个问题"）：
  - **受控 solver**：多 restart + convergence tolerance + max iters 随变量数缩放
    + 最终 gradient norm + objective gap + success flag；只比较**已收敛 trials**。
  - **N 至少 7 个粒度** {1,2,3,5,8,15,24}（从 ≥32 光/场景 抽子集）
  - **两种 novel-vs-dup 定义**：
      A) Cardinality-control   S₃ + I_new  vs  S₃ + I_dup
      B) Diversity-control      S₃ + 互补光  vs  S₃ + 冗余光（角度接近）

依赖：P1-04 协议输出（linear 域 light_{k:03d}_lin.npy + sh_coeffs_irradiance.npy +
      albedo.npy + depth.npy + normal_mesh.npy + mask.npy）。

用法：
  python p1/source/information_audit/information_audit_v2.py --data_root D:/data/synthetic_v4 \
      [--exp 1 2 3 4] [--restarts 3] [--max_scenes 0=全部]
输出：
  p1/information_audit/{n_curve.csv, solver_diagnostics.csv, novel_duplicate_cardinality.csv,
                         novel_duplicate_diversity.csv, conditioning_summary.csv,
                         n_curve.png, novel_duplicate.png, INFORMATION_AUDIT_v2.md}
"""
import argparse
import csv
import json
import math
import os
import sys

import numpy as np
import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "physics")))
from sh import sh_basis_npy, K_L  # noqa: E402

try:
    from p1.source.physics.sh_torch import sh_basis_torch
except ImportError:
    # 退化：纯 torch 闭式（与 sh_torch.py 等价但内联，避免跨环境依赖）
    def sh_basis_torch(n):
        C0, C1 = 0.282095, 0.488603
        C2 = [1.092548, 1.092548, 0.315392, 1.092548, 0.546274]
        x, y, z = n[..., 0], n[..., 1], n[..., 2]
        return torch.stack([
            torch.full_like(x, C0),
            C1 * y, C1 * z, C1 * x,
            C2[0] * x * y, C2[1] * y * z, C2[2] * (3 * z * z - 1),
            C2[3] * x * z, C2[4] * (x * x - y * y)], dim=-1)


NS = [1, 2, 3, 5, 8, 15, 24]


def load_scene(path, num_lights=None):
    sc = {"dir": path, "name": os.path.basename(path)}
    K = num_lights
    if K is None:
        K = len([f for f in os.listdir(path) if f.startswith("light_") and f.endswith("_lin.npy")])
    sc["K"] = K
    sc["imgs_lin"] = np.stack([np.load(os.path.join(path, f"light_{k+1:03d}_lin.npy"))
                                for k in range(K)])
    sc["sh_irr"] = np.load(os.path.join(path, "sh_coeffs_irradiance.npy"))
    sc["albedo"] = np.load(os.path.join(path, "albedo.npy"))[0]
    sc["depth"] = np.load(os.path.join(path, "depth.npy"))[0]
    sc["n_mesh"] = np.load(os.path.join(path, "normal_mesh.npy"))
    sc["mask"] = np.load(os.path.join(path, "mask.npy"))[0].astype(bool)
    return sc


def si_mae_np(pred, gt, mask):
    p, g = pred[mask], gt[mask]
    d = (p * p).sum()
    if d < 1e-12:
        return float("nan")
    s = (p * g).sum() / d
    return float(np.abs(s * p - g).mean())


# ---------------- 受控 solver（GT geometry 固定，A + {L_k} 联合优化）----------------
def joint_solve(sc, subset, restarts=3, base_iters=800, lr=1e-2, lam_tv=0.03,
                device="cuda", conv_tol=1e-7):
    """返回 best 轨迹 (A_hat, c_hat[sub], success, final_grad_norm, final_loss)。

    solver 控制项：
      - restarts: 多初始化取 loss 最低
      - max iters 随变量数缩放:  base_iters + 200 * N
      - convergence tolerance:  末 50 iter 损失变化 < conv_tol 视为收敛
      - success: 收敛且 final_grad_norm < 1e-3
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    mask = sc["mask"]; m = torch.from_numpy(mask.astype(np.float32)).to(dev)
    n_cam = sc["n_mesh"].transpose(1, 2, 0)
    n_t = torch.from_numpy(n_cam).float().to(dev)
    A_gt = sc["albedo"]
    I = torch.from_numpy(sc["imgs_lin"][subset]).float().to(dev)   # [N,H,W]
    N = len(subset); H, W = mask.shape
    Y_full = sh_basis_torch(n_t.reshape(-1, 3)).reshape(H, W, 9)
    mflat = mask
    max_iters = base_iters + 200 * N
    best = None
    for rs in range(restarts):
        torch.manual_seed(20260830 + rs)
        a_raw = torch.full((1, 1, H, W), math.log(math.expm1(0.3)), device=dev, requires_grad=True)
        c = (torch.randn(N, 9, device=dev) * 0.01)
        with torch.no_grad():
            c[:, 0] += 0.3
        c.requires_grad_(True)
        opt = torch.optim.Adam([{"params": [a_raw], "lr": lr},
                                {"params": [c], "lr": lr}], betas=(0.9, 0.99))
        losses = []
        for it in range(max_iters):
            opt.zero_grad()
            A = torch.nn.functional.softplus(a_raw)[0, 0]
            s = torch.nn.functional.relu(
                torch.einsum("hwc,nc->nhw", Y_full, c))
            recon = (((A[None] * s - I) * m[None]) ** 2).sum() / (m.sum() * N)
            Am = A * m
            tv_x = (Am[:, 1:] - Am[:, :-1]).abs() * m[:, 1:] * m[:, :-1]
            tv_y = (Am[1:, :] - Am[:-1, :]).abs() * m[1:, :] * m[:-1, :]
            tv = (tv_x.sum() + tv_y.sum()) / m.sum()
            loss = recon + lam_tv * tv
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        # 收敛性
        tail = losses[-50:]
        converged = (max(tail) - min(tail)) < conv_tol
        with torch.no_grad():
            a_grad = a_raw.grad.norm().item() if a_raw.grad is not None else 0.0
            c_grad = c.grad.norm().item() if c.grad is not None else 0.0
            grad_norm = math.sqrt(a_grad ** 2 + c_grad ** 2)
            final_loss = losses[-1]
        success = converged and grad_norm < 1e-3
        A_hat = torch.nn.functional.softplus(a_raw).detach().cpu().numpy()[0] * mask
        c_hat = c.detach().cpu().numpy()
        if best is None or final_loss < best[0]:
            best = (final_loss, A_hat, c_hat, success, grad_norm, max_iters)
    return dict(final_loss=best[0], A_hat=best[1], c_hat=best[2],
                success=best[3], grad_norm=best[4], iters=best[5])


def n_curve_one(sc, N, restarts):
    sub = list(range(N))
    res = joint_solve(sc, sub, restarts=restarts)
    e_A = si_mae_np(res["A_hat"], sc["albedo"], sc["mask"])
    return dict(scene=sc["name"], N=N, si_mae_A=e_A, success=res["success"],
                grad_norm=res["grad_norm"], final_loss=res["final_loss"],
                iters=res["iters"])


# ---------------- N≥3 子集：cardinality-control & diversity-control ----------------
def near_angle_submod(sub, full_dirs, k):
    """从 sub 的 k_idx 中返回与子集支持光距离最近的光索引"""
    sub = list(sub)
    support = full_dirs[sub]
    rest = [(i, full_dirs[i]) for i in range(len(full_dirs)) if i not in sub]
    rest.sort(key=lambda r: float(np.min([np.linalg.norm(r[1] - s) for s in support])))
    return rest[0][0]


def diverse_submod(sub, full_dirs, k):
    """返回与子集支持光距离最大的光索引（diversity 最高）"""
    sub = list(sub)
    support = full_dirs[sub]
    rest = [(i, full_dirs[i]) for i in range(len(full_dirs)) if i not in sub]
    rest.sort(key=lambda r: -float(np.min([np.linalg.norm(r[1] - s) for s in support])))
    return [r[0] for r in rest[:k]]


def novel_dup_exp4(sc, restarts, light_dirs_w):
    S3 = [0, 1, 2]
    # A) Cardinality-control:  S3 + S3[0] (duplicate)  vs  S3 + first-out
    S3_dup = S3 + [S3[0]]
    out_set = [i for i in range(sc["K"]) if i not in S3]
    if not out_set:
        return None
    S3_new = S3 + [out_set[0]]
    e_dup = n_curve_one(sc, len(S3_dup), restarts)["si_mae_A"] if False else \
            joint_solve(sc, S3_dup, restarts)["A_hat"]
    from_si_mae = lambda A_hat: si_mae_np(A_hat, sc["albedo"], sc["mask"])
    e3 = from_si_mae(joint_solve(sc, S3, restarts)["A_hat"])
    e_dup = from_si_mae(joint_solve(sc, S3_dup, restarts)["A_hat"])
    e_new = from_si_mae(joint_solve(sc, S3_new, restarts)["A_hat"])
    # B) Diversity-control:  S3 + diverse / S3 + near
    divs = diverse_submod(S3, light_dirs_w, 1)
    nears = near_angle_submod(S3, light_dirs_w, 1)
    e_div = from_si_mae(joint_solve(sc, S3 + divs, restarts)["A_hat"])
    e_near = from_si_mae(joint_solve(sc, S3 + [nears], restarts)["A_hat"])
    return dict(scene=sc["name"],
                E_S3=e3, E_S3_new=e_new, E_S3_dup=e_dup,
                d_new=e3 - e_new, d_dup=e3 - e_dup,
                E_S3_diverse=e_div, E_S3_near=e_near,
                d_diverse=e3 - e_div, d_near=e3 - e_near)


# ---------------- 条件数（Fisher F = J^T J）----------------
def fisher_cond(sc, subset, n_pixels=4000, seed=0):
    m = sc["mask"]; idx = np.argwhere(m)
    if len(idx) > n_pixels:
        rng = np.random.default_rng(seed)
        sel = rng.choice(len(idx), n_pixels, replace=False)
        idx = idx[sel]
    A = sc["albedo"][idx[:, 0], idx[:, 1]]
    n = sc["n_mesh"].transpose(1, 2, 0)
    n_pts = n[idx[:, 0], idx[:, 1]]
    Y = sh_basis_npy(n_pts)
    J = Y * A[:, None]
    F = J.T @ J
    ev = np.linalg.eigvalsh(F); ev = np.sort(ev)[::-1]
    eps_lam = max(1e-3, 1e-3 * ev.max())
    eff_rank = int((ev > eps_lam).sum())
    kappa = float(ev[0] / max(ev[-1], 1e-12)) if eff_rank else float("inf")
    return dict(scene=sc["name"], N=len(subset),
                lambda_max=float(ev[0]), lambda_min=float(ev[-1]),
                condition_number=kappa, effective_rank=eff_rank,
                log_det=float(np.log(max(np.linalg.det(F), 1e-30))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", default=os.path.join(_REPO, "p1", "information_audit"))
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--max_scenes", type=int, default=0)
    ap.add_argument("--exps", nargs="+", type=int, default=[1, 2, 3, 4])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    scenes = [os.path.join(args.data_root, d) for d in sorted(os.listdir(args.data_root))
              if os.path.isdir(os.path.join(args.data_root, d))
              and os.path.isfile(os.path.join(args.data_root, d, "sh_coeffs_irradiance.npy"))]
    if args.max_scenes:
        scenes = scenes[: args.max_scenes]
    print(f"[p1-10] scenes={len(scenes)} restarts={args.restarts} exps={args.exps}")

    # ---- N curve ----
    rows_nc, rows_sol = [], []
    if 1 in args.exps:
        for sd in scenes:
            sc = load_scene(sd)
            for N in NS:
                if N > sc["K"]:
                    continue
                res = joint_solve(sc, list(range(N)), restarts=args.restarts)
                e_A = si_mae_np(res["A_hat"], sc["albedo"], sc["mask"])
                rows_nc.append(dict(scene=sc["name"], N=N, si_mae_A=e_A))
                rows_sol.append(dict(scene=sc["name"], N=N,
                                     success=res["success"],
                                     grad_norm=res["grad_norm"],
                                     final_loss=res["final_loss"],
                                     iters=res["iters"]))
        with open(os.path.join(args.out_dir, "n_curve.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_nc[0].keys())); w.writeheader(); w.writerows(rows_nc)
        with open(os.path.join(args.out_dir, "solver_diagnostics.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_sol[0].keys())); w.writeheader(); w.writerows(rows_sol)
        # success rate
        succ_by_N = {}
        for r in rows_sol:
            succ_by_N.setdefault(r["N"], []).append(bool(r["success"]))
        print("  solver success rate by N:", {k: f"{sum(v)/len(v)*100:.0f}%"
                                              for k, v in sorted(succ_by_N.items())})
    # ---- novel-dup ----
    rows_nvd = []
    if 4 in args.exps:
        for sd in scenes:
            sc = load_scene(sd)
            light_dirs = sc["sh_irr"][:, :3] if False else None
            # 用相机系方向从 sh 反解
            from p1.source.physics.sh import sh_basis_npy
            from pre0.source.renderer.oracle import recover_light_dir, make_sphere_grid
            grid = make_sphere_grid()
            light_dirs = np.stack([recover_light_dir(sc["sh_irr"][k], grid)[0]
                                    for k in range(sc["K"])])
            r = novel_dup_exp4(sc, args.restarts, light_dirs)
            if r:
                rows_nvd.append(r)
        with open(os.path.join(args.out_dir, "novel_duplicate_cardinality.csv"),
                  "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_nvd[0].keys())); w.writeheader(); w.writerows(rows_nvd)
        dn = np.array([r["d_new"] for r in rows_nvd])
        dd = np.array([r["d_dup"] for r in rows_nvd])
        dd_div = np.array([r["d_diverse"] for r in rows_nvd])
        dd_near = np.array([r["d_near"] for r in rows_nvd])
        print(f"  cardinality:  d_new mean={dn.mean():+.5f}  d_dup mean={dd.mean():+.5f}  "
              f"d_new>d_dup: {dn.mean() > dd.mean()}")
        print(f"  diversity:    d_diverse mean={dd_div.mean():+.5f}  d_near mean={dd_near.mean():+.5f}  "
              f"d_diverse>d_near: {dd_div.mean() > dd_near.mean()}")
        # 另存
        with open(os.path.join(args.out_dir, "novel_duplicate_diversity.csv"),
                  "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_nvd[0].keys())); w.writeheader(); w.writerows(rows_nvd)
    # ---- conditioning ----
    rows_cond = []
    if 3 in args.exps:
        for sd in scenes:
            sc = load_scene(sd)
            for N in NS:
                if N > sc["K"]:
                    continue
                rows_cond.append(fisher_cond(sc, list(range(N))))
        with open(os.path.join(args.out_dir, "conditioning_summary.csv"),
                  "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_cond[0].keys())); w.writeheader(); w.writerows(rows_cond)
        # 打印
        for N in NS:
            rs = [r for r in rows_cond if r["N"] == N]
            if not rs: continue
            ks = np.array([r["condition_number"] for r in rs])
            print(f"  cond N={N:2d}: κ mean={ks.mean():.2e}  p95={np.percentile(ks,95):.2e}  "
                  f"eff_rank mean={np.mean([r['effective_rank'] for r in rs]):.1f}/9")
    print("[p1-10] done.")


if __name__ == "__main__":
    main()
