"""P1-R4′ · 批量化 joint solver（E2/G2/E3 确认性 Gate 的算力核心）。

与 `information_audit_v2.joint_solve` 的关系：
  - 数学逐元素等价（同一模型/损失/Adam 超参/迭代数/收敛判据/初始化种子语义）；
  - 区别仅在实现：把 (subset × restart) 打成 batch 维，一次前向/反向处理
    B 个 trial，消除 Python 循环开销（预计 ~B× 提速，8000+ runs 才可行）；
  - 冻结验证门槛（预注册 §2）：与串行版在验证用例上 SI-MAE 相对差 ≤ 1e-3。
    验证脚本入口：`python solver_batched.py --validate --data_root ... `

可复现性语义（与串行严格一致的关键）：
  串行版每个 call 内 `torch.manual_seed(20260830 + rs)` 重置 → 同一 restart
  序号的 c 初始化噪声在所有 (scene, subset) 上本来就相同。批量版按 rs 抽一次
  c_init 广播到 batch，逐位等价。a_raw 初始化为常数，无随机性。
"""
import argparse
import math
import os
import sys

import numpy as np
import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "p1", "source", "information_audit"))
from information_audit_v2 import joint_solve, si_mae_np, load_scene  # noqa: E402


def joint_solve_batched(sc, subsets, restarts=3, base_iters=800, lr=1e-2,
                        lam_tv=0.03, device="cuda", conv_tol=1e-7,
                        grad_tol=1e-3, chunk=None):
    """对同一场景的多个光照子集批量求解。

    subsets: list of list[int]（同一 N 的子集列表——按 N 分组调用）
    返回 list of dict（与 joint_solve 同构）：final_loss/A_hat/success/
    grad_norm/iters/converged/tail_range/restart。
    A_hat 仅保留 mask 内。raw 诊断 tail_range/grad_norm 随 rec 落盘，
    供统计阶段按预注册冻结阈值事后重判收敛。
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    mask = sc["mask"]
    m = torch.from_numpy(mask.astype(np.float32)).to(dev)
    n_cam = sc["n_mesh"].transpose(1, 2, 0)
    n_t = torch.from_numpy(n_cam).float().to(dev)
    H, W = mask.shape
    Y_full = torch.from_numpy(
        _sh_basis_torch_np(n_cam.reshape(-1, 3))).float().to(dev).reshape(H, W, 9)
    A_gt = sc["albedo"]
    msum = float(m.sum())
    mflat = m[None, None]                       # [1,1,H,W] 广播用
    B_sub = len(subsets)
    out = [None] * B_sub

    # (subset, restart) 展平；同一 rs 的 c_init 全 batch 相同（串行种子语义）
    trials = [(si, rs) for si in range(B_sub) for rs in range(restarts)]
    chunk = chunk or len(trials)
    N0 = len(subsets[0])
    assert all(len(s) == N0 for s in subsets), "按 N 分组后调用（本实现要求同长）"

    for t0 in range(0, len(trials), chunk):
        sel = trials[t0:t0 + chunk]
        Bp = len(sel)
        I = torch.stack([torch.from_numpy(
            sc["imgs_lin"][subsets[si]]).float() for si, _ in sel]).to(dev)  # [Bp,N,H,W]
        N = I.shape[1]
        max_iters = base_iters + 200 * N

        a_raw = torch.full((Bp, 1, H, W), math.log(math.expm1(0.3)),
                           device=dev, requires_grad=True)
        c = torch.zeros(Bp, N, 9, device=dev)
        for rs in range(restarts):
            g = torch.Generator(device="cpu").manual_seed(20260830 + rs)
            noise = torch.randn(N, 9, generator=g) * 0.01        # 串行语义：seed 按重置
            noise[:, 0] += 0.3
            for b, (si, rs2) in enumerate(sel):
                if rs2 == rs:
                    c[b] = noise
        c.requires_grad_(True)
        opt = torch.optim.Adam([{"params": [a_raw], "lr": lr},
                                {"params": [c], "lr": lr}], betas=(0.9, 0.99))
        losses = np.zeros((Bp, max_iters), dtype=np.float64)
        for it in range(max_iters):
            opt.zero_grad()
            A = torch.nn.functional.softplus(a_raw)[:, 0]        # [Bp,H,W]
            s = torch.nn.functional.relu(
                torch.einsum("hwc,bnc->bnhw", Y_full, c))        # [Bp,N,H,W]
            recon = (((A[:, None] * s - I) * m[None, None]) ** 2).sum((2, 3)) / (msum * N)
            Am = A * m
            tv_x = (Am[:, 1:] - Am[:, :-1]).abs() * m[None, 1:] * m[None, :-1]
            tv_y = (Am[1:, :] - Am[:-1, :]).abs() * m[None, 1:, :] * m[None, :-1, :]
            tv = (tv_x.sum((1, 2)) + tv_y.sum((1, 2))) / msum
            loss = recon + lam_tv * tv                           # [Bp]
            loss.sum().backward()
            opt.step()
            losses[:, it] = loss.detach().cpu().numpy()

        with torch.no_grad():
            a_grad = a_raw.grad.reshape(Bp, -1).norm(dim=1)
            c_grad = c.grad.reshape(Bp, -1).norm(dim=1)
            grad_norm = torch.sqrt(a_grad ** 2 + c_grad ** 2).cpu().numpy()
            final_loss = losses[:, -1]
            tail = losses[:, -50:]
            tail_range = tail.max(1) - tail.min(1)
            converged = tail_range < conv_tol
            success = converged & (grad_norm < grad_tol)
            A_hat = torch.nn.functional.softplus(a_raw)[:, 0].cpu().numpy()

        for b, (si, rs) in enumerate(sel):
            rec = dict(final_loss=float(final_loss[b]), restart=rs,
                       success=bool(success[b]), grad_norm=float(grad_norm[b]),
                       tail_range=float(tail_range[b]),
                       iters=int(max_iters), converged=bool(converged[b]),
                       A_hat=A_hat[b] * mask)
            if out[si] is None or rec["final_loss"] < out[si]["final_loss"]:
                out[si] = rec
    return out


def _sh_basis_torch_np(n):
    """与 physics/sh_torch 同构的 numpy SH 基（同序同常数）。"""
    x, y, z = n[:, 0], n[:, 1], n[:, 2]
    C0, C1 = 0.282095, 0.488603
    C2 = [1.092548, 1.092548, 0.315392, 1.092548, 0.546274]
    return np.stack([np.full_like(x, C0), C1 * y, C1 * z, C1 * x,
                     C2[0] * x * y, C2[1] * y * z, C2[2] * (3 * z * z - 1),
                     C2[3] * x * z, C2[4] * (x * x - y * y)], axis=-1)


def validate(data_root, scenes=2, subsets_per_scene=2, N=5, seed=7):
    """预注册验证门槛：批量 vs 串行 SI-MAE/final_loss 相对差 ≤ 1e-3。"""
    import time
    scene_dirs = sorted([os.path.join(data_root, d) for d in os.listdir(data_root)
                         if os.path.isfile(os.path.join(data_root, d, "sh_coeffs_irradiance.npy"))])
    rng = np.random.default_rng(seed)
    worst = 0.0
    for sd in scene_dirs[:scenes]:
        sc = load_scene(sd)
        subs = [sorted(rng.choice(sc["K"], N, replace=False).tolist())
                for _ in range(subsets_per_scene)]
        t0 = time.time()
        res_seq = [joint_solve(sc, s, restarts=3) for s in subs]
        t_seq = time.time() - t0
        t0 = time.time()
        res_bat = joint_solve_batched(sc, subs, restarts=3)
        t_bat = time.time() - t0
        for rseq, rbat, s in zip(res_seq, res_bat, subs):
            e_seq = si_mae_np(rseq["A_hat"], sc["albedo"], sc["mask"])
            e_bat = si_mae_np(rbat["A_hat"], sc["albedo"], sc["mask"])
            dl = abs(rseq["final_loss"] - rbat["final_loss"]) / max(abs(rseq["final_loss"]), 1e-12)
            de = abs(e_seq - e_bat) / max(abs(e_seq), 1e-12)
            worst = max(worst, dl, de)
            print(f"  {sc['name']} N={N} subset={s[:3]}...: si_mae seq={e_seq:.6f} bat={e_bat:.6f} "
                  f"(rel {de:.2e}) | loss rel {dl:.2e} | success {rseq['success']}/{rbat['success']}")
        print(f"  timing: seq {t_seq:.1f}s vs batched {t_bat:.1f}s "
              f"(speedup x{t_seq / max(t_bat, 1e-9):.1f})")
    print(f"WORST rel diff = {worst:.3e}  → {'PASS (≤1e-3)' if worst <= 1e-3 else 'FAIL'}")
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--data_root", default=os.path.join(_REPO, "p1", "calibration_set", "data_sun"))
    ap.add_argument("--scenes", type=int, default=2)
    ap.add_argument("--subsets", type=int, default=2)
    ap.add_argument("--N", type=int, default=5)
    args = ap.parse_args()
    if args.validate:
        validate(args.data_root, args.scenes, args.subsets, args.N)


if __name__ == "__main__":
    main()
