"""
T2.2 / G2.2 置换不变性测试（门禁核心：任一失败即 FAIL）

运行: python tests/test_permutation_invariance.py

用例:
  P1 置换不变性: N∈{3,5,7,10} × 各10组随机打乱，
     输出差异 <1e-5（depth/albedo/wm/features 直接比较；
     sh 按列还原后比较）
  P3 N=1 前向路径成立
  P4 N∈{1,3,5,7,10} 前向+反向冒烟（损失与梯度有限）+ 参数量统计

诚实记录（任务书允许）:
  - 当前实现按 N 分桶推理，无 pad 路径 => 不做 pad/mask 用例；
    重复光照在注意力下数学上会改变输出（非不变），以分桶协议规避。
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fusion_unet import FusionUNet

TOL = 1e-5


def make_model(seed=0):
    torch.manual_seed(seed)
    m = FusionUNet(num_images=10, in_channels=1, base_channels=16,
                   fusion_dim=64)
    m.eval()
    return m


@torch.no_grad()
def p1_permutation_invariance():
    model = make_model()
    worst = 0.0
    for N in (3, 5, 7, 10):
        torch.manual_seed(100 + N)
        for trial in range(10):
            x = torch.rand(1, N, 1, 64, 64)
            ref = model(x)
            perm = torch.randperm(N)
            out = model(x[:, perm])
            inv = torch.argsort(perm)   # 逆排列（perm 非对合时 out[:,perm] 不等于还原）
            # 还原：逐光照输出按逆排列回到原顺序；空间输出直接比较
            d_sh = (out[2][:, inv] - ref[2]).abs().max().item()
            d_pl = (out[5][:, inv] - ref[5]).abs().max().item()
            d_spatial = max((a - b).abs().max().item()
                            for a, b in zip(ref[:2] + ref[3:5],
                                            out[:2] + out[3:5]))
            dmax = max(d_sh, d_pl, d_spatial)
            worst = max(worst, dmax)
            assert dmax < TOL, f"N={N} trial={trial}: diff={dmax:.2e} >= 1e-5"
    print(f"P1 PASS: 置换不变性 max_diff={worst:.2e} (<{TOL})")


@torch.no_grad()
def p3_n1_forward():
    model = make_model(seed=1)
    x = torch.rand(1, 1, 1, 64, 64)
    out = model(x)
    shapes = [tuple(t.shape) for t in out]
    assert shapes[0] == (1, 1, 64, 64), shapes[0]
    assert shapes[2] == (1, 1, 9), shapes[2]
    assert all(torch.isfinite(t).all() for t in out), "N=1 输出含非有限值"
    print(f"P3 PASS: N=1 前向成立, shapes={shapes}")


def p4_fwd_bwd_smoke():
    model = make_model(seed=2)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    for N in (1, 3, 5, 7, 10):
        x = torch.rand(1, N, 1, 64, 64)
        t0 = time.time()
        out = model(x)
        loss = sum(t.abs().mean() for t in out[:4]) \
             + out[2].abs().mean() + out[5].abs().mean()
        opt.zero_grad()
        loss.backward()
        gnorm = torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters()
                               if p.grad is not None))
        dt = time.time() - t0
        assert torch.isfinite(loss), f"N={N} loss 非有限"
        assert torch.isfinite(gnorm), f"N={N} 梯度非有限"
        print(f"P4 N={N}: loss={loss.item():.4f} gnorm={gnorm:.3f} "
              f"fwd+bwd={dt * 1000:.0f}ms")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"P4 PASS: 全部 N 前向反向有限 | FusionUNet 参数量={n_params:,}")


if __name__ == "__main__":
    import time
    p1_permutation_invariance()
    p3_n1_forward()
    p4_fwd_bwd_smoke()
    print("ALL PERMUTATION TESTS PASSED")
