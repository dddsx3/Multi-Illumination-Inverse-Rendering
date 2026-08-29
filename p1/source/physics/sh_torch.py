"""P1 SH torch 版（仅在需要 torch 的脚本里 import）。"""
import torch
from sh import C0, C1, C2  # noqa: F401


def sh_basis_torch(n):
    x, y, z = n[..., 0], n[..., 1], n[..., 2]
    C0t, C1t = torch.tensor(C0, device=n.device), torch.tensor(C1, device=n.device)
    C2t = [torch.tensor(c, device=n.device) for c in C2]
    return torch.stack([
        C0t * torch.ones_like(x),
        C1t * y, C1t * z, C1t * x,
        C2t[0] * x * y, C2t[1] * y * z, C2t[2] * (3.0 * z * z - 1.0),
        C2t[3] * x * z, C2t[4] * (x * x - y * y),
    ], dim=-1)
