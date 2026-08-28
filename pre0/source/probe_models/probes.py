"""PRE-03 · 三个最小 Probe Model（同 encoder / 同 decoder / 同预算，只改集合融合）。

设计纪律（任务书 §5）：
  - 相同 per-image encoder、相同 decoder、相同参数量级（~2M）；
  - 不用 residual / FiLM / attention / 高级 backbone；
  - 只改变集合融合方式；
  - 输出：canonical albedo(1ch)、geometry(depth 1ch)、per-light lighting(9/light)；
  - appearance residual 全关。

Probe-A MeanSpatial     F_set = mean_k F_k              （最重要基线）
Probe-B MeanVarSpatial  decoder 输入 = 1x1conv([μ_F, σ_F]) （光照间变化是否含线索）
Probe-C GlobalSet       z = mean_k GAP(F_k)，空间广播后进 decoder（全局聚合是否
                        丢失 pixel-aligned 光度证据）

域约定：全部在 linear 域（精确 sRGB 反变换解码）——与 GT albedo / SH /
physics renderer 同域；训练渲染器 DepthToNormal(use_edge_aware=False)
与数据生成定义对齐（见 pre0/protocol/DATASET_CONTRACT.md）。
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, _REPO)


class PerImageEncoder(nn.Module):
    """1ch 256×256 -> C ch 64×64（1/4 分辨率，保空间结构供 Probe-A/B）"""

    def __init__(self, c: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 48, 3, padding=1), nn.SiLU(),
            nn.Conv2d(48, 64, 3, padding=1), nn.SiLU(),
            nn.Conv2d(64, 96, 4, stride=2, padding=1), nn.SiLU(),   # 128
            nn.Conv2d(96, 96, 3, padding=1), nn.SiLU(),
            nn.Conv2d(96, 96, 4, stride=2, padding=1), nn.SiLU(),   # 64
            nn.Conv2d(96, c, 3, padding=1), nn.SiLU(),
        )

    def forward(self, x):            # x: [B*N,1,H,W]
        return self.net(x)           # [B*N,C,H/4,W/4]


class SharedDecoder(nn.Module):
    """C ch 64×64 -> (depth 256², albedo 256²)。三个 probe 严格共享同一模块。"""

    def __init__(self, c: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, 96, 3, padding=1), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),            # 128
            nn.Conv2d(96, 96, 3, padding=1), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),            # 256
            nn.Conv2d(96, 96, 3, padding=1), nn.SiLU(),
        )
        self.head_depth = nn.Conv2d(64, 1, 3, padding=1)
        self.head_albedo = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, f):            # [B,C,h,w] -> (depth [B,1,H,W], albedo [B,1,H,W] sigmoid)
        h = self.net(f)
        depth = self.head_depth(h)
        depth = F.softplus(depth - 3.0) + 0.05   # 正深度，初始 ~0.2-0.5
        albedo = torch.sigmoid(self.head_albedo(h))
        return depth, albedo


class LightingHead(nn.Module):
    """逐图 GAP 特征 -> 9 维 SH 系数（signed）。"""

    def __init__(self, c: int = 96):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(c, 192), nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(192, 9),
        )

    def forward(self, feats):        # [B,N,C,h,w] -> [B,N,9]
        z = feats.mean(dim=(3, 4))               # GAP over 空间维
        return self.fc(z)


class ProbeBase(nn.Module):
    def __init__(self, c: int = 96):
        super().__init__()
        self.encoder = PerImageEncoder(c)
        self.decoder = SharedDecoder(c)
        self.light_head = LightingHead(c)
        self.c = c

    def _aggregate(self, feats_bnc):                     # [B,N,C,h,w]
        raise NotImplementedError

    def forward(self, images):                           # [B,N,1,H,W]
        B, N = images.shape[:2]
        feats = self.encoder(images.flatten(0, 1))       # [B*N,C,h,w]
        feats = feats.reshape(B, N, *feats.shape[1:])
        f_set = self._aggregate(feats)                   # [B,C,h,w]
        depth, albedo = self.decoder(f_set)
        sh = self.light_head(feats)                      # [B,N,9]
        return depth, albedo, sh


class ProbeA(ProbeBase):
    """MeanSpatial：F_set = mean_k F_k"""

    def _aggregate(self, feats_bnc):
        return feats_bnc.mean(1)


class ProbeB(ProbeBase):
    """MeanVarSpatial：1x1 conv([μ,σ]) -> C（唯一附加层，参数计入预算）"""

    def __init__(self, c: int = 96):
        super().__init__(c)
        self.fuse = nn.Conv2d(2 * c, c, 1)

    def _aggregate(self, feats_bnc):
        mu = feats_bnc.mean(1)
        var = feats_bnc.var(1, unbiased=False)
        return self.fuse(torch.cat([mu, torch.sqrt(var + 1e-6)], dim=1))


class ProbeC(ProbeBase):
    """GlobalSet：z = mean_k GAP(F_k)，空间广播 -> decoder"""

    def _aggregate(self, feats_bnc):
        z = feats_bnc.flatten(2).mean(-1).mean(1)        # [B,C]
        return z[..., None, None].expand(-1, -1, *feats_bnc.shape[-2:])


PROBES = {"A": ProbeA, "B": ProbeB, "C": ProbeC}


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
