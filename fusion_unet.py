"""
光照数量无关的自适应多光照逆渲染网络（T2.2 核心创新）

架构（设计冻结于 docs/design/t2_2_design.md）：
  1. Per-light stem：共享权重逐图编码（任意 N，参数量与 N 无关）
  2. 置换不变集合聚合：GAP tokens -> 无位置编码自注意力 -> PMA(seed) 池化
     （数学保证：无 PE 的自注意力置换等变 + 对称池化 = 置换不变）
  3. FiLM 条件化：z -> (gamma, beta) 调制 U-Net bottleneck
     （gamma 初始化 1、beta 初始化 0 => 初始等价无条件网络）
  4. S2 逐光照反照率：共享主反照率 + 有界残差 DeltaA_k = 0.1*tanh(conv(...))
  5. RGB/灰度双链路：stem in_channels 参数化

置换不变性：forward 与 N 及光照顺序无关（tests/test_permutation_invariance.py 验证）。

外部参考（差异化声明见设计文档 §2）：
  PS-FCN (arXiv:1807.08696)：concat 融合非置换不变、N 固定、仅法线输出；
  Deep Sets (arXiv:1703.06114)、Set Transformer (arXiv:1810.00825)：理论工具箱。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from unet_model import DownBlock, UpBlock


class LightStem(nn.Module):
    """共享权重逐图特征提取（对每张输入图独立作用）。"""

    def __init__(self, in_channels: int, out_channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x: [B*N, C, H, W] 或 [B, C, H, W]
        return self.net(x)


class SetTransformerLite(nn.Module):
    """轻量置换不变集合聚合。

    无位置编码的自注意力是置换等变的；PMA 以单一可学习 seed query 对全部
    token 做注意力加权求和（对称操作），复合后为集合的不变函数。
    （Set Transformer, ICML 2019 —— 理论依据见其 §3）
    """

    def __init__(self, in_dim: int = 32, dim: int = 128, heads: int = 4):
        super().__init__()
        self.proj = nn.Linear(in_dim, dim)
        self.mha = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.seed = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.pma = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, N, 32] -> z: [B, dim]
        t = self.proj(tokens)
        attn_out, _ = self.mha(t, t, t)
        q = self.seed.expand(attn_out.shape[0], -1, -1)
        z, _ = self.pma(q, attn_out, attn_out)
        return self.norm(z.squeeze(1))


class FusionUNet(nn.Module):
    """
    光照数量无关的自适应多光照逆渲染网络。

    Input : x [B, N, C_in, H, W]，C_in ∈ {1, 3}；N 训练期固定、推理期任意。
    Output: depth [B,1,H,W], albedo_main [B,1,H,W],
            sh_coeffs [B,N,9], weight_map [B,1,H,W],
            features [B,base,H,W],
            albedo_per_light [B,N,1,H,W] = clamp(main + DeltaA_k, 0, 2)

    网络体：与 IntrinsicUNet 相同的 Down/UpBlock 拓扑，但首层接受 stem 的
    base_channels 通道（而非 K 通道堆叠），bottleneck 经 FiLM(z) 调制。
    """

    def __init__(self, num_images: int = 5, in_channels: int = 1,
                 base_channels: int = 32, sh_order: int = 2,
                 fusion_dim: int = 128, delta_bound: float = 0.1,
                 use_per_light_albedo: bool = True,
                 sh_constraint: str = "clamp"):
        super().__init__()
        self.num_images = num_images          # 仅用于 SH 头输出形状；前向不依赖
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.sh_coeffs_dim = (sh_order + 1) ** 2
        self.delta_bound = delta_bound
        bc = base_channels

        self.stem = LightStem(in_channels, bc)

        # 主干（首层接受 stem 输出通道）
        self.down1 = DownBlock(bc, bc * 2)
        self.down2 = DownBlock(bc * 2, bc * 4)
        self.down3 = DownBlock(bc * 4, bc * 8)
        self.down4 = DownBlock(bc * 8, bc * 16)
        self.bottleneck = DownBlock(bc * 16, bc * 16, use_pooling=False)

        self.up1 = UpBlock(bc * 16, bc * 16, bc * 8)
        self.up2 = UpBlock(bc * 8, bc * 8, bc * 4)
        self.up3 = UpBlock(bc * 4, bc * 4, bc * 2)
        self.up4 = UpBlock(bc * 2, bc * 2, bc)

        # SH 头（与基线一致：GAP+FC）
        # T2.2 修正：SH 逐光照共享预测（每光照 9 系数）——任意 N 可用；
        # 原固定宽度头(num_images*9)与 N 无关目标冲突
        self.sh_fc = nn.Sequential(
            nn.Linear(bc, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.2), nn.Linear(256, self.sh_coeffs_dim),
            nn.Tanh())

        # 输出头
        def head():
            return nn.Sequential(
                nn.Conv2d(bc, bc // 2, 3, padding=1),
                nn.BatchNorm2d(bc // 2), nn.ReLU(inplace=True),
                nn.Conv2d(bc // 2, 1, 1))
        self.depth_head = head()
        self.albedo_head = head()          # 主反照率（共享）
        self.weight_head = nn.Sequential(
            nn.Conv2d(bc, bc // 2, 3, padding=1),
            nn.BatchNorm2d(bc // 2), nn.ReLU(inplace=True),
            nn.Conv2d(bc // 2, 1, 1), nn.Sigmoid())

        # 集合聚合 + FiLM
        self.aggregator = SetTransformerLite(in_dim=bc, dim=fusion_dim)
        self.film_gamma = nn.Linear(fusion_dim, bc * 16)
        self.film_beta = nn.Linear(fusion_dim, bc * 16)
        nn.init.ones_(self.film_gamma.weight)
        nn.init.zeros_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)

        # S2 逐光照反照率分支（有界残差）；F-albOff 时整体禁用
        self.use_per_light_albedo = use_per_light_albedo
        self.sh_constraint = sh_constraint if sh_constraint in ("clamp", "softplus") else "clamp"
        self.delta_head = nn.Sequential(
            nn.Conv2d(bc + 1, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1)) if use_per_light_albedo else None

        self._initialize_weights()
        self.delta_head_last = self.delta_head[-1] if self.delta_head else None

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def split_input(x: torch.Tensor):
        """[B,N,C,H,W] 或 [B,N*C,H,W] -> [B,N,C,H,W]"""
        if x.dim() == 5:
            return x
        B, KC, H, W = x.shape
        assert KC % 3 == 0 or KC % 1 == 0
        raise ValueError("扁平输入需提供 C_in；请传 [B,N,C,H,W]")

    def forward(self, x: torch.Tensor):
        """x: [B, N, C_in, H, W]；灰度模态可传 [B, N, H, W]（自动升维 C=1）。
        返回五元组 + 逐光照反照率。"""
        if x.dim() == 4:
            x = x.unsqueeze(2)
        assert x.dim() == 5, f"期望 [B,N,C,H,W]，收到 {tuple(x.shape)}"
        B, N, C_in, H, W = x.shape

        xf = x.reshape(B * N, C_in, H, W)
        f = self.stem(xf)                                   # [B*N, bc, H, W]

        tokens = F.adaptive_avg_pool2d(f, 1).reshape(B, N, -1)   # [B,N,bc]
        z = self.aggregator(tokens)                          # [B, fusion_dim]
        gamma = self.film_gamma(z).reshape(B, -1, 1, 1)
        beta = self.film_beta(z).reshape(B, -1, 1, 1)

        # U-Net 主干
        e1, skip1 = self.down1(f.reshape(B * N, -1, H, W))   # [B*N, bc*2, ..]
        e2, skip2 = self.down2(e1)
        e3, skip3 = self.down3(e2)
        e4, skip4 = self.down4(e3)
        bn_, skip_bn = self.bottleneck(e4)
        # FiLM（零初始化等价）：z 为场景级条件，同一场景的 N 个逐光照样本
        # 共享同一组调制参数（repeat_interleave 与 token 顺序 b0l0..b0lN 一致）
        g = self.film_gamma(z).unsqueeze(1).expand(B, N, -1)
        bta = self.film_beta(z).unsqueeze(1).expand(B, N, -1)
        g = g.reshape(B * N, -1, 1, 1)
        bta = bta.reshape(B * N, -1, 1, 1)
        bn_ = bn_ + g * bn_ + bta

        # 解码（batch 维保持 B*N 以共享权重；skip 同形）
        d = self.up1(bn_, skip4)
        d = self.up2(d, skip3)
        d = self.up3(d, skip2)
        feats = self.up4(d, skip1)                           # [B*N, bc, H, W]

        feats = feats.reshape(B, N, -1, H, W)

        # SH 头：逐光照全局描述子 -> 每光照 SH（输入为 stem/decoder 的 bc 维）
        sh_fc_in = feats.flatten(0, 1).mean(dim=(2, 3))          # [B*N, bc]
        raw = self.sh_fc(sh_fc_in).reshape(B, N, -1)             # [B,N,9] tanh 域 [-1,1]

        # SH[0] 物理约束变体（T2.3）：
        #   clamp    —— 与基线一致：截断到 [0,1]（hack 语义）
        #   softplus —— 可微重参数化：SH0 = softplus(x)，天然 >=0 且梯度恒正
        if self.sh_constraint == "softplus":
            sh0 = F.softplus(raw[..., 0:1])
            sh_coeffs = torch.cat([sh0, raw[..., 1:]], dim=-1)
        else:
            sh_coeffs = torch.cat([torch.clamp(raw[..., 0:1], 0.0, 1.0),
                                   raw[..., 1:]], dim=-1)

        # 主头作用于逐光照特征后再均值池化为共享输出
        depth = self.depth_head(feats.flatten(0, 1)).reshape(B, N, 1, H, W).mean(dim=1)
        albedo_main = self.albedo_head(
            feats.flatten(0, 1)).reshape(B, N, 1, H, W).mean(dim=1)
        weight_map = self.weight_head(
            feats.flatten(0, 1)).reshape(B, N, 1, H, W).mean(dim=1)

        features = feats.mean(dim=1)                         # [B, bc, H, W]

        # S2 逐光照反照率：DeltaA_k = bound*tanh(conv(cat[f_k, A_main]))
        # F-albOff：分支整体禁用，返回五元组（训练器按 5 元组走基线路径）
        if not self.use_per_light_albedo:
            return (depth, albedo_main, sh_coeffs, weight_map, features)
        a_exp = albedo_main.unsqueeze(1).expand(-1, N, -1, -1, -1)
        da_in = torch.cat([feats, a_exp], dim=2).flatten(0, 1)
        delta = self.delta_bound * torch.tanh(self.delta_head(da_in))
        albedo_per_light = torch.clamp(
            a_exp + delta.reshape(B, N, 1, H, W), 0.0, 2.0)

        return (depth, albedo_main, sh_coeffs, weight_map, features,
                albedo_per_light)
