"""
分层残差建模模块
用于处理非朗伯效应（non-Lambertian effects）和异常值（outliers）

Author: Python Engineer
Date: 2026-01-19
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class GlobalResidualMLP(nn.Module):
    """
    全局残差MLP网络

    处理每个像素的特征（法线 + 球谐系数），生成平滑的残差图
    用于建模全局的非朗伯效应（如镜面反射、次表面散射等）

    Network Architecture:
        Input:  [12] - [normal(3) + SH coefficients(9)]
        Hidden: [32] - ReLU activation
        Hidden: [32] - ReLU activation
        Output: [1]  - residual value

    Args:
        input_dim: 输入特征维度，默认12（normal 3 + SH 9）
        hidden_dim: 隐藏层维度，默认32
        num_sh_coeffs: 球谐系数数量，默认9（二阶）
    """

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 32,
        num_sh_coeffs: int = 9
    ):
        super(GlobalResidualMLP, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_sh_coeffs = num_sh_coeffs

        # MLP网络
        self.mlp = nn.Sequential(
            # Layer 1: 12 -> 32
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),

            # Layer 2: 32 -> 32
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),

            # Layer 3: 32 -> 1
            nn.Linear(hidden_dim, 1)
        )

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用Xavier初始化
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    # 偏置初始化为0
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        normal: torch.Tensor,
        sh_coeffs: torch.Tensor,
        num_images: int
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            normal: 法线图 [B, 3, H, W]
            sh_coeffs: 球谐系数 [B, K, 9]
            num_images: 图像数量K（用于确定处理哪个光照）

        Returns:
            residual: 全局残差图 [B, K, H, W]
        """
        B, _, H, W = normal.shape
        K = sh_coeffs.shape[1]

        # 准备输出
        residual_maps = []

        # 对每个光照条件单独处理
        for k in range(K):
            # 提取当前光照的球谐系数 [B, 9]
            sh_k = sh_coeffs[:, k, :]

            # 扩展球谐系数到每个像素 [B, 9, H, W]
            sh_k_expanded = sh_k.unsqueeze(-1).unsqueeze(-1).expand(B, self.num_sh_coeffs, H, W)

            # 拼接法线和球谐系数 [B, 12, H, W]
            features = torch.cat([normal, sh_k_expanded], dim=1)

            # 转置为 [B, H, W, 12]
            features = features.permute(0, 2, 3, 1)

            # 展平为像素序列 [B*H*W, 12]
            features_flat = features.reshape(-1, self.input_dim)

            # 通过MLP [B*H*W, 1]
            residual_flat = self.mlp(features_flat)

            # 恢复为图像形状 [B, H, W, 1]
            residual = residual_flat.reshape(B, H, W, 1)

            # 转置为 [B, 1, H, W]
            residual = residual.permute(0, 3, 1, 2)

            residual_maps.append(residual)

        # 堆叠所有光照的残差 [B, K, H, W]
        residual_output = torch.cat(residual_maps, dim=1)

        return residual_output


class LocalResidualNet(nn.Module):
    """
    逐场景、逐光照的局部残差网络

    早期版本使用一个全数据集共享的 [1,1,H,W] 可学习缓冲参数（create_local_residual），
    该设计有三个问题：
    1. 与输入无关——网络在训练集上过拟合，无法泛化到新场景；
    2. 所有光照共用同一张残差图——无法表达随光照变化的非朗伯效应（如镜面高光）；
    3. 推理时依然生效——等同于把数据集级系统误差写死进模型。

    新设计以解码器特征 + 反照率 + 逐光照着色为条件，逐场景、逐光照地预测残差图。
    末层 1x1 卷积零初始化，保证训练初期残差从零开始，配合课程学习平滑引入。

    Args:
        feature_channels: 解码器特征通道数（IntrinsicUNet 末层为 base_channels）
        num_images: 光照数量 K
        hidden_channels: 隐藏层通道数
    """

    def __init__(self, feature_channels: int = 32, num_images: int = 5, hidden_channels: int = 64):
        super(LocalResidualNet, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(feature_channels + 1 + num_images, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, num_images, kernel_size=1),
        )

        # 零初始化末层：初始局部残差为全零，随训练逐渐学习
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

    def forward(
        self,
        features: torch.Tensor,
        albedo: torch.Tensor,
        shading: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            features: 解码器特征图 [B, C, H, W]
            albedo: 反照率图 [B, 1, H, W]
            shading: 着色图 [B, K, H, W]

        Returns:
            local_residual: 局部残差图 [B, K, H, W]
        """
        x = torch.cat([features, albedo, shading], dim=1)  # [B, C+1+K, H, W]
        return self.net(x)


class HierarchicalResidual(nn.Module):
    """
    分层残差建模模块

    组合全局残差MLP和局部残差网络，用于完整的非朗伯残差建模
    渲染方程：final_render = albedo * shading + r_global + r_local

    Args:
        use_local_residual: 是否使用局部残差网络（基于解码器特征逐场景预测）
        residual_scales: 各训练阶段的残差缩放因子，默认课程学习下
                         阶段1/2残差关闭（scale=0），阶段3开启（scale=1.0）
        num_images: 光照数量 K
        feature_channels: 解码器特征通道数
    """

    def __init__(
        self,
        use_local_residual: bool = True,
        residual_scales: dict = None,
        num_images: int = 5,
        feature_channels: int = 32,
        hidden_channels: int = 64
    ):
        super(HierarchicalResidual, self).__init__()

        self.use_local_residual = use_local_residual
        # 残差缩放因子：阶段1/2关闭残差，阶段3开启（配合课程学习）
        self.residual_scales = residual_scales or {'stage1': 0.0, 'stage2': 0.0, 'stage3': 1.0}

        # 全局残差MLP（逐像素：法线 + 球谐系数 -> 残差，随光照变化）
        self.global_residual = GlobalResidualMLP(
            input_dim=12,
            hidden_dim=32,
            num_sh_coeffs=9
        )

        # 局部残差网络（逐场景：解码器特征 + 反照率 + 着色 -> 残差）
        if use_local_residual:
            self.local_net = LocalResidualNet(
                feature_channels=feature_channels,
                num_images=num_images,
                hidden_channels=hidden_channels
            )
        else:
            self.local_net = None

    def forward(
        self,
        albedo: torch.Tensor,
        shading: torch.Tensor,
        normal: torch.Tensor,
        sh_coeffs: torch.Tensor,
        stage: str = 'stage3',
        features: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播：计算包含残差的最终渲染

        Args:
            albedo: 反照率图 [B, 1, H, W]
            shading: 着色图 [B, K, H, W]
            normal: 法线图 [B, 3, H, W]
            sh_coeffs: 球谐系数 [B, K, 9]
            stage: 训练阶段（'stage1', 'stage2', 或 'stage3'），默认为'stage3'
            features: 解码器特征图 [B, C, H, W]，用于逐场景预测局部残差；
                      为 None 时局部残差关闭（向后兼容）

        Returns:
            final_render: 最终渲染图 [B, K, H, W]
            global_residual: 缩放后的全局残差 [B, K, H, W]
            local_residual: 缩放后的局部残差 [B, K, H, W] 或 None
        """
        B, K, H, W = shading.shape

        # 1. 基础Lambertian渲染
        # albedo: [B, 1, H, W], shading: [B, K, H, W]
        lambertian = albedo * shading  # [B, K, H, W]

        # 2. 全局残差（随光照变化：以法线 + 当前光照的SH系数为条件）
        global_res = self.global_residual(normal, sh_coeffs, K)  # [B, K, H, W]

        # 3. 局部残差（逐场景、逐光照：以解码器特征 + 反照率 + 着色为条件）
        local_res = None
        if self.local_net is not None and features is not None:
            local_res = self.local_net(features, albedo, shading)  # [B, K, H, W]

        # 4. 根据阶段获取缩放因子
        scale_factor = self.residual_scales.get(stage, self.residual_scales['stage3'])

        # 5. 对残差进行缩放
        scaled_global_res = global_res * scale_factor
        scaled_local_res = local_res * scale_factor if local_res is not None else None

        # 6. 计算最终渲染
        if scaled_local_res is not None:
            final_render = lambertian + scaled_global_res + scaled_local_res
        else:
            final_render = lambertian + scaled_global_res
        return final_render, scaled_global_res, scaled_local_res



# ============================================================================
# 测试代码（Phase 0 精简版：针对逐场景逐光照局部残差网络）
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Hierarchical Residual 模块测试")
    print("=" * 80)

    B, K, H, W = 2, 5, 256, 256
    C = 32

    normal = torch.randn(B, 3, H, W)
    normal = F.normalize(normal, p=2, dim=1)
    sh_coeffs = torch.randn(B, K, 9) * 0.1
    albedo = torch.rand(B, 1, H, W)
    shading = torch.rand(B, K, H, W)
    features = torch.randn(B, C, H, W)

    # 1. LocalResidualNet：初始输出应为全零（末层零初始化）
    print("\n【1】LocalResidualNet 零初始化检查")
    local_net = LocalResidualNet(feature_channels=C, num_images=K)
    local_res = local_net(features, albedo, shading)
    assert local_res.shape == (B, K, H, W), f"形状错误: {local_res.shape}"
    assert torch.allclose(local_res, torch.zeros_like(local_res), atol=1e-6), "初始局部残差应为全零"
    print(f"  输出形状: {local_res.shape} ✓")
    print(f"  初始输出全零 ✓")

    # 2. HierarchicalResidual：stage1 残差关闭，stage3 残差开启
    print("\n【2】课程学习阶段缩放检查")
    module = HierarchicalResidual(num_images=K, feature_channels=C)
    print(f"  残差缩放因子: {module.residual_scales}")

    final_s1, global_s1, local_s1 = module(albedo, shading, normal, sh_coeffs, stage='stage1', features=features)
    lambertian = albedo * shading
    assert torch.allclose(final_s1, lambertian, atol=1e-6), "stage1 时最终渲染应等于纯 Lambertian"
    print(f"  stage1: final == lambertian ✓ (残差贡献为 0)")

    final_s3, global_s3, local_s3 = module(albedo, shading, normal, sh_coeffs, stage='stage3', features=features)
    assert local_s3 is not None and local_s3.shape == (B, K, H, W)
    assert not torch.allclose(final_s3, lambertian, atol=1e-6), "stage3 时残差应开始贡献"
    print(f"  stage3: local_residual 形状 {local_s3.shape} ✓, global 形状 {global_s3.shape} ✓")

    # 3. features=None 时局部残差关闭（向后兼容）
    print("\n【3】向后兼容：features=None")
    final_no_f, global_no_f, local_no_f = module(albedo, shading, normal, sh_coeffs)
    assert local_no_f is None, "features=None 时局部残差应为 None"
    print(f"  local_residual = None ✓")

    # 4. 梯度传播：局部残差网络参数可学习
    print("\n【4】梯度传播检查")
    module.train()
    loss = final_s3.mean()
    loss.backward()
    local_grad_ok = any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.local_net.parameters())
    global_grad_ok = any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.global_residual.parameters())
    assert local_grad_ok, "局部残差网络参数未收到梯度"
    assert global_grad_ok, "全局残差MLP参数未收到梯度"
    print(f"  局部残差网络梯度: {'✓' if local_grad_ok else '✗'}")
    print(f"  全局残差MLP梯度: {'✓' if global_grad_ok else '✗'}")

    # 5. 多分辨率支持
    print("\n【5】多分辨率支持")
    for size in [128, 256, 512]:
        h, w = size, size
        f = torch.randn(1, C, h, w)
        a = torch.rand(1, 1, h, w)
        s = torch.rand(1, K, h, w)
        n = torch.randn(1, 3, h, w)
        n = F.normalize(n, p=2, dim=1)
        with torch.no_grad():
            out, _, _ = module(a, s, n, sh_coeffs[:1], features=f)
        assert out.shape == (1, K, h, w), f"尺寸 {size} 输出形状错误: {out.shape}"
        print(f"  输入 {size}x{size} -> 输出 {tuple(out.shape)} ✓")

    print("\n全部测试通过 ✓")
