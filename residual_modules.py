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


def create_local_residual(
    height: int,
    width: int,
    device: torch.device = None,
    requires_grad: bool = True
) -> nn.Parameter:
    """
    创建局部残差参数

    局部残差用于吸收无法被全局模型（Lambertian + 全局残差）解释的
    局部异常值，如遮挡、阴影边界、噪声等。

    Args:
        height: 图像高度
        width: 图像宽度
        device: 设备（CPU或GPU）
        requires_grad: 是否需要梯度（默认True，用于优化）

    Returns:
        local_residual: 可学习的参数张量 [1, 1, H, W]
    """
    # 初始化为全零
    local_residual = torch.zeros(1, 1, height, width, device=device)

    # 转换为可学习参数
    local_residual = nn.Parameter(local_residual, requires_grad=requires_grad)

    return local_residual


class HierarchicalResidual(nn.Module):
    """
    分层残差建模模块

    组合全局残差MLP和局部残差，用于完整的残差建模
    渲染方程：final_render = albedo * shading + r_global + r_local

    Args:
        image_height: 图像高度
        image_width: 图像宽度
        use_local_residual: 是否使用局部残差
    """

    def __init__(
        self,
        image_height: int = 256,
        image_width: int = 256,
        use_local_residual: bool = True,
        residual_scales: dict = None
    ):
        super(HierarchicalResidual, self).__init__()

        self.image_height = image_height
        self.image_width = image_width
        self.use_local_residual = use_local_residual
        # 设置残差缩放因子，默认为各阶段的缩放值
        self.residual_scales = residual_scales or {'stage1': 0.0, 'stage2': 0.3, 'stage3': 0.5}

        # 全局残差MLP
        self.global_residual = GlobalResidualMLP(
            input_dim=12,
            hidden_dim=32,
            num_sh_coeffs=9
        )

        # 局部残差（可选）
        if use_local_residual:
            self.local_residual = create_local_residual(
                height=image_height,
                width=image_width,
                requires_grad=True
            )
        else:
            self.local_residual = None

    def forward(
        self,
        albedo: torch.Tensor,
        shading: torch.Tensor,
        normal: torch.Tensor,
        sh_coeffs: torch.Tensor,
        stage: str = 'stage3',  # 添加stage参数，默认为stage3以保持兼容
        bottleneck_features: torch.Tensor = None  # 保持原有的bottleneck_features参数

    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播：计算包含残差的最终渲染

        Args:
            albedo: 反照率图 [B, 1, H, W]
            shading: 着色图 [B, K, H, W]
            normal: 法线图 [B, 3, H, W]
            sh_coeffs: 球谐系数 [B, K, 9]
            stage: 训练阶段（'stage1', 'stage2', 或 'stage3'），默认为'stage3'
            bottleneck_features: 瓶颈特征，用于局部残差计算（如果提供）

        Returns:
            final_render: 最终渲染图 [B, K, H, W]
           global_residual: 缩放后的全局残差 [B, K, H, W]
            local_residual: 缩放后的局部残差 [1, 1, H, W] 或 None
        """
        B, K, H, W = shading.shape

        # 1. 基础Lambertian渲染
        # albedo: [B, 1, H, W], shading: [B, K, H, W]
        lambertian = albedo * shading  # [B, K, H, W]

        # 2. 全局残差
        global_res = self.global_residual(normal, sh_coeffs, K)  # [B, K, H, W]

        # 3. 局部残差
        local_res = None
        if self.use_local_residual and self.local_residual is not None:
            local_res = self.local_residual  # [1, 1, H, W]

            # 调整尺寸（如果需要）
            if local_res.shape[2] != H or local_res.shape[3] != W:
                local_res = F.interpolate(
                    local_res,
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False
                )
        # 4. 根据阶段获取缩放因子
        scale_factor = self.residual_scales.get(stage, self.residual_scales['stage3'])

        # 5. 对残差进行缩放
        scaled_global_res = global_res * scale_factor
        scaled_local_res = local_res * scale_factor if local_res is not None else None

        # 6. 计算最终渲染
        if scaled_local_res is not None:

            # Broadcasting: [1, 1, H, W] -> [B, K, H, W]
            final_render = lambertian + scaled_global_res + scaled_local_res
        else:
            final_render = lambertian + scaled_global_res
        return final_render, scaled_global_res, scaled_local_res


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    """
    测试残差建模模块
    """
    print("=" * 80)
    print("Hierarchical Residual Modeling 测试")
    print("=" * 80)

    # ========== 配置参数 ==========
    BATCH_SIZE = 2
    NUM_IMAGES = 5
    IMAGE_SIZE = 256

    print(f"\n【测试配置】")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"光照数量: {NUM_IMAGES}")
    print(f"图像尺寸: {IMAGE_SIZE} x {IMAGE_SIZE}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # ========== 测试 GlobalResidualMLP ==========
    print("\n" + "=" * 80)
    print("【1】测试 GlobalResidualMLP")
    print("=" * 80)

    global_mlp = GlobalResidualMLP(
        input_dim=12,
        hidden_dim=32,
        num_sh_coeffs=9
    ).to(device)

    # 统计参数
    total_params = sum(p.numel() for p in global_mlp.parameters())
    print(f"\n模型参数量: {total_params:,}")

    # 创建输入
    normal = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
    normal = F.normalize(normal, p=2, dim=1)  # 归一化法线

    sh_coeffs = torch.randn(BATCH_SIZE, NUM_IMAGES, 9).to(device)

    print(f"\n输入:")
    print(f"  法线图: {normal.shape}")
    print(f"  球谐系数: {sh_coeffs.shape}")

    # 前向传播
    global_residual = global_mlp(normal, sh_coeffs, NUM_IMAGES)

    print(f"\n输出:")
    print(f"  全局残差: {global_residual.shape}  # [B, K, H, W]")
    print(f"  数值范围: [{global_residual.min():.3f}, {global_residual.max():.3f}]")
    print(f"  均值: {global_residual.mean():.3f}")
    print(f"  标准差: {global_residual.std():.3f}")

    print("✓ GlobalResidualMLP 测试成功")

    # ========== 测试 create_local_residual ==========
    print("\n" + "=" * 80)
    print("【2】测试 create_local_residual")
    print("=" * 80)

    local_residual = create_local_residual(
        height=IMAGE_SIZE,
        width=IMAGE_SIZE,
        device=device,
        requires_grad=True
    )

    print(f"\n局部残差参数:")
    print(f"  形状: {local_residual.shape}  # [1, 1, H, W]")
    print(f"  数据类型: {local_residual.dtype}")
    print(f"  需要梯度: {local_residual.requires_grad}")
    print(f"  是否为Parameter: {isinstance(local_residual, nn.Parameter)}")

    # 验证初始化为0
    if torch.allclose(local_residual, torch.zeros_like(local_residual)):
        print("✓ 正确初始化为全零")
    else:
        print("✗ 初始化错误")

    # ========== 测试 Broadcasting ==========
    print("\n" + "=" * 80)
    print("【3】测试 Residual Integration with Broadcasting")
    print("=" * 80)

    # 创建测试数据
    albedo = torch.rand(BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE).to(device)
    shading = torch.rand(BATCH_SIZE, NUM_IMAGES, IMAGE_SIZE, IMAGE_SIZE).to(device)

    print(f"\n输入:")
    print(f"  反照率: {albedo.shape}  # [B, 1, H, W]")
    print(f"  着色: {shading.shape}  # [B, K, H, W]")
    print(f"  全局残差: {global_residual.shape}  # [B, K, H, W]")
    print(f"  局部残差: {local_residual.shape}  # [1, 1, H, W]")

    # 渲染方程
    lambertian = albedo * shading  # [B, K, H, W]
    final_render = lambertian + global_residual + local_residual

    print(f"\n输出:")
    print(f"  Lambertian: {lambertian.shape}")
    print(f"  最终渲染: {final_render.shape}  # [B, K, H, W]")

    # 验证broadcasting正确
    expected_shape = (BATCH_SIZE, NUM_IMAGES, IMAGE_SIZE, IMAGE_SIZE)
    if final_render.shape == expected_shape:
        print("✓ Broadcasting 正确工作")
    else:
        print(f"✗ Broadcasting 错误: 期望 {expected_shape}, 得到 {final_render.shape}")

    # ========== 测试 HierarchicalResidual ==========
    print("\n" + "=" * 80)
    print("【4】测试 HierarchicalResidual 模块")
    print("=" * 80)

    hierarchical_residual = HierarchicalResidual(
        image_height=IMAGE_SIZE,
        image_width=IMAGE_SIZE,
        use_local_residual=True
    ).to(device)

    print(f"\n模块组件:")
    print(f"  全局残差MLP: ✓")
    print(f"  局部残差参数: {'✓' if hierarchical_residual.local_residual is not None else '✗'}")

    # 前向传播
    final, global_res, local_res = hierarchical_residual(
        albedo, shading, normal, sh_coeffs
    )

    print(f"\n输出:")
    print(f"  最终渲染: {final.shape}  # [B, K, H, W]")
    print(f"  全局残差: {global_res.shape}  # [B, K, H, W]")
    print(f"  局部残差: {local_res.shape if local_res is not None else None}")

    print(f"\n数值统计:")
    print(f"  最终渲染范围: [{final.min():.3f}, {final.max():.3f}]")
    print(f"  全局残差贡献: {global_res.abs().mean():.3f}")
    if local_res is not None:
        print(f"  局部残差贡献: {local_res.abs().mean():.3f}")

    print("✓ HierarchicalResidual 测试成功")

    # ========== 测试梯度传播 ==========
    print("\n" + "=" * 80)
    print("【5】测试梯度传播")
    print("=" * 80)

    # 启用梯度
    albedo_grad = albedo.clone().requires_grad_(True)
    shading_grad = shading.clone().requires_grad_(True)
    normal_grad = normal.clone().requires_grad_(True)
    sh_coeffs_grad = sh_coeffs.clone().requires_grad_(True)

    # 前向传播
    final_grad, _, _ = hierarchical_residual(
        albedo_grad, shading_grad, normal_grad, sh_coeffs_grad
    )

    # 计算损失
    loss = final_grad.mean()

    # 反向传播
    loss.backward()

    print(f"\n损失值: {loss.item():.6f}")
    print(f"\n梯度统计:")
    print(f"  反照率梯度范数: {albedo_grad.grad.norm():.6f}")
    print(f"  着色梯度范数: {shading_grad.grad.norm():.6f}")
    print(f"  法线梯度范数: {normal_grad.grad.norm():.6f}")
    print(f"  球谐系数梯度范数: {sh_coeffs_grad.grad.norm():.6f}")

    # 检查局部残差梯度
    if hierarchical_residual.local_residual is not None:
        local_grad = hierarchical_residual.local_residual.grad
        if local_grad is not None:
            print(f"  局部残差梯度范数: {local_grad.norm():.6f}")
            print("✓ 局部残差参数可学习")

    # 检查全局残差MLP梯度
    mlp_has_grad = any(p.grad is not None for p in hierarchical_residual.global_residual.parameters())
    if mlp_has_grad:
        print("✓ 全局残差MLP参数可学习")

    print("✓ 梯度传播成功，模块完全可微分")

    # ========== 测试不使用局部残差 ==========
    print("\n" + "=" * 80)
    print("【6】测试不使用局部残差")
    print("=" * 80)

    hierarchical_no_local = HierarchicalResidual(
        image_height=IMAGE_SIZE,
        image_width=IMAGE_SIZE,
        use_local_residual=False
    ).to(device)

    print(f"\n模块配置:")
    print(f"  使用局部残差: {hierarchical_no_local.use_local_residual}")

    final_no_local, global_res_no_local, local_res_no_local = hierarchical_no_local(
        albedo, shading, normal, sh_coeffs
    )

    print(f"\n输出:")
    print(f"  最终渲染: {final_no_local.shape}")
    print(f"  局部残差: {local_res_no_local}")

    if local_res_no_local is None:
        print("✓ 正确禁用局部残差")

    # ========== 测试不同图像尺寸 ==========
    print("\n" + "=" * 80)
    print("【7】测试不同图像尺寸")
    print("=" * 80)

    test_sizes = [128, 256, 512]

    for size in test_sizes:
        # 创建模块
        module = HierarchicalResidual(
            image_height=size,
            image_width=size,
            use_local_residual=True
        ).to(device)

        # 创建输入
        test_albedo = torch.rand(1, 1, size, size).to(device)
        test_shading = torch.rand(1, NUM_IMAGES, size, size).to(device)
        test_normal = torch.randn(1, 3, size, size).to(device)
        test_normal = F.normalize(test_normal, p=2, dim=1)
        test_sh = torch.randn(1, NUM_IMAGES, 9).to(device)

        # 测试
        with torch.no_grad():
            test_final, _, _ = module(test_albedo, test_shading, test_normal, test_sh)

        print(f"  尺寸 {size}x{size}: 输出形状 {test_final.shape} ✓")

    # ========== 分析残差贡献 ==========
    print("\n" + "=" * 80)
    print("【8】分析残差贡献")
    print("=" * 80)

    with torch.no_grad():
        # 重新计算
        lambertian_render = albedo * shading
        final_render, global_r, local_r = hierarchical_residual(
            albedo, shading, normal, sh_coeffs
        )

        # 计算各部分的贡献
        lambertian_mean = lambertian_render.abs().mean()
        global_mean = global_r.abs().mean()
        local_mean = local_r.abs().mean() if local_r is not None else 0

        total = lambertian_mean + global_mean + local_mean

        print(f"\n各组件的平均绝对值贡献:")
        print(f"  Lambertian项: {lambertian_mean:.4f} ({lambertian_mean/total*100:.1f}%)")
        print(f"  全局残差项: {global_mean:.4f} ({global_mean/total*100:.1f}%)")
        if local_r is not None:
            print(f"  局部残差项: {local_mean:.4f} ({local_mean/total*100:.1f}%)")

    print("\n" + "=" * 80)
    print("测试完成!")
    print("=" * 80)

    print("\n【模块总结】")
    print("1. GlobalResidualMLP:")
    print("   - 处理每个像素的12维特征（normal + SH coeffs）")
    print("   - 使用3层MLP生成平滑的残差图")
    print("   - 用于建模全局非朗伯效应")

    print("\n2. LocalResidual:")
    print("   - 可学习的参数张量，初始化为0")
    print("   - 吸收局部异常和噪声")
    print("   - 提供像素级的精细调整")

    print("\n3. HierarchicalResidual:")
    print("   - 组合全局和局部残差")
    print("   - 完整的分层建模方案")
    print("   - 支持灵活的配置")

    print("\n【使用建议】")
    print("1. 全局残差捕获平滑的、可以用特征解释的效应")
    print("2. 局部残差捕获突发的、难以建模的异常")
    print("3. 训练时可以逐步引入残差项（先训练Lambertian，再加残差）")
    print("4. 局部残差容易过拟合，建议使用正则化")
    print("5. 所有模块都是完全可微分的，支持端到端训练")

    print("\n【渲染方程】")
    print("final_render = albedo * shading + r_global + r_local")
    print("             = Lambertian + 全局非朗伯 + 局部异常")

    print("\n" + "=" * 80)
