"""
可微分物理渲染器模块
实现基于深度图的法线计算、球谐光照和渲染方程

Author: Python Engineer
Date: 2026-01-19
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class DepthToNormal(nn.Module):
    """
    深度图到法线图的转换模块

    使用Sobel算子计算深度梯度，并转换为归一化的法线向量
    支持边缘感知的梯度计算

    Args:
        use_edge_aware: 是否使用边缘感知的梯度计算
    """

    def __init__(self, use_edge_aware: bool = True):
        super(DepthToNormal, self).__init__()

        self.use_edge_aware = use_edge_aware

        # Sobel算子用于计算x和y方向的梯度
        # Sobel X: 检测垂直边缘（水平方向的梯度）
        sobel_x = torch.tensor([
            [-1., 0., 1.],
            [-2., 0., 2.],
            [-1., 0., 1.]
        ]).unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]

        # Sobel Y: 检测水平边缘（垂直方向的梯度）
        sobel_y = torch.tensor([
            [-1., -2., -1.],
            [0., 0., 0.],
            [1., 2., 1.]
        ]).unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]

        # 注册为buffer（不参与训练，但会随模型移动到GPU）
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        """
        将深度图转换为法线图

        Args:
            depth: 深度图 [B, 1, H, W]

        Returns:
            normal: 归一化的法线图 [B, 3, H, W]
        """
        # 使用Sobel算子计算梯度
        # dz/dx
        grad_x = F.conv2d(depth, self.sobel_x, padding=1)

        # dz/dy
        grad_y = F.conv2d(depth, self.sobel_y, padding=1)

        # 如果使用边缘感知，可以根据梯度大小进行加权
        if self.use_edge_aware:
            # 计算梯度幅值
            grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

            # 使用sigmoid进行软化，避免突变
            weight = torch.sigmoid(-grad_mag + 1.0)

            # 加权梯度
            grad_x = grad_x * weight
            grad_y = grad_y * weight

        # 构建法线向量：n = normalize([-dz/dx, -dz/dy, 1])
        # 注意：深度增加方向是远离相机（z正方向），所以梯度要取负
        nx = -grad_x
        ny = -grad_y
        nz = torch.ones_like(grad_x)

        # 堆叠成 [B, 3, H, W]
        normal = torch.cat([nx, ny, nz], dim=1)

        # 归一化为单位向量
        normal = F.normalize(normal, p=2, dim=1)

        return normal


class SphericalHarmonicsLighting(nn.Module):
    """
    球谐光照模块

    实现二阶球谐基函数（9个基函数）用于光照计算
    SH基函数计算公式基于标准的球谐函数定义

    二阶球谐基函数（L=0,1,2）:
    Y_0^0 = 1/2 * sqrt(1/π)
    Y_1^{-1} = sqrt(3/4π) * y
    Y_1^0 = sqrt(3/4π) * z
    Y_1^1 = sqrt(3/4π) * x
    Y_2^{-2} = 1/2 * sqrt(15/π) * xy
    Y_2^{-1} = 1/2 * sqrt(15/π) * yz
    Y_2^0 = 1/4 * sqrt(5/π) * (3z^2 - 1)
    Y_2^1 = 1/2 * sqrt(15/π) * xz
    Y_2^2 = 1/4 * sqrt(15/π) * (x^2 - y^2)
    """

    def __init__(self):
        super(SphericalHarmonicsLighting, self).__init__()

        # 预计算常数系数
        self.C0 = 0.282095  # 1/2 * sqrt(1/π)
        self.C1 = 0.488603  # sqrt(3/4π)
        self.C2 = [
            1.092548,   # 1/2 * sqrt(15/π)
            1.092548,   # 1/2 * sqrt(15/π)
            0.315392,   # 1/4 * sqrt(5/π)
            1.092548,   # 1/2 * sqrt(15/π)
            0.546274    # 1/4 * sqrt(15/π)
        ]

    def compute_sh_basis(self, normal: torch.Tensor) -> torch.Tensor:
        """
        计算球谐基函数值

        Args:
            normal: 归一化的法线向量 [B, 3, H, W]

        Returns:
            sh_basis: 9个球谐基函数值 [B, 9, H, W]
        """
        # 提取xyz分量
        x = normal[:, 0:1, :, :]  # [B, 1, H, W]
        y = normal[:, 1:2, :, :]
        z = normal[:, 2:3, :, :]

        # 计算9个球谐基函数
        # L=0
        Y0 = self.C0 * torch.ones_like(x)

        # L=1
        Y1_n1 = self.C1 * y
        Y1_0 = self.C1 * z
        Y1_p1 = self.C1 * x

        # L=2
        Y2_n2 = self.C2[0] * x * y
        Y2_n1 = self.C2[1] * y * z
        Y2_0 = self.C2[2] * (3.0 * z * z - 1.0)
        Y2_p1 = self.C2[3] * x * z
        Y2_p2 = self.C2[4] * (x * x - y * y)

        # 堆叠成 [B, 9, H, W]
        sh_basis = torch.cat([
            Y0, Y1_n1, Y1_0, Y1_p1, Y2_n2, Y2_n1, Y2_0, Y2_p1, Y2_p2
        ], dim=1)

        return sh_basis

    def forward(
        self,
        normal: torch.Tensor,
        sh_coeffs: torch.Tensor
    ) -> torch.Tensor:
        """
        使用球谐系数计算光照

        Args:
            normal: 法线图 [B, 3, H, W]
            sh_coeffs: 球谐系数 [B, K, 9]，K是光照数量

        Returns:
            shading: 着色图 [B, K, H, W]
        """
        B, _, H, W = normal.shape
        K = sh_coeffs.shape[1]

        # 计算球谐基函数 [B, 9, H, W]
        sh_basis = self.compute_sh_basis(normal)

        # 扩展维度以便批量计算
        # sh_basis: [B, 1, 9, H, W]
        sh_basis = sh_basis.unsqueeze(1)

        # sh_coeffs: [B, K, 9, 1, 1]
        sh_coeffs_expanded = sh_coeffs.unsqueeze(-1).unsqueeze(-1)

        # 计算着色：shading = Σ c_i * Y_i(n)
        # [B, K, 9, H, W]
        weighted_sh = sh_coeffs_expanded * sh_basis

        # 在球谐系数维度上求和 [B, K, H, W]
        shading = weighted_sh.sum(dim=2)

        # 使用ReLU确保非负
        shading = F.relu(shading)

        return shading


class DirectionalLight(nn.Module):
    """
    方向光照模块（可选）

    计算Lambert着色：max(0, n·d)
    其中n是法线，d是光照方向
    """

    def __init__(self):
        super(DirectionalLight, self).__init__()

    def forward(
        self,
        normal: torch.Tensor,
        light_dir: torch.Tensor,
        intensity: torch.Tensor
    ) -> torch.Tensor:
        """
        计算方向光照

        Args:
            normal: 法线图 [B, 3, H, W]
            light_dir: 光照方向 [B, K, 3]，归一化
            intensity: 光照强度 [B, K, 1]

        Returns:
            shading: 方向光着色 [B, K, H, W]
        """
        B, _, H, W = normal.shape
        K = light_dir.shape[1]

        # 扩展维度
        # normal: [B, 1, 3, H, W]
        normal_expanded = normal.unsqueeze(1)

        # light_dir: [B, K, 3, 1, 1]
        light_dir_expanded = light_dir.unsqueeze(-1).unsqueeze(-1)

        # 计算点积：n·d
        # [B, K, H, W]
        dot_product = (normal_expanded * light_dir_expanded).sum(dim=2)

        # Lambert着色：max(0, n·d)
        shading = F.relu(dot_product)

        # 乘以强度
        intensity_expanded = intensity.unsqueeze(-1).unsqueeze(-1)
        shading = shading * intensity_expanded.squeeze(2)

        return shading


class PhysicsRenderer(nn.Module):
    """
    可微分物理渲染器

    将深度图、反照率图和光照参数渲染成最终图像

    渲染流程：
    1. Depth -> Normal (使用Sobel算子)
    2. Normal + SH coeffs -> Shading (球谐光照)
    3. Albedo * Shading -> Rendered image

    Args:
        use_edge_aware: 是否在法线计算中使用边缘感知
        use_directional_light: 是否使用额外的方向光
    """

    def __init__(
        self,
        use_edge_aware: bool = True,
        use_directional_light: bool = False
    ):
        super(PhysicsRenderer, self).__init__()

        self.use_edge_aware = use_edge_aware
        self.use_directional_light = use_directional_light

        # 深度到法线模块
        self.depth_to_normal = DepthToNormal(use_edge_aware=use_edge_aware)

        # 球谐光照模块
        self.sh_lighting = SphericalHarmonicsLighting()

        # 方向光模块（可选）
        if use_directional_light:
            self.dir_lighting = DirectionalLight()

    def forward(
        self,
        depth: torch.Tensor,
        albedo: torch.Tensor,
        sh_coeffs: torch.Tensor,
        dir_light_params: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向渲染

        Args:
            depth: 深度图 [B, 1, H, W]
            albedo: 反照率图 [B, 1, H, W]，范围 [0, 1]
            sh_coeffs: 球谐系数 [B, K, 9]
            dir_light_params: 可选的方向光参数
                - light_dir: 光照方向 [B, K, 3]
                - intensity: 光照强度 [B, K, 1]

        Returns:
            rendered: 渲染图像 [B, K, H, W]
            normal: 法线图 [B, 3, H, W]
            shading: 着色图 [B, K, H, W]
        """
        # ============================
        # 1. 深度 -> 法线
        # ============================
        # [B, 1, H, W] -> [B, 3, H, W]
        normal = self.depth_to_normal(depth)

        # ============================
        # 2. 法线 + 球谐系数 -> 着色
        # ============================
        # [B, 3, H, W], [B, K, 9] -> [B, K, H, W]
        shading = self.sh_lighting(normal, sh_coeffs)

        # ============================
        # 3. 可选：添加方向光
        # ============================
        if self.use_directional_light and dir_light_params is not None:
            light_dir, intensity = dir_light_params
            dir_shading = self.dir_lighting(normal, light_dir, intensity)
            shading = shading + dir_shading

        # ============================
        # 4. 渲染方程：albedo * shading
        # ============================
        # albedo: [B, 1, H, W]
        # shading: [B, K, H, W]
        # rendered: [B, K, H, W]
        albedo_expanded = albedo  # [B, 1, H, W]
        rendered = albedo_expanded * shading

        return rendered, normal, shading


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    """
    测试PhysicsRenderer模块
    """
    print("=" * 80)
    print("PhysicsRenderer 模块测试")
    print("=" * 80)

    # ========== 配置参数 ==========
    BATCH_SIZE = 2
    NUM_IMAGES = 5  # K个光照
    IMAGE_SIZE = 256

    print(f"\n【测试配置】")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"光照数量: {NUM_IMAGES}")
    print(f"图像尺寸: {IMAGE_SIZE} x {IMAGE_SIZE}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # ========== 测试 DepthToNormal ==========
    print("\n" + "=" * 80)
    print("【1】测试 DepthToNormal 模块")
    print("=" * 80)

    depth_to_normal = DepthToNormal(use_edge_aware=True).to(device)

    # 创建一个简单的深度图（球形表面）
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, IMAGE_SIZE),
        torch.linspace(-1, 1, IMAGE_SIZE),
        indexing='ij'
    )
    # 球形深度：z = sqrt(1 - x^2 - y^2)
    r_squared = x ** 2 + y ** 2
    depth_map = torch.sqrt(torch.clamp(1 - r_squared, min=0))
    depth_map = depth_map.unsqueeze(0).unsqueeze(0).repeat(BATCH_SIZE, 1, 1, 1).to(device)

    print(f"\n输入深度图形状: {depth_map.shape}")
    print(f"深度范围: [{depth_map.min():.3f}, {depth_map.max():.3f}]")

    # 计算法线
    normal_map = depth_to_normal(depth_map)

    print(f"\n输出法线图形状: {normal_map.shape}")
    print(f"法线分量范围:")
    print(f"  X: [{normal_map[:, 0].min():.3f}, {normal_map[:, 0].max():.3f}]")
    print(f"  Y: [{normal_map[:, 1].min():.3f}, {normal_map[:, 1].max():.3f}]")
    print(f"  Z: [{normal_map[:, 2].min():.3f}, {normal_map[:, 2].max():.3f}]")

    # 验证法线是单位向量
    normal_length = torch.sqrt((normal_map ** 2).sum(dim=1))
    print(f"\n法线长度统计:")
    print(f"  均值: {normal_length.mean():.6f} (应接近1.0)")
    print(f"  标准差: {normal_length.std():.6f} (应接近0)")
    print(f"  最小值: {normal_length.min():.6f}")
    print(f"  最大值: {normal_length.max():.6f}")

    if torch.allclose(normal_length, torch.ones_like(normal_length), atol=1e-4):
        print("✓ 法线已正确归一化")
    else:
        print("✗ 法线归一化可能存在问题")

    # ========== 测试 SphericalHarmonicsLighting ==========
    print("\n" + "=" * 80)
    print("【2】测试 SphericalHarmonicsLighting 模块")
    print("=" * 80)

    sh_lighting = SphericalHarmonicsLighting().to(device)

    # 创建随机球谐系数
    sh_coeffs = torch.randn(BATCH_SIZE, NUM_IMAGES, 9).to(device)

    print(f"\n球谐系数形状: {sh_coeffs.shape}")
    print(f"球谐系数范围: [{sh_coeffs.min():.3f}, {sh_coeffs.max():.3f}]")

    # 计算着色
    shading = sh_lighting(normal_map, sh_coeffs)

    print(f"\n输出着色图形状: {shading.shape}")
    print(f"着色范围: [{shading.min():.3f}, {shading.max():.3f}]")
    print(f"着色均值: {shading.mean():.3f}")

    # 验证非负性
    if (shading >= 0).all():
        print("✓ 着色值全部非负")
    else:
        print("✗ 存在负值着色（不应该）")

    # 测试球谐基函数
    print("\n测试球谐基函数...")
    sh_basis = sh_lighting.compute_sh_basis(normal_map)
    print(f"球谐基函数形状: {sh_basis.shape}  # [B, 9, H, W]")
    print(f"基函数范围: [{sh_basis.min():.3f}, {sh_basis.max():.3f}]")

    # ========== 测试 DirectionalLight ==========
    print("\n" + "=" * 80)
    print("【3】测试 DirectionalLight 模块")
    print("=" * 80)

    dir_lighting = DirectionalLight().to(device)

    # 创建光照方向（归一化）
    light_dir = torch.randn(BATCH_SIZE, NUM_IMAGES, 3).to(device)
    light_dir = F.normalize(light_dir, p=2, dim=2)

    # 创建光照强度
    intensity = torch.rand(BATCH_SIZE, NUM_IMAGES, 1).to(device)

    print(f"\n光照方向形状: {light_dir.shape}")
    print(f"光照强度形状: {intensity.shape}")
    print(f"光照强度范围: [{intensity.min():.3f}, {intensity.max():.3f}]")

    # 计算方向光着色
    dir_shading = dir_lighting(normal_map, light_dir, intensity)

    print(f"\n方向光着色形状: {dir_shading.shape}")
    print(f"方向光着色范围: [{dir_shading.min():.3f}, {dir_shading.max():.3f}]")

    # ========== 测试完整 PhysicsRenderer ==========
    print("\n" + "=" * 80)
    print("【4】测试完整 PhysicsRenderer")
    print("=" * 80)

    # 不使用方向光
    renderer = PhysicsRenderer(
        use_edge_aware=True,
        use_directional_light=False
    ).to(device)

    # 创建反照率图
    albedo_map = torch.rand(BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE).to(device)

    print(f"\n输入:")
    print(f"  深度图: {depth_map.shape}")
    print(f"  反照率图: {albedo_map.shape}")
    print(f"  球谐系数: {sh_coeffs.shape}")

    # 渲染
    rendered, normal, shading = renderer(depth_map, albedo_map, sh_coeffs)

    print(f"\n输出:")
    print(f"  渲染图像: {rendered.shape}  # [B, K, H, W]")
    print(f"  法线图: {normal.shape}  # [B, 3, H, W]")
    print(f"  着色图: {shading.shape}  # [B, K, H, W]")

    print(f"\n数值范围:")
    print(f"  渲染图像: [{rendered.min():.3f}, {rendered.max():.3f}]")
    print(f"  法线图: [{normal.min():.3f}, {normal.max():.3f}]")
    print(f"  着色图: [{shading.min():.3f}, {shading.max():.3f}]")

    # ========== 测试使用方向光的渲染器 ==========
    print("\n" + "=" * 80)
    print("【5】测试带方向光的 PhysicsRenderer")
    print("=" * 80)

    renderer_with_dir = PhysicsRenderer(
        use_edge_aware=True,
        use_directional_light=True
    ).to(device)

    # 渲染（包含方向光）
    rendered_dir, normal_dir, shading_dir = renderer_with_dir(
        depth_map,
        albedo_map,
        sh_coeffs,
        dir_light_params=(light_dir, intensity)
    )

    print(f"\n输出:")
    print(f"  渲染图像: {rendered_dir.shape}")
    print(f"  着色范围: [{shading_dir.min():.3f}, {shading_dir.max():.3f}]")

    # 比较有无方向光的差异
    shading_diff = (shading_dir - shading).abs().mean()
    print(f"\n方向光影响:")
    print(f"  着色差异均值: {shading_diff:.3f}")

    # ========== 测试梯度传播 ==========
    print("\n" + "=" * 80)
    print("【6】测试梯度传播（可微分性）")
    print("=" * 80)

    # 启用梯度
    depth_grad = depth_map.clone().requires_grad_(True)
    albedo_grad = albedo_map.clone().requires_grad_(True)
    sh_coeffs_grad = sh_coeffs.clone().requires_grad_(True)

    # 前向传播
    rendered_grad, _, _ = renderer(depth_grad, albedo_grad, sh_coeffs_grad)

    # 计算损失
    loss = rendered_grad.mean()

    # 反向传播
    loss.backward()

    print(f"\n损失值: {loss.item():.6f}")
    print(f"\n梯度统计:")
    print(f"  深度梯度: 形状={depth_grad.grad.shape}, 范数={depth_grad.grad.norm():.6f}")
    print(f"  反照率梯度: 形状={albedo_grad.grad.shape}, 范数={albedo_grad.grad.norm():.6f}")
    print(f"  球谐系数梯度: 形状={sh_coeffs_grad.grad.shape}, 范数={sh_coeffs_grad.grad.norm():.6f}")

    if depth_grad.grad is not None and albedo_grad.grad is not None:
        print("✓ 梯度传播成功，模块完全可微分")
    else:
        print("✗ 梯度传播失败")

    # ========== 测试边缘感知 ==========
    print("\n" + "=" * 80)
    print("【7】比较边缘感知 vs 非边缘感知")
    print("=" * 80)

    # 非边缘感知
    depth_to_normal_no_edge = DepthToNormal(use_edge_aware=False).to(device)
    normal_no_edge = depth_to_normal_no_edge(depth_map)

    # 边缘感知
    normal_with_edge = depth_to_normal(depth_map)

    # 计算差异
    normal_diff = (normal_with_edge - normal_no_edge).abs().mean()

    print(f"\n法线图差异:")
    print(f"  均值差异: {normal_diff:.6f}")
    print(f"  边缘感知对法线计算有 {'显著' if normal_diff > 0.01 else '轻微'} 影响")

    # ========== 性能测试 ==========
    print("\n" + "=" * 80)
    print("【8】性能测试")
    print("=" * 80)

    import time

    renderer.eval()

    # 预热
    for _ in range(10):
        with torch.no_grad():
            _ = renderer(depth_map, albedo_map, sh_coeffs)

    # 计时
    num_iters = 100
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_iters):
            _ = renderer(depth_map, albedo_map, sh_coeffs)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    end_time = time.time()
    avg_time = (end_time - start_time) / num_iters * 1000  # ms

    print(f"\n平均推理时间: {avg_time:.2f} ms/batch")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"图像数量: {NUM_IMAGES}")
    print(f"图像尺寸: {IMAGE_SIZE}x{IMAGE_SIZE}")

    print("\n" + "=" * 80)
    print("测试完成!")
    print("=" * 80)

    print("\n【模块总结】")
    print("1. DepthToNormal: 使用Sobel算子从深度图计算法线")
    print("   - 支持边缘感知的梯度计算")
    print("   - 输出归一化的单位法线向量")
    print("   - 完全可微分")

    print("\n2. SphericalHarmonicsLighting: 二阶球谐光照")
    print("   - 实现9个球谐基函数")
    print("   - 支持多光照场景")
    print("   - 使用ReLU保证非负着色")

    print("\n3. DirectionalLight: 可选的方向光组件")
    print("   - Lambert着色模型")
    print("   - 可与球谐光照叠加")

    print("\n4. PhysicsRenderer: 完整的可微分渲染器")
    print("   - 端到端可微分")
    print("   - 支持批量渲染")
    print("   - 灵活的光照模型组合")

    print("\n【使用建议】")
    print("1. 深度图应预先归一化到合理范围")
    print("2. 球谐系数可以通过网络学习或预设")
    print("3. 边缘感知在深度不连续处效果更好")
    print("4. 所有模块都支持GPU加速")
    print("5. 可以在训练循环中直接使用，无需额外处理")

    print("\n" + "=" * 80)
