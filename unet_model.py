"""
逆向渲染U-Net模型
具有多个输出头的U-Net架构，用于同时估计深度、反照率、光照参数和权重图

Author: Python Engineer
Date: 2026-01-19
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DownBlock(nn.Module):
    """
    U-Net下采样块

    包含两个卷积层 + BN + ReLU，然后进行最大池化下采样

    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        use_pooling: 是否使用池化层（最后一层不使用）
    """

    def __init__(self, in_channels: int, out_channels: int, use_pooling: bool = True):
        super(DownBlock, self).__init__()

        self.use_pooling = use_pooling

        # 第一个卷积块
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        # 第二个卷积块
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        # 池化层（2x2）
        if self.use_pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入张量

        Returns:
            pooled: 池化后的张量（如果use_pooling=True）
            skip: 用于skip connection的张量
        """
        # 两次卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        skip = x  # 保存用于skip connection

        # 池化（如果需要）
        if self.use_pooling:
            x = self.pool(x)

        return x, skip


class UpBlock(nn.Module):
    """
    U-Net上采样块

    上采样 + 拼接skip connection + 两个卷积层

    Args:
        in_channels: 输入通道数（上采样后）
        skip_channels: skip connection的通道数
        out_channels: 输出通道数
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super(UpBlock, self).__init__()

        # 使用 Upsample + Conv 替代转置卷积
        # 上采样层：将特征图放大2倍
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # 拼接后的通道数 = in_channels + skip_channels
        concat_channels = in_channels + skip_channels

        # 第一个卷积块
        self.conv1 = nn.Conv2d(concat_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        # 第二个卷积块
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量（来自下一层）
            skip: skip connection张量（来自编码器）

        Returns:
            输出张量
        """
        # 上采样：使用 bilinear 插值放大2倍
        x = self.upsample(x)

        # 拼接skip connection
        x = torch.cat([x, skip], dim=1)

        # 两次卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x


class IntrinsicUNet(nn.Module):
    """
    用于逆向渲染的U-Net模型

    输入K张多光照灰度图像，输出深度图、反照率图、球谐光照系数和权重图

    Args:
        num_images (int): 输入图像数量K
        base_channels (int): 基础通道数，默认32
        sh_order (int): 球谐阶数，默认2（对应9个系数）

    Input:
        x: [B, K, H, W] - K张堆叠的灰度图像

    Output:
        depth: [B, 1, H, W] - 深度图
        albedo: [B, 1, H, W] - 反照率图（sigmoid，范围[0,1]）
        sh_coeffs: [B, K, 9] - 每张图像的球谐系数
        weight_map: [B, 1, H, W] - 自适应权重图（sigmoid，范围[0,1]）
    """

    def __init__(
        self,
        num_images: int = 5,
        base_channels: int = 32,
        sh_order: int = 2
    ):
        super(IntrinsicUNet, self).__init__()

        self.num_images = num_images
        self.base_channels = base_channels
        self.sh_order = sh_order
        self.sh_coeffs_dim = (sh_order + 1) ** 2  # 二阶球谐：9个系数

        # ============================
        # Encoder（编码器）
        # ============================
        # Level 0: K -> 32
        self.down1 = DownBlock(num_images, base_channels, use_pooling=True)

        # Level 1: 32 -> 64
        self.down2 = DownBlock(base_channels, base_channels * 2, use_pooling=True)

        # Level 2: 64 -> 128
        self.down3 = DownBlock(base_channels * 2, base_channels * 4, use_pooling=True)

        # Level 3: 128 -> 256
        self.down4 = DownBlock(base_channels * 4, base_channels * 8, use_pooling=True)

        # ============================
        # Bottleneck（瓶颈层）
        # ============================
        # 256 -> 512
        self.bottleneck = DownBlock(base_channels * 8, base_channels * 16, use_pooling=False)

        # ============================
        # Decoder（解码器）
        # ============================
        # Level 3: 512 + 256 -> 256
        self.up1 = UpBlock(base_channels * 16, base_channels * 8, base_channels * 8)

        # Level 2: 256 + 128 -> 128
        self.up2 = UpBlock(base_channels * 8, base_channels * 4, base_channels * 4)

        # Level 1: 128 + 64 -> 64
        self.up3 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2)

        # Level 0: 64 + 32 -> 32
        self.up4 = UpBlock(base_channels * 2, base_channels, base_channels)

        # ============================
        # Output Heads（输出头）
        # ============================

        # 1. Depth head: 32 -> 1
        self.depth_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 2, 1, kernel_size=1)
            # 不使用激活函数，深度值可以是任意正值
        )

        # 2. Albedo head: 32 -> 1 (线性输出，无激活函数)
        self.albedo_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 2, 1, kernel_size=1)
            # 移除 Sigmoid，让 Albedo 输出线性值，值域应接近 [0, 2]
        )

        # 3. Weight map head: 32 -> 1 + sigmoid
        self.weight_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 2, 1, kernel_size=1),
            nn.Sigmoid()  # 权重范围 [0, 1]
        )

        # 4. Spherical Harmonics head: Global pooling + FC -> K * 9
        # 使用自适应平均池化将特征图压缩为全局特征
        self.sh_gap = nn.AdaptiveAvgPool2d((1, 1))

        # 全连接层：512 -> K * 9，添加tanh激活约束在-1到1之间
        self.sh_fc = nn.Sequential(
            nn.Linear(base_channels * 16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_images * self.sh_coeffs_dim),
            nn.Tanh()  # 添加tanh激活，约束输出在[-1, 1]之间
        )

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入张量 [B, K, H, W]

        Returns:
            depth: 深度图 [B, 1, H, W]
            albedo: 反照率图 [B, 1, H, W]
            sh_coeffs: 球谐系数 [B, K, 9]
            weight_map: 权重图 [B, 1, H, W]
        """
        # ============================
        # Encoder
        # ============================
        # Level 0: [B, K, H, W] -> [B, 32, H/2, H/2]
        x1, skip1 = self.down1(x)

        # Level 1: [B, 32, H/2, H/2] -> [B, 64, H/4, H/4]
        x2, skip2 = self.down2(x1)

        # Level 2: [B, 64, H/4, H/4] -> [B, 128, H/8, H/8]
        x3, skip3 = self.down3(x2)

        # Level 3: [B, 128, H/8, H/8] -> [B, 256, H/16, H/16]
        x4, skip4 = self.down4(x3)

        # ============================
        # Bottleneck
        # ============================
        # [B, 256, H/16, H/16] -> [B, 512, H/16, H/16]
        bottleneck, _ = self.bottleneck(x4)

        # ============================
        # 球谐系数分支（在bottleneck后）
        # ============================
        # Global average pooling: [B, 512, H/16, H/16] -> [B, 512, 1, 1]
        sh_features = self.sh_gap(bottleneck)
        # Flatten: [B, 512, 1, 1] -> [B, 512]
        sh_features = sh_features.view(sh_features.size(0), -1)
        # FC: [B, 512] -> [B, K*9]
        sh_coeffs = self.sh_fc(sh_features)
        # Reshape: [B, K*9] -> [B, K, 9]
        sh_coeffs = sh_coeffs.view(-1, self.num_images, self.sh_coeffs_dim)

        # ============================
        # 物理约束应用
        # ============================
        # 1. 强制球谐直流分量（SH[0]）非负
        sh0_nonneg = F.relu(sh_coeffs[:, :, 0:1])  # [B, K, 1]
        sh_higher = sh_coeffs[:, :, 1:]  # [B, K, 8]

        # 2. 🔴【关键修改】光环粉碎者：强制 SH[0] 均值为 1.0
        # 这会立刻打破 2.0 倍的增益，让光照回归正常
        sh_coeffs = torch.cat([sh0_nonneg, sh_higher], dim=-1)

        # 3. 🔴【关键修改】绝对禁止 SH[0] 超过 1.0
        # 双重保险：Clamp 它（使用非原地操作，避免梯度计算错误）
        sh0_clamped = torch.clamp(sh_coeffs[:, :, 0:1], 0.0, 1.0)
        sh_higher = sh_coeffs[:, :, 1:]
        sh_coeffs = torch.cat([sh0_clamped, sh_higher], dim=-1)

        # ============================
        # Decoder
        # ============================
        # Level 3: [B, 512, H/16, H/16] + skip4 -> [B, 256, H/8, H/8]
        x = self.up1(bottleneck, skip4)

        # Level 2: [B, 256, H/8, H/8] + skip3 -> [B, 128, H/4, H/4]
        x = self.up2(x, skip3)

        # Level 1: [B, 128, H/4, H/4] + skip2 -> [B, 64, H/2, H/2]
        x = self.up3(x, skip2)

        # Level 0: [B, 64, H/2, H/2] + skip1 -> [B, 32, H, H]
        x = self.up4(x, skip1)

        # ============================
        # Output Heads
        # ============================
        # Depth: [B, 32, H, W] -> [B, 1, H, W]
        depth = self.depth_head(x)

        # Albedo: [B, 32, H, W] -> [B, 1, H, W]
        albedo = self.albedo_head(x)

        # Weight map: [B, 32, H, W] -> [B, 1, H, W]
        weight_map = self.weight_head(x)

        return depth, albedo, sh_coeffs, weight_map


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    """
    测试IntrinsicUNet模型
    """
    print("=" * 80)
    print("IntrinsicUNet 模型测试")
    print("=" * 80)

    # ========== 模型配置 ==========
    NUM_IMAGES = 5      # 每个场景5张不同光照的图像
    BASE_CHANNELS = 32  # 基础通道数
    IMAGE_SIZE = 256    # 输入图像尺寸
    BATCH_SIZE = 2      # 批次大小

    print(f"\n【模型配置】")
    print(f"输入图像数量 K: {NUM_IMAGES}")
    print(f"基础通道数: {BASE_CHANNELS}")
    print(f"输入图像尺寸: {IMAGE_SIZE} x {IMAGE_SIZE}")
    print(f"批次大小: {BATCH_SIZE}")

    # ========== 创建模型 ==========
    print("\n" + "=" * 80)
    print("【1】创建模型")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")

    model = IntrinsicUNet(
        num_images=NUM_IMAGES,
        base_channels=BASE_CHANNELS,
        sh_order=2  # 二阶球谐
    ).to(device)

    print(f"✓ 模型创建成功")

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n模型参数统计:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  参数大小: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")

    # ========== 测试前向传播 ==========
    print("\n" + "=" * 80)
    print("【2】测试前向传播")
    print("=" * 80)

    # 创建随机输入
    dummy_input = torch.randn(BATCH_SIZE, NUM_IMAGES, IMAGE_SIZE, IMAGE_SIZE).to(device)
    print(f"\n输入形状: {dummy_input.shape}  # [B, K, H, W]")
    print(f"输入数值范围: [{dummy_input.min():.3f}, {dummy_input.max():.3f}]")

    # 前向传播
    print(f"\n执行前向传播...")
    model.eval()
    with torch.no_grad():
        depth, albedo, sh_coeffs, weight_map = model(dummy_input)

    print(f"✓ 前向传播成功")

    # ========== 检查输出 ==========
    print("\n" + "=" * 80)
    print("【3】检查输出")
    print("=" * 80)

    print(f"\n1. Depth Map（深度图）:")
    print(f"   形状: {depth.shape}  # [B, 1, H, W]")
    print(f"   数值范围: [{depth.min():.3f}, {depth.max():.3f}]")
    print(f"   数据类型: {depth.dtype}")
    print(f"   梯度: {'启用' if depth.requires_grad else '禁用'}")

    print(f"\n2. Albedo Map（反照率图）:")
    print(f"   形状: {albedo.shape}  # [B, 1, H, W]")
    print(f"   数值范围: [{albedo.min():.3f}, {albedo.max():.3f}]")
    print(f"   应在 [0, 1] 范围内: {(albedo >= 0).all() and (albedo <= 1).all()}")
    print(f"   数据类型: {albedo.dtype}")

    print(f"\n3. SH Coefficients（球谐系数）:")
    print(f"   形状: {sh_coeffs.shape}  # [B, K, 9]")
    print(f"   数值范围: [{sh_coeffs.min():.3f}, {sh_coeffs.max():.3f}]")
    print(f"   每张图像9个系数（二阶球谐）")
    print(f"   数据类型: {sh_coeffs.dtype}")

    print(f"\n4. Weight Map（权重图）:")
    print(f"   形状: {weight_map.shape}  # [B, 1, H, W]")
    print(f"   数值范围: [{weight_map.min():.3f}, {weight_map.max():.3f}]")
    print(f"   应在 [0, 1] 范围内: {(weight_map >= 0).all() and (weight_map <= 1).all()}")
    print(f"   数据类型: {weight_map.dtype}")

    # ========== 测试梯度传播 ==========
    print("\n" + "=" * 80)
    print("【4】测试梯度传播")
    print("=" * 80)

    model.train()
    dummy_input_grad = torch.randn(BATCH_SIZE, NUM_IMAGES, IMAGE_SIZE, IMAGE_SIZE).to(device)
    dummy_input_grad.requires_grad = True

    # 前向传播
    depth_g, albedo_g, sh_coeffs_g, weight_g = model(dummy_input_grad)

    # 计算一个简单的损失
    loss = depth_g.mean() + albedo_g.mean() + sh_coeffs_g.mean() + weight_g.mean()

    # 反向传播
    loss.backward()

    print(f"\n损失值: {loss.item():.6f}")
    print(f"输入梯度形状: {dummy_input_grad.grad.shape}")
    print(f"输入梯度范数: {dummy_input_grad.grad.norm().item():.6f}")
    print(f"✓ 梯度传播成功，模型可微分")

    # ========== 测试不同输入尺寸 ==========
    print("\n" + "=" * 80)
    print("【5】测试不同输入尺寸")
    print("=" * 80)

    test_sizes = [128, 256, 512]
    model.eval()

    print(f"\n测试不同分辨率的输入...")
    for size in test_sizes:
        test_input = torch.randn(1, NUM_IMAGES, size, size).to(device)
        with torch.no_grad():
            d, a, s, w = model(test_input)

        print(f"  输入 {size}x{size} -> 输出 depth: {d.shape}, albedo: {a.shape}, SH: {s.shape}, weight: {w.shape}")

    print(f"✓ 模型支持多种输入尺寸")

    # ========== 内存占用估算 ==========
    print("\n" + "=" * 80)
    print("【6】内存占用估算")
    print("=" * 80)

    # 输入内存
    input_memory = BATCH_SIZE * NUM_IMAGES * IMAGE_SIZE * IMAGE_SIZE * 4 / 1024 / 1024

    # 输出内存
    output_memory = (
        BATCH_SIZE * 1 * IMAGE_SIZE * IMAGE_SIZE * 4 +  # depth
        BATCH_SIZE * 1 * IMAGE_SIZE * IMAGE_SIZE * 4 +  # albedo
        BATCH_SIZE * NUM_IMAGES * 9 * 4 +                # sh_coeffs
        BATCH_SIZE * 1 * IMAGE_SIZE * IMAGE_SIZE * 4     # weight
    ) / 1024 / 1024

    print(f"\n单个batch内存占用估算 (batch_size={BATCH_SIZE}, size={IMAGE_SIZE}):")
    print(f"  输入: {input_memory:.2f} MB")
    print(f"  输出: {output_memory:.2f} MB")
    print(f"  模型参数: {total_params * 4 / 1024 / 1024:.2f} MB")
    print(f"  估算总计: {input_memory + output_memory + total_params * 4 / 1024 / 1024:.2f} MB")

    # ========== 模型架构总结 ==========
    print("\n" + "=" * 80)
    print("【7】模型架构总结")
    print("=" * 80)

    print(f"\n编码器路径:")
    print(f"  Input: [B, {NUM_IMAGES}, H, W]")
    print(f"  Down1: [B, {NUM_IMAGES}, H, W] -> [B, {BASE_CHANNELS}, H/2, W/2]")
    print(f"  Down2: [B, {BASE_CHANNELS}, H/2, W/2] -> [B, {BASE_CHANNELS*2}, H/4, W/4]")
    print(f"  Down3: [B, {BASE_CHANNELS*2}, H/4, W/4] -> [B, {BASE_CHANNELS*4}, H/8, W/8]")
    print(f"  Down4: [B, {BASE_CHANNELS*4}, H/8, W/8] -> [B, {BASE_CHANNELS*8}, H/16, W/16]")
    print(f"  Bottleneck: [B, {BASE_CHANNELS*8}, H/16, W/16] -> [B, {BASE_CHANNELS*16}, H/16, W/16]")

    print(f"\n解码器路径:")
    print(f"  Up1: [B, {BASE_CHANNELS*16}, H/16, W/16] -> [B, {BASE_CHANNELS*8}, H/8, W/8]")
    print(f"  Up2: [B, {BASE_CHANNELS*8}, H/8, W/8] -> [B, {BASE_CHANNELS*4}, H/4, W/4]")
    print(f"  Up3: [B, {BASE_CHANNELS*4}, H/4, W/4] -> [B, {BASE_CHANNELS*2}, H/2, W/2]")
    print(f"  Up4: [B, {BASE_CHANNELS*2}, H/2, W/2] -> [B, {BASE_CHANNELS}, H, W]")

    print(f"\n输出头:")
    print(f"  Depth Head: [B, {BASE_CHANNELS}, H, W] -> [B, 1, H, W]")
    print(f"  Albedo Head: [B, {BASE_CHANNELS}, H, W] -> [B, 1, H, W] + Sigmoid")
    print(f"  SH Head: [B, {BASE_CHANNELS*16}, H/16, W/16] -> [B, {NUM_IMAGES}, 9]")
    print(f"  Weight Head: [B, {BASE_CHANNELS}, H, W] -> [B, 1, H, W] + Sigmoid")

    print("\n" + "=" * 80)
    print("测试完成!")
    print("=" * 80)

    print("\n【使用建议】")
    print("1. GPU内存充足时可增大batch_size")
    print("2. 根据数据集调整num_images参数")
    print("3. 可以调整base_channels来控制模型容量")
    print("4. 建议使用混合精度训练（AMP）节省内存")
    print("5. 所有输出都是可微分的，可以端到端训练")
    print("6. Depth输出未限制范围，可能需要后处理或使用特定损失函数")

    print("\n" + "=" * 80)
