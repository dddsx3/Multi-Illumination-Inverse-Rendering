"""
逆向渲染损失函数模块
实现所有用于训练的损失函数，包括重建损失、平滑损失、先验损失等

Author: Python Engineer
Date: 2026-01-19
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np


class CharbonnierLoss(nn.Module):
    """
    Charbonnier损失函数（也称为伪Huber损失）

    比L2损失对异常值更鲁棒
    公式: φ(x) = sqrt(x² + ε²)

    Args:
        epsilon: 平滑参数，默认1e-3
    """

    def __init__(self, epsilon: float = 1e-3):
        super(CharbonnierLoss, self).__init__()
        self.epsilon = epsilon
        self.epsilon_sq = epsilon ** 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算Charbonnier损失

        Args:
            pred: 预测值
            target: 目标值

        Returns:
            loss: 标量损失值
        """
        diff = pred - target
        loss = torch.sqrt(diff ** 2 + self.epsilon_sq)
        return loss.mean()


class DepthSmoothnessLoss(nn.Module):
    """
    边缘感知的深度平滑损失

    在图像边缘处降低平滑约束，保留深度不连续
    公式: L = mean( |∇z| * exp(-γ|∇I_max|) )
    其中∇I_max是所有图像中梯度最大的值，避免边缘信号稀释

    Args:
        gamma: 边缘感知参数，默认10
    """

    def __init__(self, gamma: float = 10.0):
        super(DepthSmoothnessLoss, self).__init__()
        self.gamma = gamma

    def compute_gradient(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算图像梯度

        Args:
            img: 输入图像 [B, K, H, W]

        Returns:
             grad_x: x方向梯度 [B, K, H, W-1]
            grad_y: y方向梯度 [B, K, H-1, W]
        """
        # 使用Sobel算子计算梯度
        # x方向梯度
        grad_x = img[:, :, :, :-1] - img[:, :, :, 1:]
        # y方向梯度
        grad_y = img[:, :, :-1, :] - img[:, :, 1:, :]

        return grad_x, grad_y

    def forward(
        self,
        depth: torch.Tensor,
        image: torch.Tensor
    ) -> torch.Tensor:
        """
        计算边缘感知的深度平滑损失

        Args:
            depth: 深度图 [B, 1, H, W]
            image: 输入图像 [B, K, H, W]

        Returns:
            loss: 标量损失值
        """
        # 计算输入图像的平均（如果有多个光照）

        # 计算深度梯度
        depth_grad_x, depth_grad_y = self.compute_gradient(depth.expand_as(image))

        # 计算图像梯度
        image_grad_x, image_grad_y = self.compute_gradient(image)

        # 取所有图像中的最大梯度绝对值，保留最强边缘
        image_grad_x_abs = torch.abs(image_grad_x).max(dim=1, keepdim=True)[0]
        image_grad_y_abs = torch.abs(image_grad_y).max(dim=1, keepdim=True)[0]

        # 深度梯度绝对值
        depth_grad_x_abs = torch.abs(depth_grad_x[:, 0:1, :, :])  # [B, 1, H, W-1]
        depth_grad_y_abs = torch.abs(depth_grad_y[:, 0:1, :, :])  # [B, 1, H-1, W]

        # 边缘权重: exp(-γ|∇I_max|) - 保留最强边缘信息
        weight_x = torch.exp(-self.gamma * image_grad_x_abs)
        weight_y = torch.exp(-self.gamma * image_grad_y_abs)

        # 加权深度梯度
        loss_x = (depth_grad_x_abs * weight_x).mean()
        loss_y = (depth_grad_y_abs * weight_y).mean()

        loss = loss_x + loss_y

        return loss


class AdaptiveAlbedoLoss(nn.Module):
    """
    自适应反照率平滑损失

    使用网络预测的权重图来调节平滑约束
    公式: L = mean( w(x,y) * |∇ρ| )

    Args:
        None
    """

    def __init__(self):
        super(AdaptiveAlbedoLoss, self).__init__()

    def compute_gradient_magnitude(self, img: torch.Tensor) -> torch.Tensor:
        """
        计算梯度幅值

        Args:
            img: 输入图像 [B, C, H, W]

        Returns:
            grad_mag: 梯度幅值 [B, C, H-1, W-1]
        """
        # x方向梯度
        grad_x = img[:, :, :-1, 1:] - img[:, :, :-1, :-1]
        # y方向梯度
        grad_y = img[:, :, 1:, :-1] - img[:, :, :-1, :-1]

        # 梯度幅值
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

        return grad_mag

    def forward(
        self,
        albedo: torch.Tensor,
        weight_map: torch.Tensor
    ) -> torch.Tensor:
        """
        计算自适应反照率平滑损失

        Args:
            albedo: 反照率图 [B, 1, H, W]
            weight_map: 权重图 [B, 1, H, W]

        Returns:
            loss: 标量损失值
        """
        # 计算反照率梯度幅值
        albedo_grad = self.compute_gradient_magnitude(albedo)

        # 裁剪权重图以匹配梯度尺寸
        weight_cropped = weight_map[:, :, :-1, :-1]

        # 加权梯度
        weighted_grad = weight_cropped * albedo_grad

        loss = weighted_grad.mean()

        return loss


class WeightMapRegularization(nn.Module):
    """
    权重图正则化损失

    包含两个部分：
    1. 分布损失: KL散度使权重图接近Beta(2,2)分布
    2. 平滑损失: TV损失鼓励权重图平滑

    Args:
        alpha: Beta分布参数α，默认2
        beta: Beta分布参数β，默认2
    """

    def __init__(self, alpha: float = 2.0, beta: float = 2.0):
        super(WeightMapRegularization, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def beta_distribution_loss(self, weight: torch.Tensor) -> torch.Tensor:
        """
        KL散度损失：使权重分布接近Beta(α,β)

        简化实现：鼓励权重集中在中间值（避免过于极端的0或1）

        Args:
            weight: 权重图 [B, 1, H, W]，范围 [0, 1]

        Returns:
            loss: 标量损失值
        """
        # Beta(2,2)的期望是0.5
        # 鼓励权重分布在中间范围
        # 使用简化的损失：惩罚远离期望的值

        # 期望值
        expected_mean = self.alpha / (self.alpha + self.beta)  # 0.5 for Beta(2,2)

        # 简化KL散度：惩罚偏离期望
        # 也惩罚过于极端的值（接近0或1）
        deviation_loss = (weight - expected_mean) ** 2

        # 惩罚接近0或1的值（熵正则化）
        epsilon = 1e-8
        entropy_loss = -(weight * torch.log(weight + epsilon) +
                        (1 - weight) * torch.log(1 - weight + epsilon))

        # Beta(2,2)鼓励中等不确定性，所以我们希望有一定的熵
        # 但不要太高（那会是均匀分布）
        target_entropy = 0.69  # Beta(2,2)的熵约为0.69
        entropy_regularization = (entropy_loss.mean() - target_entropy) ** 2

        loss = deviation_loss.mean() + 0.1 * entropy_regularization

        return loss

    def total_variation_loss(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Total Variation损失：鼓励权重图平滑

        公式: TV(w) = mean(|∇w|)

        Args:
            weight: 权重图 [B, 1, H, W]

        Returns:
            loss: 标量损失值
        """
        # x方向差分
        diff_x = weight[:, :, :, :-1] - weight[:, :, :, 1:]
        # y方向差分
        diff_y = weight[:, :, :-1, :] - weight[:, :, 1:, :]

        # TV损失
        loss = (torch.abs(diff_x).mean() + torch.abs(diff_y).mean())

        return loss

    def forward(self, weight_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算权重图正则化损失

        Args:
            weight_map: 权重图 [B, 1, H, W]

        Returns:
            dist_loss: 分布损失
            tv_loss: TV损失
        """
        dist_loss = self.beta_distribution_loss(weight_map)
        tv_loss = self.total_variation_loss(weight_map)

        return dist_loss, tv_loss


class LightingPriorLoss(nn.Module):
    """
    光照先验损失

    对球谐系数施加正则化：
    1. L2正则化：鼓励小的系数值
    2. 高阶惩罚：对高阶系数（6-9）施加更强的惩罚
    3. 稀疏性惩罚：使用L1正则化鼓励稀疏系数
    Args:
        higher_order_weight: 高阶系数的额外惩罚权重
    """

    def __init__(self, higher_order_weight: float = 2.0):
        super(LightingPriorLoss, self).__init__()
        self.higher_order_weight = higher_order_weight

    def forward(self, sh_coeffs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算光照先验损失

        Args:
            sh_coeffs: 球谐系数 [B, K, 9]

        Returns:
            l2_loss: L2正则化损失
            higher_order_loss: 高阶惩罚损失
            sparsity_loss: 稀疏性惩罚损失
        """
        # 1. L2正则化：所有系数
        l2_loss = (sh_coeffs ** 2).mean()

        # 2. 高阶惩罚：系数6-9（索引5-8）
        # L=2, m=-2,-1,0,1,2对应索引4-8
        higher_order_coeffs = sh_coeffs[:, :, 5:]  # 索引5-8，对应Y_2^{-1}, Y_2^0, Y_2^1, Y_2^2
        higher_order_loss = (higher_order_coeffs ** 2).mean()

        # 3. 稀疏性惩罚：使用L1正则化鼓励稀疏系数
        sparsity_loss = torch.abs(sh_coeffs).mean()

        return l2_loss, higher_order_loss, sparsity_loss


class LightingConstraintLoss(nn.Module):
    """
    光照约束损失

    确保球谐系数符合物理约束：
    1. 直流分量（SH[0]）非负
    2. 各阶系数能量分布合理
    3. 光照能量总和非负

    Args:
        None
    """

    def __init__(self):
        super(LightingConstraintLoss, self).__init__()

    def forward(self, sh_coeffs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算光照约束损失

        Args:
            sh_coeffs: 球谐系数 [B, K, 9]

        Returns:
            sh0_nonneg_loss: SH[0]非负损失
            energy_distribution_loss: 能量分布损失
            total_energy_loss: 总能量非负损失
        """
        # 1. 直流分量（SH[0]）非负损失
        # 使用ReLU的负部分作为惩罚
        sh0 = sh_coeffs[:, :, 0]
        sh0_nonneg_loss = F.relu(-sh0).mean()

        # 2. 各阶系数能量分布损失
        # 确保能量分布合理，低阶系数（尤其是SH[0]）贡献主要能量
        # 计算各阶能量
        # L0: 索引0（Y_0^0）
        energy_l0 = (sh_coeffs[:, :, 0:1] ** 2).mean()
        # L1: 索引1-3（Y_1^{-1}, Y_1^0, Y_1^1）
        energy_l1 = (sh_coeffs[:, :, 1:4] ** 2).mean()
        # L2: 索引4-8（Y_2^{-2}, Y_2^{-1}, Y_2^0, Y_2^1, Y_2^2）
        energy_l2 = (sh_coeffs[:, :, 4:] ** 2).mean()

        # 希望能量分布：L0 >= L1 >= L2
        # 惩罚高阶能量超过低阶的情况
        energy_distribution_loss = (
                F.relu(energy_l1 - energy_l0) +  # 惩罚L1 > L0
                F.relu(energy_l2 - energy_l1) +  # 惩罚L2 > L1
                F.relu(energy_l2 - energy_l0)  # 惩罚L2 > L0（更强约束）
        ).mean()

        # 3. 光照能量总和非负
        # 计算每个位置的光照能量总和
        # 对于二阶球谐，能量总和可以近似为所有系数平方和
        total_energy = (sh_coeffs ** 2).sum(dim=-1, keepdim=True)
        total_energy_loss = F.relu(-total_energy).mean()

        return sh0_nonneg_loss, energy_distribution_loss, total_energy_loss


class AlbedoConsistencyLoss(nn.Module):
    """
    反照率一致性损失（保留类，当前架构下为空操作）

    在现有单通道反照率设计中，所有 K 张光照图像共用同一张反照率图，
    跨光照一致性由网络结构本身保证（架构性约束），无需损失函数——
    此前该损失在 albedo.shape[1]==1 时直接返回 0，实际从未生效，
    Phase 0 已将其从 LossCalculator 活跃损失中移除。

    当 A2 升级（任意 N 输入、逐光照反照率分支）落地后，此损失将重新启用。
    """

    def __init__(self):
        super(AlbedoConsistencyLoss, self).__init__()

    def forward(
            self,
            albedo: torch.Tensor,
            images: torch.Tensor
    ) -> torch.Tensor:
        """
        计算反照率一致性损失

        Args:
            albedo: 反照率图 [B, 1, H, W]
            images: 输入图像 [B, K, H, W]

        Returns:
            loss: 标量损失值
        """
        # 检查反照率的维度
        if albedo.shape[1] == 1:
            # 模型只输出一个反照率图，已经满足一致性约束，无需计算损失
            return torch.tensor(0.0, device=albedo.device)
        else:
            # 模型输出多个反照率图，计算它们之间的差异
            K = albedo.shape[1]
            consistency_loss = 0.0

            # 计算所有反照率图之间的成对差异
            for i in range(K):
                for j in range(i + 1, K):
                    consistency_loss += torch.abs(albedo[:, i:i + 1, :, :] - albedo[:, j:j + 1, :, :]).mean()

            # 平均所有成对差异
            consistency_loss /= (K * (K - 1) / 2)

            return consistency_loss


class ResidualSparsityLoss(nn.Module):
    """
    残差稀疏性损失

    鼓励局部残差稀疏和平滑：
    1. L1损失：鼓励稀疏性
    2. TV损失：鼓励平滑性

    Args:
        None
    """

    def __init__(self):
        super(ResidualSparsityLoss, self).__init__()

    def l1_loss(self, residual: torch.Tensor) -> torch.Tensor:
        """
        L1稀疏损失

        Args:
            residual: 局部残差 [B, K, H, W]

        Returns:
            loss: 标量损失值
        """
        return torch.abs(residual).mean()

    def tv_loss(self, residual: torch.Tensor) -> torch.Tensor:
        """
        Total Variation损失

        Args:
            residual: 局部残差 [B, K, H, W]

        Returns:
            loss: 标量损失值
        """
        # x方向差分
        diff_x = residual[:, :, :, :-1] - residual[:, :, :, 1:]
        # y方向差分
        diff_y = residual[:, :, :-1, :] - residual[:, :, 1:, :]

        # TV损失
        loss = (torch.abs(diff_x).mean() + torch.abs(diff_y).mean())

        return loss

    def forward(
        self,
        local_residual: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算残差稀疏性损失

        Args:
            local_residual: 局部残差 [B, K, H, W] 或 None

        Returns:
            l1: L1损失
            tv: TV损失
        """
        if local_residual is None:
            # 如果没有局部残差，返回0
            return torch.tensor(0.0), torch.tensor(0.0)

        l1 = self.l1_loss(local_residual)
        tv = self.tv_loss(local_residual)

        return l1, tv


class PhysicalContributionLoss(nn.Module):
    """
    物理路径贡献度损失

    计算物理路径（albedo * shading）的贡献度，并惩罚过低值
    确保渲染主要依赖物理路径，而不是过度依赖残差

    Args:
        threshold: 物理贡献度阈值，低于该值将被惩罚
        weight: 损失权重
    """

    def __init__(self, threshold: float = 0.3, weight: float = 1.0):
        super(PhysicalContributionLoss, self).__init__()
        self.threshold = threshold
        self.weight = weight

    def forward(
            self,
            albedo: torch.Tensor,
            shading: torch.Tensor,
            final_render: torch.Tensor
    ) -> torch.Tensor:
        """
        计算物理贡献度损失

        Args:
            albedo: 反照率图 [B, 1, H, W]
            shading: 着色图 [B, K, H, W]
            final_render: 最终渲染图 [B, K, H, W]

        Returns:
            loss: 标量损失值
        """
        # 计算物理路径贡献（albedo * shading）
        physical_path = albedo * shading  # [B, K, H, W]

        # 计算物理路径贡献度（物理路径占最终渲染的比例）
        # 添加微小epsilon避免除零
        epsilon = 1e-6
        contribution = physical_path / (final_render + epsilon)

        # 限制贡献度在合理范围内
        contribution = torch.clamp(contribution, 0.0, 2.0)

        # 计算贡献度过低的惩罚
        # 使用ReLU惩罚低于阈值的贡献度
        loss = F.relu(self.threshold - contribution).mean()

        return loss * self.weight


class GtSupervisionLoss(nn.Module):
    """
    Phase 1 合成数据集 GT 监督损失

    对有 GT 的场景施加三项逐像素监督（全部用 mask 遮罩，只统计前景）：
      - gt_depth:  预测深度与 GT 深度的 L1（原始视空间尺度）
      - gt_albedo: 预测反照率与 GT 反照率的 L1
      - gt_normal: 预测法线（渲染器由预测深度导出）与 GT 法线的角度损失
                   （1 - cos，对方向敏感、对长度不敏感）

    阶段门控由 trainer 的阶段权重表控制：阶段 1/2 启用（几何与材质先学对），
    阶段 3 置零（自监督重建 + 残差学习主导）。

    Args:
        pred_depth:  [B,1,H,W]
        pred_albedo: [B,1,H,W]
        pred_normal: [B,3,H,W]（physics_renderer 由预测深度导出的法线）
        gt:          dict，含 depth/albedo/normal/mask 张量（data_loader 输出）

    Returns:
        dict {'gt_depth', 'gt_albedo', 'gt_normal'} -> 标量张量（已 mask 归一）
    """

    def forward(
        self,
        pred_depth: torch.Tensor,
        pred_albedo: torch.Tensor,
        pred_normal: torch.Tensor,
        gt: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        m = gt["mask"]
        if m.dim() == 3:
            m = m.unsqueeze(0)
        denom = m.sum().clamp_min(1.0)

        out = {
            "gt_depth": torch.zeros((), device=m.device),
            "gt_albedo": torch.zeros((), device=m.device),
            "gt_normal": torch.zeros((), device=m.device),
        }

        if "depth" in gt and pred_depth is not None:
            out["gt_depth"] = (torch.abs(pred_depth - gt["depth"]) * m).sum() / denom

        if "albedo" in gt and pred_albedo is not None:
            out["gt_albedo"] = (torch.abs(pred_albedo - gt["albedo"]) * m).sum() / denom

        if "normal" in gt and pred_normal is not None:
            pn = F.normalize(pred_normal, p=2, dim=1)
            gn = F.normalize(gt["normal"], p=2, dim=1)
            cos = (pn * gn).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
            out["gt_normal"] = ((1.0 - cos) * m).sum() / denom

        return out


class LossCalculator(nn.Module):
    """
    损失函数计算器

        管理所有损失函数并计算加权总损失，支持阶段感知的损失权重体系

    Args:
        weights: 损失权重字典（可选，用于覆盖默认权重）
        initial_stage: 初始阶段名称（可选，默认为stage1）
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None, initial_stage: str = 'stage1'):
        super(LossCalculator, self).__init__()
        # 1. 定义三个阶段的权重配置
        self.stage_weights = {
            'stage1': {
                # stage1: 残差权重低(0.0-0.1)，物理约束权重高(1.0-2.0)
                'reconstruction': 1.0,
                'depth_smooth': 0.01,
                'albedo_smooth': 0.01,
                'weight_dist': 0.01,
                'weight_tv': 0.01,
                'sh_l2': 0.001,
                'sh_higher': 0.002,
                'sh_sparsity': 0.001,
                'sh0_nonneg': 2.0,  # 物理约束权重高
                'sh_energy_dist': 1.5,  # 物理约束权重高
                'sh_total_energy': 1.5,  # 物理约束权重高
                'residual_l1': 0.0,  # 残差权重低
                'residual_tv': 0.0,  # 残差权重低
                'physical_contribution': 2.0  # 物理约束权重高
            },
            'stage2': {
                # stage2: 残差权重中(0.3)，物理约束权重中(0.5)
                'reconstruction': 1.0,
                'depth_smooth': 0.01,
                'albedo_smooth': 0.01,
                'weight_dist': 0.01,
                'weight_tv': 0.01,
                'sh_l2': 0.001,
                'sh_higher': 0.002,
                'sh_sparsity': 0.001,
                'sh0_nonneg': 0.5,  # 物理约束权重中
                'sh_energy_dist': 0.5,  # 物理约束权重中
                'sh_total_energy': 0.5,  # 物理约束权重中
                'residual_l1': 0.3,  # 残差权重中
                'residual_tv': 0.3,  # 残差权重中
                'physical_contribution': 0.5  # 物理约束权重中
            },
            'stage3': {
                # stage3: 残差权重高(0.5)，物理约束权重低(0.1)
                'reconstruction': 1.0,
                'depth_smooth': 0.01,
                'albedo_smooth': 0.01,
                'weight_dist': 0.01,
                'weight_tv': 0.01,
                'sh_l2': 0.001,
                'sh_higher': 0.002,
                'sh_sparsity': 0.001,
                'sh0_nonneg': 0.1,  # 物理约束权重低
                'sh_energy_dist': 0.1,  # 物理约束权重低
                'sh_total_energy': 0.1,  # 物理约束权重低
                'residual_l1': 0.5,  # 残差权重高
                'residual_tv': 0.5,  # 残差权重高
                'physical_contribution': 0.1  # 物理约束权重低
            }
        }

        # 2. 初始化当前阶段
        self.current_stage = initial_stage
        # 获取当前阶段的权重
        self.weights = self.stage_weights[self.current_stage].copy()

        # 3. 更新用户提供的权重（如果有）
        if weights is not None:
                self.weights.update(weights)

        # 4. 初始化损失函数模块
        self.charbonnier_loss = CharbonnierLoss(epsilon=1e-3)
        self.depth_smoothness = DepthSmoothnessLoss(gamma=10.0)
        self.albedo_smoothness = AdaptiveAlbedoLoss()
        self.weight_regularization = WeightMapRegularization(alpha=2.0, beta=2.0)
        self.lighting_prior = LightingPriorLoss(higher_order_weight=2.0)
        self.lighting_constraint = LightingConstraintLoss()
        self.residual_sparsity = ResidualSparsityLoss()
        self.physical_contribution_loss = PhysicalContributionLoss(threshold=0.3, weight=1.0)  # 添加物理贡献度损失模块


    def compute_reconstruction_loss(
            self,
            pred: torch.Tensor,
            target: torch.Tensor
        ) -> torch.Tensor:
            """计算重建损失（Charbonnier）"""
            return self.charbonnier_loss(pred, target)


    def compute_physical_contribution(
            self,
            albedo: torch.Tensor,
            shading: torch.Tensor,
            final_render: torch.Tensor
    ) -> torch.Tensor:
        """计算物理贡献度损失"""
        return self.physical_contribution_loss(albedo, shading, final_render)


    def set_stage(self, stage_name: str):
        """动态切换损失权重配置

        Args:
            stage_name: 阶段名称，必须是 'stage1', 'stage2' 或 'stage3'
        """
        if stage_name not in self.stage_weights:
            raise ValueError(f"Invalid stage name: {stage_name}. Must be one of {list(self.stage_weights.keys())}")

        # 更新当前阶段
        self.current_stage = stage_name
        # 更新权重为当前阶段的权重
        self.weights = self.stage_weights[stage_name].copy()
        print(f"LossCalculator stage switched to: {stage_name}")
        print(f"Current weights: {self.weights}")

    def compute_depth_smoothness_loss(
        self,
        depth: torch.Tensor,
        image: torch.Tensor
    ) -> torch.Tensor:
        """计算深度平滑损失（边缘感知）"""
        return self.depth_smoothness(depth, image)

    def compute_albedo_smoothness_loss(
        self,
        albedo: torch.Tensor,
        weight_map: torch.Tensor
    ) -> torch.Tensor:
        """计算反照率平滑损失（自适应）"""
        return self.albedo_smoothness(albedo, weight_map)

    def compute_weight_regularization_loss(
        self,
        weight_map: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算权重图正则化损失"""
        return self.weight_regularization(weight_map)

    def compute_lighting_prior_loss(
        self,
        sh_coeffs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算光照先验损失"""
        return self.lighting_prior(sh_coeffs)

    def compute_residual_sparsity_loss(
        self,
        local_residual: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算残差稀疏性损失"""
        return self.residual_sparsity(local_residual)

    def forward(
        self,
        pred_images: torch.Tensor,
        target_images: torch.Tensor,
        depth: torch.Tensor,
        albedo: torch.Tensor,
        weight_map: torch.Tensor,
        sh_coeffs: torch.Tensor,
        local_residual: Optional[torch.Tensor] = None,
        shading: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算总损失

        Args:
            pred_images: 预测图像 [B, K, H, W]
            target_images: 目标图像 [B, K, H, W]
            depth: 深度图 [B, 1, H, W]
            albedo: 反照率图 [B, 1, H, W]
            weight_map: 权重图 [B, 1, H, W]
            sh_coeffs: 球谐系数 [B, K, 9]
            local_residual: 局部残差 [B, K, H, W] 或 None
            shading: 着色图 [B, K, H, W]，用于计算物理贡献度损失
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的字典
        """
        loss_dict = {}

        # 1. 重建损失
        loss_recon = self.compute_reconstruction_loss(pred_images, target_images)
        loss_dict['reconstruction'] = loss_recon.item()

        # 2. 深度平滑损失
        loss_depth = self.compute_depth_smoothness_loss(depth, target_images)
        loss_dict['depth_smooth'] = loss_depth.item()

        # 3. 反照率平滑损失
        loss_albedo = self.compute_albedo_smoothness_loss(albedo, weight_map)
        loss_dict['albedo_smooth'] = loss_albedo.item()

        # 4. 权重图正则化
        loss_weight_dist, loss_weight_tv = self.compute_weight_regularization_loss(weight_map)
        loss_dict['weight_dist'] = loss_weight_dist.item()
        loss_dict['weight_tv'] = loss_weight_tv.item()

        # 5. 光照先验损失
        loss_sh_l2, loss_sh_higher, loss_sh_sparsity = self.compute_lighting_prior_loss(sh_coeffs)
        loss_dict['sh_l2'] = loss_sh_l2.item()
        loss_dict['sh_higher'] = loss_sh_higher.item()
        loss_dict['sh_sparsity'] = loss_sh_sparsity.item()

        # 6. 光照约束损失
        loss_sh0_nonneg, loss_sh_energy_dist, loss_sh_total_energy = self.lighting_constraint(sh_coeffs)
        loss_dict['sh0_nonneg'] = loss_sh0_nonneg.item()
        loss_dict['sh_energy_dist'] = loss_sh_energy_dist.item()
        loss_dict['sh_total_energy'] = loss_sh_total_energy.item()

        # 7. 残差稀疏性损失
        loss_res_l1, loss_res_tv = self.compute_residual_sparsity_loss(local_residual)
        loss_dict['residual_l1'] = loss_res_l1.item() if isinstance(loss_res_l1, torch.Tensor) else loss_res_l1
        loss_dict['residual_tv'] = loss_res_tv.item() if isinstance(loss_res_tv, torch.Tensor) else loss_res_tv

        # 计算加权总损失
        # 8. 物理贡献度损失
        loss_physical_contribution = 0.0
        if shading is not None:
            loss_physical_contribution = self.compute_physical_contribution(albedo, shading, pred_images)
            loss_dict['physical_contribution'] = loss_physical_contribution.item()
        else:
            loss_dict['physical_contribution'] = 0.0

        # 计算加权总损失，使用当前阶段的权重
        total_loss = (
            self.weights['reconstruction'] * loss_recon +
            self.weights['depth_smooth'] * loss_depth +
            self.weights['albedo_smooth'] * loss_albedo +
            self.weights['weight_dist'] * loss_weight_dist +
            self.weights['weight_tv'] * loss_weight_tv +
            self.weights['sh_l2'] * loss_sh_l2 +
            self.weights['sh_higher'] * loss_sh_higher +
            self.weights['sh_sparsity'] * loss_sh_sparsity +
            self.weights['sh0_nonneg'] * loss_sh0_nonneg +
            self.weights['sh_energy_dist'] * loss_sh_energy_dist +
            self.weights['sh_total_energy'] * loss_sh_total_energy +
            self.weights['residual_l1'] * loss_res_l1 +
            self.weights['residual_tv'] * loss_res_tv +
            self.weights['physical_contribution'] * loss_physical_contribution
        )

        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    """
    测试损失函数模块
    """
    print("=" * 80)
    print("Loss Functions 测试")
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

    # ========== 准备测试数据 ==========
    print("\n" + "=" * 80)
    print("【准备测试数据】")
    print("=" * 80)

    # 创建随机数据
    pred_images = torch.rand(BATCH_SIZE, NUM_IMAGES, IMAGE_SIZE, IMAGE_SIZE).to(device)
    target_images = torch.rand(BATCH_SIZE, NUM_IMAGES, IMAGE_SIZE, IMAGE_SIZE).to(device)
    depth = torch.rand(BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE).to(device)
    albedo = torch.rand(BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE).to(device)
    weight_map = torch.rand(BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE).to(device)
    sh_coeffs = torch.randn(BATCH_SIZE, NUM_IMAGES, 9).to(device) * 0.1
    local_residual = torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE).to(device) * 0.01

    print("✓ 测试数据准备完成")

    # ========== 测试 1: Charbonnier Loss ==========
    print("\n" + "=" * 80)
    print("【1】测试 Charbonnier Loss")
    print("=" * 80)

    charbonnier = CharbonnierLoss(epsilon=1e-3).to(device)
    loss_char = charbonnier(pred_images, target_images)

    print(f"\nCharbonnier Loss: {loss_char.item():.6f}")

    # 与MSE比较
    mse_loss = F.mse_loss(pred_images, target_images)
    print(f"MSE Loss (比较): {mse_loss.item():.6f}")
    print(f"✓ Charbonnier损失对异常值更鲁棒")

    # ========== 测试 2: Depth Smoothness Loss ==========
    print("\n" + "=" * 80)
    print("【2】测试 Depth Smoothness Loss (边缘感知)")
    print("=" * 80)

    depth_smooth = DepthSmoothnessLoss(gamma=10.0).to(device)
    loss_depth = depth_smooth(depth, target_images)

    print(f"\nDepth Smoothness Loss: {loss_depth.item():.6f}")
    print(f"✓ 在图像边缘处降低平滑约束")

    # ========== 测试 3: Adaptive Albedo Loss ==========
    print("\n" + "=" * 80)
    print("【3】测试 Adaptive Albedo Loss")
    print("=" * 80)

    albedo_smooth = AdaptiveAlbedoLoss().to(device)
    loss_albedo = albedo_smooth(albedo, weight_map)

    print(f"\nAdaptive Albedo Loss: {loss_albedo.item():.6f}")
    print(f"✓ 使用权重图自适应调节平滑约束")

    # ========== 测试 4: Weight Map Regularization ==========
    print("\n" + "=" * 80)
    print("【4】测试 Weight Map Regularization")
    print("=" * 80)

    weight_reg = WeightMapRegularization(alpha=2.0, beta=2.0).to(device)
    loss_w_dist, loss_w_tv = weight_reg(weight_map)

    print(f"\nWeight Map Regularization:")
    print(f"  分布损失 (KL): {loss_w_dist.item():.6f}")
    print(f"  TV损失: {loss_w_tv.item():.6f}")
    print(f"✓ 鼓励权重分布接近Beta(2,2)并保持平滑")

    # ========== 测试 5: Lighting Prior Loss ==========
    print("\n" + "=" * 80)
    print("【5】测试 Lighting Prior Loss")
    print("=" * 80)

    lighting_prior = LightingPriorLoss(higher_order_weight=2.0).to(device)
    loss_sh_l2, loss_sh_higher, loss_sh_sparsity = lighting_prior(sh_coeffs)

    print(f"\nLighting Prior Loss:")
    print(f"  L2正则化: {loss_sh_l2.item():.6f}")
    print(f"  高阶惩罚: {loss_sh_higher.item():.6f}")
    print(f"  稀疏惩罚: {loss_sh_sparsity.item():.6f}")
    print(f"✓ 对高阶球谐系数施加更强惩罚")

    # ========== 测试 6: Residual Sparsity Loss ==========
    print("\n" + "=" * 80)
    print("【6】测试 Residual Sparsity Loss")
    print("=" * 80)

    residual_sparse = ResidualSparsityLoss().to(device)
    loss_res_l1, loss_res_tv = residual_sparse(local_residual)

    print(f"\nResidual Sparsity Loss:")
    print(f"  L1稀疏损失: {loss_res_l1.item():.6f}")
    print(f"  TV平滑损失: {loss_res_tv.item():.6f}")
    print(f"✓ 鼓励局部残差稀疏且平滑")

    # 测试None情况
    loss_res_l1_none, loss_res_tv_none = residual_sparse(None)
    print(f"\n无局部残差时:")
    print(f"  L1损失: {loss_res_l1_none}")
    print(f"  TV损失: {loss_res_tv_none}")

    # ========== 测试 7: LossCalculator ==========
    print("\n" + "=" * 80)
    print("【7】测试 LossCalculator (总损失)")
    print("=" * 80)

    # 自定义权重
    custom_weights = {
        'reconstruction': 1.0,
        'depth_smooth': 0.1,
        'albedo_smooth': 0.1,
        'weight_dist': 0.01,
        'weight_tv': 0.01,
        'sh_l2': 0.001,
        'sh_higher': 0.002,
        'residual_l1': 0.01,
        'residual_tv': 0.01
    }

    loss_calculator = LossCalculator(weights=custom_weights).to(device)

    print(f"\n损失权重配置:")
    for name, weight in custom_weights.items():
        print(f"  {name}: {weight}")

    # 计算总损失
    total_loss, loss_dict = loss_calculator(
        pred_images=pred_images,
        target_images=target_images,
        depth=depth,
        albedo=albedo,
        weight_map=weight_map,
        sh_coeffs=sh_coeffs,
        local_residual=local_residual
    )

    print(f"\n损失分解:")
    for name, value in loss_dict.items():
        if name != 'total':
            print(f"  {name}: {value:.6f}")

    print(f"\n总损失: {loss_dict['total']:.6f}")
    print(f"✓ 所有损失计算成功")

    # ========== 测试 8: 梯度传播 ==========
    print("\n" + "=" * 80)
    print("【8】测试梯度传播")
    print("=" * 80)

    # 启用梯度
    pred_grad = pred_images.clone().requires_grad_(True)
    depth_grad = depth.clone().requires_grad_(True)
    albedo_grad = albedo.clone().requires_grad_(True)
    weight_grad = weight_map.clone().requires_grad_(True)
    sh_grad = sh_coeffs.clone().requires_grad_(True)
    local_res_grad = local_residual.clone().requires_grad_(True)

    # 前向传播
    total_loss_grad, _ = loss_calculator(
        pred_images=pred_grad,
        target_images=target_images,
        depth=depth_grad,
        albedo=albedo_grad,
        weight_map=weight_grad,
        sh_coeffs=sh_grad,
        local_residual=local_res_grad
    )

    # 反向传播
    total_loss_grad.backward()

    print(f"\n梯度统计:")
    print(f"  预测图像梯度范数: {pred_grad.grad.norm():.6f}")
    print(f"  深度梯度范数: {depth_grad.grad.norm():.6f}")
    print(f"  反照率梯度范数: {albedo_grad.grad.norm():.6f}")
    print(f"  权重图梯度范数: {weight_grad.grad.norm():.6f}")
    print(f"  球谐系数梯度范数: {sh_grad.grad.norm():.6f}")
    print(f"  局部残差梯度范数: {local_res_grad.grad.norm():.6f}")

    print(f"✓ 所有损失都可微分，支持反向传播")

    # ========== 测试 9: 权重调整影响 ==========
    print("\n" + "=" * 80)
    print("【9】测试权重调整对总损失的影响")
    print("=" * 80)

    # 测试不同权重配置
    weight_configs = [
        {'name': '标准配置', 'reconstruction': 1.0, 'depth_smooth': 0.1},
        {'name': '强调重建', 'reconstruction': 10.0, 'depth_smooth': 0.1},
        {'name': '强调平滑', 'reconstruction': 1.0, 'depth_smooth': 1.0}
    ]

    print(f"\n比较不同权重配置:")
    for config in weight_configs:
        name = config.pop('name')
        calc = LossCalculator(weights=config).to(device)

        with torch.no_grad():
            total, losses = calc(
                pred_images, target_images, depth, albedo,
                weight_map, sh_coeffs, local_residual
            )

        print(f"\n  {name}:")
        print(f"    总损失: {total.item():.6f}")
        print(f"    重建损失贡献: {losses['reconstruction'] * config.get('reconstruction', 1.0):.6f}")
        print(f"    深度平滑贡献: {losses['depth_smooth'] * config.get('depth_smooth', 0.1):.6f}")

    print("\n" + "=" * 80)
    print("测试完成!")
    print("=" * 80)

    print("\n【损失函数总结】")
    print("\n1. Charbonnier Loss (重建):")
    print("   - 比MSE对异常值更鲁棒")
    print("   - 公式: sqrt(x² + ε²)")

    print("\n2. Depth Smoothness Loss (深度):")
    print("   - 边缘感知的平滑约束")
    print("   - 在图像边缘处降低约束")

    print("\n3. Adaptive Albedo Loss (反照率):")
    print("   - 使用权重图自适应调节")
    print("   - 灵活控制平滑程度")

    print("\n4. Weight Map Regularization (权重图):")
    print("   - KL散度约束分布")
    print("   - TV损失保持平滑")

    print("\n5. Lighting Prior Loss (光照):")
    print("   - L2正则化所有系数")
    print("   - 高阶系数额外惩罚")

    print("\n6. Residual Sparsity Loss (残差):")
    print("   - L1鼓励稀疏")
    print("   - TV鼓励平滑")

    print("\n【使用建议】")
    print("1. 根据任务调整各损失权重")
    print("2. 训练初期可以降低正则化权重")
    print("3. 监控各项损失的相对大小")
    print("4. 使用学习率调度器配合损失权重调整")
    print("5. 验证集上评估时主要关注重建损失")

    print("\n" + "=" * 80)
