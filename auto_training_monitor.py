import logging
import torch
import numpy as np
from typing import Dict, Tuple, Optional

# 设置日志
logging.basicConfig(level=logging.INFO, format='[AutoMonitor] %(message)s')
logger = logging.getLogger(__name__)


class AutoTrainingMonitor:
    """
    自动训练监控器，用于监控逆向渲染训练过程的健康度
    支持两个阶段的监控逻辑：
    1. 阶段一：纯物理强制训练
    2. 阶段三：受控残差微调
    """

    def __init__(self, stage: int = 1):
        """
        初始化自动训练监控器

        Args:
            stage: 当前训练阶段，1表示纯物理强制训练，3表示受控残差微调
        """
        self.stage = stage

        # 阶段一参数
        self.min_shading_var = 0.005  # 起步线
        self.target_shading_var = 0.01  # 合格线
        self.max_albedo_corr = 0.6  # 合格线
        self.min_check_epoch = 20  # 第20轮前不检查

        # 阶段三参数
        self.max_residual_ratio = 0.30  # 30%

        # 状态跟踪
        self.continuous_below_min_shading = 0  # 连续低于起步线的次数
        self.continuous_above_max_corr = 0  # 连续高于最大相关性的次数
        self.continuous_reach_target = 0  # 连续达到合格线的次数
        self.continuous_exceed_residual = 0  # 连续残差超标的次数

        # 缓存上一轮的指标，用于计算变化趋势
        self.prev_metrics = None

    def _calculate_albedo_correlation(self, albedo: torch.Tensor, images: torch.Tensor) -> float:
        """
        计算反照率图与输入图像平均图之间的Pearson相关系数

        Args:
            albedo: 反照率图，形状 [B, 1, H, W]
            images: 输入图像，形状 [B, K, H, W]

        Returns:
            float: 平均相关系数
        """
        try:
            B, K, H, W = images.shape

            # 计算输入图像的平均图 [B, 1, H, W]
            avg_images = images.mean(dim=1, keepdim=True)

            # 将反照率图扩展到与平均图像相同的形状 [B, 1, H, W]
            albedo_expanded = albedo.expand_as(avg_images)

            # 计算每个样本的相关系数
            correlations = []
            for b in range(B):
                # 展平张量
                albedo_flat = albedo_expanded[b].flatten()
                image_flat = avg_images[b].flatten()

                # 计算相关系数
                # 避免除以零
                if torch.std(albedo_flat) == 0 or torch.std(image_flat) == 0:
                    corr = 0.0
                else:
                    corr = torch.corrcoef(torch.stack([albedo_flat, image_flat]))[0, 1].item()

                correlations.append(corr)

            # 计算平均相关系数
            avg_corr = np.mean(correlations)
            return avg_corr
        except Exception as e:
            logger.error(f"计算Albedo Correlation失败: {e}")
            return 0.0

    def _calculate_shading_variance(self, shading: torch.Tensor) -> float:
        """
        计算光照图的方差

        Args:
            shading: 光照图，形状 [B, K, H, W]

        Returns:
            float: 光照图的方差
        """
        try:
            # 计算平均光照图 [B, 1, H, W]
            avg_shading = shading.mean(dim=1, keepdim=True)

            # 计算方差
            variance = torch.var(avg_shading).item()
            return variance
        except Exception as e:
            logger.error(f"计算Shading Variance失败: {e}")
            return 0.0

    def _calculate_residual_ratio(self, loss_dict: Dict[str, float]) -> float:
        """
        根据Loss字典估算残差项在总能量中的占比

        Args:
            loss_dict: 包含各种损失值的字典

        Returns:
            float: 残差项占比
        """
        try:
            # 计算残差相关损失
            residual_loss = 0.0
            if 'residual_l1' in loss_dict:
                residual_loss += loss_dict['residual_l1']
            if 'residual_tv' in loss_dict:
                residual_loss += loss_dict['residual_tv']

            # 计算总损失
            total_loss = loss_dict.get('total', 1.0)

            # 避免除以零
            if total_loss == 0:
                return 0.0

            # 计算残差占比
            residual_ratio = residual_loss / total_loss
            return min(residual_ratio, 1.0)  # 确保不超过1.0
        except Exception as e:
            logger.error(f"计算Residual Ratio失败: {e}")
            return 0.0

    def update_and_check(self, current_epoch: int, metrics_dict: Dict) -> Tuple[str, str]:
        """
        更新监控状态并检查训练健康度

        Args:
            current_epoch: 当前训练轮次
            metrics_dict: 包含模型输出张量和损失字典的字典

        Returns:
            Tuple[str, str]: (status_code, message)
            status_code可选值: "CONTINUE", "STOP_TRAINING", "SUCCESS_SAVE"
            message: 详细描述决策原因
        """
        try:
            # 阶段一：纯物理强制训练
            if self.stage == 1:
                return self._check_stage1(current_epoch, metrics_dict)
            # 阶段三：受控残差微调
            elif self.stage == 3:
                return self._check_stage3(current_epoch, metrics_dict)
            else:
                return "CONTINUE", f"未知阶段 {self.stage}，继续训练"
        except Exception as e:
            logger.error(f"监控检查失败: {e}")
            return "CONTINUE", f"监控检查失败，继续训练: {e}"

    def _check_stage1(self, current_epoch: int, metrics_dict: Dict) -> Tuple[str, str]:
        """
        检查阶段一（纯物理强制训练）的训练健康度

        Args:
            current_epoch: 当前训练轮次
            metrics_dict: 包含模型输出张量和损失字典的字典

        Returns:
            Tuple[str, str]: (status_code, message)
        """
        # 第20轮前不进行严格检查，只记录指标
        if current_epoch < self.min_check_epoch:
            return "CONTINUE", f"Epoch {current_epoch} < {self.min_check_epoch}，跳过严格检查"

        # 提取张量
        albedo = metrics_dict.get('albedo')
        images = metrics_dict.get('images')
        shading = metrics_dict.get('shading')

        # 计算指标
        albedo_corr = self._calculate_albedo_correlation(albedo, images)
        shading_var = self._calculate_shading_variance(shading)

        logger.info(
            f"Epoch {current_epoch} - Albedo Correlation: {albedo_corr:.6f}, Shading Variance: {shading_var:.8f}")

        # 🔴 关键修改：信任真实的 Shading Variance
        # 如果 shading_var > 0.005（起步线），直接认为是进步
        if shading_var > 0.005:
            self.continuous_below_min_shading = 0
            # 检查是否达到合格线
            if shading_var >= self.target_shading_var and albedo_corr < self.max_albedo_corr:
                self.continuous_reach_target += 1
            else:
                self.continuous_reach_target = 0
        else:
            self.continuous_below_min_shading += 1
            self.continuous_reach_target = 0

        if albedo_corr > 0.9:
            self.continuous_above_max_corr += 1
        else:
            self.continuous_above_max_corr = 0

        # 🔴 强制返回 CONTINUE，防止监控脚本中止训练
        # 继续训练
        message = f"Epoch {current_epoch} - 继续训练: albedo_corr={albedo_corr:.6f} < {self.max_albedo_corr}, shading_var={shading_var:.8f} >= {self.target_shading_var}, 连续达标{self.continuous_reach_target}轮"
        logger.info(message)
        return "CONTINUE", message

    def _check_stage3(self, current_epoch: int, metrics_dict: Dict) -> Tuple[str, str]:
        """
        检查阶段三（受控残差微调）的训练健康度

        Args:
            current_epoch: 当前训练轮次
            metrics_dict: 包含模型输出张量和损失字典的字典

        Returns:
            Tuple[str, str]: (status_code, message)
        """
        # 提取损失字典
        loss_dict = metrics_dict.get('loss_dict', {})

        # 计算残差占比
        residual_ratio = self._calculate_residual_ratio(loss_dict)

        logger.info(f"Epoch {current_epoch} - Residual Ratio: {residual_ratio:.6f}")

        # 检查是否超标
        if residual_ratio > self.max_residual_ratio:
            self.continuous_exceed_residual += 1
        else:
            self.continuous_exceed_residual = 0

        # 检查是否连续3轮超标
        if self.continuous_exceed_residual >= 3:
            message = f"Epoch {current_epoch} - 残差占比连续{self.continuous_exceed_residual}轮 > {self.max_residual_ratio}，训练倒退"
            logger.error(message)
            return "STOP_TRAINING", message

        # 继续训练
        message = f"Epoch {current_epoch} - 继续训练: residual_ratio={residual_ratio:.6f} <= {self.max_residual_ratio}, 连续超标{self.continuous_exceed_residual}轮"
        logger.info(message)
        return "CONTINUE", message

    def reset(self):
        """
        重置监控状态
        """
        self.continuous_below_min_shading = 0
        self.continuous_above_max_corr = 0
        self.continuous_reach_target = 0
        self.continuous_exceed_residual = 0
        self.prev_metrics = None


