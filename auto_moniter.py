import torch
import numpy as np
import logging
from typing import Dict, Tuple, Optional


class AutoTrainingMonitor:
    """
    自动化训练监控脚本
    实现“严防死守”策略，自动判断训练状态并决策
    """

    STATUS_CONTINUE = "CONTINUE"
    STATUS_STOP = "STOP_TRAINING"
    STATUS_SUCCESS_SAVE = "SUCCESS_SAVE"

    def __init__(self, stage: int = 1):
        """
        初始化监控器

        Args:
            stage: 当前训练阶段 (1, 2, 3)
        """
        self.stage = stage
        self.logger = logging.getLogger("AutoMonitor")

        # 容错计数器
        self.fail_count = 0
        self.success_count = 0

        # 阶段一配置
        self.stage1_cfg = {
            'min_epochs': 20,  # 至少训练20轮才开始检查
            'stable_epochs': 5,  # 连续达标5轮才算成功
            'shading_var_min': 0.005,  # 起步线
            'shading_var_target': 0.01,  # 合格线
            'max_albedo_corr': 0.6,  # 合格线
            'redundancy': 3  # 最大容错次数
        }

        # 阶段三配置
        self.stage3_cfg = {
            'max_residual_ratio': 0.30,  # 残差占比上限
            'redundancy': 2  # 最大容错次数
        }

    def calculate_metrics(
            self,
            albedo: torch.Tensor,
            shading: torch.Tensor,
            images: torch.Tensor,
            residual_l1: float,
            total_loss: float
    ) -> Dict[str, float]:
        """
        计算核心监控指标
        """
        # 1. Albedo-Image Correlation
        # 展平并转为CPU
        albedo_flat = albedo.detach().cpu().view(-1)
        # 对多光照图像求平均作为参考
        mean_img_flat = images.mean(dim=1).detach().cpu().view(-1)

        # 计算Pearson相关系数
        # Center
        albedo_centered = albedo_flat - albedo_flat.mean()
        img_centered = mean_img_flat - mean_img_flat.mean()

        covariance = (albedo_centered * img_centered).sum()
        std_albedo = (albedo_centered ** 2).sum().sqrt()
        std_img = (img_centered ** 2).sum().sqrt()

        corr = (covariance / (std_albedo * std_img + 1e-8)).item()

        # 2. Shading Variance
        shading_var = shading.var().item()

        # 3. Residual Ratio (估算)
        # 计算Lambertian能量
        lambertian_energy = (albedo * shading).abs().mean().item()

        # 残差比例计算：只有当使用了残差模块时才计算，否则残差比例为0
        residual_ratio = 0.0
        if residual_l1 > 0 and total_loss > 0:
            # 仅当残差L1损失大于0时，才使用Loss比例作为代理指标
            # 使用更合理的比例计算：残差L1损失 / 总损失
            residual_ratio = (residual_l1 / total_loss) * 5.0  # 经验系数5.0用于估算
        # 注意：在阶段1和阶段2中，残差模块被完全切断，所以residual_l1应该为0，residual_ratio也会为0

        return {
            'albedo_corr': corr,
            'shading_var': shading_var,
            'residual_ratio': residual_ratio
        }

    def check_stage_1(
            self,
            current_epoch: int,
            metrics: Dict[str, float]
    ) -> Tuple[str, str]:
        """
        阶段一（纯物理）检查逻辑
        """
        cfg = self.stage1_cfg
        msg_list = []

        # 1. 早期跳过检查
        if current_epoch < cfg['min_epochs']:
            return self.STATUS_CONTINUE, f"Epoch {current_epoch} < Min {cfg['min_epochs']}, 早期放行"

        s_var = metrics['shading_var']
        a_corr = metrics['albedo_corr']

        # 2. 核心指标判断
        # 满足合格线
        is_pass = (s_var >= cfg['shading_var_target']) and (a_corr < cfg['max_albedo_corr'])

        msg = f"ShadingVar={s_var:.5f}(Target>{cfg['shading_var_target']}), AlbedoCorr={a_corr:.4f}(Target<{cfg['max_albedo_corr']})"

        if is_pass:
            self.fail_count = 0
            self.success_count += 1
            msg_list.append(f"[+] 指标合格 ({self.success_count}/{cfg['stable_epochs']})")

            # 检查是否达到稳定保存条件
            if self.success_count >= cfg['stable_epochs']:
                return self.STATUS_SUCCESS_SAVE, "连续达标，物理模型训练完成，建议保存并进行冻结操作。"
        else:
            self.success_count = 0
            # 检查是否触及起步线
            if s_var < cfg['shading_var_min'] or a_corr > 0.9:  # 0.9是明显的完全偷懒
                self.fail_count += 1
                msg_list.append(f"[-] 指标不达标 (容错: {self.fail_count}/{cfg['redundancy']})")
            else:
                # 虽然不达标，但在进步中，重置失败计数（可选策略，或者只对严重情况计数）
                self.fail_count = 0
                msg_list.append(f"[!] 调整中...")

        # 3. 容错溢出检查
        if self.fail_count >= cfg['redundancy']:
            return self.STATUS_STOP, f"训练坏死！连续 {self.fail_count} 轮 ShadingVar 极低或 AlbedoCorr 过高。"

        return self.STATUS_CONTINUE, ", ".join(msg_list)

    def check_stage_3(
            self,
            current_epoch: int,
            metrics: Dict[str, float]
    ) -> Tuple[str, str]:
        """
        阶段三（残差微调）检查逻辑
        """
        cfg = self.stage3_cfg
        r_ratio = metrics['residual_ratio']

        if r_ratio > cfg['max_residual_ratio']:
            self.fail_count += 1
            msg = f"ResidualRatio过高: {r_ratio:.2%} > {cfg['max_residual_ratio']:.2%} (容错: {self.fail_count}/{cfg['redundancy']})"
        else:
            self.fail_count = 0
            return self.STATUS_CONTINUE, f"ResidualRatio正常: {r_ratio:.2%}"

        if self.fail_count >= cfg['redundancy']:
            return self.STATUS_STOP, f"残差模块失控！占比连续 {self.fail_count} 轮超标。"

        return self.STATUS_CONTINUE, msg

    def update_and_check(
            self,
            current_epoch: int,
            metrics_dict: Dict[str, float]
    ) -> Tuple[str, str]:
        """
        主检查入口

        Returns:
            status: STATUS_CONTINUE, STATUS_STOP, or STATUS_SUCCESS_SAVE
            message: Human readable description
        """
        if self.stage == 1:
            return self.check_stage_1(current_epoch, metrics_dict)
        elif self.stage == 3:
            return self.check_stage_3(current_epoch, metrics_dict)
        else:
            return self.STATUS_CONTINUE, "阶段二为人工冻结阶段，监控暂停。"

# ============================================================================
# 集成示例 (放在 trainer.py 中)
# ============================================================================

# 1. 在 Trainer __init__ 中初始化
# from auto_monitor import AutoTrainingMonitor
# self.monitor = AutoTrainingMonitor(stage=1)

# 2. 在 validate() 函数末尾
# metrics = {
#     'albedo_corr': ...,
#     'shading_var': ...,
#     'residual_ratio': ...
# }
# status, msg = self.monitor.update_and_check(self.current_epoch, metrics)
# self.logger.info(f"[AutoMonitor] {msg}")

# if status == AutoTrainingMonitor.STATUS_STOP:
#     self.logger.error("!!! 自动监控检测到训练失败，中止训练 !!!")
#     raise SystemExit("Training stopped by AutoMonitor.")
# elif status == AutoTrainingMonitor.STATUS_SUCCESS_SAVE:
#     self.logger.info("!!! 阶段一目标达成，正在保存 Checkpoint !!!")
#     self.save_checkpoint(self.current_epoch, val_loss['total'], is_best=False)
#     raise SystemExit("Stage 1 Complete. Please freeze parameters and restart.")