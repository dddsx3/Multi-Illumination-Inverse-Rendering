"""
逆向渲染训练器
实现三阶段课程学习策略的完整训练pipeline

Author: Python Engineer
Date: 2026-01-24
"""

import os
import random
import time
from pathlib import Path
from typing import Dict, Tuple, List
import json

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

from unet_model import IntrinsicUNet
from loss_functions import LossCalculator, GtSupervisionLoss, CharbonnierLoss
from physics_renderer import PhysicsRenderer
from residual_modules import HierarchicalResidual
from evaluate import compute_all
from stability import StabilityGuard
from thermal_guard import ThermalGuard, ThermalStop, read_gpu_temp, wait_until_cool
from runtime_safety import check_host_memory, MemoryStop

# 热停机存档文件名。与 checkpoint_epoch_XXXX.pth 分开放：后者是"已完成 epoch"
# 的正式存档（评估与对比矩阵只认它），前者是"epoch 中途"的续跑状态，
# epoch 正常收尾即删除，绝不进入对比矩阵。
INTERRUPT_STATE = 'interrupt_state.pth'
def _initialize_weights(module, init_type='kaiming'):
    """内置权重初始化，替代外部 gradient_utils 依赖"""
    if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
        if init_type == 'kaiming':
            torch.nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
        elif init_type == 'xavier':
            torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            torch.nn.init.constant_(module.bias, 0)
    elif isinstance(module, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(module.weight, 1)
        torch.nn.init.constant_(module.bias, 0)
    elif isinstance(module, torch.nn.Linear):
        if init_type == 'kaiming':
            torch.nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
        elif init_type == 'xavier':
            torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            torch.nn.init.constant_(module.bias, 0)


class InverseRenderTrainer:
    """
    逆向渲染训练器

    实现三阶段课程学习策略：
    - Stage 1: 几何学习（深度、光照）
    - Stage 2: 材质学习（反照率、权重）
    - Stage 3: 残差学习（非朗伯效应）

    Args:
        model: IntrinsicUNet模型
        renderer: PhysicsRenderer渲染器
        residual: HierarchicalResidual残差模块
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        config: 训练配置字典
    """

    def __init__(
            self,
            model: IntrinsicUNet,
            renderer: PhysicsRenderer,
            residual: torch.nn.Module,
            train_loader,
            val_loader,
            config: Dict
    ):
        self.model = model
        self.renderer = renderer
        self.residual = residual
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # Phase 1 (T1.4)：GT 监督损失（合成数据集；无 GT 的 batch 自动跳过）
        self.gt_loss = GtSupervisionLoss()
        self.charb = CharbonnierLoss()

        # T2.1（C3）：稳定性守卫——NaN 连续跳过停机 + 梯度范数两级阈值
        # （默认值与 docs/design/t2_1_params.md 声明一致）
        self.stability = StabilityGuard(
            nan_streak_limit=self.config.get('nan_abort_streak', 10),
            warn_threshold=self.config.get('grad_norm_warn_threshold', 1e3),
            abort_threshold=self.config.get('grad_norm_abort_threshold', 1e4),
            on_abort=lambda: self.writer.flush(),
            tb_scalar=lambda name, val: self.writer.add_scalar(
                f'train/{name}', val, self.global_step)
        )

        # 强制使用GPU
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA不可用！请确保已安装CUDA版本的PyTorch并且有可用的GPU。")

        self.device = torch.device('cuda')
        print(f"训练器使用设备: {self.device}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

        # 权重初始化
        init_type = self.config.get('weight_init', 'xavier')
        self.model.apply(lambda m: _initialize_weights(m, init_type))
        if self.residual is not None:
            self.residual.apply(lambda m: _initialize_weights(m, init_type))
            # 保持 LocalResidualNet 末层零初始化（残差从零开始，课程学习平滑引入）
            if getattr(self.residual, 'local_net', None) is not None:
                torch.nn.init.zeros_(self.residual.local_net.net[2].weight)
                torch.nn.init.zeros_(self.residual.local_net.net[2].bias)
        print(f"权重初始化完成，类型: {init_type}")

        self.model.to(self.device)
        self.renderer.to(self.device)
        if self.residual is not None:
            self.residual.to(self.device)

        self.current_epoch = 0
        self.current_stage = 1
        self.best_val_loss = float('inf')
        self.global_step = 0

        # 梯度裁剪配置
        self.grad_clip = self.config.get('gradient_clip', 1.0)
        print(f"梯度裁剪配置: clip={self.grad_clip}")

        # 反照率平滑激进疗法配置
        self.aggressive_albedo_smooth = self.config.get('aggressive_albedo_smooth', False)
        print(f"反照率平滑激进疗法: {'启用' if self.aggressive_albedo_smooth else '禁用'}")

        # 残差模块在阶段1/2保持冻结（课程学习：先学几何与材质），
        # 进入阶段3（残差学习）时在 train() 中解冻
        if self.residual is not None:
            print("🔴 冻结残差模块（阶段1/2），进入阶段3时解冻")
            for param in self.residual.parameters():
                param.requires_grad = False
        else:
            print("🔴 残差模块为 None，使用纯物理渲染")

        # 相关状态变量
        self.albedo_correlation_history = []
        self.original_albedo_smooth_weight = None
        self.temp_albedo_smooth_weight = None
        
        # 达标检查相关变量
        self.continuous_qualified_epochs = 0
        self.qualified_threshold = 10  # 连续10个epoch达标
        self.epochs_below_correlation_threshold = 0

        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_loss_calculator()
        self._setup_logging()

        self.stage_configs = self._define_stage_configs()
        self._update_loss_weights()
        self._log_stage_transition()

    def _setup_optimizer(self):
        """设置优化器 - 使用改进的参数"""
        lr = self.config.get('learning_rate', 5e-5)  # 降低初始学习率
        weight_decay = self.config.get('weight_decay', 1e-6)  # 降低权重衰减

        # 创建参数组
        param_groups = [
            {'params': self.model.parameters(), 'lr': lr}
        ]
        
        # 只有当残差模块存在时，才添加其参数
        if self.residual is not None:
            param_groups.append(
                {'params': self.residual.parameters(), 'lr': lr}
            )

        # 使用更稳定的优化器参数
        self.optimizer = optim.Adam(
            param_groups,
            betas=(0.9, 0.999),
            weight_decay=weight_decay,
            eps=1e-8  # 增加eps值，提高数值稳定性
        )

        print(f"优化器设置完成: Adam, lr={lr}, weight_decay={weight_decay}")

        # 混合精度训练配置（Phase 1 T1.5：支持 BF16，规避 FP16 溢出）
        self.use_amp = self.config.get('use_amp', False)
        self.amp_dtype = str(self.config.get('amp_dtype', 'bfloat16')).lower()
        if self.amp_dtype not in ('bfloat16', 'float16'):
            self.amp_dtype = 'bfloat16'
        # 无条件初始化（fp32 路径也要有定义，train_epoch 的 autocast 引用它）
        self._autocast_dtype = torch.bfloat16 if self.amp_dtype == 'bfloat16' else torch.float16
        self._use_scaler = False
        if self.use_amp:
            if self.amp_dtype == 'bfloat16':
                # 只认**原生** BF16（张量核心自 Ampere sm_80 起）。
                # torch.cuda.is_bf16_supported() 在 Turing(sm_75，如 T4) 上可能
                # 因"仿真支持"返回 True，实测吞吐会塌到不可用——必须按算力判定。
                _cc = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
                if _cc[0] >= 8:
                    print("混合精度: bfloat16（指数位同 fp32，无溢出风险；无需 GradScaler）")
                else:
                    print(f"⚠ 该 GPU 算力 {_cc[0]}.{_cc[1]} 无原生 BF16（需 sm_80+），"
                          f"回退 float16 + GradScaler。注意：fp16 与既有 bf16 基线"
                          f"不同数值口径，跨臂对比前须声明（D10）")
                    self.amp_dtype = 'float16'
            if self.amp_dtype == 'float16':
                self._use_scaler = True
                # 新式 API（torch>=2.4）；旧版回退保持兼容
                try:
                    self.optimizer.scaler = torch.amp.GradScaler('cuda')
                except (AttributeError, TypeError):
                    self.optimizer.scaler = torch.cuda.amp.GradScaler()
                print("混合精度: float16 + GradScaler")
            self._autocast_dtype = torch.bfloat16 if self.amp_dtype == 'bfloat16' else torch.float16

    def _setup_scheduler(self):
        """设置学习率调度器"""
        scheduler_type = self.config.get('scheduler', 'cosine')
        lr = self.config.get('learning_rate', 5e-5)

        if scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('step_size', 30),
                gamma=self.config.get('gamma', 0.5)
            )
        elif scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('total_epochs', 100),
                eta_min=self.config.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True
            )
        else:
            self.scheduler = None

        print(f"调度器设置完成: {scheduler_type}")

    def _setup_loss_calculator(self):
        """设置损失计算器"""
        self.loss_calculator = LossCalculator()

    def _setup_logging(self):
        """设置日志记录"""
        self.log_dir = Path(self.config.get('log_dir', '../logs'))
        self.checkpoint_dir = Path(self.config.get('checkpoint_dir', '../checkpoints'))
        self.vis_dir = Path(self.config.get('vis_dir', '../visualizations'))

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.vis_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(str(self.log_dir))

        # 温度墙守卫（低散热平台）：批次间巡检，越线时先落盘再退出。
        # 阈值/开关见 thermal_guard.py 的环境变量说明。
        self.thermal = ThermalGuard.from_env()
        if self.thermal.enabled:
            print(f"🌡️  温度墙守卫已启用：≥{self.thermal.limit_c}°C 强制存档停机，"
                  f"安全续跑温度 ≤{self.thermal.resume_c}°C，"
                  f"巡检间隔 {self.thermal.poll_interval_s}s")
        # epoch 中途续跑状态：resume_batches>0 表示本 epoch 已训过这么多 batch，
        # 由 _apply_interrupt_state() 填入，train_epoch() 消费后立即清零。
        self._resume_batches = 0
        self._resume_partial = None

    def _define_stage_configs(self) -> Dict:
        """
        定义各阶段的配置

        Returns:
            字典包含每个阶段的epoch范围和loss权重
        """
        # INC-0010 修复：阶段边界适配 10 epoch 预算
        # 原配置 stage1_end=30, stage2_end=60 在 10 epoch 预算下整个训练被困 stage1，
        #   导致 sh_l2: 0.1 (stage1 高压制) 把 SH 系数强拉到 0 → SH[0]=0.0000。
        # 适配：保持阶段时长与训练预算成比例（4:5:1），使每个阶段都能被走到。
        # 数学等价性：原阶段边界的 epoch 标度被等比例缩放，各阶段的 loss 权重表不变，
        #   仅 end_epoch 数值按比例调整。
        total = self.config.get('total_epochs', 100)
        # INC-0010 修复：阶段边界自适应（绝对 epoch 边界）
        # 关键约束（config.py:113）：stage1_epochs + stage2_epochs < total_epochs
        #   （配置项之和；不是 stage1_end + stage2_end！）
        # 阶段边界语义：
        #   stage1_end = stage1 结束的绝对 epoch 编号 = stage1 的长度
        #   stage2_end = stage2 结束的绝对 epoch 编号 = stage1_end + stage2 的长度
        # 原配置：stage1=30, stage2=30 → 约束 30+30=60 < 100 满足，但
        #   在 total=10 时 30+30=60 不 < 10，需要自适应压缩。
        s1_cfg = self.config.get('stage1_epochs', 30)
        s2_cfg = self.config.get('stage2_epochs', 30)
        if s1_cfg + s2_cfg < total:
            # 默认配置满足约束
            stage1_end = s1_cfg
            stage2_end = s1_cfg + s2_cfg
        else:
            # 按 4:5:1 比例压缩到 total-1（给 stage3 至少 1 epoch）
            # 4:5:1 → stage1=4/10, stage2=5/10, stage3=1/10
            # 长度 floor 后求绝对边界
            s1_len = max(1, (total - 1) * 4 // 10)
            s2_len = max(1, (total - 1) * 5 // 10)
            # 兜底：保证 s1_cfg_compressed + s2_cfg_compressed < total
            if s1_len + s2_len >= total:
                # 极端情况：回退到 1:1 划分
                s1_len = max(1, (total - 1) // 2)
                s2_len = max(1, total - 1 - s1_len)
            stage1_end = s1_len
            stage2_end = s1_len + s2_len

        return {
            1: {
                'name': 'Geometry Learning',
                'end_epoch': stage1_end,
                
                # === 🔴 核心修正：Albedo 粉碎机 ===
                # 这一次，我们要把 Albedo 彻底磨平
                'loss_weights': {
                    # === 基础 ===
                    'reconstruction': 1.0,
                    
                    # === 反照率平滑（T1.5 稳定性修复）===
                    # 原值 50 在 AMP/真实数据下引发梯度爆炸与永久 NaN
                    # （实测 epoch 35 起全 nan）；降至 10 保持主导正则地位，
                    # 且阶段1 已有 GT 监督约束反照率，不再需要极端权重
                    'albedo_smooth': 10.0,
                    # 这将迫使 Albedo 降到 0.1 以下的纯色板
                    
                    # === 光照：强制干活 ===
                    'shading_gradient': 5.0,   # 保持引导，让 Shading 画阴影
                    
                    # === 光照数值 ===
                    'shading_mean': 1.0,      # 锁定均值 1.0
                    
                    # === 其他 ===
                    'sh_l2': 0.1,             # 防止系数爆炸
                    'sh_higher': 0.002,
                    'sh_sparsity': 0.001,
                    'weight_dist': 0.0,
                    'albedo_image_correlation': 0.0, # 暂时关闭，以免干扰粉碎过程
                    
                    # === 几何 ===
                    'depth_smooth': 0.001,
                    
                    # === 保持锚定 ===
                    'shading_variance_loss': 0.0, # 暂时关闭，优先粉碎
                    
                    # === 其他 ===
                    'weight_dist': 0.0,
                    'weight_tv': 0.0,
                    'shading_smooth_weight': 0.0,
                    'retinex_constraint_weight': 0.0,
                    'albedo_image_correlation': 0.0,
                    'physical_contribution': 0.0,
                    
                    # === 残差 ===
                    'residual_l1': 0.0,
                    'residual_tv': 0.0,

                    # === Phase 1: GT 监督（阶段1/2 启用）===
                    # 深度原始尺度 L1 权重取小值：深度数值量级 ~O(1)，
                    # 避免压过自监督重建项
                    'gt_depth': 0.05,
                    'gt_albedo': 0.5,
                    'gt_normal': 0.5,

                    # === T2.2（F 系列）：逐光照反照率 ===
                    'recon_per_light': 0.25,
                    'delta_l1': 0.0,
                },
                'description': 'Albedo 粉碎机 + GT 监督：几何/材质向真值对齐'
            },
            2: {
                'name': 'Material Learning',
                'end_epoch': stage2_end,
                'loss_weights': {
                    'reconstruction': 1.0,
                    'depth_smooth': 0.1,
                    'albedo_smooth': 0.1,
                    'weight_dist': 0.01,
                    'weight_tv': 0.01,
                    'sh_l2': 0.001,
                    'sh_higher': 0.002,
                    'sh_sparsity': 0.001,
                    'sh0_nonneg': 1.0,
                    'sh_energy_dist': 1.0,
                    'sh_total_energy': 1.0,
                    'residual_l1': 0.0,
                    'residual_tv': 0.0,

                    # === Phase 1: GT 监督（阶段1/2 启用）===
                    'gt_depth': 0.05,
                    'gt_albedo': 0.5,
                    'gt_normal': 0.5,

                    # === T2.2（F 系列）：阶段2 半权 ===
                    'recon_per_light': 0.5,
                    'delta_l1': 0.01
                },
                'description': '学习材质属性 + GT 监督：反照率/法线向真值对齐'
            },
            3: {
                'name': 'Residual Learning',
                'end_epoch': float('inf'),
                'loss_weights': {
                    'reconstruction': 1.0,
                    'depth_smooth': 0.1,
                    'albedo_smooth': 0.1,
                    'weight_dist': 0.01,
                    'weight_tv': 0.01,
                    'sh_l2': 0.001,
                    'sh_higher': 0.002,
                    'sh_sparsity': 0.001,
                    'sh0_nonneg': 1.0,
                    'sh_energy_dist': 1.0,
                    'sh_total_energy': 1.0,
                    'residual_l1': 0.01,
                    'residual_tv': 0.01,

                    # === Phase 1: GT 监督（阶段3 关闭：自监督重建+残差主导）===
                    'gt_depth': 0.0,
                    'gt_albedo': 0.0,
                    'gt_normal': 0.0,

                    # === T2.2（F 系列）：阶段3 全权（逐光照反照率成熟期）===
                    'recon_per_light': 0.5,
                    'delta_l1': 0.05
                },
                'description': '学习非朗伯效应，引入残差建模'
            }
        }

    def _get_current_stage(self) -> int:
        """根据当前epoch确定训练阶段（epoch 从 0 计数，用半开区间 [0, end)）"""
        if self.current_epoch < self.stage_configs[1]['end_epoch']:
            return 1
        elif self.current_epoch < self.stage_configs[2]['end_epoch']:
            return 2
        else:
            return 3

    def _update_loss_weights(self):
        """更新损失计算器的权重"""
        stage_config = self.stage_configs[self.current_stage]
        self.loss_calculator.weights.update(stage_config['loss_weights'])

    def _log_stage_transition(self):
        """记录阶段切换信息"""
        stage_config = self.stage_configs[self.current_stage]
        print(f"\n{'='*80}")
        print(f"进入阶段 {self.current_stage}: {stage_config['name']}")
        print(f"描述: {stage_config['description']}")
        print(f"损失权重: {stage_config['loss_weights']}")
        print(f"{'='*80}")

        self.writer.add_text(
            f'Stage_{self.current_stage}',
            f"Epoch {self.current_epoch}: {stage_config['name']}\n"
            f"Description: {stage_config['description']}\n"
            f"Loss weights: {json.dumps(stage_config['loss_weights'], indent=2)}",
            self.current_epoch
        )

    @staticmethod
    def _resolve_recon_target(images, gt_dev):
        """T2.2 RGB 双链路：网络输入原样透传（rgb 为 [B,K,3,H,W]）；
        重建损失与图像指标的对比目标用编码域 BT.709 luma——与灰度 PNG
        的推导公式逐位一致（见 _regen_gray.py），保证 F-N5-rgb 与灰度
        臂的监督口径同源。灰度模态原样返回。"""
        if images.dim() == 4:
            return images
        if gt_dev is not None and 'image_luma' in gt_dev:
            return gt_dev['image_luma']
        return (0.2126 * images[:, :, 0] + 0.7152 * images[:, :, 1]
                + 0.0722 * images[:, :, 2])

    def train_epoch(self) -> Dict[str, float]:
        """
        训练一个epoch

        Returns:
            字典包含各项损失的平均值
        """
        self.model.train()
        if self.residual is not None:
            self.residual.train()

        epoch_losses = {
            'total': 0.0,
            'reconstruction': 0.0,
            'depth_smooth': 0.0,
            'albedo_smooth': 0.0,
            'weight_dist': 0.0,
            'weight_tv': 0.0,
            'sh_l2': 0.0,
            'sh_higher': 0.0,
            'residual_l1': 0.0,
            'residual_tv': 0.0,
            'gt_depth': 0.0,
            'gt_albedo': 0.0,
            'gt_normal': 0.0
        }
        # 反照率质量指标
        albedo_grad_l1_total = 0.0
        albedo_image_corr_total = 0.0
        quality_metric_count = 0

        num_batches = len(self.train_loader)

        # ── epoch 中途续跑 ─────────────────────────────────────────────
        # 上次是被温度墙在 epoch 中途打断的：恢复已累计的损失/指标，
        # 本轮只补跑剩下的 batch 数（见 _save_interrupt_state 的口径说明）。
        resumed_batches = self._resume_batches
        self._resume_batches = 0
        if resumed_batches > 0:
            part = self._resume_partial or {}
            for key, val in (part.get('epoch_losses') or {}).items():
                if key in epoch_losses:
                    epoch_losses[key] = float(val)
            albedo_grad_l1_total = float(part.get('albedo_grad_l1_total', 0.0))
            albedo_image_corr_total = float(part.get('albedo_image_corr_total', 0.0))
            quality_metric_count = int(part.get('quality_metric_count', 0))
            self._resume_partial = None
            print(f"↩️  epoch {self.current_epoch} 中途续跑：已完成 {resumed_batches}"
                  f"/{num_batches} batch，本轮补跑 {num_batches - resumed_batches} 个")
        batches_to_run = max(0, num_batches - resumed_batches)

        for batch_idx, (images, gt, scene_names) in enumerate(self.train_loader):
            if batch_idx >= batches_to_run:
                break     # 中途续跑：本 epoch 的配额已补满
            images = images.to(self.device)
            gt_dev = None
            if gt is not None:
                gt_dev = {k: v.to(self.device) for k, v in gt.items()}

            B, K, H, W = *images.shape[:2], *images.shape[-2:]
            recon_target = self._resolve_recon_target(images, gt_dev)

            self.optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=self._autocast_dtype):
                out = self.model(images)
                is_fusion = len(out) == 6
                if is_fusion:
                    depth, albedo, sh_coeffs, weight_map, features, albedo_pl = out
                else:
                    depth, albedo, sh_coeffs, weight_map, features = out

                rendered, normal, shading = self.renderer(depth, albedo, sh_coeffs)

                # T2.2（F 系列）：逐光照反照率参与该光照的渲染重建。
                # A_k 独立于共享主反照率，直接与该光照的 shading 相乘。

                # 🔴【关键修改】如果不使用残差，强制纯物理渲染
                if self.residual is not None:
                    final_render, global_residual, local_residual = self.residual(
                        albedo, shading, normal, sh_coeffs,
                        stage=f'stage{self.current_stage}',
                        features=features
                    )
                    # 计算残差 Loss (Residual L1, TV)
                    loss_res_l1 = self.loss_calculator.weights.get('residual_l1', 0.0) * torch.abs(global_residual).mean() \
                        if global_residual is not None else torch.tensor(0.0, device=self.device)
                    loss_res_tv = self.loss_calculator.weights.get('residual_tv', 0.0) * (
                        torch.abs(global_residual[:, :, :, :-1] - global_residual[:, :, :, 1:]).mean() +
                        torch.abs(global_residual[:, :, :-1, :] - global_residual[:, :, 1:, :]).mean()
                    ) if global_residual is not None else torch.tensor(0.0, device=self.device)
                else:
                    # 🔴 强制使用纯物理渲染
                    final_render = rendered
                    global_residual = None
                    local_residual = None
                    # Residual Loss 为 0
                    loss_res_l1 = torch.tensor(0.0, device=self.device)
                    loss_res_tv = torch.tensor(0.0, device=self.device)

                total_loss, loss_dict = self.loss_calculator(
                    pred_images=final_render,
                    target_images=recon_target,
                    depth=depth,
                    albedo=albedo,
                    weight_map=weight_map,
                    sh_coeffs=sh_coeffs,
                    local_residual=local_residual,
                    shading=shading
                )
                
                # 🔴【关键修改】确保 shading_mean 计算生效
                # 计算光照均值
                mean_val = shading.mean()
                # 强制均值为 1.0
                loss_shading_mean = (mean_val - 1.0) ** 2
                # 在 total_loss 中加上
                total_loss = total_loss + self.loss_calculator.weights.get('shading_mean', 0.0) * loss_shading_mean

                # Phase 1 (T1.4)：GT 监督损失。无 GT 的 batch 自动跳过；
                # 阶段门控通过权重表实现（阶段3 权重为 0）。
                if gt_dev is not None:
                    gt_terms = self.gt_loss(depth, albedo, normal, gt_dev)
                    for _gk in ('gt_depth', 'gt_albedo', 'gt_normal'):
                        _gw = self.loss_calculator.weights.get(_gk, 0.0)
                        if _gw > 0.0:
                            total_loss = total_loss + _gw * gt_terms[_gk]
                        loss_dict[_gk] = float(gt_terms[_gk].item())

                # T2.2（F 系列）：逐光照反照率参与渲染重建 + DeltaA L1 正则。
                # 阶段门控经权重表 recon_per_light / delta_l1
                # （阶段1 关、阶段2 半权、阶段3 全权；见阶段配置表）。
                if is_fusion:
                    a4 = albedo_pl.squeeze(2)                       # [B,N,H,W]
                    rendered_pl = a4 * shading
                    _w_rpl = self.loss_calculator.weights.get('recon_per_light', 0.0)
                    if _w_rpl > 0.0:
                        _l_rpl = self.charb(rendered_pl, recon_target)
                        total_loss = total_loss + _w_rpl * _l_rpl
                        loss_dict['recon_per_light'] = float(_l_rpl.item())
                    _w_da = self.loss_calculator.weights.get('delta_l1', 0.0)
                    if _w_da > 0.0:
                        _l_da = (albedo_pl - albedo.unsqueeze(1)).abs().mean()
                        total_loss = total_loss + _w_da * _l_da
                        loss_dict['delta_l1'] = float(_l_da.item())

            # T1.5 NaN 守卫（INC-0001 防复发机制）：非有限损失直接跳过该
            # batch；连续超过阈值判定为发散，快速失败优于烧卡
            # T2.1：守卫逻辑已抽取为 StabilityGuard（stability.py），
            # 连续非有限损失达到 nan_abort_streak 即抛 RuntimeError 停机
            if not self.stability.check_loss(total_loss):
                self.optimizer.zero_grad(set_to_none=True)
                loss_dict['_skipped_nan'] = 1.0
                self.global_step += 1
                continue

            if self._use_scaler:
                self.optimizer.scaler.scale(total_loss).backward()
                # INC-0007：GradScaler 路径必须先 unscale_ 再裁剪/检查，
                # 否则拿到的是被放大 2^16 倍的梯度——裁剪值失真，且 fp16
                # 溢出会立刻让范数变成 inf/nan。
                self.optimizer.scaler.unscale_(self.optimizer)
            else:
                total_loss.backward()

            # 梯度裁剪
            if self.residual is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for group in [self.model.parameters(), self.residual.parameters()] for p in group],
                    self.grad_clip
                )
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.grad_clip
                )

            # C3/T2.1：梯度范数两级阈值（>1e3 预警写 TB；>1e4 硬停机），
            # 逻辑在 StabilityGuard.check_grad_norm（可单测）
            # INC-0007：fp16 + GradScaler 下，偶发非有限梯度是缩放因子标定的
            # 正常现象（scaler.step 会跳过该次更新并下调 scale），不构成发散。
            # 因此该路径改为"跳过并计数"，只有连续多次才由损失守卫判定发散；
            # bf16/fp32 路径的硬停机语义完全不变。
            if self._use_scaler and not torch.isfinite(grad_norm):
                self.stability.note_scaler_overflow()
                loss_dict['_scaler_overflow'] = 1.0
                self.optimizer.scaler.step(self.optimizer)   # 内部检测到 inf 会跳过
                self.optimizer.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
                continue
            if self._use_scaler:
                self.stability.note_scaler_ok()
            self.stability.check_grad_norm(grad_norm)

            # 参数更新
            if self._use_scaler:
                self.optimizer.scaler.step(self.optimizer)
                self.optimizer.scaler.update()
            else:
                self.optimizer.step()

            # 热节流（低散热平台）：每 batch 后微暂停压低平均功耗。
            # 由环境变量 THERMAL_PACE 控制（秒/批次），默认 0=关闭；
            # 不触碰任何超参与随机流，不影响 D10 单变量口径（只改墙钟）。
            _pace = float(os.environ.get('THERMAL_PACE', '0') or 0)
            if _pace > 0:
                time.sleep(_pace)

            for key in epoch_losses:
                if key in loss_dict:
                    epoch_losses[key] += loss_dict[key]
                    # 计算反照率质量指标
            with torch.no_grad():
                # 计算反照率梯度的L1范数
                albedo_grad_x = albedo[:, :, 1:, :] - albedo[:, :, :-1, :]
                albedo_grad_y = albedo[:, :, :, 1:] - albedo[:, :, :, :-1]
                albedo_grad_l1 = (albedo_grad_x.abs().mean() + albedo_grad_y.abs().mean()) / 2
                albedo_grad_l1_total += albedo_grad_l1.item()
                # 计算反照率与输入图像的相关系数
                # 将albedo扩展为与images相同的形状 (B, K, H, W)
                albedo_expanded = albedo.expand(B, K, H, W)
                # 计算每个样本、每个光照下的相关性
                corr_list = []
                for b in range(B):
                    for k in range(K):
                        albedo_flat = albedo_expanded[b, k].flatten()
                        image_flat = recon_target[b, k].flatten()
                        corr = torch.corrcoef(torch.stack([albedo_flat, image_flat]))[0, 1]
                        corr_list.append(corr.item())
                albedo_image_corr = sum(corr_list) / len(corr_list)
                albedo_image_corr_total += albedo_image_corr
                quality_metric_count += 1

                # 每10个批次检查反照率-图像相关性
                if self.aggressive_albedo_smooth and batch_idx % 10 == 0:
                    if self.original_albedo_smooth_weight is None:
                        self.original_albedo_smooth_weight = self.loss_calculator.weights.get('albedo_smooth',
                                                                                                      0.0)
                        self.temp_albedo_smooth_weight = self.original_albedo_smooth_weight
                    if albedo_image_corr > 0.7:
                        # 将albedo_smooth_weight临时提高50%
                        new_weight = self.original_albedo_smooth_weight * 1.5
                        self.loss_calculator.weights['albedo_smooth'] = new_weight
                        self.temp_albedo_smooth_weight = new_weight
                        print(f"批次 {batch_idx}: 反照率-图像相关性 {albedo_image_corr:.4f} > 0.7，将albedo_smooth_weight提高到 {new_weight:.4f}")

            self.global_step += 1

            if batch_idx % self.config.get('log_interval', 10) == 0:
                self._log_training_step(batch_idx, num_batches, loss_dict, total_loss, grad_norm)

            # 温度墙巡检 + 主机内存熔断（INC-0014 后新增内存项）：放在本 batch
            # 全部状态更新之后，保证越线时落盘的是一个自洽的"已完成 N 个 batch"状态。
            # 任一越线 → 存档 interrupt_state → raise → main 以 rc=42 退出续跑。
            # INC-0016 加固：epoch 0 的 worker/库加载窗口（前 10 batch）内存曲线
            # 最陡（cufft/cuDNN DLL 映射 + spawn worker 复制），该窗口内检查
            # 频率加密为每 2 batch；10 个 batch 后恢复每 10 batch 的常态间隔。
            try:
                self.thermal.poll()
                if self.current_epoch == 0 and batch_idx < 10:
                    if batch_idx % 2 == 0:
                        check_host_memory()
                elif batch_idx % 10 == 0:      # 约每 10 batch（数十秒）查一次主机内存
                    check_host_memory()
            except (ThermalStop, MemoryStop) as stop:
                done = resumed_batches + batch_idx + 1
                self._save_interrupt_state(
                    batches_done=done, num_batches=num_batches,
                    epoch_losses=epoch_losses,
                    albedo_grad_l1_total=albedo_grad_l1_total,
                    albedo_image_corr_total=albedo_image_corr_total,
                    quality_metric_count=quality_metric_count,
                    reason=str(stop), temp_c=getattr(stop, 'temp_c', None))
                raise
                # 计算平均质量指标
        if quality_metric_count > 0:
            avg_albedo_grad_l1 = albedo_grad_l1_total / quality_metric_count
            avg_albedo_image_corr = albedo_image_corr_total / quality_metric_count

            # 记录质量指标
            self.writer.add_scalar('train/albedo_grad_l1', avg_albedo_grad_l1, self.current_epoch)
            self.writer.add_scalar('train/albedo_image_corr', avg_albedo_image_corr, self.current_epoch)

            print(f"Epoch {self.current_epoch}: 反照率梯度L1范数 = {avg_albedo_grad_l1:.6f}")
            print(f"Epoch {self.current_epoch}: 反照率-图像相关性 = {avg_albedo_image_corr:.6f}")

            # 激进疗法的epoch级调整
            if self.aggressive_albedo_smooth:
                self.albedo_correlation_history.append(avg_albedo_image_corr)
                if avg_albedo_image_corr < 0.6:
                    self.epochs_below_correlation_threshold += 1
                else:
                    self.epochs_below_correlation_threshold = 0
                # 如果连续3个epoch相关性 < 0.6，恢复原始权重
                if self.epochs_below_correlation_threshold >= 3:
                    if self.original_albedo_smooth_weight is not None:
                        self.loss_calculator.weights['albedo_smooth'] = self.original_albedo_smooth_weight
                        self.temp_albedo_smooth_weight = self.original_albedo_smooth_weight
                        self.epochs_below_correlation_threshold = 0
                        print(f"Epoch {self.current_epoch}: 连续3个epoch相关性 < 0.6，恢复albedo_smooth_weight到原始值 {self.original_albedo_smooth_weight:.4f}")

        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        # Phase 1 (T1.4)：epoch 级损失分量写入 TensorBoard。
        # 逐 step 记录受 tensorboard_interval 限制，小规模冒烟可能一次都不触发；
        # GT 各项（gt_depth/gt_albedo/gt_normal）是门禁 G4 的观察项，
        # 以 epoch 步长稳定记录。
        for _gk in ('gt_depth', 'gt_albedo', 'gt_normal'):
            self.writer.add_scalar(f'train/{_gk}', epoch_losses[_gk], self.current_epoch)

        return epoch_losses

    def _log_training_step(self, batch_idx: int, num_batches: int, loss_dict: Dict, total_loss: torch.Tensor, grad_norm: torch.Tensor):
        """记录训练步骤 - 包含梯度信息"""
        print(f"Epoch [{self.current_epoch}/{self.config.get('total_epochs', 100)}] "
              f"Batch [{batch_idx}/{num_batches}] "
              f"Loss: {total_loss.item():.6f} "
              f"Grad Norm: {grad_norm.item():.4f} "
              f"LR: {self.optimizer.param_groups[0]['lr']:.8f}")

        if self.global_step % self.config.get('tensorboard_interval', 50) == 0:
            for key, value in loss_dict.items():
                self.writer.add_scalar(f'train/{key}', value, self.global_step)

            self.writer.add_scalar('train/learning_rate', self.optimizer.param_groups[0]['lr'], self.global_step)
            self.writer.add_scalar('train/grad_norm', grad_norm.item(), self.global_step)

    def validate(self) -> Tuple[Dict[str, float], Dict]:
        """
        在验证集上评估模型

        Returns:
            元组包含（损失字典，可视化结果）
        """
        self.model.eval()
        if self.residual is not None:
            self.residual.eval()

        val_losses = {
            'total': 0.0,
            'reconstruction': 0.0,
            'depth_smooth': 0.0,
            'albedo_smooth': 0.0,
            'weight_dist': 0.0,
            'weight_tv': 0.0,
            'sh_l2': 0.0,
            'sh_higher': 0.0,
            'sh_sparsity': 0.0,
            'sh0_nonneg': 0.0,
            'sh_energy_dist': 0.0,
            'sh_total_energy': 0.0,
            'residual_l1': 0.0,
            'residual_tv': 0.0
        }

        num_batches = len(self.val_loader)

        vis_results = {}
        # Phase 1 (T1.4)：量化指标累计（evaluate.compute_all 的 13 项）
        metric_acc = {}

        with torch.no_grad():
            for batch_idx, (images, gt, scene_names) in enumerate(self.val_loader):
                images = images.to(self.device)
                gt_dev = None
                if gt is not None:
                    gt_dev = {k: v.to(self.device) for k, v in gt.items()}

                B, K, H, W = *images.shape[:2], *images.shape[-2:]
                recon_target = self._resolve_recon_target(images, gt_dev)

                out = self.model(images)
                is_fusion = len(out) == 6
                if is_fusion:
                    depth, albedo, sh_coeffs, weight_map, features, albedo_pl = out
                else:
                    depth, albedo, sh_coeffs, weight_map, features = out

                rendered, normal, shading = self.renderer(depth, albedo, sh_coeffs)

                # 🔴【关键修改】如果不使用残差，强制纯物理渲染
                if self.residual is not None:
                    final_render, global_residual, local_residual = self.residual(
                        albedo, shading, normal, sh_coeffs,
                        stage=f'stage{self.current_stage}',
                        features=features
                    )
                else:
                    # 强制设置
                    final_render = rendered
                    global_residual = torch.zeros_like(rendered) # 必须存在，为了计算 Loss 不报错
                    local_residual = None

                total_loss, loss_dict = self.loss_calculator(
                    pred_images=final_render,
                    target_images=recon_target,
                    depth=depth,
                    albedo=albedo,
                    weight_map=weight_map,
                    sh_coeffs=sh_coeffs,
                    local_residual=local_residual,
                    shading=shading
                )

                for key in val_losses:
                    if key in loss_dict:
                        val_losses[key] += loss_dict[key]

                # T2.2：逐光照重建质量入验证损失（F 系列才有意义）
                if is_fusion and gt_dev is not None:
                    a4 = albedo_pl.squeeze(2)
                    rendered_pl = a4 * shading
                    _l_rpl = self.charb(rendered_pl, recon_target)
                    val_losses['recon_per_light'] = (val_losses.get('recon_per_light', 0.0)
                                                     + float(_l_rpl.item()))

                # Phase 1 (T1.4)：GT 量化指标（验证模式中心裁剪、无增强，
                # 指标走 evaluate.compute_all；无 GT 数据集自动跳过）
                if gt_dev is not None:
                    m_dict = compute_all(
                        pred={'normal': normal, 'depth': depth,
                              'albedo': albedo, 'image': final_render},
                        gt={'normal': gt_dev['normal'], 'depth': gt_dev['depth'],
                            'albedo': gt_dev['albedo'], 'image': recon_target},
                        mask=gt_dev['mask'])
                    for mk, mv in m_dict.items():
                        if mv == mv and abs(mv) != float('inf'):  # 跳过 NaN/Inf
                            metric_acc.setdefault(mk, []).append(mv)

                if batch_idx == 0:
                    vis_results = self._collect_visualization(
                    recon_target, depth, albedo, weight_map, final_render, normal, shading, sh_coeffs, scene_names
                )

        for key in val_losses:
            val_losses[key] /= num_batches

        # 合入平均后的量化指标（键前缀 metric_，TensorBoard 自动记录）
        for mk, values in metric_acc.items():
            if len(values) > 0:
                val_losses[f'metric_{mk}'] = sum(values) / len(values)

        return val_losses, vis_results

    def _collect_visualization(
            self,
            images: torch.Tensor,
            depth: torch.Tensor,
            albedo: torch.Tensor,
            weight_map: torch.Tensor,
            rendered: torch.Tensor,
            normal: torch.Tensor,
            shading: torch.Tensor,
            sh_coeffs: torch.Tensor,
            scene_names: List[str]
    ) -> Dict:
        """
        收集可视化结果

        Args:
            images: 输入图像 [B, K, H, W]
            depth: 深度图 [B, 1, H, W]
            albedo: 反照率图 [B, 1, H, W]
            weight_map: 权重图 [B, 1, H, W]
            rendered: 渲染图像 [B, K, H, W]
            normal: 法向量 [B, 3, H, W]
            shading: 着色图 [B, K, H, W]
            sh_coeffs: 球谐系数 [B, K, 9]
            scene_names: 场景名称列表

        Returns:
            可视化结果字典
        """
        B, K, H, W = images.shape

        # 计算黄金指标
        # 1. Shading Variance
        shading_var = shading.var().item()
        
        # 2. SH[0] Mean
        sh0_mean = sh_coeffs[:, :, 0].mean().item()
        
        # 3. Albedo Correlation
        albedo_expanded = albedo.expand(B, K, H, W)
        corr_list = []
        for b in range(min(B, 1)):  # 只计算第一个样本
            for k in range(K):
                albedo_flat = albedo_expanded[b, k].flatten()
                image_flat = images[b, k].flatten()
                if len(albedo_flat) > 1:
                    corr = torch.corrcoef(torch.stack([albedo_flat, image_flat]))[0, 1].item()
                    corr_list.append(corr)
        albedo_corr = sum(corr_list) / len(corr_list) if corr_list else 1.0
        
        # 4. Lambertian Ratio（INC-0010 修复：通道归一化的能量匹配度）
        # 原式 (rendered**2).mean() / (images**2).mean() 在 RGB 模态下数值不稳定：
        #   - rendered 在 stage1 高 sh_l2 压制下 → 0，平方后分子塌陷
        #   - epoch 1 反照率分支未稳时 rendered 出现大梯度 → rendered² >> images² → 146× 爆炸
        # 数学等价变换：把"能量比"改写为"通道归一化能量匹配度"，单调映射到 [0,1]
        #   等价性：原 ratio → 1 时新 ratio → 1；原 ratio → 0 或 ∞ 时新 ratio → 0
        #   物理含义不变（仍衡量"模型渲染能量是否匹配输入图像能量"）
        eps = 1e-6
        # 沿 (B, K, H, W) 求均值，保留通道维度 [C]
        rendered_ch_energy = rendered.detach().abs().mean(dim=(0, 2, 3))
        images_ch_energy = images.detach().abs().mean(dim=(0, 2, 3))
        # 通道相对偏差 → 1 - mean(|ΔE|/E)
        ch_rel_diff = ((rendered_ch_energy - images_ch_energy).abs()
                       / (images_ch_energy + eps))
        energy_match = 1.0 - ch_rel_diff.mean().item()
        lambertian_ratio = max(0.0, min(1.0, energy_match))

        results = {
            'input_images': images[0].cpu(),
            'depth': depth[0, 0].cpu(),
            'albedo': albedo[0, 0].cpu(),
            'weight_map': weight_map[0, 0].cpu(),
            'rendered': rendered[0].cpu(),
            'normal': normal[0].cpu(),
            'shading': shading[0].cpu() if shading is not None else None,
            'scene_name': scene_names[0],
            # 黄金指标
            'shading_variance': shading_var,
            'sh0_mean': sh0_mean,
            'albedo_correlation': albedo_corr,
            'lambertian_ratio': lambertian_ratio
        }

        return results

    def save_visualizations(self, vis_results: Dict, epoch: int):
        """
        保存可视化结果

        Args:
            vis_results: 可视化结果字典
            epoch: 当前epoch
        """
        epoch_dir = self.vis_dir / f'epoch_{epoch:04d}'
        epoch_dir.mkdir(parents=True, exist_ok=True)

        scene_name = vis_results['scene_name']
        scene_dir = epoch_dir / scene_name
        scene_dir.mkdir(parents=True, exist_ok=True)

        def save_tensor(tensor, path, vmin=None, vmax=None):
            """保存tensor为图像"""
            if vmin is None:
                vmin = tensor.min()
            if vmax is None:
                vmax = tensor.max()

            tensor_norm = (tensor - vmin) / (vmax - vmin + 1e-8)
            tensor_norm = (tensor_norm * 255).clamp(0, 255).byte().numpy()
            img = Image.fromarray(tensor_norm)
            img.save(path)

        K = vis_results['input_images'].shape[0]

        for k in range(K):
            save_tensor(
                vis_results['input_images'][k],
                scene_dir / f'input_{k:02d}.png'
            )
            save_tensor(
                vis_results['rendered'][k],
                scene_dir / f'rendered_{k:02d}.png'
            )

        save_tensor(vis_results['depth'], scene_dir / 'depth.png')
        save_tensor(vis_results['albedo'], scene_dir / 'albedo.png')
        save_tensor(vis_results['weight_map'], scene_dir / 'weight_map.png')

        normal = vis_results['normal']
        normal_vis = (normal + 1) / 2
        save_tensor(normal_vis[0], scene_dir / 'normal_x.png')
        save_tensor(normal_vis[1], scene_dir / 'normal_y.png')
        save_tensor(normal_vis[2], scene_dir / 'normal_z.png')

    def _atomic_save(self, obj, path):
        """先写 .tmp 再原子改名。

        热停机存档是在"随时可能被兜底看门狗硬杀 / 整机热保护关机"的处境下写的，
        直接 torch.save 到目标名一旦写到一半断电，就得到一个能骗过存在性检查
        的半截文件，续跑时才在 torch.load 里炸。原子改名保证目标路径要么是
        上一次的完好版本，要么是这一次的完好版本。
        """
        path = Path(path)
        tmp = path.with_suffix(path.suffix + '.tmp')
        torch.save(obj, tmp)
        self._atomic_replace(tmp, path)

    @staticmethod
    def _atomic_replace(tmp: Path, path: Path, retries: int = 3):
        """os.replace 的 Windows 韧性版（INC-0014 续：陈旧目标被占用/锁定→WinError5）。

        首次失败时尝试把已存在的旧目标改名挪开（.stale_<ts>）再替换；带退避重试；
        最终仍失败才抛错（由调用方决定降级策略）。
        """
        last = None
        for attempt in range(retries):
            try:
                os.replace(tmp, path)
                return
            except OSError as exc:
                last = exc
            try:
                if os.path.exists(path):
                    alt = path.with_name(f"{path.name}.stale_{int(time.time())}_{attempt}")
                    os.rename(path, alt)
                    os.replace(tmp, path)
                    return
            except OSError:
                pass
            time.sleep(0.6 * (attempt + 1))
        raise last

    def _rng_state(self) -> Dict:
        """收集全部随机源状态，供中途续跑还原。"""
        return {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

    def _restore_rng_state(self, state: Dict):
        if not state:
            return
        try:
            if state.get('python') is not None:
                random.setstate(state['python'])
            if state.get('numpy') is not None:
                np.random.set_state(state['numpy'])
            if state.get('torch') is not None:
                torch.set_rng_state(state['torch'].cpu()
                                    if hasattr(state['torch'], 'cpu') else state['torch'])
            if state.get('cuda') is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(state['cuda'])
        except Exception as exc:      # 还原失败不该阻断续跑，只降级并声明
            print(f"⚠️  随机数状态还原失败（{exc}），本 epoch 剩余部分改用当前随机流")

    def _save_interrupt_state(self, batches_done: int, num_batches: int,
                              epoch_losses: Dict, albedo_grad_l1_total: float,
                              albedo_image_corr_total: float,
                              quality_metric_count: int, reason: str,
                              temp_c=None):
        """温度墙触发时的强制存档：把 epoch 中途的完整状态落盘。

        与 save_checkpoint 的差别：
          - 它记录的是"epoch 未完成"的状态（多存 batches_done + 累计量 + 随机流），
            因此**不**参与 epochs_done 统计、不进对比矩阵、不写 best_model；
          - epoch 正常收尾时由 _clear_interrupt_state() 删除，防止陈旧状态被误用。

        续跑口径（必须如实声明）：本 epoch 剩余 `num_batches - batches_done` 个
        batch 取自新进程重建的那份 permutation 的**前若干个**，而不是原
        permutation 里尚未训到的那些——DataLoader 的 shuffle 顺序由 sampler
        自身的 generator 决定，跨进程无法接续（现有的 epoch 级续跑同样如此）。
        因此该 epoch 覆盖的场景集合是训练集的一个随机子集，梯度步数与
        batch_size 不变，超参、阶段门控、损失权重一律不变。
        """
        state = {
            'kind': 'thermal_interrupt',
            'epoch': self.current_epoch,
            'batches_done': int(batches_done),
            'num_batches': int(num_batches),
            'global_step': self.global_step,
            'current_stage': self.current_stage,
            'best_val_loss': self.best_val_loss,
            'continuous_qualified_epochs': self.continuous_qualified_epochs,
            'model_state_dict': self.model.state_dict(),
            'renderer_state_dict': self.renderer.state_dict(),
            'residual_state_dict': self.residual.state_dict() if self.residual is not None else None,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': (self.optimizer.scaler.state_dict()
                                  if self._use_scaler and hasattr(self.optimizer, 'scaler')
                                  else None),
            'partial': {
                'epoch_losses': dict(epoch_losses),
                'albedo_grad_l1_total': float(albedo_grad_l1_total),
                'albedo_image_corr_total': float(albedo_image_corr_total),
                'quality_metric_count': int(quality_metric_count),
            },
            'rng_state': self._rng_state(),
            'loss_weights': dict(self.loss_calculator.weights),
            'reason': reason,
            'temp_c': temp_c,
            'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'config': self.config,
        }
        path = self.checkpoint_dir / INTERRUPT_STATE
        try:
            self._atomic_save(state, path)
        except OSError as exc:
            # 存档文件被占用/锁定等（INC-0014 续：WinError5）：本次停机降级为
            # "按 epoch 粒度续跑"——上层仍抛 ThermalStop/MemoryStop，rc=42 让
            # 编排器从最近 checkpoint_epoch_* 续起，不阻断停机通道。
            print(f"⚠️  中途状态存档失败（{exc}）：{path}——本次按 epoch 粒度续跑，"
                  f"最后完好存档 = checkpoint_epoch_*")
        print(f"\n{'=' * 80}")
        print(f"🌡️  温度墙触发：{reason}")
        print(f"已强制存档 epoch {self.current_epoch} 第 {batches_done}/{num_batches} "
              f"batch 的完整状态 -> {path}")
        print(f"等温度降到 ≤{self.thermal.resume_c}°C 后重跑同一条命令即自动接上")
        print(f"{'=' * 80}\n")
        try:
            self.writer.add_scalar('thermal/stop_temp_c', temp_c or 0, self.global_step)
            self.writer.flush()
        except Exception:
            pass

    def _clear_interrupt_state(self):
        """epoch 正常收尾：删除中途状态，避免下次启动误用陈旧断点。"""
        path = self.checkpoint_dir / INTERRUPT_STATE
        if path.is_file():
            try:
                path.unlink()
            except OSError as exc:
                print(f"⚠️  中途状态删除失败（{exc}）：{path}")

    def _apply_interrupt_state(self):
        """启动时若存在与当前起点匹配的中途状态，则接上它。

        匹配条件：state['epoch'] == self.current_epoch（即"即将开跑的那个 epoch"）。
        epoch 不匹配说明是陈旧残留（例如中途状态之后又跑完了整个 epoch），直接丢弃。
        """
        path = self.checkpoint_dir / INTERRUPT_STATE
        if not path.is_file():
            return False
        try:
            state = torch.load(path, map_location=self.device, weights_only=False)
        except Exception as exc:
            print(f"⚠️  中途状态无法读取（{exc}），忽略并从 epoch 起点开跑：{path}")
            return False
        if state.get('kind') != 'thermal_interrupt':
            return False
        if int(state.get('epoch', -1)) != int(self.current_epoch):
            print(f"ℹ️  丢弃陈旧中途状态（记录 epoch {state.get('epoch')}，"
                  f"当前起点 epoch {self.current_epoch}）")
            self._clear_interrupt_state()
            return False
        done = int(state.get('batches_done', 0))
        total = int(state.get('num_batches', 0))
        if done <= 0 or (total and done >= total):
            self._clear_interrupt_state()
            return False

        self.model.load_state_dict(state['model_state_dict'])
        self.renderer.load_state_dict(state['renderer_state_dict'])
        if self.residual is not None and state.get('residual_state_dict') is not None:
            self.residual.load_state_dict(state['residual_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        if self.scheduler is not None and state.get('scheduler_state_dict') is not None:
            self.scheduler.load_state_dict(state['scheduler_state_dict'])
        if self._use_scaler and state.get('scaler_state_dict') is not None \
                and hasattr(self.optimizer, 'scaler'):
            self.optimizer.scaler.load_state_dict(state['scaler_state_dict'])
        self.global_step = int(state.get('global_step', self.global_step))
        self.best_val_loss = state.get('best_val_loss', self.best_val_loss)
        self.continuous_qualified_epochs = int(
            state.get('continuous_qualified_epochs', self.continuous_qualified_epochs))
        # 阶段与权重按 epoch 重算（与 load_checkpoint 同口径，INC-0005）
        self.current_stage = self._get_current_stage()
        self._update_loss_weights()
        if self.residual is not None:
            _unfrozen = self.current_stage >= 3
            for param in self.residual.parameters():
                param.requires_grad = _unfrozen
        self._restore_rng_state(state.get('rng_state'))
        self._resume_batches = done
        self._resume_partial = state.get('partial') or {}
        print(f"\n🌡️  接上热停机断点：epoch {self.current_epoch} 已完成 {done}/{total} batch"
              f"（停机原因：{state.get('reason')}，存档时间 {state.get('saved_at')}）")
        print(f"  阶段 {self.current_stage}；global_step {self.global_step}；"
              f"best_val_loss {self.best_val_loss:.6f}")
        return True

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """
        保存检查点

        Args:
            epoch: 当前epoch
            val_loss: 验证损失
            is_best: 是否为最佳模型
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'renderer_state_dict': self.renderer.state_dict(),
            'residual_state_dict': self.residual.state_dict() if self.residual is not None else None,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'current_stage': self.current_stage,
            'config': self.config
        }

        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch:04d}.pth'
        self._atomic_save(checkpoint, checkpoint_path)
        print(f"检查点已保存: {checkpoint_path}")

        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            self._atomic_save(checkpoint, best_path)
            print(f"最佳模型已保存: {best_path}")

        # 总是保存最新模型
        latest_path = self.checkpoint_dir / 'latest_model.pth'
        self._atomic_save(checkpoint, latest_path)

    def load_checkpoint(self, checkpoint_path: str):
        """
        加载检查点

        Args:
            checkpoint_path: 检查点文件路径
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device,
                                weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.renderer.load_state_dict(checkpoint['renderer_state_dict'])
        if self.residual is not None and checkpoint['residual_state_dict'] is not None:
            self.residual.load_state_dict(checkpoint['residual_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint['scheduler_state_dict'] is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # INC-0006：checkpoint 里的 epoch 是**已完成**的那一个，续跑必须从
        # 下一个 epoch 开始。原实现直接赋值导致每次续跑重做最后一个 epoch
        # 并覆盖其 checkpoint（10-epoch 分段下浪费 10% 算力 + 破坏可追溯性）。
        self.current_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint['best_val_loss']

        # INC-0005：__init__ 只按阶段1初始化损失权重表并冻结残差；断点若落在
        # 阶段2/3，必须把阶段状态同步到"即将开跑的那个 epoch"，否则整个续跑段
        # 都在错误的监督口径下训练。阶段以 epoch 重算为准（checkpoint 里存的是
        # 已完成 epoch 的阶段，跨边界续跑时会差一个阶段）。
        # 训练循环只在"检测到阶段切换"时更新权重，直接恢复的场景不会触发。
        self.current_stage = self._get_current_stage()
        self._update_loss_weights()
        if self.residual is not None:
            _unfrozen = self.current_stage >= 3
            for param in self.residual.parameters():
                param.requires_grad = _unfrozen

        print(f"检查点已加载: {checkpoint_path}")
        print(f"  已完成 Epoch: {checkpoint['epoch']}　续跑起点: {self.current_epoch}")
        print(f"  Best Val Loss: {self.best_val_loss:.6f}")
        print(f"  Current Stage: {self.current_stage}")
        print(f"  损失权重表已同步到阶段{self.current_stage}; "
              f"残差模块{'解冻' if self.current_stage >= 3 else '冻结'}")
    def train(self):
        """完整训练流程"""
        total_epochs = self.config.get('total_epochs', 100)

        # 热停机断点：若上次是被温度墙在 epoch 中途打断的，在这里接上。
        # 放在 load_checkpoint 之后、循环之前——它依赖 current_epoch 已定位。
        self._apply_interrupt_state()

        print(f"\n{'='*80}")
        print(f"开始训练")
        print(f"总Epoch数: {total_epochs}")
        print(f"设备: {self.device}")
        print(f"{'='*80}")

        start_time = time.time()

        for epoch in range(self.current_epoch, total_epochs):
            self.current_epoch = epoch

            new_stage = self._get_current_stage()

            if new_stage != self.current_stage:
                prev_stage = self.current_stage
                self.current_stage = new_stage
                self._update_loss_weights()
                self._log_stage_transition()

                # 进入阶段3（残差学习）时解冻残差模块
                if new_stage == 3 and prev_stage < 3 and self.residual is not None:
                    for param in self.residual.parameters():
                        param.requires_grad = True
                    print("🔴 解冻残差模块：进入阶段3（残差学习）")

            # 训练一个epoch
            print(f"\n开始Epoch {self.current_epoch}")
            train_losses = self.train_epoch()

            # 验证与落盘期间没有 batch 粒度的存档点，硬停在这里会丢掉整个
            # epoch 的训练成果。所以先看一眼温度：贴着阈值就原地等它降下来
            # （等待期 GPU 空载，比被兜底看门狗硬杀便宜得多）。
            if self.thermal.enabled and self.thermal.last_temp is not None:
                _t = read_gpu_temp()
                if _t is not None and _t >= self.thermal.limit_c - 2:
                    print(f"[thermal] 进入验证前温度 {_t}°C 已贴近阈值 "
                          f"{self.thermal.limit_c}°C，先等待冷却以保住本 epoch")
                    wait_until_cool(self.thermal.resume_c, poll_s=20)

            # 更新学习率
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(train_losses['total'])
                else:
                    self.scheduler.step()

            # 定期验证
            if self.current_epoch % self.config.get('val_interval', 1) == 0:
                print(f"\n验证Epoch {self.current_epoch}")
                val_losses, vis_results = self.validate()

                # 保存可视化结果
                if self.current_epoch % self.config.get('vis_interval', 5) == 0:
                    self.save_visualizations(vis_results, self.current_epoch)

                # 记录验证损失
                for key, value in val_losses.items():
                    self.writer.add_scalar(f'val/{key}', value, self.current_epoch)

                # 🔴【关键修改】达标检查逻辑
                # 计算黄金指标
                # 1. Shading Variance
                shading_var = vis_results.get('shading_variance', 0.0)
                # 2. SH[0] Mean
                sh0_mean = vis_results.get('sh0_mean', 0.0)
                # 3. Albedo Correlation
                albedo_corr = vis_results.get('albedo_correlation', 1.0)
                # 4. Lambertian Ratio
                lambertian_ratio = vis_results.get('lambertian_ratio', 0.0)
                
                # 检查是否满足所有黄金指标
                # INC-0010 修复：albedo_corr 判定语义反转
                # 原式 albedo_corr < 0.4 是"反照率与图像弱相关"判定——与 Lambertian 假设
                # I = A·S 矛盾（A 应是 I 的空间低频包络，应强正相关）。
                # 等价映射：把"< 0.4 (弱相关)"翻转为"> 0.7 (强相关)"。
                # 对正确分解的模型：两个条件都返回 True；对错误分解的模型：都返回 False。
                is_qualified = (
                    shading_var > 0.01 and
                    0.8 <= sh0_mean <= 1.2 and
                    albedo_corr > 0.7 and
                    lambertian_ratio > 0.95
                )
                
                # 更新连续达标计数器
                if is_qualified:
                    self.continuous_qualified_epochs += 1
                    print(f"✅ Epoch {self.current_epoch} 达标！连续达标: {self.continuous_qualified_epochs}/10")
                    print(f"  指标: Shading Var={shading_var:.4f}, SH[0]={sh0_mean:.4f}, Albedo Corr={albedo_corr:.4f}, Lambertian Ratio={lambertian_ratio:.4f}")
                else:
                    self.continuous_qualified_epochs = 0
                    print(f"❌ Epoch {self.current_epoch} 未达标")
                    print(f"  指标: Shading Var={shading_var:.4f}, SH[0]={sh0_mean:.4f}, Albedo Corr={albedo_corr:.4f}, Lambertian Ratio={lambertian_ratio:.4f}")
                
                # 当连续10个epoch达标时，保存模型并停止训练
                if self.continuous_qualified_epochs >= self.qualified_threshold:
                    print(f"\n{'='*80}")
                    print(f"🎉 连续 {self.qualified_threshold} 个epoch达标！")
                    print(f"模型已达到纯物理训练目标")
                    print(f"{'='*80}")
                    
                    # 保存最佳模型
                    self.save_checkpoint(self.current_epoch, val_losses['total'], True)
                    
                    # 重命名为 pure_physics_best.pth
                    best_checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch:04d}_best.pth"
                    pure_physics_path = self.checkpoint_dir / "pure_physics_best.pth"
                    import shutil
                    if best_checkpoint_path.exists():
                        shutil.copy(best_checkpoint_path, pure_physics_path)
                        print(f"模型已保存为: {pure_physics_path}")
                    
                    # 停止训练
                    print(f"\n训练已完成，模型已达标！")
                    self.writer.close()
                    return

                # 保存检查点
                is_best = val_losses['total'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_losses['total']
                    print(f"新最佳验证损失: {self.best_val_loss:.6f}")
                
                # 🔴【关键修改】每个epoch都保存模型
                self.save_checkpoint(self.current_epoch, val_losses['total'], is_best)
                # 本 epoch 已有正式存档，中途状态失效，立即删除以免下次误用
                self._clear_interrupt_state()

            print(f"\nEpoch {self.current_epoch} 完成")
            print(f"训练损失: {train_losses['total']:.6f}")
            if 'val_losses' in locals():
                print(f"验证损失: {val_losses['total']:.6f}")

        # 训练完成，保存最终模型
        print(f"\n{'='*80}")
        print(f"训练完成")
        print(f"总耗时: {time.time() - start_time:.2f} 秒")
        print(f"最佳验证损失: {self.best_val_loss:.6f}")
        print(f"{'='*80}")
        # 确保val_losses有值，如果没有则使用最佳损失
        final_val_loss = val_losses['total'] if 'val_losses' in locals() else self.best_val_loss
        self.save_checkpoint(self.current_epoch, final_val_loss, is_best=False)

        self.writer.close()
