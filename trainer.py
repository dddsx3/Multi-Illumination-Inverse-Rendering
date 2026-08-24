"""
逆向渲染训练器
实现三阶段课程学习策略的完整训练pipeline

Author: Python Engineer
Date: 2026-01-24
"""

import time
from pathlib import Path
from typing import Dict, Tuple, List
import json

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

from unet_model import IntrinsicUNet
from loss_functions import LossCalculator, GtSupervisionLoss
from physics_renderer import PhysicsRenderer
from residual_modules import HierarchicalResidual
from evaluate import compute_all
from stability import StabilityGuard
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
                if torch.cuda.is_bf16_supported():
                    print("混合精度: bfloat16（指数位同 fp32，无溢出风险；无需 GradScaler）")
                else:
                    print("⚠ GPU 不支持 BF16，回退 float16 + GradScaler")
                    self.amp_dtype = 'float16'
            if self.amp_dtype == 'float16':
                self._use_scaler = True
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

    def _define_stage_configs(self) -> Dict:
        """
        定义各阶段的配置

        Returns:
            字典包含每个阶段的epoch范围和loss权重
        """
        stage1_end = self.config.get('stage1_epochs', 30)
        stage2_end = stage1_end + self.config.get('stage2_epochs', 30)

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
                    'gt_normal': 0.5
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
                    'gt_normal': 0.0
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

        for batch_idx, (images, gt, scene_names) in enumerate(self.train_loader):
            images = images.to(self.device)
            gt_dev = None
            if gt is not None:
                gt_dev = {k: v.to(self.device) for k, v in gt.items()}

            B, K, H, W = images.shape

            self.optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=self._autocast_dtype):
                depth, albedo, sh_coeffs, weight_map, features = self.model(images)

                rendered, normal, shading = self.renderer(depth, albedo, sh_coeffs)

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
                    target_images=images,
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
            self.stability.check_grad_norm(grad_norm)

            # 参数更新
            if self._use_scaler:
                self.optimizer.scaler.step(self.optimizer)
                self.optimizer.scaler.update()
            else:
                self.optimizer.step()

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
                        image_flat = images[b, k].flatten()
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

                B, K, H, W = images.shape

                depth, albedo, sh_coeffs, weight_map, features = self.model(images)

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
                    target_images=images,
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

                # Phase 1 (T1.4)：GT 量化指标（验证模式中心裁剪、无增强，
                # 指标走 evaluate.compute_all；无 GT 数据集自动跳过）
                if gt_dev is not None:
                    m_dict = compute_all(
                        pred={'normal': normal, 'depth': depth,
                              'albedo': albedo, 'image': final_render},
                        gt={'normal': gt_dev['normal'], 'depth': gt_dev['depth'],
                            'albedo': gt_dev['albedo'], 'image': images},
                        mask=gt_dev['mask'])
                    for mk, mv in m_dict.items():
                        if mv == mv and abs(mv) != float('inf'):  # 跳过 NaN/Inf
                            metric_acc.setdefault(mk, []).append(mv)

                if batch_idx == 0:
                    vis_results = self._collect_visualization(
                    images, depth, albedo, weight_map, final_render, normal, shading, sh_coeffs, scene_names
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
        
        # 4. Lambertian Ratio
        # 计算渲染图像和输入图像的能量比
        rendered_energy = (rendered ** 2).mean().item()
        total_energy = (images ** 2).mean().item()
        lambertian_ratio = rendered_energy / total_energy if total_energy > 0 else 0.0

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
        torch.save(checkpoint, checkpoint_path)
        print(f"检查点已保存: {checkpoint_path}")

        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"最佳模型已保存: {best_path}")

        # 总是保存最新模型
        latest_path = self.checkpoint_dir / 'latest_model.pth'
        torch.save(checkpoint, latest_path)

    def load_checkpoint(self, checkpoint_path: str):
        """
        加载检查点

        Args:
            checkpoint_path: 检查点文件路径
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.renderer.load_state_dict(checkpoint['renderer_state_dict'])
        if self.residual is not None and checkpoint['residual_state_dict'] is not None:
            self.residual.load_state_dict(checkpoint['residual_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint['scheduler_state_dict'] is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.current_stage = checkpoint.get('current_stage', 1)

        print(f"检查点已加载: {checkpoint_path}")
        print(f"  Epoch: {self.current_epoch}")
        print(f"  Best Val Loss: {self.best_val_loss:.6f}")
        print(f"  Current Stage: {self.current_stage}")
    def train(self):
        """完整训练流程"""
        total_epochs = self.config.get('total_epochs', 100)

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
                is_qualified = (
                    shading_var > 0.01 and
                    0.8 <= sh0_mean <= 1.2 and
                    albedo_corr < 0.4 and
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
