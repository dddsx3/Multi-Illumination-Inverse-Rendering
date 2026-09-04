"""
逆向渲染项目主程序入口
支持训练、测试和演示模式

Author: Python Engineer
Date: 2026-01-22
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
import random
import numpy as np
import torch
from typing import Optional
from config import Config, get_default_config
from data_loader import MultiLightingDataset, create_data_loader, split_scene_names
from unet_model import IntrinsicUNet
from physics_renderer import PhysicsRenderer
from residual_modules import HierarchicalResidual
from loss_functions import LossCalculator
from trainer import InverseRenderTrainer
from thermal_guard import ThermalStop
from runtime_safety import MemoryStop

# 温度墙停机专用退出码。区别于普通失败（非 0 且非 42）：编排器看到 42
# 就知道"训练状态完好、只是太热了"，等冷却后重跑同一条命令即可续上；
# 其他非零码仍按真实故障处理，不做自动重试。
EXIT_THERMAL_STOP = 42


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"随机种子已设置为: {seed}")


def setup_device(device: str) -> torch.device:
    """设置设备 - 强制使用GPU"""
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA不可用！请确保已安装CUDA版本的PyTorch并且有可用的GPU。")

        device_obj = torch.device('cuda')
        print(f"使用设备: {device_obj}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"PyTorch版本: {torch.__version__}")
    else:
        device_obj = torch.device('cpu')
        print(f"使用设备: {device_obj}")

    return device_obj


def train_mode(config: Config, resume_checkpoint: Optional[str] = None):
    """
    训练模式

    Args:
        config: 配置对象
        resume_checkpoint: 恢复训练的检查点路径
    """
    print("\n" + "=" * 80)
    print("训练模式")
    print("=" * 80 + "\n")

    set_seed(config.seed)
    device = setup_device(config.device)

    config.print()

    print("创建数据集...")

    # Phase 1 (C1 正式化)：划分优先级 = 命令行场景列表 > 划分清单 JSON >
    # 确定性计算。清单中的 test 子集受冻结规则保护（只评不训）。
    if getattr(config.data, 'train_scenes', None) and getattr(config.data, 'val_scenes', None):
        train_names = list(config.data.train_scenes)
        val_names = list(config.data.val_scenes)
    elif getattr(config, 'split_manifest', None):
        from split_manifest import load_split
        train_names = load_split(config.split_manifest, 'train')
        val_names = load_split(config.split_manifest, 'val')
        print(f"[manifest] test 集 {len(load_split(config.split_manifest, 'test'))} 场景已冻结（只准评估）")
    else:
        train_names, val_names = split_scene_names(
            config.data.root_dir,
            train_val_split=getattr(config.data, 'train_val_split', 0.8),
            seed=getattr(config, 'seed', 42))
    print(f"场景划分: train={len(train_names)}, val={len(val_names)}")

    train_dataset = MultiLightingDataset(
        root_dir=config.data.root_dir,
        num_lights=config.data.num_lights,
        image_size=config.data.image_size,
        is_training=True,
        file_extension=config.data.file_extension,
        max_rotation_angle=config.data.max_rotation_angle,
        horizontal_flip_prob=config.data.horizontal_flip_prob,
        scene_subset=train_names,
        modality=getattr(config.data, 'modality', 'gray')
    )

    val_dataset = MultiLightingDataset(
        root_dir=config.data.root_dir,
        num_lights=config.data.num_lights,
        image_size=config.data.image_size,
        is_training=False,
        file_extension=config.data.file_extension,
        scene_subset=val_names,
        modality=getattr(config.data, 'modality', 'gray')
    )

    print(f"训练集大小: {len(train_dataset)} 个场景")
    print(f"验证集大小: {len(val_dataset)} 个场景")

    train_loader = create_data_loader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        prefetch_factor=config.data.prefetch_factor,
        persistent_workers=config.data.persistent_workers
    )

    val_loader = create_data_loader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        prefetch_factor=config.data.prefetch_factor,
        persistent_workers=False  # 验证时不需要保持worker
    )

    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")

    print("\n创建模型...")

    if getattr(config.model, 'architecture', 'unet') == 'fusion':
        from fusion_unet import FusionUNet
        in_ch = 3 if config.data.modality == 'rgb' else 1
        model = FusionUNet(
            num_images=config.model.num_images,
            in_channels=in_ch,
            base_channels=config.model.base_channels,
            sh_order=config.model.sh_order,
            use_per_light_albedo=not getattr(config, 'no_per_light_albedo', False),
            sh_constraint=getattr(config.model, 'sh_constraint', 'clamp'))
        print(f"架构: FusionUNet (in_channels={in_ch}, N-agnostic)")
    else:
        model = IntrinsicUNet(
            num_images=config.model.num_images,
            base_channels=config.model.base_channels,
            sh_order=config.model.sh_order)

    renderer = PhysicsRenderer(
        use_edge_aware=config.model.use_edge_aware,
        use_directional_light=config.model.use_directional_light
    )

    # Phase 1：恢复残差模块。Phase 0 已将全局共享缓冲重构为逐场景、
    # 逐光照的 LocalResidualNet（零初始化），阶段1/2 冻结、阶段3 解冻
    # 由 trainer 的课程学习逻辑管理。
    residual = HierarchicalResidual(
        use_local_residual=config.model.use_local_residual,
        num_images=config.model.num_images,
        feature_channels=config.model.base_channels,
        hidden_channels=getattr(config, 'res_hidden', 64)
    )
    if getattr(config, 'residual_off', False):
        # F-resA：阶段3 残差缩放恒 0（模块保留，参数量口径可解释）
        residual.residual_scales = {'stage1': 0.0, 'stage2': 0.0, 'stage3': 0.0}
        print('[F-resA] 残差缩放全部置零')

    total_params = sum(p.numel() for p in model.parameters())
    residual_params = sum(p.numel() for p in residual.parameters())
    print(f"模型参数量: {total_params:,}")
    print(f"残差模块参数量: {residual_params:,}")
    print(f"总参数量: {total_params + residual_params:,}")

    print("\n创建训练器...")

    # T2.0（INC-0003）修复：目录解析已上移至 main()，经 config 传入本函数
    # （原实现在此处引用 args 触发 NameError——args 不属于 train_mode 作用域）
    run_id = getattr(config, 'run_id', 'run_default')
    checkpoint_dir = config.paths.checkpoint_dir
    log_dir = config.paths.log_dir
    viz_dir = config.paths.vis_dir
    print(f"[T2.0] run_id={run_id}")
    print(f"[T2.0] checkpoint_dir={checkpoint_dir}")
    print(f"[T2.0] log_dir={log_dir} | viz_dir={viz_dir}")

    trainer_config = {
        'run_id': run_id,
        'architecture': getattr(config.model, 'architecture', 'unet'),
        'modality': getattr(config.data, 'modality', 'gray'),
        'sh_constraint': getattr(config.model, 'sh_constraint', 'clamp'),
        'res_hidden': getattr(config, 'res_hidden', 64),
        'residual_off': getattr(config, 'residual_off', False),
        'no_per_light_albedo': getattr(config, 'no_per_light_albedo', False),
        'data_root': config.data.root_dir,
        'train_scenes': config.data.train_scenes,
        'val_scenes': config.data.val_scenes,
        'num_lights': config.data.num_lights,
        'image_size': config.data.image_size,
        'batch_size': config.data.batch_size,
        'total_epochs': config.train.total_epochs,
        'learning_rate': config.train.learning_rate,
        'weight_decay': config.train.weight_decay,
        'stage1_epochs': config.train.stage1_epochs,
        'stage2_epochs': config.train.stage2_epochs,
        'base_channels': config.model.base_channels,
        'scheduler': config.train.scheduler,
        'step_size': config.train.step_size,
        'gamma': config.train.gamma,
        'log_dir': log_dir,
        'checkpoint_dir': checkpoint_dir,
        'vis_dir': viz_dir,
        'num_workers': config.data.num_workers,
        'use_amp': config.train.use_amp,
        'amp_dtype': getattr(config.train, 'amp_dtype', 'bfloat16'),
        'use_edge_aware': config.model.use_edge_aware,
        'log_interval': config.train.log_interval,
        'tensorboard_interval': config.train.tensorboard_interval,
        'val_interval': config.train.val_interval,
        'vis_interval': config.train.vis_interval,
        'save_interval': config.train.save_interval,
        'aggressive_albedo_smooth': getattr(config, 'aggressive_albedo_smooth', False)
    }

    trainer = InverseRenderTrainer(
        model=model,
        renderer=renderer,
        residual=residual,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_config
    )

    if resume_checkpoint is not None:
        print(f"\n从检查点恢复训练: {resume_checkpoint}")
        trainer.load_checkpoint(resume_checkpoint)

    print("\n开始训练...\n")
    try:
        trainer.train()
    except (ThermalStop, MemoryStop) as stop:
        # 温度墙 / 主机内存越界（INC-0014 新增内存通道）：trainer 已在
        # train_epoch 内把 epoch 中途状态落盘。以 rc=42 退出，编排器据此
        # 等待（冷却/内存恢复）后自动续跑。
        kind = "thermal" if isinstance(stop, ThermalStop) else "host-memory"
        print(f"\n[{kind}] 安全停机（{stop}），状态已存档，退出码 {EXIT_THERMAL_STOP}")
        try:
            trainer.writer.close()
        except Exception:
            pass
        sys.exit(EXIT_THERMAL_STOP)

    print("\n训练完成!")


def test_mode(config: Config, checkpoint_path: str):
    """
    测试模式

    Args:
        config: 配置对象
        checkpoint_path: 模型检查点路径
    """
    print("\n" + "=" * 80)
    print("测试模式")
    print("=" * 80 + "\n")

    set_seed(config.seed)
    device = setup_device(config.device)

    print("创建测试数据集...")

    test_dataset = MultiLightingDataset(
        root_dir=config.data.root_dir,
        scene_list=config.data.test_scenes,
        num_lights=config.data.num_lights,
        image_size=config.data.image_size,
        is_training=False,
        file_extension=config.data.file_extension
    )

    print(f"测试集大小: {len(test_dataset)} 个场景")

    test_loader = create_data_loader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    print("创建模型...")

    model = IntrinsicUNet(
        num_images=config.model.num_images,
        base_channels=config.model.base_channels,
        sh_order=config.model.sh_order
    ).to(device)

    renderer = PhysicsRenderer(
        use_edge_aware=config.model.use_edge_aware,
        use_directional_light=config.model.use_directional_light
    ).to(device)

    residual = HierarchicalResidual(
        use_local_residual=config.model.use_local_residual,
        num_images=config.model.num_images,
        feature_channels=config.model.base_channels
    ).to(device)

    print(f"加载检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    renderer.load_state_dict(checkpoint['renderer_state_dict'])
    residual.load_state_dict(checkpoint['residual_state_dict'])

    model.eval()
    renderer.eval()
    residual.eval()

    print("开始测试...\n")

    loss_calculator = LossCalculator()

    from PIL import Image
    import numpy as np

    output_dir = Path(config.paths.output_dir) / 'test_results'
    output_dir.mkdir(parents=True, exist_ok=True)

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, (images, _gt, scene_names) in enumerate(test_loader):
            images = images.to(device)
            scene_name = scene_names[0]

            print(f"处理场景 {batch_idx + 1}/{len(test_loader)}: {scene_name}")

            depth, albedo, sh_coeffs, weight_map, features = model(images)
            rendered, normal, shading = renderer(depth, albedo, sh_coeffs)
            final_render, global_residual, local_residual = residual(
                albedo, shading, normal, sh_coeffs,
                features=features
            )

            total_loss_batch, loss_dict = loss_calculator(
                pred_images=final_render,
                target_images=images,
                depth=depth,
                albedo=albedo,
                weight_map=weight_map,
                sh_coeffs=sh_coeffs,
                local_residual=local_residual
            )

            total_loss += total_loss_batch.item()
            num_batches += 1

            scene_dir = output_dir / scene_name
            scene_dir.mkdir(parents=True, exist_ok=True)

            def save_tensor(tensor, path, vmin=None, vmax=None):
                if vmin is None:
                    vmin = tensor.min()
                if vmax is None:
                    vmax = tensor.max()
                tensor_norm = (tensor - vmin) / (vmax - vmin + 1e-8)
                tensor_np = (tensor_norm * 255).clamp(0, 255).cpu().numpy()
                img = Image.fromarray(tensor_np.astype(np.uint8))
                img.save(path)

            B, K, H, W = images.shape

            for k in range(K):
                save_tensor(images[0, k], scene_dir / f'input_{k:02d}.png')
                save_tensor(rendered[0, k], scene_dir / f'rendered_{k:02d}.png')
                save_tensor(final_render[0, k], scene_dir / f'final_render_{k:02d}.png')

            save_tensor(depth[0, 0], scene_dir / 'depth.png')
            save_tensor(albedo[0, 0], scene_dir / 'albedo.png')
            save_tensor(weight_map[0, 0], scene_dir / 'weight_map.png')

            normal_vis = (normal[0] + 1) / 2
            for i, name in enumerate(['normal_x', 'normal_y', 'normal_z']):
                save_tensor(normal_vis[i], scene_dir / f'{name}.png')

            print(f"  损失: {total_loss_batch.item():.6f}")
            print(f"  结果已保存到: {scene_dir}")

    avg_loss = total_loss / num_batches
    print(f"\n平均测试损失: {avg_loss:.6f}")
    print(f"测试完成! 结果已保存到: {output_dir}")


def demo_mode(config: Config, checkpoint_path: Optional[str] = None):
    """
    演示模式

    Args:
        config: 配置对象
        checkpoint_path: 模型检查点路径（可选）
    """
    print("\n" + "=" * 80)
    print("演示模式")
    print("=" * 80 + "\n")

    set_seed(config.seed)
    device = setup_device(config.device)

    print("创建演示数据集...")

    demo_scenes = config.data.train_scenes[:1]

    demo_dataset = MultiLightingDataset(
        root_dir=config.data.root_dir,
        scene_list=demo_scenes,
        num_lights=config.data.num_lights,
        image_size=config.data.image_size,
        is_training=False,
        file_extension=config.data.file_extension
    )

    print(f"演示场景: {demo_scenes}")

    demo_loader = create_data_loader(
        demo_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    print("创建模型...")

    model = IntrinsicUNet(
        num_images=config.model.num_images,
        base_channels=config.model.base_channels,
        sh_order=config.model.sh_order
    ).to(device)

    renderer = PhysicsRenderer(
        use_edge_aware=config.model.use_edge_aware,
        use_directional_light=config.model.use_directional_light
    ).to(device)

    residual = HierarchicalResidual(
        use_local_residual=config.model.use_local_residual,
        num_images=config.model.num_images,
        feature_channels=config.model.base_channels
    ).to(device)

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        renderer.load_state_dict(checkpoint['renderer_state_dict'])
        residual.load_state_dict(checkpoint['residual_state_dict'])
        model.eval()
        renderer.eval()
        residual.eval()
    else:
        print("使用随机初始化的模型")

    print("\n运行演示...\n")

    with torch.no_grad():
        for images, _gt, scene_names in demo_loader:
            images = images.to(device)
            scene_name = scene_names[0]

            print(f"场景: {scene_name}")
            print(f"输入形状: {images.shape}")

            depth, albedo, sh_coeffs, weight_map, features = model(images)
            rendered, normal, shading = renderer(depth, albedo, sh_coeffs)
            final_render, global_residual, local_residual = residual(
                albedo, shading, normal, sh_coeffs,
                features=features
            )

            print(f"\n预测结果:")
            print(f"  深度图: {depth.shape}, 范围 [{depth.min():.3f}, {depth.max():.3f}]")
            print(f"  反照率图: {albedo.shape}, 范围 [{albedo.min():.3f}, {albedo.max():.3f}]")
            print(f"  权重图: {weight_map.shape}, 范围 [{weight_map.min():.3f}, {weight_map.max():.3f}]")
            print(f"  球谐系数: {sh_coeffs.shape}, 范围 [{sh_coeffs.min():.3f}, {sh_coeffs.max():.3f}]")
            print(f"  法线图: {normal.shape}, 范围 [{normal.min():.3f}, {normal.max():.3f}]")
            print(f"  着色图: {shading.shape}, 范围 [{shading.min():.3f}, {shading.max():.3f}]")
            print(f"  渲染图像: {rendered.shape}, 范围 [{rendered.min():.3f}, {rendered.max():.3f}]")
            print(
                f"  全局残差: {global_residual.shape}, 范围 [{global_residual.min():.3f}, {global_residual.max():.3f}]")

            if local_residual is not None:
                print(
                    f"  局部残差: {local_residual.shape}, 范围 [{local_residual.min():.3f}, {local_residual.max():.3f}]")

            print(f"\n最终渲染: {final_render.shape}, 范围 [{final_render.min():.3f}, {final_render.max():.3f}]")

            loss_calculator = LossCalculator()
            total_loss, loss_dict = loss_calculator(
                pred_images=final_render,
                target_images=images,
                depth=depth,
                albedo=albedo,
                weight_map=weight_map,
                sh_coeffs=sh_coeffs,
                local_residual=local_residual
            )

            print(f"\n损失:")
            for key, value in loss_dict.items():
                print(f"  {key}: {value:.6f}")
            print(f"  总损失: {total_loss.item():.6f}")

            break

    print("\n演示完成!")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='逆向渲染项目 - 多光照图像的内在属性分解',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  训练模式:
    python main.py --mode train --data_root ./data --train_scenes scene_001 scene_002 --val_scenes scene_003

  测试模式:
    python main.py --mode test --checkpoint ./checkpoints/best_model.pth --test_scenes scene_004

  演示模式:
    python main.py --mode demo --checkpoint ./checkpoints/best_model.pth

  使用配置文件:
    python main.py --config config.json --mode train
        """
    )

    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'test', 'demo'],
                        help='运行模式: train, test, demo')

    parser.add_argument('--config', type=str, default=None,
                        help='配置文件路径 (JSON格式)')

    parser.add_argument('--checkpoint', type=str, default=None,
                        help='模型检查点路径 (用于测试/演示/恢复训练)')

    parser.add_argument('--resume', action='store_true',
                        help='从检查点恢复训练')

    parser.add_argument('--data_root', type=str, default='./data/inverse_rendering',
                        help='数据集根目录')

    parser.add_argument('--train_scenes', type=str, nargs='+', default=None,
                        help='训练场景列表')

    parser.add_argument('--val_scenes', type=str, nargs='+', default=None,
                        help='验证场景列表')

    parser.add_argument('--test_scenes', type=str, nargs='+', default=None,
                        help='测试场景列表')

    parser.add_argument('--num_lights', type=int, default=5,
                        help='每个场景的光照数量')

    parser.add_argument('--image_size', type=int, nargs=2, default=[256, 256],
                        help='图像尺寸 [H, W]')

    parser.add_argument('--batch_size', type=int, default=4,
                        help='批次大小')

    parser.add_argument('--total_epochs', type=int, default=100,
                        help='总训练epoch数')

    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='学习率')

    parser.add_argument('--stage1_epochs', type=int, default=30,
                        help='阶段1的epoch数')

    parser.add_argument('--stage2_epochs', type=int, default=30,
                        help='阶段2的epoch数')

    parser.add_argument('--base_channels', type=int, default=32,
                        help='U-Net基础通道数')

    parser.add_argument('--scheduler', type=str, default='step',
                        choices=['step', 'cosine', 'plateau', 'none'],
                        help='学习率调度器')

    parser.add_argument('--use_amp', action='store_true',
                        help='使用混合精度训练')

    parser.add_argument('--amp_dtype', type=str, default='bf16',
                        choices=['bf16', 'fp16'],
                        help='AMP 精度类型：bf16 无溢出风险（Blackwell/Ampere+ 默认），fp16 需 GradScaler')

    parser.add_argument('--split_manifest', type=str, default=None,
                        help='划分清单 JSON（含 train/val/test 列表）。提供后训练/验证使用清单划分，test 冻结只评不调参')

    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'fusion'],
                        help='网络架构：unet=原堆叠 U-Net；fusion=光照数量无关融合网络（T2.2）')

    parser.add_argument('--modality', type=str, default='gray', choices=['gray', 'rgb'],
                        help='输入模态：gray 读 light_NNN.png；rgb 读 light_NNN_rgb.png 三通道')

    parser.add_argument('--sh_constraint', type=str, default='clamp', choices=['clamp', 'softplus'],
                        help='SH[0] 约束形式：clamp=基线 hack；softplus=可微重参数化（T2.3）')

    # T2.5 消融变体旗标（每变体相对基准只动一个变量）
    parser.add_argument('--residual_off', action='store_true',
                        help='F-resA：阶段3 残差缩放恒 0（模块保留，参数量口径可解释）')
    parser.add_argument('--res_hidden', type=int, default=64,
                        help='LocalResidualNet 隐藏通道数（F-resC 用 32）')
    parser.add_argument('--no_per_light_albedo', action='store_true',
                        help='F-albOff：关闭逐光照反照率分支')
    # INC-0013 判别实验变体开关（中期审计 v2 §2-P2 假设 (a)(b)(c)）
    parser.add_argument('--disable_film', action='store_true',
                        help='F-noFiLM 判别实验 (b)：FiLM 调制关闭（gamma≡1, beta≡0）')
    parser.add_argument('--albedo_smooth_stage1', type=float, default=None,
                        help='F-lowSmooth 判别实验 (c)：覆盖 Stage 1 albedo_smooth 权重（默认 10.0，'
                             'F-lowSmooth 用 1.0 验证是否权重过高）')

    parser.add_argument('--run_id', type=str, default=None,
                        help='运行标识；缺省自动生成 run_YYYYMMDD_HHMMSS')

    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='checkpoint 输出目录（INC-0003：强制独立目录，缺省 checkpoints/{run_id}/）')

    parser.add_argument('--log_dir', type=str, default=None,
                        help='TensorBoard 日志目录（缺省 logs/{run_id}/）')

    parser.add_argument('--viz_dir', type=str, default=None,
                        help='可视化输出目录（缺省 visualizations/{run_id}/）')
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',  # 确保默认值是 'cuda'
        choices=['cuda', 'cpu'],
        help='训练设备'
    )

    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')

    parser.add_argument('--verbose', action='store_true', default=True,
                        help='详细输出')

    parser.add_argument('--aggressive-albedo-smooth', action='store_true',
                        help='启用反照率平滑激进疗法')

    # INC-0008 续补：让 run_arms.py 编排器把 worker 数注入子任务。
    # 不传 = 用 config.py 的默认值（直跑口径不变）；传任意值（含 0）= 显式覆盖。
    # INC-0016 P0：0 从"哨兵=未传"改为合法显式值（单进程加载，spawn 通道绕开）。
    # 详见 docs/incidents/INC-0016_worker加载窗口熔断盲区.md §6 与
    # docs/incidents/INC-0008_win1455_n5rgb验证阶段spawn触发.md。
    parser.add_argument('--num_workers', type=int, default=None,
                        help='DataLoader worker 数；不传=config.py 默认，'
                             '显式传值(含 0)覆盖（INC-0016 P0：本机传 0 走单进程加载）')

    return parser.parse_args()


def main():
    # ========== Phase 1 修复：命令行参数优先（旧版硬编码导致 CLI 全部失效）==========
    args = parse_args()

    # 运行模式：'train'、'test' 或 'demo'
    mode = args.mode

    # 检查点路径（用于恢复训练或测试）
    checkpoint_path = args.checkpoint

    # 是否恢复训练
    resume = args.resume

    # 数据根目录
    data_root = args.data_root

    # 训练/验证/测试场景列表（留空则按种子确定性划分）
    train_scenes = args.train_scenes or []
    val_scenes = args.val_scenes or []
    test_scenes = args.test_scenes or []

    # 每个场景的光照图像数量
    num_lights = args.num_lights

    # 图像尺寸 [高度, 宽度]
    image_size = list(args.image_size)

    # 批次大小
    batch_size = args.batch_size

    # 总训练轮数
    total_epochs = args.total_epochs

    # 学习率
    learning_rate = args.learning_rate

    # 阶段1训练轮数
    stage1_epochs = args.stage1_epochs

    # 阶段2训练轮数
    stage2_epochs = args.stage2_epochs

    # 基础通道数
    base_channels = args.base_channels

    # 学习率调度器：'step'、'cosine'、'plateau' 或 'none'
    scheduler = args.scheduler

    # 是否使用混合精度训练
    use_amp = args.use_amp

    # 划分清单（C1）：提供后 train/val 取自清单，test 冻结只评不训
    split_manifest = args.split_manifest

    # AMP 精度类型：bf16 / fp16
    amp_dtype = 'bfloat16' if args.amp_dtype == 'bf16' else 'float16'

    # 设备：'cuda' 或 'cpu'
    device = args.device

    # 随机种子
    seed = args.seed

    # 是否显示详细信息
    verbose = args.verbose

    # 是否启用反照率平滑激进疗法
    aggressive_albedo_smooth = args.aggressive_albedo_smooth

    # INC-0013: F-lowSmooth 判别实验 (c) 权重覆盖
    # 在构造 Trainer 之前保存，Trainer._setup_loss_calculator 之后由阶段表覆盖；
    # 用 monkey-patch 阶段表方式（最稳）
    low_smooth_override = args.albedo_smooth_stage1

    # ========================================


    print("=" * 80)
    print("逆向渲染项目")
    print("=" * 80)

    print(f"\n运行模式: {mode}")
    print(f"设备: {device}")
    print(f"数据根目录: {data_root}")
    print(f"光照数量: {num_lights}")
    print(f"批次大小: {batch_size}")
    print(f"总训练轮数: {total_epochs}")
    print(f"学习率: {learning_rate}")

    # 创建配置对象
    config = Config()

    # 更新配置
    config.data.root_dir = data_root
    config.data.train_scenes = train_scenes
    config.data.val_scenes = val_scenes
    config.data.test_scenes = test_scenes
    config.data.num_lights = num_lights
    config.data.image_size = tuple(image_size)
    config.data.batch_size = batch_size

    config.train.total_epochs = total_epochs
    config.train.learning_rate = learning_rate
    config.train.stage1_epochs = stage1_epochs
    config.train.stage2_epochs = stage2_epochs
    config.train.scheduler = scheduler
    config.train.use_amp = use_amp
    config.train.amp_dtype = amp_dtype
    config.split_manifest = split_manifest
    config.model.architecture = getattr(args, 'model', 'unet')
    config.data.modality = args.modality; config.model.sh_constraint = args.sh_constraint
    config.model.sh_constraint = args.sh_constraint
    config.res_hidden = args.res_hidden
    config.residual_off = args.residual_off
    config.no_per_light_albedo = args.no_per_light_albedo

    # INC-0008 续补：run_arms.py 通过 --num_workers 显式注入 worker 数。
    # INC-0016 P0 根因修复（2026-09-04 22:5x）：原判断 `if args.num_workers > 0`
    # 把 --num_workers 0 当"未传"→ config 默认 4 生效 → spawn 死锁再现（第二次）。
    # 0 是合法显式值（单进程加载，绕开 spawn 通道）。改用 argparse 区分
    # "未传"(default=None)与"显式传 0"：显式传任何值(含 0)都覆盖 config；
    # 直跑 `python main.py` 不传 → config 默认值（4）→ 直跑口径不变。
    if args.num_workers is not None:
        config.data.num_workers = args.num_workers

    # T2.0（INC-0003）：产物目录参数化——缺省按 run_id 独立目录，
    # 杜绝冒烟/多实验互相覆盖生产资产。目录保持在仓库外（../ 前缀）。
    run_id = args.run_id or datetime.now().strftime('run_%Y%m%d_%H%M%S')
    config.run_id = run_id
    config.paths.checkpoint_dir = args.checkpoint_dir or f'../checkpoints/{run_id}'
    config.paths.log_dir = args.log_dir or f'../logs/{run_id}'
    config.paths.vis_dir = args.viz_dir or f'../visualizations/{run_id}'
    print(f"[T2.0] run_id={run_id}")
    print(f"[T2.0] checkpoint_dir={config.paths.checkpoint_dir}")
    print(f"[T2.0] log_dir={config.paths.log_dir} | viz_dir={config.paths.vis_dir}")

    config.model.base_channels = base_channels
    config.model.num_images = num_lights

    config.device = device
    config.seed = seed
    config.verbose = verbose
    # 添加反照率平滑激进疗法配置
    config.aggressive_albedo_smooth = aggressive_albedo_smooth
    if config.aggressive_albedo_smooth:
        print("\n启用反照率平滑激进疗法!")
        # 应用激进疗法的损失权重
        config.loss.albedo_smooth = 2.5
        config.loss.shading_smooth_weight = 0.2
        config.loss.retinex_constraint_weight = 0.3
        config.loss.reconstruction = 0.8
        print(f"激进疗法权重:")
        print(f"  albedo_smooth: {config.loss.albedo_smooth}")
        print(f"  shading_smooth_weight: {config.loss.shading_smooth_weight}")
        print(f"  retinex_constraint_weight: {config.loss.retinex_constraint_weight}")
        print(f"  reconstruction: {config.loss.reconstruction}")

    # 根据模式执行相应操作
    if mode == 'train':
        train_mode(config, checkpoint_path if resume else None)
    elif mode == 'test':
        test_mode(config, checkpoint_path)
    elif mode == 'demo':
        demo_mode(config, checkpoint_path)
    else:
        print(f"错误: 未知的模式 '{mode}'")
        print("支持的模式: train, test, demo")
        sys.exit(1)
if __name__ == "__main__":
    main()