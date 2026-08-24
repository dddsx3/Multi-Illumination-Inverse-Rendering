"""
逆向渲染推理脚本
使用训练好的模型进行预测

Author: Python Engineer
Date: 2026-01-23
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
import json

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from unet_model import IntrinsicUNet
from physics_renderer import PhysicsRenderer
from residual_modules import HierarchicalResidual
from data_loader import MultiLightingDataset, create_data_loader


class InverseRenderInference:
    """
    逆向渲染推理器

    功能：
    - 加载训练好的模型
    - 对新数据进行推理
    - 保存预测结果（深度图、反照率图、渲染图像等）
    - 支持批量处理
    """

    def __init__(
            self,
            checkpoint_path: str,
            device: str = 'cuda',
            output_dir: str = './inference_results'
    ):
        """
        初始化推理器

        Args:
            checkpoint_path: 检查点文件路径
            device: 推理设备 ('cuda' 或 'cpu')
            output_dir: 输出目录
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"初始化推理器...")
        print(f"  检查点: {self.checkpoint_path}")
        print(f"  设备: {self.device}")
        print(f"  输出目录: {self.output_dir}")

        self._load_checkpoint()
        self._setup_models()

    def _load_checkpoint(self):
        """加载检查点"""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"检查点文件不存在: {self.checkpoint_path}")

        print(f"\n加载检查点...")
        self.checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        print(f"  Epoch: {self.checkpoint['epoch']}")
        print(f"  Val Loss: {self.checkpoint['val_loss']:.6f}")
        print(f"  Best Val Loss: {self.checkpoint['best_val_loss']:.6f}")
        print(f"  Current Stage: {self.checkpoint.get('current_stage', 1)}")

        self.config = self.checkpoint['config']

    def _setup_models(self):
        """设置模型"""
        print(f"\n设置模型...")

        num_images = self.config.get('num_lights', 5)
        base_channels = self.config.get('base_channels', 32)
        sh_order = self.config.get('sh_order', 2)

        self.model = IntrinsicUNet(
            num_images=num_images,
            base_channels=base_channels,
            sh_order=sh_order
        )

        self.renderer = PhysicsRenderer(
            use_edge_aware=self.config.get('use_edge_aware', True),
            use_directional_light=self.config.get('use_directional_light', False)
        )

        self.residual = HierarchicalResidual(
            use_local_residual=self.config.get('use_local_residual', True),
            num_images=self.config.get('num_images', 5),
            feature_channels=self.config.get('base_channels', 32)
        )

        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.renderer.load_state_dict(self.checkpoint['renderer_state_dict'])
        # 🔴【关键修改】处理 residual_state_dict 为 None 的情况
        if self.checkpoint['residual_state_dict'] is not None:
            self.residual.load_state_dict(self.checkpoint['residual_state_dict'])
        else:
            print("⚠️  残差模块状态字典为 None，使用随机初始化的残差模块")

        self.model.to(self.device)
        self.renderer.to(self.device)
        self.residual.to(self.device)

        self.model.eval()
        self.renderer.eval()
        self.residual.eval()

        total_params = sum(p.numel() for p in self.model.parameters())
        residual_params = sum(p.numel() for p in self.residual.parameters())
        print(f"  模型参数量: {total_params:,}")
        print(f"  残差模块参数量: {residual_params:,}")
        print(f"  总参数量: {total_params + residual_params:,}")

    @torch.no_grad()
    def infer_single_scene(
            self,
            images: torch.Tensor,
            scene_name: str
    ) -> Dict:
        """
        对单个场景进行推理

        Args:
            images: 输入图像张量，形状 [K, H, W]
            scene_name: 场景名称

        Returns:
            推理结果字典
        """
        images = images.unsqueeze(0).to(self.device)

        B, K, H, W = images.shape

        depth, albedo, sh_coeffs, weight_map, features = self.model(images)

        rendered, normal, shading = self.renderer(depth, albedo, sh_coeffs)

        final_render, global_residual, local_residual = self.residual(
            albedo, shading, normal, sh_coeffs,
            features=features
        )

        results = {
            'input_images': images[0].cpu(),
            'depth': depth[0, 0].cpu(),
            'albedo': albedo[0, 0].cpu(),
            'weight_map': weight_map[0, 0].cpu(),
            'rendered': rendered[0].cpu(),
            'final_render': final_render[0].cpu(),
            'normal': normal[0].cpu(),
            'shading': shading[0].cpu(),
            'sh_coeffs': sh_coeffs[0].cpu(),
            'global_residual': global_residual[0].cpu() if global_residual is not None else None,
            'local_residual': local_residual[0].cpu() if local_residual is not None else None,
            'scene_name': scene_name
        }

        return results

    def save_results(self, results: Dict, save_dir: Path):
        """
        保存推理结果

        Args:
            results: 推理结果字典
            save_dir: 保存目录
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        scene_name = results['scene_name']
        scene_dir = save_dir / scene_name
        scene_dir.mkdir(parents=True, exist_ok=True)

        def save_tensor(tensor, path, vmin=None, vmax=None, colormap=False):
            """保存tensor为图像"""
            if vmin is None:
                vmin = tensor.min()
            if vmax is None:
                vmax = tensor.max()

            tensor_norm = (tensor - vmin) / (vmax - vmin + 1e-8)
            tensor_norm = (tensor_norm * 255).clamp(0, 255).byte().numpy()

            if colormap:
                import matplotlib.pyplot as plt
                import matplotlib.cm as cm
                colormap = cm.get_cmap('jet')
                tensor_norm = colormap(tensor_norm / 255.0)[:, :, :3]
                tensor_norm = (tensor_norm * 255).astype(np.uint8)

            img = Image.fromarray(tensor_norm)
            img.save(path)

        K = results['input_images'].shape[0]

        for k in range(K):
            save_tensor(
                results['input_images'][k],
                scene_dir / f'input_{k:02d}.png'
            )
            save_tensor(
                results['rendered'][k],
                scene_dir / f'rendered_{k:02d}.png'
            )
            save_tensor(
                results['final_render'][k],
                scene_dir / f'final_render_{k:02d}.png'
            )

        save_tensor(results['depth'], scene_dir / 'depth.png', colormap=True)
        save_tensor(results['albedo'], scene_dir / 'albedo.png')
        save_tensor(results['weight_map'], scene_dir / 'weight_map.png')
        save_tensor(results['shading'][0], scene_dir / 'shading.png')

        normal = results['normal']
        normal_vis = (normal + 1) / 2
        save_tensor(normal_vis[0], scene_dir / 'normal_x.png')
        save_tensor(normal_vis[1], scene_dir / 'normal_y.png')
        save_tensor(normal_vis[2], scene_dir / 'normal_z.png')

        if results['global_residual'] is not None:
            save_tensor(results['global_residual'][0], scene_dir / 'global_residual.png')

        if results['local_residual'] is not None:
            save_tensor(results['local_residual'][0], scene_dir / 'local_residual.png')

        sh_coeffs = results['sh_coeffs']
        sh_dict = {
            'sh_coeffs': sh_coeffs.numpy().tolist(),
            'scene_name': scene_name
        }

        with open(scene_dir / 'sh_coeffs.json', 'w') as f:
            json.dump(sh_dict, f, indent=2)

    def infer_dataset(
            self,
            dataset: MultiLightingDataset,
            batch_size: int = 1,
            save_visualizations: bool = True
    ) -> List[Dict]:
        """
        对整个数据集进行推理

        Args:
            dataset: 数据集
            batch_size: 批次大小
            save_visualizations: 是否保存可视化结果

        Returns:
            所有场景的推理结果列表
        """
        print(f"\n对数据集进行推理...")
        print(f"  场景数量: {len(dataset)}")
        print(f"  批次大小: {batch_size}")

        data_loader = create_data_loader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

        all_results = []

        for images, _gt, scene_names in tqdm(data_loader, desc="推理进度"):
            B, K, H, W = images.shape

            for i in range(B):
                scene_images = images[i]
                scene_name = scene_names[i]

                results = self.infer_single_scene(scene_images, scene_name)
                all_results.append(results)

                if save_visualizations:
                    self.save_results(results, self.output_dir)

        print(f"\n推理完成！")
        print(f"  结果已保存到: {self.output_dir}")

        return all_results

    def infer_single_image_folder(
            self,
            image_folder: str,
            num_lights: int = 5,
            image_size: tuple = (256, 256),
            file_extension: str = '.png'
    ):
        """
        对单个图像文件夹进行推理

        Args:
            image_folder: 图像文件夹路径
            num_lights: 光照图像数量
            image_size: 图像尺寸
            file_extension: 文件扩展名
        """
        print(f"\n对单个图像文件夹进行推理...")
        print(f"  文件夹: {image_folder}")

        image_folder = Path(image_folder)

        image_files = sorted(image_folder.glob(f'*{file_extension}'))

        if len(image_files) < num_lights:
            raise ValueError(
                f"图像数量不足: 需要 {num_lights} 张，实际 {len(image_files)} 张"
            )

        image_files = image_files[:num_lights]

        images = []
        for img_path in image_files:
            img = Image.open(img_path).convert('L')
            img = img.resize(image_size[::-1], Image.BILINEAR)
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array)
            images.append(img_tensor)

        images = torch.stack(images, dim=0)

        scene_name = image_folder.name
        results = self.infer_single_scene(images, scene_name)

        self.save_results(results, self.output_dir)

        print(f"推理完成！结果已保存到: {self.output_dir / scene_name}")

        return results



def main():
    """主函数"""

    # ========== 在这里直接修改参数 ==========

    # 检查点文件路径（必需）
    checkpoint_path = r'C:\Users\35702\Desktop\PythonProject\checkpoints\best_model.pth'

    # 推理模式：'dataset'（整个数据集）或 'single'（单个文件夹）
    mode = 'single'

    # 数据集根目录（dataset模式使用）
    data_root = r'C:\Users\35702\Desktop\PythonProject\checkpoints\best_model.pth'

    # 图像文件夹路径（single模式使用）
    image_folder = r"C:\Users\35702\Desktop\演示\rgb\scene_000000"

    # 每个场景的光照图像数量
    num_lights = 5

    # 图像尺寸 [高度, 宽度]
    image_size = [256, 256]

    # 批次大小
    batch_size = 1

    # 输出目录
    output_dir = r'../inference_results'

    # 推理设备：'cuda' 或 'cpu'
    device = 'cuda'

    # 是否保存可视化结果
    save_visualizations = True

    # ========================================

    print("=" * 80)
    print("逆向渲染推理脚本")
    print("=" * 80)

    print(f"\n配置参数:")
    print(f"  检查点: {checkpoint_path}")
    print(f"  模式: {mode}")
    print(f"  数据根目录: {data_root}")
    print(f"  图像文件夹: {image_folder}")
    print(f"  光照数量: {num_lights}")
    print(f"  图像尺寸: {image_size}")
    print(f"  批次大小: {batch_size}")
    print(f"  输出目录: {output_dir}")
    print(f"  设备: {device}")
    print(f"  保存可视化: {save_visualizations}")

    if device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA不可用，使用CPU")
        device = 'cpu'

    inference = InverseRenderInference(
        checkpoint_path=checkpoint_path,
        device=device,
        output_dir=output_dir
    )

    if mode == 'dataset':
        dataset = MultiLightingDataset(
            root_dir=data_root,
            num_lights=num_lights,
            image_size=tuple(image_size),
            is_training=False
        )

        results = inference.infer_dataset(
            dataset=dataset,
            batch_size=batch_size,
            save_visualizations=save_visualizations
        )

    elif mode == 'single':
        if image_folder is None:
            raise ValueError("single模式需要指定 image_folder 参数")

        results = inference.infer_single_image_folder(
            image_folder=image_folder,
            num_lights=num_lights,
            image_size=tuple(image_size)
        )

    print("\n" + "=" * 80)
    print("推理完成！")
    print("=" * 80)

if __name__ == '__main__':
    main()