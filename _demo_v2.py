#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
组会示例脚本：加载 v2 best_model，在 test 集上跑多个场景生成可视化

注意：v2 训练用 `--model fusion`（FusionUNet），不是 IntrinsicUNet
"""
import os
import sys
import json
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from main import (
    MultiLightingDataset, create_data_loader, setup_device, set_seed
)
from fusion_unet import FusionUNet
from physics_renderer import PhysicsRenderer
from residual_modules import HierarchicalResidual


def save_tensor_as_img(tensor, path, vmin=None, vmax=None):
    tensor = tensor.detach().cpu()
    if vmin is None:
        vmin = float(tensor.min())
    if vmax is None:
        vmax = float(tensor.max())
    t = (tensor - vmin) / (vmax - vmin + 1e-8)
    t = (t * 255).clamp(0, 255).byte().numpy()
    if t.ndim == 3 and t.shape[0] in (1, 3):
        t = t.transpose(1, 2, 0)
    if t.ndim == 3 and t.shape[-1] == 1:
        t = t[..., 0]
    Image.fromarray(t).save(path)


def load_split_manifest(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default='D:/data/synthetic_v3')
    ap.add_argument('--split-manifest', default='D:/Multi-Illumination Inverse Rendering/repo/splits/synthetic_v3.json')
    ap.add_argument('--checkpoint', default='D:/Multi-Illumination Inverse Rendering/checkpoints/p2_t22_f_n5rgb_v2/best_model.pth')
    ap.add_argument('--out-dir', default='D:/Multi-Illumination Inverse Rendering/group_meeting_demo_v2')
    ap.add_argument('--num-scenes', type=int, default=5, help='跑多少个 test 场景')
    ap.add_argument('--scene-ids', type=str, default='', help='逗号分隔的场景名列表')
    ap.add_argument('--modality', default='rgb', choices=['rgb', 'gray'])
    ap.add_argument('--num-lights', type=int, default=5)
    ap.add_argument('--image-size', type=int, default=256)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'[demo] 输出目录: {out_dir}')

    set_seed(args.seed)
    device = setup_device('cuda')
    print(f'[demo] device: {device}')

    manifest = load_split_manifest(args.split_manifest)
    test_scenes = manifest.get('test', manifest.get('test_scenes', []))
    if args.scene_ids:
        target_scenes = [s.strip() for s in args.scene_ids.split(',')]
    else:
        target_scenes = test_scenes[:args.num_scenes]
    print(f'[demo] 目标场景 ({len(target_scenes)}): {target_scenes}')

    dataset = MultiLightingDataset(
        root_dir=args.data_root,
        scene_subset=target_scenes,
        num_lights=args.num_lights,
        image_size=[args.image_size, args.image_size],
        is_training=False,
        file_extension='.png',
        modality=args.modality,
    )
    print(f'[demo] dataset: {len(dataset)} samples')

    loader = create_data_loader(
        dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False,
    )

    # 加载 checkpoint
    print(f'[demo] 加载 checkpoint: {args.checkpoint}')
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # 用 v2 训练时的配置：--model fusion --modality rgb
    # FusionUNet signature: (num_images, in_channels, base_channels, sh_order)
    in_channels = 3 if args.modality == 'rgb' else 1
    model = FusionUNet(
        num_images=args.num_lights,
        in_channels=in_channels,
        base_channels=32,
        sh_order=2,
    ).to(device)
    renderer = PhysicsRenderer(
        use_edge_aware=True,
        use_directional_light=False,
    ).to(device)
    residual = HierarchicalResidual(
        use_local_residual=True,
        num_images=args.num_lights,
        feature_channels=32,
    ).to(device)

    model.load_state_dict(ckpt['model_state_dict'])
    renderer.load_state_dict(ckpt['renderer_state_dict'])
    residual.load_state_dict(ckpt['residual_state_dict'])
    model.eval()
    renderer.eval()
    residual.eval()
    val_loss = ckpt.get('best_val_loss', None)
    epoch = ckpt.get('epoch', '?')
    print(f'[demo] 模型加载完毕（val={val_loss} epoch={epoch}）')

    metrics_summary = []

    with torch.no_grad():
        for scene_idx, (images, _gt, scene_names) in enumerate(loader):
            scene_name = scene_names[0]
            images = images.to(device)  # [1, K, 3, H, W] (rgb)
            print(f'\n[demo] [{scene_idx+1}/{len(loader)}] 场景: {scene_name} 输入形状: {tuple(images.shape)}')

            # FusionUNet forward 返回 (depth, albedo, sh_coeffs, weight_map, features, albedo_pl)
            out = model(images)
            if len(out) == 6:
                depth, albedo, sh_coeffs, weight_map, features, albedo_pl = out
            elif len(out) == 5:
                depth, albedo, sh_coeffs, weight_map, features = out
                albedo_pl = None
            else:
                raise RuntimeError(f'unexpected model output: {len(out)}')

            rendered, normal, shading = renderer(depth, albedo, sh_coeffs)
            final_render, global_residual, local_residual = residual(
                albedo, shading, normal, sh_coeffs,
                features=features,
            )

            scene_out = out_dir / scene_name
            scene_out.mkdir(parents=True, exist_ok=True)

            K = images.shape[1]
            for k in range(K):
                # input image: [1, K, C, H, W] 取 [0, k, ...]
                inp = images[0, k]  # [C, H, W]
                if inp.shape[0] == 1:
                    save_tensor_as_img(inp[0], scene_out / f'input_{k:02d}.png')
                else:
                    save_tensor_as_img(inp, scene_out / f'input_{k:02d}.png')

                out_k = final_render[0, k]
                if out_k.shape[0] == 1:
                    save_tensor_as_img(out_k[0], scene_out / f'rendered_{k:02d}.png')
                else:
                    save_tensor_as_img(out_k, scene_out / f'rendered_{k:02d}.png')

            # albedo / depth / weight_map：取 [0, 0]（主反照率）
            save_tensor_as_img(albedo[0, 0], scene_out / 'albedo.png')
            save_tensor_as_img(depth[0, 0], scene_out / 'depth.png')
            if weight_map is not None:
                save_tensor_as_img(weight_map[0, 0], scene_out / 'weight_map.png')

            # normal：3 通道 [0, 0]，映射 [-1, 1] -> [0, 1]
            n = (normal[0] + 1) / 2
            save_tensor_as_img(n, scene_out / 'normal_rgb.png')
            save_tensor_as_img(normal[0, 0], scene_out / 'normal_x.png')
            save_tensor_as_img(normal[0, 1], scene_out / 'normal_y.png')
            save_tensor_as_img(normal[0, 2], scene_out / 'normal_z.png')

            # shading：取 [0, 0]
            if shading is not None:
                save_tensor_as_img(shading[0, 0], scene_out / 'shading.png')

            # L1 误差（与 input_0）
            inp0 = images[0, 0]
            out0 = final_render[0, 0]
            l1 = (inp0 - out0).abs().mean().item()
            metrics_summary.append({
                'scene': scene_name,
                'L1_error_first_light': l1,
                'depth_min': float(depth.min()),
                'depth_max': float(depth.max()),
                'albedo_min': float(albedo.min()),
                'albedo_max': float(albedo.max()),
            })
            print(f'[demo]   场景 {scene_name} L1 误差 (input_0 vs rendered_0): {l1:.4f}')
            print(f'[demo]   depth 范围: [{depth.min():.3f}, {depth.max():.3f}]')
            print(f'[demo]   albedo 范围: [{albedo.min():.3f}, {albedo.max():.3f}]')

    summary_path = out_dir / 'summary.json'
    mean_l1 = float(np.mean([m['L1_error_first_light'] for m in metrics_summary]))
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'checkpoint': args.checkpoint,
            'val_loss': val_loss,
            'epoch': epoch,
            'num_scenes': len(target_scenes),
            'modality': args.modality,
            'per_scene': metrics_summary,
            'mean_l1': mean_l1,
        }, f, indent=2, ensure_ascii=False)
    print(f'\n[demo] 汇总写入: {summary_path}')
    print(f'[demo] 平均 L1 误差: {mean_l1:.4f}')
    print(f'\n[demo] 完成。可视化图像保存到: {out_dir}')


if __name__ == '__main__':
    main()
