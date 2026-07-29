"""""
模型诊断脚本
用于逆向渲染模型的深度问题排查
检查点：Albedo-Shading模糊、几何噪声、残差滥用、光照分布

Author: AI Assistant
"""

import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from config import Config
from unet_model import IntrinsicUNet
from physics_renderer import PhysicsRenderer
from residual_modules import HierarchicalResidual
from data_loader import create_data_loader, MultiLightingDataset

# ==================== 在这里修改参数 ====================
CHECKPOINT_PATH = r'C:\Users\35702\Desktop\PythonProject\checkpoints\best_model.pth'  # 模型检查点路径
CONFIG_PATH = r'C:\Users\35702\Desktop\PythonProject\config_text\config_test.json'     # 配置文件路径 (可选，设为None则使用默认)
DATA_ROOT =  r"C:\Users\35702\Desktop\演示"              # 数据集根目录
OUTPUT_DIR = "./diagnosis_output"                # 诊断结果输出目录
NUM_SAMPLES = 5                                  # 诊断的样本数量
NUM_LIGHTS = 5                                   # 光照数量
# ======================================================

def load_model_and_data(config_path, checkpoint_path, data_root, num_lights=5):
    """加载配置、模型和数据"""
    # 1. 加载配置
    if config_path and os.path.exists(config_path):
        config = Config.load(config_path)
        print(f"✓ 已加载配置文件: {config_path}")
    else:
        # 创建一个默认配置对象
        config = Config()
        # 设置默认值
        config.model.num_images = num_lights
        config.model.base_channels = 32
        config.model.sh_order = 2
        config.model.use_edge_aware = False
        config.model.use_directional_light = False
        config.model.use_local_residual = True
        config.data.image_size = (256, 256)
        config.data.file_extension = '.png'
        print(f"使用临时默认配置")

    # 强制覆盖路径，确保指向您想要测试的数据
    config.data.root_dir = data_root
    config.data.num_lights = num_lights

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 2. 创建模型
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
        image_height=config.data.image_size[0],
        image_width=config.data.image_size[1],
        use_local_residual=config.model.use_local_residual
    ).to(device)

    # 3. 加载权重
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # 处理可能存在的键名差异
        try:
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            if 'renderer_state_dict' in checkpoint:
                renderer.load_state_dict(checkpoint['renderer_state_dict'])
            
            if 'residual_state_dict' in checkpoint:
                residual.load_state_dict(checkpoint['residual_state_dict'])
            print("✓ 权重加载成功")
        except Exception as e:
            print(f"⚠ 权重加载部分失败: {e}")
            print("继续使用部分加载的模型")
    else:
        print("⚠ 警告：未提供有效的检查点路径，使用随机初始化模型进行测试！")

    # 4. 准备数据（取少量样本用于诊断）
    val_dataset = MultiLightingDataset(
        root_dir=config.data.root_dir,
        num_lights=config.data.num_lights,
        image_size=config.data.image_size,
        is_training=False,  # 测试时关闭增强
        file_extension=config.data.file_extension
    )

    val_loader = create_data_loader(
        val_dataset,
        batch_size=1,  # 诊断时batch size设为1，方便分析
        shuffle=True,
        num_workers=0
    )

    return model, renderer, residual, val_loader, device


def calculate_metrics(images, albedo, shading, normal, local_residual, global_residual):
    """计算诊断指标"""
    B, K, H, W = images.shape

    # 1. Albedo-Image 相关性 (Albedo是不是在偷懒，直接复制Image？)
    # 将所有光照图像求平均作为参考
    mean_image = images.mean(dim=1, keepdim=True)  # [B, 1, H, W]

    # 计算皮尔逊相关系数 (展平后)
    albedo_flat = albedo.view(B, -1).cpu()
    mean_img_flat = mean_image.view(B, -1).cpu()

    # Center
    albedo_centered = albedo_flat - albedo_flat.mean(dim=1, keepdim=True)
    img_centered = mean_img_flat - mean_img_flat.mean(dim=1, keepdim=True)

    correlation = (albedo_centered * img_centered).sum(dim=1) / (
            torch.sqrt((albedo_centered ** 2).sum(dim=1)) * torch.sqrt((img_centered ** 2).sum(dim=1)) + 1e-8
    )

    # 2. Shading 方差 (Shading是不是太均匀了？)
    shading_std = shading.std()
    shading_flat_variance = shading.var().item()

    # 3. Normal 噪声检测 (计算梯度的平滑程度)
    # 计算法线图的拉普拉斯算子，检测高频噪声
    kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=normal.device)
    kernel = kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1)  # 改为 [3, 1, 3, 3]

    # 对每个批次计算拉普拉斯
    normal_noise_scores = []
    for i in range(B):
        # 为当前批次选择法线图
        normal_batch = normal[i:i + 1]  # [1, 3, H, W]

        # 使用分组卷积，每个通道独立计算
        normal_laplacian = F.conv2d(normal_batch, kernel, padding=1, groups=3)
        normal_noise_score = torch.abs(normal_laplacian).mean().item()
        normal_noise_scores.append(normal_noise_score)

    # 取所有批次的平均噪声分数
    normal_noise_score = np.mean(normal_noise_scores) if normal_noise_scores else 0.0

    # 4. Residual 贡献占比
    # Lambertian = albedo * shading
    lambertian_render = albedo * shading
    reconstruction = lambertian_render + global_residual
    if local_residual is not None:
        reconstruction = reconstruction + local_residual

    # 计算各部分的能量占比
    energy_lambert = (lambertian_render ** 2).mean().item()
    energy_global = (global_residual ** 2).mean().item()
    energy_local = (local_residual ** 2).mean().item() if local_residual is not None else 0.0
    total_energy = energy_lambert + energy_global + energy_local

    return {
        "albedo_image_corr": correlation.mean().item(),
        "shading_variance": shading_flat_variance,
        "normal_noise": normal_noise_score,
        "energy_ratio_lambert": energy_lambert / (total_energy + 1e-8),
        "energy_ratio_global": energy_global / (total_energy + 1e-8),
        "energy_ratio_local": energy_local / (total_energy + 1e-8)
    }

def visualize_diagnosis(images, depth, albedo, shading, normal, residual, scene_name, save_dir):
    """生成可视化诊断图表"""
    os.makedirs(save_dir, exist_ok=True)

    B, K, H, W = images.shape
    num_imgs_to_show = min(K, 4)  # 最多显示4张

    fig, axes = plt.subplots(4, 1 + num_imgs_to_show, figsize=(4 * (1 + num_imgs_to_show), 16))

    # 显示范围工具
    def show_tensor(ax, tensor, title, cmap='gray', vmin=None, vmax=None):
        if tensor.is_cuda:
            tensor = tensor.cpu()
        if tensor.dim() == 3:
            tensor = tensor.squeeze()
        if vmin is None: vmin = tensor.min()
        if vmax is None: vmax = tensor.max()
        im = ax.imshow(tensor, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis('off')
        return im

    # Row 1: Inputs
    show_tensor(axes[0, 0], albedo[0], "Albedo")
    for k in range(num_imgs_to_show):
        show_tensor(axes[0, k + 1], images[0, k], f"Input Img {k}")

    # Row 2: Geometry (Depth & Normal)
    show_tensor(axes[1, 0], depth[0], "Depth Map", cmap='plasma')
    # Normal: R=x, G=y, B=z, mapped to [0,1]
    normal_vis = (normal[0] + 1) / 2
    show_tensor(axes[1, 1], normal_vis.permute(1, 2, 0), "Normal Map")

    # Shading (为了对比，显示多张)
    for k in range(min(num_imgs_to_show - 1, 3)):
        show_tensor(axes[1, k + 2], shading[0, k], f"Shading {k}")

    # Row 3: Residuals
    if residual is not None:
        res_l1 = torch.abs(residual).mean(dim=1, keepdim=True)  # 合并K个残差
        # 放大显示残差以便观察
        show_tensor(axes[2, 0], res_l1[0] * 5, "Local Residual (x5)", cmap='hot')
    else:
        axes[2, 0].text(0.5, 0.5, "No Local Residual", ha='center')
        axes[2, 0].axis('off')

    # Rendered Results
    # Re-render albedo * shading
    lambertian = albedo * shading
    show_tensor(axes[2, 1], lambertian[0, 0], "Lambertian Rendered")
    if residual is not None:
        final_render = lambertian + residual
        show_tensor(axes[2, 2], final_render[0, 0], "Final + Residual")
        diff = (images[0, 0] - final_render[0, 0]).abs()
        show_tensor(axes[2, 3], diff, "Diff (Input - Final)", cmap='hot')

    # Row 4: Comparison (Input vs Albedo vs Shading)
    # Check if Albedo looks like Input
    mean_input = images.mean(dim=1)
    show_tensor(axes[3, 0], mean_input[0], "Mean Input")
    show_tensor(axes[3, 1], albedo[0], "Albedo")
    show_tensor(axes[3, 2], (albedo[0] - mean_input[0]).abs(), "Difference (|Input - Albedo|)", cmap='hot')

    # Shading intensity check
    sh_intensity = shading.mean(dim=1)
    # Shading通常应该比较平滑，如果有纹理，说明Albedo偷懒了
    show_tensor(axes[3, 3], sh_intensity[0], "Mean Shading Intensity", cmap='plasma')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{scene_name}_diagnosis.png")
    plt.savefig(save_path)
    plt.close()
    print(f"  可视化已保存: {save_path}")


def print_diagnosis_report(metrics, sh_coeffs):
    """打印诊断报告"""
    print("\n" + "=" * 80)
    print("【诊断报告】")
    print("=" * 80)

    # 1. Albedo 检查
    corr = metrics['albedo_image_corr']
    print(f"\n1. Albedo-Shading 模糊检查:")
    print(f"   Albedo 与输入图像的相关性: {corr:.4f}")
    if corr > 0.8:
        print(f"   ⚠ 警告：相关性过高！模型可能在'偷懒'，Albedo直接复制了输入图像的纹理。")
        print(f"   ➡ 建议：增大 albedo_smooth 权重，或检查光照约束是否失效。")
    else:
        print(f"   ✓ 看起来正常。")

    # 2. Shading 检查
    var = metrics['shading_variance']
    print(f"\n2. 光照 检查:")
    print(f"   Shading 方差: {var:.6f}")
    if var < 0.01:
        print(f"   ⚠ 警告：Shading 趋近于平坦（方差极低）。")
        print(f"   ➡ 模型可能认为光照是均匀的，把所有纹理都推给了Albedo。")
    else:
        print(f"   ✓ 光照分布看起来有变化。")

    # 3. 几何检查
    noise = metrics['normal_noise']
    print(f"\n3. 几何 检查:")
    print(f"   法线图噪声分数: {noise:.6f}")
    if noise > 0.1:
        print(f"   ⚠ 警告：法线图噪声过大！")
        print(f"   ➡ 可能是Sobel算子对深度图的噪声过度敏感，或者深度图本身预测失败。")
    else:
        print(f"   ✓ 几何结构相对平滑。")

    # 4. 能量分布检查
    print(f"\n4. 能量分布检查:")
    print(f"   Lambertian项占比: {metrics['energy_ratio_lambert'] * 100:.1f}%")
    print(f"   Global Residual占比: {metrics['energy_ratio_global'] * 100:.1f}%")
    print(f"   Local Residual占比: {metrics['energy_ratio_local'] * 100:.1f}%")

    if metrics['energy_ratio_lambert'] < 0.5:
        print(f"   ⚠ 警告：Residual 项占比过高！")
        print(f"   ➡ 主网络（U-Net）可能在偷懒，让残差模块去修正所有错误。")

    # 5. 光照系数检查
    print(f"\n5. 球谐系数 分布:")
    mean_sh = sh_coeffs.abs().mean(dim=1).cpu().numpy()  # [B, K, 9] -> [B, 9]
    print(f"   平均系数绝对值: {np.mean(mean_sh):.6f}")
    print(f"   直流分量(SH[0])均值: {mean_sh[0, 0]:.6f} (应 > 0)")
    print(f"   高阶系数(SH[1-8])均值: {np.mean(mean_sh[0, 1:]):.6f}")

    print("\n" + "=" * 80)


def main():
    """主函数"""
    # 导入默认配置（如果在config.py中定义）
    try:
        from config import get_default_config
    except ImportError:
        print("无法导入默认配置，但将使用临时默认配置继续")
        # 继续执行，因为我们已经在load_model_and_data中处理了默认配置的情况

    print("正在加载模型...")
    model, renderer, residual, val_loader, device = load_model_and_data(
        CONFIG_PATH, CHECKPOINT_PATH, DATA_ROOT, NUM_LIGHTS
    )

    model.eval()
    renderer.eval()
    residual.eval()

    all_metrics = []
    sh_coeffs_list = []

    print(f"开始诊断，共 {len(val_loader)} 个批次，将处理前 {NUM_SAMPLES} 个...")

    with torch.no_grad():
        for i, (images, scene_names) in enumerate(val_loader):
            if i >= NUM_SAMPLES:
                break

            scene_name = scene_names[0]
            print(f"\n--- 处理场景 [{i + 1}/{NUM_SAMPLES}]: {scene_name} ---")

            images = images.to(device)

            # 推理
            depth, albedo, sh_coeffs, weight_map = model(images)
            rendered, normal, shading = renderer(depth, albedo, sh_coeffs)
            final_render, global_residual, local_residual = residual(albedo, shading, normal, sh_coeffs)

            # 计算指标
            metrics = calculate_metrics(images, albedo, shading, normal, local_residual, global_residual)
            all_metrics.append(metrics)
            sh_coeffs_list.append(sh_coeffs)

            # 打印单个样本指标
            print(f"  Albedo相关性: {metrics['albedo_image_corr']:.4f}")
            print(f"  Shading方差: {metrics['shading_variance']:.6f}")
            print(f"  Normal噪声: {metrics['normal_noise']:.6f}")

            # 保存可视化
            visualize_diagnosis(
                images, depth, albedo, shading, normal,
                local_residual, scene_name, OUTPUT_DIR
            )

    # 打印综合报告
    avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys()}
    # 取第一个样本的SH做代表
    print_diagnosis_report(avg_metrics, sh_coeffs_list[0])

    print(f"\n诊断完成！所有图片和详细日志已保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()