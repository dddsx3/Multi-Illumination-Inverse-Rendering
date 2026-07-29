import os
import torch
import numpy as np
from PIL import Image
import glob
import shutil


def check_scene_std(scene_folder, num_lights=5, file_extension='.png'):
    """
    检查场景文件夹中图片的标准差
    
    Args:
        scene_folder: 场景文件夹路径
        num_lights: 每个场景的光照图片数量
        file_extension: 图片文件扩展名
    
    Returns:
        float: 图片的标准差
        bool: 是否应该删除该场景
    """
    # 查找场景文件夹中的所有 light_XXX.png 图像
    image_pattern = os.path.join(scene_folder, f'light_*{file_extension}')
    image_files = sorted(glob.glob(image_pattern))
    
    if len(image_files) < num_lights:
        print(f"场景 {os.path.basename(scene_folder)} 图片数量不足，跳过")
        return 0.0, False
    
    # 取前K张图像
    image_files = image_files[:num_lights]
    
    # 加载并处理图片
    images = []
    for img_path in image_files:
        try:
            with Image.open(img_path).convert('L') as img:
                # 转换为numpy数组并归一化到[0, 1]
                img_array = np.array(img, dtype=np.float32) / 255.0
                # 应用Gamma矫正，与data_loader.py保持一致
                img_array = np.power(img_array, 1.0/2.2)
                images.append(img_array)
        except Exception as e:
            print(f"加载图片 {img_path} 时出错: {str(e)}")
            return 0.0, False
    
    # 转换为torch tensor
    images_tensor = torch.tensor(images)
    
    # 计算标准差
    std_of_batch = torch.std(images_tensor)
    std_value = std_of_batch.item()
    
    # 判断是否应该删除
    # 应用Gamma矫正后，数据的标准差会上升，所以放宽阈值
    should_delete = std_value < 0.01
    
    return std_value, should_delete


def main():
    """
    主函数：扫描所有场景文件夹并检查标准差
    """
    # 数据集根目录
    root_dir = "C:\Users\35702\Desktop\processed_data"
    num_lights = 5
    file_extension = '.png'
    
    print(f"开始检查数据集: {root_dir}")
    print(f"每个场景检查 {num_lights} 张图片")
    print("=" * 80)
    
    # 查找所有 rgb* 文件夹
    rgb_folders = []
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path) and item.startswith('rgb'):
            rgb_folders.append(item_path)
    
    if len(rgb_folders) == 0:
        print(f"未找到任何 rgb* 文件夹！请检查路径: {root_dir}")
        return
    
    print(f"找到 {len(rgb_folders)} 个 rgb* 文件夹")
    
    total_scenes = 0
    deleted_scenes = 0
    
    for rgb_folder in sorted(rgb_folders):
        print(f"\n处理: {rgb_folder}")
        
        # 查找所有 scene_* 文件夹
        scene_folders = []
        for item in os.listdir(rgb_folder):
            item_path = os.path.join(rgb_folder, item)
            if os.path.isdir(item_path) and item.startswith('scene_'):
                scene_folders.append(item_path)
        
        if len(scene_folders) == 0:
            print(f"  未找到任何 scene_* 文件夹")
            continue
        
        print(f"  找到 {len(scene_folders)} 个场景")
        
        for scene_folder in sorted(scene_folders):
            scene_name = os.path.basename(scene_folder)
            total_scenes += 1
            
            # 检查场景
            std_value, should_delete = check_scene_std(scene_folder, num_lights, file_extension)
            
            print(f"  场景 {scene_name}: std = {std_value:.4f}")
            
            if should_delete:
                print(f"  ⚠️  标准差小于0.01，删除场景 {scene_name}")
                try:
                    shutil.rmtree(scene_folder)
                    deleted_scenes += 1
                    print(f"  ✅  场景 {scene_name} 已删除")
                except Exception as e:
                    print(f"  ❌ 删除场景 {scene_name} 时出错: {str(e)}")
            else:
                print(f"  ✅  数据没问题")
    
    print("\n" + "=" * 80)
    print("检查完成！")
    print(f"总场景数: {total_scenes}")
    print(f"删除的场景数: {deleted_scenes}")
    print(f"保留的场景数: {total_scenes - deleted_scenes}")


if __name__ == "__main__":
    main()