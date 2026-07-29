import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Optional, Tuple, List
import warnings
import glob
import re


class MultiLightingDataset(Dataset):
    """
    多光照灰度图像数据集

    自动扫描数据集根目录下的所有 rgb* 文件夹，从每个 rgb* 文件夹中查找所有 scene_XXXXXX 文件夹，
    每个 scene 文件夹包含多张光照图像。

    数据结构：
    - 数据集根目录/
      - rgb/
        - scene_000000/
          - light_001.png
          - light_002.png
          - ...
      - rgb_1/
        - scene_000000/
          - ...
      - rgb_2/
        - ...

    Args:
        root_dir (str): 数据集根目录
        num_lights (int): 每个场景的光照图像数量K，默认为5
        image_size (Tuple[int, int]): 输出图像尺寸 (H, W)，用于裁剪
        is_training (bool): 是否为训练模式（启用数据增强）
        file_extension (str): 图像文件扩展名，默认为'.png'
        max_rotation_angle (float): 最大旋转角度（度），默认10度
        horizontal_flip_prob (float): 水平翻转概率，默认0.5

    Returns:
        images (torch.Tensor): 形状为 [K, H, W] 的张量
        scene_name (str): 场景名称
    """

    def __init__(
        self,
        root_dir: str,
        num_lights: int = 5,
        image_size: Tuple[int, int] = (256, 256),
        is_training: bool = True,
        file_extension: str = '.png',
        max_rotation_angle: float = 10.0,
        horizontal_flip_prob: float = 0.5
    ):
        self.root_dir = root_dir
        self.num_lights = num_lights
        self.image_size = image_size
        self.is_training = is_training
        self.file_extension = file_extension
        self.max_rotation_angle = max_rotation_angle
        self.horizontal_flip_prob = horizontal_flip_prob
        self.valid_scenes = []
        self.scene_image_paths = {}
        self._validate_and_build_dataset()

        if len(self.valid_scenes) == 0:
            raise ValueError(f"未找到有效场景！请检查路径: {root_dir}")

        # 构建数据增强pipeline
        self.transform = self._build_transforms()

        print(f"数据集初始化完成: {len(self.valid_scenes)} 个有效场景")

    def _validate_and_build_dataset(self):
        """
        自动扫描数据集根目录下的所有 rgb* 文件夹，构建数据集

        处理流程：
        1. 查找所有 rgb* 文件夹（rgb, rgb_1, rgb_2, ...）
        2. 在每个 rgb* 文件夹中查找所有 scene_XXXXXX 文件夹
        3. 在每个 scene 文件夹中查找所有 light_XXX.png 图像
        4. 验证图像数量是否满足要求
        5. 不满 num_lights 张的舍弃
        """
        # 查找所有 rgb* 文件夹
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"数据集根目录不存在: {self.root_dir}")

        if not os.path.isdir(self.root_dir):
            raise NotADirectoryError(f"路径不是目录: {self.root_dir}")
        rgb_folders = []
        for item in os.listdir(self.root_dir):
            item_path = os.path.join(self.root_dir, item)
            if os.path.isdir(item_path) and item.startswith('rgb'):
                rgb_folders.append(item_path)

        if len(rgb_folders) == 0:
            raise ValueError(f"未找到任何 rgb* 文件夹！请检查路径: {self.root_dir}")

        print(f"找到 {len(rgb_folders)} 个 rgb* 文件夹")

        total_scenes = 0
        total_images = 0
        discarded_images = 0

        for rgb_folder in sorted(rgb_folders):
            # 在每个 rgb* 文件夹中查找所有 scene_XXXXXX 文件夹
            scene_folders = []
            for item in os.listdir(rgb_folder):
                item_path = os.path.join(rgb_folder, item)
                if os.path.isdir(item_path) and item.startswith('scene_'):
                    scene_folders.append(item_path)

            if len(scene_folders) == 0:
                warnings.warn(f"rgb 文件夹 '{rgb_folder}' 中没有找到 scene_* 文件夹")
                continue

            print(f"  处理: {rgb_folder}")
            print(f"    找到 {len(scene_folders)} 个场景")

            # 处理每个场景文件夹
            for scene_folder in sorted(scene_folders):
                scene_name = os.path.basename(scene_folder)

                # 查找场景文件夹中的所有 light_XXX.png 图像
                image_pattern = os.path.join(scene_folder, f'light_*{self.file_extension}')
                image_files = sorted(glob.glob(image_pattern))

                if len(image_files) == 0:
                    warnings.warn(f"场景 '{scene_name}' 中没有找到图像文件")
                    continue

                total_images += len(image_files)

                # 检查图像数量
                if len(image_files) < self.num_lights:
                    warnings.warn(
                        f"场景 '{scene_name}' 图像数量不足: "
                        f"需要 {self.num_lights} 张，实际 {len(image_files)} 张，已舍弃"
                    )
                    discarded_images += len(image_files)
                    continue

                # 取前K张图像
                image_files = image_files[:self.num_lights]  # ← 确保num_lights是5

                # 验证所有图像尺寸一致
                try:
                    sizes = []
                    for img_path in image_files:
                        with Image.open(img_path) as img:
                            sizes.append(img.size)

                    if len(set(sizes)) > 1:
                        warnings.warn(
                            f"场景 '{scene_name}' 包含不同尺寸的图像: {set(sizes)}，已舍弃"
                        )
                        discarded_images += len(image_files)
                        continue

                    # 验证通过，添加到有效场景列表
                    self.valid_scenes.append(scene_name)
                    self.scene_image_paths[scene_name] = image_files
                    total_scenes += 1

                except Exception as e:
                    warnings.warn(f"验证场景 '{scene_name}' 时出错: {str(e)}，已舍弃")
                    discarded_images += len(image_files)
                    continue

        # 计算舍弃的图像数
        used_images = len(self.valid_scenes) * self.num_lights
        discarded_images = total_images - used_images

        print(f"\n总场景数: {total_scenes}")
        print(f"总图像数: {total_images}")
        print(f"使用的图像: {used_images}")
        print(f"舍弃的图像: {discarded_images}")
        print(f"使用率: {used_images / total_images * 100:.2f}%")

    def _build_transforms(self):
        """
        构建数据增强pipeline

        训练模式：
        - 随机裁剪到固定大小
        - 随机水平翻转
        - 随机小角度旋转

        验证/测试模式：
        - 中心裁剪到固定大小
        """
        if self.is_training:
            # 训练模式：启用数据增强
            transform_list = [
                transforms.RandomCrop(self.image_size),
                transforms.RandomHorizontalFlip(p=self.horizontal_flip_prob),
                transforms.RandomRotation(
                    degrees=(-self.max_rotation_angle, self.max_rotation_angle),
                    fill=0  # 旋转后的空白区域填充为0（黑色）
                )
            ]
        else:
            # 验证/测试模式：仅中心裁剪
            transform_list = [
                transforms.CenterCrop(self.image_size)
            ]

        return transforms.Compose(transform_list)

    def _load_grayscale_image(self, image_path: str) -> Image.Image:
        """
        加载单张灰度图像

        Args:
            image_path: 图像文件路径

        Returns:
            PIL.Image: 灰度图像对象
        """
        try:
            # 打开并转换为灰度图
            img = Image.open(image_path).convert('L')
            return img
        except Exception as e:
            raise IOError(f"无法加载图像 {image_path}: {str(e)}")

    def __len__(self) -> int:
        """返回数据集中的场景数量"""
        return len(self.valid_scenes)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        获取指定索引的场景数据

        Args:
            idx: 场景索引

        Returns:
            images: 形状为 [K, H, W] 的张量，K为光照数量
            scene_name: 场景名称
        """
        # 获取场景名称和对应的图像路径
        scene_name = self.valid_scenes[idx]
        image_paths = self.scene_image_paths[scene_name]

        # 加载所有K张图像
        images_pil = []
        for img_path in image_paths:
            img = self._load_grayscale_image(img_path)
            images_pil.append(img)

        # 对所有图像应用相同的数据增强变换
        # 使用相同的随机种子确保变换一致性
        seed = np.random.randint(2147483647)

        transformed_images = []
        for img in images_pil:
            # 设置相同的随机种子
            torch.manual_seed(seed)
            np.random.seed(seed)

            # 应用变换
            img_transformed = self.transform(img)
            transformed_images.append(img_transformed)

        # 转换为tensor并归一化到 [0, 1]
        image_tensors = []
        for img in transformed_images:
            # PIL Image转numpy数组
            img_array = np.array(img, dtype=np.float32)

            # 归一化到 [0, 1]
            img_array = img_array / 255.0

            # 🔴【关键修改】应用 Gamma 矫正
            # 将线性数据转换为感知数据，增加对比度
            # Gamma 值通常选 2.2，这是标准转换
            img_array = np.power(img_array, 1.0/2.2)  # 反 Gamma 矫正

            # 转换为torch tensor
            img_tensor = torch.from_numpy(img_array)
            image_tensors.append(img_tensor)

        # 堆叠成 [K, H, W] 的tensor
        images_stacked = torch.stack(image_tensors, dim=0)

        return images_stacked, scene_name

    def get_scene_info(self, idx: int) -> dict:
        """
        获取场景的详细信息

        Args:
            idx: 场景索引

        Returns:
            dict: 包含场景名称、图像路径等信息的字典
        """
        scene_name = self.valid_scenes[idx]
        image_paths = self.scene_image_paths[scene_name]

        return {
            'scene_name': scene_name,
            'num_images': len(image_paths),
            'image_paths': image_paths,
            'image_size': self.image_size
        }


def create_data_loader(
    dataset: MultiLightingDataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = True
) -> DataLoader:
    """
    创建优化的DataLoader

    Args:
        dataset: MultiLightingDataset实例
        batch_size: 批次大小
        shuffle: 是否打乱数据
        num_workers: 数据加载的工作进程数（Windows推荐2-4，Linux推荐CPU核心数-1）
        pin_memory: 是否将数据固定在内存中（GPU训练时推荐True）
        prefetch_factor: 每个worker预取的batch数量（推荐2-4）
        persistent_workers: 是否在epoch之间保持worker进程（训练时推荐True）

    Returns:
        DataLoader实例
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=dataset.is_training,  # 训练时丢弃最后不完整的batch
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False
    )


def get_data_loaders(
        config,
        train_val_split: float = 0.8
    ) -> Tuple[DataLoader, DataLoader]:
    """
    获取训练和验证数据加载器

    Args:
        config: 配置对象，包含数据相关配置
        train_val_split: 训练集与验证集的分割比例

    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
    """
    # 从配置中提取参数
    root_dir = config.data.root_dir
    num_lights = config.data.num_lights
    image_size = config.data.image_size
    batch_size = config.data.batch_size
    num_workers = config.data.num_workers
    pin_memory = config.data.pin_memory
    prefetch_factor = config.data.prefetch_factor
    persistent_workers = config.data.persistent_workers
    file_extension = config.data.file_extension
    max_rotation_angle = config.data.max_rotation_angle
    horizontal_flip_prob = config.data.horizontal_flip_prob

    # 创建完整数据集
    full_dataset = MultiLightingDataset(
        root_dir=root_dir,
        num_lights=num_lights,
        image_size=image_size,
        is_training=False,  # 先创建完整数据集，然后再分割
        file_extension=file_extension,
        max_rotation_angle=max_rotation_angle,
        horizontal_flip_prob=horizontal_flip_prob
    )

    # 分割数据集为训练集和验证集
    dataset_size = len(full_dataset)
    train_size = int(train_val_split * dataset_size)
    val_size = dataset_size - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # 更新训练数据集的is_training属性
    train_dataset.dataset.is_training = True
    val_dataset.dataset.is_training = False

    # 创建数据加载器
    train_loader = create_data_loader(
        train_dataset.dataset,  # 使用原始数据集，因为random_split返回的是Subset
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers
    )

    val_loader = create_data_loader(
        val_dataset.dataset,  # 使用原始数据集
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers
    )

    return train_loader, val_loader

# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    """
    测试MultiLightingDataset的完整功能
    """
    print("=" * 80)
    print("多光照灰度图像数据集测试")
    print("=" * 80)

    # ========== 配置参数 ==========
    ROOT_DIR = "C:\\Users\\35702\\Desktop\\processed_data"  # 数据集根目录
    NUM_LIGHTS = 5  # 每个场景5张不同光照的图像
    IMAGE_SIZE = (256, 256)  # 裁剪后的图像尺寸
    BATCH_SIZE = 2

    print(f"\n【数据集配置】")
    print(f"根目录: {ROOT_DIR}")
    print(f"每场景光照数: {NUM_LIGHTS}")
    print(f"图像尺寸: {IMAGE_SIZE}")
    print(f"批次大小: {BATCH_SIZE}")

    # ========== 创建训练数据集 ==========
    print("\n" + "=" * 80)
    print("【1】创建训练数据集（启用数据增强）")
    print("=" * 80)

    try:
        train_dataset = MultiLightingDataset(
            root_dir=ROOT_DIR,
            num_lights=NUM_LIGHTS,
            image_size=IMAGE_SIZE,
            is_training=True,  # 训练模式
            file_extension='.png',
            max_rotation_angle=10.0,
            horizontal_flip_prob=0.5
        )

        print(f"\n✓ 训练数据集创建成功")
        print(f"  有效场景数: {len(train_dataset)}")

        if len(train_dataset) > 0:
            # 显示第一个场景的信息
            scene_info = train_dataset.get_scene_info(0)
            print(f"\n  第一个场景信息:")
            print(f"    场景名称: {scene_info['scene_name']}")
            print(f"    图像数量: {scene_info['num_images']}")
            print(f"    输出尺寸: {scene_info['image_size']}")

            # 加载一个样本
            print(f"\n  加载样本数据...")
            images, scene_name = train_dataset[0]
            print(f"    场景名称: {scene_name}")
            print(f"    图像形状: {images.shape}  # [K, H, W]")
            print(f"    数据类型: {images.dtype}")
            print(f"    数值范围: [{images.min():.4f}, {images.max():.4f}]")
            print(f"    平均值: {images.mean():.4f}")

            # ========== 创建DataLoader ==========
            print("\n" + "=" * 80)
            print("【2】创建训练DataLoader")
            print("=" * 80)

            train_loader = create_data_loader(
                train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=0  # Windows下推荐设为0，Linux可设为4
            )

            print(f"\n✓ DataLoader创建成功")
            print(f"  批次大小: {BATCH_SIZE}")
            print(f"  总批次数: {len(train_loader)}")
            print(f"  是否打乱: True")

            # 加载一个batch
            print(f"\n  加载一个批次...")
            for batch_idx, (batch_images, batch_scenes) in enumerate(train_loader):
                print(f"\n  Batch {batch_idx + 1}:")
                print(f"    批次形状: {batch_images.shape}  # [Batch, K, H, W]")
                print(f"    场景名称: {batch_scenes}")
                print(f"    数值范围: [{batch_images.min():.4f}, {batch_images.max():.4f}]")
                print(f"    数据类型: {batch_images.dtype}")
                print(f"    占用内存: {batch_images.element_size() * batch_images.nelement() / 1024 / 1024:.2f} MB")

                # 只显示第一个batch
                break

            # 测试数据增强的随机性
            print("\n" + "=" * 80)
            print("【3】测试数据增强的随机性")
            print("=" * 80)

            print(f"\n  连续加载同一场景3次，验证数据增强效果...")
            for i in range(3):
                images, _ = train_dataset[0]
                print(f"  第{i+1}次加载 - 均值: {images.mean():.4f}, 标准差: {images.std():.4f}")

    except ValueError as e:
        print(f"\n✗ 错误: {str(e)}")
        print(f"\n提示：请确保数据集目录结构如下：")
        print(f"\n{ROOT_DIR}/")
        print(f"  ├── rgb/")
        print(f"  │   ├── scene_000000/")
        print(f"  │   │   ├── light_001.png  # 光照1")
        print(f"  │   │   ├── light_002.png  # 光照2")
        print(f"  │   │   ├── light_003.png  # 光照3")
        print(f"  │   │   ├── light_004.png  # 光照4")
        print(f"  │   │   └── light_005.png  # 光照5")
        print(f"  │   ├── scene_000001/")
        print(f"  │   │   └── ...")
        print(f"  ├── rgb_1/")
        print(f"  │   └── ...")
        print(f"  └── ...")

    # ========== 创建验证数据集 ==========
    print("\n" + "=" * 80)
    print("【4】创建验证数据集（无数据增强）")
    print("=" * 80)

    try:
        val_dataset = MultiLightingDataset(
            root_dir=ROOT_DIR,
            num_lights=NUM_LIGHTS,
            image_size=IMAGE_SIZE,
            is_training=False,  # 验证模式，不启用数据增强
            file_extension='.png'
        )

        print(f"\n✓ 验证数据集创建成功")
        print(f"  有效场景数: {len(val_dataset)}")
        print(f"  数据增强: 关闭（仅中心裁剪）")

        if len(val_dataset) > 0:
            val_loader = create_data_loader(
                val_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,  # 验证集不打乱
                num_workers=0
            )

            print(f"  总批次数: {len(val_loader)}")

            # 测试验证集的确定性
            print(f"\n  验证数据增强已关闭（连续2次加载应相同）...")
            images1, _ = val_dataset[0]
            images2, _ = val_dataset[0]
            is_same = torch.allclose(images1, images2)
            print(f"  两次加载结果相同: {is_same}")

    except ValueError as e:
        print(f"\n✗ 验证集错误: {str(e)}")

    # ========== 测试边界情况 ==========
    print("\n" + "=" * 80)
    print("【5】边界情况测试")
    print("=" * 80)

    print("\n  测试不存在的目录...")
    try:
        test_dataset = MultiLightingDataset(
            root_dir="./non_existent_dir",
            num_lights=NUM_LIGHTS,
            image_size=IMAGE_SIZE,
            is_training=False
        )
    except Exception as e:
        print(f"  ✓ 正确捕获异常: {str(e)}")

    print("\n" + "=" * 80)
    print("测试完成!")
    print("=" * 80)

    # ========== 使用建议 ==========
    print("\n【使用建议】")
    print("1. 训练时设置 is_training=True 启用数据增强")
    print("2. Windows系统建议 num_workers=0，Linux可用 num_workers=4")
    print("3. GPU训练时设置 pin_memory=True 加速数据传输")
    print("4. 根据GPU显存调整 batch_size")
    print("5. 确保数据集目录包含多个 rgb* 文件夹")
    print("6. 每个 rgb* 文件夹包含多个 scene_* 文件夹")
    print("7. 每个 scene_* 文件夹包含 light_*.png 图像")
    print("8. 每个 scene_* 文件夹至少包含 num_lights 张图像")

    print("\n" + "=" * 80)