import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from typing import Optional, Tuple, List, Dict
import warnings
import glob


class MultiLightingDataset(Dataset):
    """
    多光照逆渲染数据集（Phase 1：支持 GT 通道与同步增强）

    支持两种目录布局：

    1) Phase 1 合成数据协议（扁平布局，render_dataset.py 的输出）：
       root_dir/
         scene_A/  light_001..K.png + depth.npy + albedo.npy +
                   normal.npy + mask.npy (+ sh_coeffs.npy)
         scene_B/  ...

    2) 旧版布局（向后兼容）：
       root_dir/
         rgb/      scene_000000/light_001..K.png ...
         rgb_1/    ...

    GT 约定（与 evaluate.py / render_dataset.py 一致）：
      depth [1,H,W] float32 视空间正深度；albedo [1,H,W] float32 [0,1]；
      normal [3,H,W] float32 单位向量、面朝相机 z>0；mask [1,H,W] uint8 0/1；
      sh_coeffs [K,9] float32。

    增强策略（Phase 1 审计 A3 结论）：
      - 训练：随机裁剪 + 随机水平翻转，图像与 GT 共用同一组随机参数；
        翻转时法线 x 分量取反。
      - 旋转增强暂不启用（GT 同步旋转未实现，见升级计划 A3），配置了
        max_rotation_angle > 0 时会告警并忽略。
      - 验证：中心裁剪，无增强。
      - 图像解码保持旧约定：uint8 -> /255 -> ^(1/2.2) 反 gamma。

    Args:
        root_dir: 数据集根目录
        num_lights: 每个场景的光照图像数量 K，默认为 5
        image_size: 输出尺寸 (H, W)
        is_training: 是否启用训练增强
        file_extension: 图像扩展名，默认 .png
        max_rotation_angle: 兼容参数；旋转增强未启用，>0 时告警忽略
        horizontal_flip_prob: 水平翻转概率，默认 0.5
        scene_subset: 只使用指定场景（用于确定性划分）
        load_gt: 是否加载 GT 通道（场景缺 GT 文件时该场景 gt=None）
    """

    _GT_CORE = ("depth.npy", "albedo.npy", "normal.npy", "mask.npy")

    def __init__(
        self,
        root_dir: str,
        num_lights: int = 5,
        image_size: Tuple[int, int] = (256, 256),
        is_training: bool = True,
        file_extension: str = ".png",
        max_rotation_angle: float = 0.0,
        horizontal_flip_prob: float = 0.5,
        scene_subset: Optional[List[str]] = None,
        load_gt: bool = True,
        modality: str = "gray"
    ):
        self.root_dir = root_dir
        self.num_lights = num_lights
        self.image_size = tuple(image_size)
        self.is_training = is_training
        self.file_extension = file_extension
        self.modality = str(modality).lower()
        assert self.modality in ("gray", "rgb"), f"未知模态 {modality}"
        # v3 双输出命名：灰度 light_NNN.png / 彩色 light_NNN_rgb.png
        self._glob_pattern = (
            f"light_[0-9][0-9][0-9]{file_extension}" if self.modality == "gray"
            else f"light_[0-9][0-9][0-9]_rgb{file_extension}")
        self._pil_mode = "L" if self.modality == "gray" else "RGB"
        self.max_rotation_angle = float(max_rotation_angle)
        self.horizontal_flip_prob = float(horizontal_flip_prob)
        self.load_gt = load_gt
        if self.is_training and self.max_rotation_angle > 0:
            warnings.warn(
                "旋转增强暂未实现图像/GT 同步（含法线向量旋转），已按 0 处理；"
                "参见 Phase 1 规划 T1.4-A3。")
            self.max_rotation_angle = 0.0

        self.valid_scenes: List[str] = []
        self.scene_image_paths: Dict[str, List[str]] = {}
        self.scene_dirs: Dict[str, str] = {}
        self.scene_has_gt: Dict[str, bool] = {}

        self._validate_and_build_dataset()

        if scene_subset is not None:
            keep = set(scene_subset)
            missing = keep - set(self.valid_scenes)
            if missing:
                raise ValueError(f"scene_subset 含不存在的场景: {sorted(missing)}")
            self.valid_scenes = [s for s in self.valid_scenes if s in keep]

        if len(self.valid_scenes) == 0:
            raise ValueError(f"未找到有效场景！请检查路径: {root_dir}")

        mode_txt = "train(增强)" if self.is_training else "val(中心裁剪)"
        n_gt = sum(self.scene_has_gt.values())
        print(f"数据集初始化完成[{mode_txt}]: {len(self.valid_scenes)} 个有效场景"
              f"（含 GT: {n_gt}）")

    # ------------------------------------------------------------------
    # 扫描
    # ------------------------------------------------------------------
    def _validate_and_build_dataset(self):
        """扫描根目录构建场景列表（自动识别扁平协议布局 / 旧版 rgb* 布局）"""
        if not os.path.isdir(self.root_dir):
            raise NotADirectoryError(f"数据集根目录不存在或不是目录: {self.root_dir}")

        rgb_folders = [
            os.path.join(self.root_dir, item)
            for item in sorted(os.listdir(self.root_dir))
            if os.path.isdir(os.path.join(self.root_dir, item))
            and item.startswith("rgb")
        ]

        if rgb_folders:
            print(f"检测到旧版布局: {len(rgb_folders)} 个 rgb* 文件夹")
            scene_dirs = []
            for rgb_folder in rgb_folders:
                for item in sorted(os.listdir(rgb_folder)):
                    p = os.path.join(rgb_folder, item)
                    if os.path.isdir(p) and item.startswith("scene_"):
                        scene_dirs.append(p)
        else:
            # Phase 1 扁平协议布局：根目录下直接是场景目录
            scene_dirs = []
            for item in sorted(os.listdir(self.root_dir)):
                p = os.path.join(self.root_dir, item)
                if os.path.isdir(p) and not item.startswith("_"):
                    scene_dirs.append(p)
            if scene_dirs:
                print(f"检测到扁平协议布局: {len(scene_dirs)} 个候选场景目录")
            else:
                raise ValueError(f"未找到任何 rgb* 或场景目录！请检查路径: {self.root_dir}")

        total_images = 0
        for scene_folder in scene_dirs:
            scene_name = os.path.basename(scene_folder)
            image_files = sorted(glob.glob(
                os.path.join(scene_folder, self._glob_pattern)))

            if len(image_files) < self.num_lights:
                if len(image_files) > 0:
                    warnings.warn(
                        f"场景 '{scene_name}' 图像数量不足: 需要 "
                        f"{self.num_lights} 张，实际 {len(image_files)} 张，已舍弃")
                continue

            image_files = image_files[:self.num_lights]
            total_images += len(image_files)

            try:
                sizes = set()
                for img_path in image_files:
                    with Image.open(img_path) as img:
                        sizes.add(img.size)
                if len(sizes) > 1:
                    warnings.warn(f"场景 '{scene_name}' 图像尺寸不一致: {sizes}，已舍弃")
                    continue
            except Exception as e:
                warnings.warn(f"验证场景 '{scene_name}' 出错: {e}，已舍弃")
                continue

            has_gt = (
                self.load_gt
                and all(os.path.isfile(os.path.join(scene_folder, f))
                        for f in self._GT_CORE)
            )
            self.valid_scenes.append(scene_name)
            self.scene_image_paths[scene_name] = image_files
            self.scene_dirs[scene_name] = scene_folder
            self.scene_has_gt[scene_name] = has_gt

        used_images = len(self.valid_scenes) * self.num_lights
        print(f"有效场景: {len(self.valid_scenes)} | 图像: {used_images}/{total_images}")

    # ------------------------------------------------------------------
    # GT 加载
    # ------------------------------------------------------------------
    def _load_gt(self, scene_name: str) -> Optional[Dict[str, torch.Tensor]]:
        """加载场景 GT；缺文件时返回 None（纯自监督兼容）"""
        if not self.scene_has_gt.get(scene_name, False):
            return None
        d = self.scene_dirs[scene_name]
        gt = {
            "depth": torch.from_numpy(np.load(os.path.join(d, "depth.npy"))).float(),
            "albedo": torch.from_numpy(np.load(os.path.join(d, "albedo.npy"))).float(),
            "normal": torch.from_numpy(np.load(os.path.join(d, "normal.npy"))).float(),
            "mask": torch.from_numpy(np.load(os.path.join(d, "mask.npy"))).float(),
        }
        sh_path = os.path.join(d, "sh_coeffs.npy")
        if os.path.isfile(sh_path):
            gt["sh_coeffs"] = torch.from_numpy(np.load(sh_path)).float()
        return gt

    # ------------------------------------------------------------------
    # 同步增强
    # ------------------------------------------------------------------
    def _sample_aug_params(self, h: int, w: int):
        """采样一组裁剪/翻转参数（图像与所有 GT 通道共用）"""
        th, tw = self.image_size
        if self.is_training:
            g = torch.Generator()
            g.manual_seed(self._aug_seed)
            top = int(torch.randint(0, h - th + 1, (1,), generator=g).item())
            left = int(torch.randint(0, w - tw + 1, (1,), generator=g).item())
            flip = torch.rand((), generator=g).item() < self.horizontal_flip_prob
        else:
            top = max((h - th) // 2, 0)
            left = max((w - tw) // 2, 0)
            flip = False
        return top, left, flip

    def __len__(self) -> int:
        return len(self.valid_scenes)

    def __getitem__(self, idx: int):
        """
        Returns:
            images: [K, H, W] float32，反 gamma 解码后的线性灰度
            gt: dict 或 None：
                depth/albedo/mask [C,H,W]，normal [3,H,W]，sh_coeffs [K,9]
                （已应用与 images 完全相同的裁剪/翻转；翻转时法线 x 取反）
            scene_name: str
        """
        scene_name = self.valid_scenes[idx]
        image_paths = self.scene_image_paths[scene_name]

        images_pil = []
        for img_path in image_paths:
            try:
                images_pil.append(Image.open(img_path).convert(self._pil_mode))
            except Exception as e:
                raise IOError(f"无法加载图像 {img_path}: {e}")

        w0, h0 = images_pil[0].size
        th, tw = self.image_size

        # 尺寸不足目标时零填充到目标（正常情况下渲染尺寸即目标尺寸）
        pad_h, pad_w = max(th - h0, 0), max(tw - w0, 0)

        self._aug_seed = np.random.randint(2147483647)
        top, left, flip = self._sample_aug_params(h0 + pad_h, w0 + pad_w)

        # ---- 图像 ----
        image_tensors = []
        # rgb 模态额外产出重建目标：编码域 BT.709 luma（与 _regen_gray.py
        # 推导灰度 PNG 的公式逐位一致，保证 F-N5-rgb 与灰度臂监督同源）
        luma_tensors = [] if self.modality == "rgb" else None
        for img in images_pil:
            if pad_h > 0 or pad_w > 0:
                canvas = Image.new(self._pil_mode, (max(w0, tw), max(h0, th)))
                canvas.paste(img, (0, 0))
                img = canvas
            img = img.crop((left, top, left + tw, top + th))
            if flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            arr = np.asarray(img, dtype=np.float32)
            if luma_tensors is not None:
                luma8 = np.round(0.2126 * arr[..., 0] + 0.7152 * arr[..., 1]
                                 + 0.0722 * arr[..., 2])
                lt = np.power(luma8 / 255.0, 1.0 / 2.2).astype(np.float32)
                luma_tensors.append(torch.from_numpy(lt))
            arr = arr / 255.0
            arr = np.power(arr, 1.0 / 2.2)  # 反 gamma（与旧 data_loader 一致）
            t = torch.from_numpy(arr)
            if self.modality == "rgb":
                t = t.permute(2, 0, 1).contiguous()   # [3, H, W]
            image_tensors.append(t)
        images = torch.stack(image_tensors, dim=0)  # [K, H, W]；rgb 为 [K, 3, H, W]

        # ---- GT（与图像同一组裁剪/翻转参数）----
        gt = self._load_gt(scene_name)
        if gt is not None:
            gt_out = {}
            for key in ("depth", "albedo", "normal", "mask"):
                t = gt[key]
                if pad_h > 0:
                    t = torch.cat([t, torch.zeros(t.shape[0], pad_h, t.shape[2])], dim=1)
                if pad_w > 0:
                    t = torch.cat([t, torch.zeros(t.shape[0], t.shape[1], pad_w)], dim=2)
                t = t[:, top:top + th, left:left + tw].clone()
                if flip:
                    t = torch.flip(t, dims=[-1])
                    if key == "normal":
                        t[0] = -t[0]  # 相机空间法线 x 分量取反
                gt_out[key] = t.contiguous()
            if "sh_coeffs" in gt:
                gt_out["sh_coeffs"] = gt["sh_coeffs"]  # 全局量，不受空间增强影响
            if luma_tensors is not None:
                gt_out["image_luma"] = torch.stack(luma_tensors, dim=0)  # [K,H,W]
            gt = gt_out

        return images, gt, scene_name

    def get_scene_info(self, idx: int) -> dict:
        scene_name = self.valid_scenes[idx]
        return {
            "scene_name": scene_name,
            "num_images": len(self.scene_image_paths[scene_name]),
            "image_paths": self.scene_image_paths[scene_name],
            "has_gt": self.scene_has_gt.get(scene_name, False),
            "image_size": self.image_size,
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
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=dataset.is_training,  # 训练时丢弃最后不完整的 batch
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False
    )


def split_scene_names(root_dir: str, train_val_split: float = 0.8,
                      seed: int = 42) -> Tuple[List[str], List[str]]:
    """按场景做确定性 train/val 划分（固定种子，两次调用结果一致）"""
    probe = MultiLightingDataset(root_dir=root_dir, is_training=False, load_gt=False)
    names = sorted(probe.valid_scenes)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(names), generator=g).tolist()
    n_train = max(int(train_val_split * len(names)), 1)
    train_names = sorted(names[i] for i in perm[:n_train])
    val_names = sorted(names[i] for i in perm[n_train:])
    return train_names, val_names


def get_data_loaders(
        config,
        train_val_split: float = 0.8
    ) -> Tuple[DataLoader, DataLoader]:
    """
    获取训练和验证数据加载器（Phase 1：按场景确定性划分，
    训练/验证各自独立的数据集实例——修复旧版共享实例导致增强失效的问题）
    """
    root_dir = config.data.root_dir
    num_lights = config.data.num_lights
    image_size = config.data.image_size
    batch_size = config.data.batch_size
    num_workers = config.data.num_workers
    pin_memory = config.data.pin_memory
    prefetch_factor = config.data.prefetch_factor
    persistent_workers = config.data.persistent_workers
    file_extension = config.data.file_extension
    seed = getattr(config, "seed", 42)

    train_names, val_names = split_scene_names(root_dir, train_val_split, seed)
    print(f"场景划分: train={len(train_names)}, val={len(val_names)}")

    train_dataset = MultiLightingDataset(
        root_dir=root_dir,
        num_lights=num_lights,
        image_size=image_size,
        is_training=True,
        file_extension=file_extension,
        max_rotation_angle=config.data.max_rotation_angle,
        horizontal_flip_prob=config.data.horizontal_flip_prob,
        scene_subset=train_names
    )
    val_dataset = MultiLightingDataset(
        root_dir=root_dir,
        num_lights=num_lights,
        image_size=image_size,
        is_training=False,
        file_extension=file_extension,
        scene_subset=val_names
    )

    train_loader = create_data_loader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        prefetch_factor=prefetch_factor, persistent_workers=persistent_workers)
    val_loader = create_data_loader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        prefetch_factor=prefetch_factor, persistent_workers=False)

    return train_loader, val_loader


if __name__ == "__main__":
    """测试 MultiLightingDataset 的完整功能"""
    import sys
    from torchvision.utils import save_image

    data_root = sys.argv[1] if len(sys.argv) > 1 else "./data/inverse_rendering"

    print("=" * 80)
    print("MultiLightingDataset 测试（含 GT 通道）")
    print("=" * 80)

    dataset = MultiLightingDataset(
        root_dir=data_root,
        num_lights=5,
        image_size=(128, 128),
        is_training=True
    )

    images, gt, scene_name = dataset[0]
    print(f"\n场景: {scene_name}")
    print("images:", tuple(images.shape), images.dtype,
          "range=[%.3f, %.3f]" % (images.min(), images.max()))
    if gt is not None:
        for k, v in gt.items():
            print(f"gt[{k}]: {tuple(v.shape)} {v.dtype}")
        assert gt["depth"].shape[-2:] == images.shape[-2:], "GT 与图像空间尺寸不一致"
        nmean = gt["normal"].norm(dim=0).mean().item()
        assert abs(nmean - 1.0) < 0.05, f"法线应接近单位范数: {nmean}"
    else:
        print("gt: None（无 GT 的自监督数据）")

    loader = create_data_loader(dataset, batch_size=2, shuffle=True, num_workers=0)
    for batch_idx, (batch_images, batch_gt, batch_scenes) in enumerate(loader):
        print(f"\nBatch {batch_idx}: images {tuple(batch_images.shape)}, scenes {batch_scenes}")
        if batch_idx >= 1:
            break

    save_image(images.unsqueeze(1), "dataset_test_grid.png", nrow=5)
    print("\n测试通过! 可视化已保存至 dataset_test_grid.png")
