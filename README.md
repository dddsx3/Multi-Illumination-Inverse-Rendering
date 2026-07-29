# 多光照图像逆向渲染系统

基于深度学习的逆向渲染系统，从多光照灰度图像中恢复场景的深度图、反照率图、法线图和球谐光照系数。

## 效果演示

以下为模型在测试场景上的推理结果。输入为同一物体在 5 种不同光照条件下的灰度图像，输出为分解后的深度、反照率、法线等内在属性。

### 输入：多光照图像

| Light 1 | Light 2 | Light 3 | Light 4 | Light 5 |
|:-------:|:-------:|:-------:|:-------:|:-------:|
| ![input_00](examples/input_00.png) | ![input_01](examples/input_01.png) | ![input_02](examples/input_02.png) | ![input_03](examples/input_03.png) | ![input_04](examples/input_04.png) |

### 输出：分解结果

| 深度图 | 反照率图 | 光照图 | 法线图 (X/Y/Z) |
|:-----:|:-------:|:-----:|:-------------:|
| ![depth](examples/depth.png) | ![albedo](examples/albedo.png) | ![shading](examples/shading.png) | ![nx](examples/normal_x.png) ![ny](examples/normal_y.png) ![nz](examples/normal_z.png) |

### 输出：渲染重建

| Rendered 1 | Rendered 2 | Rendered 3 | Rendered 4 | Rendered 5 |
|:----------:|:----------:|:----------:|:----------:|:----------:|
| ![r00](examples/rendered_00.png) | ![r01](examples/rendered_01.png) | ![r02](examples/rendered_02.png) | ![r03](examples/rendered_03.png) | ![r04](examples/rendered_04.png) |

### 输出：残差修正

| 全局残差 | 局部残差 | 权重图 |
|:-------:|:-------:|:-----:|
| ![gr](examples/global_residual.png) | ![lr](examples/local_residual.png) | ![wm](examples/weight_map.png) |

## 核心功能

- 从 5 张不同光照条件下的灰度图像分解内在属性
- U-Net 架构进行多输出头预测（深度、反照率、球谐系数、权重图）
- 可微分物理渲染器（深度→法线→球谐光照→渲染图像）
- 层次化残差模块处理非朗伯反射效应
- 三阶段课程学习策略（几何→材质→残差）

## 项目结构

| 文件 | 说明 |
|------|------|
| [main.py](main.py) | 主程序入口（训练/测试/演示） |
| [config.py](config.py) | 配置管理 |
| [unet_model.py](unet_model.py) | U-Net 模型定义 |
| [physics_renderer.py](physics_renderer.py) | 可微分物理渲染器 |
| [residual_modules.py](residual_modules.py) | 层次化残差模块 |
| [loss_functions.py](loss_functions.py) | 损失函数集合 |
| [trainer.py](trainer.py) | 三阶段课程学习训练器 |
| [data_loader.py](data_loader.py) | 多光照数据加载器 |
| [inference.py](inference.py) | 推理脚本 |
| [auto_training_monitor.py](auto_training_monitor.py) | 训练健康监控 |
| [model_diagnostics.py](model_diagnostics.py) | 模型诊断与可视化 |
| [dataset_test.py](dataset_test.py) | 数据集质量检查 |
| [examples/](examples/) | 模型推理效果示例 |
| [逆向渲染项目操作手册.md](逆向渲染项目操作手册.md) | 详细操作手册 |
| [项目交接文档.md](项目交接文档.md) | 项目交接文档 |

## 环境要求

- Python 3.8+
- PyTorch 1.8+ (CUDA)
- torchvision, numpy, Pillow, tqdm
- matplotlib (可选，用于可视化)

## 快速开始

### 训练

```bash
python main.py --mode train
```

在 `main.py` 中修改 `data_root` 指向你的数据集路径。

### 推理

```bash
python inference.py
```

编辑 `inference.py` 中的 `checkpoint_path` 和 `image_folder` 参数。

### 数据格式

```
data_root/
└── rgb/
    ├── scene_000000/
    │   ├── light_001.png
    │   ├── light_002.png
    │   ├── light_003.png
    │   ├── light_004.png
    │   └── light_005.png
    └── ...
```

每个场景包含 5 张灰度光照图像，分辨率 256×256。

## 模型架构

```
输入: 5张多光照灰度图像 [B, 5, H, W]
  ↓
[IntrinsicUNet] → 深度 [B,1,H,W] + 反照率 [B,1,H,W] + 球谐系数 [B,5,9] + 权重图 [B,1,H,W]
  ↓
[PhysicsRenderer] → 深度→法线(Sobel) → 球谐光照计算 → 渲染图像
  ↓
[HierarchicalResidual] → 非朗伯效应修正
  ↓
输出: 分解后的内在属性 + 重建图像
```

## 三阶段课程学习

| 阶段 | 名称 | 重点 | 默认轮数 |
|------|------|------|----------|
| Stage 1 | 几何学习 | 深度、光照分离，Albedo 平滑 | 30 |
| Stage 2 | 材质学习 | 反照率、权重正则化 | 30 |
| Stage 3 | 残差学习 | 非朗伯效应建模 | 后续 |

## 参考文献

- **U-Net**: Ronneberger O, Fischer P, Brox T. *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI 2015. [[arXiv:1505.04597](https://arxiv.org/abs/1505.04597)]
- **球谐光照**: Ramamoorthi R, Hanrahan P. *An Efficient Representation for Irradiance Environment Maps*. SIGGRAPH 2001. [[DOI](https://doi.org/10.1145/383259.383266)]
- **内在图像分解**: Bell S, Bala K, Snavely N. *Intrinsic Images in the Wild*. ACM TOG 2014. [[DOI](https://doi.org/10.1145/2601097.2601206)]
- **深度引导本征分解**: Narihira T, Maire M, Yu S X. *Direct Intrinsics: Learning Albedo-Shading Decomposition by Convolutional Neural Networks*. ICCV 2015. [[arXiv:1512.02311](https://arxiv.org/abs/1512.02311)]
- **可微分渲染**: Li T M, Aittala M, Durand F, Lehtinen J. *Differentiable Monte Carlo Ray Tracing through Edge Sampling*. ACM TOG 2018. [[DOI](https://doi.org/10.1145/3272127.3275109)]

## 更多信息

详见 [逆向渲染项目操作手册](逆向渲染项目操作手册.md) 和 [项目交接文档](项目交接文档.md)。