# 逆渲染项目 Phase 0 完成报告

**项目**：Multi-Illumination Inverse Rendering（多光照逆渲染）
**阶段**：Phase 0 —— 代码修复与工程化底座
**日期**：2026-08-23
**交付物**：`phase0_changes.patch`（共 9 个文件，+555/-446 行）+ 本报告

---

## 1. 阶段目标

Phase 0 是升级路线图的清理阶段，解决代码级核实报告中发现的 4 个计划外关键问题中的 3 个（L1、L3、L6），并补齐工程化底座（L5 的 requirements.txt 部分）。不改变模型能力，只修复设计缺陷、激活死代码路径、建立可量化的评估契约。

| 编号 | 计划内 | 内容 | 状态 |
|------|--------|------|------|
| L1 | — | 重构全局共享的"作弊"局部残差参数为逐场景、逐光照预测网络 | ✅ 完成 |
| L3 | — | 清理从未生效的反照率一致性损失（no-op）+ 代码卫生 | ✅ 完成 |
| L5 | — | 可复现性：requirements.txt（checkpoint 对齐留待 GPU 验证） | ✅ 完成 |
| L6 | — | 新建 evaluate.py 标准评估指标模块 + CLI | ✅ 完成 |
| 附加 | — | 修复 3 个潜在训练崩溃/错误路径 bug（见 §3） | ✅ 完成 |

---

## 2. 改动总览

### 2.1 L1：局部残差重构（residual_modules.py，核心改动）

**问题**：旧实现 `create_local_residual` 是**全数据集共享**的单个可学习参数张量 `[1,1,H,W]`：

- 与输入完全无关——网络在训练集上过拟合，推理新场景时输出固定的数据集级系统误差；
- 所有 K 张光照图共用同一张残差图——无法表达随光照变化的非朗伯效应（镜面高光）；
- 三阶段课程学习中阶段 1/2 该参数**冻结**，到阶段 3 才解冻，但 trainer 从未真正把阶段信息传进残差模块（见 §3.2），且即使正确传入，一个冻结的全局缓冲也没有任何可解释性。

**新设计** `LocalResidualNet`：

```
输入: [features(解码器末层特征 C=32) | albedo(1) | shading(K)]  →  concat
层1: Conv2d(C+1+K, 64, 3×3, pad=1) + ReLU
层2: Conv2d(64, K, 1×1)  ← 权重与偏置零初始化
输出: [B, K, H, W] 逐光照残差图
```

- **逐场景**：以解码器特征为条件，新场景自动获得不同的残差估计；
- **逐光照**：以逐光照 shading 为条件，可建模镜面高光等光照相关效应；
- **零初始化末层**：训练初期残差恒为 0，渲染退化为纯 Lambertian，配合课程学习平滑引入——与阶段 3 缩放因子 1.0 无缝衔接；
- `HierarchicalResidual` 构造函数同步重构：删除 `image_height/image_width` 参数（旧作弊缓冲的遗留），`residual_scales` 默认值改为 `{stage1: 0.0, stage2: 0.0, stage3: 1.0}`；`features=None` 时自动关闭局部残差（向后兼容）。

### 2.2 L3：清理无效损失（loss_functions.py）

- `AlbedoConsistencyLoss`：原实现是**恒返回 0 的 no-op**——反照率只有 1 个通道，而该损失需要每光照独立的反照率图。其存在意义已由架构保证（多图共享反照率头即一致性）。docstring 重写说明原因，为 A2 阶段引入每光照反照率时的复活路径留了钩子；
- 从 `LossCalculator` 中移除其实例化、forward 计算块、3 个阶段权重字典中的 `albedo_consistency` 条目及加权求和项；
- 修正 `LightingPriorLoss` 自测的返回值解包 bug（返回 3 个值，旧代码解包 2 个）。

### 2.3 L5：requirements.txt（新增）

按全仓库实际 import 声明：`torch>=2.1, torchvision>=0.16, numpy>=1.24, Pillow>=10.0, tqdm>=4.60, matplotlib>=3.7, tensorboard>=2.14`。

### 2.4 L6：evaluate.py 标准评估模块（新增，293 行）

发表级论文的标准指标契约，同时作为 Phase 1 合成数据集的**数据协议**（场景目录约定：`light_001..K.png`、`depth.npy`、`albedo.npy`、`normal.npy`、`mask.npy`、`sh_coeffs.npy`）：

| 函数 | 指标 | 对应论文惯例 |
|------|------|--------------|
| `normal_metrics` | MAE°、中位°、acc@11.25°/22.5°/30°（可选符号翻转容差） | DiLiGenT 光度立体基准 |
| `depth_metrics` | RMSE、MAE、**尺度不变 RMSE**（log 空间去均值） | Eigen et al., CVPR 2014 |
| `albedo_metrics` | MSE、MAE、**尺度不变 MAE**（最小二乘尺度估计） | 反照率-光照乘积歧义 |
| `recon_metrics` | PSNR、SSIM（内置 11×11 高斯窗，零外部依赖） | 重建质量 |

- 统一输入约定 `[B,C,H,W]`，同时接受 numpy/torch、2D/3D/4D；
- 全指标支持 `mask` 有效像素掩码；
- 多光照图像逐光照计算后平均；
- CLI：`python evaluate.py --pred x.npy --gt y.npy --kind normal [--mask m.npy]`；
- `compute_all` 一键汇总全部指标（13 项，键带前缀）。

### 2.5 调用链适配（main.py / inference.py / model_diagnostics.py / unet_model.py）

- `IntrinsicUNet.forward` 返回 5 元组（新增解码器末层特征图 `features [B,C,H,W]`，供局部残差网络使用）；
- 三个调用方（训练主程序、推理、诊断脚本）同步解包 5 值并把 `features` 传入残差模块；
- 两处 `HierarchicalResidual` 构造同步适配新签名。

---

## 3. 顺带修复的 3 个潜在 bug（核实报告未覆盖）

1. **trainer.py `self.weights` AttributeError**：`__init__` 中从未定义 `self.weights`，但两处（~400/402 行）读取 `self.weights.get(...)`——只要训练开启残差相关损失（阶段 3 必然触发）就会崩溃。改为 `self.loss_calculator.weights`。
2. **残差从未收到阶段信息**：trainer 调用残差模块时未传 `stage`，导致阶段 1/2（残差应关闭、权重冻结）实际使用默认 `stage3` 的缩放 0.5（旧默认值）——冻结参数在输出中制造随机噪声，且梯度路径被激活。现在显式传 `stage=f'stage{self.current_stage}'`。
3. **`_initialize_weights` 覆盖零初始化**：trainer 初始化后统一调用权重初始化，会把 `LocalResidualNet` 末层的零初始化覆盖为 kaiming——初始残差不再为 0，破坏课程学习设计。修复：初始化后再执行一次零化。

---

## 4. 验证结果（全部通过）

| 验证项 | 结果 |
|--------|------|
| 4 个核心模块自测（unet / physics_renderer / residual / loss） | ✅ 测试完成/全部通过 |
| evaluate.py 数值测试：法线 10° 旋转 → 10.000°；si-RMSE 对尺度+偏移不变；si-MAE 对 0.7× 尺度不变；SSIM 完美=1.0、加噪下降 | ✅ 7 组全部通过 |
| evaluate.py CLI（基础 / 掩码 / depth 三种调用） | ✅ exit 0 |
| 端到端 CPU 冒烟（B=2, K=5, 128²）：model→renderer→residual→loss→backward | ✅ 梯度流到 model 与 residual |
| stage1 兼容路径（残差关闭 = 纯 Lambertian，冻结参数无梯度） | ✅ |
| 全部 11 个模块导入（含 tensorboard 依赖） | ✅ |
| patch 应用验证：干净 HEAD 工作树 `git apply --check` + 应用后 9 文件与工作区**逐字节一致**（MD5 校验） | ✅ |

---

## 5. 兼容性警告（重要）

**旧 checkpoint 无法加载**。新 `HierarchicalResidual` 的 state_dict 键与旧模型完全不同（删除了共享缓冲参数 `local_residual`，新增 `local_net.net.*`）。且旧模型是用作弊参数训练的，即使勉强加载也没有意义。**必须从零重新训练**——这本来就是 Phase 1 的计划（旧训练未在含 GT 的数据集上量化过）。

如果你的推理脚本加载旧权重，请确认 checkpoint 是 `--resume` 全量训练续点（也会因 state_dict 键不匹配而失败），正确处理方式：丢弃旧权重，用本 patch 后的代码重新训练。

---

## 6. 在你自己机器上应用补丁

```bash
cd Multi-Illumination-Inverse-Rendering
git apply phase0_changes.patch
# 或更安全：git apply --check phase0_changes.patch 先检查，然后 git apply
python -c "from loss_functions import LossCalculator; from residual_modules import HierarchicalResidual; print('OK')"
```

补丁由 `git diff`（含 `git add -N`）生成，9 个文件：7 修改 + 2 新增（evaluate.py、requirements.txt）。行尾与仓库一致的 CRLF，可干净应用于 HEAD（5a63e93）。

---

## 7. 下一步：Phase 1（B1 合成数据集）——需要 GPU 的命令

### 7.1 合成数据渲染（你的 Windows GPU 机器）

**步骤 1：安装 BlenderProc**

```bash
pip install blenderproc
blenderproc --version   # 首次运行会自动下载 Blender（约 200MB）
```

**步骤 2：下载 ShapeNet 子集**（建议先用 50-100 个模型跑通流程，全量约 5 万模型可选）

```bash
# 方式一：ShapeNet 官网账号下载（v1 约 3.5GB）
# 方式二（推荐先跑通）：任意公开 OBJ 模型集，或 https://huggingface.co/datasets/shapenet/shapenet 镜像
# 说明：BlenderProc 支持 .obj/.glb/.fbx 等格式，不强制 ShapeNet 本体
```

**步骤 3：渲染脚本**（存为 `render_dataset.py`，先跑小规模验证协议）

```python
import blenderproc as bproc
import numpy as np, os, sys, math

# ---- 场景协议：每场景 5 张 256x256 灰度图 + depth/albedo/normal/mask/sh_coeffs ----
OUT = sys.argv[1] if len(sys.argv) > 1 else "synthetic_data"
os.makedirs(OUT, exist_ok=True)

bproc.init()
for obj_path in sys.argv[2:]:           # 传入模型路径列表
    scene_name = os.path.splitext(os.path.basename(obj_path))[0]
    scene_dir = os.path.join(OUT, scene_name)
    if os.path.exists(scene_dir): continue
    os.makedirs(scene_dir, exist_ok=True)

    obj = bproc.loader.load_obj(obj_path)[0]
    bproc.object.manifold_cleanup(obj)
    obj.set_scale([0.2] * 3)            # 归一化到相机视野

    # 相机：前视图 + 轻微俯仰
    bproc.camera.set_resolution(256, 256)
    bproc.camera.add_camera_pose(bproc.math.build_transformation_mat(
        [-1.2, 0.5, 1.2], [math.radians(-30), math.radians(-15), 0]))

    # 5 个方向光：固定方位角变化 + 环境光
    bproc.lighting.set_lighting_mode_nondeterministic(False)   # 固定随机种子
    bproc.world.set_world_background(np.zeros(3))              # 纯黑背景
    for k in range(5):
        az = k * 72.0
        d = [-math.cos(math.radians(az)), math.sin(math.radians(az)), 1.0]
        light = bproc.types.Light()
        light.set_type("POINT"); light.set_location([1.5*d[0], 1.5*d[1], 2.5*d[2]])
        light.set_energy(200)

    # 渲染 G-buffer：灰度图 + depth + normal + albedo + mask
    data = bproc.renderer.render()
    bproc.renderer.enable_depth_output(True)
    bproc.renderer.enable_normals_output(True)
    bproc.renderer.enable_diffuse_color_output(True)
    bproc.renderer.enable_segmentation_output(True)

    # 按协议写文件（light_001..005.png 灰度、depth/albedo/normal/mask.npy、sh_coeffs.npy）
    # —— 具体写入逻辑见 7.2 节说明，本脚本骨架需按协议补齐
    print(f"[done] {scene_name}")
```

**说明**：渲染脚本骨架我留在 Phase 1 正式开工时交付完整版（含 SH 系数解析计算与协议文件写入），上面是让你先验证 BlenderProc 在本机能跑通的最小路径。**建议先渲染 20 个场景，把输出喂给 evaluate.py 自检**：

```bash
python evaluate.py --pred scene0/pred.npy --gt scene0/depth.npy --kind depth
```

### 7.2 训练（合成数据就绪后）

```bash
# 三阶段课程学习（默认 stage1=30, stage2=30, stage3=40 轮）
python main.py --mode train \
  --data_root C:/path/to/synthetic_data \
  --num_lights 5 --image_size 256 256 \
  --batch_size 8 --total_epochs 100 \
  --stage1_epochs 30 --stage2_epochs 30 \
  --base_channels 32 --learning_rate 1e-4 \
  --use_amp

# 断点续训
python main.py --mode train --resume --checkpoint checkpoints/latest.pth ...
```

**注意**：`trainer.py` 的 `self.weights` 崩溃 bug 已修复，现在阶段 3（残差开启）可以正常运行；本地残差初始为零，阶段 1/2 行为与旧版一致（纯 Lambertian 自监督）。建议先跑 `--total_epochs 3`（每阶段 1 轮）验证三阶段切换逻辑，再开全量。

### 7.3 验证基线（Phase 1 收尾）

- 在 BlenderProc 测试集上跑 `evaluate.py` 全套 13 项指标，写入实验日志；
- 再按核实报告 B2 修正版：复现 **DiLiGenT**（光度立体标准基准，10 个物体 × 96 光照，官网公开下载）1-2 个对比基线，而不是原计划盲复现 6 个。

---

## 8. Phase 0 遗留事项

| 事项 | 归属 | 说明 |
|------|------|------|
| examples/ 与 HEAD 不一致 | L5 剩余 | 需要 GPU checkpoint 才能对拍，Phase 1 训练后处理 |
| 每光照反照率一致性损失复活 | A2 | 等模型升级到每光照反照率输出后启用（已留钩子） |
| 新 checkpoint 格式的 inference 验证 | Phase 1 | 训练出第一个 checkpoint 后跑一遍 inference.py 全链路 |

---

*本报告与 `phase0_changes.patch` 配套使用。下一阶段工作：Phase 1（B1 合成数据集 + 首次量化训练），详见升级计划文档。*
