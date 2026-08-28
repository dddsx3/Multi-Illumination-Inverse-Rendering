# DATASET_CONTRACT · synthetic_v3 数据合同（PRE-0 v1.0）

> 本文档逐项解释 synthetic_v3 每个量的**物理意义**（不只是 tensor shape）。
> 依据：`render_dataset.py`（生成端）、`physics_renderer.py`（训练渲染端）、
> `data_loader.py`（加载端）、`config.py`、`evaluate.py`（评估端）逐行核实。
> 数值约定速查见 `pre0_protocol.yaml`。

## 0. 场景是什么

每个"场景"= 一个 Objaverse 3D 模型（可能含多部件 mesh），经联合归一化
（最长边缩放到 1.6 世界单位、几何中心移到世界原点）后，由一台固定相机
（+X 方向仰角 30°、距离 2.6、50° FOV、look-at 原点）拍摄。
5 张图 = 同一机位、同一几何，仅改变一盏点光源的位置（方位角 72° 步进、
仰角 50°、半径 2.99）。世界背景纯黑。**相机与几何跨全部 620 场景完全一致，
光照方向跨场景也完全一致**——数据集变化的只有物体形状与材质。

## 1. 图像 `light_001..005.png`（模型唯一输入）

- **物理意义**：当前点光下物体成像的**亮度（luma）通道**。成像过程是
  Blender Cycles 路径追踪：近场点光源直接照明 + 物体间/自身间接反弹 +
  几何阴影，然后 RGB 经标准 sRGB OETF 编码，取 BT.709 luma（**在编码域
  取亮度**）存 uint8 灰度。
- **编码域陷阱**：磁盘值不是线性辐亮度。训练端用 `^(1/2.2)` 近似解码
  （非精确 sRGB 反变换，暗部偏差 ~2%）。所有与线性域 GT（albedo、SH）
  的混算都在"近似解码域"进行，这一近似本身计入 renderer mismatch。
- **噪声**：Cycles 128 samples 的路径追踪噪声是数据集自带的随机噪声，
  也是 PRE-02 实验 4（novel vs duplicate）中 duplicate 不带来信息的
  物理上限来源之外的唯一随机源。
- shape `[H,W]` uint8；分辨率 256×256；无 alpha。

## 2. 深度 `depth.npy` [1,H,W] float32

- **物理意义**：视空间 z——沿该像素视线方向，相机到物体表面的距离。
  近处值小、远处值大（"近大远小"指遮挡关系，数值上是"近小远大"，
  原注释"近大远小正数"指视空间 z 语义）。z 轴指向远离相机。
- **单位**：世界单位。场景已归一化到最长边 1.6，故前景深度跨度
  通常在 0.5~1.5 之间（逐场景见 render_stats.txt）。
- **归一化**：不归一化，原始 float32。训练 loss 直接作用于此量纲。
- 背景像素深度无意义，一切深度评估以 mask 为准。

## 3. 法线 `normal.npy` [3,H,W] float32

- **物理意义**：表面单位法线在**相机空间**的方向，面朝相机（z 分量>0）。
- **定义来源（关键）**：它不是 mesh 法线！由 `depth.npy` 经无边缘感知
  Sobel 导数计算 `n = normalize([-∂z/∂x, -∂z/∂y, 1])` 得到
  （`render_dataset.sobel_normal`，生成时做过与深度导数的自检，夹角恒 0）。
  因此：
  - "render normal vs depth-derived normal" 对照在本数据集内**按构造恒等**，
    PRE-01 实验 B 无法做真实对照，必须如实报告；
  - 真实成像（Cycles 平滑着色）用的是网格法线，与本定义在曲面上和
    深度不连续处不同——这是 renderer mismatch 的一部分，由 PRE-01
    oracle 残差间接量化；
  - 训练渲染器默认 `use_edge_aware=True`，在深度不连续处会进一步偏离
    本 GT 定义。PRE-0 probe 统一 `use_edge_aware=False` 对齐数据定义。
- 通道序：`[x, y, z]`，x 向右、y 向下、z 远离相机。

## 4. 反照率 `albedo.npy` [1,H,W] float32

- **物理意义**：无光照 base color 的**亮度标量**。生成端取 Cycles
  diffuse pass 的线性 RGB，按 BT.709 luma（**线性域加权**）合成单通道，
  裁剪到 [0,1]。注意与图像的 luma 不同：图像是 sRGB 编码域取 luma，
  反照率是线性域取 luma——两者对彩色物体不等价（饱和场景可差 ~10 灰阶）。
- **值域**：[0,1]，线性（无 gamma）。
- **歧义**：ρ 与光照能量存在全局乘积歧义，凡反照率评估一律用 SI-MAE
  （逐场景全局标量 s = argmin ||s·pred − gt||²，最小二乘闭式解，
  定义同 `evaluate.albedo_metrics`）。禁止用非尺度不变指标评估反照率。
- **信息损失声明**：彩色反照率三通道不落盘，本项目所有"albedo"均指
  亮度标量反照率。

## 5. 光照 `sh_coeffs.npy` [K,9] float32

- **物理意义**：每盏光的**远场方向光近似**系数。真实光源是半径 2.99 处
  的 100W 点光；落盘系数 `c = I_eff · Y(d)`，`I_eff = 100/(4π·2.99²) ≈ 0.890`，
  `d` 为光源指向场景中心的单位方向（相机空间）。
- **基函数**：二阶实 SH，9 系数，常数与顺序与
  `physics_renderer.SphericalHarmonicsLighting` 完全一致
  （见 protocol yaml `lighting.basis_order/basis_constants`）。
- **语义边界**：sh_coeffs 描述的是"方向光近似"，不是 Cycles 实际入射光。
  近场效应（物体尺寸 ~0.8 vs 光距 2.99，跨物体面元距离差 ~27%）、
  阴影、间接光都不在系数里。**网络预测的 lighting 的 GT 就是这组系数**，
  其与真实成像的偏差由 PRE-01 oracle 实验量化，属协议内已知误差源。
- **shading 合成**：`s(n,c) = ReLU(Σ c_i Y_i(n))`——ReLU 会把阴影区截为 0，
  这是训练渲染器的定义，不是物理阴影。

## 6. 掩码 `mask.npy` [1,H,W] uint8

- **物理意义**：前景（物体）像素 = 1，背景 = 0（instance_segmaps≠0）。
  覆盖率强制在 [0.05, 0.98]（生成时校验）。
- 所有指标（PSNR/MAE/SSIM/SI-MAE/角度误差）只在 mask 内计算。

## 7. Split 合同

- `train 447 / val 49 / test 124`，冻结于 `pre0/protocol/split_manifest.json`，
  逐份 scene ID 列表的 SHA256 见该文件 `split_hash_sha256`
  （全量 hash `98e91fc5…e32`）。
- test 集来自旧版 80/20(seed=42) 划分的验证子集，从未参与任何基线训练。
- **PRE-0 纪律**：评估只在 test/val 上报告；PRE-02 的解析补光（N>5）
  不改变 scene split；任何以后的重划分都必须产生新的 dataset_id 与 hash。

## 8. 已钉死的语义冲突清单（PRE-01 审计对象）

| # | 冲突 | 影响面 |
|---|---|---|
| 1 | 近场点光 vs 远场方向光 SH 近似 | 光照幅度/方向、阴影形状 |
| 2 | Cycles 间接光 vs 单次直接漫反射 | 暗部残差、凹陷区 |
| 3 | sRGB 编码域 luma 图像 + ^(1/2.2) 近似解码 vs 线性域乘法 | 全图系统性非线性 |
| 4 | 线性域 luma 反照率 vs 编码域 luma 图像（彩色物体） | 彩色/饱和物体 |
| 5 | Cycles 平滑法线成像 vs Sobel 深度导数法线 GT | 曲面高频、不连续处 |
| 6 | 训练渲染器 edge-aware=True 默认 vs 数据 edge-aware=False | 深度不连续处 |
| 7 | ReLU 截断 vs 物理阴影 | 阴影边界 |

每一项都不允许"未知"状态存在：PRE-01 必须给出量化残差与空间分布证据。
