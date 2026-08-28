# DILIGENT_CONTRACT · DiLiGenT 真实基准评估合同（PRE-0 v1.0）

> 本阶段只建评估器与合同，不追求成绩。数据：`D:\data\DiLiGenT\pmsData`
> （10 物体 × 96 光，512×612 RGB PNG；`light_directions.txt` [96,3] 单位向量、
> `light_intensities.txt` [96,3]、`Normal_gt.mat` [512,612,3] 相机系 GT 法线）。

## 1. 输入合同（模型能看什么）

| 量 | 是否给模型 | 说明 |
|---|---|---|
| N 张图像 | ✅ | RGB PNG → 线性域解码（精确 sRGB 反变换）→ BT.709 luma 灰度。与 synthetic_v3 的灰度生成语义一致（编码域 luma 的差异见 §5 偏差声明） |
| 光照方向/强度 | ❌ | 主协议为 uncalibrated：模型不得看到 `light_directions.txt`/`light_intensities.txt` |
| GT 法线 / mask | ❌ | 只允许 evaluator 使用 |
| 物体身份 | ✅ | 模型可按物体分别推理（每物体一个 scene 等价物） |

## 2. GT 合同（evaluator 专用）

- 法线 GT：`Normal_gt.mat`（相机系，单位向量，面朝相机 z>0，与 synthetic normal 语义一致）。
- mask：本数据拷贝缺少官方 `mask.png`，暂用 `|Normal_gt| > 0` 的像素集作为
  代理掩码，并在所有结果中标注 "mask=gt-normal-proxy"。
  ⚠ 数据补全项：正式评估前必须下载官方 mask，替换代理（不允许用代理发论文数字）。
- 图像域：评估在图像的线性域进行；报告任何与模型输出域相关的换算。

## 3. Subset protocol（防"各自随机抽图"）

- 每个 N 使用固定种子生成的 K=5 个子集：S_{N,k}，k=1..5，
  N ∈ {3, 5, 10, 25, 50}；N=96 为全集。
- 生成规则：`rng(seed=20260829, object, N, k)` 从 96 光索引无放回抽样，
  种子与生成代码落盘于 `pre0/benchmark/diligent_subsets.json`，
  **所有模型必须使用完全相同的 subsets 文件**。
- 报告：每物体 × 每 N 的均值±标准差（跨 k=5 子集），禁止单子集择优。

## 4. 指标

- 角误差 MAE / median / P90（度，掩码内）。
- 标准 goodness：角度 < 11.25° 的像素百分比（DiLiGenT 惯例）。
- matched-N 比较：同 N 同子集下跨方法比较；禁止把不同 N 的结果混入同一均值。

## 5. 已知协议偏差声明

1. DiLiGenT 图像为彩色（light_intensities 为逐通道 RGB）；本项目模型为灰度。
   灰度化在**线性域 BT.709 luma** 进行，与 synthetic GT albedo 的合成方式一致；
   但彩色信息丢失对彩色物体（如 harvest）的 PS 求解是已知不利因素，如实报告。
2. 本拷贝无官方 mask（见 §2）。
3. PRE-0 只做 zero-shot 评估（不在 DiLiGenT 上训练/调参）；
   任何未来 finetune 必须更换 dataset_id 并重申合同。

## 6. 与主协议（synthetic_v3）的对应关系

| synthetic_v3 | DiLiGenT |
|---|---|
| scene | object |
| 5 光/场景（固定环） | 96 光/物体（弧形轨迹） |
| SH-2 GT 光照 | 方向光 GT（方向+RGB 强度） |
| depth/normal/albedo GT | 仅 normal GT（无 per-pixel depth/albedo GT） |
| 256² 灰度 | 512×612 灰度（luma） |

DiLiGenT 上能评估的输出只有 **normal**；albedo/lighting 无法与 GT 直接对比
（该数据集不提供），在结果表中明确标注 "normal-only"。
