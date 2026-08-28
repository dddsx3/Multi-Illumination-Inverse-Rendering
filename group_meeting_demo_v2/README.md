# 组会示例：p2_t22_f_n5rgb_v2 推理结果

> **目的**：展示目前效果最好的双模态逆渲染模型（RGB 模态，FusionUNet）
> **模型**：`p2_t22_f_n5rgb_v2`（INC-0010 A2 隔离版，与原污染目录物理分离）
> **数据**：test 集前 5 个场景（`split_manifest` 冻结 split）

---

## 1. 模型定量指标

| 指标 | 值 | 备注 |
|---|---|---|
| 训练轮数 | 100 epoch | best=epoch 96，val=0.0158 |
| **test 集 PSNR** | **37.25 dB** | 全场最佳（resA=36.54, albOff=35.69）|
| test 集 normal MAE | 8.18° | 仅比 resA 略高 0.04° |
| test 集 albedo si-MAE | 0.1304 | 仅比 resA 略高 0.001 |
| test 集 depth RMSE aligned | 0.3324 | 仅比 resA 略高 0.004 |
| 5 场景平均 L1 误差 | 0.0098 | < 1% |

---

## 2. 场景命名

| 目录 | 描述 | 原始 scene_id | L1 |
|---|---|---|---|
| `scene_1_skull/` | **头骨**（objaverse）| 00f7c6d95b7d4c359681cc3b93f8a077 | **0.0050** |
| `scene_2_leaf/` | **叶子**（objaverse，复杂拓扑）| 00ff6c772dce4e84bf66544ada6093a0 | **0.0065** |
| `scene_3/` | test 集第 3 场景 | 0104f4b9bd3c476d949785e03f2ff7c6 | 0.0116 |
| `scene_4/` | test 集第 4 场景 | 0109cde49c664b408e6a95a82d235070 | 0.0129 |
| `scene_5/` | test 集第 5 场景 | 01f2d9c189f5404c97f696d022b0ad81 | 0.0131 |

**组会推荐展示顺序**：`scene_1_skull`（L1 最低）→ `scene_2_leaf`（复杂拓扑）→ `scene_3/4/5`（多样性）

---

## 3. 文件命名约定

每个场景目录下：

| 文件 | 含义 | 物理含义 |
|---|---|---|
| `input_00.png` ~ `input_04.png` | 5 个不同光照下的输入图像（ground truth）| I_k (k=1..5) |
| `rendered_00.png` ~ `rendered_04.png` | 模型对 5 个光照的渲染输出 | \hat{I}_k = A · shading_k |
| `albedo.png` | 反照率（主反照率，跨光照一致）| A |
| `depth.png` | 深度图（相对尺度）| D |
| `normal_rgb.png` | 表面法线（RGB 编码，[-1,1] → [0,1]）| N |
| `normal_x/y/z.png` | 法线的 X/Y/Z 分量 | N_x, N_y, N_z |
| `shading.png` | 着色图（光照相关部分）| S_k |
| `weight_map.png` | 权重图（特征融合权重）| W |

---

## 4. 组会展示建议

**推荐叙事流程**：

1. **先展示输入**：5 张同一物体在不同光照下的 `input_0X.png` —— 强调"光照变化大"
2. **展示模型分解结果**：
   - `albedo.png` —— "**这是不随光照变化的内禀材质**"
   - `depth.png` —— "**这是物体三维形状**"
   - `normal_rgb.png` —— "**这是表面朝向**"
3. **展示重照对比**：`input_0X` vs `rendered_0X` —— "**模型重建的光照图像**"
4. **量化指标**：`scene_1_skull` L1=0.0050（**几乎像素级一致**）

**可选深度对比**（如时间允许）：
- 对比 p2_t25_f_resA (gray) 和 p2_t22_f_n5rgb_v2 (rgb) 的 albedo 质量
- 强调 RGB 模态在双链路下能提供更精细的材质分解

---

## 5. 复跑命令

```bash
cd "D:/Multi-Illumination Inverse Rendering/repo"

python _demo_v2.py \
  --checkpoint "D:/Multi-Illumination Inverse Rendering/checkpoints/p2_t22_f_n5rgb_v2/best_model.pth" \
  --out-dir "D:/Multi-Illumination Inverse Rendering/group_meeting_demo_v2" \
  --num-scenes 5 \
  --modality rgb
```

**前置条件**：
- 训练好的模型 `checkpoints/p2_t22_f_n5rgb_v2/best_model.pth`（PSNR 37.25 dB）
- 数据根目录 `D:/data/synthetic_v3`（620 场景）
- 冻结的 split manifest `repo/splits/synthetic_v3.json`（train 447 / val 49 / test 124）

---

## 6. 推理时间

- 5 场景 × (forward + save 18 张 PNG) ≈ **30 秒**
- GPU 占用 10.6 GB（接近额定 11.94 GB）

---

## 7. 关联文档

- 训练日志：`repo/_arm_p2_t22_f_n5rgb_v2_log.txt`
- Test 评估：`repo/eval_output/p2_t22_f_n5rgb_v2_test/eval_summary.json`
- INC-0010 审计裁决：`repo/docs/incidents/INC-0010_审计裁决_v3_20260828_最终版.md`
- 训练数据 provenance：`checkpoints/p2_t22_f_n5rgb_v2/_DECONTAMINATION_INC0010.md`
