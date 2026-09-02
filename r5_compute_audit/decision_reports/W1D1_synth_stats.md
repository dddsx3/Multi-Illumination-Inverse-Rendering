# W1-D1 Stage 1 · 合成图低层统计画像

## 数据规模
- scene: 3 × light: 5 = 15 张图

## 聚合 (跨 scene × light)

| 指标 | 均值 | 备注 |
|---|---:|---|
| 平均亮度 | 0.457 | 0=black, 1=white |
| 高光像素占比 (R,G,B 都 >0.95) | **0.0757** | Lambertian 0; 含高光 0.001-0.01 |
| 梯度直方图熵 | 1.41 | 越高=越复杂 |
| 频谱平滑度 (低频/高频) | 120766.38 | 越高=越平滑 |

## 解读 (B 轨任务书 §B.2)

**KL 检验 (与 DiLiGenT 对比) 尚未做**: 本仓库无 DiLiGenT 原始图。
**stage 1 自身画像**:
- 平均高光占比 0.0757 → 合成图有非平凡高光
- 需要查 albedo.npy 数据, 确认是否泄漏了 specular 分量

## 下一步

- **W1-D1 stage 2**: 下载 DiLiGenT 基准 (Calibrated photometric stereo, 10 objects, 96 lights each)
  URL: https://sites.google.com/site/photometricstereodata/single-object
  路径: `pre0/raw_data/diligent/`
- 跑同 4 个指标, 计算 KL(合成 || DiLiGenT)
- 三项 KL 全 < 0.1 → 域差假设**死**, 转查架构容量 (B→C)
- 任何 KL > 0.5 → 域差假设**获强支持**, 推进 B0 协议
