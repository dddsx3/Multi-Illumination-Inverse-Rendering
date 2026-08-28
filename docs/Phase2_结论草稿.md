# Phase 2 结论草稿（2026-08-28 后期，T-MATRIX 起步）

> **目的**：本稿是 T-MATRIX 任务的"骨架"——等所有 T-ARM 变体（physcon / resC / noFiLM / lowSmooth）训完 + A3-bis 3-seed 综合版完成后，扩展为论文 Results/Discussion 章节。
> 当前覆盖：v2 best 已交付 + INC-0012 物理约束修复 + N 敏感性双轨 + 文档链注。
> **未覆盖部分**（标注"待补"）：5 臂消融矩阵完整 6 变体 × 13 项 + 3-seed mean ± std。

---

## 1. Phase 2 主要成就

### 1.1 T2.2 架构升级（光照数量无关的注意力融合）
- 核心创新落地：FusionUNet 引入 SetTransformerLite + FiLM + ΔA 残差模块
- **置换不变性测试 PASS**（`tests/test_permutation_invariance.py`）：max_diff=3.34e-06 < 1e-5
- N 敏感性双轨（合成 v3 + DiLiGenT）：架构对 N 极其鲁棒（相对变化 < 1%）
- **N_min = 1**（实测双轨 N=1 不退化，论文可维持"任意 N"措辞）

### 1.2 T2.3 物理约束补建（INC-0012）
- albedo 头 Sigmoid + depth 头 Softplus 落地（key 兼容版，旧 checkpoint 严格 load OK）
- 评估脚本物理断言：albedo∈[0,1] / depth>0 违规像素占比统计入库
- 二次风险验证：albedo_smooth=10.0 **未触发 0.5 退化**（v2 warm-start + 3 epoch 冒烟，violation 0%）
- **维持 albedo_smooth=10.0 默认**（不需下调）

### 1.3 T2.5 消融与判别实验
- F-albOff 完成 100 epoch（PSNR 35.69）—— ΔA 价值验证
- F-resA 完成 100 epoch（PSNR 36.54）—— 残差价值验证
- F-resC / F-physcon / F-noFiLM / F-lowSmooth **待训**（A3-bis 完成后启动）

### 1.4 文档链注（INC-0011 / INC-0012 / v3 链注段）
- INC-0011（F-resA 单 run 反转方法论沉淀）已创建
- INC-0012（物理约束补建 + 二次风险验证）已 PASS
- INC-0010 §8 v3 链注段已加（不修改 §1-§7 已结案内容）
- t2_2_design.md §10 物理约束补建加注已完成
- t2_5_n_sensitivity_report.md N 曲线解读文档已完成
- 决策 1-5 全部落地（5 项决策 0 失效）

---

## 2. Phase 2 关键数字（v2 best + INC-0012）

### 2.1 13+2 项指标（v2 best, A6 表扩展）

| 指标 | v2 best | INC-0012 冒烟 | 趋势 |
|---|---|---|---|
| image_psnr ↑ | 37.25 dB | – | ✓ G2.2 放行 |
| normal_mae_deg ↓ | 8.18° | – | ✓ |
| albedo_si_mae ↓ | 0.0532 | 0.0532（N=5，synth v3）| ✓ 不退 |
| depth_rmse ↓ | 0.3554 | 0.3554（N=5）| ✓ |
| **albedo_violation_ratio** ↓ | – | **0.0000%** | ✅ INC-0012 物理约束完全生效 |
| **depth_violation_ratio** ↓ | – | **0.0000%** | ✅ INC-0012 物理约束完全生效 |

**扩展 2 项**：原 13 项 + albedo/depth violation ratio（INC-0012）= 15 项。

### 2.2 N 敏感性（Phase 2 T2.5 核心证据）

| N | synth v3 normal_mae° | DiLiGenT MAE° |
|---|---|---|
| 1 | 10.372 | 39.88 |
| 3 | 10.342 | 39.67 |
| 5 | 10.347 | 39.67 |
| 10 | – | 39.61 |
| 15 | – | 39.56 |
| **极差** | **0.030°（< 0.3%）** | **0.33°（< 1%）** |

**结论**：架构在 N 维度上同时避免了"信息不足"和"聚合器缺陷"——曲线近乎平坦。

### 2.3 DiLiGenT zero-shot 对比

| 模型 | N=5 MAE° | 备注 |
|---|---|---|
| Phase 1 R0 v3gray | 40.39° | T1.7 历史基线 |
| **Phase 2 v2 best** | **39.67°** | 本次 N 曲线（N=5 随机子集）|
| 改善 | -0.72° | G2.2 门禁"不降即放行"达标 |

---

## 3. 反照率退化问题的处置（中期审计 v2 §2-P2 决策 1 重定义）

### 3.1 根因（决策 1 重定义版）
- **不是"重构时丢失 Phase 0 Sigmoid"**——Phase 0 `unet_model.py:226` 注释显式记录已"移除 Sigmoid"
- **真实根因**：
  - `fusion_unet.py` 主干 head 层无显式激活约束（沿袭 Phase 0 模式）
  - `albedo_smooth=10.0` 强平滑项把无约束输出压向常数（同时满足平滑 + 重建）
  - 评估脚本无物理断言，违规未在入库即查

### 3.2 修复路径
1. **补建新约束**（不是"恢复丢失"）：albedo 头 Sigmoid + depth 头 Softplus
2. **验证二次风险**：3 epoch 冒烟 × 2 组（albedo10/albedo1），violation 均 0%
3. **评估脚本物理断言**：albedo∈[0,1] / depth>0 违规占比入库即查
4. **维持 albedo_smooth=10.0**：Sigmoid 头下不再需要靠"压值域"满足约束

### 3.3 反照率退化是否被治愈？
- **当前 v2 best（无 INC-0012 约束）仍存在退化**——13 项 §2 albedo_si_mae=0.0532（N=5，synth v3），
  Phase 1 R0 v3gray 历史 0.055
- **重训 v2 + INC-0012 约束后预期** albedo_si_mae 应进一步降低（"平坦 shading + 平均图像 albedo" 欺骗路径被截断）
- **正式验证**需 A3-bis 3-seed 完成 + 完整重训 v2 + INC-0012 约束启用（计划在 T-ARM 后续阶段）

---

## 4. Phase 2 残余任务（T-MATRIX 后续）

### 4.1 必做（Phase 2 收口前）
- [ ] A3-bis seed 42 完成 100 epoch（续跑中，已到 epoch 42）
- [ ] A3-bis seed 123 / 2024 启动并完成
- [ ] 3-seed 综合 13+2 项指标 mean ± std
- [ ] T2.5 消融补训：F-physcon / F-resC / F-noFiLM / F-lowSmooth
- [ ] 6 变体完整对比矩阵（resA / albOff / v2 / physcon / resC / noFiLM / lowSmooth）
- [ ] 判别实验 (a)(b)(c) 全部完成 → culprit 判定表

### 4.2 必做（论文成稿前）
- [ ] 重训 v2 + INC-0012 物理约束 → 重测 13+2 项 → 验证反照率退化治愈
- [ ] 论文方法节加入 INC-0012 物理约束修复描述（v2.1 §10 约束第 6 条）
- [ ] 论文 N_min 声明（N_min=1 维持"任意 N"措辞）
- [ ] 论文 N 曲线解读框架（"信息不足" vs "聚合器缺陷"双假设判定）
- [ ] 论文图：从 `report_assets/n_curve_*.png` 重渲染（高清版 + 中文标注）

### 4.3 可选（Phase 3 真实世界验证）
- [ ] 真实世界手机实拍数据（10 组场景）
- [ ] DiLiGenT 同协议重训（vs PS-FCN / SDPS-Net / LCNet SOTA 对比）

---

## 5. 门禁对齐状态

| 门禁 | 状态 | 阻塞任务 |
|---|---|---|
| G2.1 训练稳定化 | ✅ 已 PASS | – |
| G2.2 架构升级 | ⚠️ 部分 PASS | 置换测试 + N 冒烟已过；新架构 PSNR ≥ 原架构（T2.2 达成）；RGB 链路通过；参数量表（3,422,829）已记录 |
| G2.3 物理约束重写 | ❌ 未 PASS | T2.3 p2_t23_f_physcon 未训（已建占位目录）|
| G2.4 增强同步 | ✅ 已 PASS | – |
| G2.5 消融矩阵 | ⚠️ 部分 PASS | albOff/resA/v2 已交付；resC/physcon/noFiLM/lowSmooth 未训 |
| G2.6 test 集正式化 | ⚠️ 部分 PASS | test 基线 + 划分文件已就位；`--num_lights` N 子集协议（INC-0012 阶段实现）|
| G2.7 Phase 2 收尾 | ❌ 未 PASS | 等 G2.3 / G2.5 完整 + 3-seed 综合版 |

**Phase 2 放行阻塞**：G2.3（physcon 未训）+ G2.5（4 变体未训）+ G2.7（矩阵不完整）。
预计补训耗时 ~24h（physcon 5.5h + resC 5.5h + noFiLM 5.5h + lowSmooth 5.5h，串行温度节律）。

---

## 6. 引用关系

- 关联文档：
  - `docs/design/t2_2_design.md` §8/§9/§10（架构升级 + 早停 + 物理约束补建）
  - `docs/design/t2_5_n_sensitivity_report.md`（N 敏感性实测报告）
  - `docs/incidents/INC-0010_数学底层三重偏差与编排器续跑冲突.md` §8（v3 链注）
  - `docs/incidents/INC-0011_F-resA单run反转与seed噪声影响.md`（方法论沉淀）
  - `docs/incidents/INC-0012_物理约束补建与albedo_smooth二次风险验证.md`（物理约束修复）
  - `docs/incidents/INC-0010_A6_13项指标_执行版.md` §9（决策 1-5 增补）
- 关联产物：
  - `eval_output/n_curve_synth_v3/n_curve_{raw,agg}.json`（synth v3 N 曲线）
  - `eval_diligent/n_curve/diligent_n_curve.json`（DiLiGenT N 曲线）
  - `report_assets/n_curve_{synth,diligent,combined}.png`（N 曲线图）
  - `_SMOKE_phys_constraints/{albedo10,albedo1}/*.json`（T-PHYS 冒烟产物）

---

*本稿由 2026-08-28 23:00 阶段决策落地，作为 Phase 2 论文 Results/Discussion 骨架。后续 T-ARM 变体训完后扩展为最终版。*
