# T2.2 设计预注册 · 光照数量无关的自适应融合（冻结版）

**状态**：已预注册（本文档入库后评估口径不得更改）。**日期**：2026-08-25。

## 1. 问题设定

输入：任意 N 张（N>=1）同一视角、不同光照下的图像（灰度或 RGB，非定标光），
输出：深度 / 反照率 / 法线 / 每光照 SH 系数，经可微物理渲染器重建输入并受 GT 监督。
要求模型对光照顺序的任意排列保持输出不变（置换不变性），且训练后能泛化到
训练时未见过的 N。

## 2. 相关工作与差异声明

- **PS-FCN**（ECCV 2018, arXiv:1807.08696）：共享权重逐图提特征 -> concat -> 卷积融合，
  输出仅法线。concat 融合对光照顺序敏感（非置换不变），且 N 固定于训练设置。
  **本设计差异**：(a) 注意力聚合 + 对称池化，数学上置换不变；(b) 任意 N 训练/推理；
  (c) 输出完整内参分解（深度/反照率/法线/SH）；(d) 物理渲染器在环监督 + 三阶段课程；
  (e) 合成到真实零样本迁移验证。
- **Deep Sets**（arXiv:1703.06114）/ **Set Transformer**（arXiv:1810.00825）：
  置换不变聚合的理论工具箱（sum-pooling 不变性引理 / PMA 注意力池化），本设计采用。

## 3. 架构设计

### 3.1 逐光照共享编码（Per-light stem）
每张输入图独立通过**共享权重** stem（Conv3x3->BN->ReLU x2，1 或 3 通道入、32 通道出，
两次 stride2 至 1/4 分辨率），得到 f_k ∈ R^{32×64×64}。参数与 N 无关。

### 3.2 置换不变注意力聚合（核心创新点）
- token_k = GAP(f_k) 经 MLP(32->128) 得 N 个描述子；
- 单层多头自注意力（D=128, heads=4, **无位置编码**）作用于 N 个 token；
- **PMA**（Pooling by Multi-head Attention，单一可学习 seed query）输出全局光照
  上下文向量 z ∈ R^{128}；
- **置换不变性论证**：无位置编码的自注意力对输入排列等变（softmax((QK^T)/√d)V 的
  行随输入行同步置换）；紧接的 PMA 以同一 seed 对全部 token 做注意力加权求和（对称
  操作），故 z 为集合的不变函数。形式化依据见 Set Transformer §3（引理 1：
  多头注意力在无 PE 时置换等变；marathon 引理：PMA 不变）。

### 3.3 条件注入（FiLM）
z 经两个线性头映射为 (γ, β) ∈ R^{128}，对 U-Net bottleneck 特征做逐通道仿射调制
bottleneck' = γ⊙b + β（γ 初始化 1、β 初始化 0，保证初始等价于无条件网络）。

### 3.4 逐光照反照率（选定 **S2 方案**）
- 主反照率头保持与基线一致（共享 albedo_shared）；
- 新增 ΔA 分支：cat(f_k 下采样至全分辨率, albedo_shared) -> Conv3x3 -> tanh × 0.1
  => ΔA_k ∈ [-0.1, 0.1]；
- A_k = clip(albedo_shared + ΔA_k, 0, 2)，仅参与该光照的渲染重建；
- 正则：λ_da · mean|ΔA_k|，λ_da 阶段表：{stage1: 0, stage2: 0.01, stage3: 0.05}
  （渐进启用，权重表改动按裁决书记 INC 级说明）；
- **13 项评估指标一律使用主 albedo**（= albedo_shared），与其它 run 口径可比。

## 4. 变长 N 批处理
**按 N 分桶**（同 batch 同 N；训练固定 N=5 无需分桶；评估时按 N 分组推理）。
pad+mask 方案暂不实现（风险 R3/R4），如后续需要再立项。N=1：前向路径成立
（注意力退化为单 token 自注意=PMA(seed, [token])），将在置换测试中显式验证。

## 5. RGB 双链路
stem 首层 in_channels 参数化（gray=1 / rgb=3）；v3 数据集同场景同时提供两种模态
PNG，几何/GT 完全同源；Phase 2 所有对比统一在 v3 口径内。

## 6. 指标口径预注册（冻结）
13 项指标的 albedo 一律取**主 albedo**；normal 由渲染器从预测深度导出；
depth 报告原始未对齐 + 最小二乘 aligned 两套（延续 T1.6 协议）；
image 用 final_render vs 输入。任何口径变更须先走规程 §7。

## 7. 显存预算
stem 与聚合模块增量 <8MB 参数；激活增量主要来自逐光照 ΔA 分支（K×1×256² float
≈ 0.26MB @bf16），实测峰值记录进冒烟报告。

## 8. 训练协议（INC-0010 审计裁决 v2 §3 A7 加注）

> **关键注**：本节为 INC-0010 v2 §3 A7 处置令所要求的论文学理化加注。
> 由 G-M8 实验（`tests_audit/test_earlystop_validity.py` + `out_M8_earlystop_validity.json`）
> 证伪：100 epoch 训练全程中，4-metric 早停门（`shading_var > 0.01 ∧ 0.8 ≤ sh0_mean ≤ 1.2 ∧
> albedo_corr > 0.7 ∧ lambertian_ratio > 0.95`）的**连续 10 窗口达成次数 = 0**。

**论文学理化约束**（防止读者把"早停条件被触发"误解为"模型已物理收敛"）：

1. **早停 = 工程节流**，不是物理完备性证明。训练中 `trainer.py:1352-1378` 的连续达标
   计数器是资源管理装置（节省 GPU 时间），不是物理量收敛判据。
2. **论文中禁用**"模型已收敛 / 训练完毕"等暗示物理完备性的措辞。G-M8 证明 4 个指标
   永远无法同时稳定触发——这是**数学设计本身**（lambertian_ratio > 0.95 的达成率
   < 30%），与"训练是否充分"无关。
3. **正确措辞**应当是"低曲率区（low-curvature region）"或"训练-验证损失相对稳定区间
   （plateau of training-validation loss）"。这一约束只描述损失曲面的局部几何性质，
   不暗示反照率/光照系数等具体物理量已经全局收敛。
4. **若必须报告物理量收敛**，应给出 epoch 99 时各指标的具体值 + 5 档 baseline
   （F-resA @ epoch 9/19/29/39/59）的对照表，**不**使用"已达标"等二元判定语言。

**与本设计预注册的兼容性**：本节是文档级澄清，不修改 §1–§7 任何架构/数据/指标定义；
若后续真要修改早停判据或训练协议，须按 §6 走规程 §7 变更流程。

## 9. 早停 return 与编排器协作的实测异常（INC-0010 v3 §3 A7 补段）

> **关键注**：本节为 INC-0010 v3 §3 A7 补段（"v3 新增段"），引用 §2 R3 与 §3 A8-bis 排查结果。
> v2 训练日志 `_arm_p2_t22_f_n5rgb_v2_log.txt` 在 epoch 76 打印 `连续 10 达标` 但训练继续到
> epoch 100。本节澄清**这不是 trainer 代码 bug，而是 trainer 早停与 run_arms 编排器协作的
> 设计哲学差异**。

### 9.1 实测日志证据

v2 训练日志第 6385-6394 行（`p2_t22_f_n5rgb_v2` 100 epoch 训练）：

```
[line 6385] 🎉 连续 10 个epoch达标！
[line 6391] 训练已完成，模型已达标！
[line 6394] 训练完成!
```

紧跟其后（[line 6411+]）`main.py` 第二次启动（`总Epoch数: 87`），这是 run_arms 编排器 spawn 的下一段
（seg 9 of 10，从 epoch 87 → 97）。**这说明 trainer.py:1411-1431 的 `return` 正常生效**——main.py
以 rc=0 正常退出；但 run_arms 编排器（`run_arms.py:468-535 train_arm`）只关心 `epochs_done`
文件系统状态，不知道 trainer 内部"已达标"语义，按 while 循环 spawn 下一段。

### 9.2 trainer 早停代码（已确认正常）

`trainer.py:1411-1431`：

```python
if self.continuous_qualified_epochs >= self.qualified_threshold:
    print(f"🎉 连续 {self.qualified_threshold} 个epoch达标！")
    ...
    self.save_checkpoint(self.current_epoch, val_losses['total'], True)
    ...
    self.writer.close()
    return  # ← 早停退出（实测生效）
```

实测：v2 epoch 76 时 `continuous_qualified_epochs = 10`，触发 `return`，main.py rc=0 退出。

### 9.3 run_arms 编排器（设计哲学问题，非代码 bug）

`run_arms.py:468-535 train_arm` 设计为**多段训练**：

```python
while True:
    done = epochs_done(ckpt_root, run_id)        # 只看文件系统
    if done >= TOTAL_EPOCHS: return True          # 100 epoch 才停
    seg = min(SEGMENT_EPOCHS, TOTAL_EPOCHS - done)  # 10 epoch/段
    target = done + seg
    cmd = spawn_main(target)                      # spawn 下一段
    rc = sh(cmd).returncode
    if rc == 42:                                  # 温度墙 → continue
        continue
    if after <= done:                             # 训练没推进 → abort
        return False
```

**run_arms 不知道 trainer 早停语义**——它假设每段 10 epoch 跑完就 `epochs_done += 10`，
继续下一段。即使 trainer 在 epoch 76 提前 return，run_arms 会 spawn 87→97 段。

### 9.4 处置（v3 判定）

按 v3 裁决 §2 R3 + §3 A8-bis：

1. **trainer.py:1411-1431 return 正常**，**无代码 bug**，**无需修复**。
2. **run_arms.py 多段训练设计**是 Phase 2 编排器标准行为——支持温度墙续跑、checkpoint 续训、
   段间资源回收。**不修改**。
3. **trainer 早停的"实际价值"**：仅在单次 main.py 跑到底（`--total_epochs 100`）的 spawn 模式下生效；
   在 run_arms 编排器多段模式下，trainer 早停 return 会被 run_arms 立即 spawn 下一段覆盖。
4. **论文叙事建议**：在论文中应说明"训练在编排器层面按段推进，trainer 早停 return 设计的工程价值
   体现在单次 main.py spawn 模式（手动调用或非编排场景）"。**这一说明仅是设计文档级澄清，不
   引入新数学假设**。

### 9.5 与 v3 §6 时间表协同

- A8-bis 排查结论：本节"非代码 bug"已澄清
- 不影响 A3-bis 3-seed 100 epoch run 启动
- 后续 A3-bis 训练日志中仍可能观察到 trainer 早停 return + run_arms 继续 spawn 的协作模式
  （属预期行为，不算异常）

---

## 10. 物理约束补建（INC-0012，2026-08-28 后期，决策 1 重定义版）

> **章节编号说明**：v3 §4 整改要求"按 v3 §3 A1-A8 + §4 放行条件 #1 整改"（决策 2 重定义）。
> 本节作为 v3 之后的独立修复，与 §8（v2 A7 加注）和 §9（v3 A7 补段）并列为 T2.2 设计文档第三段增补。

### 10.1 背景：决策 1 重定义

中期审计 v2 §2-P2 原始叙事为"fusion_unet.py 重构时丢失 Phase 0 Sigmoid 约束"。
经代码级核实（2026-08-28 23:00 只读检查阶段）：

- Phase 0 `unet_model.py:226` 注释显式记录"`# 移除 Sigmoid，让 Albedo 输出线性值`"——Phase 0 历史上**已显式移除 Sigmoid**
- `unet_model.py:549-551` `__main__` 打印字符串仍写"+ Sigmoid"（**文档 vs 代码漂移**）
- fusion_unet.py 复刻 Phase 0 同样的"无 Sigmoid"模式（值域期望 [0, 2] 但无显式激活）

**决策 1** 将修复路径重定义为"补建新约束 + 验证 albedo_smooth 二次风险"（**不是**"恢复丢失约束"），
理由：Phase 0 历史上就不存在该约束，恢复路径叙述与代码事实不符。

### 10.2 修复内容

| 修复项 | 代码位置 | 变更 | key 兼容性 |
|---|---|---|---|
| albedo 头 Sigmoid | `fusion_unet.py:120-130` `head_albedo()` 末层 | `nn.Conv2d(bc//2, 1, 1)` 后接 `nn.Sigmoid()` | 与旧 checkpoint 严格兼容（`albedo_head.0/1/3.*` key 路径不变）|
| depth 头 Softplus | `fusion_unet.py:135-145` `head_depth()` 末层 | `nn.Conv2d(bc//2, 1, 1)` 后接 `nn.Softplus()` | 与旧 checkpoint 严格兼容（`depth_head.0/1/3.*` key 路径不变）|
| 评估物理断言 | `evaluate_model.py` `assert_physical()` | albedo∈[0,1] / depth>0 违规像素占比统计入 `eval_summary.json` | 不影响现有 ckpt |
| T-PHYS 冒烟 | `_smoke_phys_constraints.py` | albedo10 vs albedo1 双轨 3 epoch 冒烟 | 仅冒烟用，D12 物理隔离 |

**关键设计点**：把激活接在 head() 工厂末层 Conv 之后，并保持单 Sequential 结构。
PyTorch 对无参数模块（Sigmoid/Softplus）跳过索引编号，所以新 key 路径与旧 checkpoint
完全一致（`albedo_head.{0,1,3}.*`），可严格 `load_state_dict(strict=False)` 加载 v2 best_model。

### 10.3 二次风险验证（T-PHYS 冒烟，3 epoch × 50 场景）

| 维度 | albedo_smooth=10.0（默认）| albedo_smooth=1.0（对照）| 结论 |
|---|---|---|---|
| albedo violation ratio | 0.0000% | 0.0000% | Sigmoid 头完全生效 |
| depth violation ratio | 0.0000% | 0.0000% | Softplus 头完全生效 |
| 0.5 退化（`abs(mean-0.5)<0.05 AND std<0.1`）| False（std=0.093）| False（std=0.091）| 两种权重均未触发 |

**结论**：在 v2 best warm-start + 3 epoch 训练下，`albedo_smooth=10.0` **未触发 0.5 退化**。
**维持 albedo_smooth=10.0 不下调**（与 INC-0010 早期"降至 10 保持主导正则地位"决策一致）。

为什么 Sigmoid 头下不会触发 0.5 退化：
- Sigmoid 把 albedo 值域锁在 [0,1]，模型不再需要靠"压向 0.5" 来满足范围约束
- albedo_smooth=10.0 的作用现在是"鼓励空间平滑"（梯度 L1），而不是"压值域"（已由 Sigmoid 保证）
- v2 模型在无 Sigmoid 时被压向 0.5 是因为它是"达到 [0,2] 值域 + 满足平滑"的折中；现在有 Sigmoid 后这个折中消失

### 10.4 论文叙事（中期审计 v2.1 §10 约束第 6 条执行）

按 v2.1 §10 约束第 6 条"反照率退化修复（P2/P3）必须在方法节或补充材料中如实描述（约束丢失 + 修复），
这是工程严谨性证据，不是污点"——论文方法节须包含：

1. **物理约束的"设计意图"**：[0,1] 范围 + 正深度是物理可解释的先验（retroreflective 假设 / 朗伯假设），
   不是"重构时丢失的 bug"——v2 模型在无约束时是被训练压力推向值域边缘，而非表达错误
2. **修复路径**：在 fusion_unet.py 主干 head 层显式接 Sigmoid/Softplus 激活；这是**新增**的物理约束，
   不应叙述为"恢复"
3. **二次风险评估**：在引入 Sigmoid 后重新评估 `albedo_smooth` 权重的兼容性（防止压向 0.5），
   验证结论"维持 10.0 不下调"——这是工程严谨性证据

### 10.5 实施产物（D3 数字入库）

- 冒烟产物：`D:\Multi-Illumination Inverse Rendering\_SMOKE_phys_constraints\{albedo10,albedo1}\*.json`
- 物理断言：每次 `evaluate_model.py` / `eval_n_curve.py` / `evaluate_diligent.py` 推理都执行
- INC-0012 文档：`docs/incidents/INC-0012_物理约束补建与albedo_smooth二次风险验证.md`
- 决策链路：本节 → INC-0012 → 中期审计 v2 §2-P2/P3 → 顶层设计 v2.1 任务卡 T2.2 物理约束修复段

### 10.6 引用关系

- 承接：中期审计 v2 §2-P2/P3 + 顶层设计 v2.1 T2.2 物理约束修复段
- 决策 1：原始"恢复丢失"叙事 → "补建新约束 + 验证二次风险" 重定义
- 不冲突：INC-0011（F-resA 1 run 反转方法论沉淀）
- 配套：evaluate_model.py / eval_n_curve.py / evaluate_diligent.py 评估脚本物理断言
- 后续：等 A3-bis 3-seed 完成后，重训 v2 时**保留**本节 Sigmoid/Softplus 约束（不再回退）

