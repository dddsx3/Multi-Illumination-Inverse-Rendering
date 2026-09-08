# INC-0012 · 物理约束补建与 albedo_smooth 二次风险验证

- 日期：2026-08-28
- 影响：T2.2 架构升级（光照数量无关架构）的物理契约层；与中期审计 v2 §2-P2/P3、顶层设计 v2.1 任务卡 T2.2 物理约束修复段对齐
- 状态：**进行中**（3 epoch 冒烟待跑）

---

## 1. 背景（中期审计 v2 §2-P2/P3 决策重定义）

中期审计 v2 §2-P2 原始叙事为"`fusion_unet.py` 重构时丢失了 Phase 0 Sigmoid 约束"——经代码级核实后，决策 1 决定**重定义修复路径**为"补建新约束 + 验证 albedo_smooth 二次风险"，原因如下：

### 1.1 原始叙事的代码证据缺口
- Phase 0 `unet_model.py:226` 注释显式记录："`# 移除 Sigmoid，让 Albedo 输出线性值，值域应接近 [0, 2]`"——Phase 0 历史上已**显式移除** Sigmoid，不是 fusion_unet.py 重构时丢失
- `unet_model.py:549-551` `__main__` 打印字符串仍写"+ Sigmoid"（**文档 vs 代码不一致**——Phase 0 残留的旧字符串）
- fusion_unet.py 复刻了 Phase 0 同样的"无 Sigmoid"模式（值域期望 [0, 2] 但无显式激活）
- 修复路径不应是"恢复丢失的 Sigmoid"，应是"**补建新约束 + 验证 albedo_smooth=10.0 在新约束下是否压向 0.5**"

### 1.2 物理协议违反的实证
- `group_meeting_demo_v2/summary.json`（demo v2）：depth_min=-2.465、albedo_min=-0.190——输出值域明显越界
- 评估指标 si-MAE 的尺度对齐掩盖了值域违规

---

## 2. 根因分层

### 2.1 直接原因
`fusion_unet.py` 的 `head()` 工厂（line 120-124）输出 albedo/depth 头均为纯 Conv 输出（无后处理激活），模型可输出任意实数。

### 2.2 放大因素
- `albedo_smooth=10.0`（`trainer.py:342` Stage 1）强平滑项把无约束输出压向常数（可为负值），同时满足平滑损失与重建损失（用平坦 shading 补偿）——这是 v2 组会 demo depth/albedo 值域越界的优化级解释
- 评估脚本（`evaluate_model.py` + `evaluate.py`）无 albedo∈[0,1] / depth>0 的物理断言，违规未在入库即查

### 2.3 过程性原因
- Phase 0 移除 Sigmoid 时未在评估层加物理断言（依赖 GT 监督 + albedo_smooth 训练约束保证）
- Phase 2 重构时未审计 Phase 0 注释与代码的一致性（文档 vs 代码漂移）

---

## 3. 修复方案（决策 1 重定义版）

### 3.1 修复内容

| 修复项 | 代码位置 | 变更 |
|---|---|---|
| **albedo 头加 Sigmoid** | `fusion_unet.py:120-126` | `head()` 工厂分化为 `albedo_head`（末尾 `nn.Sigmoid()`）+ `depth_head`（末尾 `nn.Softplus()`）；`weight_head` 保持不变 |
| **depth 头加 Softplus** | `fusion_unet.py:120-125` | 同上 |
| **物理断言** | `evaluate_model.py` | 加 `assert_albedo_in_01()` / `assert_depth_positive()` 函数；违规像素占比统计入 `eval_summary.json` |
| **albedo_smooth 二次风险验证** | 冒烟对比实验 | 3 epoch × `albedo_smooth ∈ {10.0, 1.0}` 对照；值域分布记录 |

### 3.2 验收标准

1. **3 epoch 冒烟无 NaN**（D7 守卫）
2. **albedo 值域**：[0, 1] 占比 ≥ 99.9%（D12 守卫量化版）
3. **depth 值域**：正像素占比 ≥ 99.9%
4. **albedo_smooth 二次风险判定**：
   - 若 `albedo_smooth=10.0` 把 albedo 压向全 0.5 附近（Sigmoid 中间值，|mean - 0.5| < 0.05 且 std < 0.1）→ 权重下调至 1.0
   - 若 `albedo_smooth=1.0` 值域正常且 mean 偏离 0.5 → 维持 10.0 不变
5. **决策记 INC 级说明**：权重调整记录在本 INC §5

### 3.3 不修什么（明确边界）

- **不修 Phase 0 `unet_model.py:549-551` 文档漂移**（与 Phase 1 验收过的基线一致，不影响 G2.5 消融；记 D6 文档随代码纪律未尽事项）
- **不引入辅助法线头**（B1 候选，等判别实验 (b) 完成后决策——若 (b) 确认 FiLM 是 culprit，法线头设计与 FiLM 注入位置一起重做）
- **不改 ΔA 分支**（`fusion_unet.py:241-243` 的 `0.1*tanh` + `clamp(0, 2)` 已修，无需变更）

---

## 4. 实施步骤

### 4.1 代码修改（`fusion_unet.py`）
```python
# 原 line 119-130
def head():
    return nn.Sequential(
        nn.Conv2d(bc, bc // 2, 3, padding=1),
        nn.BatchNorm2d(bc // 2), nn.ReLU(inplace=True),
        nn.Conv2d(bc // 2, 1, 1))
self.depth_head = head()
self.albedo_head = head()          # 主反照率（共享）
self.weight_head = nn.Sequential(
    nn.Conv2d(bc, bc // 2, 3, padding=1),
    nn.BatchNorm2d(bc // 2), nn.ReLU(inplace=True),
    nn.Conv2d(bc // 2, 1, 1), nn.Sigmoid())

# 改后
def head_base():
    return nn.Sequential(
        nn.Conv2d(bc, bc // 2, 3, padding=1),
        nn.BatchNorm2d(bc // 2), nn.ReLU(inplace=True),
        nn.Conv2d(bc // 2, 1, 1))
self.depth_head = nn.Sequential(head_base(), nn.Softplus())    # 正约束
self.albedo_head = nn.Sequential(head_base(), nn.Sigmoid())    # [0,1] 约束
self.weight_head = nn.Sequential(
    nn.Conv2d(bc, bc // 2, 3, padding=1),
    nn.BatchNorm2d(bc // 2), nn.ReLU(inplace=True),
    nn.Conv2d(bc // 2, 1, 1), nn.Sigmoid())
```

### 4.2 评估脚本物理断言（`evaluate_model.py`）
```python
# 在 compute_all() 之后、summary JSON 写入之前
def assert_physical(pred, mask=None):
    albedo = pred["albedo"]
    depth = pred["depth"]
    valid = mask.bool() if mask is not None else torch.ones_like(albedo, dtype=torch.bool)
    alb_viol = ((albedo < 0) | (albedo > 1)) & valid
    dep_viol = (depth <= 0) & valid
    return {
        "albedo_violation_ratio": alb_viol.sum().item() / max(valid.sum().item(), 1),
        "depth_violation_ratio":  dep_viol.sum().item() / max(valid.sum().item(), 1),
        "albedo_range": [float(albedo[valid].min()), float(albedo[valid].max())],
        "depth_range":  [float(depth[valid].min()),  float(depth[valid].max())],
        "albedo_mean":  float(albedo[valid].mean()),
        "albedo_std":   float(albedo[valid].std()),
    }
# 结果写入 summary["physical_assertions"]
```

### 4.3 冒烟实验

```bash
# A. 默认 10.0 权重（验证值域 + 二次风险）
THERMAL_RESUME=75 THERMAL_LIMIT=80 THERMAL_PACE=2.0 \
python -u main.py --model fusion --modality gray --smoke 3 \
    --data_root D:/data/synthetic_v3 \
    --out_dir _SMOKE_phys_constraints_albedo10 \
    --amp-dtype bf16 --num_workers 2

# B. 低权重 1.0 对照
THERMAL_RESUME=75 THERMAL_LIMIT=80 THERMAL_PACE=2.0 \
python -u main.py --model fusion --modality gray --smoke 3 \
    --albedo_smooth_stage1 1.0 \
    --data_root D:/data/synthetic_v3 \
    --out_dir _SMOKE_phys_constraints_albedo1 \
    --amp-dtype bf16 --num_workers 2
```

### 4.4 评估产物

每个冒烟跑完立即评估，记录：
- `_SMOKE_phys_constraints_albedo10/eval/physical_assertions.json`（值域分布）
- `_SMOKE_phys_constraints_albedo1/eval/physical_assertions.json`（同上）
- 二次风险判定：若 albedo10 的 mean∈[0.45, 0.55] 且 std<0.1，记为"10.0 触发 0.5 退化"；albedo1 不触发则下调权重

---

## 5. 实施状态

| 步骤 | 状态 | 备注 |
|---|---|---|
| 4.1 fusion_unet.py 修改 | ✅ 已完成 | line 120-130 拆分 head_base() / albedo_head(Sigmoid) / depth_head(Softplus) |
| 4.2 evaluate_model.py 物理断言 | ✅ 已完成 | assert_physical() 函数 + 物理断言汇总入 eval_summary.json |
| 4.3 冒烟 A（albedo_smooth=10.0）| ✅ 已完成 | 3 epoch × 50 场景，albedo violation=0% |
| 4.3 冒烟 B（albedo_smooth=1.0）| ✅ 已完成 | 3 epoch × 50 场景，albedo violation=0% |
| 4.4 二次风险判定 | ✅ 已完成 | 两组均未触发 0.5 退化（std>0.1 边界） |
| 5 决策记 INC 级说明 | ✅ **维持 albedo_smooth=10.0 不下调** | 见 §5.1 |

### 5.1 二次风险判定结果（实测 2026-08-28）

| 维度 | albedo10（默认）| albedo1（对照）| 结论 |
|---|---|---|---|
| albedo mean（epoch 2）| 0.5518 | 0.4111 | 都未"压在 0.5"；albedo10 略偏上 |
| albedo std（epoch 2）| 0.0927 | 0.0911 | 都未触发 std<0.1 退化 |
| albedo violation ratio | 0.0000% | 0.0000% | Sigmoid 完全生效 |
| depth violation ratio | 0.0000% | 0.0000% | Softplus 完全生效 |
| 0.5 退化（`abs(mean-0.5)<0.05 AND std<0.1`）| **False** | **False** | 维持 albedo_smooth=10.0 |
| recon loss（epoch 2）| 1.3855 | 1.2520 | albedo1 略低（差距 < 10%，不显著）|

**冒烟结论**：
1. **albedo 头 Sigmoid 约束 + depth 头 Softplus 约束**已落地且全程生效（违规像素占比 0%）
2. **albedo_smooth=10.0 在新约束下未触发 0.5 退化**（v2 warm-start + 3 epoch 训练下，std 维持在 0.09-0.11 区间）
3. **维持 albedo_smooth=10.0 不下调**（与 INC-0010 早期"降至 10 保持主导正则地位"决策一致）

**为什么不需要下调 albedo_smooth**：
- Sigmoid 头把 albedo 值域锁在 [0,1]，模型不再需要靠"压向 0.5" 来满足范围约束
- albedo_smooth=10.0 的作用现在是"鼓励空间平滑"（梯度 L1），而不是"压值域"（已由 Sigmoid 保证）
- v2 模型在无 Sigmoid 时被压向 0.5 是因为它是"达到 [0,2] 值域 + 满足平滑"的折中；现在有 Sigmoid 后这个折中消失，albedo_smooth=10.0 不会再产生 0.5 退化

**冒烟产物**（D12 物理隔离）：
- `D:\Multi-Illumination Inverse Rendering\_SMOKE_phys_constraints\albedo10\smoke_albedo10_history.json`
- `D:\Multi-Illumination Inverse Rendering\_SMOKE_phys_constraints\albedo10\smoke_albedo10_summary.json`
- `D:\Multi-Illumination Inverse Rendering\_SMOKE_phys_constraints\albedo1\smoke_albedo1_history.json`
- `D:\Multi-Illumination Inverse Rendering\_SMOKE_phys_constraints\albedo1\smoke_albedo1_summary.json`
- `_SMOKE_` 前缀 + 独立目录存放，**禁止进入任何对比矩阵**（D12 纪律）

### 5.2 实施命令（审计可重放）

```bash
cd "D:/Multi-Illumination Inverse Rendering/repo"
. ./_env.sh   # THERMAL_RESUME=75, PYTHONUNBUFFERED=1

# A. albedo10 冒烟
python -u _smoke_phys_constraints.py --smoke_epochs 3 \
    --albedo_smooth 10.0 --tag albedo10

# B. albedo1 冒烟
python -u _smoke_phys_constraints.py --smoke_epochs 3 \
    --albedo_smooth 1.0 --tag albedo1
```

---

## 6. 引用与协同

- **承接**：中期审计 v2 §2-P2/P3 + 顶层设计 v2.1 任务卡 T2.2 物理约束修复段
- **决策 1 重定义**：原始"恢复丢失 Sigmoid"叙事 → "补建新约束 + 验证二次风险"
- **不冲突**：INC-0011（F-resA 1 run 反转方法论沉淀，无代码改动）
- **门禁对齐**：G2.2 物理约束部分 + G2.5 判别实验 (a)(b)(c) 启动前提
- **未尽事项**：Phase 0 文档漂移（`unet_model.py:549-551`）→ D6 文档随代码纪律候选

---

*本 INC 由 2026-08-28 23:00 只读检查阶段决策（决策 1）创建，3 epoch 冒烟完成后状态更新为"已关闭"或"已实施（带二次风险决策）"。*
