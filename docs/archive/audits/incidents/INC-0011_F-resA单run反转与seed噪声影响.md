# INC-0011 · F-resA 单 run 反转与 seed 噪声影响的方法论沉淀

- 日期：2026-08-28
- 影响：早停判据可信度（G-M8 1 run 结论被 Q1 4 run 复跑反转）
- 状态：**已关闭**（v3 处置：MIXED + 软约束 + A3-bis 3-seed 验证；不需新代码改动）

---

## 1. 时间线

| 时间 | 事件 |
|---|---|
| 2026-08-27 | G-M8 单 run 跑出 "F-resA 100 epoch never triggers"（1 run 噪声） |
| 2026-08-28 凌晨 | INC-0010 v1 采纳 G-M8 结论，判早停为 GATE_INEFFECTIVE |
| 2026-08-28 03:55 | INC-0010 v2 把 G-M8 1 run 结论保留为硬约束 |
| 2026-08-28 15:18 | Q1 复跑扩展到 4 run（`tests_audit/test_earlystop_q1.py`），verdict=MIXED（3/4 触发，1/4 不触发）|
| 2026-08-28 15:26 | INC-0010 v3 反转：早停判据从 GATE_INEFFECTIVE → MIXED；A3-bis 3-seed 验证列为新增项 |
| 2026-08-28 23:00 | INC-0011 登记本条（沉淀方法论，无新代码修复）|

---

## 2. 证据链

| 文件 | 角色 | 关键数字 |
|---|---|---|
| `tests_audit/test_earlystop_validity.py` | G-M8 单 run 测试脚本 | "F-resA 100 epoch never triggers" |
| `tests_audit/out_M8_*.json` | G-M8 单 run 详细输出 | 触发数=0 / 100 epoch |
| `tests_audit/test_earlystop_q1.py` | Q1 4 run 复跑脚本 | 11420 字节，2026-08-28 15:17 |
| `tests_audit/out_Q1_earlystop_4runs.json` | Q1 4 run 详细输出 | 3/4 run 触发，1/4 run 不触发，verdict=MIXED |
| `tests_audit/Q1_merged_verdict.md` | Q1 复跑裁决 | 4585 字节，2026-08-28 15:18 |
| `docs/incidents/INC-0010_审计裁决_v3_20260828_最终版.md` §7 v2→v3 差异表 | 处置依据 | "早停判据：GATE_INEFFECTIVE → MIXED" |

---

## 3. 根因分层

### 3.1 直接原因
G-M8 阶段只跑了 1 个 seed run，恰好抽到不触发早停的 seed 样本——**单 run 数据的统计噪声主导结论**。

### 3.2 放大因素
早停判据依赖"连续 10 个 batch 达标"统计量（val_loss ≤ 阈值），小样本下 std 偏大，触发与否的判定对单 run 噪声敏感。G-M7 已知跨 run std=12.3%（软约束预算），未在 G-M8 阶段同步收紧。

### 3.3 过程性原因
v1 / v2 阶段均以单 run 数据作为裁决依据（"10 epoch 够用"、"100 epoch 中间档位"、"早停是内部逻辑"），未建立"早停/健康判据必须 ≥3 run 复跑"的工程纪律。**这是流程级缺口，不是代码级 bug**。

---

## 4. 已实施处置（v3 裁决 + A3-bis 流程）

| 处置 | 出处 | 状态 |
|---|---|---|
| 早停判据从硬约束改为软约束（MIXED + mean val_loss@99 ∈ [0.0167, 0.022]）| INC-0010 v3 §4 放行条件 #2 | ✅ 已实施 |
| A3-bis 3-seed 100 epoch 抖动实验 | INC-0010 v3 §3 A3-bis | ⏳ seed 42 33/100 中断态已存档，seed 123/2024 未启动 |
| Q1 4 run 复跑扩展（从 1 run → 4 run）| `tests_audit/test_earlystop_q1.py` | ✅ 已完成 |
| F-resA 单 run 数据保留为反例素材 | 本 INC-0011 §5 | ✅ 已登记 |

**结论**：无新代码改动需求（v3 流程处置已覆盖）；本 INC-0011 仅为方法论沉淀。

---

## 5. 未尽事项 / 后续观察

| # | 事项 | 转交目标 |
|---|------|---------|
| 1 | 任何依赖 1 run 数据的早停/健康判据必须 ≥3 run 复跑 | 写入《顶层设计 · 任务工作指导书》v2.2 纪律（D14 候选）|
| 2 | A3-bis 3-seed 跑完后，重算 13 项指标均值 ± std，写入 INC-0010 v4 验收报告 | T-ARM 完成后触发 |
| 3 | 早停 trainer.py:1411-1431 `return` 与 run_arms 编排器协作的 R3 异常（v3 §3 A8-bis）—— 已被 t2_2_design.md §9.4 声明"无需修复" | 维持 v3 处置，不登记新 INC |

---

## 6. 可借鉴的失败点（沉淀为纪律或规程修改建议）

### 6.1 失败点
G-M8 阶段对"早停判据"这一**双值性结论**（触发 / 不触发）做了单 run 验证——单 run 验证对**单值性指标**（如 PSNR）勉强可接受，对**双值性结论**则完全不充分。

### 6.2 沉淀建议
- **D14 候选 · 判据充分性纪律**：早停 / 健康 / 收敛等双值性判据必须 ≥3 run 复跑；单 run 结论仅作初判；多 run 结论才可作裁决依据。
- **审计规程 §5 抽检规则扩展**：判据类门禁（G2.5 / G2.6）的抽检应优先核对 "≥3 run" 而非单 run 数字。
- **复跑命令模板**：所有 G-M* 脚本必须接受 `--num_runs N` 参数（默认 3），单 run 调用必须显式 `--num_runs 1` 警示。

### 6.3 适用范围
本教训不仅限于早停判据——任何**双值性裁决**（触发 / 不触发、通过 / 失败、合格 / 不合格）都应同等对待。

---

## 7. 引用关系

- **承接**：INC-0010 v3 §2 R3（早停 R3 三可能根因经实测全部为伪）+ §3 A8-bis（排查结论）
- **引用**：tests_audit/test_earlystop_validity.py / out_M8_*.json + tests_audit/test_earlystop_q1.py / out_Q1_earlystop_4runs.json
- **配合**：docs/HANDOFF_20260828.md §5.2 TODO 4
- **不冲突**：T-PHYS（物理约束修复）使用 INC-0012 编号

---

*本 INC 由 2026-08-28 23:00 只读检查阶段决策（决策 3）落地，方法论沉淀不涉及代码改动。*
