# PRE0_VERDICT · PRE-0 历史结论永久冻结

> **本文件由 P1 阶段签发（2026-08-30）**，对 PRE-0 阶段所有产物作最终结论。
> 此后任何引用 PRE-0 数字作为“多光照分解”证据的文档、论文、汇报均视为不成立。
> 本文件不可被 PRE-0 内部产物自行修改；只有新的 P1 数据修复完成 + 新审计
> 通过后，才能以"取代声明"形式重写本文件。

## 1. 永久固化结论（permanently valid as of 2026-08-30）

1. **`synthetic_v3` 不构成有效多光照数据**。
   每场景 `light_001..005.png` 实为同一张图（两两平均灰差 0.000~0.049/255，
   99.9% 像素逐位相同，差异仅为 Cycles 路径追踪采样噪声 ±1~2 灰阶；
   `sh_coeffs.npy` 5 行参数确实不同但渲染帧未随之变化——BlenderProc
   `Light.set_location(..., frame=k)` 逐帧动画在本数据集生成中未生效）。
   抽检 ≥20 场景、PRE-01 oracle 逐光 PSNR 5/5 近似常数、PRE-03 probe
   encoder 跨 5 图特征 std ≈ 1e-6，三条独立证据链一致。

2. **旧 N curve / cross-subset / held-out relighting / Probe A/B/C
   不具有方法区分力**。N 曲线对 N 逐位平坦不是因为模型鲁棒，
   是因为输入本身相同（数据缺陷）；A/B/C test SI-MAE 差 <0.001
   不是三种聚合方法真实无差，而是真多光照维度未到位。

3. **旧 synthetic_v3 上任何"多光照融合有效/无效"的论文结论均作废**。
   包含但不限于：v2 best 主结果中"多光照分解"叙事、N 曲线"鲁棒"
   结论、置换不变性"成功"叙述、DiLiGenT zero-shot 数字的"跨数据集
   泛化"含意。v2 best 的 PSNR/SI-MAE 数字本身仍可作为"单光照重建"
   参考，但不能作为"多光照逆渲染"证据。

4. **PRE-0 的 Probe 仅作为管线验证资产保留**。其 checkpoint 不入
   正式实验；其架构（~0.71M Encoder/Decoder/三个 aggregation 变体）
   仍可作为 P1 阶段参考，但行为不可作为 P1 训练结果参考。

5. **原始 synthetic_v3 不删除**。数据集原地保留，标注：
   `status: invalid_for_multi_illumination_claims`。
   理由：可能仍用于"P 域单光照 + R 域部分场景"等狭窄对照；
   删除会破坏可复现性。

## 2. machine-readable header（已自动写入所有 PRE-0 报告）

```yaml
protocol_status: superseded
valid_for_multi_illumination_claims: false
reason: duplicated illumination frames
supersedes: synthetic_v3 (commit 2c23026, before 2026-08-29)
next_valid_artifact: p1/calibration_set/, p1/physics_clean/
verdict_issuer: P1 task (docs/P1_任务书.md, 2026-08-30)
```

## 3. 受影响报告与文件清单

| 路径 | 状态 |
|---|---|
| `pre0/HANDOFF.md` | **superseded**；Q4/Q6 数字仅作"工具链验证"参考 |
| `pre0/oracle_renderer/ORACLE_AUDIT.md` | **superseded**；§1-§6 仍作物理协议分析参考，§7 数据缺陷结论永久生效 |
| `pre0/information_audit/INFORMATION_AUDIT.md` | **superseded**；仅"双线性病态"等方法论发现仍生效 |
| `pre0/probe_results/probe_*_summary.json` | **superseded**；仅作管线可跑通的证据 |
| `pre0/checkpoints/probe_*_best.pth` | **superseded**；保留供参考不参与 P1 正式实验 |
| `pre0/literature/literature_matrix.csv` | **provisional novelty map**（per P1-12 升级为 `closest_prior_verified.md`） |
| `pre0/evidence_accumulation/*` | **superseded**；N 曲线对 N 平坦的结论与"数据有真多光照"前提不一致 |
| `pre0/heldout_relighting/*` | **superseded**；在本数据上 query = support 是伪 held-out |
| `pre0/benchmark/DILIGENT_CONTRACT.md` | **继续生效**；数据合同模板在 P1 阶段直接复用 |

## 4. P1 阶段必须重新做的实验（per P1-08/10/15/16）

数据修复（每灯独立 render call + 新增生成端门禁 G1/G2/G3）→ Calibration
Set 通过 → Physics-clean 主数据集生成 → PRE-02 重做（受控 solver）→
最小 Probe 重训 → 5 项 Learnability Gate（C1 N curve / C2 novel-vs-dup /
C3 diversity / C4 cross-subset / C5 oracle-query-light held-out）。

## 5. P1 阶段不重做的实验

- 旧 v2 best 主训练（任务书 §1 禁止）
- 任何 100 epoch 主方法实验
- 论文主表比较
- "3-seed 抖动"训练
- 任何基于 PRE-0 checkpoint 的微调

---

**签发**：P1 任务书 §P1-00 · ZCode agent · 2026-08-30
**取代**：PRE-0 阶段全部"多光照"相关结论
**未取代**：管线/工具/物理协议/方法论发现（见各文件 §1 重声明）
