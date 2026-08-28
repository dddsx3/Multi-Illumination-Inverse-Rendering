# splits 划分文件审计 · C4 报告

> **目的**：验证 G2.6 放行条件 C1（test 集冻结），并发现 v2/v3 划分的差异是否影响对比公平性（D11）。
> **执行日期**：2026-08-28 23:00 后期阶段
> **关联门禁**：G2.6 test 集正式化 / D11 公平对比纪律

---

## 1. 划分文件元信息

| 文件 | 字节 | mtime | 格式 |
|---|---|---|---|
| `splits/synthetic_v2.json` | 25,183 | 2026-08-25 04:35 | JSON with train/val/test 列表 |
| `splits/synthetic_v3.json` | 24,676 | 2026-08-25 11:55 | JSON with train/val/test 列表 |

---

## 2. 划分统计

| 划分 | synthetic_v2 | synthetic_v3 | 差异 |
|---|---|---|---|
| train | 456 | 447 | -9（v3 训练集缩小）|
| val | 50 | 49 | -1（v3 val 集缩小）|
| test | 127 | 124 | -3（v3 test 集缩小）|
| **总** | **633** | **620** | **-13（v3 渲染减少 13 场景）**|

---

## 3. 划分完整性自检

| 检查 | 结果 | 状态 |
|---|---|---|
| v2 train 完全在 v2 test 之外 | ✓ True | G2.6 C1 PASS |
| v2 val 完全在 v2 test 之外 | ✓ True | G2.6 C1 PASS |
| v3 train 完全在 v3 test 之外 | ✓ True | G2.6 C1 PASS |
| v3 val 完全在 v3 test 之外 | ✓ True | G2.6 C1 PASS |
| v3 test 完全在 v3 train 之外 | ✓ True | G2.6 C1 PASS |
| **v2 train ∩ v3 test** | **89 场景重叠** | ⚠️ **D11 公平对比影响** |
| **v2 test ∩ v3 train** | **89 场景重叠** | ⚠️ **D11 公平对比影响** |
| v2 test == v3 test | False（仅 26 场景重叠）| ⚠️ **D11 公平对比影响** |

---

## 4. 关键发现：v2/v3 划分差异对 D11 的影响

**问题**：v2 best 训练时 89 个场景在训练集；v3 best 测试时这 89 个场景在 test 集。
v3 best 训练时这 89 个场景被移到了 test 集（**没有 v3 best 在 v2 test 上的指标**）。
v2 best 在 v2 test 上的指标（如 F-N5-gray 7.79° 法线）与 v3 best 在 v3 test 上的指标（如 v2 best 8.18° 法线）**不能直接对比**。

**涉及数据流**：
- Phase 1 R0 v3gray / F-N5-gray：v2 划分，v2 test 127 场景
- Phase 2 v2 best (n5rgb)：v3 划分，v3 test 124 场景
- Phase 2 albOff / resA：v3 划分，v3 test 124 场景
- Phase 2 T-PHYS 冒烟：v2 划分（v2 best warm-start），但只用了 50 训练子集

**影响**：
- **F-N5-gray albedo_si_mae 0.128 vs R0 v3gray albedo_si_mae 0.055**——这两个数字**不能直接对比**（不同 test 集）
- **v2 best PSNR 37.25 vs albOff 35.69 vs resA 36.54**——三者均在 v3 test 上评估，**可对比**（同 test 集）

**审计提示（D11 合规）**：
- Phase 1 vs Phase 2 对比表**必须声明 test 集不同**（v2 test 127 ≠ v3 test 124）
- Phase 2 内变体对比（v2 / albOff / resA / physcon / resC / noFiLM / lowSmooth）**统一 v3 test**，**可直接对比**

---

## 5. 划分可复现性核验（D4 纪律）

| 项 | 状态 |
|---|---|
| 划分文件已落盘 | ✓ `splits/synthetic_v3.json` 24,676 bytes |
| 加载函数 `load_split(manifest_path, split_name)` | ✓ 可复现 |
| 训练脚本使用 `args.split_manifest` 显式指定 | ✓ |
| test 集不参与训练（写死）| ✓ 训练脚本默认 split=val，test 需显式指定 |

**D4 纪律 PASS**。

---

## 6. 改进建议

1. **Phase 2 论文对比表**：所有 Phase 2 内变体对比统一标注"v3 test 124 场景"
2. **Phase 1 vs Phase 2 对比表**：单独标注"v2 test 127 场景"与"v3 test 124 场景"——**禁止混合**
3. **后续阶段（Phase 3 真实世界验证）**：test 集继续冻结，扩展到手机实拍数据时**新建独立 val/test 划分**（不动 synth v3 test）

---

## 7. 引用

- 顶层设计 v2.1 任务卡：`文档类材料/顶层设计-任务工作指导书 (1).md` §4 T2.6 任务卡（test 集正式化）
- 独立审计规程 v2.1：`独立审计执行/顶层设计-独立审计与门槛验收规程 (1).md` §2.4 G2.6 门禁 + §3.2 报告数字可追溯
- D11 公平对比纪律：`文档类材料/顶层设计-任务工作指导书 (1).md` §3 纪律 D11
- 划分文件：`splits/synthetic_v2.json` + `splits/synthetic_v3.json`

---

*本报告由 2026-08-28 23:00 阶段决策落地。*
