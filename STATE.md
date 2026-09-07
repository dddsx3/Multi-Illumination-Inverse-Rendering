# STATE · 进度账本（每次会话开始必读、结束必更）

> **主线一句话**：校准置信度连续谱 → 逐模式信息预测 → 受控真实数据验证（CI01–CI05 → TCI 投稿）。
> **宪法**：[docs/Calibration_Confidence_Continuum_TCI_实验设计书_v1.0.md](docs/Calibration_Confidence_Continuum_TCI_实验设计书_v1.0.md)
> **执行条例**：[docs/主控计划书_20260907_TCI_从当前状态到投稿_v1.0.md](docs/主控计划书_20260907_TCI_从当前状态到投稿_v1.0.md) · 卡状态只有 pending / in_progress / done / blocked / void
> **交接指令**：读 STATE.md + REPO_MIGRATION.md 锚点列 + 主控计划书 §4 当前卡，继续执行。

| 卡 | 状态 | 结论一行 | 证据/artifact | commit | 更新日期 |
|----|------|----------|----------------|--------|----------|
| C01 仓库地基 | done | 目录树与宪法§6逐目录一致；CLAIMS_REGISTRY 含 C1–C6+红队增补 N/V+红线；pip install -e . 绿 + pytest 1 passed | CLAIMS_REGISTRY.yaml; pyproject.toml; src/calibinfo/ | c23cedb | 2026-09-07 |
| C02 V1–V6 测试迁移 | done | 23/23 绿（~4s）；V1–V6+rank-deficient 阈值分层锁定；legacy 对照逐位一致（V1 median 1.0045/V4 rel err 6.44e-15/V2 2.23e-8 均与红队报告吻合）；legacy 无被任何实验 import；exp9d 结果已在本 clone 查证 | tests/unit/×8; legacy_redteam/ | 本commit(C02) | 2026-09-07 |
| C03 迁移矩阵执行 | pending | — | — | — | — |
| C04 run manifest 骨架 | pending | — | — | — | — |
| C05 Gate A 冒烟 | pending | — | — | — | — |
| C06 双路线 ΔF | pending | — | — | — | — |
| C07 CI01 正式 | pending | — | — | — | — |
| C08 理论手稿定稿 | pending | — | — | — | — |
| C09 scene 工厂 | pending | — | — | — | — |
| C10 CI02 gauge/λ⋆ | pending | — | — | — | — |
| C11 retention+tracking | pending | — | — | — | — |
| C12 CI03 线性 MC | pending | — | — | — | — |
| C13 CI03 非线性包络 | pending | — | — | — | — |
| C14 OpenIllumination 适配 | pending | — | — | — | — |
| C15 Σ_c 生成器+pilot | pending | — | — | — | — |
| C16 CI04 正式 | pending | — | — | — | — |
| C17 DiLiGenT 适配 | pending | — | — | — | — |
| C18 CI05 sanity | pending | — | — | — | — |
| C19 ablation 附录 | pending | — | — | — | — |
| C20 图表冻结 | pending | — | — | — | — |
| C21 初稿组装 | pending | — | — | — | — |
| C22 复现审计 | pending | — | — | — | — |
| C23 内审投稿 | pending | — | — | — | — |

## 条件卡（触发才执行）

| 卡 | 触发条件 | 依据 |
|----|----------|------|
| OPT-1 VarPro | CI03/CI04 估计器单解 > 数秒且成瓶颈 | 主控计划书 §1；五判据冻结于 workspace/卡OPT1 |
| 卡R2 复活（exp12v4） | CI05 需要合成侧判别力失败案例 | v3.3 卡 R2 原文步骤仍有效 |

## 阻塞登记

| 事项 | 阻塞物 | 解除条件 | 登记日期 |
|------|--------|----------|----------|
| ~~exp9d 结果状态未证实~~ **已解除（2026-09-07，卡 C02 查证）** | 云 clone 缺文件；本机 Windows clone **存在完整结果**：`critical_experiments/exp9d_terminator_localization.json`，数字与处置裁决表一致（λ terminator 带 1.49× / μ 0.919× vs 预注册 ≥5×，contour 对照 3.43×） | 已解除；分支维持归档，数字允许按红队报告口径引用 | 2026-09-07 |
