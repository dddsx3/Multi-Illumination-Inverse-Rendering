# STATE · 进度账本（每次会话开始必读、结束必更）

> **主线一句话**：校准置信度连续谱 → 逐模式信息预测 → 受控真实数据验证（CI01–CI05 → TCI 投稿）。
> **宪法**：[docs/Calibration_Confidence_Continuum_TCI_实验设计书_v1.0.md](docs/Calibration_Confidence_Continuum_TCI_实验设计书_v1.0.md)
> **执行条例**：[docs/主控计划书_20260907_TCI_从当前状态到投稿_v1.0.md](docs/主控计划书_20260907_TCI_从当前状态到投稿_v1.0.md) · 卡状态只有 pending / in_progress / done / blocked / void
> **交接指令**：读 STATE.md + REPO_MIGRATION.md 锚点列 + 主控计划书 §4 当前卡，继续执行。

| 卡 | 状态 | 结论一行 | 证据/artifact | commit | 更新日期 |
|----|------|----------|----------------|--------|----------|
| C01 仓库地基 | done | 目录树与宪法§6逐目录一致；CLAIMS_REGISTRY 含 C1–C6+红队增补 N/V+红线；pip install -e . 绿 + pytest 1 passed | CLAIMS_REGISTRY.yaml; pyproject.toml; src/calibinfo/ | c23cedb | 2026-09-07 |
| C02 V1–V6 测试迁移 | done | 23/23 绿（~4s）；V1–V6+rank-deficient 阈值分层锁定；legacy 对照逐位一致（V1 median 1.0045/V4 rel err 6.44e-15/V2 2.23e-8 均与红队报告吻合）；legacy 无被任何实验 import；exp9d 结果已在本 clone 查证 | tests/unit/×8; legacy_redteam/ | 6e28ad8 | 2026-09-07 |
| C03 迁移矩阵执行 | done | B1–B3 逐行迁移（information/estimators/datasets 单源化）+B4 归档标注+AGENT_HANDOFF 重写；估计器与 legacy 对账 x max\|Δ\|=0.0；诊断稠密对照 5/5 绿；迁移表无空锚点行；48 tests 绿 | src/calibinfo/; docs/REPO_MIGRATION.md 锚点 | 528de66 | 2026-09-07 |
| C04 run manifest 骨架 | done | manifest 字段齐全（14 项）；config hash 对内容敏感/键序不敏感；run_ci.py+make_figures(--figure 1-9)+make_tables+reproduce_paper.sh 占位 | src/calibinfo/io/manifest.py; scripts/ | 本commit(C04/C05) | 2026-09-07 |
| C05 Gate A 冒烟 | done | CI01 pilot 端到端绿：config→run_ci→results/raw(manifest)→artifacts/frozen→fig1_draft.png；6/6 checks 过（dual ~1e-12、param ~1e-12、scale_inv 0.0、photometric shift 0.81=V5 同构复现） | configs/ci01/pilot.yaml; artifacts/frozen/ci01_pilot_summary.json | 本commit(C04/C05) | 2026-09-07 |
| C06 双路线 ΔF | done | 交付物随 C03 落地（schur.py 双路线+whitening）；验收测试含 m/q 欠定/临界/过定五区+Λ=0 秩亏，双路线 rel 8.4e-16 | tests/unit/test_information_modules.py | 528de66(交付)/本commit(验收扩区) | 2026-09-07 |
| C07 CI01 正式 | done | **Gate A 通过**：100/100 checks 绿（dual 2.21e-12/param 2.23e-12/scale_inv 1.72e-15/idem 8.6e-16/photometric shift 0.799）；首轮暴露 2 个 harness 缺陷（未归一 J 病态、λ 锚定退化带）如实记录并修复后整卡重跑 | artifacts/frozen/ci01_formal_summary.json; docs/memo_CI01_GateA_20260907.md | 本commit(C07) | 2026-09-07 |
| C08 理论手稿定稿 | done | Lemma1/Prop1/Prop2/Lemma2/λ⋆ 逐条按宪法§3 成文；四元组齐（假设→结论→新颖性权重→实验）；R1 引用谱系落实（HCRB/Harville/GUM）；无 novel theorem 措辞；λ⋆ 仅绝对信息语境 | paper/theory_note_v1.md | 本commit(C08) | 2026-09-07 |
| C09 scene 工厂 | done | 分层网格 54 场景（3几何×3albedo×3仰角×{1,3}灯）；gauge 恒等式 A·a=−B·c̄ 逐位成立（<1e-12）；μ_floor 接口落地 | src/calibinfo/datasets/synthetic.py; tests/unit/test_scene_factory.py | 本commit(C09-C11) | 2026-09-08 |
| C10 CI02 gauge/λ⋆ | done | 双 gate 绿：闭式 vs direct 良条件区 p50 3.1e-9/layered 0.132（消减地板分层 gate，cancel_margin=100 预注册）；λ⋆ 一阶预测 vs 闭式二分求根 median\|log10\|=0.00000（54/54 线性域，cond≤0.99）；实现层处置：灾难消减地板分层+读出升级二分求根（memo §2） | artifacts/frozen/ci02_formal_summary.json; docs/memo_CI02_20260907.md | 本commit(C09-C11) | 2026-09-08 |
| C11 retention+tracking | done | 双路线（白化平方根 vs generalized-eig）54 场景 max rel 1.55e-15（<1e-10）；Fig.4 三 panel 数据入 frozen：trace 反例（Δ≈4% 而 tracked ρ 0.001→0.87）、tracked heatmap（track_modes 链，N=3 简 1 步）、V6 角度 N=1 0.00°/N=3 25.9°；首轮 helper 缺陷（追错 F∞ 基）修正为 R(λ) 特征基 | experiments/ci02_gauge.py::fig4; tests 51 绿 | 本commit(C09-C11) | 2026-09-08 |
| C12 CI03 线性 MC | done | **Gate B 素材过**：18/18 绿（varratio 0.977–1.009/cov68 .68±.013/cov95 .95±.008/bias 闭式 0.96–1.01/condvar 0.98–1.03）；3 处实现事故如实入 memo（np.empty 混入→0.40 假象/条件方差归一化错位/bias SNR 可测性门） | artifacts/frozen/ci03_formal_summary.json; docs/memo_CI03_linear_20260908.md | 本commit(C12) | 2026-09-08 |
| C13 CI03 非线性包络 | done | **Gate C 过（有效域存在）**：B 臂解析 SH+ReLU 42 cells；<10% 域边界 flip≤3.0%/0.8%/2.2%（sphere/bumpy/term），误差随 flip 单调爬升；mask-flip 逐 cell 记录；A 臂 BlenderProc 按卡条款为增强项待 GPU 排期（非阻塞） | artifacts/frozen/ci03nl_nl_formal_summary.json; docs/memo_CI03_nl_20260908.md; fig6_draft.png | 本commit(C13) | 2026-09-08 |
| C14 OpenIllumination 适配 | done | 数据侦察→降采样裁决（全量 41GB/对象 >> 2GB 线 → thumbnail 层 200×273 + alpha 掩码，裁决入 manifest）；development 8 对象冻结（官方 material 8 族各 1）；loader GT 逐位复现 light_pos.npy + alpha 掩码与全分辨率覆盖率交叉校验 ±0.02；manifest 8×143 文件 sha256 入 frozen | src/calibinfo/datasets/openillumination.py; artifacts/frozen/openillumination_dev_manifest.json; tests/unit/test_openillumination.py | 本commit(C14) | 2026-09-08 |
| C15 Σ_c 生成器+pilot | done | corruption 生成器三类（强度/方向/联合，物理单位 φ 空间）；pilot 全链跑通：spearman 0.676 [0.590,0.757] > 0.5 预注册 pass；首轮教训×3（level=4 超线性化域→0.23，截档回收；δc=0 对照臂恒等解爆炸→残差 bootstrap；强度维被尺度 gauge 吸收→估计后对齐）；量级失配 10²（R-A/R-D 实证）→ 主 claim 冻结为秩相关+全局仿射尺度 | src/calibinfo/models/corruption.py; artifacts/frozen/ci04_pilot_summary.json; docs/memo_CI04_pilot_20260908.md | 本commit(C15) | 2026-09-08 |
| C16 CI04 正式 | done | **Gate D 通过**：test 11 对象（冻结清单）×6 档×20 seeds，spearman 0.728 [0.668,0.783]>0.5 且 CI 下界≫0；逐对象 11/11 正、带 [0.72,0.80] 稳定；greenhead 缺 com_masked 层→exclusion 如实记录；首轮误用 dev 清单发现后重跑（config 显式 objects 字段固化） | artifacts/frozen/ci04_formal_summary.json; docs/memo_CI04_formal_20260908.md | 本commit(C16) | 2026-09-08 |
| C17 DiLiGenT 适配 | done | loader 于 C03 迁移（真数据对账 0.0）；CI05 口径与 exp8R 系逐对象朗伯残差一致（ball 0.244/pot1 0.137/harvest 0.671）——迁移后口径未漂移 | src/calibinfo/datasets/diligent.py; artifacts/frozen/ci05_formal_summary.json | 本commit(C17/C18) | 2026-09-08 |
| C18 CI05 sanity | done | 10/10 弱模式存在（floor/med 2.7e-3–7.1e-3 ≪1）；λ⋆ 预测全对象有限（1.8e-3–5.9e-3）；taxonomy 10/10 校准传播与模型失配可比（量级对比，不混写因果）；OOM(24.7GB)+死代码两次如实记录 | experiments/ci05_sanity_robustness.py; docs/memo_CI05_20260908.md | 本commit(C17/C18) | 2026-09-08 |
| C19 ablation 附录 | done | 4 项 within-scene 效应量：白化vs raw λI 中位差 0.034（红队 V4 10% 同量级）；Λ 4× misspec 弱模式迹 −74.8%（4 场景 IQR 窄带）；灯数 1→3 最弱模式 log10 −6.2→−3.3（多灯混合抬升）；异方差口径 retention 差中位 0.008（V5 尺度纪律稳定） | experiments/ci05_ablation.py; artifacts/frozen/ci05abl_ablation_summary.json | 本commit(C19) | 2026-09-08 |
| C20 图表冻结 | done | Fig.1–9 全部由 make_figures.py --figure N 从 artifacts/frozen 一键生成（Fig.3/4/7 主图 + Fig.2/5/6/8/9 全 draft）；provenance（git SHA + artifact sha256 前 16）入 paper/provenance/×9；无 notebook 手工导出 | paper/figures/×8; paper/provenance/×9; scripts/figure_gens.py | 本commit(C20) | 2026-09-08 |
| C21 初稿组装 | done | 手稿 v0.8 全 11 节（宪法 §9 逐节）：Abstract 四句/三贡献；数字全部取 frozen（λ⋆ 0.00000/coverage ±0.013/Gate D 0.728 [0.668,0.783]/包络 flip≤3%）；§13 checklist 9 项全 PASS（禁词=否定声明豁免）；CLAIMS_REGISTRY C1–C6 证据指针补全（artifact+sha256+commit） | paper/manuscript_v0.8.md | 本commit(C21) | 2026-09-08 |
| C22 复现审计 | done | clean checkout（GitHub 克隆）抓到 .gitignore 静默排除 datasets 三文件（补交 8bcc139）→ 补后 clone: install 绿+pytest 54/54+CI01 重跑 100/100+Fig.3 重建；reproduce_paper.sh 一键链本仓库全通过；第三方异机 smoke 待投稿前异机执行（单机限制如实记） | scripts/reproduce_paper.sh; docs/memo_C22_20260908.md | 本commit(C22) | 2026-09-08 |
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
