# REPO_MIGRATION · 仓库迁移矩阵（2026-09-07 审计版）

> **依据**：《Calibration_Confidence_Continuum_TCI_实验设计书_v1.0》§6/§15
> **审计基线**：commit `58b3397`（main = origin/main）
> **用法**：本表是迁移的唯一清单。执行时逐行落位，每完成一行在本表"完成锚点"列填 commit SHA。
> 迁移原则：**公式单源化**（所有 Fisher/Schur/ΔF 只存在于 `src/calibinfo/information/`）；旧脚本只读归档；每行迁移必须绑定对应单元测试。

## A. 状态审计快照（迁移前事实）

| 事实 | 锚点 |
|---|---|
| 理论层已冻结（Lemma1/Prop1/Prop2/λ⋆/retention + 验证阶梯），三轮红队 R1–R4 修订入版 | 《红队报告三_20260907》+ 实验设计书 §0 |
| 新宪法 = 实验设计书 v1.0（CI01–CI05 + Gate A–E + CLAIMS_REGISTRY C1–C6） | 实验设计书全文 |
| 旧主线（GA-ISI/R4'/ICCV2027 四支柱）已被专家第六轮降格/替换 | 专家第六轮回复 + 记忆 |
| exp12v3 判 VOID（真值初始化等四缺陷）；exp8R v3 主诊断两处代数错误（headline 0/10正6/10负 被污染）；exp8S 变体 A = 正确参考实现 | 任务书增补 v3.3 §1 |
| exp8R v3 估计器侧合格（trf+解析稀疏 Jacobian、FD 4e-9、ALS 初值、多起点、像素守卫）；LAE 响应数据保留有效 | v3.3 §1.2 |
| exp9d（terminator 定位）预注册 ≥5× 富集未达成（λ 1.49×/μ 0.919×/ν 0.963×），分支已归档；**结果文件未在本 clone 发现，状态待确认** | 记忆 + 专家第六轮 |
| 云主机：CPU-only，磁盘余 7.1GB，numpy 2.4.6/scipy 1.18.1；DiLiGenT 未下载（936MB，可容纳）；OpenIllumination 未下载 | 本机实测 |

## B. 迁移矩阵

### B1 核心数学（单源化到 src/calibinfo/information/）

| 现有资产 | 目标 | 动作 | 必绑测试 | 完成锚点 |
|---|---|---|---|---|
| `redteam3_exp.py` 的 `DeltaF()`（lstsq 路径，已验证） | `src/calibinfo/information/schur.py::delta_f` | 重写（加 rank_policy 参数与 Λ=0 SVD 分支；返回 ΔF/M/rank diagnostics） | `test_rank_deficient_lambda0.py` | 528de66 |
| `redteam3_exp.py` V1 的 marginal 路线（σ²I+BΣ_cBᵀ 直接逆） | `schur.py::delta_f_marginal` | 移植；**只用于交叉验证**，禁止实验主路径调用 | `test_v1_covariance.py` 双路线互检 | 528de66 |
| （新建）W=Σ_y^{-1/2} 白化 | `information/whitening.py::whiten_system` | 新写；下游只接受白化系统 | CI01 异方差用例 | 528de66 |
| `redteam3_exp.py` V2 闭式 | `information/gauge.py::gauge_response` | 移植；返回 exact/small-λ slope/saturation | `test_v2_gauge_closed_form.py` | 528de66 |
| `redteam3_exp.py` V3 白化（**注意：正确形式是 F∞^{-1/2}=diag(1/s)，不是 1/√s**） | `information/retention.py::retention_spectrum` | 重写；限制到 range(F∞)；返回 ρ/basis/cond 诊断 | `test_v3_retention_bounds.py` | 528de66 |
| （新建）模式追踪 | `information/mode_tracking.py::track_modes` | 新写（\|V_prevᵀV_curr\| assignment + 简并子空间处理） | `test_v6_mode_tracking.py`（N=1 必须 0°） | 528de66 |

### B2 估计器（src/calibinfo/estimators/）

| 现有资产 | 目标 | 动作 | 必绑测试 | 完成锚点 |
|---|---|---|---|---|
| `critical_experiments/exp8r_diligent_discrimination_v3.py` 估计器侧（LM+解析稀疏 Jacobian+ALS 初值+多起点+像素守卫） | `estimators/joint_map.py` + `estimators/gauss_newton.py` | 移植合格部分；**禁止**移植其 `diagnose_trace()`（两处代数错误） | FD-vs-解析 Jacobian rel<1e-6 回归 | 528de66 |
| `critical_experiments/exp8s_oracle_schur_audit.py` 变体 A（精确逐像素 ρ-Schur + S_aa 联合 α 边缘化） | `estimators/diagnostics.py`（或 information 层） | **移植为诊断参考实现**（v3.3 已核验为正确代数） | 随机小规模稠密对照 rel<1e-10（红线 #8） | 528de66 |
| exp12v3 的教训固化 | 估计器入口 assert | 数据驱动初始化；`assert not np.allclose(init, gt)`；GT 只进评分（红线 #11） | init 断言测试 | 528de66 |
| exp14 的 Λ 匹配诊断机制（估计器实际先验 Λ_z 进 CRB 迹） | `estimators/diagnostics.py` | 按需移植（CI03 用） | 与 exp14 json 数字对账用例 | 推迟→C12（矩阵原文：按需移植） |

### B3 数据层（src/calibinfo/datasets/）

| 现有资产 | 目标 | 动作 | 必绑测试 | 完成锚点 |
|---|---|---|---|---|
| `eval_diligent/` + `evaluate_diligent.py` | `datasets/diligent.py` | 收敛为单一 loader + manifest 接入 | 小样本 checksum/形状断言 | 528de66 |
| （新建）OpenIllumination loader | `datasets/openillumination.py` | 新写：HF `OpenIllumination/OpenIllumination`（CC BY 4.0）OLAT 子集；GT 光照/掩码解析；对象清单 manifest | 已知对象光照 GT 数值 sanity | 推迟→C14（新建） |
| `p1/source/generation/render_multilight.py`（BlenderProc，Windows 3.10 环境） | **留在原位**，Windows 侧调用 | 只读引用；CI03 非线性增强臂时包一层适配器 | oracle_gate 28.25dB 基线不回退 | 原位（C13 时包适配器） |
| `make_split_manifest.py` / `split_manifest.py` / `splits/` | `io/manifest.py` 统一 | 并入 run_manifest 体系 | manifest round-trip 测试 | 本commit(C04) |

### B4 归档（只读，禁止再修改）

| 资产 | 去向 | 说明 |
|---|---|---|
| `redteam_exp.py`、`redteam2_exp.py`、`redteam3_exp.py`（逆渲染工作目录） | `legacy_redteam/` | 原样归档 + README 注明非主实验入口 |
| `critical_experiments/` exp1–exp14 全部（含 VOID 的 exp12v3、被污染判定的 exp8R v3） | 原位只读 + `CLOSURE_20260906.md` 增补指针 | 诚实负结果是 CI05 failure taxonomy 素材 |
| `p1/protocol/CLAIM_REGISTRY.md`（GA-ISI 三句话宪法） | 头部加 `[SUPERSEDED 20260907 → CLAIMS_REGISTRY.yaml]` | 历史保留 |
| `docs/ICCV2027_SKELETON_v1.md`（若存在） | 头部加 SUPERSEDED 标注 | venue 已冻结 TCI |
| `任务书 v3.0–v3.3`（workspace，不入仓库） | 红线 8–11 + 检查表第 5 行 + 统计纪律**继续有效**；卡体系废止 | 处置裁决见主计划书 §1 |
| `AGENT_HANDOFF.md` | 重写为 calibration-continuum 版（C03 卡交付） | 旧版 §7 待办（R4'/P1-13）作废 |

### B5 不迁移 / 独立处置

| 资产 | 处置 | 理由 |
|---|---|---|
| exp12v4 / exp8R v3.1 云端重跑（v3.3 卡 R2/S2 主体） | **不执行**（条件复活条款见主计划书 §1） | 证据目的（旧支柱③判别力）在新 CLAIMS_REGISTRY 无位置 |
| 卡 OPT-1（VarPro 加速） | 条件卡：仅当 CI03/CI04 估计器成瓶颈时启用；五判据保持冻结 | 原目的（加速已取消的云跑）消失 |
| FusionUNet / 训练管线（`fusion_unet.py`、`trainer.py` 等） | 原位冻结，不进入 CI 链 | 专家第六轮已降 network 为 case study，主线无训练 |
| 云迁移脚本（`run_exp8r_cloud*.sh`、`cloud_migration/`） | 保留可运行，标记 legacy | 卡 S2 取消后暂无调用方；云跑经验（断点续跑/线程钉扎）是 C17 的参考 |
