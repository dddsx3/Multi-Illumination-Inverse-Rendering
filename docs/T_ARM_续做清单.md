# T-ARM 续做清单（2026-08-28 23:00 之后）

> **目的**：handoff §5.1 TODO 1/2 + 中期审计 v2 任务卡 T2.5 残差消融 + 判别实验 (b)(c) 的下一位 agent 接力模板。
> **当前状态**：seed 42 续跑中（已到 epoch 44/100，温度节律下预计 3-4h 内 100/100）。
> **决策依据**：决策 1-5 + INC-0010 v3 链注 + INC-0012 物理约束补建 + N 敏感性双轨 PASS。

---

## 1. 接力状态

### 1.1 已落地（无需重复）
- ✅ 5 项决策 1-5 全部落地
- ✅ INC-0011（F-resA 单 run 反转）已创建
- ✅ INC-0012（物理约束补建）已 PASS，违规 0%
- ✅ T-PHYS 3 epoch 冒烟（albedo10/albedo1）已通过
- ✅ T-NCURVE 双轨产物（synth v3 + DiLiGenT）已落盘
- ✅ INC-0010 §8 v3 链注段 + t2_2_design.md §10 + t2_5_n_sensitivity_report.md
- ✅ A6 §9 决策 1-5 增补段 + Phase 2 结论草稿
- ✅ T-ARM A3-bis seed 42 续跑启动（已到 epoch 44）

### 1.2 接力待做
- T-ARM.1：续跑 A3-bis seed 42 → 100（**进行中**）
- T-ARM.2：启动 A3-bis seed 123 / 2024（seed 42 完成后顺序启动）
- T-ARM.3：补训 p2_t23_f_physcon（T2.3 G2.3 门禁）
- T-ARM.4：补训 p2_t25_f_resC（T2.5 G2.5 门禁）
- T-ARM.5：实现 F-noFiLM 变体开关 + 补训（判别实验 b）
- T-ARM.6：实现 F-lowSmooth 变体开关 + 补训（判别实验 c）
- T-DOC.1：A6 13 项指标重写为 3-seed mean ± std 综合版（A3-bis 完成后）
- T-MATRIX.1：6 变体完整对比矩阵（全部 T-ARM 完成后）

---

## 2. 启动命令模板

### 2.1 A3-bis seed 42 续跑（当前已启动，等完）
```bash
cd "D:/Multi-Illumination Inverse Rendering/repo"
. ./_env.sh   # THERMAL_RESUME=75

# run_arms.py 临时改 v2 → v2_seed42
sed -i 's/"p2_t22_f_n5rgb",$/"p2_t22_f_n5rgb_v2_seed42",/' run_arms.py

python -u run_arms.py --data_root D:/data/synthetic_v3 --budget-hours 24 \
    --max-lanes 1 --only p2_t22_f_n5rgb_v2_seed42 \
    --amp-dtype bf16 --skip-package > _orchestrator_seed42_resume_log.txt 2>&1

# 完跑后恢复 run_arms.py 原名
sed -i 's/"p2_t22_f_n5rgb_v2_seed42",$/"p2_t22_f_n5rgb",/' run_arms.py
```

### 2.2 A3-bis seed 123 / 2024（seed 42 完成后顺序启动）
```bash
# seed 123
mkdir -p checkpoints/p2_t22_f_n5rgb_v2_seed123
sed -i 's/"p2_t22_f_n5rgb",$/"p2_t22_f_n5rgb_v2_seed123",/' run_arms.py
python -u run_arms.py --data_root D:/data/synthetic_v3 --budget-hours 8 \
    --max-lanes 1 --only p2_t22_f_n5rgb_v2_seed123 \
    --amp-dtype bf16 --skip-package --seed 123 > _orchestrator_seed123_log.txt 2>&1
sed -i 's/"p2_t22_f_n5rgb_v2_seed123",$/"p2_t22_f_n5rgb",/' run_arms.py

# seed 2024
mkdir -p checkpoints/p2_t22_f_n5rgb_v2_seed2024
sed -i 's/"p2_t22_f_n5rgb",$/"p2_t22_f_n5rgb_v2_seed2024",/' run_arms.py
python -u run_arms.py --data_root D:/data/synthetic_v3 --budget-hours 8 \
    --max-lanes 1 --only p2_t22_f_n5rgb_v2_seed2024 \
    --amp-dtype bf16 --skip-package --seed 2024 > _orchestrator_seed2024_log.txt 2>&1
sed -i 's/"p2_t22_f_n5rgb_v2_seed2024",$/"p2_t22_f_n5rgb",/' run_arms.py
```

### 2.3 补训 p2_t23_f_physcon（T2.3 G2.3 门禁）
```bash
# 占位目录已建：checkpoints/p2_t23_f_physcon/
python -u run_arms.py --data_root D:/data/synthetic_v3 --budget-hours 24 \
    --max-lanes 1 --only p2_t23_f_physcon \
    --amp-dtype bf16 --skip-package > _orchestrator_physcon_log.txt 2>&1
# 训练旗标：--model fusion --modality gray --sh_constraint softplus
```

### 2.4 补训 p2_t25_f_resC（T2.5 残差容量消融）
```bash
python -u run_arms.py --data_root D:/data/synthetic_v3 --budget-hours 24 \
    --max-lanes 1 --only p2_t25_f_resC \
    --amp-dtype bf16 --skip-package > _orchestrator_resC_log.txt 2>&1
# 训练旗标：--model fusion --modality gray --res_hidden 32
```

### 2.5 判别实验 (b) F-noFiLM
```bash
# 步骤 1: 在 fusion_unet.py 加 --disable_film 旗标（film_gamma=1, film_beta=0）
# 步骤 2: main.py 加 argparse
# 步骤 3: run_arms.py ARMS 列表新增
# ARMS.append(("p2_t25_f_noFiLM",
#              ["--model", "fusion", "--modality", "gray", "--disable_film"],
#              ["--model", "fusion"],
#              "G2.5 判别实验 (b) F-noFiLM：FiLM 调制影响验证"))

python -u run_arms.py --data_root D:/data/synthetic_v3 --budget-hours 8 \
    --max-lanes 1 --only p2_t25_f_noFiLM \
    --amp-dtype bf16 --skip-package > _orchestrator_noFiLM_log.txt 2>&1
```

### 2.6 判别实验 (c) F-lowSmooth
```bash
# 步骤 1: main.py 加 --albedo_smooth_stage1 旗标
# 步骤 2: run_arms.py ARMS 列表新增
# ARMS.append(("p2_t25_f_lowSmooth",
#              ["--model", "fusion", "--modality", "gray", "--albedo_smooth_stage1", "1.0"],
#              ["--model", "fusion"],
#              "G2.5 判别实验 (c) F-lowSmooth：albedo_smooth=1.0 权重验证"))

python -u run_arms.py --data_root D:/data/synthetic_v3 --budget-hours 8 \
    --max-lanes 1 --only p2_t25_f_lowSmooth \
    --amp-dtype bf16 --skip-package > _orchestrator_lowSmooth_log.txt 2>&1
```

---

## 3. 时间预算（24h + 24h 编排）

| 阶段 | 任务 | 预计耗时 | 串行依赖 |
|---|---|---|---|
| 阶段 1 | A3-bis seed 42 续跑（67 epoch）| 3-4h | – |
| 阶段 2 | A3-bis seed 123（100 epoch）| 5.5h | 阶段 1 完 |
| 阶段 3 | A3-bis seed 2024（100 epoch）| 5.5h | 阶段 2 完 |
| 阶段 4 | T2.3 physcon（100 epoch）| 5.5h | 阶段 3 完 |
| 阶段 5 | T2.5 resC（100 epoch）| 5.5h | 阶段 4 完 |
| 阶段 6 | 判别实验 noFiLM（100 epoch）| 5.5h | 阶段 5 完 |
| 阶段 7 | 判别实验 lowSmooth（100 epoch）| 5.5h | 阶段 6 完 |
| **总耗时** | **37.5-40h**（温度节律下）| | |

**降级方案**（如时间紧）：跳过 stage 5/6/7 中部分任务，按优先级 T2.3 > 判别实验 > A3-bis > 残差变体。

---

## 4. 文档交付顺序

| 顺序 | 任务 | 触发条件 |
|---|---|---|
| 1 | A6 13 项指标 3-seed 综合版重写 | A3-bis 3 seed 全完 |
| 2 | t2_5 判别实验 culprit 判定表 | 判别实验 (a)(b)(c) 全完 |
| 3 | 6 变体完整对比矩阵 | 6 变体全训完 |
| 4 | Phase 2 验收报告 | G2.1-G2.7 全 PASS |

---

## 5. 注意事项

1. **温度墙持续撞停**：保留 `interrupt_state.pth` + 使用 `THERMAL_RESUME=75`/`THERMAL_LIMIT=80`/`THERMAL_PACE=2.0` 固化值
2. **run_arms.py 临时改名**：每个 seed / 变体启动前 `sed` 改 `:80` 改 ARMS 名称，完跑后 `sed` 改回
3. **D12 冒烟产物物理隔离**：`_SMOKE_*` 前缀或 `smoke_artifacts/` 目录，禁入对比矩阵
4. **D11 公平对比纪律**：6 变体对比时每行标注协议（zero-shot / 同协议 / 重训）
5. **D13 残差健康监测**：residual_magnitude + physical_ratio 入评估与训练日志
6. **判别实验结论必须单一 culprit**：若多个变体都"反照率恢复"，用叠加实验（noFiLM+lowSmooth）判定主次
7. **3-seed 综合**才允许做"std < 5% 噪声预算"的最终判定（中期审计 v2 §2-P2 软约束）

---

## 6. 引用

- handoff 文档：`repo/HANDOFF_20260828.md` §5.1 TODO 1/2 + §6 未授权修改清单
- 中期审计 v2：`文档类材料/Phase2中期审计-问题发现与改进方向 (1).md` §2-P2 判别实验表
- 顶层设计 v2.1 任务卡：`文档类材料/顶层设计-任务工作指导书 (1).md` T2.5 + T2.6 + T2.7
- 独立审计规程 v2.1：`独立审计执行/顶层设计-独立审计与门槛验收规程 (1).md` §4 放行规则
- INC-0010 v3 链注：`repo/docs/incidents/INC-0010_数学底层三重偏差与编排器续跑冲突.md` §8
- INC-0011：`repo/docs/incidents/INC-0011_F-resA单run反转与seed噪声影响.md`
- INC-0012：`repo/docs/incidents/INC-0012_物理约束补建与albedo_smooth二次风险验证.md`
- T-NCURVE 报告：`repo/docs/design/t2_5_n_sensitivity_report.md`
- Phase 2 结论草稿：`repo/docs/Phase2_结论草稿.md`

---

*本清单由 2026-08-28 23:00 阶段决策落地，作为下一位 agent 接力模板。*
