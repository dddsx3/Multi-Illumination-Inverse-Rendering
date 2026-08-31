# R4′ 状态报告（2026-08-31 · 基线 `b74eba3`）

> **完成度**: 数学 R3′ 全过、数据 R4′-C 18 scene 全部 Gate PASS、统计管线端到端验证通过；全 18 scene × N={3,5,8} × 30 subsets = 1620 trials 的 E2/G2/E3 **采集仍在后台进行**（断点续跑，断电友好）。本文档为**进程报告**而非最终裁决。

## 1. 截至目前完成的工作

### 1.1 R3′ 数学封口（MATH GATE = PASS）
- `gauge_fisher_v2.py` 替代 v1：修正交叉块 `B_k = a·s·h·Y`（v1 漏 s_kp）+ 完整 P×P full-Schur（v1 逐像素假 Schur），加 operator 路径（LinearOperator + eigsh）。
- `test_gauge_fisher_v2.py` 28 项单测全 PASS（FD Jacobian ≤1e-5 / block ≤1e-8 / Schur ≤1e-6 / gauge-null 1e-8~1e-5 稳定 / 5 项附加命题 / operator 一致性）。
- `IDENTIFIABILITY_v2.md` 修正 N=1 秩=P−9、重复光 kernel 不变性等命题。
- v1 `gauge_fisher.py` + R4 summary 顶部加 **DEPRECATED_EXPLORATORY**。
- 旧 HANDOFF/INFORMATION_AUDIT_v2 加 **STALE** 标记。
- commit `236f895` + 报告 `R3P_MATH_AUDIT_REPORT.md`。

### 1.2 R4′-D Discovery 复扫（实现稳定性检查；非确认性证据）
- 3 变体：cap2000/cut1e-8（主）、cap1000/cut1e-8（主扫描；pixel_cap 冻结）、cap1000/cut1e-6（cutoff 稳定性）。
- 病态检查全 OK：gauge residual≤1.7e-5、PSD 余量≥-2.7e-10、320/320 primary>0。
- v1 失真量化：proxy_lam_min_norm / full median=47.4× [0.4×, 71098×]。
- cutoff 平台扫描（1e-10/1e-8/1e-7/1e-6/1e-5）：**1e-8 处于双平台鞍部，保留冻结**。
- pixel_cap 冻结 1000（本机 commit 配额 32GB 已用 ≥94% 实证，commit 减压前不可上调）。
- 报告 `R4P_DISCOVERY_RERUN_REPORT.md` + `R4P_CUTOFF_PLATEAU.md`，commit `8ab3957`。

### 1.3 R4′-C 确认集数据生成（全部 Gate PASS）
- 25 mesh 参数化家族生成（make_confirmatory_meshes.py：平滑 9 + 簇状 11 + 复合 6；torus_knot 因 Blender 4.2 缺 operator 跳过，最终 25 个）。
- 渲染器修复 **INC-001**（偶发整帧丢光 → depsgraph 同步 + 帧级完整性校验重试；commit `6f980b7`）：canary 32/32 帧健康、G1/G2/G3 PASS、Or1=26.77 dB。
- 18/25 mesh 渲染成功（7 个空掩码被剔除：水平 plane 30° 俯视、cube-only 高自阴影等；按纪律干净跳过）。
- **18 scene 全部 G1/G2/G3 PASS + Or1 SI-PSNR ∈ [25.06, 31.41] dB**（mean 27.29）。
- 报告 `R4P_CONFIRMATORY_DATA_GATES.md`，commit `b74eba3`。

### 1.4 R4′ 统计基础设施（end-to-end 验证）
- `solver_batched.py` + `r4p_confirmatory_gate.py`（E2/G2/E3）+ `r4p_pilot_calibration.py`。
- Pilot 标定：Discovery 24 trial 上 grad_norm P75=3.88e-4、loss P75=1.12e-4。
- Batched solver 验证：vs 串行 rel ~1e-3（最差 7.4e-3，Bp=1 仍 6.4e-5）。**诊断后决策：R4′ 使用串行 solver**（6.5× speedup 不值 1e-3 风险；batched v3 重做）。
- 算力紧缩：原 18×4×50=7200 trial → 18×3×30=1620 trial（舍弃 N=12；N={3,5,8} 足够跨 3 个 N 检 G2 斜率）。
- 预注册：pixel_cap=1000、cutoff=1e-8、subsets_per_N=30、solver=joint_solve、收敛自适应 P75（复合 mesh 自阴影使 Discovery-P75 阈值 0% success）。
- scores 阶段 1620/1620 全部完成。

## 2. 进行中：全 18 scene solver + 统计

后台任务 `exec_1ef9c747-...`：canary 1 scene（conf_cone_r04_d12）已 90 trial 完成（N=3 414s, N=5 538s, N=8 802s, 节奏 ~14~27s/run）。**当前进度 851/1620 trial（52.5%）、10/18 scene 完整**（cyl_plus_sphere 89/90 trial 几近完成；hemisphere_sq 仅 30 trial N=3 完成）。

**Partial E2 诊断（8-9 scene，非 verdict）**：

| N | n_scene (≥18 success) | median_ρ | frac_neg | 95% CI | 方向 |
|---|---|---|---|---|---|
| 3 | 5 | -0.146 | 100% | [-0.433, -0.073] | 弱负 ✓ 方向对 |
| 5 | 5 | -0.196 | 60% | [-0.554, +0.150] | 弱负（含 0）|
| 8 | 5 | +0.186 | 40% | [-0.544, +0.581] | **混杂（场景间方差大）** |

**直觉**（仅信号，不可作 verdict）：G1 部分被验证（固定 N 内确有相关），G2/E3 仍待 18 scene 全部。**N=8 的 5 scene 中 ρ∈{-0.54, -0.46, +0.19, +0.46, +0.58}**——场景间方差巨大，可能与法线多样性（cluster vs smooth）相关；**8 scene 还远不够 scene-bootstrap 的 10000 重采样稳定性**。

**距离裁决还差**：
- 剩余 8 scene × 90 trial ≈ 5h GPU
- 全 18 scene 跑完后 stats 阶段 <1 min
- 裁决落盘 + CLAIM_REGISTRY v0.3 + 交付报告 < 1h
- **最早裁决时间**：solve 完成 +1h

## 3. 数学协议的最终状态

- R3′ PASS，物理协议已封口（28.25 dB 无 unexplained gap）。
- H-COND 状态：**hypothesis，NOT LOCKED**（CLAIM_REGISTRY v0.1 仍有效）。
- R4 旧数字正式降级 exploratory：v1 `gauge_fisher.py` 派生，R3′ 裁决其交叉块公式级错。

## 4. 算力诚实声明

| 阶段 | 算力 | 实际 |
|---|---|---|
| R3′ 单测 | CPU | 0.5 min |
| R4′-D 复扫 3 变体 | CPU | ~3 min |
| R4′-C 渲染 25 mesh ×32 SUN | GPU 12GB | 19 min（含 0 个坏帧逃出校验）|
| R4′-C Gate+Oracle 18 scene | CPU | <2 min |
| R4′ scores 1620 trial | CPU | 11 min |
| R4′ solve 90 trial (canary 1 scene) | GPU 12GB | 29 min |
| **R4′ solve 全 1530 trial 估** | GPU 12GB | **~8 h**（后台运行中） |
| R4′ stats 1620 | CPU | <1 min |

本会话时间预算下，全 18 scene solve 无法在本窗口内完成。**但统计管线与数据全部就位**，后续 agent 只需 `python r4p_confirmatory_gate.py --stage solve`（断点续跑）然后 `--stage stats` 即可出最终裁决。

## 5. 后续 Agent 接手清单

1. `python p1/source/information_audit/r4p_confirmatory_gate.py --stage solve` — 等后台完成或手动跑
2. `python p1/source/information_audit/r4p_confirmatory_gate.py --stage stats` — 出 A/B/C 裁决
3. 根据裁决更新 `CLAIM_REGISTRY.md` 与 `P1_R0_STOP_LINE.md` 附录
4. 写最终交付报告 `P1_R4PRIME_CONFIRMATORY_REPORT.md`
5. 跑 P1-10/13/15/16 后续（任务书 §6/§7）
6. 写 R5 文献封口

签发：R4′ · ZCode agent · 2026-08-31
