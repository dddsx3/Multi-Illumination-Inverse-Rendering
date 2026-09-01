# R5-B′ · A10 启动操作表（傻瓜式）

> **作者**: ZCode agent · 2026-09-01
> **适用算力**: 1× NVIDIA A10 24GB / 20 vCPU / 116 GB RAM / Ubuntu 22.04
> **免费额度**: 24 机时（抵扣因子 3.3）= 实际 ~7.27 GPU-小时等效
> **本任务**: P1-A full + Task G（选项 A）= ~21.5 机时
> **GitHub 仓库**: https://github.com/dddsx3/Multi-Illumination-Inverse-Rendering
> **基线 commit**: 9796884 (R4§ sprint 收官)

---

## 0. 一分钟总览

你要做的全部事情，按顺序：

```
1. 启动 A10 实例（Ubuntu 22.04 image）
2. SSH 进实例
3. git clone dddsx3/Multi-Illumination-Inverse-Rendering
4. cd 进去
5. bash scripts/launcher/01_a10_env_setup.sh        ← 5-10 min
6. bash scripts/launcher/02_a10_run_all.sh          ← 6.5 h（看提示输入 'y' commit）
7. git push (脚本里已经做了, 但要再确认)
8. 关机
```

如果你中间 OOM / 被抢占，用 `02b_a10_per_scene.sh` 单 cell 重跑（详见 §6）。

---

## 1. 启动 A10 实例

### 1.1 选择镜像

| 项 | 推荐值 |
|---|---|
| OS | Ubuntu 22.04 LTS |
| GPU | 1× NVIDIA A10 24GB |
| vCPU | 20（实例规格自带）|
| RAM | 116 GB（实例规格自带）|
| Storage | ≥200 GB（数据 + 中间产物 + LFS）|
| Python | 3.10+（系统自带）|
| 预装 | NVIDIA driver（CUDA 12.x）|

> 镜像里**不需要**预装 PyTorch。01_env_setup.sh 会自己装。

### 1.2 SSH 进去

```bash
ssh -i <your-key>.pem ubuntu@<instance-ip>
```

进入后第一个命令：

```bash
nvidia-smi
```

应能看到 **NVIDIA A10** 24 GB。**如果看不到**，联系平台客服确认镜像是否含 GPU 驱动。

---

## 2. 一次性准备（首次运行约 5-10 分钟）

### 2.1 克隆仓库（如果 /workspace 不是本仓库）

```bash
cd ~
git clone https://github.com/dddsx3/Multi-Illumination-Inverse-Rendering.git
cd Multi-Illumination-Inverse-Rendering
```

> **如果已经在 `/workspace` 且是本仓库（看到 `(main)` 在 prompt 里）**：
> 跳过 2.1，直接 `cd /workspace` 即可。

### 2.2 安装 git-lfs（LFS 拉取 data_sun_confirmatory/*.npy）

```bash
# Ubuntu 22.04 通常需要 apt 装
# 注意包名是 git-lfs (git-l-ess, 末尾是 ess), 不是 git-lf
sudo apt-get update
sudo apt-get install -y git-lfs
```

### 2.3 拉取 LFS 数据

```bash
cd ~/Multi-Illumination-Inverse-Rendering
git lfs install                  # 一次性
git lfs pull                     # 拉数据, 约 300 MB
```

验证：

```bash
ls p1/calibration_set/data_sun_confirmatory/ | wc -l   # 应显示 19
```

如果显示不是 19，看 §7 故障处理。

### 2.4 跑环境脚本

```bash
bash scripts/launcher/01_a10_env_setup.sh
```

脚本会做：
1. sanity check（uname/nproc/free/df）
2. 验证 git status 干净
3. 验证 LFS 数据齐全
4. 创建 `.venv_a10/` 虚拟环境（Python 3.10+）
5. 装 `torch + torchvision`（cu128 wheel，A10 SM86 兼容）
6. 装 `numpy + scipy + pandas`
7. CUDA sanity（小 matmul）
8. 输出 OpenBLAS thread 配置（OMP_NUM_THREADS=10）

**期望输出末行**：
```
[01 env] DONE — ready for 02_a10_run_all.sh
venv: /home/ubuntu/Multi-Illumination-Inverse-Rendering/.venv_a10
data: ... (19 scenes)
GPU: NVIDIA A10
next: bash scripts/launcher/02_a10_run_all.sh
```

**耗时**: 5-10 分钟（pip install 慢）

---

## 3. 主任务跑（约 6.5 小时）

```bash
bash scripts/launcher/02_a10_run_all.sh
```

### 3.1 脚本会做什么（不需要交互）

| Step | 动作 | wall-clock |
|---|---|---|
| 1 | sanity + GPU 检查 | 5 s |
| 2 | 进入 venv + 设 BLAS thread=10 | instant |
| 3 | **后台启动 Task G**（240 run × ~2.5s = 10 min）| (并行) |
| 4 | **P1-A GSIQ**（6 scenes × 13,920 subsets × 2 score = 83,520 calls × 0.25 s）| **~5.8 h** |
| 5 | **P1-A solver arm**（360 runs × ~2.5 s）| **~0.3 h** |
| 6 | 等 Task G 后台完成 | (并行) |
| 7 | 验证 6 个产物文件存在 | instant |
| 8 | 解析 gate verdict | instant |
| 9 | **询问** `Commit + push results? (y/N)`，输入 `y` 自动 commit + push | 30 s |

**期望总 wall-clock**: **~6.5 h**（Task G 与 GSIQ 完全并行；GSIQ CPU-bound, Task G GPU-bound）

### 3.2 监控

开**第二个 SSH session**（不要 kill 主 session）：

```bash
# 看 P1-A 进度
tail -f r5/r5_p1_a_full_run.log

# 看 GPU 占用（Task G 跑时会有 CUDA process）
nvidia-smi

# 看 CSV 已写多少行
wc -l r5/r5_p1_albedo_ablation.csv

# 看 Task G 进度
tail -f r4pp/task_g_run.log
```

### 3.3 进度判断

- **正常进度**: P1-A GSIQ 每行约 0.25-0.5 s（BLAS 调度可能慢一些）
- **83,520 行** × 0.3 s/call ≈ **7 h upper bound**；实际 **~5.8 h**
- Task G 在 GSIQ 启动后约 10 min 完成（在另一个 tail 里能看到 `++n` 进度）

### 3.4 跑完后

脚本会问 `Commit + push results? (y/N)`。输入 `y`，脚本自动 commit 并 push 到 `origin HEAD`。

如果脚本意外退出（OOM / 抢占），**结果已经写在 CSV 里**，下次 SSH 进来 `git status` 应能看到未 commit 的 `r5/r5_p1_*.csv`，手动 commit + push 即可。

---

## 4. 预期产物清单

### 4.1 必须存在（脚本会验证）

```
r5/r5_p1_albedo_ablation.csv               # 83,520+ 数据行 (6 scenes × 13,920)
r5/r5_p1_albedo_ablation_ranking.csv       # 12 cell ranking metrics
r5/r5_p1_albedo_ablation_outliers.csv      # boundary outliers（通常 <100 行）
r5/r5_p1_albedo_ablation_gate.md           # PASS-A / CONDITIONAL / FAIL-A
r5/r5_p1_albedo_ablation_selection.csv     # solver arm 输出 (~720 行)
r5/r5_p1_a_full_run.log                    # 完整 stdout 日志
r4pp/07_local_vs_global_init.csv           # Task G 输出 (240 run)
r4pp/task_g_run.log                        # Task G 日志
```

### 4.2 数值成功标准

**P1-A**（看 `gate.md`）：
- `median rho >= 0.95` 且 `median top10 overlap >= 0.80` → **PASS-A**
- 不满足 → CONDITIONAL / FAIL-A，需进一步分析

**Task G**（看 `r4pp/07_local_vs_global_init.csv`，自己跑一段分析）：

```bash
python3 -c "
import pandas as pd, numpy as np
df = pd.read_csv('r4pp/07_local_vs_global_init.csv')
for mode in ['global', 'oracle_local']:
    sub = df[df['init_mode'] == mode]
    I = sub['information'].values
    logE = np.log(sub['reconstruction_error'].values + 1e-12)
    beta = np.polyfit(I, logE, 1)[0]
    print(f'{mode}: beta = {beta:.4f}, n_runs = {len(sub)}')
"
```

期望：
- `global: beta < 0`（信息效应在 global init 下成立）
- `oracle_local: beta < 0` → Case 1（intrinsic identifiability，最佳）
- `oracle_local: beta >= 0` → Case 2（仅 practical optimization recoverability）

---

## 5. 完整流程 checklist

```
[ ] 启动 A10 实例（Ubuntu 22.04，24 GB GPU）
[ ] SSH 进实例
[ ] sudo apt-get install -y git-lfs
[ ] git clone https://github.com/dddsx3/Multi-Illumination-Inverse-Rendering.git
[ ] cd Multi-Illumination-Inverse-Rendering
[ ] git lfs install && git lfs pull
[ ] ls p1/calibration_set/data_sun_confirmatory/ | wc -l   # 应是 19
[ ] nvidia-smi   # 应看到 A10
[ ] bash scripts/launcher/01_a10_env_setup.sh
[ ] bash scripts/launcher/02_a10_run_all.sh
[ ] 看 6.5 h...（第二个 SSH session 用 tail -f 监控）
[ ] 脚本问 "Commit + push?" 时输 y
[ ] 验证 r5/r5_p1_*.csv + r4pp/07_local_vs_global_init.csv 都在
[ ] 关机
[ ] 本机 git pull 拉结果
[ ] 跑 §4.2 的 P1-A gate 解析 + Task G beta 解析
[ ] 邮件 / Slack 通知项目 owner: "P1-A + Task G done, gate verdict = XXX"
```

---

## 6. 故障处理

### 6.1 OOM 中途退出

A10 24 GB 应该不会 OOM（P1-A GSIQ 是 CPU-only）。但如果意外 OOM：

**症状**: 看到 `MemoryError` 或 CUDA OOM

**解决**: 跑单 cell 重跑：

```bash
# 例: cube_axis N=3 OOM 了, 重跑
bash scripts/launcher/02b_a10_per_scene.sh conf_cube_axis 3
```

`02b` 会自动 append 到现有 CSV（不覆盖）。然后手动跑一次 `02_a10_run_all.sh` 重新生成 ranking / outliers / selection / gate。

### 6.2 Spot / 抢占中断

A10 实例如果是 spot，平台可能随时回收。**R5-B' 脚本都支持 incremental**：

| 脚本 | 增量策略 |
|---|---|
| `r5_p1_albedo_ablation.py` | `csv_mode='a'` 自动 append (scene, N, subset_id) 已存在行被覆盖 |
| `r4pp_local_vs_global.py` | 自带 incremental：跳过 `(scene, N, subset, init_mode)` 已 done |

被抢占后：

```bash
# SSH 重新进（如果实例被关, 需要重启）
bash scripts/launcher/02_a10_run_all.sh   # 自动跳过已完成部分
```

### 6.3 GSIQ 太慢

正常 5.8 h 完成。如果 6.5 h 还没完成 80%，检查：

```bash
# 1. CPU 占用
top -bn 1 | head -20

# 2. 看 OMP_NUM_THREADS 是否生效
echo $OMP_NUM_THREADS    # 应 10

# 3. 看 BLAS 是否真用了多核
# Linux: lscpu | grep "NUMA\|Core"
# top 单进程应显示 ~1000% CPU（10 核）
```

如果 CPU 只用单核：

```bash
# 重设环境变量（在 venv 内）
source .venv_a10/bin/activate
export OMP_NUM_THREADS=10
export OPENBLAS_NUM_THREADS=10
# 重跑
```

### 6.4 LFS 数据缺失

**症状**: `ls p1/calibration_set/data_sun_confirmatory/ | wc -l` 显示 < 19

**解决**:

```bash
git lfs pull
# 或强制重拉
git lfs fetch --all
git lfs checkout
```

### 6.5 torch 安装失败

**症状**: `pip install torch` 报 SSL 或网络错误

**解决**:

```bash
# 用清华镜像
pip install torch torchvision --index-url https://pypi.tuna.tsinghua.edu.cn/simple
# 或 aliyun
pip install torch torchvision --index-url https://mirrors.aliyun.com/pytorch/
```

### 6.6 Git push 失败

**症状**: `git push` 报权限 / auth 错误

```bash
# 1. 检查 remote
git remote -v
# 应是 https://github.com/dddsx3/Multi-Illumination-Inverse-Rendering.git

# 2. 检查 token
git config --get user.name
git config --get user.email

# 3. 手动 push (会要求 username + Personal Access Token)
git push origin HEAD
```

---

## 7. 不要做的事

- ❌ **不要**手动修改 `gauge_fisher_v2.py`（任务书 §25 KILL 条件：换 metric 救 correlation）
- ❌ **不要**调 `spec_cutoff=1e-8` / `cutoff=1e-8`（任务书 §5 冻结值）
- ❌ **不要**为了赶时间跳过 `--solver`（solver arm 是 PASS-A Gate 的第二支柱）
- ❌ **不要**在 P1-A full 未完成前跑 P1-B / P2（任务书 §27 严格顺序）
- ❌ **不要**用 `git push --force` 覆盖已 push 的 commit
- ❌ **不要**让 A10 实例跑超过 24 机时（超出免费额度）

---

## 8. 跑完后通知项目 owner

需要给项目 owner 发一条消息，包含：

```
R5-B′ P1-A full + Task G 完成 (A10 实例).

产物: 
  - r5/r5_p1_albedo_ablation.csv (83,520 rows @ P=2000)
  - r5/r5_p1_albedo_ablation_gate.md (Gate verdict = XXX)
  - r4pp/07_local_vs_global_init.csv (240 run)

数值:
  - median rho = Y.YYYY
  - median top10 overlap = Y.YY
  - Task G beta_global = X.XXXX, beta_oracle_local = X.XXXX

总 wall-clock: X.X h / 抵扣机时 X.X / 24 机时

下一步 (待 owner 决策):
  - 如果 PASS-A + Task G Case 1: 申请 P1-B 算力 (~$30 spot, 9 h wall-clock)
  - 如果 PASS-A + Task G Case 2/3: 重新评估 claim
  - 如果 FAIL-A: 转 identifiability diagnostic paper
```

---

## 9. 关键文件路径

| 用途 | 路径 |
|---|---|
| 数据 | `p1/calibration_set/data_sun_confirmatory/` |
| M1 / GSIQ 实现 | `p1/source/information_audit/gauge_fisher_v2.py` |
| P1-A 评分脚本 | `p1/source/information_audit/r5_p1_albedo_ablation.py` |
| Task G 脚本（R4§ 已就绪）| `p1/source/information_audit/r4pp_local_vs_global.py` |
| 数学文档 | `p1/protocol/IDENTIFIABILITY_v3.md` |
| 论文 Claim | `p1/protocol/CLAIM_REGISTRY.md` (v0.4) |
| 算力清点 | `r5/R5_H100_RESOURCE_INVENTORY.md`, `r5/R5_RESOURCE_PRECISION_MATRIX.md` |
| A10 校准 | `r5/R5_A10_FIT_CHECK.md` |
| P1-A smoke 报告 | `r5/r5_p1_a_closure.md` |

---

## 10. 附录：完整时间线（A10 wall-clock 推算）

```
0:00   start
0:05   clone + git-lfs pull
0:15   apt install git-lfs
0:30   apt install + pip install torch + packages
1:00   venv ready, env_setup.sh done
1:05   run_all.sh starts; Task G launched in background
1:15   P1-A GSIQ N=3 scene 1 begins (~50 min)
2:05   P1-A GSIQ N=3 scene 2 (~50 min)
2:55   P1-A GSIQ N=3 scene 3 (~50 min)
3:45   P1-A GSIQ N=3 scene 4 (~50 min)
4:35   P1-A GSIQ N=3 scene 5 (~50 min)
5:25   P1-A GSIQ N=3 scene 6 (~50 min)
6:15   P1-A GSIQ N=5 scene 1 (~17 min, faster because sampled 2000 not full 4960)
6:32   ... scene 2-6, ~17 min each
7:30   P1-A GSIQ done; solver arm starts (360 run × 2.5s ≈ 15 min)
7:45   solver arm done; Task G already done at 1:15 (10 min)
7:45   gate verdict + commit prompt
7:46   user types 'y' → git push
7:50   shutdown A10 instance
```

总 wall-clock ≈ **6.5 h**，抵扣机时 ≈ **21.5 / 24**，剩 **2.5 机时**做缓冲。

---

*作者: ZCode agent · 2026-09-01 · A10 24机时免费额度 · 选项 A (P1-A full + Task G)*
*本文档所有命令可直接 copy-paste；不需要任何编程或项目知识*