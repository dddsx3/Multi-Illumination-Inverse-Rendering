# R5-B′ · A10 启动操作表（v2 · 0 知识版）

> **作者**: ZCode agent · 2026-09-01 (v2)
> **适用算力**: 1× NVIDIA A10 24GB / 20 vCPU / 116 GB RAM / Ubuntu 22.04
> **免费额度**: 24 机时（抵扣因子 3.3）= 实际 ~7.27 GPU-小时等效
> **本任务**: P1-A full + Task G（选项 A）= ~21.5 机时
> **GitHub 仓库**: https://github.com/dddsx3/Multi-Illumination-Inverse-Rendering
> **基线 commit**: `ed0a068` (本版本包含全部修复)

---

## 0. 一句话总览

```
你要做的全部事情 (按顺序):
  1. SSH 进 A10 实例
  2. 确认仓库路径 (已经在 /workspace 或类似路径)
  3. bash scripts/launcher/01_a10_env_setup.sh    # 5-10 min
  4. bash scripts/launcher/02_a10_run_all.sh      # 6.5 h
  5. 看提示输 y → 自动 commit + push
  6. 关机
```

**操作员不需要任何 Python / 项目知识**。所有命令可直接 copy-paste。

---

## 1. 启动 A10 实例

### 1.1 选择镜像

| 项 | 推荐值 |
|---|---|
| OS | **Ubuntu 22.04 LTS** |
| GPU | **1× NVIDIA A10 24GB** |
| vCPU | 20（实例规格自带）|
| RAM | 116 GB（实例规格自带）|
| Storage | **≥200 GB SSD**（LFS 数据 + 中间产物）|
| Python | 3.10+（系统自带）|
| 预装 | NVIDIA driver（CUDA 12.x）|

> ⚠️ **镜像里**不需要预装 PyTorch。`01_a10_env_setup.sh` 会自己装。

### 1.2 SSH 进去

```bash
ssh -i <your-key>.pem ubuntu@<instance-ip>
```

进入后第一个验证命令（应看到 A10）：

```bash
nvidia-smi
```

**期望输出**（前两行）：
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.xx       Driver Version: 525.xx    CUDA Version: 12.x     |
| GPU  Name        Persistence-M  Bus-Id        Memory-Usage  GPU-Util  ... |
|   0  NVIDIA A10                      ...              0MiB / 24576MiB ... |
```

如果看不到 A10 → 联系平台客服确认镜像含 GPU 驱动。

### 1.3 找到仓库目录

云实例可能已经预拉 GitHub。验证：

```bash
pwd                   # 看到当前目录
ls scripts/launcher/  # 应看到 3 个 .sh 文件
ls .git/ 2>/dev/null | head -3   # 应是 git repo
```

**常见情况**：
- `pwd = /workspace` 且 `ls scripts/launcher/` 显示文件 → 已经在仓库根目录，继续 §3
- `pwd = ~` 且没看到 `scripts/launcher/` → 仓库在别的路径，跑：
  ```bash
  find / -name "01_a10_env_setup.sh" 2>/dev/null
  ```
- 仓库完全不存在 → §2 先 clone

---

## 2. 一次性准备（首次运行约 5-10 分钟）

### 2.1 克隆仓库（仅当 §1.3 没找到）

```bash
cd ~
git clone https://github.com/dddsx3/Multi-Illumination-Inverse-Rendering.git
cd Multi-Illumination-Inverse-Rendering
```

### 2.2 拉取最新代码

```bash
cd <REPO_ROOT>                  # 替换为实际仓库路径
git pull origin main            # 拉取最新 commit (ed0a068)
git log --oneline -3            # 验证 head 是 ed0a068
```

### 2.3 安装 git-lfs（LFS 拉取 data_sun_confirmatory/*.npy）

⚠️ **包名拼写易错**：是 `git-lfs`（git-l-**ess**），不是 `git-lf`：

```bash
sudo apt-get update
sudo apt-get install -y git-lfs    # 注意末尾是 l-f-s, 不是 l-f
```

### 2.4 拉数据

```bash
cd <REPO_ROOT>
git lfs install                  # 一次性配置 git hooks
git lfs pull                     # 拉 ~300 MB 数据
```

**期望进度条显示** ~300 MB 下载。**如果没进度条**，说明 `git lfs` 装的版本有问题，参考 §6.4。

### 2.5 验证数据

```bash
ls p1/calibration_set/data_sun_confirmatory/conf_*/albedo.npy | head -5
# 期望 5 行真实路径
# 每个 albedo.npy 应 ~52-65 KB（不是 134 字节的 LFS pointer）
stat -c %s p1/calibration_set/data_sun_confirmatory/conf_sphere_r05/albedo.npy
# 期望：>50000
```

如果显示 134 bytes，是 LFS pointer，需要再 `git lfs pull`。

### 2.6 跑环境脚本

```bash
bash scripts/launcher/01_a10_env_setup.sh
```

**脚本会做**（不需要交互）：
1. sanity check（uname / nproc / free / df）
2. 验证 git status 干净
3. **验证 LFS 数据真实存在**（自动检测 pointer 文件，若发现直接退出并提示）
4. 创建 `.venv_a10/` 虚拟环境
5. 装 `torch + torchvision`（cu128 wheel，A10 SM86 兼容）
6. 装 `numpy + scipy + pandas`
7. CUDA sanity（小 matmul on GPU）
8. 输出 OpenBLAS thread 配置（OMP_NUM_THREADS=10）

**期望输出末行**：
```
[01 env] DONE — ready for 02_a10_run_all.sh
venv: /workspace/.venv_a10
data: /workspace/p1/calibration_set/data_sun_confirmatory (19 scenes)
GPU: NVIDIA A10
next: bash scripts/launcher/02_a10_run_all.sh
```

**耗时**: 5-10 分钟（pip install 慢）。如果卡卡超过 15 分钟，看 §6.5。

---

## 3. 主任务跑（约 6.5 小时）

```bash
bash scripts/launcher/02_a10_run_all.sh
```

### 3.1 脚本会做什么（不需要交互，除了最后的 commit 确认）

| Step | 动作 | wall-clock |
|---|---|---|
| 1 | sanity + GPU 检查 | 5 s |
| 2 | 进入 venv + 设 BLAS thread=10 | instant |
| 3 | **后台启动 Task G**（240 run × ~2.5s ≈ 17 min）| (并行) |
| 4 | **P1-A GSIQ**（6 scenes × 4960 + 2000 subsets × 2 score = 83,520 calls × 0.25 s）| **~5.8 h** |
| 5 | **P1-A solver arm**（360 runs × ~2.5 s）| **~0.3 h** |
| 6 | 等 Task G 后台完成 | (并行) |
| 7 | 验证 6 个产物文件存在 | instant |
| 8 | 解析 gate verdict | instant |
| 9 | **询问** `Commit + push results? (y/N)`，输入 `y` 自动 commit + push | 30 s |

**期望总 wall-clock**: **~6.5 h**（Task G 与 GSIQ 完全并行；GSIQ CPU-bound, Task G GPU-bound）

**抵扣机时** (wall-clock × 3.3): **~21.5 / 24**

### 3.2 监控（第二个 SSH session）

```bash
# P1-A 进度（每行 1 subset, 约 0.25s/call）
tail -f r5/r5_p1_a_full_run.log

# GPU 占用（Task G 跑时会有 CUDA process）
nvidia-smi

# 已写多少行
wc -l r5/r5_p1_albedo_ablation.csv

# Task G 进度
tail -f r4pp/task_g_run.log
```

### 3.3 进度判断

**P1-A GSIQ 正常进度**：
- 每行 ~0.25-0.5 s（BLAS 调度可能慢一些）
- 总 83,520 行 × ~0.3 s ≈ **~7 h upper bound**；实际 **~5.8 h**

**Task G 正常进度**：
- 240 run × 2.5 s/run ≈ **10 min**
- 在 GSIQ 启动后约 10 min 完成

**正常 wall-clock 切片**：
```
0:00   P1-A GSIQ N=3 scene 1 (~50 min)
0:50   P1-A GSIQ N=3 scene 2 (~50 min)
1:40   P1-A GSIQ N=3 scene 3-6 (剩余 ~3 h)
4:40   P1-A GSIQ N=5 scenes 1-6 (剩余 ~1.7 h)
6:20   P1-A solver arm (~15 min)
6:35   DONE
```

### 3.4 跑完后

脚本会问 `Commit + push results? (y/N)`：
- 输入 **`y`**：自动 commit + push 到 `origin HEAD`
- 输入 **`N` 或回车**：不 push，下次手动 push

如果脚本意外退出（OOM / 抢占），**结果已经在 CSV 里**，下次 SSH 进来：
```bash
git status                              # 应有 un-committed r5/ + r4pp/
git add r5/r5_p1_albedo_ablation*.csv r5/r5_p1_albedo_ablation_gate.md \
        r5/r5_p1_a_full_run.log r4pp/07_local_vs_global_init.csv
git commit -m "feat(r5-p1): A10 P1-A full + Task G results"
git push origin HEAD
```

---

## 4. 预期产物清单

### 4.1 必须存在（脚本会验证）

```
r5/r5_p1_albedo_ablation.csv               # 83,520 数据行 (6 scenes × 13,920)
r5/r5_p1_albedo_ablation_ranking.csv       # 12 cell ranking metrics
r5/r5_p1_albedo_ablation_outliers.csv      # boundary outliers (通常 <100 行)
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

**Task G**（自己跑一段分析）：

```bash
source .venv_a10/bin/activate
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
[ ] 启动 A10 实例（Ubuntu 22.04, A10 24GB）
[ ] SSH 进实例
[ ] nvidia-smi   # 应看到 A10
[ ] pwd + ls scripts/launcher/   # 确认仓库路径
[ ] cd <REPO_ROOT>
[ ] git pull origin main   # 拉最新代码
[ ] git log --oneline -3   # 验证 head = ed0a068
[ ] sudo apt-get install -y git-lfs   # 注意是 l-f-s
[ ] git lfs install
[ ] git lfs pull   # 应有 ~300 MB 下载进度
[ ] stat -c %s p1/calibration_set/data_sun_confirmatory/conf_sphere_r05/albedo.npy
    # 期望 >50000 (真实数据)
[ ] bash scripts/launcher/01_a10_env_setup.sh
[ ] bash scripts/launcher/02_a10_run_all.sh
[ ] 看 6.5 h... (第二个 SSH session 用 tail -f 监控)
[ ] 脚本问 "Commit + push?" 时输 y
[ ] 验证 r5/r5_p1_*.csv + r4pp/07_local_vs_global_init.csv 都在
[ ] 关机
[ ] 本机 git pull 拉结果
[ ] 跑 §4.2 的 P1-A gate 解析 + Task G beta 解析
[ ] 邮件 / Slack 通知项目 owner
```

---

## 6. 故障处理

### 6.1 OOM 中途退出

A10 24 GB 不会 OOM（P1-A GSIQ 是 CPU-only）。但如果意外 OOM：

**症状**: `MemoryError` 或 CUDA OOM

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
| `r5_p1_albedo_ablation.py` | `csv_mode='a'` 自动 append |
| `r4pp_local_vs_global.py` | 自带 incremental（按 `(scene, N, subset, init_mode)` skip done）|

被抢占后：

```bash
# SSH 重新进（如果实例被关, 需要重启）
bash scripts/launcher/02_a10_run_all.sh   # 自动跳过已完成部分
```

### 6.3 GSIQ 太慢

正常 5.8 h 完成。如果 6.5 h 还没完成 80%，检查：

```bash
top -bn 1 | head -20       # CPU 占用
echo $OMP_NUM_THREADS      # 应 10
lscpu | grep "Core"        # 应 20 核
# top 单进程应显示 ~1000% CPU（10 核）
```

如果 CPU 只用单核：

```bash
source .venv_a10/bin/activate
export OMP_NUM_THREADS=10
export OPENBLAS_NUM_THREADS=10
# 重跑
```

### 6.4 LFS 数据缺失或仍是 pointer 文件

**症状**: `albedo.npy` < 1000 bytes (LFS pointer)

**解决**:

```bash
git lfs pull
# 或强制重拉
git lfs fetch --all
git lfs checkout
```

如果 `git-lfs` 命令本身没装：

```bash
sudo apt-get install -y git-lfs    # 注意包名是 git-lfs (末尾是 ess)
```

### 6.5 pip install torch 超时

**症状**: `pip install torch` 报 SSL 或网络错误 / 卡死

**解决**（按优先级）：

```bash
# 选项 1: 重试 (PyPI 中国 CDN 偶发不通)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 选项 2: 清华 anaconda 镜像
pip install torch torchvision --index-url https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/linux/

# 选项 3: aliyun PyPI (numpy/scipy 可走)
pip install numpy scipy --index-url https://mirrors.aliyun.com/pypi/simple/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 选项 4: 平台工单系统申请出网白名单加 pypi.org + download.pytorch.org
```

> **现实经验**: A10 实例的 PyPI 默认可达；若不是，优先选项 1 重试 3 次。

### 6.6 Git push 失败

**症状**: `git push` 报权限 / auth 错误

```bash
git remote -v
# 应是 https://github.com/dddsx3/Multi-Illumination-Inverse-Rendering.git

git config --get user.name
git config --get user.email

# 手动 push (会要求 username + Personal Access Token)
git push origin HEAD
```

### 6.7 LFS 包名错（这是我们已修过的，prevention）

⚠️ **不要写 `git-lf`**（没有这个包）。正确包名是 `git-lfs`（git-l-ess）。

如果你看到 "E: Unable to locate package git-lf" → 你写错了，加个 s。

---

## 7. 不要做的事

- ❌ **不要**手动修改 `gauge_fisher_v2.py`（任务书 §25 KILL 条件）
- ❌ **不要**调 `spec_cutoff=1e-8` / `cutoff=1e-8`（任务书 §5 冻结值）
- ❌ **不要**为了赶时间跳过 `--solver`（solver arm 是 PASS-A Gate 的第二支柱）
- ❌ **不要**在 P1-A full 未完成前跑 P1-B / P2（任务书 §27 严格顺序）
- ❌ **不要**用 `git push --force` 覆盖已 push 的 commit
- ❌ **不要**让 A10 实例跑超过 24 机时（超出免费额度）
- ❌ **不要**同时跑 GSIQ + 多个 solver（GPU 16 GB 装不下 2 个 batched solver）

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
| 算力清点 | `r5/R5_H100_RESOURCE_INVENTORY.md`, `r5/R5_RESOURCE_PRECISION_MATRIX.md`, `r5/R5_A10_FIT_CHECK.md` |
| P1-A smoke 报告 | `r5/r5_p1_a_closure.md` |
| 边界异常诊断 | `r5/r5_p1_a_boundary_diagnostic.md` |

---

## 10. v1 → v2 修订清单

本次 v2 修复了 v1 操作表的 5 处低级错误：

1. **包名错误** `git-lf` → `git-lfs`（git-l-ess，末尾是 ess）
2. **脚本缺 N3 full enumerate**：`02_a10_run_all.sh` 加 `--n3_limit 4960`，否则 P1-A 只跑前 500 个 N=3 subsets（任务书 §10 要求 enumerate all 4960）
3. **`bc` 依赖**：`02_a10_run_all.sh` 用纯 shell arithmetic（`bc` 在 Ubuntu minimal 上不一定预装）
4. **LFS pointer 误判**：`01_a10_env_setup.sh` 加 `albedo.npy size > 1000` 验证
5. **PyPI 镜像 URL 错误**：手册 `pip install torch` 镜像改为 `download.pytorch.org/whl/cu128`（清华 anaconda + aliyun pypi 备选）

---

*作者: ZCode agent · 2026-09-01 (v2)*
*本文档所有命令可直接 copy-paste；操作员不需要任何编程或项目知识*