# A4 云端运行包 · 操作员手册（傻瓜式 · 全程只需复制粘贴）

> 目标：在 Ubuntu 云实例上跑完 A4（29,700 次重建），产出唯一交付物
> `a4_results_<时间戳>.zip`，下载后交给作者。
> **总耗时：约 30–60 分钟（32–64 核实例），全程三条命令。**

---

## 第 0 步 · 上传两个文件（作者已提供）

把作者给你的这两个文件上传到云实例的同一个目录（例如 `~/a4/`）：

| 文件 | 大小 | 用途 |
|---|---|---|
| `cloud_a4.zip` | ~30 KB | 全部脚本（本目录打包） |
| `openillumination_data_bundle.zip` | ~17 MB | 11 个对象的数据（精确所需子集） |

上传方式任选：`scp`、云控制台文件上传、或对象存储中转。

```zsh
# 示例（本地电脑执行）：
scp cloud_a4.zip openillumination_data_bundle.zip user@<云实例IP>:~/a4/
```

## 第 1 步 · 解压（云实例上执行）

```zsh
cd ~/a4
unzip -o cloud_a4.zip -d cloud_a4
unzip -o openillumination_data_bundle.zip -d data_extract
mkdir -p data
mv data_extract/data/OpenIllumination data/
mv data_extract/data/OpenIllumination_meta data/
rm -rf data_extract
```

## 第 2 步 · 一键运行（唯一需要执行的命令）

```zsh
cd ~/a4/cloud_a4
bash run_all.sh
```

脚本会**自动完成全部 8 步**：环境预检 → 克隆审计代码（`ddd44b5`）→ venv 依赖
→ Pre-Run Seal 七项检查 → A4 正式运行（29,700 次）→ 完整性检查 → A5 统计 → 打包。
每一步之间会显示 `[y/N]` 确认——**一路按 `y` 回车即可**。

无人值守模式（可选，跳过所有确认）：

```zsh
bash run_all.sh --auto
```

## 第 3 步 · 期间想看进度（开第二个终端）

```zsh
cd ~/a4/cloud_a4
bash status.sh          # 单次查看
watch -n 30 bash status.sh   # 每 30 秒自动刷新
```

## 第 4 步 · 完成后取结果

脚本结束时打印 `PACKAGE READY: a4_results_<时间戳>.zip`，把它下载回本地：

```zsh
# 本地电脑执行：
scp user@<云实例IP>:~/a4/cloud_a4/a4_results_*.zip ./
```

然后把 zip 交给作者。**到此操作员的工作全部结束。**

---

## 常见问题（操作员必读）

| 现象 | 处理 |
|---|---|
| `FATAL: python3 not found` | `sudo apt update && sudo apt install -y python3 python3-venv git zip` 后重跑 |
| `DATASET NOT FOUND` | 数据没上传或解压路径不对——按第 1 步重做，确认 `~/a4/data/OpenIllumination/OLAT/obj_03_pumpkin/Lights/000/com_masked_thumbnail/A1.png` 存在 |
| `某对象 has N/142 thumbnails` | 数据不完整——重新解压 `openillumination_data_bundle.zip` |
| `FATAL: clone failed` | 新仓库是 private——改用作者提供的 `repo-new.zip`（如有），解压到 `cloud_a4/repo-new` 后重跑 |
| `FATAL: pre-run seal FAILED` | **不要重试**，把 `logs/precheck.log` 原样发给作者 |
| `CELL FAILED ... INCOMPLETE` | 断点续跑：**直接重跑 `bash run_all.sh`**，已完成的 cell 自动跳过 |
| 中途想停止 | `Ctrl+C` 即可；重启后重跑 `bash run_all.sh`，从断点继续 |
| 内存不足报错 | `MRC_WORKERS=8 bash run_all.sh --auto`（降低并行度） |

## 纪律（写给任何想"顺手改点东西"的人）

- **不要**修改 `a4_cloud_driver.py` / `a4_stage2_check.py` / `a5_analyze.py` 的任何内容——
  它们对应审计过的代码状态（F5 manifest 门会校验仓库文件 hash，改了会直接拒绝运行）；
- **不要**改网格/种子/策略/budget——运行日志与结果文件都会暴露不一致，实验作废；
- 唯一允许调整的环境变量：`MRC_WORKERS`（并行进程数）；
- 遇到本手册未覆盖的情况：**停止操作，把 `logs/` 全部目录打包发作者**。
