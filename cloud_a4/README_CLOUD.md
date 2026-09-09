# A4 云端运行包 · 操作员手册（傻瓜式 · 全程只需复制粘贴）

> 目标：在 Ubuntu 云实例上跑完 A4（29,700 次重建），产出唯一交付物
> `a4_results_<时间戳>.zip`，下载后交给作者。
> **总耗时：约 30–60 分钟（32–64 核实例），全程三条命令。**

---

## 第 1 步 · 上传一个文件（作者已提供）

把作者给你的 `A4_CLOUD_PACKAGE.zip`（**18.1 MB，自包含**：代码快照 + git bundle
+ 全部数据 + 全部脚本）上传到云实例任意目录（例：`/workspace/`）。

```zsh
# 示例（本地电脑执行）：
scp A4_CLOUD_PACKAGE.zip user@<云实例IP>:/workspace/
```

## 第 2 步 · 一条命令运行（云实例上执行，zsh）

```zsh
python3 -m zipfile -e /workspace/A4_CLOUD_PACKAGE.zip ~/a4_cloud/ && zsh ~/a4_cloud/RUN.sh
```

第一条解压（自动放到 `~/a4_cloud/`），第二条启动一键脚本。之后**全自动**：
环境预检 → 恢复审计代码（`ddd44b5`，离线 bundle，无需 GitHub）→ venv 依赖 →
Pre-Run Seal 七项检查 → A4 正式运行（29,700 次）→ 完整性检查 → A5 统计 → 打包。
失败会打印 `FATAL: ...` 并停止——按下方 FAQ 处理或把 `~/a4_cloud/logs/` 发给作者。

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
