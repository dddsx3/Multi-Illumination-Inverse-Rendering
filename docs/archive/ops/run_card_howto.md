# RUN_CARD 生成手册（run_card_howto）

> 归属：任务书 T v2.0 OP-2 §2 · 每个训练/评估 run 必产 RUN_CARD.json（与 eval 产物同目录）。

## RUN_CARD.json 模板（照抄；含 FIX-05 三指纹）

```json
{"run":"<run_id>","generation":"Gen-Ax",
 "code_commit_sha":"<训练启动前 git rev-parse HEAD；不可回溯时如实写 unrecorded>",
 "config":{<训练 config 全量>},
 "config_sha256":"<config JSON(SHA, sort_keys) 哈希>",
 "data_manifest_sha256":"<splits/synthetic_v3.json 哈希>",
 "data_dir":{"scene_dirs":<数>,"total_mb":<大小>},
 "eval":{"test":"./eval_output/<run>/eval_summary.json"},
 "ckpt_sha256":"<计算值>",
 "produced":"2026-xx-xx","status":"ok|interrupted"}
```

## 三指纹生成命令（FIX-05，可直接照抄）

① `code_commit_sha` —— 训练启动前：
```bash
git rev-parse HEAD
```
② `config_sha256` —— 训练 config 全量 json（以 run_arms 对应臂段配置为准）的 SHA256：
```powershell
# 先把 config 存为 config_<run>.json（sort_keys, ensure_ascii=False），再：
Get-FileHash -Algorithm SHA256 -Path "config_<run>.json"
# 或 certutil -hashfile "config_<run>.json" SHA256
```
（python 等价：`hashlib.sha256(json.dumps(cfg,sort_keys=True,ensure_ascii=False).encode()).hexdigest()`）
③ `data_manifest_sha` —— 数据清单哈希 + 目录指纹：
```powershell
Get-FileHash -Algorithm SHA256 -Path "D:\MIR_Archive_20260829\Multi-Illumination-Inverse-Rendering\splits\synthetic_v3.json"
# 目录指纹（场景目录数 + 总 MB）用文件管理器属性或 python os.scandir 汇总
```

## ckpt SHA-256 计算（Windows 两种都可用）

PowerShell：
```powershell
Get-FileHash -Algorithm SHA256 -Path "D:\MIR_Archive_20260829\checkpoints\<run>\best_model.pth"
```
certutil（cmd/PowerShell 通用）：
```
certutil -hashfile "D:\MIR_Archive_20260829\checkpoints\<run>\best_model.pth" SHA256
```
> 云训/远端场景补充细则（审计意见 §三-5）：ckpt 不在本机时，在远端执行上述命令取哈希，人工回填 json 后本机提交。

## 世代变更行登记（FIX-05 · 防再犯）

任何**代码/协议改动后首次训练**前，必须在 `docs/gpu_ledger.md` 登记一行世代变更：

```
| 日期 | 世代变更 Gen-Ax → Gen-Ay | 理由 | 相关 commit | code_commit_sha |
```

规则：无世代标注的数字禁止引用（CLAIM_CARDS S-01 口径同款）；三个指纹中任一缺失即
RUN_CARD 不合格（status 不许写 ok，写 interrupted/partial 并注明缺失项）。

*示例（A3-0 回填）：Gen-历史(bs8) → Gen-A3(bs4)；理由=本机 12GB 不可跑 bs8（INC-0014）；code_commit_sha=unrecorded（13:34 启动时刻无法回溯，诚实注记）。*
