# RUN_CARD 生成手册（run_card_howto）

> 归属：任务书 T v2.0 OP-2 §2 · 每个训练/评估 run 必产 RUN_CARD.json（与 eval 产物同目录）。

## RUN_CARD.json 模板（照抄）

```json
{"run":"A3-3_fw_main","ckpt":"ckpt/fw_main_seed42.pt","ckpt_sha256":"<计算值>",
 "config":{"model":"fusion","modality":"rgb","split_manifest":"splits/synthetic_v3.json",
 "epochs":100,"amp":"bf16","seed":42},
 "eval":{"test":"./eval_output/A3-3_fw_main/eval_summary.json"},
 "produced":"2026-xx-xx","status":"ok|interrupted"}
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
