# Phase 1 T1.2 全量数据集自主流水线（下载 -> 渲染 -> 校验）
# 各阶段幂等可续跑；任何阶段失败即中止并保留现场
$ErrorActionPreference = "Continue"
Set-Location "D:/Multi-Illumination Inverse Rendering/repo"

Write-Output "=== STAGE 1: download 820 models ==="
python download_objaverse.py --count 820 --out D:/data/objaverse_raw --list_file obj_models_list.txt --workers 5 --max_seconds 14400 > _dl_log.txt 2>&1
Write-Output "stage1_exit=$LASTEXITCODE"
if ($LASTEXITCODE -ne 0) { Write-Output 'CHAIN ABORT at stage1'; exit 1 }

$models = (Get-Content obj_models_list.txt | Measure-Object).Count
Write-Output "models_listed=$models"
if ($models -lt 100) { Write-Output 'CHAIN ABORT: too few models'; exit 1 }

Write-Output "=== STAGE 2: render $models scenes ==="
py -3.10 -m blenderproc run render_dataset.py --obj_list obj_models_list.txt --out_dir D:/data/synthetic --size 256 --gpu > _render_full_log.txt 2>&1
Write-Output "stage2_exit=$LASTEXITCODE"

Write-Output "=== STAGE 3: validate ==="
python validate_dataset.py --root D:/data/synthetic --sample 50 > _validate_log.txt 2>&1
Write-Output "stage3_exit=$LASTEXITCODE"

Get-Content _validate_log.txt -Tail 10 | Write-Output