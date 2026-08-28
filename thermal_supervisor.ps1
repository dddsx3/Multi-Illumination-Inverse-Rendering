# 训练热保护随车看门狗（兜底，无需管理员权限）
# 进程内温度墙（thermal_guard.py + trainer.py）是主防线：越线先落盘再以 rc=42
# 优雅退出，编排器 run_arms.py 等温度回落后自动续跑，损失不超过一个 batch。
# 本脚本只负责两件防火墙外的事：
#   1) 无训练进程且任务未完成时，等 GPU 冷却到安全线后拉起编排器；
#   2) 异常兜底硬杀：进程内守卫失效（如 CUDA hang 卡死无法轮询温度）时，
#      在 >= HARD_KILL_C 强杀，防止整机热保护关机。阈值刻意抬到主防线之上，
#      正常路径永远轮不到它。
$repo = "D:\Multi-Illumination Inverse Rendering\repo"
$log = Join-Path $repo "_supervisor_log.txt"
$done = Join-Path $repo "_orchestrator_log.txt"
$HARD_KILL_C = 88        # 主防线 80 之上 + 读数滞后余量，仅兜底
$RESUME_C = 70           # 与 THERMAL_RESUME（thermal_guard.py）一致

function Get-GpuTemp {
    $raw = (& nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)
    return [int]($raw -replace '[^0-9]', '')
}

function Get-TrainingProcs {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -match 'main\.py|run_arms\.py|run_phase2_all\.py|evaluate_model\.py|eval_n_curve\.py' }
}

Add-Content $log "supervisor start $(Get-Date)"
while ($true) {
    if ((Test-Path $done) -and (Select-String -Path $done -Pattern "ALL ARMS COMPLETE" -Quiet)) {
        Add-Content $log "ALL ARMS COMPLETE $(Get-Date) -> exit"
        break
    }
    if (-not (Get-TrainingProcs)) {
        # 启动前等温度回落到安全线（进程内守卫与冷却重启都以 70 为界）
        $t = Get-GpuTemp
        while ($t -gt $RESUME_C) {
            Add-Content $log "pre-launch wait t=$t $(Get-Date)"
            Start-Sleep -Seconds 60
            $t = Get-GpuTemp
        }
        Add-Content $log "launch orchestrator t=$t $(Get-Date)"
        # 训练内热节流：1.5s/批次（只改墙钟不碰训练数学，见本机长跑前置清单 §1.2）
        $env:THERMAL_PACE = "1.5"
        Start-Process -FilePath "python" `
            -ArgumentList "-u", "run_arms.py", "--data_root", "D:/data/synthetic_v3", "--budget-hours", "24", "--max-lanes", "1" `
            -WorkingDirectory $repo `
            -RedirectStandardOutput $done `
            -RedirectStandardError (Join-Path $repo "_orchestrator_err.txt") `
            -WindowStyle Hidden
        Start-Sleep -Seconds 150
        continue
    }
    $t = Get-GpuTemp
    Add-Content $log "t=$t $(Get-Date)"
    if ($t -ge $HARD_KILL_C) {
        Add-Content $log "HARD THERMAL GUARD t=$t -> force kill $(Get-Date)"
        Get-TrainingProcs | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
    }
    Start-Sleep -Seconds 30
}