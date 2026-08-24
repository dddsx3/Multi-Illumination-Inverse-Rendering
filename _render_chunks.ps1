# 分块渲染（参数化版）：每 ChunkSize 场景重启 Blender
param(
  [int]$ChunkSize = 40,
  [string]$ObjList = "obj_models_list.txt",
  [string]$OutDir = "D:/data/synthetic",
  [string]$ExtraArgs = ""
)
$ErrorActionPreference = "Continue"
Set-Location "D:/Multi-Illumination Inverse Rendering/repo"

$all = Get-Content $ObjList
$remaining = @()
foreach ($p in $all) {
    $sha = [IO.Path]::GetFileNameWithoutExtension($p)
    if (-not (Test-Path (Join-Path $OutDir ($sha + "/sh_coeffs.npy")))) { $remaining += $p }
}
Write-Output ("remaining_scenes=" + $remaining.Count)

$chunkIdx = 0
for ($i = 0; $i -lt $remaining.Count; $i += $ChunkSize) {
    $end = [Math]::Min($i + $ChunkSize - 1, $remaining.Count - 1)
    $batch = @($remaining[$i..$end])
    $chunkIdx++
    $listFile = Join-Path (Get-Location) "_chunk_list.txt"
    [IO.File]::WriteAllLines($listFile, $batch)
    Write-Output ("=== chunk " + $chunkIdx + " (" + $batch.Count + " models) ===")
    py -3.10 -m blenderproc run render_dataset.py --obj_list $listFile --out_dir $OutDir --size 256 --gpu $ExtraArgs.Split(" ") >> _render_full_log.txt 2>&1
    Write-Output ("chunk" + $chunkIdx + "_exit=" + $LASTEXITCODE)
}
Write-Output "ALL CHUNKS DONE"