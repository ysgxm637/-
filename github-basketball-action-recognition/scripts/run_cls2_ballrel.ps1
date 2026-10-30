# BallFused dual-branch (appearance + ball-relation GRU) probe run.
# Validates --ball-rel before committing the full x3 + MViT chain.
# ASCII only; Chinese paths resolved via $PSScriptRoot (project rule).
$ErrorActionPreference = 'Continue'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = 'utf-8'

$py  = 'F:\zjl\envs\yolo\python\python.exe'
$tr  = Join-Path $PSScriptRoot 'train_action_cls2.py'
$root   = Split-Path -Parent $PSScriptRoot
$basket = Split-Path -Parent $root
$runroot = Join-Path $root 'runs\cls2'
$clog = Join-Path (Join-Path $root 'logs') 'cls2_ballrel_log.txt'

$ds1 = Get-ChildItem $basket -Directory | Where-Object {
    Test-Path (Join-Path $_.FullName 'datasets\dataset_action_clip_val35_sliding\train')
} | Select-Object -First 1
$ds = Join-Path $ds1.FullName 'datasets\dataset_action_clip_val35_sliding'

$tag = 'r21db_s0'
$best = Join-Path $runroot "$tag\best.pt"
if (Test-Path $best) {
    Write-Output "SKIP $tag (best.pt exists)"
} else {
    $line = "===== [$((Get-Date -Format 'MM-dd HH:mm:ss'))] RUN $tag ballrel=1 seed=0"
    Write-Output $line
    Add-Content -Path $clog -Value $line -Encoding UTF8
    & $py -u $tr --data-root $ds --run $runroot --tag $tag `
        --arch r2plus1d18 --seed 0 --batch 16 --lr 0.0003 --ball-rel 1 `
        --patience 0 2>&1 | Tee-Object -FilePath $clog -Append
    Add-Content -Path $clog -Value "===== DONE $tag exit=$LASTEXITCODE" -Encoding UTF8
}
Write-Output 'BALLREL PROBE DONE'
