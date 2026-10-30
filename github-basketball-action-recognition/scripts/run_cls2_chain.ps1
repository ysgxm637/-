# Two-stage 2.0 classifier horse-race chain (ASCII only, no CJK literals in code)
# Order per user spec: A) R(2+1)D-18 multi-seed x3  ->  B) MViT v2 S (x3 + 1 LR guard)
#                      ->  C) R(2+1)D-18 + BiGRU x3
# Each run: train_action_cls2.py, sqrt sampler, same aug/hparams as 1.0 baseline.
# Judge rule: diffs < 5pt = noise (multi-seed mandatory).
$ErrorActionPreference = 'Continue'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = 'utf-8'

$py  = 'F:\zjl\envs\yolo\python\python.exe'
$tr  = Join-Path $PSScriptRoot 'train_action_cls2.py'
$root   = Split-Path -Parent $PSScriptRoot          # stage 2.0 root
$basket = Split-Path -Parent $root                  # F:\zjl\basket
$logs   = Join-Path $root 'logs'
$runroot = Join-Path $root 'runs\cls2'
New-Item -ItemType Directory -Force -Path $logs, $runroot | Out-Null
$clog = Join-Path $logs 'cls2_chain_log.txt'

# locate the 1.0 folder (CJK name, derived by dataset probe, no literal needed)
$ds1 = Get-ChildItem $basket -Directory | Where-Object {
    Test-Path (Join-Path $_.FullName 'datasets\dataset_action_clip_val35_sliding\train')
} | Select-Object -First 1
if (-not $ds1) { Write-Output 'FATAL: dataset folder not found'; exit 1 }
$ds = Join-Path $ds1.FullName 'datasets\dataset_action_clip_val35_sliding'

function Run-One($tag, $arch, $seed, $bigru, $batch, $lr) {
    $args = @('-u', $tr, '--data-root', $ds, '--run', $runroot, '--tag', $tag,
              '--arch', $arch, '--seed', $seed, '--batch', $batch, '--lr', $lr,
              '--bigru', $bigru)
    $line = "===== [$((Get-Date -Format 'MM-dd HH:mm:ss'))] RUN $tag arch=$arch seed=$seed bigru=$bigru batch=$batch lr=$lr"
    Write-Output $line
    Add-Content -Path $clog -Value $line -Encoding UTF8
    & $py @args 2>&1 | Tee-Object -FilePath $clog -Append
    $code = $LASTEXITCODE
    $done = "===== [$((Get-Date -Format 'MM-dd HH:mm:ss'))] DONE $tag exit=$code"
    Write-Output $done
    Add-Content -Path $clog -Value $done -Encoding UTF8
}

Write-Output "STAGE A: R(2+1)D-18 multi-seed x3 (baseline)"
Run-One 'r21d_s0' 'r2plus1d18' 0 0 16 '0.0003'
Run-One 'r21d_s1' 'r2plus1d18' 1 0 16 '0.0003'
Run-One 'r21d_s2' 'r2plus1d18' 2 0 16 '0.0003'

Write-Output "STAGE B: MViT v2 S (224 input, batch 8) x3 + 1 LR guard"
Run-One 'mvit_s0'    'mvitv2s' 0 0 8 '0.0003'
Run-One 'mvit_s1'    'mvitv2s' 1 0 8 '0.0003'
Run-One 'mvit_s2'    'mvitv2s' 2 0 8 '0.0003'
Run-One 'mvit_lr1e4_s0' 'mvitv2s' 0 0 8 '0.0001'

Write-Output "STAGE C: R(2+1)D-18 + BiGRU(256x2) x3"
Run-One 'bigru_s0' 'r2plus1d18' 0 1 16 '0.0003'
Run-One 'bigru_s1' 'r2plus1d18' 1 1 16 '0.0003'
Run-One 'bigru_s2' 'r2plus1d18' 2 1 16 '0.0003'

Write-Output "ALL DONE. Summaries: $runroot\<tag>\train_log.txt ; chain log: $clog"
