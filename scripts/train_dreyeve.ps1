# Dr(eye)ve end-to-end training workflow (PowerShell)
# Run from the project root:
#   cd D:\TW\Final_Project
#   powershell -ExecutionPolicy Bypass -File scripts\train_dreyeve.ps1

$ErrorActionPreference = "Stop"

# Activate the project virtualenv
$activate = Join-Path (Split-Path $PSScriptRoot) ".venv\Scripts\Activate.ps1"
if (Test-Path $activate) {
    . $activate
    Write-Host "venv activated." -ForegroundColor DarkGray
} else {
    Write-Warning "No .venv found. Run: python -m venv .venv"
    Write-Warning "Then: .venv\Scripts\pip install -r requirements.txt"
}

$DATASET_ROOT = "D:\TW\dr(eye)ve"
$OUTPUT_JSON  = "data\dreyeve_train.json"
$MODEL_OUT    = "models\intent_dreyeve.pth"

Write-Host "=== Step 1: Preparing dr(eye)ve dataset ===" -ForegroundColor Cyan
python scripts\prepare_dreyeve.py `
    --dataset  $DATASET_ROOT `
    --output   $OUTPUT_JSON `
    --frame-skip 1 `
    --min-frames 30

Write-Host ""
Write-Host "=== Step 2: Training intent model ===" -ForegroundColor Cyan
python scripts\train_intent.py train `
    --data      $OUTPUT_JSON `
    --output    $MODEL_OUT `
    --epochs    60 `
    --batch     32 `
    --lr        1e-3 `
    --seq-len   90 `
    --stride    15 `
    --early-stop 15

Write-Host ""
Write-Host "=== Step 3: Evaluating ===" -ForegroundColor Cyan
python scripts\train_intent.py evaluate `
    --data    $OUTPUT_JSON `
    --output  $MODEL_OUT `
    --seq-len 90

Write-Host ""
Write-Host "Done. Model saved to $MODEL_OUT" -ForegroundColor Green
Write-Host "Run inference:"
Write-Host "  python inference.py --driver 0 --scene 1 --model $MODEL_OUT"
