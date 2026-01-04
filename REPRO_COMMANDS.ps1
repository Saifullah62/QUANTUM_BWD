# SemanticPhase v3 - 3-Seed Validation Reproduction Script
# Generated: 2026-01-04
#
# This script sets up and runs the 3-seed validation for SemanticPhase polysemy disambiguation.
# It expects to be run from a QUANTUM_BWD repository root with the required dependencies installed.
#
# REQUIREMENTS:
#   - Python 3.8+
#   - PyTorch 2.0+ with CUDA support
#   - transformers
#   - The bundle file: data/dress_rehearsal_bundles.jsonl (MD5: 8f4e13b7ae4ab5e793ea47f2a80000a4)
#
# EXPECTED RUNTIME: ~3-4 hours on a modern GPU (RTX 3090 or equivalent)

param(
    [string]$BundlePath = "data/dress_rehearsal_bundles.jsonl",
    [switch]$SkipDependencyCheck,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "  SemanticPhase v3 - 3-Seed Validation Reproduction" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host ""

# Check Python availability
Write-Host "[1/5] Checking Python..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "  Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "  ERROR: Python not found in PATH" -ForegroundColor Red
    exit 1
}

# Check required files
Write-Host "[2/5] Checking required files..." -ForegroundColor Yellow

$requiredFiles = @(
    "scripts/run_3seed_validation.py",
    "scripts/train_semantic_phase_v2.py",
    "paradigm_factory/bundle_dataset.py",
    "qllm/layers/sense_head.py"
)

$missingFiles = @()
foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Host "  [OK] $file" -ForegroundColor Green
    } else {
        Write-Host "  [MISSING] $file" -ForegroundColor Red
        $missingFiles += $file
    }
}

if ($missingFiles.Count -gt 0) {
    Write-Host ""
    Write-Host "ERROR: Missing required files. Copy from evidence bundle:" -ForegroundColor Red
    foreach ($f in $missingFiles) {
        Write-Host "  - $f"
    }
    exit 1
}

# Check bundle file
Write-Host "[3/5] Checking bundle file..." -ForegroundColor Yellow
if (Test-Path $BundlePath) {
    Write-Host "  [OK] Bundle file exists: $BundlePath" -ForegroundColor Green

    # Verify hash
    $expectedHash = "8f4e13b7ae4ab5e793ea47f2a80000a4"
    try {
        $actualHash = (Get-FileHash -Path $BundlePath -Algorithm MD5).Hash.ToLower()
        if ($actualHash -eq $expectedHash) {
            Write-Host "  [OK] Bundle hash verified: $expectedHash" -ForegroundColor Green
        } else {
            Write-Host "  [WARN] Bundle hash mismatch!" -ForegroundColor Yellow
            Write-Host "    Expected: $expectedHash" -ForegroundColor Yellow
            Write-Host "    Actual:   $actualHash" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "  [WARN] Could not verify bundle hash" -ForegroundColor Yellow
    }
} else {
    Write-Host "  [MISSING] Bundle file not found: $BundlePath" -ForegroundColor Red
    Write-Host "  Please provide the bundle file (15,780 bundles, 86,073 items)" -ForegroundColor Red
    exit 1
}

# Check CUDA availability
Write-Host "[4/5] Checking CUDA availability..." -ForegroundColor Yellow
if (-not $SkipDependencyCheck) {
    $cudaCheck = python -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>&1
    Write-Host "  $cudaCheck" -ForegroundColor Green
}

# Build command
Write-Host "[5/5] Building validation command..." -ForegroundColor Yellow

$command = @"
python scripts/run_3seed_validation.py `
  --bundles $BundlePath `
  --epochs 3 `
  --seeds 42 123 456 `
  --sense-head `
  --hard-neg-top-k 3 `
  --track-killers 20 `
  --killer-log-every 50
"@

Write-Host ""
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "  COMMAND TO EXECUTE:" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host ""
Write-Host $command -ForegroundColor White
Write-Host ""

if ($DryRun) {
    Write-Host "[DRY RUN] Command not executed. Remove -DryRun to run." -ForegroundColor Yellow
    exit 0
}

Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "  STARTING VALIDATION RUN" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host ""

# Execute
python scripts/run_3seed_validation.py `
    --bundles $BundlePath `
    --epochs 3 `
    --seeds 42 123 456 `
    --sense-head `
    --hard-neg-top-k 3 `
    --track-killers 20 `
    --killer-log-every 50

$exitCode = $LASTEXITCODE

Write-Host ""
Write-Host "=" * 70 -ForegroundColor Cyan
if ($exitCode -eq 0) {
    Write-Host "  VALIDATION COMPLETE" -ForegroundColor Green
} else {
    Write-Host "  VALIDATION FAILED (exit code: $exitCode)" -ForegroundColor Red
}
Write-Host "=" * 70 -ForegroundColor Cyan

Write-Host ""
Write-Host "Key metrics to verify:" -ForegroundColor Yellow
Write-Host "  - Seeds passed: 3/3"
Write-Host "  - Late hard positive rate: >= 30% (expected: ~95%)"
Write-Host "  - Avg final slack: > 0 (expected: ~+1.5)"
Write-Host ""
Write-Host "Check logs in: checkpoints/validation_seed_*/dashboard_log.json"

exit $exitCode
