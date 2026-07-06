<#
.SYNOPSIS
  The quality gate — fmt, clippy, tests — ordered cheapest-fail-first.

.DESCRIPTION
  Runs the three checks from the end-of-batch ritual in the order that fails
  fastest: `cargo fmt --check` (instant) -> `cargo clippy --all-targets -D warnings`
  (~20s) -> `cargo test --workspace` (the slow one). A formatting or lint slip
  then costs seconds instead of waiting out the full test run.

  Test execution is the dominant cost (compute-bound f64 N-body / SPH proptests
  plus the GPU suite). It is kept fast by `[profile.dev] opt-level = 2` in the
  workspace Cargo.toml — the tests link optimized code with debug-assertions and
  overflow-checks still on. See DESIGN.md / Cargo.toml for the rationale.

.PARAMETER SkipTests
  Run fmt + clippy only (the seconds-scale checks) — a quick pre-commit sanity
  pass. NOT a substitute for the full gate.

.EXAMPLE
  ./gate.ps1            # full gate
  ./gate.ps1 -SkipTests # fmt + clippy only
#>
param([switch]$SkipTests)

$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $root
$overall = [System.Diagnostics.Stopwatch]::StartNew()

function Invoke-Phase {
    param([string]$Name, [string[]]$CargoArgs)
    Write-Host "=== $Name ===" -ForegroundColor Cyan
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & cargo @CargoArgs
    $code = $LASTEXITCODE
    $sw.Stop()
    $secs = [math]::Round($sw.Elapsed.TotalSeconds, 1)
    if ($code -ne 0) {
        Write-Host "FAIL  $Name  (${secs}s)" -ForegroundColor Red
        Pop-Location
        exit $code
    }
    Write-Host "ok    $Name  (${secs}s)`n" -ForegroundColor Green
}

Invoke-Phase 'fmt --check' @('fmt', '--check')
Invoke-Phase 'clippy'      @('clippy', '--all-targets', '--workspace', '--', '-D', 'warnings')
if (-not $SkipTests) {
    Invoke-Phase 'test'    @('test', '--workspace')
}

$overall.Stop()
$total = [math]::Round($overall.Elapsed.TotalSeconds, 1)
Write-Host "GATE GREEN  (${total}s total)" -ForegroundColor Green
Pop-Location
