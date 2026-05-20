# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
#
# Unsloth Studio environment installer for Windows (invoked by the Tauri desktop app).
# Usage: install.ps1 --tauri [--local]
#
# Creates %USERPROFILE%\.unsloth\studio\unsloth_studio\ with a Python venv and installs
# the `unsloth` package so the desktop app can spawn:
#   unsloth studio --api-only -H 127.0.0.1 -p <port>
#
# Protocol lines written to stdout (parsed by install.rs):
#   [TAURI:STEP] <name>        -- current installation phase
#   [TAURI:PROGRESS] <detail>  -- human-readable progress detail
#   [TAURI:DIAG] <marker>      -- machine-readable diagnostic marker
#
# Exit codes:
#   0  success
#   1  non-recoverable failure

param(
    [switch]$Tauri,
    [switch]$Local
)

$ErrorActionPreference = 'Stop'
$ProgressPreference    = 'SilentlyContinue'  # suppress slow Write-Progress bars

function Write-Step($msg)     { Write-Output "[TAURI:STEP] $msg" }
function Write-Progress2($msg) { Write-Output "[TAURI:PROGRESS] $msg" }
function Write-Diag($msg)     { Write-Output "[TAURI:DIAG] $msg" }
function Write-Fail($msg)     { Write-Output "[TAURI:PROGRESS] ERROR: $msg"; exit 1 }

$MinVersion  = "2026.5.5"
$StudioHome  = "$env:USERPROFILE\.unsloth\studio"
$VenvPath    = "$StudioHome\unsloth_studio"

# ── Prepare ──────────────────────────────────────────────────────────────────
Write-Step "Preparing"
Write-Progress2 "Setting up Unsloth Studio environment at $StudioHome"
$null = New-Item -ItemType Directory -Force -Path $StudioHome
Write-Diag "studio_home_ready"

# ── Find Python 3.9+ ─────────────────────────────────────────────────────────
Write-Step "Checking Python"
$PythonExe = $null

foreach ($candidate in @('python', 'python3', 'py -3')) {
    try {
        $verStr = Invoke-Expression "$candidate -c `"import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')`"" 2>$null
        if (-not $verStr) { continue }
        $parts  = $verStr.Trim().Split('.')
        $major  = [int]$parts[0]
        $minor  = [int]$parts[1]
        if ($major -ge 3 -and $minor -ge 9) {
            $PythonExe = $candidate
            break
        }
    } catch {
        continue
    }
}

if (-not $PythonExe) {
    Write-Fail "Python 3.9 or later is required. Download from https://python.org and retry."
}

$pyVer = Invoke-Expression "$PythonExe --version" 2>&1
Write-Progress2 "Using $pyVer"
Write-Diag "python_found"

# ── Create virtual environment ────────────────────────────────────────────────
Write-Step "Creating virtual environment"
Write-Progress2 "Creating environment at $VenvPath"

Invoke-Expression "$PythonExe -m venv `"$VenvPath`""
if ($LASTEXITCODE -ne 0) { Write-Fail "Failed to create Python virtual environment" }
Write-Diag "venv_created"

$Pip     = "$VenvPath\Scripts\pip.exe"
$Unsloth = "$VenvPath\Scripts\unsloth.exe"

# ── Upgrade pip ───────────────────────────────────────────────────────────────
Write-Step "Upgrading pip"
& $Pip install --upgrade pip --quiet
if ($LASTEXITCODE -ne 0) { Write-Fail "Failed to upgrade pip" }
Write-Diag "pip_upgraded"

# ── Install Unsloth ───────────────────────────────────────────────────────────
Write-Step "Installing Unsloth Studio"
Write-Progress2 "Downloading and installing unsloth>=$MinVersion (this may take a few minutes)..."
& $Pip install "unsloth>=$MinVersion" --quiet
if ($LASTEXITCODE -ne 0) { Write-Fail "Failed to install unsloth" }
Write-Diag "unsloth_installed"

# ── Verify ────────────────────────────────────────────────────────────────────
Write-Step "Verifying installation"
if (-not (Test-Path $Unsloth)) {
    Write-Fail "unsloth.exe not found at $Unsloth after installation"
}
Write-Diag "verification_complete"

Write-Step "Done"
Write-Progress2 "Unsloth Studio environment is ready"
