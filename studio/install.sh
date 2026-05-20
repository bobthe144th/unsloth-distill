#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
#
# Unsloth Studio environment installer (invoked by the Tauri desktop app).
# Usage: install.sh --tauri [--local]
#
# Creates ~/.unsloth/studio/unsloth_studio/ with a Python venv and installs
# the `unsloth` package so the desktop app can spawn:
#   unsloth studio --api-only -H 127.0.0.1 -p <port>
#
# Protocol lines written to stdout (parsed by install.rs):
#   [TAURI:STEP] <name>        — current installation phase
#   [TAURI:PROGRESS] <detail>  — human-readable progress detail
#   [TAURI:DIAG] <marker>      — machine-readable diagnostic marker
#   [TAURI:NEED_SUDO] <pkg...> — (Linux) apt packages needing elevation; exits 2
#
# Exit codes:
#   0  success
#   1  non-recoverable failure
#   2  elevated system-package install required (Linux only)

set -euo pipefail

TAURI=false
LOCAL=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tauri) TAURI=true ;;
        --local) LOCAL=true ;;
        *) ;;
    esac
    shift
done

step()     { echo "[TAURI:STEP] $1"; }
progress() { echo "[TAURI:PROGRESS] $1"; }
diag()     { echo "[TAURI:DIAG] $1"; }
fail()     { echo "[TAURI:PROGRESS] ERROR: $1" >&2; exit 1; }

STUDIO_HOME="${HOME}/.unsloth/studio"
VENV_PATH="${STUDIO_HOME}/unsloth_studio"
MIN_VERSION="2026.5.5"

# ── Prepare ──────────────────────────────────────────────────────────────────
step "Preparing"
progress "Setting up Unsloth Studio environment at ${STUDIO_HOME}"
mkdir -p "${STUDIO_HOME}"
diag "studio_home_ready"

# ── Find Python 3.9+ ─────────────────────────────────────────────────────────
step "Checking Python"
PYTHON=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3.9 python3 python; do
    if ! command -v "${candidate}" >/dev/null 2>&1; then
        continue
    fi
    ver=$("${candidate}" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || true)
    if [[ -z "${ver}" ]]; then
        continue
    fi
    major="${ver%%.*}"
    minor="${ver#*.}"
    if [[ "${major}" -ge 3 && "${minor}" -ge 9 ]]; then
        PYTHON="${candidate}"
        break
    fi
done

if [[ -z "${PYTHON}" ]]; then
    fail "Python 3.9 or later is required. Install Python from python.org and retry."
fi
py_ver=$("${PYTHON}" --version 2>&1)
progress "Using ${py_ver}"
diag "python_found"

# ── Check for required system libraries (Linux) ───────────────────────────────
if [[ "$(uname -s)" == "Linux" ]]; then
    step "Checking system libraries"
    MISSING_PKGS=()

    for pkg_check in "openssl:libssl-dev" "ffi.h:libffi-dev"; do
        header="${pkg_check%%:*}"
        pkg="${pkg_check##*:}"
        if ! "${PYTHON}" -c "import ctypes; ctypes.cdll.LoadLibrary('libssl.so.3')" 2>/dev/null \
           && ! "${PYTHON}" -c "import ssl" 2>/dev/null; then
            MISSING_PKGS+=("${pkg}")
        fi
    done

    if [[ ${#MISSING_PKGS[@]} -gt 0 ]]; then
        echo "[TAURI:NEED_SUDO] ${MISSING_PKGS[*]}"
        exit 2
    fi
    diag "system_libs_ok"
fi

# ── Create virtual environment ────────────────────────────────────────────────
step "Creating virtual environment"
progress "Creating environment at ${VENV_PATH}"
"${PYTHON}" -m venv "${VENV_PATH}"
diag "venv_created"

PIP="${VENV_PATH}/bin/pip"

step "Upgrading pip"
"${PIP}" install --upgrade pip --quiet
diag "pip_upgraded"

# ── Install Unsloth ───────────────────────────────────────────────────────────
step "Installing Unsloth Studio"
progress "Downloading and installing unsloth>=${MIN_VERSION} (this may take a few minutes)..."
"${PIP}" install "unsloth>=${MIN_VERSION}" --quiet
diag "unsloth_installed"

# ── Verify ────────────────────────────────────────────────────────────────────
step "Verifying installation"
UNSLOTH_BIN="${VENV_PATH}/bin/unsloth"
if [[ ! -x "${UNSLOTH_BIN}" ]]; then
    fail "unsloth binary not found at ${UNSLOTH_BIN} after installation"
fi
diag "verification_complete"

step "Done"
progress "Unsloth Studio environment is ready"
