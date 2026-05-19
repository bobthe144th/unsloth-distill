# Unsloth Studio

Desktop app for [Unsloth](https://unsloth.ai) — fine-tune and run large language models on your own machine.

## Download

Get the latest installer from the [Releases](../../releases) page:

| Platform | File |
|---|---|
| Windows | `Unsloth Studio (Desktop)_*_x64-setup.exe` |
| macOS (Apple Silicon) | `Unsloth Studio (Desktop)_*.dmg` |
| Linux (Ubuntu/Debian) | `unsloth-studio-desktop_*_amd64.deb` |
| Linux (universal) | `unsloth-studio-desktop_*_amd64.AppImage` |

> **Windows:** SmartScreen may show an "unknown publisher" warning on unsigned dev builds — click **More info → Run anyway**.

## First-run setup

The installer does not bundle Python. On first launch the app shows a setup screen — click **Install** to download the Python environment automatically. This takes a few minutes and requires an internet connection.

The Python environment is installed to `~/.unsloth/studio/` and is not touched by app updates.

## Build from source

### Prerequisites

| Tool | Version |
|---|---|
| Node.js | ≥ 22.12.0 |
| Rust (stable) | via [rustup](https://rustup.rs) |
| Python | 3.9 – 3.13 |

### Development build (Linux)

```bash
# Install system libraries (Ubuntu/Debian)
sudo apt-get install -y libwebkit2gtk-4.1-dev libgtk-3-dev \
  libayatana-appindicator3-dev librsvg2-dev libxdo-dev \
  libssl-dev libdbus-1-dev libwayland-dev patchelf

# Install Tauri CLI
npm install --save-dev --prefix studio @tauri-apps/cli@2.10.1

# Build frontend
cd studio/frontend && npm ci && npm run build && cd ../..

# Build and run (debug binary, no installer)
cd studio && npx tauri dev
```

### Unsigned Windows installer (CI)

Trigger the **Build Windows .exe (dev, unsigned)** workflow from the Actions tab. The resulting NSIS installer is uploaded as a downloadable artifact.

### Signed production release

Use the **Release Desktop App** workflow (`workflow_dispatch`) with a `studio_version` SemVer tag and the corresponding PyPI `unsloth` version. Requires Azure Trusted Signing, Apple Developer, and Tauri signing secrets configured in repository settings.

## Project layout

```
studio/
  frontend/        React 19 + Vite + TypeScript UI
  src-tauri/       Tauri 2 Rust shell (window, backend spawn, auto-update)
  backend/         FastAPI Python backend (spawned at runtime from installed unsloth)
  install.sh       Linux/macOS environment installer (bundled in app)
  install.ps1      Windows environment installer (bundled in app)
  scripts/         CI helper scripts
.github/workflows/
  studio-tauri-smoke.yml    PR smoke build (Linux debug, no codesign)
  studio-frontend-ci.yml    Frontend typecheck + bundle sanity
  studio-backend-ci.yml     Python backend pytest matrix
  build-windows-dev.yml     On-demand unsigned Windows .exe
  release-desktop.yml       Signed production release (all platforms)
  security-audit.yml        Supply-chain audit (pip + npm + cargo)
  lint-ci.yml               Python + shell + YAML + JSON lint
```

## License

AGPL-3.0-only — see [`studio/`](studio/) for source.
