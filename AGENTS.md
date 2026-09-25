# Agent Instructions

## Package Management

This project uses **pnpm** exclusively for package management in the frontend (`invokeai/frontend/web/`).

- ✅ Use `pnpm` commands (e.g., `pnpm install`, `pnpm run`)
- ❌ Never use `npm` or `yarn` commands
- ❌ Never suggest creating or using `package-lock.json` or `yarn.lock`
- ✅ The lock file is `pnpm-lock.yaml`

Use the following pnpm commands for typical operations:

- pnpm -C invokeai/frontend/web install
- pnpm -C invokeai/frontend/web build
- pnpm -C invokeai/frontend/web lint:tsc
- pnpm -C invokeai/frontend/web lint:dpdm
- pnpm -C invokeai/frontend/web lint:eslint
- pnpm -C invokeai/frontend/web lint:prettier

## Project Structure

- Backend: Python in `invokeai/`
- Frontend: TypeScript/React in `invokeai/frontend/web/` (uses pnpm)

## Active Branch Context

Current primary work is on the `qt6-migration` branch.

- Qt6 migration is in progress
- Local GUI mode uses PySide6
- macOS/darwin support has been removed
- Minimum platform versions: Windows 11+, Ubuntu 24.04+, Arch Linux rolling
- Python: 3.13 stable; 3.14/3.15 are WIP pending Qt6 validation
- Key dependency changes: `onnx>=1.23.0`, `PySide6` added, `numpy` unpinned, `torch>=2.14.0,<3.0`
- ROCm 10.0 and Intel XPU support are present in `pyproject.toml`
- FLUX/SD/SDXL backend trees have been removed; remaining architecture-specific imports are being cleaned up to restore server startup
