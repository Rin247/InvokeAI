# InvokeAI Fork — Summary

## Goal
Prepare a maintained, locally-runnable fork of InvokeAI focused on currently supported model families, targeting Windows and Linux, with CUDA as the primary backend.

## Completed
- Fork created at `https://github.com/Rin247/InvokeAI`
- `qwen-image-2.1` branch merged into `main`
- `README.md` trimmed: Contributing/sponsors sections removed; model list refocused on maintained families; Qt6 refactor noted as `>v7.00`
- `pyproject.toml` updated:
  - ROCm removed
  - macOS removed
  - Windows classifiers focused on Windows 11+
  - Platform scope narrowed to Windows and Linux
- Windows/Linux debug path created:
  - `build.bat`
  - `run.bat`
  - `compile.bat`
- App made runnable on Windows for debugging by:
  - restoring kept-model backends (`pid`, `krea2`, `ideogram4`, `z_image`, `ernie_image`, `anima`, `flux`, `krea2`, `ideogram4`, `z_image`)
  - removing removed-model invocation files that were still imported
  - fixing imports/configs that blocked startup

## In Progress
- Removing FLUX/FLUX2 references from backend configs and frontend
- Removing remaining SD1/SD2/SDXL/SD3.5 dead references
- Keeping the Windows/Linux debug build runnable while cleaning

## Planned
- Context-menu “use model” and “use loras” frontend work
- Intel XPU refactor after the current model-family cleanup stabilizes
- Qt6 frontend refactor planned for a future `>v7.00` rebase
