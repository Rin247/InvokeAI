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
  - Python 3.13 set as working local dev target; 3.14/3.15 marked WIP
- Removed obsolete model families: SD1.x, SD2.x, SDXL, FLUX.1, FLUX.2, SD3.5
- Retained maintained model families: Anima, Qwen Image 2.1, Krea 2, Ideogram 4, ERNIE-Image, ERNIE-Image-Turbo, Wan 2.2, Z-Image Turbo, Z-Image Base
- Old build/deploy scripts removed
- Windows/Linux debug path created:
  - `build.bat`
  - `run.bat`
  - `compile.bat`
- App made runnable on Windows for debugging by:
  - restoring kept-model backends (`pid`, `krea2`, `ideogram4`, `z_image`, `ernie_image`, `anima`, `flux`, `krea2`, `ideogram4`, `z_image`)
  - removing removed-model invocation files that were still imported
  - fixing imports/configs that blocked startup

## In Progress
- Verifying local build on Python 3.13
- Verifying Anima model loader compatibility with newer fine-tunes
- Verifying Qwen-Image 2.1 support

## Planned
- Context-menu “use model” and “use loras” frontend work
- Intel XPU refactor after the current model-family cleanup stabilizes
- Qt6 frontend refactor planned for a future `>v7.00` rebase
- Python 3.14/3.15 support once dependency wheels are available
