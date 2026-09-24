# Build / Run (Windows)

- Prereqs: `uv` (0.12.18+), Git
- From repo root:
  - `build.bat` — installs runtime deps and the project in editable mode
  - `run.bat` — runs `invokeai.app.run_app`
  - `compile.bat` — frontend typegen/check (requires Node deps)

Python versions:
- 3.13: working local development target
- 3.14 / 3.15: WIP, not yet fully supported due to missing dependency wheels

Notes:
- Current debug path uses CPU PyTorch until CUDA tooling is available
- Frontend UI mount is skipped if `invokeai\frontend\web\dist` is missing
