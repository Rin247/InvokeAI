# InvokeAI Fork — TODO

## Done
- [x] Fork and trim README
- [x] Merge `qwen-image-2.1`
- [x] Remove ROCm and macOS from `pyproject.toml`
- [x] Create `build.bat`, `run.bat`, `compile.bat`
- [x] Restore kept-model backends needed for Windows debug
- [x] Remove removed-model invocation files still imported
- [x] Document Qt6 refactor as `>v7.00`

## In Progress
- [ ] Remove FLUX/FLUX2 backend config classes and imports
- [ ] Remove FLUX/FLUX2 frontend types, schema, and generators
- [ ] Remove FLUX/FLUX2 tests and fixtures
- [ ] Remove SD1/SD2/SDXL/SD3.5 dead references

## Next
- [ ] Context-menu “use model” and “use loras”
- [ ] Intel XPU refactor after model-family cleanup
- [ ] Qt6 frontend refactor (`>v7.00`)
- [ ] Final Windows/Linux debug verification
- [ ] Commit and push
