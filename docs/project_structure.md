# Project Structure

This repo now has a clean top-level layout for new work while keeping legacy
scripts compatible.

## Main folders

- `modules/fkg_cpp`: native C++/CUDA FIS/FKG engine and pybind11 bindings.
- `modules/fkg_python`: Python runtime helpers and adapters around FKG.
- `runners`: executable workflows. Runners call modules; they do not own model
  code.
- `tools`: short Windows entrypoints for build, validation, and common runners.
- `Source`: build/package wrapper, flow scripts, tests, and current ICTA data
  pipeline assets.
- `Source_code`: legacy research/scenario code. Keep it stable unless a legacy
  scenario is being migrated.
- `GPU`: GPU build wrapper/output location.
- `result` and `Source/Data/result`: generated experiment outputs.

## Ownership rules

- Put C++/CUDA algorithm changes in `modules/fkg_cpp`.
- Put reusable Python FKG helpers in `modules/fkg_python`.
- Put one-click workflows in `runners`.
- Put batch wrappers and developer commands in `tools`.
- Do not add new runner logic to the C++ module folder.
- Do not hard-code `sys.path` and DLL paths in new runners; use
  `modules.fkg_python.fkg_runtime.module_loader`.

## Recommended commands

```bat
tools\build_fkg_cuda.bat
tools\test_backend.bat auto source
tools\validate_fkg_gpu.bat
tools\run_icta_flow.bat
tools\run_icta_flow.bat --under5
```

The old `.bat` files still work, but new work should use the `tools` and
`runners` folders.
