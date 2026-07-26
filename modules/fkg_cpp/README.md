# FKG C++ Module

This folder owns the native FIS/FKG engine code.

- `include/`: C++ public headers used by the pybind11 extension.
- `src/`: CPU/CUDA implementation files.
- `python/bindings.cpp`: Python binding layer that exposes `fisa_module`.

Build configuration remains in `Source/CMakeLists.txt` so existing scripts can
still output `fisa_module*.pyd` into `Source/` and `GPU/Source/`.

Use:

```bat
tools\build_fkg_cuda.bat --fallback-cpu
tools\test_backend.bat auto source
```
