# GPU Source Wrapper

This folder keeps the GPU build wrapper and optional GPU-local output.

- `CMakeLists.txt` only delegates to the shared `Source` build wrapper.
- Native C++/CUDA source lives in `modules/fkg_cpp/`.
- GPU build output (`fisa_module*.pyd`) can be placed in `GPU/Source/`.
