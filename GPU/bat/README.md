# GPU Bat Scripts

Các script build/test liên quan GPU được gom tại đây:

- `Install_MSYS2_Dependencies.bat`: cài/kiểm tra dependency qua MSYS2.
- `Build_FKG_CUDA.bat`: build CUDA chính (Windows cần MSVC + nvcc).
- `Build_FKG_CUDA_MSYS2.bat`: chạy bước MSYS2 trước rồi build CUDA.
- `Test_Backend_GPU_CPU.bat`: test runtime chọn backend CPU/GPU.
- `Build_Wheel.bat`: build wheel theo backend (`cpu` hoặc `gpu`).

Các file `.bat` cùng tên ở thư mục root chỉ còn là wrapper để tương thích lệnh cũ.
