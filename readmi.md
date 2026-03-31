# Báo cáo tiến độ - nhánh thanh/GPU_version

Cập nhật: 31/03/2026

## Tóm tắt nhanh
Mình vừa hoàn thành phần build module C++ sang `.pyd` cho Python, đồng thời test xong luồng chọn backend CPU/GPU có fallback an toàn.

## Đã làm xong những gì
- Sửa API để chọn backend bằng biến đầu vào: `set_use_gpu`, `get_use_gpu`, `is_using_gpu`.
- Bổ sung kiểm tra runtime:
  - module có build kèm GPU hay không,
  - máy có GPU khả dụng hay không.
- Nếu yêu cầu GPU nhưng không đáp ứng điều kiện, hệ thống tự in thông báo và chuyển sang CPU.
- Chuẩn hóa pipeline `.bat` cho build/test:
  - `Build_FISA_CPU.bat`
  - `Build_FISA_CUDA.bat`
  - `Test_Backend_GPU_CPU.bat`
- Fix lỗi crash trong `FKG::train` (tràn chỉ số khi tạo tổ hợp 4 thuộc tính).

## Kết quả build và thử nghiệm
- Build CPU: thành công.
- File đầu ra: `Source/fisa_module.cp314-mingw_x86_64_msvcrt_gnu.pyd`.
- Số bài test backend đã chạy: 2 bài.
- Test `--backend cpu`: PASS.
- Test `--backend gpu`: PASS theo đúng cơ chế fallback, có in thông báo chuyển sang CPU khi không có GPU/build GPU.
- Test bộ ICTA (anh Đức Hoàng): đã chạy xong bằng script `Source/tests/test_icta_gpu.py`
  - dataset: `Source_code/data/ICTA/ICTA.csv`
  - backend request: `gpu`
  - backend used: `gpu`
  - GPU compiled/available: `True/True`
  - train time: `153.948 ms`
  - infer time: `39.825 ms` (231 mẫu test)
  - accuracy: `65.80%`

## Trạng thái hiện tại
Build vừa mới hoàn tất, test backend ổn định, và đã có kết quả chạy thực tế trên ICTA.
