# GPU Source Wrapper

Folder này giữ cấu hình build dành cho pipeline GPU.

- `CMakeLists.txt` ở đây chỉ đóng vai trò wrapper.
- Code C++ thực tế vẫn nằm ở `Source/`.
- Output module Python (`fisa_module*.pyd`) của pipeline GPU sẽ được đặt tại `GPU/Source/`.
