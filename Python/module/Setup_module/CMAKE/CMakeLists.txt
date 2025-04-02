cmake_minimum_required(VERSION 3.10)
project(FisaModule)

# Đảm bảo dùng đúng compiler và tiêu chuẩn C++
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Tìm Python
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

# Xác định thư mục include & phiên bản Python
execute_process(COMMAND ${Python3_EXECUTABLE} -c 
    "import sys; print('.'.join(map(str, sys.version_info[:2])))"
    OUTPUT_VARIABLE PYTHON_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)

# 🔹 Nếu lỗi vẫn xảy ra, thử **tắt** Py_LIMITED_API để đảm bảo tương thích
# add_definitions(-DPy_LIMITED_API=0x03060000)

# Tạo module Python
pybind11_add_module(fisa_module fisa_module.cpp)

# Đảm bảo module có thể chạy với mọi Python 3.x
if(WIN32)
    set_target_properties(fisa_module PROPERTIES SUFFIX ".pyd" PREFIX "")
else()
    set_target_properties(fisa_module PROPERTIES SUFFIX ".so" PREFIX "")
endif()

message(STATUS "Building for Python ${PYTHON_VERSION}")
