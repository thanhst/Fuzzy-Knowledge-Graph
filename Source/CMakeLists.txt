cmake_minimum_required(VERSION 3.18)
project(FisaModule VERSION 2.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(USE_CUDA "Enable NVIDIA CUDA backend" OFF)
option(USE_GPU "Enable SYCL backend" OFF)
set(FISA_OUTPUT_DIR "${FISA_OUTPUT_DIR}" CACHE PATH "Output directory for fisa_module")
if(NOT FISA_OUTPUT_DIR)
    set(FISA_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(PYBIND11_CMAKE_DIR)
        list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
        find_package(pybind11 CONFIG REQUIRED)
    else()
        message(FATAL_ERROR "pybind11 not found. Install with: pip install pybind11")
    endif()
endif()

find_package(OpenMP QUIET)

set(GPU_COMPILE_DEFS FUZZY_USE_CUDA=0 FUZZY_USE_GPU=0)
set(GPU_LINK_LIBS "")
set(GPU_INCLUDE_DIRS "")
set(GPU_BACKEND_NAME "CPU")

if(USE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 14)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        find_package(CUDAToolkit REQUIRED)
        list(REMOVE_ITEM GPU_COMPILE_DEFS FUZZY_USE_CUDA=0)
        list(APPEND GPU_COMPILE_DEFS FUZZY_USE_CUDA=1)
        list(APPEND GPU_LINK_LIBS CUDA::cudart)
        set(GPU_BACKEND_NAME "CUDA")
    else()
        message(WARNING "USE_CUDA=ON but CUDA compiler was not found. Falling back to CPU.")
    endif()
elseif(USE_GPU)
    find_path(SYCL_INCLUDE_DIR "sycl/sycl.hpp"
        PATHS
            "$ENV{ONEAPI_ROOT}/compiler/latest/include"
            "C:/Program Files (x86)/Intel/oneAPI/compiler/latest/include"
    )
    if(SYCL_INCLUDE_DIR)
        list(REMOVE_ITEM GPU_COMPILE_DEFS FUZZY_USE_GPU=0)
        list(APPEND GPU_COMPILE_DEFS FUZZY_USE_GPU=1)
        list(APPEND GPU_INCLUDE_DIRS ${SYCL_INCLUDE_DIR})
        set(GPU_BACKEND_NAME "SYCL")
    else()
        message(WARNING "USE_GPU=ON but SYCL headers were not found. Falling back to CPU.")
    endif()
endif()

set(SOURCE_FILES
    Src/FKG.cpp
    Src/FIS.cpp
    Src/Utils.cpp
)

if(USE_CUDA AND CMAKE_CUDA_COMPILER)
    list(APPEND SOURCE_FILES
        Src/FKG_CUDA_Kernels.cu
    )
endif()

set(HEADER_FILES
    Include/FKG.h
    Include/FIS.h
    Include/FKG_CUDA_Kernels.h
    Include/Utils.h
)

pybind11_add_module(fisa_module MODULE
    Python/bindings.cpp
    ${SOURCE_FILES}
    ${HEADER_FILES}
)

target_include_directories(fisa_module PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/Include
    ${GPU_INCLUDE_DIRS}
)

target_compile_definitions(fisa_module PRIVATE ${GPU_COMPILE_DEFS})

if(OpenMP_CXX_FOUND)
    target_link_libraries(fisa_module PRIVATE OpenMP::OpenMP_CXX)
endif()

if(GPU_LINK_LIBS)
    target_link_libraries(fisa_module PRIVATE ${GPU_LINK_LIBS})
endif()

if(MSVC)
    target_compile_options(fisa_module PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/O2 /DNDEBUG /D_CRT_SECURE_NO_WARNINGS>
    )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(fisa_module PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -ffast-math -funroll-loops -DNDEBUG>
    )
endif()

if(USE_CUDA AND CMAKE_CUDA_COMPILER)
    target_compile_options(fisa_module PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -lineinfo>
    )
endif()

if(MINGW)
    # Python.org on Windows ships MSVC import libs; MinGW can still build the
    # extension by deferring unresolved Python symbols to runtime.
    target_link_options(fisa_module PRIVATE -Wl,--allow-shlib-undefined)
endif()

set_target_properties(fisa_module PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${FISA_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${FISA_OUTPUT_DIR}
    OUTPUT_NAME "fisa_module"
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

foreach(_cfg IN ITEMS Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER "${_cfg}" _cfg_upper)
    set_target_properties(fisa_module PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY_${_cfg_upper} ${FISA_OUTPUT_DIR}
        RUNTIME_OUTPUT_DIRECTORY_${_cfg_upper} ${FISA_OUTPUT_DIR}
    )
endforeach()

install(TARGETS fisa_module
    LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}
)

message(STATUS "===========================================")
message(STATUS "FisaModule Configuration")
message(STATUS "===========================================")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Python: ${Python3_EXECUTABLE} (${Python3_VERSION})")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "OpenMP: ${OpenMP_CXX_FOUND}")
message(STATUS "Requested backend: USE_CUDA=${USE_CUDA}, USE_GPU=${USE_GPU}")
message(STATUS "Effective backend: ${GPU_BACKEND_NAME}")
message(STATUS "===========================================")
