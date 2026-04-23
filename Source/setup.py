#!/usr/bin/env python
"""
Wheel/extension build for fisa_module via CMake.

Environment variables:
  USE_CUDA=ON|OFF        Enable CUDA backend in this build.
  USE_GPU=ON|OFF         Enable SYCL backend in this build.
  CUDAToolkit_ROOT=...   Optional CUDA toolkit path.
  CMAKE_GENERATOR=...    Optional CMake generator override.
  CMAKE_BUILD_PARALLEL_LEVEL=N
"""

from __future__ import annotations

import os
import pathlib
import platform
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

import pybind11


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = ""):
        super().__init__(name, sources=[])
        self.sourcedir = str(pathlib.Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def run(self) -> None:
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext: CMakeExtension) -> None:
        ext_fullpath = pathlib.Path(self.get_ext_fullpath(ext.name)).resolve()
        extdir = ext_fullpath.parent

        build_temp = pathlib.Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)

        cfg = "Release"
        use_cuda = os.environ.get("USE_CUDA", "OFF").upper()
        use_gpu = os.environ.get("USE_GPU", "OFF").upper()

        cmake_args = [
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-Dpybind11_DIR={pybind11.get_cmake_dir()}",
            f"-DFISA_OUTPUT_DIR={extdir}",
            f"-DUSE_CUDA={use_cuda}",
            f"-DUSE_GPU={use_gpu}",
        ]

        cuda_root = os.environ.get("CUDAToolkit_ROOT", "").strip()
        if cuda_root:
            cmake_args.append(f"-DCUDAToolkit_ROOT={cuda_root}")

        generator = os.environ.get("CMAKE_GENERATOR", "").strip()
        if not generator and platform.system() == "Windows":
            generator = "Visual Studio 17 2022"
        if generator:
            cmake_args.extend(["-G", generator])
            if generator.startswith("Visual Studio"):
                cmake_args.extend(["-A", "x64"])

        build_args = ["--config", cfg]
        parallel_level = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", "").strip()
        if parallel_level:
            build_args.extend(["--parallel", parallel_level])

        subprocess.check_call(["cmake", ext.sourcedir, *cmake_args], cwd=build_temp)
        subprocess.check_call(["cmake", "--build", ".", *build_args], cwd=build_temp)


setup(
    name="fisa-module",
    version=(
        "2.1.0"
        if not os.environ.get("FISA_LOCAL_VERSION", "").strip()
        else f"2.1.0+{os.environ.get('FISA_LOCAL_VERSION', '').strip().replace('-', '.')}"
    ),
    author="FISA Team",
    description="FKG/FIS C++ extension with CPU/GPU backend selection",
    ext_modules=[CMakeExtension("fisa_module", ".")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "pybind11>=2.10.0",
    ],
)
