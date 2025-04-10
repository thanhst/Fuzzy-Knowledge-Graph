from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import os, sys, subprocess

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-Dpybind11_DIR={self._find_pybind11_cmake_dir()}",
        ]

        build_temp = self.build_temp
        os.makedirs(build_temp, exist_ok=True)

        # Run CMake
        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=build_temp)
        subprocess.check_call(["cmake", "--build", ".", "--config", "Release"], cwd=build_temp)

        def _find_pybind11_cmake_dir(self):
            try:
                import pybind11
                return pybind11.get_cmake_dir()
            except Exception as e:
                print("❌ Không tìm được pybind11 cmake dir:", e)
                raise


setup(
    name="fisa_module",
    version="0.1.0",
    author="Bạn",
    description="Mô-đun Python viết bằng C++ sử dụng pybind11",
    ext_modules=[CMakeExtension("fisa_module")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
