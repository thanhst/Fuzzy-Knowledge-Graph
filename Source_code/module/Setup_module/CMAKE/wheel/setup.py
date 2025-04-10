from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11

ext_modules = [
    Extension(
        "fisa_module",
        ["fisa_module.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["/std:c++20"] if __import__('platform').system() == "Windows" else ["-std=c++20"],
    ),
]

setup(
    name="fisa_module",
    version="0.1.0",
    author="Your Name",
    description="A test module built with pybind11",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
