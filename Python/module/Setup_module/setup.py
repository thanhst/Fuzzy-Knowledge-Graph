from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "fisa_module",
        ["fisa_module.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++"
    ),
]

setup(
    name="fisa_module",
    ext_modules=ext_modules,
)
