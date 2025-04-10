python setup.py build_ext --inplace

python -m pybind11 --cmakedir
cd build
cmake .. -Dpybind11_DIR="C:/Python311/Lib/site-packages/pybind11/share/cmake/pybind11"
cmake --build .

cd build-linux
cmake ..
cmake --build .


