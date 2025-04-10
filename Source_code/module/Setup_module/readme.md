python setup.py build_ext --inplace

# window
python -m pybind11 --cmakedir
cd build
cmake .. -Dpybind11_DIR="C:/Python311/Lib/site-packages/pybind11/share/cmake/pybind11"
cmake --build .

# linux
cd build-linux
cmake ..
cmake --build .

# install
cd \Source_code\module\Setup_module\wheelhouse
pip install fisa_module --find-links=.


