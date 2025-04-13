python setup.py bdist_wheel
cd dist
pip install fisa_module-0.1.0-cp311-cp311-win_amd64.whl

cibuildwheel --platform windows --output-dir wheelhouse/window

cd \Source_code\module\Setup_module\wheelhouse
pip install fisa_module --find-links=.