### third_party/meson.build ###

py_mod = import('python')
py = py_mod.find_installation(pure: false)
py_dep = py.dependency()

python_path = run_command(py, ['-c', 'import sys; print(sys.executable)'], check: true).stdout().strip()
message('Using Python: ' + python_path)

# LibTIFF
libtiff_dep = dependency('libtiff-4', required : false)
if not libtiff_dep.found()
    libtiff_dep = dependency('tiff', required : false)
endif
if not libtiff_dep.found()
    libtiff_dep = cc.find_library('tiff', required : false)
endif
if not libtiff_dep.found()
    error('libtiff not found. Please install libtiff development files.')
endif

# ZSTD
zstd_dep = dependency('libzstd', required : false)
have_zstd = zstd_dep.found()
if have_zstd
    message('ZSTD support enabled')
else
    message('ZSTD support disabled')
endif

# Numpy and PYBIND11
numpy_include = run_command(py, ['-c', '''
import os
import numpy
print(os.path.relpath(numpy.get_include(), os.getcwd()))
'''], check: true).stdout().strip()

pybind11_include = run_command(py, ['-c', '''
import os
import pybind11
print(os.path.relpath(pybind11.get_include(), os.getcwd()))
'''], check: true).stdout().strip()

incdir_numpy = include_directories(numpy_include)
incdir_pybind11 = include_directories(pybind11_include)

# Find Boost with necessary components
boost_modules = ['system', 'serialization']
boost_dep = dependency('boost', modules : boost_modules, required : true)

# OpenCV
opencv_dep = dependency('opencv4', required : true)

# Shared dependencies
base_deps = [libtiff_dep]
if have_zstd
    base_deps += [zstd_dep]
    cpp_args += ['-DHAVE_ZSTD']
endif

# Export dependencies and includes
incdir_numpy = incdir_numpy
incdir_pybind11 = incdir_pybind11
base_deps = base_deps
boost_dep = boost_dep
opencv_dep = opencv_dep
