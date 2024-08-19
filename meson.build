project('dlup', 'cpp', 'cython',
  version : '0.7.0',
  default_options : ['buildtype=release', 'warning_level=3', 'cpp_std=c++17'])

cpp_args = ['-O3', '-march=native', '-ffast-math', '-funroll-loops', '-flto', '-pipe', '-fomit-frame-pointer']
link_args = ['-flto']

b_unity = true


### Includes ####

py_mod = import('python')
py = py_mod.find_installation(pure: false)
py_dep = py.dependency()

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

# Numpy and PYBIND
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

# Use the relative paths
incdir_numpy = include_directories(numpy_include)
incdir_pybind11 = include_directories(pybind11_include)

# Find Boost with necessary components
boost_modules = ['system', 'serialization']
boost_dep = dependency('boost', modules : boost_modules, required : true)

# OpenCV
opencv_dep = dependency('opencv4', required : true)

### End Includes ###


_background = py.extension_module('_background',
                                  'dlup/_background.pyx',
                                  include_directories : [incdir_numpy],
                                  install : true,
                                  subdir : '',
                                  link_args : link_args,
                                  c_args : cpp_args + ['-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION']
)

# Define the base dependencies and compiler arguments

# Add ZSTD support if available
base_deps = [libtiff_dep]
if have_zstd
    base_deps += [zstd_dep]
    cpp_args += ['-DHAVE_ZSTD']
endif

# tiff writer extension
_libtiff_tiff_writer = py.extension_module('_libtiff_tiff_writer',
                                           'src/libtiff_tiff_writer.cpp',
                                           include_directories : [incdir_pybind11],
                                           install : true,
                                           cpp_args : cpp_args,
                                           link_args : link_args,
                                           dependencies : base_deps)

_geometry = py.extension_module('_geometry',
                                'src/geometry.cpp',
                                include_directories : [incdir_pybind11],
                                install : true,
                                cpp_args : cpp_args,
                                link_args : link_args,
                                dependencies : base_deps + boost_dep + opencv_dep)
