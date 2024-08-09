project('dlup', 'cpp', 'cython',
  version : '0.6.1',
  default_options : ['warning_level=3', 'cpp_std=c++17'])


py_mod = import('python')
py = py_mod.find_installation(pure: false)
py_dep = py.dependency()

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

# Capture the output of the run_command and convert to relative paths
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

install_dir = py.get_install_dir(pure: false)
install_data('README.md', install_dir : install_dir)
install_data('LICENSE', install_dir : install_dir)
install_subdir('dlup', install_dir : install_dir)

_background = py.extension_module('_background',
                                  'dlup/_background.pyx',
                                  include_directories : [incdir_numpy],
                                  subdir : 'dlup',
                                  install: true,
                                  cpp_args : ['-O3', '-march=native', '-ffast-math'])

# pybind11 extension
_libtiff_tiff_writer = py.extension_module('_libtiff_tiff_writer',
                                           'src/libtiff_tiff_writer.cpp',
                                           include_directories : [incdir_pybind11],
                                           install : true,
                                           subdir : 'dlup',
                                           cpp_args: ['-std=c++17', '-O3', '-march=native', '-ffast-math'],
                                           dependencies : [libtiff_dep])
