project('dlup', 'cpp', 'cython',
  version : '0.7.0',
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

# Check for ZSTD library
zstd_dep = dependency('libzstd', required : false)
have_zstd = zstd_dep.found()
if have_zstd
    message('ZSTD support enabled')
else
    message('ZSTD support disabled')
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

# Check if the CODECOV_CI environment variable is set to 'true'
codecov_ci = run_command('sh', ['-c', '''
if [ "$CODECOV_CI" = "true" ]; then
    echo "true"
else
    echo "false"
fi
'''], check: true).stdout().strip()

# Conditionally set install_dir based on the CODECOV_CI variable
if codecov_ci == 'true'
    install_dir = run_command('sh', ['-c', '''
    if [ -z "$PYTHONPATH" ]; then
        python -c "import sysconfig; print(sysconfig.get_path('purelib'))"
    else
        echo $PYTHONPATH
    fi
    '''], check: true).stdout().strip()

    # Ensure the install_dir is not empty
    if install_dir == ''
        error('Could not determine install_dir from PYTHONPATH or sysconfig.get_path().')
    endif
else
    install_dir = py.get_install_dir(pure: false)
endif

message('Installing to: ' + install_dir)
install_subdir('dlup', install_dir : install_dir)

_background = py.extension_module('_background',
                                  'dlup/_background.pyx',
                                  include_directories : [incdir_numpy],
                                  install : true,
                                  install_dir : install_dir / 'dlup',
                                  cpp_args : ['-O3', '-march=native', '-ffast-math'])

# Define the base dependencies and compiler arguments
base_deps = [libtiff_dep]
base_cpp_args = ['-std=c++17', '-O3', '-march=native', '-ffast-math']

# Add ZSTD support if available
if have_zstd
    base_deps += [zstd_dep]
    base_cpp_args += ['-DHAVE_ZSTD']
endif

# pybind11 extension
_libtiff_tiff_writer = py.extension_module('_libtiff_tiff_writer',
                                           'src/libtiff_tiff_writer.cpp',
                                           include_directories : [incdir_pybind11],
                                           install : true,
                                           install_dir : install_dir / 'dlup',
                                           cpp_args : base_cpp_args,
                                           dependencies : base_deps)
