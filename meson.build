project(
  'dlup',
  'cpp',
  version: '0.9.1',
  default_options: [
    'cpp_std=c++20',
    'warning_level=3',
    'werror=false',
  ],
)

cpp = meson.get_compiler('cpp')

add_project_arguments([
  '-Wno-deprecated-declarations',
  '-Wno-unused-parameter',
], language: 'cpp')

python3 = import('python').find_installation('python3', pure: false)

incdir = include_directories('include')

numpy_inc = include_directories(
  run_command(
    python3,
    '-c',
    'import numpy; print(numpy.get_include())',
    check: true
  ).stdout().strip()
)

pybind11_dep = dependency('pybind11', required: true)
boost_dep = dependency('boost', modules: ['geometry'], required: true)

libtiff_dep = dependency('libtiff-4', required: false)
if not libtiff_dep.found()
  libtiff_dep = dependency('libtiff', required: true)
endif

zstd_dep = dependency('libzstd', required: false)

aifocore_proj = subproject('aifocore', default_options: [
  'default_library=static',
  'warning_level=0',
])
aifocore_dep = aifocore_proj.get_variable('aifocore_dep')

geometry_sources = files(
  'src/geometry/box.cpp',
  'src/geometry/collection.cpp',
  'src/geometry/marching_squares.cpp',
  'src/geometry/polygon.cpp',
  'src/geometry/transforms.cpp',
  'src/geometry/utilities.cpp',
)

geometry_lib = static_library(
  'dlup_geometry',
  geometry_sources,
  include_directories: [incdir],
  dependencies: [
    aifocore_dep,
    boost_dep,
  ],
  install: false,
)

geometry_link_args = []
if host_machine.system() == 'darwin'
  geometry_link_args = ['-Wl,-exported_symbol,_PyInit__geometry']
endif

python3.extension_module(
  '_geometry',
  'src/geometry/geometry_module.cpp',
  include_directories: [incdir, numpy_inc],
  dependencies: [
    pybind11_dep,
    aifocore_dep,
    boost_dep,
  ],
  link_with: geometry_lib,
  link_args: geometry_link_args,
  install: true,
  subdir: 'dlup',
)

libtiff_link_args = []
if host_machine.system() == 'darwin'
  libtiff_link_args = ['-Wl,-exported_symbol,_PyInit__libtiff_tiff_writer']
endif

libtiff_cpp_args = []
libtiff_deps = [pybind11_dep, libtiff_dep]
if zstd_dep.found()
  libtiff_cpp_args = ['-DHAVE_ZSTD']
  libtiff_deps += [zstd_dep]
endif

python3.extension_module(
  '_libtiff_tiff_writer',
  'src/_libtiff_tiff_writer.cpp',
  include_directories: [incdir, numpy_inc],
  dependencies: libtiff_deps,
  link_args: libtiff_link_args,
  cpp_args: libtiff_cpp_args,
  install: true,
  subdir: 'dlup',
)

background_link_args = []
if host_machine.system() == 'darwin'
  background_link_args = ['-Wl,-exported_symbol,_PyInit__background']
endif

python3.extension_module(
  '_background',
  'src/_background.cpp',
  include_directories: [incdir, numpy_inc],
  dependencies: [pybind11_dep],
  link_args: background_link_args,
  install: true,
  subdir: 'dlup',
)

install_subdir(
  'dlup',
  install_dir: python3.get_install_dir(),
  exclude_files: ['*.cpp', '*.h', 'meson.build'],
  exclude_directories: ['__pycache__'],
)

