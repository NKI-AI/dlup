project('dlup', 'cpp',
  version : '0.9.2',
  license : 'Apache-2.0',
  meson_version : '>=1.3',
  default_options : [
    'warning_level=1',
    'cpp_std=c++20',
    'default_library=static',
    # Optimized by default (mirrors the Bazel `--config=release` -O3 -DNDEBUG).
    # Override on the CLI with e.g. `meson setup builddir --buildtype=debug`.
    'buildtype=release',
    'b_ndebug=if-release',
  ])

cpp = meson.get_compiler('cpp')

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
# aifocore (git wrap) exposes the `aifocore` dependency via
# meson.override_dependency in its own meson.build; it provides the ndarray /
# concepts / status / fmt-backed utilities the geometry code relies on.
aifocore_dep = dependency('aifocore')

# Boost.Geometry, header-only, vendored via subprojects/boost.wrap.
boost_dep = dependency('boost')

# libtiff (+ zstd) back the _libtiff_tiff_writer extension. Both have wrap
# fallbacks; with --wrap-mode=forcefallback they build from source.
tiff_dep = dependency('libtiff-4')
zstd_dep = dependency('libzstd')

# `-Wno-unused-parameter` / `-Wno-deprecated-declarations` are GCC/Clang-only;
# drop unsupported flags on MSVC.
dlup_cpp_args = cpp.get_supported_arguments([
  '-Wno-unused-parameter',
  '-Wno-deprecated-declarations',
  # Boost.Geometry pulls in Boost.QVM, whose `x.template member` usages trip the
  # newer Clang diagnostic that became default-error. The Boost code is valid; we
  # only consume the headers, so downgrade it (no-op on compilers without it).
  '-Wno-missing-template-arg-list-after-template-kw',
])

inc = include_directories('include')

# ---------------------------------------------------------------------------
# Geometry library
# ---------------------------------------------------------------------------
geometry_lib = static_library('dlup_geometry',
  [
    'src/geometry/box.cpp',
    'src/geometry/collection.cpp',
    'src/geometry/marching_squares.cpp',
    'src/geometry/polygon.cpp',
    'src/geometry/transforms.cpp',
    'src/geometry/utilities.cpp',
  ],
  include_directories : inc,
  dependencies : [aifocore_dep, boost_dep],
  cpp_args : dlup_cpp_args,
  install : false)

geometry_dep = declare_dependency(
  include_directories : inc,
  link_with : geometry_lib,
  dependencies : [aifocore_dep, boost_dep])

# ---------------------------------------------------------------------------
# Python extensions (nanobind)
# ---------------------------------------------------------------------------
# Built only for wheels (meson-python sets -Dbuild_python=true). nanobind and
# robin-map resolve from subprojects/*.wrap; with --wrap-mode=forcefallback they
# compile from source, so the extensions are self-contained.
if get_option('build_python')
  py = import('python').find_installation(pure : false)
  nanobind_dep = dependency('nanobind')

  py.extension_module('_geometry',
    'src/geometry/geometry_module.cpp',
    include_directories : inc,
    dependencies : [nanobind_dep, geometry_dep],
    cpp_args : dlup_cpp_args,
    install : true,
    subdir : 'dlup')

  py.extension_module('_libtiff_tiff_writer',
    'src/_libtiff_tiff_writer.cpp',
    include_directories : inc,
    dependencies : [nanobind_dep, tiff_dep, zstd_dep],
    cpp_args : dlup_cpp_args + ['-DHAVE_ZSTD'],
    install : true,
    subdir : 'dlup')

  py.extension_module('_background',
    'src/_background.cpp',
    include_directories : inc,
    dependencies : [nanobind_dep],
    cpp_args : dlup_cpp_args,
    install : true,
    subdir : 'dlup')

  # Pure-Python package (all .py/.pyi/.xsd/py.typed under dlup/). The native
  # extensions above install alongside these into the same `dlup` package dir.
  install_subdir('dlup',
    install_dir : py.get_install_dir(),
    exclude_files : ['meson.build'],
    exclude_directories : ['__pycache__'])
endif
