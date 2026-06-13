project('boost', 'cpp', version : '1.86.0', license : 'BSL-1.0')

# Header-only view of Boost. dlup only uses Boost.Geometry (and its header-only
# transitive deps), all of which live under `boost/` in the release tarball, so
# exposing the unpacked root as an include directory is sufficient. No Boost
# libraries are compiled.
boost_dep = declare_dependency(
  include_directories : include_directories('.'),
)

meson.override_dependency('boost', boost_dep)
