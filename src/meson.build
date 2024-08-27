# Include the dependencies module
third_party_dir = include_directories('..')

# Importing necessary variables from the dependencies build file
incdir_pybind11 = get_variable('incdir_pybind11', [])
base_deps = get_variable('base_deps', [])
boost_dep = dependency('boost', modules : ['system', 'serialization'])
opencv_dep = dependency('opencv4')

# Pybind11 modules
libtiff_tiff_writer = py.extension_module('_libtiff_tiff_writer',
    'libtiff_tiff_writer.cpp',
    install : true,
    subdir : 'dlup',  # This is crucial
    include_directories : [third_party_dir, incdir_pybind11],
    cpp_args : cpp_args,
    link_args : link_args,
    dependencies : base_deps)

geometry = py.extension_module('_geometry',
    'geometry.cpp',
    install : true,
    subdir : 'dlup',  # This is crucial
    include_directories : [third_party_dir, incdir_pybind11],
    cpp_args : cpp_args,
    link_args : link_args,
    dependencies : base_deps + [boost_dep, opencv_dep])
