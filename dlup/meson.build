# Include the NumPy headers
incdir_numpy = get_variable('incdir_numpy', [])

# Cython extension module
background = py.extension_module('_background',
    '_background.pyx',
    include_directories : [incdir_numpy],
    install : true,
    subdir : 'dlup',
    link_args : link_args,
    c_args : cpp_args + ['-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION'],
    dependencies : [py_dep])  # Add Python dependency
