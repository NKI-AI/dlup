project('dlup', 'cpp', 'cython',
 version : '0.7.0',
 default_options : ['buildtype=release', 'warning_level=3', 'cpp_std=c++20'])

ninja = find_program('ninja', required : true)

# Import Python module

py_mod = import('python')
if host_machine.system() == 'darwin'
    py = py_mod.find_installation(pure: false)
else
    py = py_mod.find_installation()
endif
py_dep = py.dependency()

# Base compiler and linker arguments
cpp_args = ['-O3', '-march=native', '-ffast-math', '-funroll-loops', '-flto', '-pipe', '-fomit-frame-pointer']
link_args = ['-flto']

# Unity build option
unity_build = true


subdir('third_party')
subdir('src')
subdir('dlup')

# Include subdirectories for building
install_subdir('dlup', install_dir : py.get_install_dir())
