cmake_minimum_required(VERSION 3.12)
project(dlup)

set(CMAKE_CXX_STANDARD 17)

find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)
find_package(pybind11 REQUIRED)

pybind11_add_module(_libtiff_tiff_writer src/libtiff_tiff_writer.cpp)
install(TARGETS _libtiff_tiff_writer DESTINATION .)
