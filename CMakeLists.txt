cmake_minimum_required(VERSION 3.12)
cmake_policy(SET CMP0148 NEW)
project(dlup)

set(CMAKE_CXX_STANDARD 17)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(Cython REQUIRED)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_libtiff_tiff_writer src/libtiff_tiff_writer.cpp)
install(TARGETS _libtiff_tiff_writer DESTINATION .)
