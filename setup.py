#!/usr/bin/env python
"""The setup script."""

import ast
from typing import Any

from Cython.Build import cythonize  # type: ignore
from setuptools import Extension, find_packages, setup  # type: ignore
import pybind11
import subprocess
import os


with open("dlup/__init__.py") as f:
    for line in f:
        if line.startswith("__version__"):
            version = ast.parse(line).body[0].value.s  # type: ignore
            break


# Get the long description from the README file
with open("README.md") as f:
    LONG_DESCRIPTION = f.read()

install_requires = [
    "numpy==1.26.4",
    "tifftools>=1.5.2",
    "tifffile>=2024.7.2",
    "pyvips>=2.2.3",
    "tqdm>=2.66.4",
    "pillow>=10.3.0",
    "openslide-python>=1.3.1",
    "opencv-python>=4.9.0.80",
    "shapely>=2.0.4",
    "packaging>=24.0",
    "pybind11>=2.8.0",
]


class NumpyImportDefer:
    def __getattr__(self, attr: Any) -> Any:
        import numpy

        return getattr(numpy, attr)


numpy = NumpyImportDefer()


def get_brew_path(package):
    return subprocess.check_output(["brew", "--prefix", package]).decode("utf-8").strip()


vips_path = get_brew_path("vips")
glib_path = get_brew_path("glib")

print(f"Using vips from {vips_path}")
print(f"Using glib from {glib_path}")

os.environ["VIPS_INCLUDE_PATH"] = f"{vips_path}/include"
os.environ["VIPS_LIB_PATH"] = f"{vips_path}/lib"

# Add these environment variables
os.environ["CPPFLAGS"] = f"-I{glib_path}/include/glib-2.0 -I{glib_path}/lib/glib-2.0/include"
os.environ["LDFLAGS"] = f"-L{glib_path}/lib"


extension = Extension(
    name="dlup._background",
    sources=["dlup/_background.pyx"],
    include_dirs=[numpy.get_include()],
    extra_compile_args=["-O3", "-march=native", "-ffast-math"],
    extra_link_args=["-O3"],
)

fast_tiff_writer_extension = Extension(
    name="dlup.fast_tiff_writer",
    sources=["src/fast_tiff_writer.cpp"],
    include_dirs=[
        pybind11.get_include(),
        f"{vips_path}/include",
        f"{glib_path}/include/glib-2.0",
        f"{glib_path}/lib/glib-2.0/include",
    ],
    library_dirs=[f"{vips_path}/lib", f"{glib_path}/lib"],
    libraries=["vips", "glib-2.0"],
    extra_compile_args=["-std=c++17", "-O3", "-march=native", "-ffast-math"],
    extra_link_args=["-lvips", "-lglib-2.0"],
)


setup(
    author="Jonas Teuwen",
    author_email="j.teuwen@nki.nl",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    entry_points={
        "console_scripts": [
            "dlup=dlup.cli:main",
        ],
    },
    setup_requires=["Cython>=0.29", "pybind11>=2.8.0"],
    install_requires=install_requires,
    extras_require={
        "dev": [
            "psutil",
            "pytest>=8.2.1",
            "mypy>=1.10.0",
            "pytest-mock>=3.14.0",
            "sphinx_copybutton>=0.5.2",
            "numpydoc>=1.7.0",
            "myst_parser>=3.0.1",
            "sphinx-book-theme>=1.1.2",
            "pylint>=3.2.2",
            "pydantic>=2.7.2",
            "types-Pillow>=10.2.0",
            "darwin-py>=0.8.62",
        ],
        "darwin": ["darwin-py>=0.8.59"],
    },
    license="Apache Software License 2.0",
    include_package_data=True,
    keywords="dlup",
    name="dlup",
    packages=find_packages(include=["dlup", "dlup.*"]),
    url="https://github.com/NKI-AI/dlup",
    version=version,
    ext_modules=cythonize([extension], compiler_directives={"language_level": "3"}) + [fast_tiff_writer_extension],
    include_dirs=[numpy.get_include(), pybind11.get_include()],
    zip_safe=False,
)
