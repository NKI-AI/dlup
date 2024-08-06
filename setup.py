#!/usr/bin/env python
"""The setup script."""

import ast
from typing import Any, List

from skbuild import setup
from Cython.Build import cythonize
from setuptools import Extension

with open("dlup/__init__.py") as f:
    for line in f:
        if line.startswith("__version__"):
            version = ast.parse(line).body[0].value.s
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

extension = Extension(
    name="dlup._background",
    sources=["dlup/_background.pyx"],
    include_dirs=[numpy.get_include()],
    extra_compile_args=["-O3", "-march=native", "-ffast-math"],
    extra_link_args=["-O3"],
)


def get_ext_modules() -> List[Extension]:
    return cythonize([extension], compiler_directives={"language_level": "3"})


setup(
    name="dlup",
    version=version,
    author="Jonas Teuwen",
    author_email="j.teuwen@nki.nl",
    description="A package for digital pathology image analysis",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/NKI-AI/dlup",
    packages=["dlup"],
    cmake_install_dir="dlup",
    cmake_with_sdist=True,
    cmake_languages=("C", "CXX"),
    python_requires=">=3.10",
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
    ext_modules=get_ext_modules(),
    include_package_data=True,
    license="Apache Software License 2.0",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="dlup",
    entry_points={
        "console_scripts": [
            "dlup=dlup.cli:main",
        ],
    },
)
