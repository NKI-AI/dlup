#!/usr/bin/env python
# coding=utf-8

"""The setup script."""

import ast
import os
from typing import List

from setuptools import find_packages, setup  # type: ignore

with open("dlup/__init__.py") as f:
    for line in f:
        if line.startswith("__version__"):
            version = ast.parse(line).body[0].value.s  # type: ignore
            break


# Get the long description from the README file
with open("README.md") as f:
    LONG_DESCRIPTION = f.read()

install_requires = [
    "numpy>=1.21",
    "scikit-image>=0.19",
    "tifftools",
    "tifffile>=2022.5.4",
    "pyvips>=2.2.1",
    "tqdm",
    "pillow>=9.2.0",
    "openslide-python",
    "opencv-python>=4.6.0",
]

# This is a hack as PyPi does not want package installed from github, but tox will fail and we are using
# the upstream version of shapely
if os.environ.get("IS_TOX", False):
    install_requires.append("shapely @ git+https://github.com/shapely/shapely.git")


setup(
    author="Jonas Teuwen",
    author_email="j.teuwen@nki.nl",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    entry_points={
        "console_scripts": [
            "dlup=dlup.cli:main",
        ],
    },
    install_requires=install_requires,
    extras_require={
        "dev": [
            "pytest",
            "pytest-mock",
            "sphinx_copybutton",
            "numpydoc",
            "myst_parser",
            "sphinx-book-theme",
            "pylint",
            "pydantic",
            "types-requests",
        ],
    },
    license="Apache Software License 2.0",
    include_package_data=True,
    keywords="dlup",
    name="dlup",
    packages=find_packages(include=["dlup", "dlup.*"]),
    url="https://github.com/NKI-AI/dlup",
    version=version,
    zip_safe=False,
)
