#!/usr/bin/env python
# coding=utf-8

"""The setup script."""

import ast
from typing import List

from setuptools import find_packages, setup

with open("dlup/__init__.py") as f:
    for line in f:
        if line.startswith("__version__"):
            version = ast.parse(line).body[0].value.s  # type: ignore
            break


# Get the long description from the README file
with open("README.md") as f:
    LONG_DESCRIPTION = f.read()


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
    install_requires=[
        "matplotlib",
        "joblib",
        "numpy",
        "openslide-python",
        "requests",
        "scikit-image",
        "tifftools",
        "torch",
        "torchvision",
        "tqdm",
        "pillow",
        "staintools",
    ],
    extras_require={
        "dev": ["pytest", "sphinx_copybutton", "numpydoc", "myst_parser", "sphinx-book-theme", "pylint"],
    },
    license="Apache Software License 2.0",
    include_package_data=True,
    keywords="dlup",
    name="dlup",
    packages=find_packages(include=["dlup", "dlup.*"]),
    url="https://github.com/NKI-AI/dlup",
    version="0.1",
    zip_safe=False,
)
