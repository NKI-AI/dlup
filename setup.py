#!/usr/bin/env python
"""The setup script."""

import ast

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
    "tifftools",
    "tifffile>=2024.5.22",
    "pyvips>=2.2.3",
    "tqdm",
    "pillow>=10.3.0",
    "openslide-python>=1.3.1",
    "opencv-python>=4.9.0.80",
    "shapely>=2.0.4",
    "packaging>=24.0",
]

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
    install_requires=install_requires,
    extras_require={
        "dev": [
            "pytest",
            "mypy>=1.10.0",
            "pytest-mock",
            "sphinx_copybutton",
            "numpydoc",
            "myst_parser",
            "sphinx-book-theme",
            "pylint",
            "pydantic>=2.1.1",
            "types-requests",
            "types-Pillow",
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
    zip_safe=False,
)
