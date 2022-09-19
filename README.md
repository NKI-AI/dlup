# Deep Learning Utilities for Pathology
[![Tox](https://github.com/NKI-AI/dlup/actions/workflows/tox.yml/badge.svg)](https://github.com/NKI-AI/dlup/actions/workflows/tox.yml)
[![mypy](https://github.com/NKI-AI/dlup/actions/workflows/mypy.yml/badge.svg)](https://github.com/NKI-AI/dlup/actions/workflows/mypy.yml)
[![Pylint](https://github.com/NKI-AI/dlup/actions/workflows/pylint.yml/badge.svg)](https://github.com/NKI-AI/dlup/actions/workflows/pylint.yml)
[![Black](https://github.com/NKI-AI/dlup/actions/workflows/black.yml/badge.svg)](https://github.com/NKI-AI/dlup/actions/workflows/black.yml)
[![codecov](https://codecov.io/gh/NKI-AI/dlup/branch/main/graph/badge.svg?token=OIJ7F9G7OO)](https://codecov.io/gh/NKI-AI/dlup)

Dlup offers a set of utilities to ease the process of running Deep Learning algorithms on
Whole Slide Images.

Check the [full documentation](https://docs.aiforoncology.nl/dlup) for more details on how to use dlup.

## Features
- Dataset classes to handle whole-slide images in a tile-by-tile manner
- Annotation classes which can load GeoJSON and ASAP formats and convert these to masks.

### Quickstart
The package can be installed using `pip install dlup`. However, that requires the installation of the beta (upstream)
version of [shapely](https://github.com/shapely/shapely) using `pip install git+https://github.com/shapely/shapely.git`.
