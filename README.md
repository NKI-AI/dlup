# Deep Learning Utilities for Pathology
[![pypi](https://img.shields.io/pypi/v/dlup.svg)](https://pypi.python.org/pypi/dlup)
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
- Annotation classes which can load GeoJSON, [V7 Darwin](https://www.v7labs.com/), [HALO](https://indicalab.com/halo/) and [ASAP](https://computationalpathologygroup.github.io/ASAP/) formats and convert these to masks.
- Command-line utilities to report on the metadata of WSIs, tile WSIs, and convert masks to polygons.

## Quickstart
The package can be installed using `python -m pip install dlup`.

## Used by
- [ahcore](https://github.com/NKI-AI/ahcore.git): a pytorch lightning based-library for computational pathology
