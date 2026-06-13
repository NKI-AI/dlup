# dlup - Deep Learning Utilities for Pathology

`dlup` provides tools to work with large whole-slide images (WSIs) for
computational pathology: tiled reading across multiple backends, tiling
datasets, geometry/annotation handling, and pyramidal TIFF writing. Performance
critical pieces are implemented in C++20 and exposed to Python via
[nanobind](https://github.com/wjakob/nanobind).

This repository is automatically synced from the AI for Oncology monorepo. It
builds both with [Bazel](https://bazel.build/) (via bzlmod) and with
[Meson](https://mesonbuild.com/) (used to produce the Python wheels).

## Features

- **Multiple slide backends**: FastSlide, OpenSlide, tifffile, DeepZoom and remote/SlideScore
- **Tiling datasets**: grid-based tile extraction with masks, annotations and metadata
- **Geometry**: fast C++ polygons/points/boxes, marching squares, and Shapely interop
- **Annotations**: import/export GeoJSON, HALO, SlideScore and dlup XML
- **Writers**: pyramidal (libtiff-backed) and tifffile TIFF writers
- **Background estimation**: native foreground/background masking

## Installation (Python)

```bash
pip install dlup

# To enable the OpenSlide backend (bundled prebuilt libopenslide):
pip install "dlup[openslide]"
```

```python
import dlup
from dlup import SlideImage

slide = SlideImage.from_file_path("slide.svs")
region = slide.read_region((0, 0), 0, (512, 512))
```

## Building from source

### Bazel

```bash
# Build the C++ libraries and Python package (native extensions)
bazelisk build //:dlup

# Run the Python test suite
bazelisk test //tests/python:tests
```

### Meson (wheels)

```bash
# Build a wheel + sdist for the current interpreter
uv build
```

The Meson build fetches every native dependency through `subprojects/*.wrap`
(aifocore, Boost.Geometry, libtiff, nanobind, ...) and links them statically,
producing self-contained extensions. The pure-Python runtime dependencies
(FastSlide, fim, NumPy, Shapely, ...) are installed by pip from PyPI.

## Development install

For day-to-day development you can work either through the Python (uv/Meson)
toolchain or through Bazel.

### uv (editable, Meson backend)

`dlup` is a compiled (nanobind) package, so the editable install uses
meson-python's rebuild-on-import hook. That hook re-invokes `meson`/`ninja` at
import time, which means it must NOT be installed with build isolation — with
isolation the build tools live in a throwaway environment that is deleted right
after install, and the first `import dlup` fails with
`FileNotFoundError: .../bin/ninja`.

Install the build tools into your environment **first**, then pass
`--no-build-isolation` (with build isolation, the backend is not visible to the
editable build and you get `ModuleNotFoundError: No module named 'mesonpy'`):

```bash
# Install the build backend + tools into the active venv FIRST.
# NB: the PyPI package is `meson-python`; it provides the `mesonpy` module that
# uv's error hint refers to (there is no separate `mesonpy` distribution).
uv pip install meson-python meson ninja nanobind

# Editable install without build isolation (so the backend + rebuild hook's
# `ninja` are found in the venv, not a deleted temp build env).
uv pip install -e . --no-build-isolation
```

With this setup, pure-Python edits are picked up live and C++ changes trigger a
ninja rebuild on the next `import dlup`.

If you do not need an editable install, building and installing a regular wheel
avoids the rebuild hook entirely:

```bash
uv build --wheel && uv pip install --find-links dist --force-reinstall dlup
```

### Bazel

Bazel needs no separate install step — it builds the native extensions and
resolves the Python dependencies hermetically:

```bash
# Build everything (C++ libraries + native extensions + Python package)
bazelisk build //...

# Run the Python test suite
bazelisk test //tests/python:tests
```

## License

Apache 2.0 - see [LICENSE](LICENSE) for details.
