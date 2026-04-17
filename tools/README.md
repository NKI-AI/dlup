# Building dlup wheels

## Single wheel

```bash
bazelisk build //aifo/dlup:dlup_wheel_cp312
```

Replace `cp312` with `cp310`, `cp311`, or `cp313` as needed. The wheel is written to `bazel-bin/` (the exact path is printed in the build output).

## Full matrix

Build wheels for all platforms and Python versions at once using the helper script (no extra dependencies, runs with any Python 3):

```bash
cd aifo/dlup/tools
python build_wheels.py
```

Or filter by platform/version:

```bash
python build_wheels.py --platform linux_x86_64 --python cp312
```

Wheels are collected in `aifo/dlup/artifacts/wheels/`.

## Usage

```bash
pip install path/to/dlup-0.8.0-cp312-none-manylinux2014_x86_64.whl
```

```python
import dlup
from dlup._geometry import Polygon, Point, GeometryCollection
from dlup._libtiff_tiff_writer import LibtiffTiffWriter
```

Note: `import dlup` requires OpenSlide at runtime. Install via `pip install openslide-bin`.
