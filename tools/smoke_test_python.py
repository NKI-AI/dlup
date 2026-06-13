#!/usr/bin/env python3
"""On-target smoke test for an installed dlup Python wheel.

Run this with the interpreter of a venv that has the wheel installed (NOT from
the repo source tree), e.g. in CI after ``uv pip install --find-links dist
dlup``. It imports ``dlup`` (which loads the native extensions ``dlup._geometry``,
``dlup._background``, ``dlup._libtiff_tiff_writer``), checks the version, and
exercises the geometry extension end to end -- proving the wheel's native
libraries load and execute on the real OS/arch/Python of the runner.
"""

from __future__ import annotations

import sys


def main() -> int:
    # Imported from the installed wheel in the CI venv, not the repo source tree.
    import dlup  # pylint: disable=import-error,import-outside-toplevel

    version = getattr(dlup, "__version__", "<unknown>")
    print(f"\u25b6\ufe0e Imported dlup {version}")
    print(f"  interpreter: {sys.executable}")
    print(f"  python:      {sys.version.split()[0]} ({sys.implementation.name})")

    # Native extensions must import.
    import dlup._background  # noqa: F401  # pylint: disable=import-error,import-outside-toplevel,unused-import,no-name-in-module
    import dlup._geometry  # noqa: F401  # pylint: disable=import-error,import-outside-toplevel,unused-import,no-name-in-module
    import dlup._libtiff_tiff_writer  # noqa: F401  # pylint: disable=import-error,import-outside-toplevel,unused-import,no-name-in-module

    print("\u2714 Imported native extensions (_geometry, _background, _libtiff_tiff_writer)")

    # Exercise the geometry extension: a 3x3 square has area 9.
    from dlup.geometry import Polygon  # pylint: disable=import-error,import-outside-toplevel,no-name-in-module

    polygon = Polygon([(0, 0), (0, 3), (3, 3), (3, 0)])
    area = polygon.area
    if abs(area - 9.0) > 1e-6:
        print(f"\u2717 Unexpected polygon area: {area} (expected 9.0)")
        return 1
    print(f"\u2714 Built a Polygon and computed area = {area}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
