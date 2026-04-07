# Multi-stage build for dlup base images.
# Produces two targets:
#   builder  -- build toolchain for compiling dlup C++ extensions
#   runtime  -- runtime shared libraries only (no Python packages)
#
# Python packages are installed into isolated venvs by downstream images.
#
# Build:
#   docker build --target builder -t dlup:builder .
#   docker build -t dlup:latest .

FROM python:3.13.3-slim AS builder

# Build toolchain for dlup C++ extensions (geometry backend uses boost + pybind11).
# Boost is header-only at runtime — only needed here for compilation.
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libboost-all-dev \
    libtiff-dev \
    libzstd-dev \
 && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------------------------
FROM python:3.13.3-slim AS runtime

# Runtime shared libraries required by dlup C extensions (libtiff for
# _libtiff_tiff_writer, libgomp for OpenMP in numpy/numcodecs).
# Boost is not needed at runtime — dlup uses it header-only via pybind11.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libtiff6 \
    libzstd1 \
 && rm -rf /var/lib/apt/lists/*

CMD ["bash"]
