# Multi-stage build for dlup.
# Produces two useful targets:
#   builder  -- has build tools, used as base for compiling downstream packages
#   runtime  -- slim image with only runtime shared libraries + installed packages
#
# Build:
#   docker build --target builder -t slideforge-dlup:builder .
#   docker build -t slideforge-dlup:latest .

FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Build toolchain for dlup C++ extensions (geometry backend uses boost + pybind11)
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libboost-all-dev \
    libtiff-dev \
    libzstd-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Use openslide-bin (PyPI) instead of libopenslide-dev (apt) because the
# openslide-bin provides pre-built OpenSlide 4.x binaries bundled inside the Python package,
# so no system-level .so install or source build is needed. Debian/Ubuntu packages ship OpenSlide 3.4.1.
RUN uv pip install --system openslide-bin ".[remote-backends]"

# Collect boost runtime .so files for the slim stage
RUN mkdir -p /runtime-libs && \
    cp -a /usr/lib/*-linux-gnu/libboost_*.so* /runtime-libs/

# -------------------------------------------------------------------
FROM python:3.13-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libtiff6 \
    libzstd1 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /runtime-libs/ /usr/lib/
RUN ldconfig

COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin/ /usr/local/bin/

CMD ["bash"]
