FROM python:3.13-slim AS dlup

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# System toolchain + C/C++ libs
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libboost-all-dev \
    libopenslide-dev \
    libtiff-dev \
    libzstd-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY . .

# Build & install dlup from source into this image system
RUN uv pip install --system ".[remote-backends]"

# Default command: drop into a shell
CMD ["bash"]