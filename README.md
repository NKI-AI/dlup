# Deep Learning Utilities for Pathology

DLUP offers a set of of utilities to ease the process of running Deep Learning algorithms on
Whole Slide Images.

## Requirements

DLUP supports python3.8+ and depends on the `openslide` open source library.

**Note:** Not all slides can be read using the OpenSlide version available in the official repository.
You could install the version from our GitHub https://github.com/NKI-AI/openslide.git.
Alternatively, you can also run DLUP using the `Dockerfile` provided, which includes this version of OpenSlide.


## Installation

To install DLUP from the latest main version, simply run:

``` sh
pip install git+https://github.com/NKI-AI/DLUP
```

## Quickstart

### Tiling

With dlup it's extremely easy to tile gigapixel WSIs. Given a set of images found in an `<input folder>` it's possible to tile
the image by 1024x1024 tiles with no overlap and with target mpp 0.45 by simply running:

``` sh
dlup wsi tiling --no-log --mpp=0.45 --tile-size=1024 --tile-overlap=0\
                --num-workers=4 <input folder> <output folder>
```
