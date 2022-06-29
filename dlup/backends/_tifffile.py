# coding=utf-8
# Copyright (c) dlup contributors
import numpy as np
import PIL.Image
import tifffile

from dlup.backends.common import AbstractSlideBackend, check_mpp


def open_slide(filename):
    return TifffileSlide(filename)


def get_tile(page, coordinates, size):
    # https://gist.github.com/rfezzani/b4b8852c5a48a901c1e94e09feb34743

    """Extract a crop from a TIFF image file directory (IFD).

    Only the tiles englobing the crop area are loaded and not the whole page.
    This is usefull for large Whole slide images that can't fit int RAM.

    Parameters
    ----------
    page : TiffPage
        TIFF image file directory (IFD) from which the crop must be extracted.
    coordinates: (int, int)
        Coordinates of the top left and right corner corner of the desired crop.
    size: (int, int)
        Desired crop height and width.

    Returns
    -------
    out : ndarray of shape (imagedepth, h, w, sampleperpixel)
        Extracted crop.

    """
    i0, j0 = coordinates
    w, h = size

    if not page.is_tiled:
        raise ValueError("Input page must be tiled.")

    im_width = page.imagewidth
    im_height = page.imagelength

    if h < 1 or w < 1:
        raise ValueError("h and w must be strictly positive.")

    if i0 < 0 or j0 < 0 or i0 + h > im_height or j0 + w > im_width:
        raise ValueError("Requested crop area is out of image bounds.")

    tile_width, tile_height = page.tilewidth, page.tilelength
    i1, j1 = i0 + h, j0 + w

    tile_i0, tile_j0 = i0 // tile_height, j0 // tile_width
    tile_i1, tile_j1 = np.ceil([i1 / tile_height, j1 / tile_width]).astype(int)

    tile_per_line = int(np.ceil(im_width / tile_width))

    out = np.empty(
        (page.imagedepth, (tile_i1 - tile_i0) * tile_height, (tile_j1 - tile_j0) * tile_width, page.samplesperpixel),
        dtype=page.dtype,
    )

    fh = page.parent.filehandle

    jpegtables = page.tags.get("JPEGTables", None)
    if jpegtables is not None:
        jpegtables = jpegtables.value

    for i in range(tile_i0, tile_i1):
        for j in range(tile_j0, tile_j1):
            index = int(i * tile_per_line + j)

            offset = page.dataoffsets[index]
            bytecount = page.databytecounts[index]

            fh.seek(offset)
            data = fh.read(bytecount)
            tile, indices, shape = page.decode(data, index, jpegtables=jpegtables)

            im_i = (i - tile_i0) * tile_height
            im_j = (j - tile_j0) * tile_width
            out[:, im_i : im_i + tile_height, im_j : im_j + tile_width, :] = tile

    im_i0 = i0 - tile_i0 * tile_height
    im_j0 = j0 - tile_j0 * tile_width

    return out[:, im_i0 : im_i0 + h, im_j0 : im_j0 + w, :]


class TifffileSlide(AbstractSlideBackend):
    def __init__(self, path: str) -> None:
        super().__init__(path)
        # You can have pyvips figure out the reader
        self._image = tifffile.TiffFile(path)
        self._level_count = len(self._image.pages)

        self.__parse_tifffile()

    def __parse_tifffile(self):
        unit_dict = {1: 1, 2: 254000, 3: 100000, 4: 1000000, 5: 10000000}
        self._downsamples.append(1.0)
        for idx, page in enumerate(self._image.pages):
            self._shapes.append(page.shape[::-1])

            x_res = page.tags["XResolution"].value
            x_res = x_res[0] / x_res[1]
            y_res = page.tags["YResolution"].value
            y_res = y_res[0] / y_res[1]
            unit = int(page.tags["ResolutionUnit"].value)

            mpp_x = unit_dict[unit] / x_res
            mpp_y = unit_dict[unit] / y_res
            check_mpp(mpp_x, mpp_y)
            self._spacings.append(mpp_x)

            if idx >= 1:
                downsample = mpp_x / self._spacings[0]
                self._downsamples.append(downsample)

    @property
    def properties(self):
        """Metadata about the image.
        This is a map: property name -> property value."""
        properties = {}
        for idx, page in enumerate(self._image.pages):
            for tag in page.tags:
                # Not so relevant at this point
                if tag.name in [
                    "TileOffsets",
                    "TileByteCounts",
                    "SMinSampleValue",
                    "JPEGTables",
                    "ReferenceBlackWhite",
                ]:
                    continue

                properties[f"tifffile.level[{idx}].{tag.name}"] = tag.value

        return properties

    @property
    def associated_images(self):
        """Images associated with this whole-slide image.
        This is a map: image name -> PIL.Image."""

        raise NotImplementedError

    def set_cache(self, cache):
        """Use the specified cache to store recently decoded slide tiles.
        cache: an OpenSlideCache object."""
        raise NotImplementedError

    def read_region(self, coordinates, level, size):
        if level > self._level_count - 1:
            raise RuntimeError(f"Level {level} not present.")

        page = self._image.pages[level]
        tile = get_tile(page, coordinates, size)[0]
        bands = tile.shape[-1]

        if bands == 1:
            mode = "L"
            tile = tile[:, :, 0]
        elif bands == 3:
            mode = "RGB"
        elif bands == 4:
            mode = "RGBA"
        else:
            raise RuntimeError(f"Incorrect number of channels.")

        return PIL.Image.fromarray(tile, mode=mode)

    def close(self):
        self._image.close()


if __name__ == "__main__":
    from dlup.backends._pyvips import open_slide as openslide_pyvips

    path = "/mnt/archive/data/pathology/TIGER/wsitils/images/105S.tif"
    tissue_path = "/mnt/archive/data/pathology/TIGER/wsitils/tissue-masks/105S_tissue.tif"

    vslide = openslide_pyvips(path)
    tslide = open_slide(tissue_path)

    location = (0, 0)
    size = (465, 368)
    level = 6
    img = np.asarray(vslide.read_region(location, level, size))
    mask = np.asarray(tslide.read_region(location, level, size))

    from dlup.viz.plotting import plot_2d

    output = np.asarray(plot_2d(img, mask=mask))

    print("hi")
