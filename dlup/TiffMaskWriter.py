import pyvips
from pathlib import Path
from dlup import SlideImage
import numpy as np
from PIL import Image, ImageDraw

class TiffMaskWriter(object):
    def __init__(self, anns: dict, mpp: float):
        """Class initialization.
        Input:
            1. anns: A dictionary containing shapely objects
            2. mpp: The mpp value desired for reading WSI."""
        self.anns = anns
        self.mpp =mpp

    @staticmethod
    def _get_ann_coords(anns: dict, ann_attr: dict) -> list:
        """Get the coorinates for the points in the annotation files. Filter for slide name, classname.

        Input:
            1. anns: Dictionary object containing preprocessed annotations from any slidescore study.
            2. ann_attr: Dictionary object containing the attributes of the annotations (slidename, classname).

        Output:
            1. mycoordslist: A list containing all the annotation points for a particular class label by a particular author.
            """
        mycoordslist = []
        if anns is not None:
            slidename = ann_attr["slidename"]
            label = ann_attr["classname"]
            blob = []
            for i in range(len(anns[slidename][label])):
                blob.append([list(x.exterior.coords) for x in anns[slidename][label][i].geoms])
            if blob not in mycoordslist:
                mycoordslist.append(blob)
        return mycoordslist

    @staticmethod
    def _numpy2vips(a: np.ndarray) -> pyvips.Image:
        """Return a pyvips.Image object given a numpy image.
        Input:
            1. a: a numpy array
        output:
            1. vi: pyvips.Image object"""
        height, width = a.shape
        flat = a.reshape(width * height)
        vi = pyvips.Image.new_from_memory(flat.data, width, height, 1, 'uchar')
        return vi

    def _gen_pyvips_mask(self, polygons:list, scaled_dims:tuple, real_dims:tuple) -> pyvips.Image:
        """Generate the annotation masks from the annotation points marked for a WSI.

        Input:
            1. polygons: A list of annotation polygons.
            2. scaled_dims: The dimensions of the WSI after scaling to required mpp.
            3. real_dims: The dimensions of the WSI at full resolution.

        Output:
            1. mask: A pyvips.Image object """
        mask = Image.new("L", (scaled_dims[0], scaled_dims[1]))
        mask_draw = ImageDraw.Draw(mask)
        if len(polygons) > 0:
            for polygon in polygons:
                for element in polygon:
                    xy = []
                    for coord in element[0]:
                        coord = list(coord)
                        coord[0] = coord[0] * scaled_dims[0] / real_dims[0]
                        coord[1] = coord[1] * scaled_dims[1] / real_dims[1]
                        coord = tuple(coord)
                        xy.append(coord)
                    mask_draw.polygon(xy, fill=255)
        mask = self._numpy2vips(np.asarray(mask))
        return mask

    def _get_wsi_dims(self, path_to_slide: str) -> [list, list]:
        """Get the dimensional properties of a whole slide image.
        Input:
            1. path_to_slide: A string containing the path to a WSI.
        Output:
            1. real_dims: A two-tuple containing the original dimensions of the WSI.
            2. scaled_dims: A two-tuple containing the dimensions of the WSI scaled at some mpp."""
        slide_image = SlideImage.from_file_path(Path(path_to_slide))
        real_dims = slide_image.size
        scaled_slide_image = slide_image.get_scaled_view(slide_image.get_scaling(self.mpp))
        scaled_dims = scaled_slide_image.size
        return real_dims, scaled_dims

    def get_wsi_props(self, path_to_slide: str) -> [tuple, tuple]:
        real_dims, scaled_dims = self._get_wsi_dims(path_to_slide)
        return real_dims, scaled_dims

    def get_ann_props(self, ann_attr: dict) -> list:
        return self._get_ann_coords(self.anns, ann_attr)

    def get_tiff_mask(self, coords: list, scaled_dims: tuple, real_dims: tuple) -> pyvips.Image:
        return self._gen_pyvips_mask(coords, scaled_dims=scaled_dims, real_dims=real_dims)

    def save_tiff_mask(self, path_to_save: str, mask: pyvips.Image) -> None:
        mask.tiffsave(path_to_save,
            compression='jpeg',
            tile=True,
            tile_width=256,
            tile_height=256,
            pyramid=True,
            squash=True,
            bitdepth=1,
            bigtiff=True,
            depth='onetile',
            background=[0]
        )

    def write_tiff_mask(self, path_to_slide: str, ann_attr: dict, path_to_save: str) -> None:
        real_dims, scaled_dims = self.get_wsi_props(path_to_slide)
        annotation_polygons = self.get_ann_props(ann_attr)
        tiff_mask = self.get_tiff_mask(annotation_polygons, scaled_dims=scaled_dims, real_dims=real_dims)
        self.save_tiff_mask(path_to_save=path_to_save, mask=tiff_mask)