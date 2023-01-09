import pyvips
import numpy as np
from ahcore.utils.pyvips_utils import numpy_to_vips
from tqdm import tqdm
from PIL import Image
from PIL import ImageDraw

def tiff_from_dataset(dataset, save_path):
    vips_image = None
    for tile_index, sample in tqdm(
        enumerate(dataset), disable=False, unit="tiles", total=len(dataset)
    ):

        TILE_SIZE = (10, 10)
        TARGET_MPP = 100

        scaled_region_view = dataset.slide_image.get_scaled_view(dataset.slide_image.get_scaling(TARGET_MPP))
        output_tile = Image.new("RGBA", TILE_SIZE, (255, 255, 255, 255))

        target = sample["target"]
        tile = sample["image"]
        # TILE has to be something else (using the target)

        coords = np.array(sample["coordinates"])
        box = tuple(np.array((*(0, 0), *(coords + TILE_SIZE))).astype(int))
        output_tile.paste(tile, box)
        draw = ImageDraw.Draw(output_tile)
        draw.rectangle(box, outline="red")

        _tile = np.asarray(output_tile)
        if vips_image is None:
            # Assumes last axis is the channel!
            vips_image = pyvips.Image.black(*scaled_region_view.size, bands=_tile.shape[-1])

        vips_tile = numpy_to_vips(_tile)
        vips_image = vips_image.insert(vips_tile, *coords)

