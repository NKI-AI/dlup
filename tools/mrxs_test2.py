# /mnt/sw/spack/opt/spack/linux-ubuntu20.04-zen2/gcc-9.4.0/openslide-aifo-3.4.1-nki-lz5wygwipinkd6utxjppmgkcg7vugze6/lib:

from dlup.data.dataset import TiledROIsSlideImageDataset
from dlup.experimental_backends import ImageBackend
from tqdm import tqdm

# IMAGE_FN = "/Users/jteuwen/images/Extern: T11-94135 a1 HENKI.mrxs"


mpp = 8.0
tile_size = 256
tile_overlap = 0
dataset = TiledROIsSlideImageDataset.from_standard_tiling(
    IMAGE_FN, mpp, (tile_size, tile_size), (tile_overlap, tile_overlap), backend=ImageBackend.OPENSLIDE
)

print(dataset.slide_image.size)

for region in tqdm(dataset):
    curr_region = region
