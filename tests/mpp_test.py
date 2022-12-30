from dlup import SlideImage
from dlup.data.dataset import TiledROIsSlideImageDataset
from dlup.experimental_backends import ImageBackend

image = SlideImage.from_file_path(
    "/Users/jteuwen/TCGA-OL-A5RY-01Z-00-DX1.AE4E9D74-FC1C-4C1E-AE6D-5DF38899BBA6.svs",
    overwrite_mpp=(0.25, 0.25),
    backend=ImageBackend.OPENSLIDE,
)

fn = "/Users/jteuwen/TCGA-OL-A5RY-01Z-00-DX1.AE4E9D74-FC1C-4C1E-AE6D-5DF38899BBA6.svs"
ds = TiledROIsSlideImageDataset.from_standard_tiling(
    fn, tile_size=(1024, 1024), tile_overlap=(0, 0), mpp=1, overwrite_mpp=(0.25, 0.25)
)


print(ds)
