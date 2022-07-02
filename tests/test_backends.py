import numpy as np

from dlup.experimental_backends import autodetect_backend
from dlup.experimental_backends._pyvips import open_slide as openslide_pyvips
from dlup.experimental_backends._tifffile import open_slide as openslide_tifffile

if __name__ == "__main__":

    path = "/mnt/archive/data/pathology/TIGER/wsitils/images/105S.tif"
    tissue_path = "/mnt/archive/data/pathology/TIGER/wsitils/tissue-masks/105S_tissue.tif"

    vslide = openslide_pyvips(tissue_path)
    tslide = openslide_tifffile(tissue_path)

    location = (0, 0)
    size = (465, 368)
    level = 6
    img = np.asarray(vslide.read_region(location, level, size))
    mask = np.asarray(tslide.read_region(location, level, size))

    from dlup.viz.plotting import plot_2d

    output = np.asarray(plot_2d(img, mask=mask))

    backend = autodetect_backend(tissue_path)

    thumbnail = np.asarray(vslide.get_thumbnail((256, 256)))
    thumbnail2 = np.asarray(tslide.get_thumbnail((256, 256)))

    print("hi")

    path = "/mnt/archive/data/pathology/TCGA/images/gdc_manifest.2021-11-01_tissue_uterus_nos.txt/076fde5e-e081-4316-8279-3de6108dd021/TCGA-N6-A4VD-01A-01-TSA.E009A1D0-3BB5-45A4-BC4A-9C505F71AA22.svs"
    # slide = OpenSlideSlide(path)

    from dlup import SlideImage
    from dlup.experimental_backends import ImageBackends

    # Try to autodetect the backend
    slide_autodetect = SlideImage.from_file_path(path, backend=ImageBackends.AUTODETECT)

    print("hi")
