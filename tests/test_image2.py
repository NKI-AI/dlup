from dlup import SlideImage
from dlup import BoundaryMode

if __name__ == "__main__":
    slide_image = SlideImage.from_file_path("/mnt/archive/data/pathology/TIGER/wsitils/images/104S.tif")

    region_view = slide_image.get_scaled_view(0.3)
    region_view.boundary_mode = BoundaryMode.zero

    coordinates = (-256, -256)
    region_size = (512, 512)

    region = region_view.read_region(coordinates, region_size)

    pass
