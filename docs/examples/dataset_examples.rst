Working with the SlideImage and creating a TiledROIsSlideImageDataset
---------------------------------------------------------------------

Below you can find code snippets that show how to work with the DLUP SlideImage class, and create
a tiled dataset on the fly without any preprocessing. The visualizations show the result
of the automatic masking and thresholding.

.. code-block:: python

    from dlup.data.dataset import TiledROIsSlideImageDataset
    from dlup import SlideImage
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw
    import numpy as np
    from dlup.background import get_mask

    INPUT_FILE_PATH = "TCGA-B6-A0IG-01Z-00-DX1.4238CB6E-0561-49FD-9C49-9B8AEAFC4618.svs"
    slide_image = SlideImage.from_file_path(INPUT_FILE_PATH)

    TILE_SIZE = (10, 10)
    TARGET_MPP = 100

    # Generate the mask
    mask = get_mask(slide_image)
    scaled_region_view = slide_image.get_scaled_view(slide_image.get_scaling(TARGET_MPP))

    dataset = TiledROIsSlideImageDataset.from_standard_tiling(INPUT_FILE_PATH, TARGET_MPP, TILE_SIZE, (0, 0), mask=mask)
    output = Image.new("RGBA", tuple(scaled_region_view.size), (255, 255, 255, 255))

    for d in dataset:
        tile = d["image"]
        coords = np.array(d["coordinates"])
        box = tuple(np.array((*coords, *(coords + TILE_SIZE))).astype(int))
        output.paste(tile, box)
        draw = ImageDraw.Draw(output)
        draw.rectangle(box, outline="red")

    output.save("dataset_example.png")

.. figure:: img/dataset_example.png

.. code-block:: python

    grid1 = Grid.from_tiling(
        (100, 120),
        size=(100, 100),
        tile_size=TILE_SIZE,
        tile_overlap=(0, 0)
    )

    grid2 = Grid.from_tiling(
        (65, 62, 0),
        size=(100, 100),
        tile_size=TILE_SIZE,
        tile_overlap=(0, 0)
    )

    dataset = TiledROIsSlideImageDataset(INPUT_FILE_PATH, [(grid1, TILE_SIZE, TARGET_MPP), (grid2, TILE_SIZE, TARGET_MPP)], mask=mask)


    output = Image.new("RGBA", tuple(scaled_region_view.size), (255, 255, 255, 255))

    for i, d in enumerate(dataset):
        tile = d["image"]
        coords = np.array(d["coordinates"])
        print(coords, d["grid_local_coordinates"], d["grid_index"])
        box = tuple(np.array((*coords, *(coords + TILE_SIZE))).astype(int))
        output.paste(tile, box)
        draw = ImageDraw.Draw(output)
        draw.rectangle(box, outline="red")

    output.save("dataset_example2.png")

.. figure:: img/dataset_example2.png
