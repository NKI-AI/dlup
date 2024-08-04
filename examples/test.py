import openslide

slide = openslide.open_slide("output_normal.tiff")
for idx, dim in enumerate(slide.level_dimensions):
    region = slide.read_region((0, 0), idx, dim)
    region.save("output_normal_level_{}.png".format(idx))

slide = openslide.open_slide("output_c.tiff")
for idx, dim in enumerate(slide.level_dimensions):
    region = slide.read_region((0, 0), idx, dim)
    region.save("output_c_level_{}.png".format(idx))
