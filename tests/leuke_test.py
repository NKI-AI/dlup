from pathlib import Path

from dlup.annotations import WsiAnnotations

data = WsiAnnotations.from_geojson(Path("/Users/jteuwen/specimen_broken.json"))
