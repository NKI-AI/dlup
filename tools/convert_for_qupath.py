import pathlib
from dlup.annotations import WsiAnnotations
import json
from tqdm import tqdm
PATH_TO_ANNS = pathlib.Path("/Users/jteuwen/v20221018")

for dir in tqdm(PATH_TO_ANNS.glob("*")):
    for json_fn in dir.glob("*.json"):
        annotation = WsiAnnotations.from_geojson(json_fn)
        json_dict = annotation.as_geojson()
        with open(json_fn, "w") as f:
            json.dump(json_dict, f, indent=2)
