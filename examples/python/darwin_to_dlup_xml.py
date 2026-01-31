# Copyright 2024 AI for Oncology Research Group. All Rights Reserved.
# Copyright 2024 Jonas Teuwen. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This script converts Darwin annotations to DLUP XML format.
In addition it generates an index map based on the order of incoming polygons.
"""

import argparse

from dlup import SlideAnnotations
from pathlib import Path
from tqdm import tqdm


def convert_darwin_to_dlup(darwin_annotations: Path, output_xmls: Path) -> None:
    index_map = {}
    current_max_index = 1
    output_xmls.mkdir(parents=True, exist_ok=True)

    for json_fn in tqdm(darwin_annotations.glob("*.json")):
        output_fn = output_xmls / (json_fn.stem + ".xml")
        tqdm.write(f"Processing {json_fn} to {output_fn}")
        annotations = SlideAnnotations.from_file_path(json_fn, reader="darwin_json")

        if not annotations.available_classes:
            tqdm.write(f"Skipping {json_fn} as it has no annotations")
            continue

        for polygon_name in annotations.available_polygons:
            if polygon_name not in index_map:
                index_map[polygon_name] = current_max_index
                current_max_index += 1

        annotations.reindex_polygons(index_map)
        with open(output_fn, "w", encoding="utf-8") as f:
            f.write(annotations.as_dlup_xml())


def main():
    parser = argparse.ArgumentParser(description="Convert Darwin annotations to DLUP XML format")
    parser.add_argument("darwin_annotations", type=Path, help="Path to Darwin annotation folder")
    parser.add_argument("output_xmls", type=Path, help="Path to output dlup xml files")
    args = parser.parse_args()

    convert_darwin_to_dlup(args.darwin_annotations, args.output_xmls)


if __name__ == "__main__":
    main()
