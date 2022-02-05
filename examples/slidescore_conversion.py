import json
import pathlib

import shapely.geometry
from annotations import TIGER_LABEL_CONVERSIONS
from shapely.geometry import mapping
from tqdm import tqdm

from dlup.data._annotations import AnnotationType, SlideAnnotations


class SlideScoreConvertor:
    def __init__(self, asap_filenames, slidescore_mapping):
        self.annotations = [
            (filename, SlideAnnotations.from_asap_xml(filename, label_map=TIGER_LABEL_CONVERSIONS))
            for filename in asap_filenames
        ]
        self._slidescore_mapping = slidescore_mapping
        image_names = [filename.with_suffix("").name for filename in asap_filenames]
        self._image_ids = {image_name: slidescore_mapping[image_name] for image_name in image_names}

    def execute(self, filename, author):
        with open(filename, "w") as f:
            f.write("")
        #     f.write("ImageID\tImage Name\tBy\tQuestion\tAnswer\n")

        for xml_filename, annotation in self.annotations:
            image_name = xml_filename.with_suffix("").name
            image_id = self._slidescore_mapping[image_name]
            for label in annotation.available_labels:
                curr_data = annotation[label].as_list()
                annotation_type = annotation.label_to_type(label)
                answer = []
                for line in self.annotation_iterator(curr_data, annotation_type):
                    answer.append(line)
                with open(filename, "a") as f:
                    slidescore_line = (
                        f"{image_id}\t{image_name}\t{author}\t{label}\t{json.dumps(answer).replace(' ', '')}\n"
                    )
                    f.write(slidescore_line)

    @staticmethod
    def to_slidescore(curr_shape, annotation_type):

        data = mapping(curr_shape)
        if annotation_type == AnnotationType.POLYGON:
            assert len(data["coordinates"]) == 1
            coordinates = data["coordinates"][0]
            output = {"type": "brush", "positivePolygons": [[{"x": int(x), "y": int(y)} for x, y in coordinates]]}

        elif annotation_type == AnnotationType.POINT:
            x = data["coordinates"][0]
            y = data["coordinates"][1]
            output = {"x": int(x), "y": int(y)}

        elif annotation_type == AnnotationType.BOX:
            x0, y0, x1, y1 = curr_shape.bounds
            h = x1 - x0
            w = y1 - y0
            output = {"type": "rect", "corner": {"x": int(x0), "y": int(y0)}, "size": {"x": int(h), "y": int(w)}}

        else:
            raise NotImplementedError

        return output

    def annotation_iterator(self, data, annotation_type):
        for curr_shape in data:
            yield self.to_slidescore(curr_shape, annotation_type)


if __name__ == "__main__":
    with open("/tmp/pycharm_dlup/slidescore_858_mapping.txt", "r") as f:
        slidescore_mapping = {k.split("\t")[1].strip(): k.split("\t")[0] for k in f.readlines()}

    asap_filenames = pathlib.Path(
        "/mnt/archive/data/pathology/TIGER/tiger-training-data/wsirois/wsi-level-annotations/annotations-tissue-cells-xmls/"
    ).glob("*.xml")

    for filename in tqdm(asap_filenames):
        annotations_fn = f"{filename.with_suffix('').name}_annotations.txt"
        SlideScoreConvertor([filename], slidescore_mapping).execute(annotations_fn, "j.teuwen@nki.nl")
    print()
