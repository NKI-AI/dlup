from dlup.experimental_annotations import WsiAnnotations

if __name__ == "__main__":
    fn = "/mnt/archive/data/pathology/TIGER/wsirois/wsi-level-annotations/annotations-tissue-cells-xmls/100B.xml"
    annotations = WsiAnnotations.from_asap_xml(fn, scaling=5)
    _annotations = WsiAnnotations.from_asap_xml(fn, scaling=1)
    geojson = annotations.as_geojson()

    annotations2 = WsiAnnotations.from_geojson(["/home/j.teuwen/roi.json"], scaling=2)
    geojson = annotations2.as_geojson()
    print()
