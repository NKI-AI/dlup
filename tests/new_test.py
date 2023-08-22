# import logging
# from pathlib import Path
# from typing import Generator, Optional
#
# from darwin.client import Client
# from darwin.dataset import RemoteDataset
# from darwin.dataset.release import Release
# from darwin.exceptions import NotFound
# from darwin.utils import parse_darwin_json
#
# from dlup.annotations import AnnotationClass, AnnotationType, WsiAnnotations
#
#
# def _has_release(dataset: RemoteDataset, release_name: str) -> bool:
#     return release_name in [_.name for _ in dataset.get_releases()]
#
#
# def _get_release(dataset: RemoteDataset, release_name: str) -> Release:
#     if not _has_release(dataset, release_name):
#         raise ValueError(
#             f"Release {release_name} does not exist in {dataset.team}/{dataset.slug}."
#             "Create one in the dataset overview on Darwin V7"
#         )
#     return dataset.get_release(release_name)
#
#
# class DarwinV7ReleaseWrapper:
#     def __init__(self, dataset_slug: str, release_name: str, api_key: Optional[str] = None):
#         self._logger = logging.getLogger(type(self).__name__)
#         if api_key is not None:
#             client = Client.from_api_key(api_key)
#         else:  # This requires that you have run `darwin authenticate`
#             client = Client.local()
#
#         # TODO: Caching might help
#         try:
#             self._dataset: RemoteDataset = client.get_remote_dataset(dataset_slug)
#         except NotFound:
#             raise ValueError(f"Dataset {dataset_slug} not found")
#
#         self._release_name: str = release_name
#         self._release: Release = _get_release(self._dataset, release_name)
#
#         self._annotated_files = []
#         self._release_path: Optional[Path] = None
#         # TODO: Maybe only call this on demand
#         self._pull_from_remote()
#
#     def _pull_from_remote(self):
#         self._release_path = self._dataset.local_releases_path / self._release.name / "annotations"
#         if not (self._release_path / "completed").is_file():
#             self._dataset.pull(release=self._release, only_annotations=True)
#             with open(self._release_path / "completed", "w") as f:
#                 f.write("")
#
#     def annotated_files(self) -> Generator:
#         for filename in self._release_path.glob("*.json"):
#             parsed = parse_darwin_json(filename, 0)
#             yield filename, parsed.filename
#
#     def _get_annotation_json(self, filename: str):
#         _path = self._release_path / filename
#         if not (_path.with_suffix(".json")).is_file():
#             raise ValueError(f"Filename {filename} is not available in release {self._release_name}")
#
#         return _path.with_suffix(".json")
#
#     def build_dlup_wsi_annotations(self):
#         output = {}
#         for json_path, filename in self.annotated_files():
#             output[filename] = WsiAnnotations.from_darwin_json(json_path)
#         return output
#
#
# if __name__ == "__main__":
#     # z = WsiAnnotations.from_darwin_json(
#     #     "/Users/jteuwen/Downloads/test_complex_polygons.json"
#     # )
#     #
#     # z.filter(["lymphocyte (cell)", "ROI (detection)"])
#
#     # a = AnnotationClass("lymphocyte (cell)", AnnotationType.POINT)
#     # b = AnnotationClass("lymphocyte cell", AnnotationType.POINT)
#     #
#     # z.relabel(((a, b),))
#     #
#     # y = z.bounding_box
#     #
#     # b = z[b]
#     #
#     # x = z.as_geojson()
#     #
#     # w = z.read_region(y[0], 1, y[1])
#
#     release_name = "test"
#     dataset_slug = "tcga-lung"
#     wrapper = DarwinV7ReleaseWrapper(dataset_slug=dataset_slug, release_name=release_name)
#
#     z = wrapper.build_dlup_wsi_annotations()
#
#     pass
