# coding=utf-8
# Copyright (c) dlup contributors
import pathlib

from dlup.preprocessors.slide_manager import BaseSlideManager


class SlideManager(BaseSlideManager):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )

    def extract_unique_file_id(self, file: pathlib.Path):
        return file.stem.replace(" ", "_")
