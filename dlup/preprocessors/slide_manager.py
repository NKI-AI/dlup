# coding=utf-8
# Copyright (c) DLUP Contributors
import abc
import logging
import pathlib
from typing import Any, Dict, List, Union

from dlup import SlideImage
from dlup.utils.types import PathLike


class BaseSlideManager(abc.ABC):
    """
    This works like a dataset class of sorts. Given a directory containing .svs slides,
    it lists all of them and allows opening them and iterating them by index.
    """

    def __init__(
        self,
        input_dir: PathLike,
        extension: str,
        start_idx: int = 0,
        end_idx: int = -1,
        *args,
        **kwargs,
    ):
        """


        Parameters
        ----------
        root_dir : PathLike
        start_idx : int
            Indices to create a subset of the entire dataset. Can be used to do multi-node
            processing of several subsets of the entire dataset.
            start_idx is inclusive
            end_idx is inclusive
            start_idx (int) and end_idx(int):
            i.e. we set filenames = filenames[start_idx:end_idx+1], so that the first three elements of [0,1,2,3,4,5] are accessed
            with start_idx=0, end_idx=2, and the last three with start_idx=3, end_idx=5.

        end_idx : int
            See start_idx
        """
        input_dir = pathlib.Path(input_dir)
        self.input_dir = pathlib.Path(input_dir)
        if not self.input_dir.is_dir():
            raise NotADirectoryError(f"Given root directory '{input_dir}' is not a valid directory.")

        self.logger = logging.getLogger(type(self).__name__)
        self.s_idx_start = start_idx
        self.s_idx_end = end_idx
        self.extension = extension
        self.file_names = self.init_file_names()

    def init_file_names(self) -> List[Dict[str, Union[pathlib.Path, Any]]]:
        """
        Navigate self.input_dir folder and get the file paths of all WSI files
        """
        file_names = []
        self.logger.info(f"Initializing file names from the root dir: {self.input_dir}")
        slides_fn_generator = self.input_dir.rglob(f"*{self.extension}")
        for file in slides_fn_generator:
            file_id = self.extract_unique_file_id(file)
            file_names.append(
                {
                    "path": pathlib.Path(file),
                    "id": file_id,
                }
            )
        num_files = len(file_names)
        self.logger.info(f"Slide manager found a total of {num_files} files")
        if num_files == 0:
            return []

        self.check_given_indices(num_files)
        self.logger.info(f"Creating Slide Dataset from index {self.s_idx_start} to index {self.s_idx_end}")
        return file_names[self.s_idx_start : self.s_idx_end + 1]

    def check_given_indices(self, num_files):
        """
        Checks if the given start- and end-index are valid. That is:
        1. Set the end index to the last file index if set to -1
        2. Check if the end index is larger than the starting index
        3. Check if the starting index is not larger than the last file

        This function will throw a RuntimeError if something is wrong. If not, the rest of __init__ can continue.
        """
        if self.s_idx_end == -1:
            self.s_idx_end = num_files - 1
        elif self.s_idx_end >= num_files:
            self.logger.info(
                f"The provided slide ending idx {self.s_idx_end} is larger than the full dataset ({num_files})."
                f"The ending index is set to the last file of the full dataset: {num_files - 1}"
            )
            self.s_idx_end = num_files - 1
        else:
            pass

        if self.s_idx_end < self.s_idx_start:
            raise RuntimeError(
                f"Your starting index {self.s_idx_start} should be smaller than the end index {self.s_idx_end}."
            )

        if self.s_idx_start >= num_files:
            raise RuntimeError(f"The index to start at is {self.s_idx_start} while there are only {num_files} files")

    def open_slide(self, idx):
        """
        Opens a specific slide from the file path list, by index
        """
        if len(self.file_names) > idx >= 0:
            return SlideImage(
                file_name=self.file_names[idx]["path"],
                slide_uid=self.file_names[idx]["id"],
            )
        self.logger.info(f"There are only {len(self)} slides")

    def __iter__(self):
        for f_name in self.file_names:
            try:
                yield SlideImage(
                    file_name=f_name["path"],
                    slide_uid=f_name["id"],
                )
            except KeyError:
                self.logger.warning(f"Slide {f_name['id']} cannot be parsed. Skipping...")
                continue

    def __len__(self):
        return len(self.file_names)

    def get_file_names(self):
        return self.file_names

    @abc.abstractmethod
    def extract_unique_file_id(self, file: pathlib.Path):
        """
        Method to be implemented by subclasses, which extract a unique file id from the data.
        This depends on the dataset.

        Parameters
        ----------
        file : pathlib.Path
        """
        pass
