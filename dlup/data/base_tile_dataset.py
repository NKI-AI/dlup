# coding=utf-8
# Copyright (c) DLUP Contributors

import json
import logging
import pathlib
from typing import Callable, Generator, List, Optional

import torchvision
from PIL import Image
from torch.utils import data


class BaseTileDataset(data.Dataset):
    """This dataset retrieves tiles produced by DLUP preprocessing."""

    def __init__(
        self,
        root_dir: pathlib.Path,
        transforms: Callable = torchvision.transforms.ToTensor(),
        slide_names: Optional[List[str]] = None,
        tile_extension: str = "png",
    ) -> None:
        """
        Initializes a BaseTileDataset instance.

        Parameters
        ------
        root_dir : pathlib.Path
            Path to the directory containing the tiled slides.
        transforms : Callable
            Any sequence of transforms to be applied to the image at fetching time.
        slide_names : List[str]
            List of strings corresponding to the names of the slides to be included in this dataset instance.
        tile_extension : str
            Extension of the tile images produced by the preprocessing algorithm.

        Returns
        ------
        None
        """
        self.root_dir = root_dir
        self.transforms = transforms
        self.slide_names = slide_names
        self.tile_extension = tile_extension

        self.logger = logging.getLogger(type(self).__name__)

        self.dicts = self.get_dicts()

    def __len__(self) -> int:
        return len(self.dicts)

    def __getitem__(self, item: int) -> dict:
        """
        Returns a dictionary for a dataset sample.

        Parameters
        ------
        item : int
            Numerical index specifying which dataset sample to retrieve.

        Returns
        ------
        dict
            A dictionary containing image, image path, and other useful metadata.
        """
        # load image with pil, covert it to a torch tensor
        img = Image.open(self.dicts[item]["image_path"])

        # apply transforms
        img = self.transforms(img)

        # include image to the tile dictionary and return it
        img_dict = {"image": img, **self.dicts[item]}
        return img_dict

    def get_slide_dirs(self) -> Generator[pathlib.Path, None, None]:
        """Get the list of tiled slides used by this dataset."""
        if self.slide_names:
            return (self.root_dir / slide_name for slide_name in self.slide_names)
        return self.root_dir.glob("*/")

    @staticmethod
    def add_fields_to_tile(
        tile_dict: dict,
    ) -> dict:
        """
        Fetch additional information to be included with each dataset sample.

        This method should be overridden in any class that inherits from BaseTileDataset.
        that wishes to fetch labels (e.g. bounding boxes, paths to label files) or any
        other tile metadata to be included in the tile dictionary retrieved by this class.

        Parameters
        ------
        tile_dict: dict
            The tile dictionary as built by BaseTileDataset. It should include all necessary information
            to uniquely identify each tile, and fetch any corresponding label (i.e. slide name, tile coordinates).
            It contains keys: ['image_path', 'slide_name', 'coordinates', 'level', 'size', 'overflow',
            'target_tile_size', 'idx']

        Returns
        ------
        dict
            A dictionary containing any additional key-value pairs to be added to the default tile dictionary.
        """
        return {}

    def get_dicts(self) -> List[dict]:
        """
        Generates a dataset map.

        Used to generate a dataset map, to be used both for easier retrieval during training,
        and to initialize the bounding boxes for each tile at the start.

        Returns
        ------
        List[dict]
            A list of dicts containing essential data for each sample.
        """
        dicts = []

        # iterate over the directories of requested tiled slides
        for slide_dir in self.get_slide_dirs():

            # get a list of all tile paths and iterate over it
            tile_dir = slide_dir / pathlib.Path("tiles")
            for tile_path in tile_dir.glob(f"*.{self.tile_extension}"):
                # generate dictionary basic information
                tile_dict = {"image_path": tile_path, "slide_name": slide_dir.stem}

                # add tile metadata to the dictionary
                json_path = slide_dir / pathlib.Path("json") / pathlib.Path(f"{tile_path.stem}.json")
                with open(json_path, "r") as file:
                    tile_dict.update(json.load(file))

                # fetch any additional information for each tile
                tile_dict.update(self.add_fields_to_tile(tile_dict))

                # build the tile dictionary
                dicts.append(tile_dict)

        return dicts
