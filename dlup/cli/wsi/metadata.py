# coding=utf-8
# Copyright (c) dlup contributors
import argparse

from tqdm import tqdm

from dlup.cli.wsi import WsiMetadataParser
from dlup.slide import Slide
from dlup.utils.io import write_json

parser_kwargs = {
    "parents": [
        WsiMetadataParser(epilog="Save all metadata from slides to disk, e.g. to find optimal mpp/magnification.")
    ],
    "add_help": False,
    "description": "Save metadata of folder of WSIs.",
}


def main(args: argparse.Namespace):
    """Main function that recursively parses metadata from a folder of slides.

    Parameters
    ----------
    args: argparse.Namespace
        The arguments passed from the CLI. Run with `-h` to see the required parameters

    Returns
    -------
    None
    """
    # TODO: add possibility to not dump in one file but for each slide a new one.
    # Get all svs
    metadata = {}
    all_slides = args.input_dir.glob(f"**/*.{args.extension}")
    for curr_slide in tqdm(all_slides):
        tqdm.write(f"Reading: {curr_slide}.")
        try:
            slide = Slide(curr_slide, curr_slide.name.split(".{args.extension}")[0])
        except RuntimeError as e:
            tqdm.write(f"RuntimeError: {e}. Skipping...")
            continue
        data = {"mpp": slide.mpp, "magnification": slide.magnification, "levels": slide.levels}
        if args.save_full_metadata:
            data["properties"] = slide.properties

        metadata[str(curr_slide)] = data

    write_json(args.output_file, metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parser_kwargs)  # type: ignore
    cli_args = parser.parse_args()
    main(cli_args)
