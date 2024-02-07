# Copyright (c) dlup contributors
import argparse
import json
import pathlib

from dlup import SlideImage


def info(args: argparse.Namespace) -> None:
    """Return available slide properties."""
    slide = SlideImage.from_file_path(args.slide_file_path)
    props = slide.properties
    if not props:
        return print("No properties found.")
    if args.json:
        print(json.dumps(dict(props)))
        return

    for k, v in props.items():
        print(f"{k}\t{v}")


def register_parser(parser: argparse._SubParsersAction) -> None:  # type: ignore
    """Register wsi commands to a root parser."""
    wsi_parser = parser.add_parser("wsi", help="WSI parser")
    wsi_subparsers = wsi_parser.add_subparsers(help="WSI subparser")
    wsi_subparsers.required = True
    wsi_subparsers.dest = "subcommand"

    # Get generic slide infos.
    info_parser = wsi_subparsers.add_parser("info", help="Return available slide properties.")
    info_parser.add_argument(
        "slide_file_path",
        type=pathlib.Path,
        help="Input slide image.",
    )
    info_parser.add_argument(
        "--json",
        action="store_true",
        help="Print available properties in json format.",
    )
    info_parser.set_defaults(subcommand=info)
