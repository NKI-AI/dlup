# coding=utf-8
# Copyright (c) dlup contributors
"""Dlup command-line interface. This is the file which builds the main parser."""
import argparse
import sys

from dlup.cli.slidescore.download_labels import main as dl_slidescore_labels
from dlup.cli.slidescore.download_labels import parser_kwargs as dl_slidescore_labels_parser_kwargs  # type: ignore
from dlup.cli.slidescore.download_wsis import main as dl_slidescore_wsis
from dlup.cli.slidescore.download_wsis import parser_kwargs as dl_slidescore_wsis_parser_kwargs
from dlup.cli.slidescore.upload_labels import main as ul_slidescore_labels
from dlup.cli.slidescore.upload_labels import parser_kwargs as ul_slidescore_labels_parser_kwargs  # type: ignore
from dlup.cli.wsi.metadata import main as extract_metadata
from dlup.cli.wsi.metadata import parser_kwargs as metadata_parser_kwargs
from dlup.cli.wsi.tiling import main as tiling
from dlup.cli.wsi.tiling import parser_kwargs as tiling_parser_kwargs


def main():
    """
    Console script for dlup.
    """
    # From https://stackoverflow.com/questions/17073688/how-to-use-argparse-subparsers-correctly
    root_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    root_subparsers = root_parser.add_subparsers(help="Possible dlup CLI utils to run.")
    root_subparsers.required = True
    root_subparsers.dest = "subcommand"

    # --- WSI subparser ---
    wsi_parser = root_subparsers.add_parser("wsi", help="WSI parser")
    wsi_subparsers = wsi_parser.add_subparsers(help="WSI subparser")
    wsi_subparsers.required = True
    wsi_subparsers.dest = "subcommand"

    # Tiling subparser
    tiling_parser = wsi_subparsers.add_parser("tiling", **tiling_parser_kwargs)
    tiling_parser.set_defaults(subcommand=tiling)

    # Metadata subparser
    metadata_parser = wsi_subparsers.add_parser("metadata", **metadata_parser_kwargs)
    metadata_parser.set_defaults(subcommand=extract_metadata)

    # --- SlideScore subparser ---
    slidescore_parser = root_subparsers.add_parser("slidescore", help="SlideScore parser")
    slidescore_subparsers = slidescore_parser.add_subparsers(help="SlideScore subparser")
    slidescore_subparsers.required = True
    slidescore_subparsers.dest = "subcommand"

    # SlideScore download wsi
    dl_slidescore_parser = slidescore_subparsers.add_parser("download-wsi", **dl_slidescore_wsis_parser_kwargs)
    dl_slidescore_parser.set_defaults(subcommand=dl_slidescore_wsis)

    # SlideScore download labels
    dl_slidescore_parser = slidescore_subparsers.add_parser("download-labels", **dl_slidescore_labels_parser_kwargs)
    dl_slidescore_parser.set_defaults(subcommand=dl_slidescore_labels)

    # SlideScore upload labels
    ul_slidescore_parser = slidescore_subparsers.add_parser("upload-labels", **ul_slidescore_labels_parser_kwargs)
    ul_slidescore_parser.set_defaults(subcommand=ul_slidescore_labels)

    args = root_parser.parse_args()
    args.subcommand(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
