# coding=utf-8
# Copyright (c) dlup contributors
"""DLUP Command-line interface. This is the file which builds the main parser."""
import argparse


def main():
    """
    Console script for dlup.
    """
    # From https://stackoverflow.com/questions/17073688/how-to-use-argparse-subparsers-correctly
    root_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    root_subparsers = root_parser.add_subparsers(help="Possible DLUP CLI utils to run.")
    root_subparsers.required = True
    root_subparsers.dest = "subcommand"

    # Prevent circular import
    from dlup.cli.wsi import register_parser as register_wsi_subcommand

    # Whole slide images related commands.
    register_wsi_subcommand(root_subparsers)

    args = root_parser.parse_args()
    args.subcommand(args)


if __name__ == "__main__":
    main()
