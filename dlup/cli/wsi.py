import argparse


def register_tiling_subparser(parse: argparse.ArgumentParser):

    tiling_parser = wsi_subparsers.add_parser("tiling", **test)
    tiling_parser.set_defaults(subcommand=tiling)
