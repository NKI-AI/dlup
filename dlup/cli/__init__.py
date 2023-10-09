# Copyright (c) dlup contributors
"""DLUP Command-line interface. This is the file which builds the main parser."""
import argparse
import pathlib


def dir_path(path: str) -> pathlib.Path:
    """Check if the path is a valid directory.
    Parameters
    ----------
    path : str
    Returns
    -------
    pathlib.Path
        The path as a pathlib.Path object.
    """
    _path = pathlib.Path(path)
    if _path.is_dir():
        return _path
    raise argparse.ArgumentTypeError(f"{path} is not a valid directory.")


def file_path(path: str, need_exists: bool = True) -> pathlib.Path:
    """Check if the path is a valid file.
    Parameters
    ----------
    path : str
    need_exists : bool

    Returns
    -------
    pathlib.Path
        The path as a pathlib.Path object.
    """
    _path = pathlib.Path(path)
    if need_exists:
        if _path.is_file():
            return _path
        raise argparse.ArgumentTypeError(f"{path} is not a valid file.")
    return _path


def main() -> None:
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

    # Prevent circular import
    from dlup.cli.mask import register_parser as register_mask_subcommand

    # Whole slide images related commands.
    register_mask_subcommand(root_subparsers)

    args = root_parser.parse_args()
    args.subcommand(args)


if __name__ == "__main__":
    main()
