# coding=utf-8
# Copyright (c) DLUP Contributors
import argparse


# From: https://stackoverflow.com/a/52025430/576363
class ArgumentParserWithDefaults(argparse.ArgumentParser):
    def __init__(
        self,
        add_logging_disable=True,
        add_verbose=True,
        add_debug=True,
        **kwargs,
    ):
        super().__init__()
        if add_logging_disable:
            self.add_argument(
                "--no-log",
                help="Switch off logging. Otherwise will log to current directory.",
                action="store_true",
            )
        if add_verbose:
            self.add_argument(
                "-v",
                "--verbose",
                action="count",
                help="Verbosity level, as an integer.",
                default=1,
            )
        if add_debug:
            self.add_argument(
                "--debug",
                type=bool,
                default=False,
                help="Enable debug mode.",
            )

    def add_argument(self, *args, help=None, default=None, **kwargs):
        if help is not None:
            kwargs["help"] = help
        if default is not None and args[0] != "-h":
            kwargs["default"] = default
            if help is not None:
                kwargs["help"] += f" (Default: {default})"
        super().add_argument(*args, **kwargs)
