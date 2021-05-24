# coding=utf-8
# Copyright (c) dlup contributors
"""Code generator for dlup. This script converts torchvision transforms to a dlup variant."""
import pathlib
import warnings

import packaging.version
import torchvision
from jinja2 import Environment

from dlup.utils import str_to_class

# The format is: torchvision.transform name,
# whether the transform can be similarly applied to the mask, and to which keys should the transform be applied.
available_transforms = [
    # Transforms leaving the mask invariant
    ("RandomGrayscale", True, ["image"]),
    ("RandomErasing", True, ["image"]),
    ("LinearTransformation", True, ["image"]),
    ("ColorJitter", True, ["image"]),
    ("Grayscale", True, ["image"]),
    ("Normalize", True, ["image"]),
    # Transforms that adjust the mask without interpolation defects
    ("CenterCrop", True, ["image", "mask"]),
    ("Pad", True, ["image", "mask"]),
    ("FiveCrop", True, ["image", "mask"]),
    ("TenCrop", True, ["image", "mask"]),
    # Wrap transforms which cannot have any mask
    ("RandomCrop", False, ["image"]),
    ("RandomHorizontalFlip", False, ["image"]),
    ("RandomVerticalFlip", False, ["image"]),
    ("RandomPerspective", False, ["image"]),
    ("RandomRotation", False, ["image"]),
    ("RandomAffine", False, ["image"]),
]

newer_available_transforms = [
    ("RandomInvert", True, ["image"]),
    ("RandomPosterize", True, ["image"]),
    ("RandomSolarize", True, ["image"]),
    ("RandomAdjustSharpness", True, ["image"]),
    ("RandomAutocontrast", True, ["image"]),
    ("RandomEqualize", True, ["image"]),
]

# Only supported in newer versions
if packaging.version.parse(torchvision.__version__) >= packaging.version.parse("0.9.0"):
    available_transforms += newer_available_transforms
else:
    warnings.warn(
        f"torchvision version {torchvision.__version__} does not support all transforms. Requires at least 0.9.0. "
        f"These transforms are missing: {', '.join(newer_available_transforms)}"
    )


def code_generator(template_filename: pathlib.Path, output_filename: pathlib.Path):
    """
    Generate code based on Jinja2 template

    Parameters
    ----------
    template_filename : pathlib.Path
    output_filename : pathlib.Path

    Returns
    -------

    """
    env = Environment(autoescape=False, optimized=False)
    env.globals.update(zip=zip)

    print(f"Generating code from template {template_filename}. Writing to {output_filename}.")
    with open(template_filename, "r") as f:
        template = env.from_string(f.read())

    # Render the template
    with open(output_filename, "w") as f:
        # Stream renders the template to a stream for writing to a file.
        # Pass your variables as kwargs

        # Parse docstrings
        docstrings = []
        for class_name, _, _ in available_transforms:
            curr_docstring = str_to_class("torchvision.transforms", class_name).__doc__
            new_docstring = (
                f'"""Dlup wrapped version of {class_name}. '
                f"Note that the examples might require modifications.\n\n"
                f'    **Original docs**:\n    {curr_docstring}\n    """'
            )
            docstrings.append(new_docstring)

        template.stream(
            docstrings=docstrings,
            torchvision_transforms=available_transforms,
        ).dump(f)


def main():
    """Main entrance point for code generator."""
    template_filename = pathlib.Path("dlup/transforms/torchvision_transforms.py.j2")
    output_filename = template_filename.with_suffix("")
    code_generator(template_filename, output_filename)


if __name__ == "__main__":
    main()
