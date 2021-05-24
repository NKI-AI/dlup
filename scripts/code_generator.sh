#!/bin/bash

python scripts/code_generators/generate_torchvision_code.py
black dlup/transforms/torchvision_transforms.py

