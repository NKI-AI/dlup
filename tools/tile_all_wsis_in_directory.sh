#!/bin/bash

# Example script to find all .svs files in a specific directory, run tiling for each WSI, and save the output in a
# similar directory structure as in the given directory

# If file paths are absolute, run the .sh script from anywhere
# If file paths are relative, run the .sh script from the appropriate directory

# Run with 
# bash tile_all_wsis_in_direectory.sh source_directory extension output_directory mpp tile_size foreground_threshold
# e.g. 
# bash tile_all_wsis_in_directory.sh /absolute/or/relative/path/to/directory/containing/wsis .svs /absolute/or/relative/path/to/directory/to/save 2 2048 0.8

SOURCE_DIRECTORY=$1 # "/absolute/or/relative/path/to/directory/containing/wsis"
LEN_SOURCE_DIRECTORY=${#SOURCE_DIRECTORY}
EXTENSION=$2 # e.g. .svs
OUTPUT_DIRECTORY=$3 # "/absolute/or/relative/path/to/directory/to/save"
MPP=$4 # e.g. 2
TILE_SIZE=$5 # e.g. 2048
FOREGROUND_THRESHOLD=$6 # e.g. 0.8

# If the input dir looks like
# SOURCE_DIRECTORY
# ├── patient_dir_1
# │    ├── wsi_1.svs
# │    └── wsi_2.svs
# ├── patient_dir_2
# │    ├── wsi_1.svs
# └──  └── wsi_2.svs

# The output dir would look like, e.g., like this (depending on the given parameters, used WSIs, and version of DLUP)

# TARGET_DIRECTORY
# ├── patient_dir_1
# │    ├── wsi_1
# │    │   ├── mask.png
# │    │   ├── thumbnail.png
# │    │   ├── thumbnail_with_mask.png
# │    │   ├── tiles
# │    │   │   ├── 1_1.png
# │    │   │   ...
# │    │   │   └── 5_3.png
# │    │   └── tiles.json
# │    └── wsi_2.svs
# │    │   ├── mask.png
# │    │   ├── thumbnail.png
# │    │   ├── thumbnail_with_mask.png
# │    │   ├── tiles
# │    │   │   ├── 1_1.png
# │    │   │   ...
# │    │   │   └── 4_5.png
# │    │   └── tiles.json
# └── patient_dir_2
#      ├── wsi_1.svs
#      │   ├── mask.png
#      │   ├── thumbnail.png
#      │   ├── thumbnail_with_mask.png
#      │   ├── tiles
#      │   │   ├── 1_1.png
#      │   │   ...
#      │   │   └── 8_8.png
#      │   └── tiles.json
#      └── wsi_2.svs
#          ├── mask.png
#          ├── thumbnail.png
#          ├── thumbnail_with_mask.png
#          ├── tiles
#          │   ├── 1_1.png
#          │   ...
#          │   └── 6_5.png
#          └── tiles.json

find $SOURCE_DIRECTORY -name "*$EXTENSION" | while read line
do
    RELATIVE_DIR=${line:$LEN_SOURCE_DIRECTORY+1} # Strip the source directory from the found file path and the /
    RELATIVE_DIR_WITHOUT_FILE_EXTENSION=${RELATIVE_DIR%.*} # Strip the extension from the found file path

    dlup wsi tile \
    --mpp $MPP \
    --tile-size $TILE_SIZE \
    --foreground-threshold $FOREGROUND_THRESHOLD \
    $line \
    $OUTPUT_DIRECTORY/$RELATIVE_DIR_WITHOUT_FILE_EXTENSION
    # Pass the found filepath as input
    # Save the output in the same tree structure as source directory
done
