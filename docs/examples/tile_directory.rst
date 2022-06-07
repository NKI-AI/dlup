
Use DLUP CLI to tile all WSIs in a directory
-----------------------------------------------------

Below is an example of a `.sh` script that can be used to tile all WSIs in a directory, saving the tiles in a similar directory structure as the directory structure of the source.

This script finds all files of a given extension (e.g. `.svs` for TCGA) in a given directory,
tiles each WSI according to the given configuration, and saves it in a similar directory structure.

If given file paths to the directories are absolute, you can run the .sh script from anywhere. Otherwise, run it from the proper directory.

This simple script can then be run with

.. code-block:: bash

    bash tile_all_wsis_in_directory.sh \
    source_directory \
    extension \
    output_directory \
    mpp \
    tile_size \
    foreground_threshold

e.g.

.. code-block:: bash

    bash tile_all_wsis_in_directory.sh \
    /absolute/or/relative/path/to/directory/containing/wsis \
    .svs \
    /absolute/or/relative/path/to/directory/to/save \
    2 \
    2048 \
    0.8

.. code-block:: bash

    #!/bin/bash

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

    # The output dir would look like this (depending on the given parameters, used WSIs, and version of DLUP)

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
