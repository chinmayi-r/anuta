#!/bin/bash

#* Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <directory> <string_to_prepend>"
    exit 1
fi

DIRECTORY=$1
PREFIX=$2

#* Check if the provided directory exists
if [ ! -d "$DIRECTORY" ]; then
    echo "Error: Directory '$DIRECTORY' does not exist."
    exit 1
fi

#* Iterate over all files in the directory
for FILE in "$DIRECTORY"/*; do
    #* Skip if not a file
    if [ -f "$FILE" ]; then
        BASENAME=$(basename "$FILE") #* Get the file name without the path
        NEWNAME="${DIRECTORY}/${PREFIX}${BASENAME}" #* Create the new name
        mv "$FILE" "$NEWNAME" #* Rename the file
    fi
done

echo "All files in '$DIRECTORY' have been renamed with the prefix '$PREFIX'."