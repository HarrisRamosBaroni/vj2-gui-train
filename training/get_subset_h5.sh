#!/bin/bash

# Define command-line arguments and their descriptions
REMOTE_DIR=$1
OUTPUT_DIR=$2
NUM_FILES=$3
RCLONE_LSF_OPTS="--checkers 8 --fast-list"
RCLONE_COPY_OPTS="--transfers 6 --checkers 6 --drive-chunk-size 128M --fast-list"

# Check if all required arguments are provided
if [ -z "$REMOTE_DIR" ] || [ -z "$OUTPUT_DIR" ] || [ -z "$NUM_FILES" ]; then
    echo "Usage: $0 <remote_directory> <output_directory> <number_of_files | 'all'>"
    exit 1
fi

# Define intermediate directory and file names
# TEMP_DIR="temp"
FILES_LIST="files_to_copy.txt"

# Step 1: List files from the remote directory and pick a random sample
if [ "$NUM_FILES" = "all" ]; then
    echo "Listing all files from $REMOTE_DIR..."
    rclone lsf ${RCLONE_LSF_OPTS} --include "*.h5" "$REMOTE_DIR" > "$FILES_LIST"
else
    echo "Listing files from $REMOTE_DIR and sampling $NUM_FILES..."
    rclone lsf ${RCLONE_LSF_OPTS} --include "*.h5" "$REMOTE_DIR" | shuf -n "$NUM_FILES" > "$FILES_LIST"
fi
if [ $? -ne 0 ]; then
    echo "Error: Failed to list and sample files. Check your rclone configuration and remote path."
    exit 1
fi

# Step 2: Create a temporary directory for copying files
# mkdir -p "$TEMP_DIR"

# Step 3: Copy the sampled files from the remote to the temporary directory
echo "Copying sampled files to $OUTPUT_DIR..."
rclone copy -P ${RCLONE_COPY_OPTS} "$REMOTE_DIR" "$OUTPUT_DIR" --files-from "$FILES_LIST"
if [ $? -ne 0 ]; then
    echo "Error: Failed to copy files with rclone."
    exit 1
fi

# Step 4: Run the Python script to migrate the files
echo "Running migration script..."
python preprocess/generate_split_manifest.py --data-dir "$OUTPUT_DIR" --output-dir "$OUTPUT_DIR/manifests" --name experiment_A
if [ $? -ne 0 ]; then
    echo "Error: The Python manifests generation failed."
    exit 1
fi

# Step 5: Clean up temporary files and directories
echo "Cleaning up temporary files..."
# rm -r "$TEMP_DIR" "$FILES_LIST"
rm -r "$FILES_LIST"

echo "Pipeline complete. Processed files are in $OUTPUT_DIR"