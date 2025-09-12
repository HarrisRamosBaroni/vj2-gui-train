#!/bin/bash

# Define paths
SOURCE_REPO="/home/harris/Documents/y2/industry/vj2-gui"
TARGET_REPO="/home/harris/Documents/y2/industry/train"

# List of files/directories to sync (same as original script)
FILES_TO_SYNC=(
    "config.py"
    "src/"
    "gui_world_model/utils/"
    "gui_world_model/__init__.py"
    "gui_world_model/predictor.py"
    "gui_world_model/predictor_additive.py"
    "gui_world_model/predictor_a_encoded.py"
    "gui_world_model/predictor_cross_attention.py"
    "gui_world_model/predictor_film.py"
    "gui_world_model/predictor_noop.py"
    "gui_world_model/predictor_prob.py"
    "gui_world_model/predictor_prob_film.py"
    "conditioned_gesture_generator/"
    "latent_action_model/"
    # "training/test/"
    "training/validators/"
    "training/__init__.py"
    "training/dataloader.py"
    "training/losses.py"
    "training/migrate_npz_to_npy.py"
    "training/train_v2.py"
    "training/utils.py"
    "preprocess/generate_split_manifest.py"
    # ".gitignore"
    # "RCLONE_README.md"
    # "TRAIN_README.md"
)

echo "Starting safe synchronization of selected files from GUI repo..."

# Copy each file/directory individually using cp
for item in "${FILES_TO_SYNC[@]}"; do
    if [[ "$item" == *"#"* ]]; then
        echo "Skipping commented item: $item"
        continue
    fi
    
    echo "Syncing: $item"
    
    if [ -e "$SOURCE_REPO/$item" ]; then
        # Create target directory if it doesn't exist
        target_path="$TARGET_REPO/$item"
        target_dir="$(dirname "$target_path")"
        mkdir -p "$target_dir"
        
        if [[ "$item" == */ ]]; then
            # For directories, copy recursively and update newer files
            echo "  Copying directory: $SOURCE_REPO/$item -> $target_path"
            rsync -avu --existing "$SOURCE_REPO/$item" "$target_path" 2>/dev/null || true
            rsync -avu --ignore-existing "$SOURCE_REPO/$item" "$target_path"
        else
            # For files, copy if target doesn't exist or files are different
            if [ ! -f "$target_path" ]; then
                echo "  Copying new file: $SOURCE_REPO/$item -> $target_path"
                cp "$SOURCE_REPO/$item" "$target_path"
            elif ! cmp -s "$SOURCE_REPO/$item" "$target_path"; then
                echo "  Updating file (content differs): $SOURCE_REPO/$item -> $target_path"
                cp "$SOURCE_REPO/$item" "$target_path"
            else
                echo "  File $item is up to date"
            fi
        fi
    else
        echo "Warning: $item does not exist in source repo, skipping"
    fi
done

echo "Safe synchronization complete!"
echo "Note: This script only copies/updates files, it never deletes anything from the target repo."
