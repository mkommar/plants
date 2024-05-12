#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <source_directory> <destination_directory>"
    exit 1
fi

# Assign arguments to variables
src_dir=$1
dest_dir=$2

# Function to move files from a given directory
move_files () {
    local source=$1
    local destination=$2

    # Ensure the destination directory exists
    mkdir -p "$destination"

    # Get the list of files
    files=($(find "$source" -maxdepth 1 -type f))
    num_files=${#files[@]}
    num_files_to_move=$(($num_files * 20 / 100))

    # Adjust if 20% calculation results in less than 1 file
    if [ "$num_files_to_move" -eq 0 ] && [ "$num_files" -gt 0 ]; then
        num_files_to_move=1
    fi

    # Move the calculated number of files
    echo "Moving $num_files_to_move files from $source to $destination..."
    for ((i=0; i<$num_files_to_move; i++)); do
        mv "${files[$i]}" "$destination"
    done
}

# Export the function so it can be used in subshells
export -f move_files

# Use find to iterate over each directory and move files accordingly
find "$src_dir" -type d | while read dir; do
    # Compute the corresponding destination directory
    subpath="${dir#$src_dir}"
    destination_path="$dest_dir$subpath"
    move_files "$dir" "$destination_path"
done

echo "Done moving files."
