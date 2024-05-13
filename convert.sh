#!/bin/bash

# Define the directory to search
base_dir="./plant_images"

# Find all files in the directory recursively
find "$base_dir" -type f | while read file; do
  # Check if the file is an image and not a JPEG
  if file "$file" | grep -qE 'image' && ! file "$file" | grep -qE 'JPEG image data'; then
    # Construct new JPEG filename
    new_file="${file%.*}.jpg"
    
    # Convert the file to JPEG using ImageMagick
    convert "$file" "$new_file"
    
    # Optionally delete the original file
    # rm "$file"
    
    echo "Converted $file to $new_file"
  fi
done
