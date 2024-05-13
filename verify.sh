#!/bin/bash

# Define the directory to search
base_dir="./plant_images"

# Find all files in the directory recursively
find "$base_dir" -type f | while read file; do
  # Check if the file is a JPEG image
  if file "$file" | grep -qE 'JPEG image data'; then
    echo "Verified JPEG: $file"
  else
    echo "Non-JPEG file found: $file"
  fi
done