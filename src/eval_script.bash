#!/bin/bash

directory="../networks/txts"
output_dir="../evals_test_data"

# Create output directory if it doesn't exist
mkdir -p "$output_dir"

# Check if the specific file exists
for file in "$directory"/*.txt; do
  # Check if the file exists (handles the case where no .txt files are present)
  if [[ -f "$file" ]]; then
    echo "Processing file: $file"
  
  # Get filename without directory and extension for the output file
  filename=$(basename "$file" .txt)
  output_file="$output_dir/${filename}_eval"
  
  # Execute the Python script with the appropriate arguments
  python ids_driver.py -a evaluate --network_filename "$file" \
    --eons_params ../config/eons.json \
    --proc_params ../config/ucaspian.json \
    --dataset_path ../data/ \
    --processes 10 \
    --save_eval "$output_file"
  
else
  echo "File $file_path not found"
  exit 1
fi

echo "Evaluation complete."

done