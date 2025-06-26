#!/bin/bash

# Compute monthly climatologies (2011–2022) for all NEMO files in ./model
# and write results to ./processed
# Assumes CDO is installed and available in PATH

# Directories
input_dir="model"
output_dir="processed"
tmp_dir=".tmp"

# Create temporary directory
mkdir -p "$tmp_dir"

for i in $(seq -w 00 14); do
    input_file="${input_dir}/nemo${i}_1m_201001_202212_grid_T.nc"
    tmp_file="${tmp_dir}/nemo${i}_2011_2022.nc"
    output_file="${output_dir}/nemo${i}_clim_2011_2022.nc"

    echo "Processing ${input_file} ..."

    # Step 1: Remove 2010
    cdo seldate,2011-01-01,2022-12-31 "$input_file" "$tmp_file"

    # Step 2: Compute monthly climatology
    cdo ymonmean "$tmp_file" "$output_file"
done

# Optional: remove temporary files
rm -r "$tmp_dir"

echo "All climatologies written to ${output_dir}/"

