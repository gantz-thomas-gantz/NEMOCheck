#!/bin/bash

# Compute monthly climatologies (2011–2022) for selected NEMO files in ../model
# and write results to ../processed
# Assumes CDO is installed and available in PATH

# Directories
input_dir="../model"
output_dir="../processed"
tmp_dir=".tmp"

# File codes matching xx_codes in Python
cfgs=(00 01 02 04 05 06 07 08 09 10 12 14 15 16 17)

# --clean option: remove generated climatology files
if [[ "$1" == "--clean" ]]; then
    echo "Removing climatology output files from ${output_dir}..."
    for code in "${cfgs[@]}"; do
        output_file="${output_dir}/nemo${code}_clim_2011_2022.nc"
        if [[ -f "$output_file" ]]; then
            rm "$output_file"
            echo "Deleted: $output_file"
        else
            echo "Not found: $output_file"
        fi
    done
    exit 0
fi

# Create temporary and output directories
mkdir -p "$tmp_dir" "$output_dir"

for code in "${cfgs[@]}"; do
    input_file="${input_dir}/nemo${code}_1m_201001_202212_grid_T.nc"
    tmp_file="${tmp_dir}/nemo${code}_2011_2022.nc"
    output_file="${output_dir}/nemo${code}_clim_2011_2022.nc"

    echo "Processing ${input_file} ..."

    # Step 1: Remove 2010
    cdo seldate,2011-01-01,2022-12-31 "$input_file" "$tmp_file"

    # Step 2: Compute monthly climatology
    cdo ymonmean "$tmp_file" "$output_file"
done

# Remove temporary files
rm -r "$tmp_dir"

echo "All climatologies written to ${output_dir}/"



