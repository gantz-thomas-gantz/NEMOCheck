#!/usr/bin/env bash
set -euo pipefail

env_name="nemocheck_env"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lock_file="${script_dir}/conda-lock.yml"
MICROMAMBA="${HOME}/bin/micromamba"

# Download micromamba if missing
if [ ! -f "$MICROMAMBA" ]; then
  mkdir -p "$HOME/bin"
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -C "$HOME/bin" -xvj bin/micromamba
  chmod +x "$MICROMAMBA"
fi

# Create or update environment (no activation needed)
"$MICROMAMBA" create -y -n "$env_name" --file "$lock_file"

# Install local utils package in editable mode
"$MICROMAMBA" run -n "$env_name" pip install -e "$script_dir/../"

# Install Jupyter kernel
"$MICROMAMBA" run -n "$env_name" python -m ipykernel install --user --name "$env_name"

echo "Environment '$env_name' is ready as a Jupyter kernel."
echo "You can select it in Jupyter Lab's kernel list."

