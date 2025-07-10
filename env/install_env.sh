#!/usr/bin/env bash
set -euo pipefail

env_name="nemocheck_env"
lock_file="conda-lock.yml"

MICROMAMBA="${HOME}/bin/micromamba"

if ! command -v micromamba &> /dev/null && [ ! -f "$MICROMAMBA" ]; then
  echo "Downloading micromamba..."
  mkdir -p "$HOME/bin"
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -C "$HOME/bin" -xvj bin/micromamba
  chmod +x "$MICROMAMBA"
  export PATH="$HOME/bin:$PATH"
elif [ -f "$MICROMAMBA" ]; then
  export PATH="$HOME/bin:$PATH"
fi

# Use the lock file for reproducible install
echo "Creating or updating environment '$env_name' from lock file..."
micromamba create -y -n "$env_name" --file "$lock_file"

echo "Installing Jupyter kernel..."
micromamba run -n "$env_name" python -m ipykernel install --user --name "$env_name"

echo ""
echo "Environment '$env_name' is ready."
echo "To activate it: micromamba activate $env_name"
