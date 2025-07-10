#!/usr/bin/env bash
set -euo pipefail

env_name="nemocheck_env"

# Get directory of this script (so paths work regardless of where you run it from)
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

lock_file="${script_dir}/conda-lock.yml"
MICROMAMBA="${HOME}/bin/micromamba"

# Download micromamba if missing
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

# Initialize micromamba shell hook for the current bash session if not done yet
if [[ -n "${BASH_VERSION-}" ]]; then
  # Only do this if micromamba shell hook isn't already active
  if ! micromamba shell hook --shell bash &>/dev/null; then
    eval "$(micromamba shell hook --shell bash)"
  fi
fi

# Check lock file exists
if [ ! -f "$lock_file" ]; then
  echo "Error: Lock file not found at $lock_file"
  exit 1
fi

echo "Creating or updating environment '$env_name' from lock file..."
micromamba create -y -n "$env_name" --file "$lock_file"

echo "Installing Jupyter kernel..."
micromamba run -n "$env_name" python -m ipykernel install --user --name "$env_name"

echo ""
echo "Environment '$env_name' is ready."
echo "To activate it: micromamba activate $env_name"

