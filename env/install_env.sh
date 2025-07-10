#!/usr/bin/env bash
set -euo pipefail

env_name="nemocheck_env"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MICROMAMBA="${HOME}/bin/micromamba"

# Detect OS and architecture
unameOut="$(uname -s)"
archOut="$(uname -m)"

case "${unameOut}" in
    Linux*)
        OS=linux
        ;;
    Darwin*)
        OS=osx
        ;;
    *)
        echo "Unsupported OS: ${unameOut}"
        exit 1
        ;;
esac

case "${archOut}" in
    x86_64*)
        ARCH=64
        ;;
    arm64*)
        ARCH=arm64
        ;;
    *)
        echo "Unsupported architecture: ${archOut}"
        exit 1
        ;;
esac

# Use the single conda-lock.yml file for all platforms
lock_file="${script_dir}/conda-lock.yml"

if [ ! -f "$lock_file" ]; then
    echo "Lock file $lock_file not found!"
    exit 1
fi

# Compose download URL for micromamba
if [[ "$OS" == "linux" && "$ARCH" == "64" ]]; then
    MICROMAMBA_URL="https://micro.mamba.pm/api/micromamba/linux-64/latest"
elif [[ "$OS" == "osx" && "$ARCH" == "64" ]]; then
    MICROMAMBA_URL="https://micro.mamba.pm/api/micromamba/osx-64/latest"
elif [[ "$OS" == "osx" && "$ARCH" == "arm64" ]]; then
    MICROMAMBA_URL="https://micro.mamba.pm/api/micromamba/osx-arm64/latest"
else
    echo "Unsupported OS/architecture combination: ${OS}/${ARCH}"
    exit 1
fi

# Download micromamba if missing
if [ ! -f "$MICROMAMBA" ]; then
  mkdir -p "$HOME/bin"
  curl -Ls "$MICROMAMBA_URL" | tar -xj -C "$HOME/bin" --strip-components=1 bin/micromamba
  chmod +x "$MICROMAMBA"
fi

# Create or update environment (no activation needed)
"$MICROMAMBA" create -y -n "$env_name" --file "$lock_file"

# Install local utils package in editable mode
"$MICROMAMBA" run -n "$env_name" pip install -e "$script_dir/../"

# Install Jupyter kernel
"$MICROMAMBA" run -n "$env_name" python -m ipykernel install --user --name "$env_name"

echo "Environment '$env_name' is ready and available as a Jupyter kernel."
echo "To launch JupyterLab using this environment, run:"
echo "  \"$MICROMAMBA\" run -n \"$env_name\" jupyter-lab"


