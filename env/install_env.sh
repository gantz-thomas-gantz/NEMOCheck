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

# Detect user shell init file (bash priority, fallback to .bash_profile or .profile)
if [ -f "$HOME/.bashrc" ]; then
  shell_init_file="$HOME/.bashrc"
elif [ -f "$HOME/.bash_profile" ]; then
  shell_init_file="$HOME/.bash_profile"
else
  shell_init_file="$HOME/.profile"
fi

# Append micromamba shell init to the shell init file if not already present
if ! grep -q 'micromamba shell init --shell bash' "$shell_init_file"; then
  echo "Adding micromamba shell init to $shell_init_file ..."
  micromamba shell init --shell bash --root-prefix=~/.local/share/mamba >> "$shell_init_file"
  echo "" >> "$shell_init_file"
fi

# Initialize micromamba shell hook for the current session (immediate effect)
if [[ -n "${BASH_VERSION-}" ]]; then
  eval "$(micromamba shell hook --shell bash)"
fi

# Verify lock file exists
if [ ! -f "$lock_file" ]; then
  echo "Error: Lock file not found at $lock_file"
  exit 1
fi

echo "Creating or updating environment '$env_name' from lock file..."
micromamba create -y -n "$env_name" --file "$lock_file"

echo "Installing local utils package in editable mode..."
micromamba run -n "$env_name" pip install -e "$script_dir/../src/utils"

echo "Installing Jupyter kernel..."
micromamba run -n "$env_name" python -m ipykernel install --user --name "$env_name"

echo ""
echo "Environment '$env_name' is ready."
echo "To activate it in this session: micromamba activate $env_name"
echo "In future sessions, just open a new terminal and run: micromamba activate $env_name"


