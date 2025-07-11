# 🌊 NEMOCheck

> **A comprehensive, modular, and reproducible framework for validating and analyzing NEMO ocean model simulations**

---

## 🚀 Abstract

**NEMOCheck** is a robust validation and analysis toolkit for the [NEMO ocean model](https://www.nemo-ocean.eu/). It empowers to perform thorough, reproducible, and insightful assessments of model performance. Supported comparison modes include both model–model and model–observation evaluations for essential oceanographic variables:

- **Sea Surface Temperature (SST)**
- **Sea Surface Salinity (SSS)**
- **Mixed Layer Depth (MLD)**

NEMOCheck provides both interactive Jupyter notebooks for exploration and scripting, as well as non-interactive workflows for automation—all within a locked, reproducible conda environment. The modular design means you can easily extend analyses to new model configurations and diagnostics.

---

## 🧰 Features

- **Interactive Analysis:** Explore, plot, and tune parameters in Jupyter notebooks.
- **Quantitative Diagnostics:** Powerful statistics tools (PCA, vertical profiles, zonal means, and more).
- **Flexible Visualization:** Effortless side-by-side model/observation comparisons.
- **Automated Processing:** Scripts for climatology computation, data normalization, and standardization.
- **Reproducible Environments:** Conda-lock guarantees bitwise identical dependencies on Linux and macOS.
- **Easy Extensibility:** Plug in new configurations and methods with minimal friction.

---

## 🧩 Supported Model Configurations

For a comprehensive overview of supported NEMO configurations, check out:

📄 [Model Configurations Guide (PDF)](docs/tests_orca05.pdf)

This document provides:
- Overview of supported NEMO configurations
- Configuration-specific parameters and settings

---

## ⚡ Quick Start

Get started with NEMOCheck by following these steps:

1. **Install Git LFS (Large File Storage)**  
   Git LFS is required to efficiently handle large files in this repository.
   - On Ubuntu/Linux:
     ```bash
     sudo apt-get install git-lfs
     ```
   - On macOS (with Homebrew):
     ```bash
     brew install git-lfs
     ```

2. **Initialize Git LFS**  
   Set up Git LFS for your user account:
   ```bash
   git lfs install
   ```

3. **Clone the Repository**
Download the project code to your local machine (note: this may take a while due to \~7 GB of data):

```bash
git clone https://github.com/gantz-thomas-gantz/NEMOCheck.git
cd NEMOCheck
```

4. **Set Up the Environment**  
   Run the provided script to create a Python environment and install dependencies:
   ```bash
   ./env/install_env.sh
   ```

5. **Start JupyterLab**  
   Once setup is complete, activate the environment and launch JupyterLab. Follow the on-screen instructions.

You are now ready to explore and use NEMOCheck in your browser via **JupyterLab**!

- Open and run any notebook as usual in JupyterLab.
- **Be sure to select the `nemocheck_env` kernel** in JupyterLab for compatibility.
- You can also run interactive plotting scripts directly in the terminal, for example:

  ```bash
  python src/utils/plot/model-model/horizontal.py
  ```

---

### 🔒 About the Environment Setup

- The install script uses `conda-lock.yml`, which pins **all dependencies** with exact versions for reproducibility.
- For a more human-readable list of dependencies, see `environment.yml`.
- The `conda-lock.yml` file was generated from `environment.yml` by running:

  ```bash
    conda-lock lock -f environment.yml --micromamba -p linux-64 -p osx-64 -p osx-arm64
  ```

- This means the environment and code are intended to work on both **Linux** and **macOS** platforms.

---

## 🗃️ Data Directory Structure

The `data/` directory is structured as follows:

```text
data/
  ├── model/         # Raw NEMO outputs (horizontal fields, sections) & ORCA05 mesh
  ├── obs/           # Observational datasets (SST, SSS, MLD, EKE)
  ├── processed/     # Preprocessed, standardized files ready for analysis
  ├── weights/       # Interpolation weight files
  └── processing/    # Scripts for data processing
```

### 🛠️ Data Processing

Processed data is generated via one of the following methods:

- **Climatology computation bash script**
  - Script: `data/compute_climatology.sh`
  - Output: `data/processed/nemo<00>_clim_2011_2022.nc`
  - Usage: Horizontal model–model comparison: `src/interactive/plotting_tool`

- **Processing notebook**
  - Notebook: `data/processing.ipynb`
  - Output: `data/processed/model.nc`, `data/processed/combined_observations.nc`
  - Usage:
    - Horizontal model–obs comparison: `src/interactive/plotting_tool`
    - Error profiles for model–obs: `src/interactive_plotting_tool/`
    - Non-interactive SST diagnostics: `src/non-interactive/sst.ipynb`

- **Model normalization function**
  - Function: `normalize_model_file()` in `src/utils/data/general.py`
  - Output: `data/processed/nemo<00>.nc`
  - Usage:
    - `src/quantitative/eke.ipynb`
    - `src/quantitative/zonal_sst.ipynb`

Further details on the data pipeline are documented at the top of `data/processing.ipynb`.

#### 📄 File Types

- **`model.nc`**: Derived from the `00` configuration and uses an MLD criterion that matches the observational dataset.
- **`nemo<00>.nc`**: Contains SST, SSS, and MLD (model definition), interpolated to a regular 1° grid.
- **`nemo<00>_clim_2011_2022.nc`**: Retains all model variables on the native grid for inter-model comparisons.

---

## 📁 Repository Structure

```text
NEMOCheck/
├── 📄 README.md              # You are here!
├── 📄 LICENSE               # CeCILL FREE SOFTWARE License
├── 📄 pyproject.toml        # Python package configuration
├── 🗂️ env/                   # Environment setup
│   ├── environment.yml       # Human-readable dependencies
│   ├── conda-lock.yml        # Locked dependencies for reproducibility
│   └── install_env.sh        # One-click environment setup
├── 🗂️ data/                  # Data storage and processing
│   └── processing/           # Data pipeline scripts & notebooks
├── 🗂️ docs/                  # Documentation files
└── 🗂️ src/                   # Main source code
    ├── 🎯 interactive/        # ⭐ Interactive Jupyter notebooks (START HERE!)
    │   ├── plotting_tool.ipynb     # 🌟 Beautiful interactive plots & comparisons
    │   └── numerical_schemes.ipynb # Model scheme analysis
    ├── 📊 quantitative/       # Statistical analysis notebooks
    │   ├── eke.ipynb          # Eddy kinetic energy analysis
    │   ├── pca.ipynb          # Principal component analysis
    │   ├── vertical.ipynb     # Vertical profile diagnostics
    │   └── zonal_sst.ipynb    # Zonal SST analysis
    ├── 🔬 non-interactive/    # Automated analysis scripts
    │   └── sst.ipynb          # Sea surface temperature diagnostics
    ├── 🧠 understanding/      # Model understanding tools
    │   ├── mesh.ipynb         # Grid and mesh analysis
    │   └── plotting.ipynb     # Plotting utilities exploration
    └── 🛠️ utils/              # Reusable utility modules
        ├── data/              # Data processing utilities
        └── plot/              # Visualization utilities
            ├── model_model/   # Model-to-model comparison plots
            └── model_obs/     # Model-to-observation comparison plots
```

### 🌟 Where to Start

**New users should begin with the `src/interactive/` notebooks**, particularly:

- **`plotting_tool.ipynb`** 🎨 - **The crown jewel!** Features interactive visualizations for comparing models and observations. 

- **`numerical_schemes.ipynb`** 🔍 - Deep dive into model numerical schemes and their impact on results.

The interactive notebooks provide an intuitive entry point to NEMOCheck's capabilities, with rich visualizations that make complex oceanographic data immediately accessible and interpretable.

---

## ➕ Adding a New Configuration

To include a new NEMO configuration in the analysis:

1. Open the last cell of `data/processing/processing.ipynb`, update the `cfgs` list, and re-run the cell.
2. Open `data/compute_climatology.sh`, update the `cfgs` list, and re-run the script.

---

## 📚 References

**Gurvan Madec (2024).** *NEMO Ocean Engine Reference Manual* (v5.0). Zenodo. https://doi.org/10.5281/zenodo.14515373

---

## 📖 Citation

If you use NEMOCheck, please consider citing this repository:

```bibtex
@software{nemocheck,
  author = {Thomas Gantz},
  title = {NEMOCheck: Validation and Analysis Framework for NEMO Ocean Model},
  url = {https://github.com/gantz-thomas-gantz/NEMOCheck},
  year = {2025}
}
```



