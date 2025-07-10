# NEMOCheck

## Abstract

NEMOCheck is a comprehensive validation and analysis framework for NEMO (Nucleus for European Modelling of the Ocean). This repository provides a suite of tools for evaluating ocean model performance through model-model and model-observation comparisons across multiple oceanographic variables including sea surface temperature (SST), sea surface salinity (SSS), mixed layer depth (MLD), and eddy kinetic energy (EKE).

The framework offers both interactive and non-interactive analysis capabilities:
- **Interactive tools**: Jupyter notebooks for exploratory data analysis, plotting, and parameter tuning
- **Quantitative analysis**: Statistical evaluation including PCA, vertical profiles, and zonal averaging
- **Visualization**: Horizontal and vertical plotting utilities for model output and observational data comparison
- **Data processing**: Automated climatology computation and data standardization workflows

NEMOCheck supports multiple NEMO configurations and provides a reproducible environment with locked dependencies for consistent results across Linux and macOS platforms. The modular design allows users to easily incorporate new model configurations and extend the analysis capabilities for their specific validation needs.

## Model Configurations

For detailed information about the different NEMO model configurations supported by this framework, please refer to our comprehensive documentation:

📄 **[Model Configurations Guide](docs/tests_orca05.pdf)**

This document provides:
- Overview of supported NEMO configurations
- Configuration-specific parameters and settings

## Quick Start

```bash
git clone git@github.com:gantz-thomas-gantz/NEMOCheck.git
cd NEMOCheck

./env/install_env.sh

jupyter-lab
````

* Run any notebook as usual in JupyterLab.
* Make sure to select the `nemocheck_env` kernel in JupyterLab to use the correct environment.
* You can also run interactive plotting scripts directly in the terminal, for example:

```bash
python src/utils/plot/model-model/horizontal.py
```

### About the environment setup

* The install script uses `conda-lock.yml`, which pins **all dependencies** with exact versions for reproducibility.
* For a more human-readable list of dependencies, see `environment.yml`.
* The `conda-lock.yml` file was generated from `environment.yml` by running:

```bash
conda-lock lock -f environment.yml --micromamba -p linux-64 -p osx-64
```

* This means the environment and code are intended to work on both **Linux** and **macOS** platforms.

## Data

The `data/` directory is structured as follows:

* `data/model/`: Raw NEMO output (horizontal fields and equatorial sections) and the ORCA05 mesh.
* `data/obs/`: Observational data for SST, SSS, MLD, and EKE.
* `data/processed/`: Preprocessed data ready for analysis.

### Data Processing

Processed data is generated via one of the following methods:

* **Climatology computation bash script**
  Script: `data/compute_climatology.sh`
  Output: `data/processed/nemo<00>_clim_2011_2022.nc`
  Usage: Horizontal model–model comparison: : `src/interactive/plotting_tool`

* **Processing notebook**
  Notebook: `data/processing.ipynb`
  Output: `data/processed/model.nc`, `data/processed/combined_observations.nc`
  Usage:

  * Horizontal model–obs comparison: `src/interactive/plotting_tool`
  * Error profiles for model–obs: `src/interactive_plotting_tool/`
  * Non-interactive SST diagnostics: `src/non-interactive/sst.ipynb`

* **Model normalization function**
  Function: `normalize_model_file()` in `src/utils/data/general.py`
  Output: `data/processed/nemo<00>.nc`
  Usage:

  * `src/quantitative/eke.ipynb`
  * `src/quantitative/zonal_sst.ipynb`

Further details on the data pipeline are documented at the top of `data/processing.ipynb`.

### File Types

* `model.nc` is derived from the `00` configuration and uses an MLD criterion that matches the observational dataset.
* `nemo<00>.nc` contains SST, SSS, and MLD with an internally consistent MLD definition (but not necessarily consistent with observations) and is interpolated to a regular 1° grid.
* `nemo<00>_clim_2011_2022.nc` retains all model variables on the native grid for inter-model comparisons.

### Adding a New Configuration

To include a new NEMO configuration in the analysis:

1. Open the last cell of `data/processing/processing.ipynb`, update the `cfgs` list, and re-run the cell.
2. Open `data/compute_climatology.sh`, update the `cfgs` list, and re-run the script.

## References

**Gurvan Madec (2024).** *NEMO Ocean Engine Reference Manual* (v5.0). Zenodo. https://doi.org/10.5281/zenodo.14515373


### How to Cite This Framework

If you use NEMOCheck please consider citing this repository:

```
@software{nemocheck,
  author = {Thomas Gantz},
  title = {NEMOCheck: Validation and Analysis Framework for NEMO Ocean Model},
  url = {https://github.com/gantz-thomas-gantz/NEMOCheck},
  year = {2025}
}
```




