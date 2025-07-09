# NEMOCheck

## Abstract

## Quick Start

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




