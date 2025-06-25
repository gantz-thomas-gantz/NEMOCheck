import numpy as np
import xarray as xr
import scipy.ndimage
from typing import Union
import os
import contextlib
import regionmask

def fill_coastal_points_in_time(
    data: Union[xr.DataArray, xr.Dataset],
    max_iterations: int,
    time_dim: str = "time"
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Fill coastal NaNs in each time slice using 2D convolution with periodic longitude (x-dimension).
    - Works for both DataArrays and Datasets.
    - For Datasets: applies filling to all data variables.

    Parameters
    ----------
    data : xr.DataArray or xr.Dataset
        Input data with at least two spatial dimensions and a time dimension.
    max_iterations : int
        Number of filling iterations per time slice.
    time_dim : str, default "time"
        Name of the time dimension.

    Returns
    -------
    xr.DataArray or xr.Dataset
        Data with coastal NaNs filled.
    """
    def fill_single_var(da: xr.DataArray) -> xr.DataArray:
        filled = da.copy(deep=True)
        kernel = np.array([[1,1,1], [1,0,1], [1,1,1]])
        # Detect spatial dimensions
        spatial_dims = [d for d in filled.dims if d != time_dim]
        if len(spatial_dims) != 2:
            raise ValueError(f"Expected 2 spatial dimensions, found: {spatial_dims}")
        y_dim, x_dim = spatial_dims

        for t in filled[time_dim]:
            slice2d = filled.sel({time_dim: t}).copy()
            arr = slice2d.values

            for _ in range(max_iterations):
                nan_mask = np.isnan(arr)
                # Pad latitude (y) manually with zeros
                padded_mask = np.pad(~nan_mask, ((1,1), (0,0)), mode="constant", constant_values=False)
                padded_data = np.pad(np.nan_to_num(arr), ((1,1), (0,0)), mode="constant", constant_values=0)

                neighbor_count = scipy.ndimage.convolve(
                    padded_mask.astype(int),
                    kernel,
                    mode="wrap"
                )[1:-1, :]  # remove padded rows

                neighbor_sum = scipy.ndimage.convolve(
                    padded_data,
                    kernel,
                    mode="wrap"
                )[1:-1, :]  # remove padded rows

                fill_condition = nan_mask & (neighbor_count > 0)
                if not np.any(fill_condition):
                    break

                arr[fill_condition] = neighbor_sum[fill_condition] / neighbor_count[fill_condition]

            # Assign back using correct dims
            filled.loc[{time_dim: t}] = xr.DataArray(arr, dims=(y_dim, x_dim))

        return filled

    if isinstance(data, xr.DataArray):
        return fill_single_var(data)
    elif isinstance(data, xr.Dataset):
        # Apply to all data variables; preserve coordinates and attributes
        filled_vars = {vn: fill_single_var(data[vn]) for vn in data.data_vars}
        return xr.Dataset(filled_vars, coords=data.coords, attrs=data.attrs)
    else:
        raise TypeError("Input must be an xarray DataArray or Dataset")

target_grid = xr.Dataset({
    'lon': (['lon'], np.arange(0.5, 360, 1)),      
    'lat': (['lat'], np.arange(-89.5, 89.5 + 1))   
})

def normalize_model_file(
    input_path: str,
    mesh_path: str,
    output_path: str,
    weights_path: str
) -> None:
    """
    Normalizes, regrids, and processes ocean model data for climatology analysis.
    
    Steps:
        1. Reads model and mesh files.
        2. Subsets and renames variables, computes monthly climatology.
        3. Applies land/ocean masking.
        4. Regrids model data to a global 1x1 deg target grid.
        5. Applies ocean-only mask.
        6. Sets CF-compliant metadata.
        7. Saves output as NetCDF.

    Args:
        input_path (str): Path to the input model NetCDF file.
        mesh_path (str): Path to the model mesh (domain config) NetCDF file.
        output_path (str): Path where processed NetCDF will be written.
        weights_path (str): Path to store/read the regridding weights file.

    Returns:
        None

    Raises:
        IOError: If input files cannot be read or output cannot be written.
        KeyError: If expected variables are missing from input files.
    """
    # Prepare target regrid grid
    target_grid = xr.Dataset({
        'lon': (['lon'], np.arange(0.5, 360, 1)),      
        'lat': (['lat'], np.arange(-89.5, 90.5, 1))    # 90.5 (not 89.5+1) to include 89.5
    })

    # Compute land/ocean mask for target grid
    land_mask = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(target_grid)
    ocean_mask = land_mask.isnull()

    # Read model and mesh files
    model = xr.open_dataset(input_path)
    mesh = xr.open_dataset(mesh_path)

    # Select and rename variables, and restrict to 2012-2022 time period
    model = model[["tos", "sos", "mldr10_1"]].rename({
        "tos": "sst",
        "sos": "sss",
        "mldr10_1": "mld"
    }).sel(time_counter=slice("2012", "2022"))

    # Compute monthly climatology (average over years by month)
    model = model.groupby("time_counter.month").mean(dim="time_counter")

    # Rename spatial coordinates and assign mesh coordinates
    model = model.rename({
        'nav_lon': 'lon',
        'nav_lat': 'lat',
    })
    model = model.assign_coords(lon=mesh['glamt'], lat=mesh['gphit'])

    # Apply land mask from mesh bathymetry (mask = True where bathymetry is zero)
    model_mask = mesh['bathy_metry'] == 0
    model = model.where(~model_mask)

    # Fill coastal points in time (e.g., nearest-neighbor fill over 20 gridpoints per month)
    model = fill_coastal_points_in_time(model, 20, "month")

    import xesmf as xe
    # Silence stdout during regridder construction (xesmf can be noisy)
    with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
        regridder = xe.Regridder(
            model, target_grid,
            method='bilinear',
            filename=weights_path,
            reuse_weights=True,
            ignore_degenerate=True,
            periodic=True
        )

    # Regrid model data to target 1x1 grid
    model = regridder(model)

    # Ensure longitude is in [0, 360), sort by lon/lat, transpose axes
    model['lon'] = (model['lon'] + 360) % 360
    model = model.sortby('lon')
    model = model.sortby('lat')
    model = model.transpose('month', 'lat', 'lon')

    # Apply ocean-only mask
    model = model.where(ocean_mask)

    # Set CF-compliant coordinate metadata
    model.coords['lon'].attrs.update({
        'long_name': 'Longitude',
        'units': 'degrees_east',
        'standard_name': 'longitude'
    })
    model.coords['lat'].attrs.update({
        'long_name': 'Latitude',
        'units': 'degrees_north',
        'standard_name': 'latitude'
    })

    # Set variable attributes for output clarity
    model['sst'].attrs.update({
        'long_name': 'Monthly Mean of Sea Surface Temperature',
        'units': 'degC',
        'standard_name': 'sea_surface_temperature'
    })
    model['sss'].attrs.update({
        'long_name': 'Monthly Mean of Sea Surface Salinity',
        'units': 'psu',
        'standard_name': 'sea_surface_salinity'
    })
    model['mld'].attrs.update({
        'long_name': 'Monthly Mean of Mixed Layer Depth',
        'units': 'm',
        'standard_name': 'ocean_mixed_layer_thickness'
    })

    # Write processed model to NetCDF file
    model.to_netcdf(output_path)



    