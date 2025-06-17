import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import xarray as xr
import scipy.ndimage

def show_coverage_mask_model(model, mesh, time_index=0, title="Coverage Mask Model"):
    """
    Display a global coverage mask based on the presence of NaNs in the model at a given time index.

    Parameters:
    - model: xr.DataArray, 3D (time, y, x) with possible NaNs
    - mesh: dict-like, containing 'glamf' and 'gphif' arrays for longitudes and latitudes
    - time_index: int, time slice to visualize
    - title: str, title for the plot
    """
    coverage_mask = model.isel(time=time_index).notnull()
    non_coverage_mask = ~coverage_mask

    lonf = mesh['glamf'].values
    latf = mesh['gphif'].values

    fig = plt.figure(figsize=(12, 6))
    crs = ccrs.PlateCarree()
    ax = plt.subplot(1, 1, 1, projection=crs)
    ax.set_extent([-180, 180, -80, 90], crs)

    im = ax.pcolormesh(
        lonf, latf, non_coverage_mask[1:,1:],
        transform=crs,
        cmap='Greys',
        shading='auto',
        alpha=0.4
    )

    ax.coastlines(resolution='50m')
    ax.set_title(title)
    ax.gridlines(draw_labels=True)
    plt.tight_layout()
    plt.show()

def fill_coastal_points_in_time(
    data_array: xr.DataArray, 
    max_iterations: int, 
    time_dim: str = "time"
) -> xr.DataArray:
    """
    Fill coastal NaNs using 2D convolution with periodic longitude.
    Latitude is padded manually with zeros.

    Parameters:
    - data_array: xr.DataArray, shape (..., y, x), must have a time-like dimension
    - max_iterations: int, number of filling iterations
    - time_dim: str, name of the time dimension (default 'time')

    Returns:
    - xr.DataArray of same shape, coastal NaNs filled
    """
    filled = data_array.copy(deep=True)
    kernel = np.array([[1,1,1], [1,0,1], [1,1,1]])

    # Automatically detect the two spatial dims (excluding time_dim)
    spatial_dims = [d for d in filled.dims if d != time_dim]
    if len(spatial_dims) != 2:
        raise ValueError(f"Expected 2 spatial dimensions, found: {spatial_dims}")
    y_dim, x_dim = spatial_dims

    for t in filled[time_dim]:
        slice2d = filled.sel({time_dim: t}).copy()
        arr = slice2d.values

        for _ in range(max_iterations):
            nan_mask = np.isnan(arr)

            # Pad latitude (y-dim) manually with zeros
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

            fill_condition = nan_mask & (neighbor_count > 0) # coastal points
            if not np.any(fill_condition):
                break

            arr[fill_condition] = neighbor_sum[fill_condition] / neighbor_count[fill_condition]

        # assign back using the correct spatial dimension names
        filled.loc[{time_dim: t}] = xr.DataArray(arr, dims=(y_dim, x_dim))

    return filled