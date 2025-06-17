import numpy as np
import xarray as xr
import scipy.ndimage
from typing import Union

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
    