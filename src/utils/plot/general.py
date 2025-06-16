import os, socket
from typing import List
import xarray as xr
import pandas as pd


def get_vars(path: str) -> List[str]:
    """
    Retrieve all variable names from a NetCDF file.

    Args:
        path (str): Path to the NetCDF file.

    Returns:
        List[str]: List of variable names, or an empty list if reading fails.
    """
    try:
        ds = xr.open_dataset(path)
        vars_ = list(ds.data_vars)
        ds.close()
        return vars_
    except Exception:
        # Return empty list if file cannot be read or is corrupted
        return []


def get_common_vars(path1: str, path2: str) -> List[str]:
    """
    Identify variable names common to two NetCDF datasets.

    Args:
        path1 (str): Path to the first NetCDF file.
        path2 (str): Path to the second NetCDF file.

    Returns:
        List[str]: Alphabetically sorted list of common variable names.
    """
    vars1 = set(get_vars(path1))
    vars2 = set(get_vars(path2))
    return sorted(list(vars1 & vars2))


def find_free_port(start_port: int = 8050, max_tries: int = 100) -> int:
    """
    Find a free TCP port on localhost starting from a given port.

    This function attempts to bind a socket to consecutive port numbers 
    starting from `start_port` up to `start_port + max_tries - 1`.
    If a free port is found, it is returned.
    If no free port is found within the given range, a RuntimeError is raised.

    Args:
        start_port (int): Port number to start searching from. Default is 8050.
        max_tries (int): Maximum number of ports to try. Default is 100.

    Returns:
        int: A free TCP port on localhost.

    Raises:
        RuntimeError: If no free port is found after `max_tries` attempts.
    """
    port = start_port
    for _ in range(max_tries):
        # Create a new socket using IPv4 and TCP
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                # Attempt to bind the socket to localhost and the current port
                s.bind(("127.0.0.1", port))
                # Binding succeeded: port is available
                return port
            except OSError:
                # Binding failed: port is likely in use, try the next one
                port += 1

    # All attempts failed: no available port found in the specified range
    raise RuntimeError("No free port found")


def compute_seasonal_mean(
    path: str,
    time_dim: str = "time",
    year_range: tuple[int, int] = None
) -> xr.Dataset:
    """
    Compute seasonal means (DJF, MAM, JJA, SON) from a NetCDF file,
    allowing specification of time dimension and year range. Only full
    seasons are included (e.g., DJF requires Dec of previous year + Jan/Feb).

    Parameters
    ----------
    path : str
        Path to the NetCDF file.
    time_dim : str, optional
        Name of the time dimension in the dataset. Default is "time_counter".
    year_range : tuple of int, optional
        (start_year, end_year) range to consider for seasonal means.
        Both ends are inclusive. If None, all available years are used.

    Returns
    -------
    xr.Dataset
        Dataset with mean values for each complete season.
    """

    ds = xr.open_dataset(path)
    time = pd.to_datetime(ds[time_dim].values)

    # Assign datetime index
    ds = ds.assign_coords({time_dim: time})

    # Create a DataFrame to work with dates and filter for full seasons
    df = pd.DataFrame({time_dim: time})
    df["year"] = df[time_dim].dt.year
    df["month"] = df[time_dim].dt.month

    # Define season labels
    def get_season_label(row):
        m = row["month"]
        if m in [12, 1, 2]:
            return "DJF"
        elif m in [3, 4, 5]:
            return "MAM"
        elif m in [6, 7, 8]:
            return "JJA"
        else:
            return "SON"

    df["season"] = df.apply(get_season_label, axis=1)

    # Adjust DJF year (DJF 2022 = Dec 2021 + Jan/Feb 2022)
    df["season_year"] = df["year"]
    df.loc[df["month"] == 12, "season_year"] += 1

    # Keep only complete seasons
    valid_seasons = (
        df.groupby(["season", "season_year"])
        .filter(lambda g: len(g) == 3)
        .drop_duplicates(subset=["season", "season_year"])
    )

    # Filter by year range if specified
    if year_range is not None:
        start_year, end_year = year_range
        valid_seasons = valid_seasons[
            (valid_seasons["season_year"] >= start_year) &
            (valid_seasons["season_year"] <= end_year)
        ]

    # Build a mask to keep only valid times
    valid_times = valid_seasons[time_dim].unique()
    ds = ds.sel({time_dim: ds[time_dim].isin(valid_times)})

    # Recreate season labels for filtered dataset
    def assign_season(time_index):
        seasons = []
        for t in time_index:
            m = pd.Timestamp(t).month
            if m in [12, 1, 2]:
                seasons.append("DJF")
            elif m in [3, 4, 5]:
                seasons.append("MAM")
            elif m in [6, 7, 8]:
                seasons.append("JJA")
            else:
                seasons.append("SON")
        return xr.DataArray(seasons, dims=time_dim)

    ds = ds.assign_coords(season=assign_season(ds[time_dim].values))

    return ds.groupby("season").mean(time_dim)


