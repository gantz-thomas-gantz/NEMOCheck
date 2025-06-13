import os, socket
from typing import List
import xarray as xr


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

