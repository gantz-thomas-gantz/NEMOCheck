# model_model/vertical.py
# Goal: Interactive Dash app for visualizing and comparing vertical (longitude-depth) sections between two model datasets.

# ---- Imports ----
import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd
from dash import Dash, dcc, html, Input, Output, State, callback_context
from dash.dependencies import ALL
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from utils.plot.general import get_vars, get_common_vars, find_free_port

# ---- Path configuration ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '..', '..', '..', 'data', 'model')
MESH_PATH = os.path.join(DATA_DIR, 'orca05l75_domain_cfg_nemov5_10m.nc')

# ---- Load mesh data ----
y = 249  # Chosen latitude index for section (e.g., equator)
mesh = xr.open_dataset(MESH_PATH)
nx = mesh.sizes["x"]
nlev = mesh.sizes["nav_lev"]
depths = np.cumsum(mesh.e3t_1d.values)  # 1D array of depth cell thicknesses
files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith("_Eq.nc")])
default_files = ["nemo00_1m_201001_202212_Eq.nc", "nemo01_1m_201001_202212_Eq.nc"]

SEASON_ORDER = ['DJF', 'MAM', 'JJA', 'SON']

ALLOWED_VARS = {
    "uo": "zonal current (uo, m/s)",
    "to": "temperature (to, °C)",
    "so": "salinity (so, PSU)"
}
ALLOWED_VARS_LIST = list(ALLOWED_VARS.keys())

def compute_seasonal_mean(
    path: str,
    time_dim: str = 'time_counter',
    years: list = None
) -> xr.Dataset:
    """
    Compute the seasonal mean for a dataset, grouping by standard seasons and optionally limiting to specific years.

    Parameters
    ----------
    path : str
        Path to the NetCDF file.
    time_dim : str, optional
        Name of the time dimension (default is 'time_counter').
    years : list, optional
        Years to include in averaging. If None, includes all years.

    Returns
    -------
    xr.Dataset
        Dataset averaged by season, filtered for full seasons only.
    """
    ds = xr.open_dataset(path)
    time_values = pd.to_datetime(ds[time_dim].values)
    seasons = []
    years_ = []
    months = []
    # Classify each time step by season and season-year
    for t in time_values:
        month = t.month
        year = t.year
        # Assign DJF's December to following year
        if month == 12:
            season = "DJF"
            season_year = year + 1
        elif month in [1, 2]:
            season = "DJF"
            season_year = year
        elif month in [3, 4, 5]:
            season = "MAM"
            season_year = year
        elif month in [6, 7, 8]:
            season = "JJA"
            season_year = year
        else:
            season = "SON"
            season_year = year
        seasons.append(season)
        years_.append(season_year)
        months.append(month)
    seasons = np.array(seasons)
    years_ = np.array(years_)
    ds = ds.assign_coords(
        season=(time_dim, seasons),
        season_year=(time_dim, years_)
    )
    all_years = np.unique(years_)
    # Determine valid years (exclude edge years with incomplete seasons)
    if years is not None:
        valid_years = np.array([y for y in all_years if y in years])
    else:
        valid_years = all_years
    full_season_mask = np.isin(years_, valid_years)
    # Remove DJF entries for edge years where December/January/February are missing
    for yr in [all_years[0], all_years[-1]]:
        mask = (seasons == "DJF") & (years_ == yr)
        full_season_mask[mask] = False
    ds_full = ds.isel({time_dim: full_season_mask})
    return ds_full.groupby("season").mean(time_dim)

def slider_marks_new(nlev: int, depths: np.ndarray) -> dict:
    """
    Create marks for the depth slider.

    Parameters
    ----------
    nlev : int
        Number of vertical levels.
    depths : np.ndarray
        Array of cumulative depths.

    Returns
    -------
    dict
        Dict of slider marks keyed by integer depth level.
    """
    marks = {}
    marks[1] = str(1)
    marks[nlev] = str(nlev)
    # Add a few evenly-spaced intermediate marks for context
    if nlev > 2:
        for i in np.linspace(1, nlev, num=5, dtype=int):
            i = int(i)
            if i in marks:
                continue
            marks[i] = str(i)
    marks = {int(k): v for k, v in marks.items()}
    marks = {k: marks[k] for k in sorted(marks.keys())}
    return marks

def range_slider_year_marks(years, max_marks=5) -> dict:
    """
    Create marks for the year range slider.

    Parameters
    ----------
    years : list or range
        Years included.
    max_marks : int
        Maximum number of marks to show.

    Returns
    -------
    dict
        Dict of slider marks keyed by year.
    """
    if isinstance(years, range):
        years = list(years)
    n_years = len(years)
    marks = {int(years[0]): str(years[0])}
    if n_years > 1:
        marks[int(years[-1])] = str(years[-1])
    # If only a few years, add all as marks; else, space them out
    if n_years <= max_marks:
        for y in years[1:-1]:
            marks[int(y)] = str(y)
    else:
        step = (n_years - 1) // (max_marks - 1)
        for j in range(1, max_marks - 1):
            idx = j * step
            if idx < n_years - 1:
                y = years[idx]
                marks[int(y)] = str(y)
    marks = {k: marks[k] for k in sorted(marks.keys())}
    return marks

def wrap180(lon: float) -> float:
    """
    Convert a longitude to the [-180, 180] range.

    Parameters
    ----------
    lon : float
        Input longitude.

    Returns
    -------
    float
        Longitude in [-180, 180].
    """
    return ((lon + 180) % 360) - 180

"""
Dash application for interactive vertical (longitude-depth) model-model difference visualization.

- Lets the user choose and compare two processed model files as vertical sections for selected variables.
- Supports interactive selection of variable, depth, years, and longitude center.
- Visualizes differences as a grid of PNG images (matplotlib), with zoom modal functionality.
- Designed for coder-style clarity: includes docstrings, type hints, and inline comments.
- Data directory and file conventions are fixed; see README for requirements.

Dependencies and definitions required elsewhere in the project:
    - get_vars, get_common_vars: Utility functions for variable selection.
    - find_free_port: Function to find an available port for local hosting.
    - DATA_DIR structure and processed NetCDF files as described in README.
"""
external_stylesheets = [dbc.themes.FLATLY]
app = Dash(__name__, external_stylesheets=external_stylesheets)

def get_year_range_from_netcdf(path: str, time_dim: str) -> tuple:
    """
    Extract the available year range from a NetCDF file.

    Parameters
    ----------
    path : str
        Path to NetCDF file.
    time_dim : str
        Name of the time dimension.

    Returns
    -------
    tuple
        (first_year, last_year)
    """
    ds = xr.open_dataset(path)
    time_vals = pd.to_datetime(ds[time_dim].values)
    years = pd.Series(time_vals).dt.year.unique()
    return int(years.min()), int(years.max())

default_time_dim = "time_counter"
default_first_year, default_last_year = get_year_range_from_netcdf(
    os.path.join(DATA_DIR, default_files[0]), default_time_dim
)

DEFAULT_VAR = "to"
DEFAULT_CENTER = -160  # Pacific
DEFAULT_YEAR_RANGE = [default_first_year+1, default_last_year]

def filtered_variables_with_explanation(variables: list) -> list:
    """
    Return filtered list of variable dropdown options with explanations.

    Parameters
    ----------
    variables : list
        List of variable names.

    Returns
    -------
    list
        List of dicts for use as dropdown options.
    """
    return [
        {"label": ALLOWED_VARS[v], "value": v}
        for v in variables if v in ALLOWED_VARS
    ]

@app.callback(
    Output('center-input', 'value'),
    Input('center-preset-dropdown', 'value'),
    State('center-input', 'value'),
)
def update_center_input(preset: str, current: float) -> float:
    """
    Callback to update the custom center longitude input.

    Returns
    -------
    float
        The new center value.
    """
    # Preset dropdown controls the longitude center input unless user chooses 'custom'
    if preset == 'custom':
        return current
    else:
        return preset

@app.callback(
    Output('variable-dropdown', 'options'),
    Output('variable-dropdown', 'value'),
    Input('model-dropdown', 'value'),
)
def update_variable_options(model_files: list) -> tuple:
    """
    Callback to update the variable dropdown based on selected model files.

    Returns
    -------
    tuple
        (options for dropdown, default value)
    """
    vars_to_show = []
    # Only show variables present in both files
    if model_files and len(model_files) == 2:
        path1 = os.path.join(DATA_DIR, model_files[0])
        path2 = os.path.join(DATA_DIR, model_files[1])
        all_vars = get_common_vars(path1, path2)
        vars_to_show = [v for v in all_vars if v in ALLOWED_VARS]
    elif model_files and len(model_files) == 1:
        path = os.path.join(DATA_DIR, model_files[0])
        all_vars = get_vars(path)
        vars_to_show = [v for v in all_vars if v in ALLOWED_VARS]
    else:
        vars_to_show = []
    options = filtered_variables_with_explanation(vars_to_show)
    default_val = []
    # Prefer temperature as default, else first available
    if 'to' in vars_to_show:
        default_val = ['to']
    elif vars_to_show:
        default_val = [vars_to_show[0]]
    return options, default_val

@app.callback(
    Output('depth-slider-value', 'children'),
    Input('depth-slider', 'value')
)
def update_depth_text(depth_level: int) -> html.Div:
    """
    Callback to update the depth slider label text.

    Returns
    -------
    html.Div
        Div showing the current depth level and depth in meters.
    """
    depth = int(round(depths[depth_level-1]))
    return html.Div([
        f"Current depth level: {depth_level}",
        html.Br(),
        html.Span(f"({depth} meters)", style={"font-size": "90%", "color": "#555"})
    ])

@app.callback(
    Output('year-range-slider-value', 'children'),
    Input('year-range-slider', 'value')
)
def update_year_range_text(years: list) -> str:
    """
    Callback to update the display for the selected year range.

    Returns
    -------
    str
        Text representation of the year range.
    """
    y0, y1 = years
    return f"Selected years: {y0}–{y1}"

@app.callback(
    Output('note-collapse', 'is_open'),
    Input('note-toggle-btn', 'n_clicks'),
    State('note-collapse', 'is_open')
)
def toggle_note(n: int, is_open: bool) -> bool:
    """
    Callback to show/hide the season note.

    Returns
    -------
    bool
        New open state.
    """
    if n:
        return not is_open
    return is_open

def get_grid_info(var_name: str, ds: xr.Dataset, mesh: xr.Dataset, y: int) -> dict:
    """
    Get grid info and depth for a variable and dataset at a given latitude index.

    Parameters
    ----------
    var_name : str
        Variable name.
    ds : xr.Dataset
        Dataset.
    mesh : xr.Dataset
        Mesh dataset.
    y : int
        y-index.

    Returns
    -------
    dict
        Dict with grid info and sorted depths.
    """
    dims = ds[var_name].dims
    nlev = mesh.sizes["nav_lev"]
    nx = mesh.sizes["x"]
    bottom_levels = mesh.bottom_level.sel(y=y).values
    e3_1d = mesh.e3t_1d.values
    # Determine grid type (T or U) based on dims
    if "y_grid_T" in dims and "x_grid_T" in dims:
        grid_type = "T"
        y_var, x_var = "y_grid_T", "x_grid_T"
        nav_lon = mesh.glamf.sel(y=y).values
        e3_0 = mesh.e3t_0.sel(y=y).values
    elif "y_grid_U" in dims and "x_grid_U" in dims:
        grid_type = "U"
        y_var, x_var = "y_grid_U", "x_grid_U"
        nav_lon = mesh.glamt.sel(y=y).values
        e3_0 = mesh.e3u_0.sel(y=y).values 
    else:
        raise ValueError(f"Unknown grid for variable {var_name} with dims {dims}")
    # valid_mask: True where water column exists (below bottom level is False)
    valid_mask = np.zeros((nlev, nx), dtype=bool)
    for x in range(nx):
        level = bottom_levels[x]
        valid_mask[:level, x] = True
    # Compute cumulative depth (negative down, as per ocean convention)
    depth_2d = np.zeros((nlev, nx))
    for x in range(nx):
        level = bottom_levels[x]
        depth_2d[:, x] = e3_1d
        depth_2d[level-1, x] = e3_0[level-1, x]
    depth_cumsum_2d = -np.cumsum(depth_2d, axis=0)
    # Sort by longitude for plotting
    depth_cumsum_2d_sorted = depth_cumsum_2d[:, np.argsort(nav_lon)]
    depth_cumsum_2d_sorted_padded = np.pad(depth_cumsum_2d_sorted, ((1, 0), (0, 0)), mode='constant', constant_values=0)
    return dict(
        grid_type=grid_type,
        y_var=y_var, x_var=x_var,
        nav_lon=nav_lon,
        valid_mask=valid_mask,
        depth_cumsum_2d_sorted=depth_cumsum_2d_sorted,
        depth_cumsum_2d_sorted_padded=depth_cumsum_2d_sorted_padded,
        sort_idx=np.argsort(nav_lon),
        nlev=nlev,
        nx=nx
    )

# ---- Dash Layout ----
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("Model-Model Difference Viewer (Vertical)"), width="auto"),
    ], className="mb-2"),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Select two model files:"),
                            dcc.Dropdown(
                                id='model-dropdown',
                                options=[{'label': fn, 'value': fn} for fn in files],
                                value=default_files,
                                multi=True,
                                style={'margin-bottom': '8px'},
                            ),
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Variable(s):"),
                            dcc.Dropdown(
                                id='variable-dropdown',
                                options=filtered_variables_with_explanation(ALLOWED_VARS_LIST),
                                value=[DEFAULT_VAR],
                                multi=True,
                                style={'margin-bottom': '8px'},
                            ),
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Depth level:"),
                            dcc.Slider(
                                id='depth-slider',
                                min=1,
                                max=nlev,
                                value=30,
                                marks=slider_marks_new(nlev, depths),
                                step=1,
                                tooltip={'always_visible': False}
                            ),
                            html.Div(id='depth-slider-value', style={'marginBottom': '8px'}),
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Time dimension:"),
                            dcc.Input(
                                id='time-dim-input',
                                type='text',
                                value=default_time_dim,
                                style={'width': '100%'}
                            ),
                        ], width=4),
                        dbc.Col([
                            dbc.Label("Year range (full seasons):"),
                            dcc.RangeSlider(
                                id='year-range-slider',
                                min=default_first_year,
                                max=default_last_year,
                                value=DEFAULT_YEAR_RANGE,
                                marks=range_slider_year_marks(range(default_first_year, default_last_year+1)),
                                step=1,
                                tooltip={'always_visible': False}
                            ),
                            html.Div(id='year-range-slider-value', style={'marginBottom': '8px'}),
                        ], width=8),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Center longitude:"),
                            dcc.Dropdown(
                                id='center-preset-dropdown',
                                options=[
                                    {'label': 'Atlantic Ocean (-30°)', 'value': -30},
                                    {'label': 'Pacific Ocean (-160°)', 'value': -160},
                                    {'label': 'Indian Ocean (80°)', 'value': 80},
                                    {'label': 'Custom', 'value': 'custom'}
                                ],
                                value=DEFAULT_CENTER,
                                clearable=False,
                                style={'width': '100%', 'margin-bottom': '8px'}
                            ),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Custom center:"),
                            dcc.Input(
                                id='center-input',
                                type='number',
                                value=DEFAULT_CENTER,
                                min=-180,
                                max=180,
                                step=1,
                                style={'width': '100%'}
                            ),
                        ], width=6),
                    ]),
                    dbc.Row([
                        dbc.Col(
                            html.Button("Plot", id="plot-btn", n_clicks=0, className="btn btn-primary", style={'marginTop': '8px'}),
                            width='auto'),
                    ], className="mt-2"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Collapse(
                                dbc.Alert(
                                    [
                                        html.Strong("Note: "),
                                        "December of a year (e.g. December 2010) counts into DJF of the following year (e.g. DJF 2011). Only full seasons are taken into account (e.g. January 2010 will never be taken into account)."
                                    ],
                                    color="warning",
                                    className="py-2 mb-2",
                                    style={"fontWeight": "bold", "fontSize": "1em"}
                                ),
                                id="note-collapse",
                                is_open=False
                            ),
                            dbc.Button(
                                "Show/Hide Note",
                                id="note-toggle-btn",
                                color="secondary",
                                outline=True,
                                size="sm",
                                style={"marginBottom": "1rem"}
                            ),
                        ])
                    ]),
                ])
            ], className="mb-2"),
        ], width=4),
        dbc.Col([
            dcc.Loading(
                id="plot-loading",
                type="default",
                children=html.Div(id='plot-container')
            ),
            dcc.Store(id='plot-imgs-store', storage_type='memory'),
            dcc.Store(id='plot-config-store', storage_type='memory'),
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Zoomed-in Plot")),
                    dbc.ModalBody(
                        html.Div(id="modal-plot-content")
                    ),
                    dbc.ModalFooter(
                        dbc.Button("Close", id="close-modal", className="ms-auto", n_clicks=0)
                    ),
                ],
                id="plot-modal",
                size="xl",
                is_open=False,
                scrollable=True,
            ),
        ], width=8),
    ])
], fluid=True)

# --- Remaining callbacks and logic unchanged ---

@app.callback(
    Output('plot-container', 'children'),
    Output('plot-imgs-store', 'data'),
    Output('plot-config-store', 'data'),
    Input('plot-btn', 'n_clicks'),
    State('model-dropdown', 'value'),
    State('variable-dropdown', 'value'),
    State('depth-slider', 'value'),
    State('center-input', 'value'),
    State('time-dim-input', 'value'),
    State('year-range-slider', 'value'),
    prevent_initial_call=True,
)
def make_plot_grid(
    n_clicks: int,
    model_files: list,
    var_list: list,
    depth_level: int,
    center: float,
    time_dim: str,
    year_range: list
):
    """
    Callback to build the grid of PNG section plots for seasonal means and differences.

    Returns
    -------
    Tuple
        (HTML grid Div, image store, plot config)
    """
    import json
    # Validate input selections
    if not model_files or not var_list or len(model_files) != 2:
        raise PreventUpdate
    if isinstance(var_list, str):
        var_list = [var_list]
    var_list = [v for v in var_list if v in ALLOWED_VARS]
    if not var_list:
        return html.Div("No valid variable selected."), None, None
    path1 = os.path.join(DATA_DIR, model_files[0])
    path2 = os.path.join(DATA_DIR, model_files[1])
    years = list(range(year_range[0], year_range[1]+1))
    # Compute seasonal means for both datasets
    ds1_seasonal = compute_seasonal_mean(path1, time_dim=time_dim, years=years)
    ds2_seasonal = compute_seasonal_mean(path2, time_dim=time_dim, years=years)
    seasons = SEASON_ORDER
    n_vars = len(var_list)
    n_cols = 1 + n_vars * 3  # 1 for season label, 3 per variable (model1, model2, diff)
    grid_template_columns = f"60px {' '.join(['170px'] * (n_vars * 3))}"

    gridinfos = {}
    global_vmin = {}
    global_vmax = {}
    diff_absmax = {}
    # Precompute vmin/vmax for each variable for consistent color scaling
    for var in var_list:
        gridinfos[var] = get_grid_info(var, ds1_seasonal, mesh, y)
        model1_vals = []
        model2_vals = []
        diff_vals = []
        gridinfo = gridinfos[var]
        valid_mask = gridinfo["valid_mask"]
        y_var = gridinfo["y_var"]
        # Gather values for each season
        for season in seasons:
            if season not in ds1_seasonal['season'].values or season not in ds2_seasonal['season'].values:
                continue
            T1 = ds1_seasonal[var].sel(season=season).isel({y_var: 0}).values
            T2 = ds2_seasonal[var].sel(season=season).isel({y_var: 0}).values
            T1 = np.where(valid_mask, T1, np.nan)
            T2 = np.where(valid_mask, T2, np.nan)
            T1_slice = T1[:depth_level, :]
            T2_slice = T2[:depth_level, :]
            model1_vals.append(T1_slice)
            model2_vals.append(T2_slice)
            diff_vals.append(T1_slice - T2_slice)
        if not model1_vals or not model2_vals:
            # If no valid data, set dummy range
            global_vmin[var] = 0
            global_vmax[var] = 1
            diff_absmax[var] = 1
        else:
            model1_all = np.concatenate([a.flatten() for a in model1_vals])
            model2_all = np.concatenate([a.flatten() for a in model2_vals])
            diff_all = np.concatenate([a.flatten() for a in diff_vals])
            global_vmin[var] = np.nanmin([model1_all, model2_all])
            global_vmax[var] = np.nanmax([model1_all, model2_all])
            diff_absmax[var] = np.nanmax(np.abs(diff_all))
    plot_config = dict(
        model_files=model_files,
        var_list=var_list,
        depth_level=depth_level,
        center=center,
        time_dim=time_dim,
        year_range=year_range,
        seasons=seasons,
        global_vmin=global_vmin,
        global_vmax=global_vmax,
        diff_absmax=diff_absmax,
    )
    plot_imgs = []
    plot_img_params = []
    # Loop through seasons, variables, and plot types (model1, model2, diff)
    for season_idx, season in enumerate(seasons):
        for var_idx, var in enumerate(var_list):
            for type_idx, label in enumerate([model_files[0], model_files[1], "Δ"]):
                gridinfo = gridinfos[var]
                nav_lon = gridinfo["nav_lon"]
                sort_idx = gridinfo["sort_idx"]
                valid_mask = gridinfo["valid_mask"]
                depth_cumsum_2d_sorted = gridinfo["depth_cumsum_2d_sorted"]
                depth_cumsum_2d_sorted_padded = gridinfo["depth_cumsum_2d_sorted_padded"]
                y_var = gridinfo["y_var"]
                x_var = gridinfo["x_var"]
                if season not in ds1_seasonal['season'].values or season not in ds2_seasonal['season'].values:
                    plot_imgs.append(None)
                    plot_img_params.append(None)
                    continue
                T1 = ds1_seasonal[var].sel(season=season).isel({y_var: 0}).values
                T2 = ds2_seasonal[var].sel(season=season).isel({y_var: 0}).values
                T1 = np.where(valid_mask, T1, np.nan)
                T2 = np.where(valid_mask, T2, np.nan)
                T1_sorted = T1[:depth_level, sort_idx]
                T2_sorted = T2[:depth_level, sort_idx]
                diff_sorted = T1_sorted - T2_sorted
                # Re-center longitude for Pacific/Atlantic/Indian etc.
                lon_wrapped = ((nav_lon[sort_idx] - center + 180) % 360) - 180 + center
                sort_idx2 = np.argsort(lon_wrapped)
                nav_lon_centered = lon_wrapped[sort_idx2]
                T1_centered = T1_sorted[:, sort_idx2]
                T2_centered = T2_sorted[:, sort_idx2]
                diff_centered = diff_sorted[:, sort_idx2]
                depth_cumsum_centered = depth_cumsum_2d_sorted_padded[:, sort_idx2]
                cmap = plt.get_cmap('viridis').copy()
                cmap.set_bad(color='white')
                cmap_diff = plt.get_cmap('seismic').copy()
                cmap_diff.set_bad(color='white')
                fig, ax = plt.subplots(figsize=(3, 2.5))
                # Hide axes for compact grid
                ax.set_xticks([])
                ax.set_yticks([])
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                # Plot based on type (model1, model2, or difference)
                if type_idx == 0:
                    im = ax.pcolormesh(
                        nav_lon_centered,
                        depth_cumsum_centered[:depth_level+1, :],
                        T1_centered[:, 1:],
                        shading='auto',
                        cmap=cmap,
                        vmin=global_vmin[var], vmax=global_vmax[var]
                    )
                elif type_idx == 1:
                    im = ax.pcolormesh(
                        nav_lon_centered,
                        depth_cumsum_centered[:depth_level+1, :],
                        T2_centered[:, 1:],
                        shading='auto',
                        cmap=cmap,
                        vmin=global_vmin[var], vmax=global_vmax[var]
                    )
                else:
                    im = ax.pcolormesh(
                        nav_lon_centered,
                        depth_cumsum_centered[:depth_level+1, :],
                        diff_centered[:, 1:],
                        shading='auto',
                        cmap=cmap_diff,
                        vmin=-diff_absmax[var], vmax=diff_absmax[var]
                    )
                # Set longitude ticks and labels to [-180, 180]
                xticks = np.linspace(np.nanmin(nav_lon_centered), np.nanmax(nav_lon_centered), 7)
                xticklabels = [f"{wrap180(l):.0f}°" for l in xticks]
                ax.set_xticks(xticks)
                ax.set_xticklabels(xticklabels)
                buf = io.BytesIO()
                plt.savefig(buf, format="png", dpi=100, bbox_inches='tight', pad_inches=0.02)
                plt.close(fig)
                buf.seek(0)
                img_base64 = base64.b64encode(buf.read()).decode("ascii")
                buf.close()
                plot_imgs.append(img_base64)
                plot_img_params.append(dict(
                    season_idx=season_idx,
                    var_idx=var_idx,
                    type_idx=type_idx,
                ))

    # --- GRID HEADER PART: Use CSS Grid for perfect centering ---
    # Top header row: variable names, spanning 3 columns each
    top_header_cells = [html.Div("", style={})]
    for i, var in enumerate(var_list):
        # Use gridColumn to span the 3 columns for this variable
        top_header_cells.append(html.Div(
            ALLOWED_VARS[var],
            style={
                "gridColumn": f"{2 + i*3} / {2 + (i+1)*3}",
                "textAlign": "center", "fontWeight": 600, "fontSize": "14px", "lineHeight": "18px"
            }
        ))
    while len(top_header_cells) < n_cols:
        top_header_cells.append(html.Div(""))

    # Second header row: file1, file2, Δ for each variable
    second_header_cells = [html.Div("", style={})]
    for var in var_list:
        for label in [model_files[0], model_files[1], "Δ"]:
            second_header_cells.append(html.Div(
                label,
                style={
                    "textAlign": "center", "fontSize": "9px", "fontWeight": 400,
                    "whiteSpace": "nowrap", "overflow": "hidden", "textOverflow": "ellipsis",
                    "lineHeight": "13px"
                },
                title=label
            ))

    # Build the grid rows (each as a grid row)
    html_grid = []
    img_idx = 0
    for season_idx, season in enumerate(seasons):
        row_cells = []
        # Leftmost cell: season label, vertically oriented
        row_cells.append(html.Div(
            season,
            style={'writingMode': 'vertical-rl', 'textAlign': 'center',
                   'fontWeight': 600, 'fontSize': '18px', 'padding': '10px 2px 10px 0'}
        ))
        for var_idx, var in enumerate(var_list):
            for k in range(3):
                zoom_id = {'type': 'zoom-btn', 'row': img_idx}
                if plot_imgs[img_idx]:
                    cell = html.Div([
                        html.Div([
                            html.Img(
                                src="data:image/png;base64," + plot_imgs[img_idx],
                                style={'width': '170px', 'border': '1px solid #ccc', 'borderRadius': '4px'}
                            ),
                            html.Button(
                                html.Span("\U0001F50D", style={'fontSize': '20px'}),
                                id=zoom_id,
                                n_clicks=0,
                                style={
                                    'position': 'absolute',
                                    'top': '7px',
                                    'right': '8px',
                                    'background': 'rgba(255,255,255,0.7)',
                                    'border': 'none',
                                    'borderRadius': '50%',
                                    'padding': '2px 5px',
                                    'cursor': 'pointer',
                                    'zIndex': 10,
                                }
                            )
                        ], style={'position': 'relative', 'display': 'inline-block'})
                    ], style={'display': 'inline-block', 'margin': '2px'})
                else:
                    cell = html.Div()
                row_cells.append(cell)
                img_idx += 1
        html_grid.append(html.Div(row_cells, style={
            "display": "grid",
            "gridTemplateColumns": grid_template_columns,
            "alignItems": "center",
            "marginBottom": "8px"
        }))

    # Insert the header rows at the top, using grid as well
    html_grid.insert(0, html.Div(second_header_cells, style={
        "display": "grid", "gridTemplateColumns": grid_template_columns, "marginBottom": "2px"
    }))
    html_grid.insert(0, html.Div(top_header_cells, style={
        "display": "grid", "gridTemplateColumns": grid_template_columns, "marginBottom": "2px"
    }))

    return html.Div(html_grid), dict(imgs=plot_imgs, img_params=plot_img_params), plot_config

@app.callback(
    Output('modal-plot-content', 'children'),
    Output('plot-modal', 'is_open'),
    Input({'type': 'zoom-btn', 'row': ALL}, 'n_clicks'),
    Input('close-modal', 'n_clicks'),
    State('plot-imgs-store', 'data'),
    State('plot-config-store', 'data'),
    prevent_initial_call=True,
)
def zoom_modal(
    zoom_btn_clicks: list,
    close_modal_clicks: int,
    plot_imgs_store: dict,
    plot_config: dict
):
    """
    Callback to show modal with zoomed-in matplotlib section plot.

    Returns
    -------
    Tuple
        (modal image, modal open/close bool)
    """
    import json
    ctx = callback_context

    # Only trigger the modal if an actual action occurred
    if not ctx.triggered:
        raise PreventUpdate

    triggered = ctx.triggered[0]
    triggered_id = triggered["prop_id"].split(".")[0]

    # If the close button was clicked, close the modal
    if triggered_id == "close-modal":
        return None, False

    # Otherwise, check if a zoom button was clicked
    try:
        triggered_dict = json.loads(triggered_id)
    except Exception:
        # If not a zoom button, do not update
        raise PreventUpdate

    # Only process if it's a zoom button (contains type and row)
    if (
        isinstance(triggered_dict, dict)
        and triggered_dict.get("type") == "zoom-btn"
    ):
        zoom_idx = triggered_dict.get("row")
        # Only open modal if n_clicks for that button is > 0 (ignore programmatic triggers)
        if zoom_btn_clicks[zoom_idx] <= 0:
            raise PreventUpdate
    else:
        raise PreventUpdate

    # Retrieve parameters for the image to be shown in the modal
    img_params = plot_imgs_store.get('img_params', [])[zoom_idx]
    if not img_params or not plot_config:
        raise PreventUpdate

    # Extract the plot/config parameters for this image
    season_idx = img_params['season_idx']
    var_idx = img_params['var_idx']
    type_idx = img_params['type_idx']
    model_files = plot_config['model_files']
    var_list = plot_config['var_list']
    depth_level = plot_config['depth_level']
    center = plot_config['center']
    time_dim = plot_config['time_dim']
    year_range = plot_config['year_range']
    seasons = plot_config['seasons']
    global_vmin = plot_config['global_vmin']
    global_vmax = plot_config['global_vmax']
    diff_absmax = plot_config['diff_absmax']

    # Paths to model files
    path1 = os.path.join(DATA_DIR, model_files[0])
    path2 = os.path.join(DATA_DIR, model_files[1])
    years = list(range(year_range[0], year_range[1]+1))

    # Compute up-to-date seasonal means (ensures plot is always current with selections)
    ds1_seasonal = compute_seasonal_mean(path1, time_dim=time_dim, years=years)
    ds2_seasonal = compute_seasonal_mean(path2, time_dim=time_dim, years=years)
    season = seasons[season_idx]
    var = var_list[var_idx]

    # Get grid info (longitude, sorting, valid mask, etc.) for selecting and plotting
    gridinfo = get_grid_info(var, ds1_seasonal, mesh, y)
    nav_lon = gridinfo["nav_lon"]
    sort_idx = gridinfo["sort_idx"]
    valid_mask = gridinfo["valid_mask"]
    depth_cumsum_2d_sorted = gridinfo["depth_cumsum_2d_sorted"]
    depth_cumsum_2d_sorted_padded = gridinfo["depth_cumsum_2d_sorted_padded"]
    y_var = gridinfo["y_var"]
    x_var = gridinfo["x_var"]

    # Extract the variable fields for both model datasets, select the right season and y-row
    T1 = ds1_seasonal[var].sel(season=season).isel({y_var: 0}).values
    T2 = ds2_seasonal[var].sel(season=season).isel({y_var: 0}).values
    # Mask out invalid points (land, below bottom)
    T1 = np.where(valid_mask, T1, np.nan)
    T2 = np.where(valid_mask, T2, np.nan)

    # Sort fields and depth arrays in longitude for correct display
    T1_sorted = T1[:depth_level, sort_idx]
    T2_sorted = T2[:depth_level, sort_idx]
    diff_sorted = T1_sorted - T2_sorted
    # Re-center longitude so Pacific, Atlantic, etc. are in the middle if desired
    lon_wrapped = ((nav_lon[sort_idx] - center + 180) % 360) - 180 + center
    sort_idx2 = np.argsort(lon_wrapped)
    nav_lon_centered = lon_wrapped[sort_idx2]
    T1_centered = T1_sorted[:, sort_idx2]
    T2_centered = T2_sorted[:, sort_idx2]
    diff_centered = diff_sorted[:, sort_idx2]
    depth_cumsum_centered = depth_cumsum_2d_sorted_padded[:, sort_idx2]

    # Choose colormap
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad(color='white')
    cmap_diff = plt.get_cmap('seismic').copy()
    cmap_diff.set_bad(color='white')

    # Create zoomed-in matplotlib plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the correct panel type (model1, model2, or their difference)
    if type_idx == 0:
        im = ax.pcolormesh(
            nav_lon_centered,
            depth_cumsum_centered[:depth_level+1, :],
            T1_centered[:, 1:],
            shading='auto',
            cmap=cmap,
            vmin=global_vmin[var], vmax=global_vmax[var]
        )
        title = f"{season} – {ALLOWED_VARS[var]}\n{model_files[0]}"
    elif type_idx == 1:
        im = ax.pcolormesh(
            nav_lon_centered,
            depth_cumsum_centered[:depth_level+1, :],
            T2_centered[:, 1:],
            shading='auto',
            cmap=cmap,
            vmin=global_vmin[var], vmax=global_vmax[var]
        )
        title = f"{season} – {ALLOWED_VARS[var]}\n{model_files[1]}"
    else:
        im = ax.pcolormesh(
            nav_lon_centered,
            depth_cumsum_centered[:depth_level+1, :],
            diff_centered[:, 1:],
            shading='auto',
            cmap=cmap_diff,
            vmin=-diff_absmax[var], vmax=diff_absmax[var]
        )
        title = f"{season} – {ALLOWED_VARS[var]}\n Δ {model_files[0]} - {model_files[1]}"

    # Set longitude ticks and labels to [-180, 180] for interpretability
    xticks = np.linspace(np.nanmin(nav_lon_centered), np.nanmax(nav_lon_centered), 7)
    xticklabels = [f"{wrap180(l):.0f}°" for l in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Depth (m)", fontsize=12)
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, label=ALLOWED_VARS[var] if type_idx < 2 else f"Δ {ALLOWED_VARS[var]}")
    plt.tight_layout()

    # Render to PNG and encode as base64
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("ascii")
    buf.close()

    # Return wrapped image for modal and open flag True
    modal_img = html.Img(src="data:image/png;base64," + img_base64, style={'width': '100%', 'maxWidth': '900px'})
    return modal_img, True

if __name__ == "__main__":
    port = find_free_port()
    app.run(debug=True, port=port)