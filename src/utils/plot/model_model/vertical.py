import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd
from dash import Dash, dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from utils.plot.general import get_vars, get_common_vars, find_free_port

DATA_DIR = "~/NEMOCheck/data/model"
DATA_DIR = os.path.expanduser(DATA_DIR)
MESH_PATH = "~/NEMOCheck/data/model/orca05l75_domain_cfg_nemov5_10m.nc"

y = 249
mesh = xr.open_dataset(MESH_PATH)
nx = mesh.sizes["x"]
nlev = mesh.sizes["nav_lev"]
depths = np.cumsum(mesh.e3t_1d.values)

SEASON_ORDER = ['DJF', 'MAM', 'JJA', 'SON']

# Only allow these variables, with explanations
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
    ds = xr.open_dataset(path)
    time_values = pd.to_datetime(ds[time_dim].values)
    seasons = []
    years_ = []
    months = []
    for t in time_values:
        month = t.month
        year = t.year
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
    if years is not None:
        valid_years = np.array([y for y in all_years if y in years])
    else:
        valid_years = all_years
    full_season_mask = np.isin(years_, valid_years)
    for yr in [all_years[0], all_years[-1]]:
        mask = (seasons == "DJF") & (years_ == yr)
        full_season_mask[mask] = False
    ds_full = ds.isel({time_dim: full_season_mask})
    return ds_full.groupby("season").mean(time_dim)

def slider_marks_new(nlev, depths):
    # Only show the level numbers as strings in the marks
    marks = {}
    marks[1] = str(1)
    marks[nlev] = str(nlev)
    if nlev > 2:
        for i in np.linspace(1, nlev, num=5, dtype=int):
            i = int(i)
            if i in marks:
                continue
            marks[i] = str(i)
    # Ensure all keys are int (no numpy types)
    marks = {int(k): v for k, v in marks.items()}
    marks = {k: marks[k] for k in sorted(marks.keys())}
    return marks

def range_slider_year_marks(years, max_marks=5):
    if isinstance(years, range):
        years = list(years)
    n_years = len(years)
    marks = {int(years[0]): str(years[0])}
    if n_years > 1:
        marks[int(years[-1])] = str(years[-1])
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

external_stylesheets = [dbc.themes.FLATLY]
app = Dash(__name__, external_stylesheets=external_stylesheets)

files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith("_Eq.nc")])
default_files = ["nemo00_1m_201001_202212_Eq.nc", "nemo01_1m_201001_202212_Eq.nc"]

def get_year_range_from_netcdf(path, time_dim):
    ds = xr.open_dataset(path)
    time_vals = pd.to_datetime(ds[time_dim].values)
    years = pd.Series(time_vals).dt.year.unique()
    return int(years.min()), int(years.max())

default_time_dim = "time_counter"
default_first_year, default_last_year = get_year_range_from_netcdf(
    os.path.join(DATA_DIR, default_files[0]), default_time_dim
)

# Only show allowed variables, and show explanation in brackets
def filtered_variables_with_explanation(variables):
    return [
        {"label": ALLOWED_VARS[v], "value": v}
        for v in variables if v in ALLOWED_VARS
    ]

first_var = ALLOWED_VARS_LIST[0] if ALLOWED_VARS_LIST else None

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("Vertical Slice Viewer"), width="auto"),
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
                                value=[first_var] if first_var else [],
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
                                value=[default_first_year, default_last_year],
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
                                value=0,
                                clearable=False,
                                style={'width': '100%', 'margin-bottom': '8px'}
                            ),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Custom center:"),
                            dcc.Input(
                                id='center-input',
                                type='number',
                                value=0,
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
            )
        ], width=8),
    ])
], fluid=True)

@app.callback(
    Output('center-input', 'value'),
    Input('center-preset-dropdown', 'value'),
    State('center-input', 'value'),
)
def update_center_input(preset, current):
    if preset == 'custom':
        return current
    else:
        return preset

@app.callback(
    Output('variable-dropdown', 'options'),
    Output('variable-dropdown', 'value'),
    Input('model-dropdown', 'value'),
)
def update_variable_options(model_files):
    # Only show allowed variables that are present in both files
    vars_to_show = []
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
    default_val = [vars_to_show[0]] if vars_to_show else []
    return options, default_val

@app.callback(
    Output('depth-slider-value', 'children'),
    Input('depth-slider', 'value')
)
def update_depth_text(depth_level):
    depth = int(round(depths[depth_level-1]))
    # Always show level and meters, meters in a line below
    return html.Div([
        f"Current depth level: {depth_level}",
        html.Br(),
        html.Span(f"({depth} meters)", style={"font-size": "90%", "color": "#555"})
    ])

@app.callback(
    Output('year-range-slider-value', 'children'),
    Input('year-range-slider', 'value')
)
def update_year_range_text(years):
    y0, y1 = years
    return f"Selected years: {y0}–{y1}"

@app.callback(
    Output('note-collapse', 'is_open'),
    Input('note-toggle-btn', 'n_clicks'),
    State('note-collapse', 'is_open')
)
def toggle_note(n, is_open):
    if n:
        return not is_open
    return is_open

def get_grid_info(var_name, ds, mesh, y):
    dims = ds[var_name].dims
    nlev = mesh.sizes["nav_lev"]
    nx = mesh.sizes["x"]
    bottom_levels = mesh.bottom_level.sel(y=y).values
    e3_1d = mesh.e3t_1d.values
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
    valid_mask = np.zeros((nlev, nx), dtype=bool)
    for x in range(nx):
        level = bottom_levels[x]
        valid_mask[:level, x] = True
    depth_2d = np.zeros((nlev, nx))
    for x in range(nx):
        level = bottom_levels[x]
        depth_2d[:, x] = e3_1d
        depth_2d[level-1, x] = e3_0[level-1, x]
    depth_cumsum_2d = -np.cumsum(depth_2d, axis=0)
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

@app.callback(
    Output('plot-container', 'children'),
    Input('plot-btn', 'n_clicks'),
    State('model-dropdown', 'value'),
    State('variable-dropdown', 'value'),
    State('depth-slider', 'value'),
    State('center-input', 'value'),
    State('time-dim-input', 'value'),
    State('year-range-slider', 'value'),
)
def make_plot(n_clicks, model_files, var_list, depth_level, center, time_dim, year_range):
    if n_clicks == 0 or not model_files or not var_list or len(model_files) != 2:
        raise PreventUpdate
    try:
        if isinstance(var_list, str):
            var_list = [var_list]
        # Only allow allowed variables, just in case
        var_list = [v for v in var_list if v in ALLOWED_VARS]
        if not var_list:
            return html.Div("No valid variable selected.")
        path1 = os.path.join(DATA_DIR, model_files[0])
        path2 = os.path.join(DATA_DIR, model_files[1])
        years = list(range(year_range[0], year_range[1]+1))
        ds1_seasonal = compute_seasonal_mean(path1, time_dim=time_dim, years=years)
        ds2_seasonal = compute_seasonal_mean(path2, time_dim=time_dim, years=years)
        seasons = SEASON_ORDER
        n_vars = len(var_list)
        n_seasons = len(seasons)
        fig, axs = plt.subplots(n_seasons * n_vars, 3, figsize=(24, 2.5 * n_seasons * n_vars), sharey=True)
        if axs.ndim == 1:
            axs = axs.reshape(-1, 3)
        # Precompute gridinfos for all variables
        gridinfos = {}
        for var in var_list:
            gridinfos[var] = get_grid_info(var, ds1_seasonal, mesh, y)
        # Precompute global vmin/vmax and diff_absmax for each variable
        global_vmin = {}
        global_vmax = {}
        diff_absmax = {}
        for var in var_list:
            model1_vals = []
            model2_vals = []
            diff_vals = []
            gridinfo = gridinfos[var]
            valid_mask = gridinfo["valid_mask"]
            y_var = gridinfo["y_var"]
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
        row_idx = 0
        for season in seasons:
            for var in var_list:
                gridinfo = gridinfos[var]
                nav_lon = gridinfo["nav_lon"]
                sort_idx = gridinfo["sort_idx"]
                valid_mask = gridinfo["valid_mask"]
                depth_cumsum_2d_sorted = gridinfo["depth_cumsum_2d_sorted"]
                depth_cumsum_2d_sorted_padded = gridinfo["depth_cumsum_2d_sorted_padded"]
                y_var = gridinfo["y_var"]
                x_var = gridinfo["x_var"]
                if season not in ds1_seasonal['season'].values or season not in ds2_seasonal['season'].values:
                    for ax in axs[row_idx]:
                        ax.axis('off')
                    row_idx += 1
                    continue
                T1 = ds1_seasonal[var].sel(season=season).isel({y_var: 0}).values
                T2 = ds2_seasonal[var].sel(season=season).isel({y_var: 0}).values
                T1 = np.where(valid_mask, T1, np.nan)
                T2 = np.where(valid_mask, T2, np.nan)
                T1_sorted = T1[:depth_level, sort_idx]
                T2_sorted = T2[:depth_level, sort_idx]
                diff_sorted = T1_sorted - T2_sorted
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
                im1 = axs[row_idx, 0].pcolormesh(
                    nav_lon_centered,
                    depth_cumsum_centered[:depth_level+1, :],
                    T1_centered[:, 1:],
                    shading='auto',
                    cmap=cmap,
                    vmin=global_vmin[var], vmax=global_vmax[var]
                )
                axs[row_idx, 0].set_title(f"{model_files[0]} – {season} – {ALLOWED_VARS[var]}")
                plt.colorbar(im1, ax=axs[row_idx, 0])
                im2 = axs[row_idx, 1].pcolormesh(
                    nav_lon_centered,
                    depth_cumsum_centered[:depth_level+1, :],
                    T2_centered[:, 1:],
                    shading='auto',
                    cmap=cmap,
                    vmin=global_vmin[var], vmax=global_vmax[var]
                )
                axs[row_idx, 1].set_title(f"{model_files[1]} – {season} – {ALLOWED_VARS[var]}")
                plt.colorbar(im2, ax=axs[row_idx, 1], label=ALLOWED_VARS[var])
                im3 = axs[row_idx, 2].pcolormesh(
                    nav_lon_centered,
                    depth_cumsum_centered[:depth_level+1, :],
                    diff_centered[:, 1:],
                    shading='auto',
                    cmap=cmap_diff,
                    vmin=-diff_absmax[var], vmax=diff_absmax[var]
                )
                axs[row_idx, 2].set_title(f"Δ ({model_files[0]} - {model_files[1]}) – {season} – {ALLOWED_VARS[var]}")
                plt.colorbar(im3, ax=axs[row_idx, 2], label=f"Δ {ALLOWED_VARS[var]}")
                for ax in axs[row_idx]:
                    ax.set_xlabel("Longitude")
                    xticks = np.arange(-180, 181, 60)
                    xticks_wrapped = ((xticks - center + 180) % 360) - 180 + center
                    ax.set_xticks(xticks_wrapped)
                    ax.set_xticklabels([f"{x}°" for x in xticks])
                axs[row_idx, 0].set_ylabel("Depth (m)")
                for ax in axs[row_idx]:
                    ax.set_xlim(np.min(nav_lon_centered), np.max(nav_lon_centered))
                row_idx += 1
        plt.suptitle(
            f"Vertical slice at the equator, variables: {', '.join([ALLOWED_VARS[v] for v in var_list])} (centered at {center}°)",
            fontsize=22
        )
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("ascii")
        buf.close()
        return html.Img(src="data:image/png;base64," + img_base64, style={'width': '100%', 'maxWidth': '1500px'})
    except Exception as e:
        return html.Pre(str(e))

if __name__ == "__main__":
    port = find_free_port()
    app.run(debug=True, port=port)

