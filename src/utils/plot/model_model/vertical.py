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

def compute_seasonal_mean(
    path: str,
    time_dim: str = 'time_counter',
    years: list = None
) -> xr.Dataset:
    ds = xr.open_dataset(path)

    # Assign seasons as coords
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
    months = np.array(months)
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

def slider_marks(nlev, depths, max_marks=5):
    marks = {1: f"1 ({int(round(depths[0]))}m)"}
    if nlev > 1:
        marks[nlev] = f"{nlev} ({int(round(depths[-1]))}m)"
    if nlev <= max_marks:
        for i in range(2, nlev):
            marks[i] = f"{i} ({int(round(depths[i-1]))}m)"
    else:
        step = (nlev - 1) // (max_marks - 1)
        for j in range(1, max_marks - 1):
            idx = 1 + j * step
            if idx < nlev and idx not in marks:
                marks[idx] = f"{idx} ({int(round(depths[idx-1]))}m)"
    marks = {k: marks[k] for k in sorted(marks.keys())}
    return marks

def range_slider_year_marks(years, max_marks=5):
    """
    Given a list or range of years, return a dict of at most max_marks evenly spaced marks,
    always including the first and last year.
    """
    if isinstance(years, range):
        years = list(years)
    n_years = len(years)
    marks = {years[0]: str(years[0])}
    if n_years > 1:
        marks[years[-1]] = str(years[-1])
    if n_years <= max_marks:
        for y in years[1:-1]:
            marks[y] = str(y)
    else:
        step = (n_years - 1) // (max_marks - 1)
        for j in range(1, max_marks - 1):
            idx = j * step
            if idx < n_years - 1:
                y = years[idx]
                marks[y] = str(y)
    marks = {k: marks[k] for k in sorted(marks.keys())}
    return marks

external_stylesheets = [dbc.themes.FLATLY]
app = Dash(__name__, external_stylesheets=external_stylesheets)

files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith("_Eq.nc")])
default_files = ["nemo00_1m_201001_202212_Eq.nc", "nemo01_1m_201001_202212_Eq.nc"]

variables = get_common_vars(
    os.path.join(DATA_DIR, default_files[0]),
    os.path.join(DATA_DIR, default_files[1])
) if len(default_files) == 2 else get_vars(os.path.join(DATA_DIR, default_files[0]))
first_var = variables[0] if variables else None

def get_year_range_from_netcdf(path, time_dim):
    ds = xr.open_dataset(path)
    time_vals = pd.to_datetime(ds[time_dim].values)
    years = pd.Series(time_vals).dt.year.unique()
    return int(years.min()), int(years.max())

default_time_dim = "time_counter"
default_first_year, default_last_year = get_year_range_from_netcdf(
    os.path.join(DATA_DIR, default_files[0]), default_time_dim
)

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("Vertical Slice Viewer"), width="auto"),
    ], className="mb-2"),
    dbc.Row([
        dbc.Alert(
            "Note: December of a year (e.g. December 2010) counts into DJF of the following year (e.g. DJF 2011). Only full seasons are taken into account (e.g. January 2010 will never be taken into account).",
            color="warning",
            className="py-2 mb-2",
            style={"fontWeight": "bold", "fontSize": "1em"}
        ),
    ]),
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
                            dbc.Label("Variable:"),
                            dcc.Dropdown(
                                id='variable-dropdown',
                                options=[{'label': v, 'value': v} for v in variables],
                                value=first_var,
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
                                marks=slider_marks(nlev, depths),
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
    if model_files and len(model_files) == 2:
        path1 = os.path.join(DATA_DIR, model_files[0])
        path2 = os.path.join(DATA_DIR, model_files[1])
        variables = get_common_vars(path1, path2)
        opts = [{'label': v, 'value': v} for v in variables]
        return opts, variables[0] if variables else None
    elif model_files and len(model_files) == 1:
        path = os.path.join(DATA_DIR, model_files[0])
        variables = get_vars(path)
        opts = [{'label': v, 'value': v} for v in variables]
        return opts, variables[0] if variables else None
    return [], None

@app.callback(
    Output('depth-slider-value', 'children'),
    Input('depth-slider', 'value')
)
def update_depth_text(depth_level):
    depth = int(round(depths[depth_level-1]))
    return f"Current depth level: {depth_level} ({depth} meters)"

@app.callback(
    Output('year-range-slider-value', 'children'),
    Input('year-range-slider', 'value')
)
def update_year_range_text(years):
    y0, y1 = years
    return f"Selected years: {y0}–{y1}"

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
def make_plot(n_clicks, model_files, var, depth_level, center, time_dim, year_range):
    if n_clicks == 0 or not model_files or not var or len(model_files) != 2:
        raise PreventUpdate
    try:
        path1 = os.path.join(DATA_DIR, model_files[0])
        path2 = os.path.join(DATA_DIR, model_files[1])
        years = list(range(year_range[0], year_range[1]+1))

        ds1_seasonal = compute_seasonal_mean(path1, time_dim=time_dim, years=years)
        ds2_seasonal = compute_seasonal_mean(path2, time_dim=time_dim, years=years)

        nav_lon = mesh.glamf.sel(y=y).values
        sort_idx = np.argsort(nav_lon)
        nav_lon_sorted = nav_lon[sort_idx]

        bottom_levels = mesh.bottom_level.sel(y=y).values
        depth_2d = np.zeros((nlev, nx))
        for x in range(nx):
            level = bottom_levels[x]
            depth_2d[:, x] = mesh.e3t_1d.values
            depth_2d[level-1, x] = mesh.e3t_0.sel(y=y, x=x).isel(nav_lev=level-1).values
        depth_cumsum_2d = -np.cumsum(depth_2d, axis=0)
        depth_cumsum_2d_sorted = depth_cumsum_2d[:, sort_idx]
        depth_cumsum_2d_sorted_padded = np.pad(depth_cumsum_2d_sorted, ((1, 0), (0, 0)), mode='constant', constant_values=0)

        valid_mask = np.zeros((nlev, nx), dtype=bool)
        for x in range(nx):
            valid_mask[:bottom_levels[x], x] = True

        seasons = ['DJF', 'MAM', 'JJA', 'SON']

        model1_vals = []
        model2_vals = []
        diff_vals = []
        for season in seasons:
            if season not in ds1_seasonal['season'].values or season not in ds2_seasonal['season'].values:
                continue
            T1 = ds1_seasonal[var].sel(season=season).isel(y_grid_T=0).values
            T2 = ds2_seasonal[var].sel(season=season).isel(y_grid_T=0).values
            T1 = np.where(valid_mask, T1, np.nan)
            T2 = np.where(valid_mask, T2, np.nan)
            T1_slice = T1[:depth_level, :]
            T2_slice = T2[:depth_level, :]
            model1_vals.append(T1_slice)
            model2_vals.append(T2_slice)
            diff_vals.append(T1_slice - T2_slice)
        if not model1_vals or not model2_vals:
            return html.Div("No valid data found for the selected years and seasons.")

        model1_all = np.concatenate([a.flatten() for a in model1_vals])
        model2_all = np.concatenate([a.flatten() for a in model2_vals])
        diff_all = np.concatenate([a.flatten() for a in diff_vals])
        global_vmin = np.nanmin([model1_all, model2_all])
        global_vmax = np.nanmax([model1_all, model2_all])
        diff_absmax = np.nanmax(np.abs(diff_all))

        fig, axs = plt.subplots(4, 3, figsize=(24, 30), sharey=True)

        for i, season in enumerate(seasons):
            if season not in ds1_seasonal['season'].values or season not in ds2_seasonal['season'].values:
                for ax in axs[i]:
                    ax.axis('off')
                continue
            T1 = ds1_seasonal[var].sel(season=season).isel(y_grid_T=0).values
            T2 = ds2_seasonal[var].sel(season=season).isel(y_grid_T=0).values
            T1 = np.where(valid_mask, T1, np.nan)
            T2 = np.where(valid_mask, T2, np.nan)
            T1_sorted = T1[:depth_level, sort_idx]
            T2_sorted = T2[:depth_level, sort_idx]
            diff_sorted = T1_sorted - T2_sorted

            lon_wrapped = ((nav_lon_sorted - center + 180) % 360) - 180 + center
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

            im1 = axs[i,0].pcolormesh(
                nav_lon_centered,
                depth_cumsum_centered[:depth_level+1, :],
                T1_centered[:, 1:],
                shading='auto',
                cmap=cmap,
                vmin=global_vmin, vmax=global_vmax
            )
            axs[i,0].set_title(f"{model_files[0]} – {season}")
            plt.colorbar(im1, ax=axs[i,0])

            im2 = axs[i,1].pcolormesh(
                nav_lon_centered,
                depth_cumsum_centered[:depth_level+1, :],
                T2_centered[:, 1:],
                shading='auto',
                cmap=cmap,
                vmin=global_vmin, vmax=global_vmax
            )
            axs[i,1].set_title(f"{model_files[1]} – {season}")
            plt.colorbar(im2, ax=axs[i,1], label=f"{var}")

            im3 = axs[i,2].pcolormesh(
                nav_lon_centered,
                depth_cumsum_centered[:depth_level+1, :],
                diff_centered[:, 1:],
                shading='auto',
                cmap=cmap_diff,
                vmin=-diff_absmax, vmax=diff_absmax
            )
            axs[i,2].set_title(f"Δ ({model_files[0]} - {model_files[1]}) – {season}")
            plt.colorbar(im3, ax=axs[i,2], label=f"Δ {var}")

            for ax in axs[i]:
                ax.set_xlabel("Longitude")
                xticks = np.arange(-180, 181, 60)
                xticks_wrapped = ((xticks - center + 180) % 360) - 180 + center
                ax.set_xticks(xticks_wrapped)
                ax.set_xticklabels([f"{x}°" for x in xticks])

            axs[i,0].set_ylabel("Depth (m)")
            for ax in axs[i]:
                ax.set_xlim(np.min(nav_lon_centered), np.max(nav_lon_centered))
        plt.suptitle(f"Vertical slice at the equator, variable: {var} (centered at {center}°)", fontsize=22)
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

