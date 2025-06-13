import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from dash import Dash, dcc, html, Input, Output, State
from utils.plot.general import get_vars, get_common_vars, find_free_port


DATA_DIR = "~/NEMOCheck/data/model"
DATA_DIR = os.path.expanduser(DATA_DIR)
MESH_PATH = "~/NEMOCheck/data/model/orca05l75_domain_cfg_nemov5_10m.nc"

y = 249
mesh = xr.open_dataset(MESH_PATH)
nx = mesh.sizes["x"]
nlev = mesh.sizes["nav_lev"]
depths = np.cumsum(mesh.e3t_1d.values)

# --- Build the Dash app ---
app = Dash(__name__)

files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith("_Eq.nc")])
default_files = ["nemo00_1m_201001_202212_Eq.nc", "nemo01_1m_201001_202212_Eq.nc"]

variables = get_common_vars(
    os.path.join(DATA_DIR, default_files[0]),
    os.path.join(DATA_DIR, default_files[1])
) if len(default_files) == 2 else get_vars(os.path.join(DATA_DIR, default_files[0]))
first_var = variables[0] if variables else None

# Helper for slider marks: show every ~nlev//10 with depth in meters
def slider_marks(nlev, depths):
    step = max(1, nlev // 10)
    marks = {}
    for i in range(1, nlev+1, step):
        d = int(round(depths[i-1]))
        marks[i] = f"{i} ({d}m)"
    # Always include the last level
    if nlev not in marks:
        d = int(round(depths[-1]))
        marks[nlev] = f"{nlev} ({d}m)"
    return marks

app.layout = html.Div([
    html.H2("Vertical Slice Viewer (Matplotlib, Static Image)"),
    html.Label("Select TWO model files:"),
    dcc.Dropdown(
        id='model-dropdown',
        options=[{'label': fn, 'value': fn} for fn in files],
        value=default_files,
        multi=True
    ),
    html.Label("Select variable:"),
    dcc.Dropdown(
        id='variable-dropdown',
        options=[{'label': v, 'value': v} for v in variables],
        value=first_var
    ),
    html.Label("Select depth level:"),
    dcc.Slider(
        id='depth-slider',
        min=1,
        max=nlev,
        value=30,
        marks=slider_marks(nlev, depths),
        step=1,
        tooltip={'always_visible': False}
    ),
    html.Div(id='depth-slider-value', style={'marginBottom': '16px'}),

    html.Label("Center longitude:"),
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
        style={'width': '250px', 'display': 'inline-block', 'marginRight': '16px'}
    ),
    dcc.Input(
        id='center-input',
        type='number',
        value=0,
        min=-180,
        max=180,
        step=1,
        style={'width': '100px', 'display': 'inline-block'}
    ),

    html.Button("Plot", id="plot-btn", n_clicks=0, style={'marginTop': '16px'}),
    html.Div(id='plot-container')
])

@app.callback(
    Output('center-input', 'value'),
    Input('center-preset-dropdown', 'value'),
    State('center-input', 'value'),
)
def update_center_input(preset, current):
    if preset == 'custom':
        return current  # don't change user's custom value
    else:
        return preset   # update to selected preset

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
    # Show the corresponding depth
    depth = int(round(depths[depth_level-1]))
    return f"Current depth level: {depth_level} ({depth} meters)"

@app.callback(
    Output('plot-container', 'children'),
    Input('plot-btn', 'n_clicks'),
    State('model-dropdown', 'value'),
    State('variable-dropdown', 'value'),
    State('depth-slider', 'value'),
    State('center-input', 'value'),
)
def make_plot(n_clicks, model_files, var, depth_level, center):
    if n_clicks == 0 or not model_files or not var or len(model_files) != 2:
        return html.Div("Please select two files and a variable.")
    try:
        path1 = os.path.join(DATA_DIR, model_files[0])
        path2 = os.path.join(DATA_DIR, model_files[1])
        model1 = xr.open_dataset(path1)
        model2 = xr.open_dataset(path2)

        nav_lon = mesh.glamf.sel(y=y).values
        sort_idx = np.argsort(nav_lon)
        nav_lon_sorted = nav_lon[sort_idx]

        bottom_levels = mesh.bottom_level.sel(y=y).values  # shape (nx,)
        depth_2d = np.zeros((nlev, nx))
        for x in range(nx):
            level = bottom_levels[x]
            depth_2d[:, x] = mesh.e3t_1d.values
            depth_2d[level-1, x] = mesh.e3t_0.sel(y=y, x=x).isel(nav_lev=level-1).values
        depth_cumsum_2d = -np.cumsum(depth_2d, axis=0) 
        depth_cumsum_2d_sorted = depth_cumsum_2d[:, sort_idx]
        depth_cumsum_2d_sorted_padded = np.pad(depth_cumsum_2d_sorted, ((1, 0), (0, 0)), mode='constant', constant_values=0)

        # Use user-chosen depth_level from slider
        valid_mask = np.zeros((nlev, nx), dtype=bool)
        for x in range(nx):
            valid_mask[:bottom_levels[x], x] = True

        T1 = (model1[var].isel(time_counter=0, y_grid_T=0).values)
        T2 = (model2[var].isel(time_counter=0, y_grid_T=0).values)
        T1 = np.where(valid_mask, T1, np.nan)
        T2 = np.where(valid_mask, T2, np.nan)
        T1_sorted = T1[:depth_level, sort_idx]
        T2_sorted = T2[:depth_level, sort_idx]
        diff_sorted = T1_sorted - T2_sorted 

        # --- Center at selected longitude ---
        lon_wrapped = ((nav_lon_sorted - center + 180) % 360) - 180 + center
        sort_idx2 = np.argsort(lon_wrapped)
        nav_lon_centered = lon_wrapped[sort_idx2]
        T1_centered = T1_sorted[:, sort_idx2]
        T2_centered = T2_sorted[:, sort_idx2]
        diff_centered = diff_sorted[:, sort_idx2]
        depth_cumsum_centered = depth_cumsum_2d_sorted_padded[:, sort_idx2]

        fig, axs = plt.subplots(1, 3, figsize=(24, 8), sharey=True)
        cmap = plt.get_cmap('viridis').copy()
        cmap.set_bad(color='white')
        vmin = min(np.nanmin(T1_centered), np.nanmin(T2_centered))
        vmax = max(np.nanmax(T1_centered), np.nanmax(T2_centered))
        diff_vmax = np.nanmax(np.abs(diff_centered))

        im1 = axs[0].pcolormesh(
            nav_lon_centered,
            depth_cumsum_centered[:depth_level+1, :],
            T1_centered[:, 1:],
            shading='auto',
            cmap=cmap,
            vmin=vmin, vmax=vmax
        )
        axs[0].set_title(f"{model_files[0]}")
        plt.colorbar(im1, ax=axs[0])

        im2 = axs[1].pcolormesh(
            nav_lon_centered,
            depth_cumsum_centered[:depth_level+1, :],
            T2_centered[:, 1:],
            shading='auto',
            cmap=cmap,
            vmin=vmin, vmax=vmax
        )
        axs[1].set_title(f"{model_files[1]}")
        plt.colorbar(im2, ax=axs[1], label=f"{var}")

        cmap_diff = plt.get_cmap('seismic').copy()
        cmap_diff.set_bad(color='white')
        im3 = axs[2].pcolormesh(
            nav_lon_centered,
            depth_cumsum_centered[:depth_level+1, :],
            diff_centered[:, 1:],
            shading='auto',
            cmap=cmap_diff,
            vmin=-diff_vmax, vmax=diff_vmax
        )
        axs[2].set_title(f"{model_files[0]} - {model_files[1]}")
        plt.colorbar(im3, ax=axs[2], label=f"Δ {var}")

        for ax in axs:
            ax.set_xlabel("Longitude")
            xticks = np.arange(-180, 181, 60)
            xticks_wrapped = ((xticks - center + 180) % 360) - 180 + center
            ax.set_xticks(xticks_wrapped)
            ax.set_xticklabels([f"{x}°" for x in xticks])

        axs[0].set_ylabel("Depth (m)")
        axs[0].set_xlim(np.min(nav_lon_centered), np.max(nav_lon_centered))
        axs[1].set_xlim(np.min(nav_lon_centered), np.max(nav_lon_centered))
        axs[2].set_xlim(np.min(nav_lon_centered), np.max(nav_lon_centered))
        plt.suptitle(f"Vertical slice at the equator, variable: {var} (centered at {center}°)")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("ascii")
        buf.close()

        return html.Img(src="data:image/png;base64," + img_base64, style={'width': '100%', 'maxWidth': '1500px'})
    except Exception as e:
        return html.Pre(str(e))

# To keep script executable as such
if __name__ == "__main__":
    port = find_free_port()
    app.run(debug=True, port=port)


