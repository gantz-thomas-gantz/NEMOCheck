# model_obs/profiler.py
# Goal: Interactive Dash app for visualizing and analyzing model-observation error profiles along user-selected transects on a map.

# ---- Imports ----
import os
import xarray as xr
import numpy as np
from scipy.interpolate import griddata
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from utils.plot.general import find_free_port

# ---- Constants and data setup ----

ALLOWED_VARS = {
    "sst": "Sea Surface Temperature (SST, °C)",
    "sss": "Sea Surface Salinity (SSS, PSU)",
    "mld": "Mixed Layer Depth (MLD, m)"
}
COLOR_CYCLE = px.colors.qualitative.Plotly

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '..', '..', '..', 'data')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

# Load model and observation climatologies
clim_obs = xr.open_dataset(os.path.join(PROCESSED_DIR, 'combined_observations.nc'))
clim_model = xr.open_dataset(os.path.join(PROCESSED_DIR, 'model_new.nc'))
clim_error = clim_model - clim_obs  # Compute error (model - obs) for all variables

def get_common_vars(ds1: xr.Dataset, ds2: xr.Dataset) -> list:
    """
    Return variables present in both datasets and allowed for plotting.

    Parameters
    ----------
    ds1 : xr.Dataset
        First dataset.
    ds2 : xr.Dataset
        Second dataset.

    Returns
    -------
    list
        List of common variable names.
    """
    return [v for v in ds1.data_vars if v in ds2.data_vars and v in ALLOWED_VARS]

# Variable setup
common_vars = get_common_vars(clim_model, clim_obs)
DEFAULT_VAR = 'sst' if 'sst' in common_vars else (common_vars[0] if common_vars else None)
initial_scale = {
    'sst': {'min': -4, 'max': 4},
    'sss': {'min': -2, 'max': 2},
    'mld': {'min': -100, 'max': 100}
}
initial_month = 3

"""
Dash application for interactive error profile exploration between model and observations.

- Lets the user select a variable, color scale, and month for map and profile error plots.
- Supports interactive selection of pairs of points on the map, drawing transects and error profiles between them.
- Visualizes differences on a map (with transects) and as 1D error profiles along selected lines.
- Designed for coder-style clarity: includes docstrings, type hints, and rich inline comments.
- Data directory and file conventions are fixed; see README for requirements.

Dependencies and definitions required elsewhere in the project:
    - find_free_port: Function to find an available port for local hosting.
    - DATA_DIR structure and processed NetCDF files as described in README.
"""
external_stylesheets = [dbc.themes.FLATLY]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

def create_map_figure(
    variable: str,
    scale: dict,
    month: int,
    selected_pairs: list,
    relayout_data: dict = None
) -> go.Figure:
    """
    Create the map figure showing model-observation error and user-selected transects.

    Parameters
    ----------
    variable : str
        Variable to plot.
    scale : dict
        Color scale dict with 'min' and 'max'.
    month : int
        Month to plot (1-based).
    selected_pairs : list
        List of dicts, each containing two points and a color.
    relayout_data : dict, optional
        Plotly relayout data (for zoom/pan).

    Returns
    -------
    go.Figure
        Plotly figure for the map.
    """
    error_map = clim_error[variable].sel(month=month)
    z = error_map.values
    x = error_map['lon'].values
    y = error_map['lat'].values

    fig = go.Figure(go.Heatmap(
        z=z,
        x=x,
        y=y,
        colorscale='RdBu_r',
        zmin=scale['min'],
        zmax=scale['max'],
        colorbar=dict(title="")
    ))

    # Draw user-selected transects (pairs of points)
    for i, pair in enumerate(selected_pairs):
        p1, p2 = pair['points']
        color = pair['color']

        fig.add_trace(go.Scatter(
            x=[p1['lon'], p2['lon']],
            y=[p1['lat'], p2['lat']],
            mode='lines+markers+text',
            line=dict(color=color, width=2, dash='dash'),
            marker=dict(color=color, size=10, symbol='circle'),
            text=["Start", "End"],
            textposition="top right",
            showlegend=False,
            hoverinfo='skip'
        ))

    fig.update_layout(
        showlegend=False,
        title=f'{ALLOWED_VARS[variable]} Δ model - obs in Month {month}',
        xaxis_title='Longitude',
        yaxis_title='Latitude',
        dragmode='pan',
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=38, l=38, b=38, r=38),
        xaxis=dict(showgrid=True, gridcolor='lightgray', zeroline=False, showline=True, linecolor='black', mirror=True, range=[min(x), max(x)]),
        yaxis=dict(showgrid=True, gridcolor='lightgray', zeroline=False, showline=True, linecolor='black', mirror=True, range=[min(y), max(y)]),
    )

    # Preserve axis ranges if user zoomed/panned
    if relayout_data:
        x0 = relayout_data.get('xaxis.range[0]')
        x1 = relayout_data.get('xaxis.range[1]')
        y0 = relayout_data.get('yaxis.range[0]')
        y1 = relayout_data.get('yaxis.range[1]')
        if None not in (x0, x1, y0, y1):
            if y0 > y1:
                y0, y1 = y1, y0
            fig.update_layout(
                xaxis=dict(range=[x0, x1]),
                yaxis=dict(range=[y0, y1])
            )

    return fig

def variable_dropdown() -> dcc.Dropdown:
    """
    Create the variable selection dropdown.

    Returns
    -------
    dcc.Dropdown
        Dropdown for variable selection.
    """
    return dcc.Dropdown(
        id='variable-dropdown',
        options=[{'label': ALLOWED_VARS[v], 'value': v} for v in common_vars],
        value=DEFAULT_VAR,
        clearable=False,
        style={'marginBottom': '6px'}
    )

def error_scale_inputs() -> dbc.InputGroup:
    """
    Create the input controls for color scale min/max.

    Returns
    -------
    dbc.InputGroup
        Input group for color scale.
    """
    return dbc.InputGroup([
        dbc.InputGroupText("Min"),
        dbc.Input(id='scale-min', type='number', value=initial_scale.get(DEFAULT_VAR, {'min': -1})['min'], step=0.1, style={'width': '80px'}),
        dbc.InputGroupText("Max"),
        dbc.Input(id='scale-max', type='number', value=initial_scale.get(DEFAULT_VAR, {'max': 1})['max'], step=0.1, style={'width': '80px'}),
        dbc.Button("Update", id='update-scale-btn', n_clicks=0, color="secondary", outline=True, size="sm")
    ], size="sm", className="mb-1")

# ---- App Layout ----

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("Model-Observation Error Profile Explorer", className="text-center mb-2", style={'fontWeight': 'bold', "letterSpacing": "0.02em"}), width=12),
    ], className="mb-0 mt-1"),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Variable"),
                            variable_dropdown()
                        ], width=3),
                        dbc.Col([
                            dbc.Label("Scale"),
                            error_scale_inputs(),
                        ], width=3),
                        dbc.Col([
                            dbc.Label("Month"),
                            dcc.Slider(
                                id='month-slider',
                                min=1, max=12, step=1,
                                value=initial_month,
                                marks={i: str(i) for i in range(1, 13)},
                                tooltip={'always_visible': False}
                            ),
                        ], width=3),
                        dbc.Col([
                            html.Div(id='legend-container', style={
                                'padding': '8px',
                                'border': '1px solid #eee',
                                'backgroundColor': '#fafbfc',
                                'fontFamily': 'monospace',
                                'fontSize': '8px',
                                'maxHeight': '70px',
                                'overflowY': 'auto'
                            }),
                        ], width=2),
                        dbc.Col([
                            dbc.Button('Reset All', id='reset-all-btn', n_clicks=0, color="outline-danger", size="sm", className="mt-4"),
                        ], width=1),
                    ], align="end", justify="start")
                ])
            ], className="mb-2 shadow-sm")
        ], width=12)
    ], className="mb-1"),
    dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='profile-plot', style={'height': '60vh', 'minHeight': 400}),
                ])
            ], className="shadow-sm"),
            width=6
        ),
        dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='map-plot', style={'height': '60vh', 'minHeight': 400}),
                ])
            ], className="shadow-sm"),
            width=6
        ),
    ], justify='center', align='top', style={'height': '66vh', 'marginTop': 0}),
    dcc.Store(id='current-scale', data=initial_scale.get(DEFAULT_VAR, {'min': -1, 'max': 1})),
    dcc.Store(id='pair-store', data={'points': [], 'pairs': []}),
    dcc.Store(id='current-variable', data=DEFAULT_VAR),
], fluid=True)

# ---- Callbacks ----

@app.callback(
    Output('scale-min', 'value'),
    Output('scale-max', 'value'),
    Output('current-scale', 'data'),
    Input('variable-dropdown', 'value'),
    Input('update-scale-btn', 'n_clicks'),
    State('scale-min', 'value'),
    State('scale-max', 'value'),
)
def sync_scale_and_variable(
    variable: str,
    n_clicks: int,
    cur_min: float,
    cur_max: float
):
    """
    Synchronize color scale min/max with variable selection or update.

    Returns
    -------
    Tuple
        (new min, new max, new scale dictionary)
    """
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    # If variable changed, load its default scale
    if triggered_id == 'variable-dropdown':
        defaults = initial_scale.get(variable, {'min': cur_min, 'max': cur_max})
        return defaults['min'], defaults['max'], defaults
    # If user pressed update, use the entered min/max
    if triggered_id == 'update-scale-btn':
        if cur_min is not None and cur_max is not None and cur_min < cur_max:
            return cur_min, cur_max, {'min': cur_min, 'max': cur_max}
        else:
            return cur_min, cur_max, dash.no_update
    # Fallback/default
    defaults = initial_scale.get(variable, {'min': cur_min, 'max': cur_max})
    return defaults['min'], defaults['max'], defaults

@app.callback(
    Output('current-variable', 'data'),
    Input('variable-dropdown', 'value')
)
def store_current_variable(variable: str) -> str:
    """
    Store the currently selected variable in a hidden dcc.Store.

    Returns
    -------
    str
        Selected variable.
    """
    return variable

@app.callback(
    Output('map-plot', 'figure'),
    Input('current-variable', 'data'),
    Input('current-scale', 'data'),
    Input('month-slider', 'value'),
    Input('pair-store', 'data'),
    Input('map-plot', 'relayoutData'),
)
def update_map(
    variable: str,
    scale: dict,
    month: int,
    store: dict,
    relayout_data: dict
) -> go.Figure:
    """
    Update the map plot based on variable, scale, month, and selected pairs.

    Returns
    -------
    go.Figure
        Updated map figure.
    """
    return create_map_figure(variable, scale, month, store['pairs'], relayout_data)

@app.callback(
    Output('pair-store', 'data'),
    Input('map-plot', 'clickData'),
    Input('reset-all-btn', 'n_clicks'),
    Input({'type': 'delete-pair-btn', 'index': dash.ALL}, 'n_clicks'),
    State('pair-store', 'data'),
    prevent_initial_call=True
)
def update_pairs(
    click_data,
    reset_clicks,
    delete_clicks,
    store
) -> dict:
    """
    Update the pair-store: handle point selection, reset, and deletion.

    Returns
    -------
    dict
        Updated pair-store with current points/pairs.
    """
    ctx = callback_context
    triggered = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    if triggered == 'reset-all-btn':
        return {'points': [], 'pairs': []}

    elif triggered and triggered.startswith('{'):
        import json
        triggered_obj = json.loads(triggered)
        if triggered_obj.get('type') == 'delete-pair-btn':
            index = triggered_obj.get('index')
            if index is not None and 0 <= index < len(store['pairs']):
                del store['pairs'][index]
                return store

    elif triggered == 'map-plot' and click_data:
        lon = click_data['points'][0]['x']
        lat = click_data['points'][0]['y']
        color = COLOR_CYCLE[len(store['pairs']) % len(COLOR_CYCLE)]
        # Add a point, and if two selected, convert to a pair and reset
        store['points'].append({'lat': lat, 'lon': lon})
        if len(store['points']) == 2:
            store['pairs'].append({
                'points': store['points'],
                'color': color
            })
            store['points'] = []
    return store

@app.callback(
    Output('legend-container', 'children'),
    Input('pair-store', 'data')
)
def update_legend(store: dict):
    """
    Update the legend showing all selected pairs.

    Returns
    -------
    list or str
        List of HTML children for the legend, or a message.
    """
    selected_pairs = store['pairs']
    if not selected_pairs:
        return "No point pairs selected."

    children = []
    for i, pair in enumerate(selected_pairs):
        p1, p2 = pair['points']
        color = pair['color']
        children.append(html.Div([
            html.Span("⬤", style={'color': color, 'fontSize': '16px', 'marginRight': '6px'}),
            f"({p1['lat']:.2f}, {p1['lon']:.2f}) → ({p2['lat']:.2f}, {p2['lon']:.2f})",
            html.Button("\u274C", id={'type': 'delete-pair-btn', 'index': i}, n_clicks=0, style={
                'color': 'red',
                'marginLeft': '10px',
                'border': 'none',
                'background': 'transparent',
                'cursor': 'pointer'
            })
        ], style={'marginBottom': '4px'}))
    return children

@app.callback(
    Output('profile-plot', 'figure'),
    Input('pair-store', 'data'),
    Input('month-slider', 'value'),
    Input('current-scale', 'data'),
    Input('current-variable', 'data')
)
def plot_profiles(
    store: dict,
    month: int,
    scale: dict,
    variable: str
) -> go.Figure:
    """
    Plot error profiles for all user-selected pairs (transects).

    Returns
    -------
    go.Figure
        Plotly figure with error profiles.
    """
    fig = go.Figure()
    for pair in store['pairs']:
        p1, p2 = pair['points']
        color = pair['color']
        n_samples = 100  # Number of points along transect
        lats = np.linspace(p1['lat'], p2['lat'], n_samples)
        lons = np.linspace(p1['lon'], p2['lon'], n_samples)

        # Interpolate field to transect
        lat_grid, lon_grid = np.meshgrid(clim_error[variable].lat.values, clim_error[variable].lon.values, indexing='ij')
        field = clim_error[variable].sel(month=month).values
        sample_points = np.vstack([lat_grid.ravel(), lon_grid.ravel()]).T
        values = field.ravel()
        profile_vals = griddata(sample_points, values, (lats, lons), method='linear')

        # Approximate distance along transect in km (using crude conversion)
        distances = np.sqrt((lats - lats[0]) ** 2 + (lons - lons[0]) ** 2) * 111

        fig.add_trace(go.Scatter(
            x=distances,
            y=profile_vals,
            mode='lines+markers',
            line=dict(color=color),
            marker=dict(size=4),
            name=f"({p1['lat']:.1f}, {p1['lon']:.1f}) → ({p2['lat']:.1f}, {p2['lon']:.1f})"
        ))

    fig.update_layout(
        title=f"{ALLOWED_VARS.get(variable, variable)} Error Profiles",
        xaxis_title="Distance from Start (km)",
        yaxis_title=ALLOWED_VARS.get(variable, variable),
        yaxis=dict(range=[scale['min'], scale['max']]),
        template='plotly_white'
    )
    return fig

if __name__ == '__main__':
    port = find_free_port()
    app.run(debug=True, port=port)