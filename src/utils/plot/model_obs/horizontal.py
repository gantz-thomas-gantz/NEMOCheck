# model_obs/horizontal.py
# Goal: Interactive Dash app for visualizing and analyzing horizontal (map-based and time series) differences between a model and observational dataset.

# ---- Imports ----
import os
import xarray as xr
import plotly.graph_objects as go
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import numpy as np
from utils.plot.general import find_free_port

# ---- Data loading and configuration ----

# Define data directory relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '..', '..', '..', 'data', 'processed')

# Load climatology datasets for model and observations
clim_obs = xr.open_dataset(os.path.join(DATA_DIR, 'combined_observations.nc'))
clim_model = xr.open_dataset(os.path.join(DATA_DIR, 'model.nc'))
clim_error = clim_model - clim_obs  # Compute error (model - obs) for all variables

# Variable configuration
ALLOWED_VARS = {
    "sst": "Sea Surface Temperature (SST, °C)",
    "sss": "Sea Surface Salinity (SSS, PSU)",
    "mld": "Mixed Layer Depth (MLD, m)"
}
COLOR_CYCLE = px.colors.qualitative.Plotly

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

common_vars = get_common_vars(clim_model, clim_obs)
DEFAULT_VAR = 'sst' if 'sst' in common_vars else (common_vars[0] if common_vars else None)

# Initial color scales for each variable
initial_scale = {
    'sst': {'min': -4, 'max': 4},
    'sss': {'min': -2, 'max': 2},
    'mld': {'min': -100, 'max': 100}
}

initial_month = 3  # Default starting month

"""
Dash application for interactive horizontal (map-based and time series) model-observation error visualization.

- Lets the user select a variable, color scale, and month for map and time series error plots.
- Supports interactive selection of spatial points on the map, which are tracked and shown in the legend and time series.
- Designed for coder-style clarity: includes docstrings, type hints, and key inline comments.
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
    selected_points: list,
    relayout_data: dict = None
) -> go.Figure:
    """
    Create the map figure for model-observation error for one variable and month.

    Parameters
    ----------
    variable : str
        Variable name (must be present in both datasets).
    scale : dict
        Dict with 'min' and 'max' for color scale.
    month : int
        Month to plot (1-based).
    selected_points : list
        List of dicts with lat, lon, and color for selected spatial points.
    relayout_data : dict, optional
        Plotly relayout data (for zoom/pan).

    Returns
    -------
    go.Figure
        Plotly figure for the map.
    """
    error_data = clim_error[variable]  
    error_map = error_data.sel(month=month)
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
        colorbar=dict()  # No label on colorbar
    ))

    # Overlay selected points
    for i, pt in enumerate(selected_points):
        fig.add_trace(go.Scatter(
            x=[pt['lon']], y=[pt['lat']],
            mode='markers',
            marker=dict(color=pt['color'], size=12, line=dict(color='black', width=1)),
            name=f"Selected ({pt['lat']:.2f}, {pt['lon']:.2f})",
            hoverinfo='text',
            hovertext=f"Point {i+1}: ({pt['lat']:.2f}, {pt['lon']:.2f})"
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
        xaxis=dict(
            showgrid=True, gridcolor='lightgray', zeroline=False,
            showline=True, linecolor='black', mirror=True, range=[min(x), max(x)]
        ),
        yaxis=dict(
            showgrid=True, gridcolor='lightgray', zeroline=False,
            showline=True, linecolor='black', mirror=True, range=[min(y), max(y)]
        ),
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
    # Header
    dbc.Row([
        dbc.Col(html.H2("Model-Observation Error Explorer", className="text-center mb-2", style={'fontWeight': 'bold', "letterSpacing": "0.02em"}), width=12),
    ], className="mb-0 mt-1"),
    # Controls row
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Variable", style={'marginBottom': 0}),
                            variable_dropdown(),
                        ], width=3),
                        dbc.Col([
                            dbc.Label("Scale", style={'marginBottom': 0}),
                            error_scale_inputs(),
                        ], width=3),
                        dbc.Col([
                            dbc.Label("Month", style={'marginBottom': 0}),
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
                                'fontSize': '13px',
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
    # Plot row
    dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='ts-plot', style={'height': '60vh', 'minHeight': 400}),
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
    dcc.Store(id='selected-points-store', data=[]),
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
    Store the current variable in a hidden dcc.Store.

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
    Input('selected-points-store', 'data'),
    Input('map-plot', 'relayoutData'),
)
def update_map(
    variable: str,
    scale: dict,
    month: int,
    selected_points: list,
    relayout_data: dict
) -> go.Figure:
    """
    Update the map plot based on selection.

    Returns
    -------
    go.Figure
        Map figure.
    """
    return create_map_figure(variable, scale, month, selected_points, relayout_data)

@app.callback(
    Output('selected-points-store', 'data'),
    Input('map-plot', 'clickData'),
    Input({'type': 'remove-point-btn', 'index': dash.ALL}, 'n_clicks'),
    Input('reset-all-btn', 'n_clicks'),
    State('selected-points-store', 'data'),
    prevent_initial_call=True
)
def update_selected_points(
    clickData,
    remove_clicks,
    reset_clicks,
    selected_points
):
    """
    Update the list of selected points based on map clicks, remove-point, or reset.

    Returns
    -------
    list or dash.no_update
        Updated list of points, or no update.
    """
    ctx = callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if selected_points is None:
        selected_points = []
    # Handle reset button
    if trigger_id == 'reset-all-btn' and reset_clicks:
        return []
    # Handle point removal
    if trigger_id.startswith('{'):
        import json
        triggered_obj = json.loads(trigger_id)
        idx_to_remove = triggered_obj['index']
        if 0 <= idx_to_remove < len(selected_points):
            selected_points.pop(idx_to_remove)
        return selected_points
    # Handle new map point click, avoiding duplicates (within tolerance)
    if trigger_id == 'map-plot' and clickData:
        lon = clickData['points'][0]['x']
        lat = clickData['points'][0]['y']
        tol = 0.1
        for pt in selected_points:
            if abs(pt['lat'] - lat) < tol and abs(pt['lon'] - lon) < tol:
                return selected_points
        color = COLOR_CYCLE[len(selected_points) % len(COLOR_CYCLE)]
        selected_points.append({'lat': lat, 'lon': lon, 'color': color})
        return selected_points
    return dash.no_update

@app.callback(
    Output('legend-container', 'children'),
    Input('selected-points-store', 'data')
)
def update_legend(selected_points: list) -> list:
    """
    Update the legend showing selected spatial points.

    Returns
    -------
    list or str
        HTML children for the legend, or empty string.
    """
    if not selected_points:
        return ""
    legend_items = []
    for i, pt in enumerate(selected_points):
        legend_items.append(
            html.Div([
                html.Span("⬤", style={'color': pt['color'], 'fontSize': '16px', 'marginRight': '5px'}),
                html.Span(f"({pt['lat']:.2f}, {pt['lon']:.2f})", style={'marginRight': '4px'}),
                html.Button('×', id={'type': 'remove-point-btn', 'index': i}, n_clicks=0,
                            style={
                                'color': 'red',
                                'background': 'none',
                                'border': 'none',
                                'cursor': 'pointer',
                                'fontWeight': 'bold',
                                'fontSize': '14px',
                                'lineHeight': '14px',
                                'padding': '0 2px',
                                'marginLeft': '2px',  # less margin here!
                            })
            ], style={
                'display': 'flex',
                'alignItems': 'center',
                'marginBottom': '4px',
                'gap': '0px'  # no extra gap between elements
            })
        )
    return legend_items

@app.callback(
    Output('ts-plot', 'figure'),
    Input('selected-points-store', 'data'),
    Input('current-scale', 'data'),
    Input('current-variable', 'data')
)
def update_timeseries(
    selected_points: list,
    scale: dict,
    variable: str
) -> go.Figure:
    """
    Update the time series plot for selected points.

    Returns
    -------
    go.Figure
        Time series figure.
    """
    fig = go.Figure()
    all_months = np.arange(1, 13)
    if not selected_points:
        fig.update_layout(
            title=f'{ALLOWED_VARS.get(variable, variable)} Δ Time Series for Selected Points',
            xaxis_title='Month', yaxis_title=ALLOWED_VARS.get(variable, variable),
            xaxis=dict(tickmode='linear', tick0=1, dtick=1, range=[0.5, 12.5], tickvals=all_months, ticktext=[str(m) for m in all_months]),
            yaxis=dict(range=[scale['min'], scale['max']])
        )
        return fig

    # Calculate time series for each selected point
    clim_error = clim_model[variable] - clim_obs[variable]
    for pt in selected_points:
        ts = clim_error.sel(lat=pt['lat'], lon=pt['lon'], method='nearest')
        ts_months = ts['month'].values
        ts_vals = ts.values
        ts_full = np.full(12, np.nan)
        for i, m in enumerate(ts_months):
            if 1 <= m <= 12:
                ts_full[int(m)-1] = ts_vals[i]
        fig.add_trace(go.Scatter(
            x=all_months,
            y=ts_full,
            mode='lines+markers',
            name=f"({pt['lat']:.2f}, {pt['lon']:.2f})",
            line=dict(color=pt['color']),
            marker=dict(color=pt['color'])
        ))

    fig.update_layout(
        title=f'{ALLOWED_VARS.get(variable, variable)} Δ Time Series for Selected Points',
        xaxis_title='Month', yaxis_title=ALLOWED_VARS.get(variable, variable),
        xaxis=dict(tickmode='linear', tick0=1, dtick=1, range=[0.5, 12.5], tickvals=all_months, ticktext=[str(m) for m in all_months]),
        yaxis=dict(range=[scale['min'], scale['max']]),
        template='plotly_white'
    )
    return fig

if __name__ == '__main__':
    port = find_free_port()
    app.run(debug=True, port=port)