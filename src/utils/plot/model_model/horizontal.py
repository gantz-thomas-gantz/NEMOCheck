import os
import re
import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objs as go
import plotly.express as px
import xarray as xr
import numpy as np
import contextlib
import xesmf as xe
from utils.plot.general import find_free_port
from utils.data.general import fill_coastal_points_in_time
import regionmask
import dash_bootstrap_components as dbc

# ---- Configuration ----
DATA_DIR = "~/NEMOCheck/data/processed"

DATA_DIR = os.path.expanduser(DATA_DIR)
WEIGHTS_PATH = os.path.expanduser("~/NEMOCheck/data/weights/weights_bilinear_pmmh.nc")
MESH_PATH = "~/NEMOCheck/data/model/orca05l75_domain_cfg_nemov5_10m.nc"


VARIABLE_LABELS = {
    'tos': 'Sea Surface Temperature (tos, °C)',
    'sos': 'Sea Surface Salinity (sos, PSU)',
    'zos': 'Sea Surface Height (zos, m)',
    'mldkz5': 'Mixed Layer Depth (MLD Kz5, m)',
    'mldr10_1': 'Mixed Layer Depth (MLD R10, m)',
    'tob': 'Bottom Temperature (tob, °C)',
    'sob': 'Bottom Salinity (sob, PSU)',
    'taubot': 'Bottom Stress (taubot, N/m²)'
}
AVAILABLE_VARIABLES = list(VARIABLE_LABELS.keys())
COLOR_CYCLE = px.colors.qualitative.Plotly
initial_scale = {'min': -4, 'max': 4}
initial_month = 3  

# ---- Load mesh and land mask ONCE ----
mesh = xr.open_dataset(MESH_PATH)

# ---- Grid for interpolation ----
target_grid = xr.Dataset({
    'lon': (['lon'], np.arange(-179.5, 180, 1)),
    'lat': (['lat'], np.arange(-89.5, 90, 1))
})

land_mask = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(target_grid)
ocean_mask = land_mask.isnull()

files = [
    fn for fn in sorted(os.listdir(DATA_DIR))
    if re.match(r"^nemo.*_clim_.*_.*.nc$", fn)
]

def build_regridder():
    if not files:
        raise FileNotFoundError("No .nc files found in DATA_DIR for regridder construction")
    sample_path = os.path.join(DATA_DIR, files[0])
    ds = xr.open_dataset(sample_path)
    for variable_name in AVAILABLE_VARIABLES:
        if variable_name in ds.variables:
            arr = ds[variable_name]
            break
    else:
        raise ValueError("None of AVAILABLE_VARIABLES found in sample file for regridder construction")

    arr = arr.isel(time_counter=0, drop=True)
    arr = arr.rename({'nav_lon': 'lon', 'nav_lat': 'lat'})
    arr = arr.assign_coords(lon=mesh['glamt'], lat=mesh['gphit'])
    with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
        regridder = xe.Regridder(
            arr, target_grid, method='bilinear',
            filename=WEIGHTS_PATH,
            reuse_weights=True,
            ignore_degenerate=True,
            periodic=True
        )
    return regridder

regridder = build_regridder()

external_stylesheets = [dbc.themes.FLATLY]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

def model_file_dropdown(component_id, label):
    return dbc.Col([
        dbc.Label(label, style={'marginBottom': 0}),
        dcc.Dropdown(
            id=component_id,
            options=[{'label': fn, 'value': fn} for fn in files],
            placeholder=label,
            clearable=False
        ),
    ], width=3)

def variable_dropdown():
    return dbc.Col([
        dbc.Label("Variable", style={'marginBottom': 0}),
        dcc.Dropdown(
            id='variable-dropdown',
            options=[{'label': VARIABLE_LABELS[v], 'value': v} for v in AVAILABLE_VARIABLES],
            placeholder='Choose variable',
            clearable=False,
        ),
    ], width=3)

def start_button_col():
    return dbc.Col([
        dbc.Label(""),
        dbc.Button("Start", id="start-btn", n_clicks=0, color="primary", style={'width': '100%'})
    ], width=3)

def error_scale_inputs():
    return dbc.Col([
        dbc.Label("Scale", style={'marginBottom': 0}),
        dbc.InputGroup([
            dbc.InputGroupText("Min"),
            dbc.Input(id='scale-min', type='number', value=initial_scale['min'], step=0.1, style={'width': '80px'}),
            dbc.InputGroupText("Max"),
            dbc.Input(id='scale-max', type='number', value=initial_scale['max'], step=0.1, style={'width': '80px'}),
            dbc.Button("Update", id='update-scale-btn', n_clicks=0, color="secondary", outline=True, size="sm")
        ], size="sm", className="mb-1"),
    ], width=4)

def month_slider():
    return dbc.Col([
        dbc.Label("Month", style={'marginBottom': 0}),
        dcc.Slider(
            id='month-slider',
            min=1,
            max=12,
            step=1,
            value=initial_month,
            marks={i: str(i) for i in range(1, 13)},
            tooltip={'always_visible': False}
        ),
    ], width=4)

def app_legend():
    return html.Div(id='legend-container', style={
        'padding': '8px',
        'border': '1px solid #eee',
        'backgroundColor': '#fafbfc',
        'fontFamily': 'monospace',
        'fontSize': '13px',
        'maxHeight': '70px',
        'overflowY': 'auto'
    })

def reset_points_button():
    return dbc.Button('Reset All', id='reset-all-btn', n_clicks=0, color="outline-danger", size="sm", className="mt-4")

# --- Layout ---
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col(html.H2("Model-Model Difference Viewer (Horizontal)", className="text-center mb-2", style={'fontWeight': 'bold', "letterSpacing": "0.02em"}), width=12),
    ], className="mb-0 mt-1"),

    # Initial selection box (now all cols same width and Start button in same row)
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        model_file_dropdown('model1-dropdown', "First model file"),
                        model_file_dropdown('model2-dropdown', "Second model file"),
                        variable_dropdown(),
                        start_button_col()
                    ], align="end", justify="start", className="g-2"),
                ])
            ], className="mb-2 shadow-sm")
        ], width=12)
    ], className="mb-1"),

    # --- Controls for plot (in their own card!) ---
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        error_scale_inputs(),
                        month_slider(),
                        dbc.Col([
                            app_legend(),
                        ], width=3),
                        dbc.Col([
                            reset_points_button(),
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

    # Stores and loading
    dcc.Store(id='clim-error-store'),
    dcc.Store(id='selected-points-store', data=[]),
    dcc.Store(id='current-scale', data=initial_scale),
    dcc.Store(id='current-variable', data=AVAILABLE_VARIABLES[0]),
    dcc.Loading(
        id="loading-global",
        type="circle",
        children=html.Div(id="loading-dummy", style={"display": "none"})
    )
], fluid=True)

@app.callback(
    Output('current-scale', 'data'),
    Input('update-scale-btn', 'n_clicks'),
    State('scale-min', 'value'),
    State('scale-max', 'value')
)
def update_scale(n_clicks, scale_min, scale_max):
    if scale_min is not None and scale_max is not None and scale_min < scale_max:
        return {'min': scale_min, 'max': scale_max}
    return dash.no_update

@app.callback(
    Output('current-variable', 'data'),
    Input('variable-dropdown', 'value')
)
def store_current_variable(selected_variable):
    return selected_variable

@app.callback(
    [Output('clim-error-store', 'data'),
     Output('loading-dummy', 'children')],
    [Input('start-btn', 'n_clicks')],
    [State('model1-dropdown', 'value'),
     State('model2-dropdown', 'value'),
     State('variable-dropdown', 'value')]
)
def compute_clim_error(n_clicks, first_model_file, second_model_file, selected_variable):
    if n_clicks == 0:
        return dash.no_update, ""
    if first_model_file and second_model_file and selected_variable:
        path1 = os.path.join(DATA_DIR, first_model_file)
        path2 = os.path.join(DATA_DIR, second_model_file)
        try:
            ds1 = xr.open_dataset(path1)
            ds2 = xr.open_dataset(path2)
            arr1 = ds1[selected_variable]
            arr2 = ds2[selected_variable]
            difference = arr1 - arr2
            difference = difference.rename({
                'nav_lon': 'lon',
                'nav_lat': 'lat',
                'time_counter': 'time'
            })
            difference = difference.assign_coords(lon=mesh['glamt'], lat=mesh['gphit'])
            filled_difference = fill_coastal_points_in_time(difference, 20)
            regridded_difference = regridder(filled_difference)
            regridded_difference = regridded_difference.where(ocean_mask)
            return regridded_difference.to_dataset(name='error').to_dict(), ""
        except Exception:
            return dash.no_update, ""
    return dash.no_update, ""

@app.callback(
    Output('map-plot', 'figure'),
    Input('clim-error-store', 'data'),
    Input('current-scale', 'data'),
    Input('selected-points-store', 'data'),
    Input('month-slider', 'value'),
    Input('map-plot', 'relayoutData'),
    Input('current-variable', 'data'),
)
def update_map(clim_error_dict, scale, selected_points, selected_month, relayout_data, selected_variable):
    if not clim_error_dict or not selected_variable:
        return go.Figure()
    clim_error = xr.Dataset.from_dict(clim_error_dict)['error']
    data_for_month = clim_error.isel(time=selected_month-1)
    lons = data_for_month['lon'].values
    lats = data_for_month['lat'].values
    z = data_for_month.values
    fig = go.Figure(go.Heatmap(
        z=z,
        x=lons,
        y=lats,
        colorscale='RdBu_r',
        zmin=scale['min'],
        zmax=scale['max'],
        colorbar=dict()  # No label on colorbar to match reference app
    ))
    for i, point in enumerate(selected_points or []):
        fig.add_trace(go.Scatter(
            x=[point['lon']],
            y=[point['lat']],
            mode='markers',
            marker=dict(color=point['color'], size=12, line=dict(color='black', width=1)),
            name=f"Selected ({point['lat']:.2f}, {point['lon']:.2f})",
            hoverinfo='text',
            hovertext=f"Point {i+1}: ({point['lat']:.2f}, {point['lon']:.2f})"
        ))
    variable_label = VARIABLE_LABELS.get(selected_variable, selected_variable)
    fig.update_layout(
        showlegend=False,
        title=f'{variable_label} Difference (Model 1 - Model 2) in Month {selected_month}',
        xaxis_title='Longitude',
        yaxis_title='Latitude',
        dragmode='pan',
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=38, l=38, b=38, r=38),
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            zeroline=False,
            showline=True,
            linecolor='black',
            mirror=True,
            range=[np.nanmin(lons), np.nanmax(lons)]
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            zeroline=False,
            showline=True,
            linecolor='black',
            mirror=True,
            range=[np.nanmin(lats), np.nanmax(lats)]
        )
    )
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

@app.callback(
    Output('selected-points-store', 'data'),
    Input('map-plot', 'clickData'),
    Input({'type': 'remove-point-btn', 'index': dash.ALL}, 'n_clicks'),
    Input('reset-all-btn', 'n_clicks'),
    State('selected-points-store', 'data'),
    prevent_initial_call=True
)
def update_selected_points(clickData, remove_clicks, reset_clicks, selected_points):
    ctx = callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    selected_points = selected_points or []
    if trigger_id == 'reset-all-btn' and reset_clicks:
        return []
    if trigger_id.startswith('{'):
        import json
        idx_to_remove = json.loads(trigger_id)['index']
        if 0 <= idx_to_remove < len(selected_points):
            selected_points.pop(idx_to_remove)
        return selected_points
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
def update_legend(selected_points):
    if not selected_points:
        return ""
    legend_items = []
    for i, point in enumerate(selected_points):
        legend_items.append(
            html.Div([
                html.Span("⬤", style={'color': point['color'], 'fontSize': '16px', 'marginRight': '5px'}),
                html.Span(f"({point['lat']:.2f}, {point['lon']:.2f})", style={'marginRight': '4px'}),
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
                                'marginLeft': '2px',
                            })
            ], style={
                'display': 'flex',
                'alignItems': 'center',
                'marginBottom': '4px',
                'gap': '0px'
            })
        )
    return legend_items

@app.callback(
    Output('ts-plot', 'figure'),
    Input('selected-points-store', 'data'),
    Input('clim-error-store', 'data'),
    Input('current-scale', 'data'),
    Input('current-variable', 'data')
)
def update_timeseries(selected_points, clim_error_dict, scale, selected_variable):
    fig = go.Figure()
    all_months = np.arange(1, 13)
    variable_label = VARIABLE_LABELS.get(selected_variable, selected_variable)
    if not selected_points or not clim_error_dict:
        fig.update_layout(
            title=f'{variable_label} Difference Time Series for Selected Points',
            xaxis_title='Month',
            yaxis_title=variable_label,
            xaxis=dict(
                tickmode='linear',
                tick0=1,
                dtick=1,
                range=[0.5, 12.5],
                tickvals=all_months,
                ticktext=[str(m) for m in all_months]
            ),
            yaxis=dict(range=[scale['min'], scale['max']])
        )
        return fig
    clim_error = xr.Dataset.from_dict(clim_error_dict)['error']
    months = np.arange(1, clim_error.sizes['time'] + 1)
    for point in selected_points:
        ts = clim_error.sel(lat=point['lat'], lon=point['lon'], method='nearest')
        fig.add_trace(go.Scatter(
            x=months,
            y=ts.values,
            mode='lines+markers',
            name=f"({point['lat']:.2f}, {point['lon']:.2f})",
            line=dict(color=point['color']),
            marker=dict(color=point['color'])
        ))
    fig.update_layout(
        title=f'{variable_label} Difference Time Series for Selected Points',
        xaxis_title='Month',
        yaxis_title=variable_label,
        xaxis=dict(
            tickmode='linear',
            tick0=1,
            dtick=1,
            range=[0.5, 12.5],
            tickvals=all_months,
            ticktext=[str(m) for m in all_months]
        ),
        yaxis=dict(range=[scale['min'], scale['max']]),
        template='plotly_white'
    )
    return fig

if __name__ == '__main__':
    port = find_free_port()
    app.run(debug=True, port=port)