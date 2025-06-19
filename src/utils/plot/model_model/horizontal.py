import os
import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objs as go
import plotly.express as px
import xarray as xr
import numpy as np
import contextlib
from utils.plot.general import find_free_port
from utils.data.general import fill_coastal_points_in_time

# ---- Configuration ----
DATA_DIR = "~/NEMOCheck/data/processed"
DATA_DIR = os.path.expanduser(DATA_DIR)
MESH_PATH = "~/NEMOCheck/data/model/orca05l75_domain_cfg_nemov5_10m.nc"
AVAILABLE_VARIABLES = ['tos', 'sos', 'zos', 'mldkz5', 'mldr10_1', 'tob', 'sob', 'taubot']
COLOR_CYCLE = px.colors.qualitative.Plotly
initial_scale = {'min': -4, 'max': 4}
initial_month = 1  # 1-based for slider display; see below for 0-based indexing

# ---- Load mesh and land mask ONCE ----
mesh = xr.open_dataset(MESH_PATH)
if "bathy_metry" not in mesh:
    raise RuntimeError("Mesh file must contain 'bathy_metry'")
land_mask = mesh['bathy_metry'] == 0

# ---- Grid for interpolation ----
target_grid = xr.Dataset({
    'lon': (['lon'], np.arange(-179.5, 180, 1)),
    'lat': (['lat'], np.arange(-89.5, 90, 1))
})

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("Interactive Model Difference Viewer (Regridded to Regular Lon/Lat)"),

    # Controls
    html.Div([
        html.Label("Select first model file:"),
        dcc.Dropdown(
            id='model1-dropdown',
            options=[{'label': fn, 'value': fn} for fn in sorted(os.listdir(DATA_DIR)) if fn.endswith('.nc')],
            placeholder='Choose model 1'
        ),
        html.Label("Select second model file:"),
        dcc.Dropdown(
            id='model2-dropdown',
            options=[{'label': fn, 'value': fn} for fn in sorted(os.listdir(DATA_DIR)) if fn.endswith('.nc')],
            placeholder='Choose model 2'
        ),
        html.Label("Select variable:"),
        dcc.Dropdown(
            id='variable-dropdown',
            options=[{'label': v, 'value': v} for v in AVAILABLE_VARIABLES],
            placeholder='Choose variable'
        ),
        html.Button("Start", id="start-btn", n_clicks=0, style={'marginTop': '16px', 'width': '100%'})
    ], style={'width': '40%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginBottom': '20px'}),

    html.Div([
        html.Label("Set colorbar scale:"),
        dcc.Input(id='scale-min', type='number', value=initial_scale['min'], step=0.1, style={'width': '80px'}),
        dcc.Input(id='scale-max', type='number', value=initial_scale['max'], step=0.1, style={'width': '80px'}),
        html.Button("Update Scale", id='update-scale-btn', n_clicks=0),
    ], style={'marginBottom': '20px'}),

    html.Div([
        html.Label("Select Month:"),
        dcc.Slider(
            id='month-slider',
            min=1,
            max=12,
            step=1,
            value=initial_month,
            marks={i: str(i) for i in range(1, 13)},
        )
    ], style={'width': '40%', 'marginBottom': '20px'}),

    dcc.Store(id='clim-error-store'),  # will store dict of regridded data
    dcc.Store(id='selected-points-store', data=[]),
    dcc.Store(id='current-scale', data=initial_scale),

    dcc.Loading(
        id="loading-global",
        type="circle",
        children=html.Div(id="loading-dummy", style={"display": "none"})
    ),

   html.Div([
        html.Div([
            dcc.Graph(
                id='ts-plot',
                style={'height': '500px'}
            )
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
    
        html.Div([
            dcc.Graph(
                id='map-plot',
                style={'height': '500px'}
            ),
            html.Button('Reset All Points', id='reset-all-btn', n_clicks=0, style={'marginTop': '10px'}),
            html.Div(id='legend-container', style={
                'padding': '10px',
                'border': '1px solid lightgray',
                'marginTop': '10px',
                'backgroundColor': '#f9f9f9',
                'fontFamily': 'monospace',
                'fontSize': '13px',
                'maxHeight': '200px',
                'overflowY': 'auto'
            }),
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
    ])
])

# ---- Callbacks ----

@app.callback(
    Output('current-scale', 'data'),
    Input('update-scale-btn', 'n_clicks'),
    State('scale-min', 'value'),
    State('scale-max', 'value')
)
def update_scale(n_clicks, vmin, vmax):
    if vmin is not None and vmax is not None and vmin < vmax:
        return {'min': vmin, 'max': vmax}
    return dash.no_update

@app.callback(
    [Output('clim-error-store', 'data'),
     Output('loading-dummy', 'children')],
    [Input('start-btn', 'n_clicks')],
    [State('model1-dropdown', 'value'),
     State('model2-dropdown', 'value'),
     State('variable-dropdown', 'value')]
)
def compute_clim_error(n_clicks, file1, file2, var):
    if n_clicks == 0:
        return dash.no_update, ""
    if file1 and file2 and var:
        path1 = os.path.join(DATA_DIR, file1)
        path2 = os.path.join(DATA_DIR, file2)
        try:
            ds1 = xr.open_dataset(path1)
            ds2 = xr.open_dataset(path2)
            arr1 = ds1[var]
            arr2 = ds2[var]

            # Use mesh-provided land mask
            arr1 = arr1.where(~land_mask)
            arr2 = arr2.where(~land_mask)

            # Compute difference on the original grid
            error = arr1 - arr2
            error = error.rename({
                'nav_lon': 'lon',
                'nav_lat': 'lat',
                'time_counter': 'time'
            })
            error = error.assign_coords(lon=mesh['glamt'], lat=mesh['gphit'])

            # Fill coasts for interpolation (optional, but often recommended)
            error_filled = fill_coastal_points_in_time(error, 20)

            # Interpolate difference to regular grid (on time_counter axis)
            import xesmf as xe
            with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
                regridder = xe.Regridder(
                    error_filled, target_grid, 
                    method='bilinear', 
                    filename='weights_bilinear_pmmh.nc',
                    reuse_weights=False, 
                    ignore_degenerate=True, 
                    periodic=True
                )
            error_regridded = regridder(error)

            # Store as dict (no print, no return of xarray object!)
            return error_regridded.to_dataset(name='error').to_dict(), ""
        except Exception as e:
            # return empty, no error printing
            return dash.no_update, ""
    return dash.no_update, ""

@app.callback(
    Output('map-plot', 'figure'),
    Input('clim-error-store', 'data'),
    Input('current-scale', 'data'),
    Input('selected-points-store', 'data'),
    Input('month-slider', 'value'),
    Input('map-plot', 'relayoutData')
)
def update_map(clim_error_dict, scale, selected_points, month, relayout_data):
    if not clim_error_dict:
        return go.Figure()
    clim_error = xr.Dataset.from_dict(clim_error_dict)['error']
    # Use .isel(time=month-1) since your time coordinate is named "time"
    da = clim_error.isel(time=month-1)  # (lat, lon)
    lons = da['lon'].values
    lats = da['lat'].values
    z = da.values  # shape (lat, lon)
    fig = go.Figure(go.Heatmap(
        z=z,
        x=lons,
        y=lats,
        colorscale='RdBu_r',
        zmin=scale['min'],
        zmax=scale['max'],
        colorbar=dict(title='Δ (model1 - model2)')
    ))

    # Overlay selected points
    for i, pt in enumerate(selected_points or []):
        fig.add_trace(go.Scatter(
            x=[pt['lon']],
            y=[pt['lat']],
            mode='markers',
            marker=dict(color=pt['color'], size=12, line=dict(color='black', width=1)),
            name=f"Selected ({pt['lat']:.2f}, {pt['lon']:.2f})",
            hoverinfo='text',
            hovertext=f"Point {i+1}: ({pt['lat']:.2f}, {pt['lon']:.2f})"
        ))

    fig.update_layout(
        showlegend=False,
        title=f'Model Difference - Month {month}',
        xaxis_title='Longitude',
        yaxis_title='Latitude',
        dragmode='pan',
        plot_bgcolor='white',
        paper_bgcolor='white',
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

    # Reset all points
    if trigger_id == 'reset-all-btn' and reset_clicks:
        return []

    # Remove one point
    if trigger_id.startswith('{'):
        import json
        triggered_obj = json.loads(trigger_id)
        idx_to_remove = triggered_obj['index']
        if 0 <= idx_to_remove < len(selected_points):
            selected_points.pop(idx_to_remove)
        return selected_points

    # Add point on map click
    if trigger_id == 'map-plot' and clickData:
        lon = clickData['points'][0]['x']
        lat = clickData['points'][0]['y']
        tol = 0.1
        for pt in selected_points:
            if abs(pt['lat'] - lat) < tol and abs(pt['lon'] - lon) < tol:
                return selected_points  # already selected
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
        return "No points selected."
    legend_items = []
    for i, pt in enumerate(selected_points):
        legend_items.append(
            html.Div([
                html.Span("⬤ ", style={'color': pt['color'], 'fontSize': '16px'}),
                f"({pt['lat']:.2f}, {pt['lon']:.2f})",
                html.Button('×', id={'type': 'remove-point-btn', 'index': i}, n_clicks=0,
                            style={
                                'marginLeft': '8px',
                                'color': 'red',
                                'background': 'none',
                                'border': 'none',
                                'cursor': 'pointer',
                                'fontWeight': 'bold',
                                'fontSize': '16px',
                                'lineHeight': '16px',
                                'padding': '0',
                            })
            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '6px'})
        )
    return legend_items

@app.callback(
    Output('ts-plot', 'figure'),
    Input('selected-points-store', 'data'),
    Input('clim-error-store', 'data'),
    Input('current-scale', 'data')
)
def update_timeseries(selected_points, clim_error_dict, scale):
    fig = go.Figure()
    if not selected_points or not clim_error_dict:
        fig.update_layout(
            title='Select points on the map to see their time series',
            xaxis_title='Month',
            yaxis_title='Δ Value',
            yaxis=dict(range=[scale['min'], scale['max']])
        )
        return fig

    clim_error = xr.Dataset.from_dict(clim_error_dict)['error']
    months = np.arange(1, clim_error.sizes['time'] + 1)
    for pt in selected_points:
        # Find nearest grid point
        ts = clim_error.sel(lat=pt['lat'], lon=pt['lon'], method='nearest')
        fig.add_trace(go.Scatter(
            x=months,
            y=ts.values,
            mode='lines+markers',
            name=f"({pt['lat']:.2f}, {pt['lon']:.2f})",
            line=dict(color=pt['color']),
            marker=dict(color=pt['color'])
        ))

    fig.update_layout(
        title='Difference Time Series for Selected Points',
        xaxis_title='Month',
        yaxis_title='Δ Value',
        xaxis=dict(tickmode='linear', tick0=1, dtick=1),
        yaxis=dict(range=[scale['min'], scale['max']]),
        template='plotly_white'
    )
    return fig

if __name__ == '__main__':
    port = find_free_port()
    app.run(debug=True, port=port)