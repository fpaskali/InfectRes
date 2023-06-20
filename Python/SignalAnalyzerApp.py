import base64
import datetime
from datetime import date
import io
import json

import scipy.io as sio
import numpy as np
from scipy.signal import find_peaks, peak_widths
from scipy import sparse
from scipy.sparse.linalg import spsolve

import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash import dash_table as dt
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go

import pandas as pd

app = dash.Dash(__name__)
app.title = "InfectResonator"

######################################
# Sidebar modules
######################################

expInfo_module = html.Div([
                    dbc.Label("Select date"),
                    html.Br(),
                    dcc.DatePickerSingle(
                        id='date-picker',
                        display_format='DD/MM/YYYY',
                        date=date.today(),
                        clearable=True
                    ),
                    html.Hr(),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Experiment ID"),
                            html.Br(),
                            dbc.Input(
                                id="exp-id",
                                type="text",
                                value=1,
                                style={"width": "5vw"}),
                        ]),
                        dbc.Col([
                            dbc.Label("Particle Type"),
                            html.Br(),
                            dbc.Input(
                                id="par-type",
                                type="text",
                                style={"width": "5vw"}),
                        ]),
                    ])
                ])

sigLoad_module = html.Div([
                    dbc.Label("Choose signal data"),
                    dbc.Row([
                        dbc.Col(
                            dcc.Upload(id="upload-sig-1", 
                                children=[dbc.Button(id="up1", children="Load Signal 1")],
                                multiple=True)
                            ),
                        dbc.Col(
                            dcc.Upload(id="upload-sig-2", 
                                children=[dbc.Button(id="up2", children="Load Signal 2")],
                                multiple=True)
                            ),
                    ], justify="start"),
                    html.Br(),
                    dbc.Button(id="reset", children="Reset", outline=True, color="secondary"),
                    html.Div(id="msg"),
                    dcc.Store(id="signal-1-org", clear_data=False),
                    dcc.Store(id="signal-2-org", clear_data=False)
                ])

peakDet_module = html.Div([
                    dbc.Label("Number of peak pairs"),
                    dcc.Slider(
                        id="n_peaks",
                        min=1,
                        max=4,
                        step=1,
                        marks={i: str(i) for i in range(10)},
                        value=4),
                    html.Br(),
                    dbc.Label("Prominence"),
                    html.Br(),
                    dbc.Input(
                        id="prom",
                        type="number",
                        min=0.01,
                        max=1,
                        step=0.01,
                        value=0.04,
                        style={"width": "10vw"})
                ])

baseline_module = html.Div([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Lambda"),
                            html.Br(),
                            dbc.Label("Positive residuals")
                        ]),
                        dbc.Col([
                            dbc.Input(
                                id="baseline-lam",
                                type="number",
                                min=0,
                                max=10000000,
                                step=1,
                                value=1000,
                                style={"width": "10vw"}),
                            dbc.Input(
                                id="baseline-p",
                                type="number",
                                min=0,
                                max=1,
                                step=0.00005,
                                value=0.0001,
                                style={"width": "10vw"}),
                        ])
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([dbc.Button(id="apply_correction", children="Apply correction")]),
                        dbc.Col([dbc.Button(id="revert_correction", children="Revert correction")])
                    ])
                ])

keepPar_module = html.Div([
                    html.P("Keep particles"),
                    dbc.Card(id="par_sel",children="Particles not loaded.", body=True, style={"height": "12vh", "overflow":"scroll"}),
                    dcc.Store(id="signal-1-preprocessed", clear_data=False),
                    dcc.Store(id="signal-2-preprocessed", clear_data=False),
                ])

def limit_module(value_l=0, value_u=0, step=0,  disabled=True):
    return html.Div(id="limit", 
                    children=[
                        dbc.Label("Set plot limit"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Lower limit"),
                                dbc.Input(
                                    id="l_limit",
                                    type="number",
                                    min=value_l,
                                    value=value_l,
                                    step=step,
                                    disabled=disabled,
                                    style={"width": "5vw"})
                            ]),
                            dbc.Col([
                                dbc.Label("Upper limit"),
                                dbc.Input(
                                    id="u_limit",
                                    type="number",
                                    value=520,
                                    max=value_u,
                                    step=step,
                                    disabled=disabled,
                                    style={"width": "5vw"})
                            ])
                        ], justify="start")
                    ])

#####################################
# Sidebars
#####################################

tab1_sidebar = dbc.Card([
                    expInfo_module,
                    html.Hr(),
                    sigLoad_module,
                    html.Hr(),
                    baseline_module,
                ], body=True)

tab2_sidebar = dbc.Card([
                    html.Div([
                        keepPar_module,
                        html.Hr(),
                        limit_module(step=1),
                        html.Hr(),
                        peakDet_module,
                        html.Hr(),
                        dbc.Button(id="peakDet", children="Refresh plots"),
                        html.Hr(),
                        dbc.Checklist(
                                    id="check",
                                    options=[
                                        {"label":"peak positions", "value":"pos"},
                                        {"label":"peak shift", "value":"shift"},
                                        {"label":"halfwidth", "value":"hw"},
                                    ],
                                    value=["pos", "shift", "hw"],
                                    inline=False
                            ),
                        html.Hr(),
                        dbc.Row([
                            dbc.Col(dbc.Button(id="extract", children="Extract values")),
                            dbc.Col(dbc.Button(id="undo", children="Undo", disabled=True)),
                            dcc.Store(id="area-diff")
                        ])
                    ])
                ], body=True)

tab3_sidebar = dbc.Card([
                    html.Div([dbc.Button(id="btn_csv", children="Save Features"),
                                dcc.Download(id="download-intens-csv")])
                ], body=True)

#################################
# Main layout
#################################

app.layout = dbc.Container([
        html.H1("InfectResonator Signal Analyzer"),
        html.Hr(),
        html.Div(
            dbc.Tabs(id="tabs", active_tab="tab1",
                children=[
                    dbc.Tab(label="Baseline correction", tab_id="tab1",
                        children=[
                            html.Br(),
                            dbc.Row([
                                dbc.Col(tab1_sidebar, md=3),
                                dbc.Col(dbc.Card(html.Div(id="tab1-mainplot"), body=True, style={"height": "80vh", "overflow":"scroll"}), md=9)
                            ]), 
                        ]),
                dbc.Tab(label="Peak detection", tab_id="tab2",
                        children=[
                            html.Br(),
                            dbc.Row([
                                dbc.Col(tab2_sidebar, md=3),
                                dbc.Col(dbc.Card(html.Div(id="tab2-mainplot"), body=True, style={"height": "80vh", "overflow":"scroll"}), md=9)
                            ])
                        ]),
                dbc.Tab(
                    label="Features", tab_id="tab3",
                    children=[
                        html.Br(),
                        dbc.Row([
                            dbc.Col(tab3_sidebar, md=3),
                            dbc.Col([
                                dbc.Card([
                                    dbc.Label("Features data table"),
                                    html.Div(id="intens-table")], body=True, style={"height": "80vh", "overflow":"scroll"})
                            ], md=9),
                            dcc.Store(id="intens-data"),
                            dcc.Store(id="restore-data")
                        ])
                    ]
                ),
            ])
        )
    ],fluid=True)

########################################
# Helper functions
########################################

def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'nba' in filename:
            name = filename.split(".")[0]
            mat = sio.loadmat(io.BytesIO(decoded))
            df = pd.DataFrame({'par': [name for i in range(len(mat['data'][0]))],
                               'nm': mat['data'][0], 
                               'inten': mat['data'][1]})
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    
    return df

def scale_intensity(bb_inten, ab_inten):
    max_i, min_i = max(np.concatenate((bb_inten, ab_inten), axis=None)), min(np.concatenate((bb_inten, ab_inten), axis=None))
    bb_inten = (bb_inten - min_i) / (max_i - min_i)
    ab_inten = (ab_inten - min_i) / (max_i - min_i)

    return bb_inten, ab_inten

def generate_base_graph(bb_wl, bb_inten, ab_wl, ab_inten):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bb_wl, y=bb_inten, name="before binding"))
    fig.add_trace(go.Scatter(x=ab_wl, y=ab_inten, name="after binding"))

    return fig

def baseline_als(y, lam, p, niter=50):
    L = len(y)
    D = sparse.diags([1, -2, 1],[0,-1,-2], shape=(L,L-2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z

####################################
# Feature extraction functions
####################################

# def _measure_distances(df1, df2, n_peaks, prom):
#     bb_distances = {}
#     ab_distances = {}
#     bb_heights = {}
#     ab_heights = {}

#     particles = np.intersect1d(np.unique(df1["par"]), np.unique(df2["par"]))

#     for par in particles:
#         dff1 = df1[df1["par"]==par]
#         dff2 = df2[df2["par"]==par]

#         bb_inten = dff1.loc[:,"inten"].to_numpy()
#         ab_inten = dff2.loc[:,"inten"].to_numpy()

#         bb_wl = dff1.loc[:,"nm"].to_numpy()
#         ab_wl = dff1.loc[:,"nm"].to_numpy()

#         bb_inten, ab_inten = scale_intensity(dff1.loc[:,"inten"].to_numpy(), dff2.loc[:,"inten"].to_numpy())
        
#         left = 1
#         right = n_peaks
        
#         bb_peaks, _ = find_peaks(bb_inten, prominence=prom)
#         ab_peaks, _ = find_peaks(ab_inten, prominence=prom)

#         bb_half = peak_widths(bb_inten, bb_peaks, rel_height=0.5)
#         ab_half = peak_widths(ab_inten, ab_peaks, rel_height=0.5)

#         # Locate the highest peak in the before binding signal
#         bb_peak_idx = np.argmax(bb_inten[bb_peaks])

#         # # Find the wavelength of the highest peak
        
#         bb_hp = bb_wl[bb_peaks][bb_peak_idx]

#         if n_peaks <= len(ab_peaks) and n_peaks <= len(bb_peaks):
#             # Find the index of the closest peak in the after binding signal
#             ab_hp = min(ab_wl[ab_peaks], key=lambda x:abs(x-bb_hp))
#             ab_peak_idx = int(np.where(ab_wl[ab_peaks] == ab_hp)[0])

#             # Calculate distance between peaks
#             bb_peak_dis = bb_wl[bb_peaks][bb_peak_idx - left:bb_peak_idx + right]
#             ab_peak_dis = ab_wl[ab_peaks][ab_peak_idx - left:ab_peak_idx + right]

#             if len(bb_peak_dis) != len(ab_peak_dis):
#                 bb_distances[par] = [np.nan for i in range(n_peaks)]
#                 ab_distances[par] = [np.nan for i in range(n_peaks)]
#                 bb_heights[par] = [np.nan for i in range(n_peaks)]
#                 ab_heights[par] = [np.nan for i in range(n_peaks)]
#                 continue
            
#             bb_distances[par] = bb_peak_dis
#             ab_distances[par] = ab_peak_dis

#             # Calculate height difference
#             bb_peak_h = bb_inten[bb_peaks][bb_peak_idx - left:bb_peak_idx + right]
#             ab_peak_h = ab_inten[ab_peaks][ab_peak_idx - left:ab_peak_idx + right]

#             bb_heights[par] = bb_peak_h
#             ab_heights[par] = ab_peak_h

#             bb_halfwidths[par] = bb_half
#             ab_halfwidths[par] = ab_half
#         else:
#             bb_distances[par] = [np.nan for i in range(n_peaks)]
#             ab_distances[par] = [np.nan for i in range(n_peaks)]
#             bb_heights[par] = [np.nan for i in range(n_peaks)]
#             ab_heights[par] = [np.nan for i in range(n_peaks)]
#             bb_halfwidths[par] = [np.nan for i in range(n_peaks)]
#             ab_halfwidths[par] = [np.nan for i in range(n_peaks)]

#     return bb_distances, ab_distances, bb_heights, ab_heights, bb_halfwidths, ab_halfwidths

# def _measure_features(df1, df2):
#     avg_distance = {}
#     bb_sd = {}
#     ab_sd = {}
#     bb_iqr = {}
#     ab_iqr = {}
#     for par in np.intersect1d(np.unique(df1["par"]), np.unique(df2["par"])):
#         dff1 = df1[df1["par"]==par]
#         dff2 = df2[df2["par"]==par]

#         bb_inten = dff1.loc[:,"inten"].to_numpy()
#         ab_inten = dff2.loc[:,"inten"].to_numpy()

#         bb_wl = dff1.loc[:,"nm"].to_numpy()
#         ab_wl = dff1.loc[:,"nm"].to_numpy()

#         bb_inten, ab_inten = scale_intensity(dff1.loc[:,"inten"].to_numpy(), dff2.loc[:,"inten"].to_numpy())

#         avg_distance[par] = np.abs(np.mean(ab_inten - bb_inten))

#         bb_sd[par], ab_sd[par] = np.std(bb_inten), np.std(ab_inten)
        
#         q3, q1 = np.percentile(bb_inten, [75 ,25])
#         bb_iqr[par] = q3 - q1
#         q3, q1 = np.percentile(ab_inten, [75 ,25])
#         ab_iqr[par] = q3 - q1

#     return avg_distance, bb_sd, ab_sd, bb_iqr, ab_iqr

#####################################
# Server
#####################################

# Tab 1

@app.callback(Output("signal-1-org", "data"),
              Output("upload-sig-1", "children"),
              Output("upload-sig-1", "disabled"),
              Input('upload-sig-1', 'contents'),
              Input('upload-sig-1', 'filename'),
              Input('upload-sig-1', 'last_modified'))
def load_signal1(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        data = pd.DataFrame()
        for c, n, d in zip(list_of_contents, list_of_names, list_of_dates):
                data = pd.concat([data, parse_contents(c, n, d)], axis=0)
        return data.to_json(date_format='iso', orient='split'), dbc.Button(children="Signal 1 Loaded", color="primary"), True
    else:
        return None, dbc.Button(children="Load Signal 1", color="primary"), False

@app.callback(Output("signal-2-org", "data"),
              Output("upload-sig-2", "children"),
              Output("upload-sig-2", "disabled"),
              Input('upload-sig-2', 'contents'),
              Input('upload-sig-2', 'filename'),
              Input('upload-sig-2', 'last_modified'))
def load_signal2(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        data = pd.DataFrame()
        for c, n, d in zip(list_of_contents, list_of_names, list_of_dates):
                data = pd.concat([data, parse_contents(c, n, d)], axis=0)
        return data.to_json(date_format='iso', orient='split'), dbc.Button(children="Signal 2 Loaded", color="primary"), True
    else:
        return None, dbc.Button(children="Load Signal 2", color="primary"), False

@app.callback(Output("signal-2-org", "clear_data"),
              Output("signal-1-org", "clear_data"),
              Output("upload-sig-1", "contents"),
              Output("upload-sig-2", "contents"),
              Input("reset", "n_clicks"))
def reset_data(n_clicks):
    return True, True, None, None

@app.callback(Output("apply_correction", "n_clicks"),
              Input("revert_correction", "n_clicks"))
def revert_bc(n_clicks):
    return None

@app.callback(Output("tab1-mainplot", "children"),
            Output("par_sel", "children"),
            Output("signal-1-preprocessed", "data"),
            Output("signal-2-preprocessed", "data"),
            Output("limit", "children"),
            Input("signal-1-org", "data"),
            Input("signal-2-org", "data"),
            Input("baseline-lam", "value"),
            Input("baseline-p", "value"),
            Input("apply_correction", "n_clicks"),
            prevent_initial_call=True)
def render_plots_tab1(s1, s2, lam, p, n_clicks):
    if s1 == None or s2 == None:
        return "", "Signals not loaded.", None, None, limit_module()
    else:
        df1 = pd.read_json(s1, orient="split")
        df2 = pd.read_json(s2, orient="split")

        children=[]
        checklist=[]

        particles = np.intersect1d(np.unique(df1["par"]), np.unique(df2["par"]))

        checklist = dbc.Checklist(id="keep_par",
                                    options=[
                                        {"label":par, "value":par} for par in particles
                                    ],
                                    value=list(particles),
                                    inline=False)
        
        df1_pp = pd.DataFrame()
        df2_pp = pd.DataFrame()

        for par in particles:
            dff1 = df1[df1["par"]==par]
            dff2 = df2[df2["par"]==par]

            bb_inten = dff1.loc[:,"inten"].to_numpy()
            ab_inten = dff2.loc[:,"inten"].to_numpy()

            bb_wl = dff1.loc[:,"nm"].to_numpy()
            ab_wl = dff1.loc[:,"nm"].to_numpy()

            bb_inten, ab_inten = scale_intensity(dff1.loc[:,"inten"].to_numpy(), dff2.loc[:,"inten"].to_numpy())

            bb_baseline = baseline_als(bb_inten,lam, p)
            ab_baseline = baseline_als(ab_inten,lam, p)

            if n_clicks:
                bb_inten = bb_inten - bb_baseline
                ab_inten = ab_inten - ab_baseline   

            df1_pp = pd.concat([df1_pp,
                                pd.DataFrame({'par': [par for i in range(len(bb_wl))],
                                             'nm': bb_wl, 
                                             'inten': bb_inten})], axis=0)

            df2_pp = pd.concat([df2_pp,
                                pd.DataFrame({'par': [par for i in range(len(ab_wl))],
                                             'nm': ab_wl, 
                                             'inten': ab_inten})], axis=0)

            # Plotting
            fig = generate_base_graph(bb_wl, bb_inten, ab_wl, ab_inten)
            if not n_clicks:
                fig.add_trace(go.Scatter(x=bb_wl, y=bb_baseline, name="Signal 1 baseline"))
                fig.add_trace(go.Scatter(x=ab_wl, y=ab_baseline, name="Signal 2 baseline"))

            children.append(html.Div([html.H4(par),
                        dcc.Graph(figure=fig)]))

        limit_mod = limit_module(bb_wl[0], bb_wl[-1], 0.01, False)

        return children, checklist, df1_pp.to_json(date_format='iso', orient='split'), df2_pp.to_json(date_format='iso', orient='split'), limit_mod

# Tab 2

@app.callback(Output("tab2-mainplot", "children"),
            Output("intens-data", "data"),
            Output("restore-data", "data"),
            Input("peakDet", "n_clicks"),
            State("signal-1-preprocessed", "data"),
            State("signal-2-preprocessed", "data"),
            State("keep_par", "value"),
            State("n_peaks", "value"),
            State("l_limit", "value"),
            State("u_limit", "value"),
            State("prom","value"),
            State("date-picker", "date"),
            State("exp-id", "value"),
            State("par-type", "value"),
            State("check", "value"),
            State("intens-data", "data"),
            prevent_initial_call=True)
def render_plots_tab2(n_clicks, s1, s2, par_list, n_peaks, low_lim, upp_lim, prom, d, exp_id, par_type, checklist, current_d):
    if n_clicks == 0 or s1 == None or s2 == None:
        return "", None, None
    else:
        df1 = pd.read_json(s1, orient="split")
        df2 = pd.read_json(s2, orient="split")
        
        children = []

        df_dict = {
            "date": [],
            "parType": [],
            "expID": [],
            "parID": []
        }

        df_dict.update({f"bb_pos_{i+1}":[] for i in range(n_peaks*2)})
        df_dict.update({f"ab_pos_{i+1}":[] for i in range(n_peaks*2)})
        df_dict.update({f"shift_{i+1}":[] for i in range(n_peaks*2)})
        df_dict.update({f"bb_hw_{i+1}":[] for i in range(n_peaks*2)})
        df_dict.update({f"ab_hw_{i+1}":[] for i in range(n_peaks*2)})

        for par in sorted(par_list):
            dff1 = df1[df1["par"]==par]
            dff2 = df2[df2["par"]==par]

            bb_inten = dff1.loc[:,"inten"].to_numpy()
            ab_inten = dff2.loc[:,"inten"].to_numpy()

            bb_wl = dff1.loc[:,"nm"].to_numpy()
            ab_wl = dff1.loc[:,"nm"].to_numpy()

            # Limit plots
            lower = np.where(bb_wl > low_lim)
            upper = np.where(bb_wl > upp_lim)

            if low_lim < upp_lim and np.any(lower):
                low_idx = np.min(lower)
            else:
                low_idx = 0
            
            if low_lim < upp_lim and np.any(upper):
                upp_idx = np.min(upper)
            else:
                upp_idx = len(bb_wl)

            bb_inten = bb_inten[low_idx:upp_idx]
            ab_inten = ab_inten[low_idx:upp_idx]
            bb_wl = bb_wl[low_idx:upp_idx]
            ab_wl = ab_wl[low_idx:upp_idx]

            # Plotting
            fig = generate_base_graph(bb_wl, bb_inten, ab_wl, ab_inten)

            # Peak detection
            bb_peaks, _ = find_peaks(bb_inten, prominence=prom)
            ab_peaks, _ = find_peaks(ab_inten, prominence=prom)

            # Locate the highest peak in the before binding signal
            bb_peak_idx = np.argmax(bb_inten[bb_peaks])
            bb_hp = bb_wl[bb_peaks][bb_peak_idx]

            # Find the index of the closest peak in the after binding signal
            if np.any(bb_hp - ab_wl[ab_peaks]):
                ab_peak_idx = np.argmin(np.absolute(bb_hp - ab_wl[ab_peaks]))
            else:
                ab_peak_idx = None
            
            df_dict["date"].append(d)
            df_dict["parType"].append(par_type)
            df_dict["expID"].append(exp_id)
            df_dict["parID"].append(par)
            
            msg = ""
            # TODO change the code to avoid this nesting.
            if not np.all([ab_peak_idx, bb_peak_idx]):
                msg = "There was problem with the peak detection (Highest peak not detected!)"

                for i in range(n_peaks*2):
                    df_dict[f"bb_pos_{i+1}"].append(np.nan)
                    df_dict[f"ab_pos_{i+1}"].append(np.nan)
                    df_dict[f"bb_hw_{i+1}"].append(np.nan)
                    df_dict[f"ab_hw_{i+1}"].append(np.nan)
                    df_dict[f"shift_{i+1}"].append(np.nan)
            else:
                # Calculate the peaks wavelength
                bb_peaks_wl = bb_wl[bb_peaks][bb_peak_idx - 3:bb_peak_idx+(n_peaks*2-3)]
                ab_peaks_wl = ab_wl[ab_peaks][ab_peak_idx - 3:ab_peak_idx+(n_peaks*2-3)]
                if len(bb_peaks_wl) != len(ab_peaks_wl):
                    msg = "There was problem with the peak detection (The number of peaks in both signals not same)"

                    for i in range(n_peaks*2):
                        df_dict[f"bb_pos_{i+1}"].append(np.nan)
                        df_dict[f"ab_pos_{i+1}"].append(np.nan)
                        df_dict[f"bb_hw_{i+1}"].append(np.nan)
                        df_dict[f"ab_hw_{i+1}"].append(np.nan)
                        df_dict[f"shift_{i+1}"].append(np.nan)
                else:
                    # Calculate peaks height
                    bb_peaks_h = bb_inten[bb_peaks][bb_peak_idx - 3:bb_peak_idx+(n_peaks*2-3)]
                    ab_peaks_h = ab_inten[ab_peaks][ab_peak_idx - 3:ab_peak_idx+(n_peaks*2-3)]

                    bb_peaks_half = peak_widths(bb_inten, bb_peaks[bb_peak_idx - 3:bb_peak_idx+(n_peaks*2-3)])
                    ab_peaks_half = peak_widths(ab_inten, ab_peaks[ab_peak_idx - 3:ab_peak_idx+(n_peaks*2-3)])

                    bb_peaks_hw = [bb_peaks_h[i]/(bb_wl[int(bb_peaks_half[3][i])]-bb_wl[int(bb_peaks_half[2][i])]) for i in range(len(bb_peaks_half[0]))]
                    ab_peaks_hw = [ab_peaks_h[i]/(ab_wl[int(ab_peaks_half[3][i])]-ab_wl[int(ab_peaks_half[2][i])]) for i in range(len(ab_peaks_half[0]))]

                    diff = np.subtract(ab_peaks_wl, bb_peaks_wl)
                    
                    if len(bb_peaks_wl) < n_peaks*2:
                        bb_peaks_wl = np.append(bb_peaks_wl,[np.nan for i in range(n_peaks*2-len(bb_peaks_wl))])
                        ab_peaks_wl = np.append(ab_peaks_wl,[np.nan for i in range(n_peaks*2-len(ab_peaks_wl))])
                        bb_peaks_hw = np.append(bb_peaks_hw,[np.nan for i in range(n_peaks*2-len(bb_peaks_hw))])
                        ab_peaks_hw = np.append(ab_peaks_hw,[np.nan for i in range(n_peaks*2-len(ab_peaks_hw))])
                        diff = np.append(diff,[np.nan for i in range(8-len(diff))])

                    for i in range(n_peaks*2):
                        df_dict[f"bb_pos_{i+1}"].append(bb_peaks_wl[i])
                        df_dict[f"ab_pos_{i+1}"].append(ab_peaks_wl[i])
                        df_dict[f"bb_hw_{i+1}"].append(bb_peaks_hw[i])
                        df_dict[f"ab_hw_{i+1}"].append(ab_peaks_hw[i])
                        df_dict[f"shift_{i+1}"].append(diff[i])
                
                    fig.add_trace(go.Scatter(mode="markers",
                                            x=bb_peaks_wl, 
                                            y=bb_peaks_h,
                                            name="Peaks",
                                            marker=dict(size=12,
                                                        color="LightSkyBlue")))

                    
                    fig.add_trace(go.Scatter(mode="text",
                                                x=bb_peaks_wl + 0.3, # with offset
                                                y=bb_peaks_h,
                                                text=np.round(diff, decimals=2),
                                                name="Distances",
                                                textfont_color="blue",
                                                textposition="middle right"))
                    
                    bb_hw_x, bb_hw_y, bb_hw_ratio = [], [], []
                    ab_hw_x, ab_hw_y, ab_hw_ratio = [], [], []

                    for i in range(len(bb_peaks_half[1])):
                        bb_hw_x.extend([bb_wl[int(bb_peaks_half[2][i])], bb_wl[int(bb_peaks_half[3][i])], None])
                        bb_hw_y.extend([bb_peaks_half[1][i], bb_peaks_half[1][i], None])
                        ab_hw_x.extend([ab_wl[int(ab_peaks_half[2][i])], ab_wl[int(ab_peaks_half[3][i])], None])
                        ab_hw_y.extend([ab_peaks_half[1][i], ab_peaks_half[1][i], None])


                    fig.add_trace(go.Scatter(x=bb_hw_x, # with offset
                                            y=bb_hw_y,
                                            name="BB half-width"))

                    fig.add_trace(go.Scatter(mode="text",
                                                x=bb_hw_x[::3],
                                                y=bb_hw_y[::3],
                                                text=np.round(bb_peaks_hw, decimals=2),
                                                name="BB HW Ratio",
                                                textfont_color="blue",
                                                textposition="middle right"))

                    fig.add_trace(go.Scatter(x=ab_hw_x, # with offset
                                            y=ab_hw_y,
                                            name="AB half-width"))
                    
                    fig.add_trace(go.Scatter(mode="text",
                                                x=ab_hw_x[::3],
                                                y=ab_hw_y[::3],
                                                text=np.round(ab_peaks_hw, decimals=2),
                                                name="AB HW Ratio",
                                                textfont_color="blue",
                                                textposition="middle right"))

            children.append(html.Div([html.H4(par),
                                      dcc.Graph(figure=fig),
                                      html.H6(msg)]))

        df = pd.DataFrame(df_dict)

        if current_d != None:
            cd = pd.read_json(current_d, orient="split", convert_dates=False)
            df = pd.concat([cd, df], ignore_index=True)

        return children, df.to_json(date_format='iso', orient='split'), current_d

# Tab 3

@app.callback(Output("intens-table", "children"),
              Output("undo", "disabled"),
              Input("extract", "n_clicks"),
              Input("undo", "n_clicks"),
              State("intens-data", "data"),
              State("restore-data", "data"),
              State("undo", "disabled"),
              State("check", "value"),
              prevent_initial_call=True)
def render_dataframe(extract, restore, current_d, old_d, restore_disabled, checklist):
    #TODO implement restore
    df = pd.read_json(current_d, orient='split', convert_dates=False)
    df = df.filter(["date","parType","expID","parID"]+[x for x in df.columns for w in checklist if w in x])

    if current_d != None:
        return dbc.Table.from_dataframe(df, bordered=True, striped=True, responsive=True), True
    
    return None, True
        
@app.callback(
    Output("download-intens-csv", "data"),
    Input("btn_csv", "n_clicks"),
    State("intens-data", "data"),
    State("check", "value"),
    prevent_initial_call=True)
def func(n_clicks, current_d, checklist):

    df = pd.read_json(current_d, orient='split', convert_dates=False)
    df = df.filter(["date","parType","expID","parID"]+[x for x in df.columns for w in checklist if w in x])
    return dcc.send_data_frame(df.to_csv, "featuresData.csv", index=False)

if __name__ == '__main__':
    app.run_server(debug=False, port=8040)