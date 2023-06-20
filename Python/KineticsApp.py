import base64
import datetime
from datetime import date
import io, json

import scipy.io as sio
import numpy as np
from scipy.signal import find_peaks
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
app.title = "InfectResonator Kinetics"

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
                    dbc.Label("Choose kinetics data"),
                    dbc.Row([
                        dbc.Col(
                            dcc.Upload(id="upload-sig-1", 
                                children=[dbc.Button(id="up1", children="Load Kinetics Data")],
                                multiple=False)
                            )
                    ], justify="start"),
                    html.Br(),
                    dbc.Button(id="reset", children="Reset", outline=True, color="secondary"),
                    html.Div(id="msg"),
                    dcc.Store(id="signal-1-org", clear_data=False),
                    dcc.Store(id="signal-1-preprocessed", clear_data=False),
                ])

peakDet_module = html.Div([
                    dbc.Label("Number of peaks"),
                    dcc.Slider(
                        id="n_peaks",
                        min=1,
                        max=8,
                        step=1,
                        marks={i: str(i) for i in range(8)},
                        value=8),
                    html.Br(),
                    dbc.Label("Prominence"),
                    html.Br(),
                    dbc.Input(
                        id="prom",
                        type="number",
                        min=0.01,
                        max=1,
                        step=0.01,
                        value=0.25,
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
                    dbc.Card(id="par_sel",children="Particles not loaded.", body=True, style={"height": "12vh", "overflow":"scroll"})
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
                ], body=True)

tab2_sidebar = dbc.Card([
                    html.Div([
                        limit_module(),
                        html.Hr(),
                        dbc.Button(id="peakDet", children="Refresh plots"),
                        html.Hr(),
                        peakDet_module,
                        html.Hr(),
                        dbc.Row([
                            dbc.Col(dbc.Button(id="plotKinetics", children="Plot Kinetics")),
                            dcc.Store(id="peak-kinetics")
                        ])
                    ])
                ], body=True)

tab3_sidebar = dbc.Card([
                    html.H3("Kinetics Visualization")
                ], body=True)

#################################
# Main layout
#################################

app.layout = dbc.Container([
        html.H1("InfectResonator Kinetics Analyzer"),
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
                    label="Kinetics", tab_id="tab3",
                    children=[
                        html.Br(),
                        dbc.Row([
                            dbc.Col(tab3_sidebar, md=3),
                            dbc.Col(dbc.Card(dbc.Row(id="tab3-mainplot"), body=True, style={"height": "80vh", "overflow":"scroll"}), md=9),
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
            kinetics_dict = {'wl': mat['data'][0]}
            kinetics_dict.update({f"frame{i}": mat['data'][i] for i in range(1,len(mat['data']))})
            df = pd.DataFrame(kinetics_dict)
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    
    return df

def scale_intensity(spectra):
    max_i, min_i = max(spectra), min(spectra)
    spectra = (spectra - min_i) / (max_i - min_i)

    return spectra

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

def _measure_distances(df1, df2, n_peaks, prom):
    bb_distances = {}
    ab_distances = {}
    bb_heights = {}
    ab_heights = {}

    particles = np.intersect1d(np.unique(df1["par"]), np.unique(df2["par"]))

    for par in particles:
        dff1 = df1[df1["par"]==par]
        dff2 = df2[df2["par"]==par]

        bb_inten = dff1.loc[:,"inten"].to_numpy()
        ab_inten = dff2.loc[:,"inten"].to_numpy()

        bb_wl = dff1.loc[:,"nm"].to_numpy()
        ab_wl = dff1.loc[:,"nm"].to_numpy()

        bb_inten, ab_inten = scale_intensity(dff1.loc[:,"inten"].to_numpy(), dff2.loc[:,"inten"].to_numpy())
        
        left = 1
        right = n_peaks
        
        bb_peaks, _ = find_peaks(bb_inten, prominence=prom)
        ab_peaks, _ = find_peaks(ab_inten, prominence=prom)

        # Locate the highest peak in the before binding signal
        bb_peak_idx = np.argmax(bb_inten[bb_peaks])

        # # Find the wavelength of the highest peak
        
        bb_hp = bb_wl[bb_peaks][bb_peak_idx]

        if n_peaks <= len(ab_peaks) and n_peaks <= len(bb_peaks):
            # Find the index of the closest peak in the after binding signal
            ab_hp = min(ab_wl[ab_peaks], key=lambda x:abs(x-bb_hp))
            ab_peak_idx = int(np.where(ab_wl[ab_peaks] == ab_hp)[0])

            # Calculate distance between peaks
            bb_peak_dis = bb_wl[bb_peaks][bb_peak_idx - left:bb_peak_idx + right]
            ab_peak_dis = ab_wl[ab_peaks][ab_peak_idx - left:ab_peak_idx + right]

            if len(bb_peak_dis) != len(ab_peak_dis):
                bb_distances[par] = [np.nan for i in range(n_peaks)]
                ab_distances[par] = [np.nan for i in range(n_peaks)]
                bb_heights[par] = [np.nan for i in range(n_peaks)]
                ab_heights[par] = [np.nan for i in range(n_peaks)]
                continue
            
            bb_distances[par] = bb_peak_dis
            ab_distances[par] = ab_peak_dis

            # Calculate height difference
            bb_peak_h = bb_inten[bb_peaks][bb_peak_idx - left:bb_peak_idx + right]
            ab_peak_h = ab_inten[ab_peaks][ab_peak_idx - left:ab_peak_idx + right]

            bb_heights[par] = bb_peak_h
            ab_heights[par] = ab_peak_h
        else:
            bb_distances[par] = [np.nan for i in range(n_peaks)]
            ab_distances[par] = [np.nan for i in range(n_peaks)]
            bb_heights[par] = [np.nan for i in range(n_peaks)]
            ab_heights[par] = [np.nan for i in range(n_peaks)]

    return bb_distances, ab_distances, bb_heights, ab_heights

def _measure_features(df1, df2):
    avg_distance = {}
    bb_sd = {}
    ab_sd = {}
    bb_iqr = {}
    ab_iqr = {}
    for par in np.intersect1d(np.unique(df1["par"]), np.unique(df2["par"])):
        dff1 = df1[df1["par"]==par]
        dff2 = df2[df2["par"]==par]

        bb_inten = dff1.loc[:,"inten"].to_numpy()
        ab_inten = dff2.loc[:,"inten"].to_numpy()

        bb_wl = dff1.loc[:,"nm"].to_numpy()
        ab_wl = dff1.loc[:,"nm"].to_numpy()

        bb_inten, ab_inten = scale_intensity(dff1.loc[:,"inten"].to_numpy(), dff2.loc[:,"inten"].to_numpy())

        avg_distance[par] = np.abs(np.mean(ab_inten - bb_inten))

        bb_sd[par], ab_sd[par] = np.std(bb_inten), np.std(ab_inten)
        
        q3, q1 = np.percentile(bb_inten, [75 ,25])
        bb_iqr[par] = q3 - q1
        q3, q1 = np.percentile(ab_inten, [75 ,25])
        ab_iqr[par] = q3 - q1

    return avg_distance, bb_sd, ab_sd, bb_iqr, ab_iqr

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
def load_signal1(c, n, d):
    if c is not None:
        data = parse_contents(c, n, d)
        return data.to_json(date_format='iso', orient='split'), dbc.Button(children="Kinetics Data Loaded", color="primary"), True
    else:
        return None, dbc.Button(children="Load Kinetics", color="primary"), False

@app.callback(Output("signal-1-org", "clear_data"),
              Output("upload-sig-1", "contents"),
              Input("reset", "n_clicks"))
def reset_data(n_clicks):
    return True, None

@app.callback(Output("tab1-mainplot", "children"),
            Output("signal-1-preprocessed", "data"),
            Output("limit", "children"),
            Input("signal-1-org", "data"),
            prevent_initial_call=True)
def render_plots_tab1(s1):
    if s1 == None:
        return "", None, limit_module()
    else:
        df1 = pd.read_json(s1, orient="split")

        wl = df1.iloc[:,0].to_numpy()

        normalized_data = []

        for i in range(1, len(df1.columns)-2):
            normalized_data.append(scale_intensity(df1.iloc[:,i].to_numpy()))

        df1_pp_dict = {"wl": wl}
        df1_pp_dict.update({f"frame{i}": normalized_data[i] for i in range(len(normalized_data))})

        df1_pp = pd.DataFrame(df1_pp_dict)


        # Plotting
        fig = go.Figure()

        for i in range(len(normalized_data)):
            fig.add_trace(go.Scatter(x=wl, y=normalized_data[i]))

        limit_mod = limit_module(wl[0], wl[-1], 0.01, False)

        return dcc.Graph(figure=fig, style={'height':'70vh'}), df1_pp.to_json(date_format='iso', orient='split'), limit_mod

@app.callback(Output("tab2-mainplot", "children"),
            Output("peak-kinetics", "data"),
            Input("peakDet", "n_clicks"),
            State("signal-1-preprocessed", "data"),
            State("l_limit", "value"),
            State("u_limit", "value"),
            State("n_peaks", "value"),
            State("prom", "value"),
            prevent_initial_call=True)
def render_plots_tab2(n_clicks, s1, low_lim, upp_lim, n_peaks, prom):
    if s1 == None:
        return ""
    else:
        df1 = pd.read_json(s1, orient="split")

        wl = df1.iloc[:,0].to_numpy()
        intens = df1.iloc[:,1:]

        # Limit plots
        lower = np.where(wl > low_lim)
        upper = np.where(wl > upp_lim)

        if low_lim < upp_lim and np.any(lower):
            low_idx = np.min(lower)
        else:
            low_idx = 0
        
        if low_lim < upp_lim and np.any(upper):
            upp_idx = np.min(upper)
        else:
            upp_idx = len(wl)

        peaks = []

        # Peak detection
        for i in range(len(intens.columns)):
            p, _ = find_peaks(intens.iloc[:,i].to_numpy()[low_idx:upp_idx], prominence=prom)
            peaks.append(p)
        # Plotting
        fig = go.Figure()
        
        peaks_wl = {}

        for i in range(len(intens.columns)):
            fig.add_trace(go.Scatter(x=wl[low_idx:upp_idx], y=intens.iloc[:,i].to_numpy()[low_idx:upp_idx]))
        for idx in range(n_peaks):
            p_wl = [wl[low_idx:upp_idx][peaks[i][idx]] for i in range(len(peaks))]
            peaks_wl.update({f"p_{idx}":p_wl})
            fig.add_trace(go.Scatter(mode="markers",
                                    x=p_wl, 
                                    y=[intens.iloc[:,i].to_numpy()[low_idx:upp_idx][peaks[i][idx]] for i in range(len(peaks))],
                                    name="Peaks",
                                    marker=dict(size=12,
                                    color=px.colors.qualitative.Plotly[idx])))

        kinetics_df = pd.DataFrame(peaks_wl)

        limit_mod = limit_module(wl[0], wl[-1], 0.01, False)

        return dcc.Graph(figure=fig, style={'height':'70vh'}), kinetics_df.to_json(date_format='iso', orient='split')

@app.callback(Output("tab3-mainplot", "children"),
            Input("plotKinetics", "n_clicks"),
            State("peak-kinetics", "data"),
            prevent_initial_call=True)
def render_plots_tab3(n_clicks, kinetics):
    df = pd.read_json(kinetics, orient="split")

    children = []

    # Plotting
    for i in range(len(df.columns)):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[*range(len(df.iloc[:,i]))],
                                y=df.iloc[:,i],
                                name=f"Peak {i}",
                                marker=dict(size=12,
                                color=px.colors.qualitative.Plotly[i])))
        children.append(dbc.Col([
                            html.H4(f"Peak {i+1}"), 
                            dcc.Graph(figure=fig, style={'height':'50vh', 'width':'35vw'})
                            ])
                        )

    return children


# @app.callback(Output("tab2-mainplot", "children"),
#             Output("intens-data", "data"),
#             Output("restore-data", "data"),
#             Input("peakDet", "n_clicks"),
#             State("signal-1-preprocessed", "data"),
#             State("signal-2-preprocessed", "data"),
#             State("keep_par", "value"),
#             State("l_limit", "value"),
#             State("u_limit", "value"),
#             State("prom","value"),
#             State("date-picker", "date"),
#             State("exp-id", "value"),
#             State("par-type", "value"),
#             State("intens-data", "data"),
#             prevent_initial_call=True)
# def render_plots_tab2(n_clicks, s1, s2, par_list, low_lim, upp_lim, prom, d, exp_id, par_type, current_d):
#     if n_clicks == 0 or s1 == None or s2 == None:
#         return "", None, None
#     else:
#         df1 = pd.read_json(s1, orient="split")
#         df2 = pd.read_json(s2, orient="split")
        
#         children = []

#         df_dict = {
#             "date": [],
#             "parType": [],
#             "expID": [],
#             "parID": []
#         }

#         df_dict.update({f"bb_peak_{i+1}":[] for i in range(8)})
#         df_dict.update({f"ab_peak_{i+1}":[] for i in range(8)})
#         df_dict.update({f"shift_{i+1}":[] for i in range(8)})

#         for par in sorted(par_list):
#             dff1 = df1[df1["par"]==par]
#             dff2 = df2[df2["par"]==par]

#             bb_inten = dff1.loc[:,"inten"].to_numpy()
#             ab_inten = dff2.loc[:,"inten"].to_numpy()

#             bb_wl = dff1.loc[:,"nm"].to_numpy()
#             ab_wl = dff1.loc[:,"nm"].to_numpy()

#             # Limit plots
#             lower = np.where(bb_wl > low_lim)
#             upper = np.where(bb_wl > upp_lim)

#             if low_lim < upp_lim and np.any(lower):
#                 low_idx = np.min(lower)
#             else:
#                 low_idx = 0
            
#             if low_lim < upp_lim and np.any(upper):
#                 upp_idx = np.min(upper)
#             else:
#                 upp_idx = len(bb_wl)

#             bb_inten = bb_inten[low_idx:upp_idx]
#             ab_inten = ab_inten[low_idx:upp_idx]
#             bb_wl = bb_wl[low_idx:upp_idx]
#             ab_wl = ab_wl[low_idx:upp_idx]

#             # Plotting
#             fig = generate_base_graph(bb_wl, bb_inten, ab_wl, ab_inten)

#             # Peak detection
#             bb_peaks, _ = find_peaks(bb_inten, prominence=prom)
#             ab_peaks, _ = find_peaks(ab_inten, prominence=prom)

#             # Locate the highest peak in the before binding signal
#             bb_peak_idx = np.argmax(bb_inten[bb_peaks])
#             bb_hp = bb_wl[bb_peaks][bb_peak_idx]

#             # Find the index of the closest peak in the after binding signal
#             if np.any(bb_hp - ab_wl[ab_peaks]):
#                 ab_peak_idx = np.argmin(np.absolute(bb_hp - ab_wl[ab_peaks]))
#             else:
#                 ab_peak_idx = None
            
#             df_dict["date"].append(d)
#             df_dict["parType"].append(par_type)
#             df_dict["expID"].append(exp_id)
#             df_dict["parID"].append(par)
            
#             msg = ""
#             # TODO change the code to avoid this nesting.
#             if not np.all([ab_peak_idx, bb_peak_idx]):
#                 msg = "There was problem with the peak detection (Highest peak not detected!)"

#                 bb_peaks_wl = [np.nan for i in range(8)]
#                 ab_peaks_wl = [np.nan for i in range(8)]
#                 diff = [np.nan for i in range(8)]

#                 for i in range(len(bb_peaks_wl)):
#                     df_dict[f"bb_peak_{i+1}"].append(bb_peaks_wl[i])
#                     df_dict[f"ab_peak_{i+1}"].append(ab_peaks_wl[i])
#                     df_dict[f"shift_{i+1}"].append(diff[i])
#             else:
#                 # Calculate the peaks wavelength
#                 bb_peaks_wl = bb_wl[bb_peaks][bb_peak_idx - 3:bb_peak_idx + 5]
#                 ab_peaks_wl = ab_wl[ab_peaks][ab_peak_idx - 3:ab_peak_idx + 5]
#                 if len(bb_peaks_wl) != len(ab_peaks_wl):
#                     msg = "There was problem with the peak detection (The number of peaks in both signals not same)"

#                     bb_peaks_wl = [np.nan for i in range(8)]
#                     ab_peaks_wl = [np.nan for i in range(8)]
#                     diff = [np.nan for i in range(8)]

#                     for i in range(len(bb_peaks_wl)):
#                         df_dict[f"bb_peak_{i+1}"].append(bb_peaks_wl[i])
#                         df_dict[f"ab_peak_{i+1}"].append(ab_peaks_wl[i])
#                         df_dict[f"shift_{i+1}"].append(diff[i])
#                 else:
#                     # Calculate peaks height
#                     bb_peaks_h = bb_inten[bb_peaks][bb_peak_idx - 3:bb_peak_idx + 5]
#                     ab_peaks_h = ab_inten[ab_peaks][ab_peak_idx - 3:ab_peak_idx + 5]
                    
#                     diff = np.subtract(ab_peaks_wl, bb_peaks_wl)
                    
#                     if len(bb_peaks_wl) < 8:
#                         bb_peaks_wl = np.append(bb_peaks_wl,[np.nan for i in range(8-len(bb_peaks_wl))])
#                         ab_peaks_wl = np.append(ab_peaks_wl,[np.nan for i in range(8-len(ab_peaks_wl))])
#                         diff = np.append(diff,[np.nan for i in range(8-len(diff))])

#                     for i in range(len(bb_peaks_wl)):
#                         df_dict[f"bb_peak_{i+1}"].append(bb_peaks_wl[i])
#                         df_dict[f"ab_peak_{i+1}"].append(ab_peaks_wl[i])
#                         df_dict[f"shift_{i+1}"].append(diff[i])
                
#                     fig.add_trace(go.Scatter(mode="markers",
#                                             x=bb_peaks_wl, 
#                                             y=bb_peaks_h,
#                                             name="Peaks",
#                                             marker=dict(size=12,
#                                                         color="LightSkyBlue")))

                    
#                     fig.add_trace(go.Scatter(mode="text",
#                                                 x=bb_peaks_wl + 0.3, # with offset
#                                                 y=bb_peaks_h,
#                                                 text=np.round(diff, decimals=2),
#                                                 name="Distances",
#                                                 textfont_color="blue",
#                                                 textposition="middle right"))


#             children.append(html.Div([html.H4(par),
#                                       dcc.Graph(figure=fig),
#                                       html.H6(msg)]))

#         df = pd.DataFrame(df_dict)

#         if current_d != None:
#             cd = pd.read_json(current_d, orient="split", convert_dates=False)
#             df = pd.concat([cd, df], ignore_index=True)

#         return children, df.to_json(date_format='iso', orient='split'), current_d


# @app.callback(Output("graphs", "children"),
#             Input("signal-1-org", "data"),
#             Input("signal-2-org", "data"),
#             Input("n_peaks", "value"),
#             Input("prom", "value"))
# def render_graphs(s1, s2, n_peaks, prom):
#     if s1 != None and s2 != None:
#         df1 = pd.read_json(s1, orient="split")
#         df2 = pd.read_json(s2, orient="split")
#         children = []
#         for par in np.intersect1d(np.unique(df1["par"]), np.unique(df2["par"])):
#             dff1 = df1[df1["par"]==par]
#             dff2 = df2[df2["par"]==par]

#             bb_inten = dff1.loc[:,"inten"].to_numpy()
#             ab_inten = dff2.loc[:,"inten"].to_numpy()

#             bb_wl = dff1.loc[:,"nm"].to_numpy()
#             ab_wl = dff1.loc[:,"nm"].to_numpy()

#             bb_inten, ab_inten = scale_intensity(dff1.loc[:,"inten"].to_numpy(), dff2.loc[:,"inten"].to_numpy())

#             fig = generate_base_graph(bb_wl, bb_inten, ab_wl, ab_inten)

#             # Peak detection
#             left = 1
#             right = n_peaks
            
#             bb_peaks, _ = find_peaks(bb_inten, prominence=prom)
#             ab_peaks, _ = find_peaks(ab_inten, prominence=prom)

#             # Locate the highest peak in the before binding signal
#             bb_peak_idx = np.argmax(bb_inten[bb_peaks])

#             # # Find the wavelength of the highest peak
#             bb_hp = bb_wl[bb_peaks][bb_peak_idx]
            
#             if n_peaks <= len(ab_peaks) and n_peaks <= len(bb_peaks):
#                 # Find the index of the closest peak in the after binding signal
#                 ab_hp = min(ab_wl[ab_peaks], key=lambda x:abs(x-bb_hp))
#                 ab_peak_idx = int(np.where(ab_wl[ab_peaks] == ab_hp)[0])
                
#                 fig.add_trace(go.Scatter(mode="markers",
#                                             x=bb_wl[bb_peaks][bb_peak_idx - left:bb_peak_idx + right], 
#                                             y=bb_inten[bb_peaks][bb_peak_idx - left:bb_peak_idx + right],
#                                             name="Peaks",
#                                             marker=dict(size=12,
#                                                         color="LightSkyBlue")))

#                 # Calculate distance between peaks
#                 bb_peak_dis = bb_wl[bb_peaks][bb_peak_idx - left:bb_peak_idx + right]
#                 ab_peak_dis = ab_wl[ab_peaks][ab_peak_idx - left:ab_peak_idx + right]
                
#                 if len(bb_peak_dis) != len(ab_peak_dis):
#                     children.append(
#                         html.Div([html.H4(par),
#                                 html.H6("*There was a problem with the peak detection"),
#                                 dcc.Graph(figure=fig)
#                                 ])
#                     )
#                     continue

#                 # Calculate peaks height
#                 bb_peak_h = bb_inten[bb_peaks][bb_peak_idx - left:bb_peak_idx + right]
#                 ab_peak_h = ab_inten[ab_peaks][ab_peak_idx - left:ab_peak_idx + right]

#                 diff = np.subtract(ab_peak_dis, bb_peak_dis)

#                 fig.add_trace(go.Scatter(mode="text",
#                                             x=bb_peak_dis + 0.3, # with offset
#                                             y=bb_peak_h,
#                                             text=np.round(diff, decimals=2),
#                                             name="Distances",
#                                             textfont_color="blue",
#                                             textposition="middle right"))

#                 # Measuring of the height is not needed

#                 # h_diff = np.subtract(ab_peak_h, bb_peak_h)

#                 # fig.add_trace(go.Scatter(mode="text",
#                 #                             x=bb_peak_dis + 0.3, # with offset
#                 #                             y=bb_peak_h + h_diff/2,
#                 #                             text=np.round(h_diff, decimals=2),
#                 #                             name="Heights",
#                 #                             textfont_color="red",
#                 #                             textposition="middle right"))

#                 # # Render the lines on the graph
#                 # for i in range(len(bb_peak_dis)):
#                 #     fig.add_shape(type="line",
#                 #         x0=bb_peak_dis[i], y0=bb_peak_h[i], x1=bb_peak_dis[i], y1=ab_peak_h[i],
#                 #         line=dict(
#                 #             color="LightSeaGreen",
#                 #             width=2,
#                 #             dash="dot",
#                 #         )
#                 #     )

                

#                 children.append(
#                     html.Div([html.H4(par),
#                             dcc.Graph(figure=fig)])
#                 )
#             else:
#                 children.append(
#                     html.Div([html.H4(par),
#                             html.H6("*There was a problem with the peak detection"),
#                             dcc.Graph(figure=fig)
#                             ])
#                 )

#         return children
#     else:
#         return "Load signal data"

# Tab 2

# @app.callback(Output("analysis-graphs", "children"),
#             Input("signal-1-org", "data"),
#             Input("signal-2-org", "data"),
#             Input("baseline-lam", "value"),
#             Input("baseline-p", "value"),
#             Input("apply_correction", "n_clicks"),
#             Input("show_peaks", "n_clicks"),
#             State("mode", "value"),
#             State("n_peaks", "value"),
#             State("prom", "value"))
# def render_analysis_graphs(s1, s2, lam, p, n_clicks, show_peaks, mode, n_peaks, prom):
#     if s1 != None and s2 != None:
#         df1 = pd.read_json(s1, orient="split")
#         df2 = pd.read_json(s2, orient="split")
#         children=[]
#         for par in np.intersect1d(np.unique(df1["par"]), np.unique(df2["par"])):
#             dff1 = df1[df1["par"]==par]
#             dff2 = df2[df2["par"]==par]

#             bb_inten = dff1.loc[:,"inten"].to_numpy()
#             ab_inten = dff2.loc[:,"inten"].to_numpy()

#             bb_wl = dff1.loc[:,"nm"].to_numpy()
#             ab_wl = dff1.loc[:,"nm"].to_numpy()

#             bb_inten, ab_inten = scale_intensity(dff1.loc[:,"inten"].to_numpy(), dff2.loc[:,"inten"].to_numpy())

#             # Peak detection
#             if n_peaks > 1:
#                 if mode == "before":
#                     right = 1
#                     left = n_peaks - 1
#                 elif mode == "after":
#                     right = n_peaks
#                     left = 0
#                 else:
#                     right = int(np.floor(n_peaks/2)) + 1
#                     left = int(np.ceil(n_peaks/2)) - 1
#             else:
#                 left = 0
#                 right = 1
            
#             bb_peaks, _ = find_peaks(bb_inten, prominence=prom)
#             ab_peaks, _ = find_peaks(ab_inten, prominence=prom)

#             # Locate the highest peak in the before binding signal
#             bb_peak_idx = np.argmax(bb_inten[bb_peaks])

#             # # Find the wavelength of the highest peak
#             bb_hp = bb_wl[bb_peaks][bb_peak_idx]
#             # Locate the highest peak in the before binding signal
#             bb_peak_idx = np.argmax(bb_inten[bb_peaks])

#             # # Find the wavelength of the highest peak
#             bb_hp = bb_wl[bb_peaks][bb_peak_idx]

#             # Find the index of the closest peak in the after binding signal
#             ab_hp = min(ab_wl[ab_peaks], key=lambda x:abs(x-bb_hp))
#             ab_peak_idx = int(np.where(ab_wl[ab_peaks] == ab_hp)[0])

#             bb_baseline = baseline_als(bb_inten,lam, p)
#             ab_baseline = baseline_als(ab_inten,lam, p)

#             if n_clicks and n_clicks %2:
#                 bb_inten = bb_inten - bb_baseline
#                 ab_inten = ab_inten - ab_baseline   

#             fig = generate_base_graph(bb_wl, bb_inten, ab_wl, ab_inten)
#             if not n_clicks:
#                 fig.add_trace(go.Scatter(x=bb_wl, y=bb_baseline, name="Signal 1 baseline"))
#                 fig.add_trace(go.Scatter(x=ab_wl, y=ab_baseline, name="Signal 2 baseline"))
        
#             if show_peaks != None and show_peaks % 2: # adds toggle functionality to the button
#                 if n_clicks and n_clicks %2:
#                     bb_minima = np.argwhere(bb_inten <= 0).flatten()
#                     ab_minima = np.argwhere(ab_inten <= 0).flatten()
                    
#                     if len(bb_peaks[bb_peak_idx - left:bb_peak_idx + right]) != len(ab_peaks[ab_peak_idx - left:ab_peak_idx + right]):
#                         children.append(
#                             html.Div([html.H4(par),
#                                     html.H6("*There was a problem with the peak detection"),
#                                     dcc.Graph(figure=fig)
#                                     ])
#                         )
#                         continue

#                     # Find peak start and end
#                     bb_p_start = np.array([int(max([i for i in bb_minima - p if i < 0])) for p in bb_peaks[bb_peak_idx - left:bb_peak_idx + right]])
#                     bb_p_end = np.array([int(min([i for i in bb_minima - p if i > 0])) for p in bb_peaks[bb_peak_idx - left:bb_peak_idx + right]])

#                     ab_p_start = np.array([int(max([i for i in ab_minima - p if i < 0])) for p in ab_peaks[ab_peak_idx - left:ab_peak_idx + right]])
#                     ab_p_end = np.array([int(min([i for i in ab_minima - p if i > 0])) for p in ab_peaks[ab_peak_idx - left:ab_peak_idx + right]])

#                     # Calculate area under the peak
#                     bb_chosen_peaks = bb_peaks[bb_peak_idx - left:bb_peak_idx + right]
#                     ab_chosen_peaks = ab_peaks[ab_peak_idx - left:ab_peak_idx + right]

#                     bb_areas = [np.trapz(bb_inten[bb_chosen_peaks[pi]+bb_p_start[pi]:bb_chosen_peaks[pi]+bb_p_end[pi]],
#                                         bb_wl[bb_chosen_peaks[pi]+bb_p_start[pi]:bb_chosen_peaks[pi]+bb_p_end[pi]]) for pi in range(len(bb_chosen_peaks))]
#                     ab_areas = [np.trapz(ab_inten[ab_chosen_peaks[pi]+ab_p_start[pi]:ab_chosen_peaks[pi]+ab_p_end[pi]],
#                                         ab_wl[ab_chosen_peaks[pi]+ab_p_start[pi]:ab_chosen_peaks[pi]+ab_p_end[pi]]) for pi in range(len(ab_chosen_peaks))]

#                     # bb_areas = [np.trapz(bb_inten[bb_chosen_peaks[pi]+bb_p_start[pi]:bb_chosen_peaks[pi]+bb_p_end[pi]]) for pi in range(len(bb_chosen_peaks))]
#                     # ab_areas = [np.trapz(ab_inten[ab_chosen_peaks[pi]+ab_p_start[pi]:ab_chosen_peaks[pi]+ab_p_end[pi]]) for pi in range(len(ab_chosen_peaks))]

#                     fig.add_trace(go.Scatter(mode="text",
#                             x=bb_wl[bb_chosen_peaks], # with offset
#                             y=bb_inten[bb_chosen_peaks]/2,
#                             text=np.round(bb_areas, decimals=4),
#                             name="Area",
#                             textfont_color="blue",
#                             textposition="middle right"))

#                     fig.add_trace(go.Scatter(mode="text",
#                             x=ab_wl[ab_chosen_peaks], # with offset
#                             y=ab_inten[ab_chosen_peaks]/2,
#                             text=np.round(ab_areas, decimals=4),
#                             name="Area",
#                             textfont_color="red",
#                             textposition="middle right"))
                    

#                     fig.add_trace(go.Scatter(mode="markers",
#                                                 x=np.concatenate((bb_wl[bb_peaks[bb_peak_idx - left:bb_peak_idx + right] + bb_p_start], bb_wl[bb_peaks[bb_peak_idx - left:bb_peak_idx + right] + bb_p_end]),axis=None),
#                                                 y=np.concatenate((bb_inten[bb_peaks[bb_peak_idx - left:bb_peak_idx + right] + bb_p_start], bb_inten[bb_peaks[bb_peak_idx - left:bb_peak_idx + right] + bb_p_end]),axis=None),
#                                                 name="bb signal lim",
#                                                 marker=dict(size=8,
#                                                             symbol="x",
#                                                             color="blue")))

#                     fig.add_trace(go.Scatter(mode="markers",
#                                                 x=np.concatenate((ab_wl[ab_peaks[ab_peak_idx - left:ab_peak_idx + right] + ab_p_start], ab_wl[ab_peaks[ab_peak_idx - left:ab_peak_idx + right] + ab_p_end]),axis=None),
#                                                 y=np.concatenate((ab_inten[ab_peaks[ab_peak_idx - left:ab_peak_idx + right] + ab_p_start], ab_inten[ab_peaks[ab_peak_idx - left:ab_peak_idx + right] + ab_p_end]),axis=None),
#                                                 name="ab signal lim",
#                                                 marker=dict(size=8,
#                                                             symbol="x",
#                                                             color="red")))

#             children.append(html.Div([html.H4(par),
#                         dcc.Graph(figure=fig)]))

#         return children

# Tab 3

# @app.callback(
#         Output("intens-data", "data"),
#         Output("tabs", "active_tab"),
#         Input("extract_areas", "n_clicks"),
#         State("signal-1-org", "data"),
#         State("signal-2-org", "data"),
#         State("baseline-lam", "value"),
#         State("baseline-p", "value"),
#         State("n_peaks", "value"),
#         State("prom", "value"),
#         State("date-picker", "date"),
#         State("exp-id", "value"),
#         State("par-type", "value"),
#         State("intens-data", "data"),
#         State("check", "value"),
#         prevent_initial_call=True)
# def generate_dataframe(n_clicks, s1, s2, lam, p, n_peaks, prom, d, exp_id, par_type, current_d, ch):
#     if s1 != None and s2 !=None:
#         df1 = pd.read_json(s1, orient="split")
#         df2 = pd.read_json(s2, orient="split")

#         bb_dis_dict, ab_dis_dict, bb_h_dict, ab_h_dict = _measure_distances(df1, df2, n_peaks, prom)
#         if "areas" in ch:
#             bb_a_dict, ab_a_dict = _measure_areas(df1, df2, lam, p, n_peaks, prom)

#         df_dict = {
#             "date": [d for i in range(len(bb_dis_dict))],
#             "parType": [par_type for i in range(len(bb_dis_dict))],
#             "expID": [exp_id for i in range(len(bb_dis_dict))],
#             "parID": [*bb_dis_dict.keys()]}

#         if "wavelength" in ch:
#             for i in range(len([*bb_dis_dict.values()][0])):
#                 df_dict[f"bb_peak_{i+1}"] = [bb_dis_dict[par][i] for par in df_dict["parID"]]

#             for i in range(len([*ab_dis_dict.values()][0])):
#                 df_dict[f"ab_peak_{i+1}"] = [ab_dis_dict[par][i] for par in df_dict["parID"]]

#         if "heights" in ch:
#             for i in range(len([*bb_h_dict.values()][0])):
#                 df_dict[f"bb_height_{i+1}"] = [bb_h_dict[par][i] for par in df_dict["parID"]]

#             for i in range(len([*ab_h_dict.values()][0])):
#                 df_dict[f"ab_height_{i+1}"] = [ab_h_dict[par][i] for par in df_dict["parID"]]
        
#         if "areas" in ch:
#             for i in range(len([*bb_a_dict.values()][0])):
#                 df_dict[f"bb_area_{i+1}"] = [bb_a_dict[par][i] for par in df_dict["parID"]]

#             for i in range(len([*ab_a_dict.values()][0])):
#                 df_dict[f"ab_area_{i+1}"] = [ab_a_dict[par][i] for par in df_dict["parID"]]

#         if "extra" in ch:
#             avg, bb_sd, ab_sd, bb_iqr, ab_iqr = _measure_features(df1, df2)

#             df_dict["avg_dis"] = [avg[par] for par in df_dict["parID"]]
#             df_dict["bb_sd"] = [bb_sd[par] for par in df_dict["parID"]]
#             df_dict["ab_sd"] = [ab_sd[par] for par in df_dict["parID"]]
#             df_dict["bb_iqr"] = [bb_iqr[par] for par in df_dict["parID"]]
#             df_dict["ab_iqr"] = [ab_iqr[par] for par in df_dict["parID"]]

#         df = pd.DataFrame(df_dict)

#         if current_d != None and ch:
#             cd = pd.read_json(current_d, orient="split", convert_dates=False)
#             df = pd.concat([cd, df], ignore_index=True)
        
#         return df.to_json(date_format='iso', orient='split'), "tab3"

# @app.callback(Output("intens-table", "children"),
#               Output("undo", "disabled"),
#               Input("extract", "n_clicks"),
#               Input("undo", "n_clicks"),
#               State("intens-data", "data"),
#               State("restore-data", "data"),
#               State("undo", "disabled"),
#               prevent_initial_call=True)
# def render_dataframe(extract, restore, current_d, old_d, restore_disabled):
#     #TODO fix restore
#     if restore_disabled:
#         if current_d != None:
#             df = pd.read_json(current_d, orient='split', convert_dates=False)
#             return dbc.Table.from_dataframe(df, bordered=True, striped=True, responsive=True),False
#     else:
#         if old_d == None:
#             return None,True
#         df = pd.read_json(old_d, orient='split', convert_dates=False)
#         return dbc.Table.from_dataframe(df, bordered=True, striped=True, responsive=True),True
        
#     return None,True
        
# @app.callback(
#     Output("download-intens-csv", "data"),
#     Input("btn_csv", "n_clicks"),
#     State("intens-data", "data"),
#     prevent_initial_call=True)
# def func(n_clicks, current_d):

#     df = pd.read_json(current_d, orient='split', convert_dates=False)
#     return dcc.send_data_frame(df.to_csv, "featuresData.csv")

if __name__ == '__main__':
    app.run_server(debug=True, port=3333)