from dash import dcc
import dash
from dash import html, State
from dash.dependencies import Input, Output
from graph import get_networx_graph, plotly_graph, read_graphs
from latin import prepare_seeds
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from gensim.models import Word2Vec
import os
import plotly.graph_objs as go
import pandas as pd
from timeline import plot_word_occurance, plot_word_use

model = Word2Vec.load("../dat/word2vec.model")
token_table = pd.read_csv("../dat/token_table.csv")
word_use = token_table.drop("tokens", "columns").groupby("Årstal").sum().reset_index()

app = dash.Dash(__name__)
server = app.server

def Label(text = ""):
    return html.Label(
                text,
                style={
                    'padding-top': '20px',
                    "font": "15px Helvetica",
                    "margin-bottom": "7px"
                }
            )

app.layout = html.Div([
    html.Div([
        dcc.Tabs(id="tabs", value="network-tab", children=[
            dcc.Tab(label="Semantisk netværk", value="network-tab"),
            dcc.Tab(label="Tidslinje og ordanalyse", value="timeline-tab"),
        ], colors={
        "border": "white",
        "primary": "#8100d1",
        "background": "#fbf5ff"
        }),
        html.Div(
            dcc.Graph(id='network', style={"height": "100%", "width": "100%"}),
            id="network-container",
        ),
        html.Div([
            dcc.Graph(id='timeline', style={"display": "flex", "flex": "2"}),
            dcc.Graph(id="word-occurance", style={"display": "flex", "flex": "1"})
        ],
        id="timeline-container"),
    ],
    style={
            #'padding': '10px',
            "display": "flex",
            "flex-direction": "column",
            "flex": "4",
            "height": "100%",
            "font": "15px Helvetica",
    }),
    html.Div([
                Label('Please write in the seeds (comma-separated):'),
                html.Div([dcc.Input(
                    id='seeds',
                    type='text',
                    placeholder="Your seeds here",
                    style={
                        'padding' : '10px',
                        "margin-right": "10px",
                        'font': '15px Helvetica',
                        'width' : '95%',
                    }
                    )],
                    style={
                        'padding' : '10px',
                        'font': '10px Helvetica',
                        #'width' : '100%',
                        'margin-bottom': '20px'
                    }),
                Label('Number of words from first level of association:'),
                dcc.Slider(
                    id='k',
                    min=1,
                    max=10,
                    step=1,
                    value=3,
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
                Label('Number of words from second level of association:'),
                dcc.Slider(
                    id='m',
                    min=1,
                    max=10,
                    step=1,
                    value=3,
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
                Label("Please select which genres to analyse:"),
                dcc.Checklist(
                    options=[
                        {"label": "Sermon", "value": "Sermon"},
                        {"label": "Letter", "value": "Letter"},
                        {"label": "Writing", "value": "Writing"},
                    ],
                    value=["Sermon", "Letter", "Writing"],
                    id="genres-list"
                ),
                html.Button(
                    'Submit',
                    id='submit',
                    n_clicks=0,
                    style={
                        'margin-top' : '10px',
                        'padding' : '15px',
                        'padding-left' : '20px',
                        'padding-right' : '20px',
                        'outline': 'false',
                        'font': 'bold 10px Helvetica',
                        'background': '#8100d1',
                        'color' : 'white',
                        'border' : 'none',
                        'border-radius': '8px'
                    }
                ),
        ],
        style={
            'padding': '10px',
            "display": "flex",
            "flex-direction": "column",
            "margin": "0",
            "flex": "0 0 420px",
            "background": "#fcfcfc",
            "box-shadow": "-2px 0 5px #00000066",
            "z-index": "5",
        }),
],
style={
        "top": "0",
        "left": "0",
        "display": "flex",
        "flex-direction": "row",
        "justify-content": "space-around",
        "align-items": "stretch",
        #"flex": "1",
        "height": "100%",
        "width": "100%",
        "position": "fixed"
    }
)

@app.callback(
    [Output('network', 'figure'), Output('timeline', 'figure'), Output("word-occurance", "figure")],
    [Input('submit', 'n_clicks'),
    State("genres-list", "value"),
    State('k', 'value'),
    State('m', 'value'),
    State('seeds', 'value')]
)
def update_network(n_clicks,genres, k, m, seeds_text):
    if seeds_text:
        seeds = [seed.strip() for seed in seeds_text.split(",")]
        network_seeds = prepare_seeds(model,seeds)
        #print(f"creating kernel with seeds: {seeds}")
        G, pos, connections = get_networx_graph(network_seeds, model, k = k, m = m)
        return (
            plotly_graph(G, pos, connections),
            plot_word_use(word_use,token_table, seeds, genres),
            plot_word_occurance(token_table, seeds, genres)
        )
    else:
        return {}, {}, {}

@app.callback([Output('network-container', 'style'), Output("timeline-container", "style")],
              [Input('tabs', 'value')])
def render_tabs(tab):
    if tab == 'network-tab':
        return (
            {
                "display": "block",
                #"flex": "1",
                "width": "100%",
                "height": "100%"
            }, {
                "display" : "none"
            }
        )
    elif tab == 'timeline-tab':
        return (
            {
                "display" : "none"
            }, {
                "display": "block",
            }
        )

if __name__ == '__main__':
    app.run_server(debug = True)#host = "0.0.0.0", debug=True, port = 8080)