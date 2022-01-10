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

app.layout = html.Div([
    html.Div([
        html.Div(
            dcc.Graph(id='network'),
        ),
        html.Div([
            dcc.Graph(id='timeline', style={"display": "flex", "flex": "2"}),
            dcc.Graph(id="word-occurance", style={"display": "flex", "flex": "1"})
        ],
        style={
            "display": "flex",
            "flex-direction": "row",
        })
    ],
    style={
            #'padding': '10px',
            "display": "flex",
            "flex-direction": "column",
            "flex": "4",
            "height": "100%"
    }),
    html.Div([
                html.Label(
                    ['Please write in the seeds (comma-separated):'],
                    style={
                        'padding-top': '20px',
                        "font": "15px Helvetica"
                    }
                ),
                html.Div([dcc.Input(
                    id='seeds',
                    type='text',
                    placeholder="Your seeds here",
                    style={
                        'padding' : '10px',
                        'font': '15px Helvetica',
                        'width' : '100%',
                    }
                    )],
                    style={
                        'padding-top' : '10px',
                        'font': '10px Helvetica',
                        'width' : '100%',
                        'margin-bottom': '20px'
                    }),
                html.Label(
                    ['Number of words from first level of association:'],
                    style={
                        'margin-top': '10px',
                        'padding-top': '20px',
                        "font": "15px Helvetica"
                    }
                ),
                dcc.Slider(
                    id='k',
                    min=1,
                    max=10,
                    step=1,
                    value=3,
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
                html.Label(
                    ['Number of words from second level of association:'],
                    style={
                        'padding-top': '20px',
                        "font": "15px Helvetica"
                    }
                ),
                dcc.Slider(
                    id='m',
                    min=1,
                    max=10,
                    step=1,
                    value=3,
                    tooltip={"placement": "bottom", "always_visible": True}
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
            "flex": "2",
            "background": "#f0f0f0",
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
    State('k', 'value'),
    State('m', 'value'),
    State('seeds', 'value')]
)
def update_network(n_clicks,k, m, seeds_text):
    if seeds_text:
        seeds = [seed.strip() for seed in seeds_text.split(",")]
        network_seeds = prepare_seeds(model,seeds)
        #print(f"creating kernel with seeds: {seeds}")
        G, pos, connections = get_networx_graph(network_seeds, model, k = k, m = m)
        return (
            plotly_graph(G, pos, connections),
            plot_word_use(word_use,token_table, seeds),
            plot_word_occurance(token_table, seeds)
        )
    else:
        return {}, {}, {}

if __name__ == '__main__':
    app.run_server(debug = True)#host = "0.0.0.0", debug=True, port = 8080)