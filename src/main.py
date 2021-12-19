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

model = Word2Vec.load("../dat/word2vec.model")

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.Div(
        dcc.Graph(id='graph',
                  style={'height': "70%", 'width': "100%"}
                  ),
    ),
    html.Div([
                html.Label(
                    ['Please write in the seeds (comma-separated):'],
                    style={
                        'padding-top': '20px',
                        "font": "15px Helvetica"
                    }
                ),
                html.Div([dcc.Input(
                    id='input-on-submit',
                    type='text',
                    placeholder="Your seeds here",
                    style={
                        'padding' : '10px',
                        'font': '15px Helvetica',
                        'width' : '90%',
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
                    id='submit-val',
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
            'padding': '10px'
        })
])

@app.callback(
    Output('graph', 'figure'),
    Input('submit-val', 'n_clicks'),
    State('k', 'value'),
    State('m', 'value'),
    State('input-on-submit', 'value')
)
def update_figure(n_clicks,k, m, seeds_text):
    if seeds_text:
        seeds = [seed.strip() for seed in seeds_text.split(",")]
        seeds = prepare_seeds(model,seeds)
        print(f"creating kernel with seeds: {seeds}")
        G, pos = get_networx_graph(seeds, model, k = k, m = m)
        return plotly_graph(G, pos)
    else:
        return {}

if __name__ == '__main__':
    app.run_server(host = "0.0.0.0", debug=True, port = 8080)