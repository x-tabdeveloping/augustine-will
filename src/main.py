import click
from dash import dcc
import dash
from dash import html, State
from dash.dependencies import Input, Output
from graph import build_plot, get_graph, get_neighbours
from latin import prepare_seeds
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from gensim.models import Word2Vec
import os
import plotly.graph_objs as go
import pandas as pd
from timeline import filter_tokens, plot_word_occurance, plot_word_use

model = Word2Vec.load("../dat/word2vec.model")
token_table = pd.read_csv("../dat/token_table.csv")
word_use = token_table.drop("tokens", "columns").groupby("Årstal").sum().reset_index()

app = dash.Dash(__name__)
server = app.server


def Label(text=""):
    return html.Label(
        text,
        style={"padding-top": "20px", "font": "15px Helvetica", "margin-bottom": "7px"},
    )


popup_style = {
    "display": "none",
    "position": "fixed",
    "top": "50%",
    "left": "50%",
    "transform": "translate(-50%, -50%)",
    "width": "50%",
    "height": "50%",
    "padding": "10px",
    "z-index": "10",
    "background": "white",
    "box-shadow": "2px 2px 5px #00000066",
    "border-radius": "8px",
    "overflow": "hidden",
}

close_button_style = {
    "position": "absolute",
    "top": "22",
    "left": "93%",
    "background": "#00000000",
    "color": "#8100d1",
    "padding": "10px",
    "padding-right": "15px",
    "padding-left": "15px",
    "outline": "false",
    "font": "bold 12px Helvetica",
    "border": "none",
    "transform": "translate(-50%)",
}

sidebar_style = {
    "padding": "10px",
    "display": "flex",
    "flex-direction": "column",
    "margin": "0",
    "flex": "0 0 420px",
    "background": "white",  # "#fcfcfc",
    "box-shadow": "-2px 0 5px #00000066",
    "z-index": "5",
}

app.layout = html.Div(
    [
        dcc.Store(id="network-state", storage_type="local"),
        dcc.Store(id="analysis-state", storage_type="local"),
        html.Div(
            [
                dcc.Tabs(
                    id="tabs",
                    value="network-tab",
                    children=[
                        dcc.Tab(label="Semantisk netværk", value="network-tab"),
                        dcc.Tab(label="Tidslinje og ordanalyse", value="timeline-tab"),
                    ],
                    colors={
                        "border": "white",
                        "primary": "#8100d1",
                        "background": "#fbf5ff",
                    },
                ),
                html.Div(
                    [
                        dcc.Dropdown(
                            id="network-switch",
                            options=[
                                {"label": "Avanceret Netværk", "value": "sej"},
                                {
                                    "label": "Simpelt Netværk",
                                    "value": "kedelig",
                                },
                            ],
                            value="sej",
                            clearable=False,
                            style={"height": "5%"},
                        ),
                        dcc.Graph(
                            id="network",
                            style={
                                "height": "95%",
                                "width": "100%",
                            },
                        ),
                    ],
                    id="network-container",
                ),
                html.Div(
                    [
                        dcc.Graph(id="timeline", style={"width": "100%"}),
                        dcc.Dropdown(
                            id="timeline-switch",
                            options=[
                                {"label": "Procentvis", "value": "procent"},
                                {"label": "Absolute tal", "value": "absolute"},
                            ],
                            value="absolute",
                            clearable=False,
                        ),
                        dcc.Graph(id="word-occurance", style={"width": "100%"}),
                    ],
                    id="timeline-container",
                ),
            ],
            style={
                # 'padding': '10px',
                "display": "flex",
                "flex-direction": "column",
                "flex": "4",
                "height": "100%",
                "font": "15px Helvetica",
                "overflow-y": "auto",
            },
        ),
        html.Div(
            [
                Label("Indtast dine seeds (komma separeret):"),
                html.Div(
                    [
                        dcc.Input(
                            id="seeds",
                            type="text",
                            placeholder="Your seeds here",
                            style={
                                "padding": "10px",
                                "margin-right": "10px",
                                "font": "15px Helvetica",
                                "width": "95%",
                            },
                        )
                    ],
                    style={
                        "padding": "10px",
                        "font": "10px Helvetica",
                        # 'width' : '100%',
                        "margin-bottom": "20px",
                    },
                ),
                Label("Antal af ord fra first level assoc.:"),
                dcc.Slider(
                    id="k",
                    min=1,
                    max=10,
                    step=1,
                    value=3,
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
                Label("Antal af ord fra second level assoc.:"),
                dcc.Slider(
                    id="m",
                    min=1,
                    max=10,
                    step=1,
                    value=3,
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
                Label("Vælg genrerne til analysen:"),
                dcc.Checklist(
                    options=[
                        {"label": "Sermon", "value": "Sermon"},
                        {"label": "Letter", "value": "Letter"},
                        {"label": "Writing", "value": "Writing"},
                    ],
                    value=["Sermon", "Letter", "Writing"],
                    id="genres-list",
                ),
                html.Button(
                    "Vælg værker",
                    id="open-works",
                    n_clicks_timestamp=0,
                    style={
                        "margin-top": "10px",
                        "padding": "15px",
                        "padding-left": "20px",
                        "padding-right": "20px",
                        "outline": "false",
                        "font": "bold 10px Helvetica",
                        "background": "#d5b2eb",
                        "color": "black",
                        "border": "none",
                        "border-radius": "8px",
                    },
                ),
                html.Button(
                    "Anvend",
                    id="submit",
                    n_clicks=0,
                    style={
                        "margin-top": "10px",
                        "padding": "15px",
                        "padding-left": "20px",
                        "padding-right": "20px",
                        "outline": "false",
                        "font": "bold 10px Helvetica",
                        "background": "#8100d1",
                        "color": "white",
                        "border": "none",
                        "border-radius": "8px",
                    },
                ),
            ],
            style=sidebar_style,
        ),
        html.Div(
            [
                html.Button(
                    "Anvend",
                    id="close-works",
                    n_clicks_timestamp=0,
                    style=close_button_style,
                ),
                html.Button(
                    "Fravælg alt",
                    id="deselect-works",
                    n_clicks=0,
                    style={
                        **close_button_style,
                        "margin-top": "40px",
                        "color": "black",
                    },
                ),
                html.Div(
                    dcc.Dropdown(
                        options=[],
                        value=[],
                        id="works-list",
                        multi=True,
                    ),
                    style={"width": "90%", "overflow-y": "auto"},
                ),
            ],
            id="work-popup",
            style=popup_style,
        ),
        html.Div(
            [
                html.Button(
                    "Close",
                    id="close-connections",
                    n_clicks_timestamp=0,
                    style=close_button_style,
                ),
                dcc.Clipboard(
                    target_id="connections-text",
                    n_clicks=0,
                    style={
                        **close_button_style,
                        "margin-top": "40px",
                        "color": "black",
                    },
                ),
                html.Div(
                    "",
                    id="connections-text",
                    style={"width": "90%", "overflow-y": "auto"},
                ),
            ],
            id="connections-popup",
            style=popup_style,
        ),
    ],
    style={
        "top": "0",
        "left": "0",
        "display": "flex",
        "flex-direction": "row",
        "justify-content": "space-around",
        "align-items": "stretch",
        # "flex": "1",
        "height": "100%",
        "width": "100%",
        "position": "fixed",
    },
)


@app.callback(
    Output("network-state", "data"),
    [
        Input("submit", "n_clicks"),
        State("k", "value"),
        State("m", "value"),
        State("seeds", "value"),
    ],
)
def update_network(n_clicks, k, m, seeds_text):
    if (not n_clicks) or (not seeds_text):
        raise dash.exceptions.PreventUpdate
    seeds = [seed.strip() for seed in seeds_text.split(",")]
    seeds = prepare_seeds(model, seeds)
    return get_graph(seeds, model, k, m)


@app.callback(
    Output("analysis-state", "data"),
    [
        Input("submit", "n_clicks"),
        State("seeds", "value"),
        State("works-list", "value"),
        State("genres-list", "value"),
    ],
)
def update_word_analysis(n_clicks, seeds_text, works, genres):
    if (not n_clicks) or (not seeds_text):
        raise dash.exceptions.PreventUpdate
    seeds = [seed.strip() for seed in seeds_text.split(",")]
    return dict(analysis_df=filter_tokens(token_table, seeds, genres, works).to_dict())


@app.callback(
    Output("network", "figure"),
    [
        State("network-state", "data"),
        Input("network-state", "modified_timestamp"),
        Input("network-switch", "value"),
    ],
)
def update_network_plot(network, ts, style):
    if network is None:
        raise dash.exceptions.PreventUpdate
    return build_plot(network, style)


@app.callback(
    [
        Output("timeline", "figure"),
        Output("word-occurance", "figure"),
    ],
    [
        State("analysis-state", "data"),
        Input("analysis-state", "modified_timestamp"),
        Input("timeline-switch", "value"),
    ],
)
def update_word_analysis(data, ts, timeline_type):
    if data is None:
        raise dash.exceptions.PreventUpdate
    analysis_df = pd.DataFrame(data["analysis_df"])
    return (
        plot_word_use(analysis_df, word_use, timeline_type),
        plot_word_occurance(analysis_df),
    )


@app.callback(
    Output("work-popup", "style"),
    [
        Input("open-works", "n_clicks_timestamp"),
        Input("close-works", "n_clicks_timestamp"),
    ],
)
def open_works(open_b, close_b):
    if open_b > close_b:
        return {**popup_style, "display": "flex"}
    else:
        return popup_style


@app.callback(
    [Output("network-container", "style"), Output("timeline-container", "style")],
    [Input("tabs", "value")],
)
def render_tabs(tab):
    if tab == "network-tab":
        return (
            {
                "display": "block",
                # "flex": "1",
                "width": "100%",
                "height": "100%",
            },
            {"display": "none"},
        )
    elif tab == "timeline-tab":
        return (
            {"display": "none"},
            {
                "display": "block",
            },
        )


@app.callback(
    [Output("works-list", "options"), Output("works-list", "value")],
    [Input("genres-list", "value"), Input("deselect-works", "n_clicks")],
)
def update_works_list(genres, deselect):
    changed_id = [p["prop_id"] for p in dash.callback_context.triggered][0]
    if not genres:
        genres = ["Writing", "Letter", "Sermon"]
    df = token_table[token_table["Genre"].isin(genres)]
    _, uniques = pd.factorize(df["Forkortelse"])
    works = list(uniques)
    options = [{"label": work, "value": work} for work in works]
    if "deselect-works" in changed_id:
        return options, []
    else:
        return options, works


def markdown_list(values):
    return "\n".join([f"* {value}" for value in values])


@app.callback(
    Output("connections-text", "children"),
    [Input("network", "clickData"), State("network-state", "data")],
)
def update_connections(click_data, graph):
    if (not click_data) or (graph is None):
        raise dash.exceptions.PreventUpdate
    node = click_data["points"][0]["customdata"]
    neighbours = get_neighbours(graph, node)
    title = "Connections of {}({}):".format(graph["labels"][node], len(neighbours))
    return [
        html.H1(title),
        html.Ul([
            html.Li(neighbour) for neighbour in neighbours
        ])
    ]


@app.callback(
    Output("connections-popup", "style"),
    [Input("close-connections", "n_clicks_timestamp"), Input("network", "clickData")],
)
def close_connections(ts, click_data):
    changed_id = [p["prop_id"] for p in dash.callback_context.triggered][0]
    if (not ts) and (not click_data):
        raise dash.exceptions.PreventUpdate
    if "close-connections" not in changed_id:
        return {**popup_style, "display": "flex"}
    else:
        return popup_style


if __name__ == "__main__":
    app.run_server(debug=True)  # host = "0.0.0.0", debug=True, port = 8080)
