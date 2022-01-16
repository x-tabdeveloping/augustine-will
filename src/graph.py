import plotly.express as px
import plotly.graph_objects as go
from community import community_louvain
import networkx as nx
import os
import pickle
import numpy as np
from gensim.models import Word2Vec
import os
import matplotlib.pyplot as plt
import networkx as nx

# from networkx.drawing.nx_agraph import graphviz_layout
import itertools
import gensim
import re
from networkx.drawing.layout import spring_layout


def distinct(l):
    return list(set(l))


def empty(s):
    return not s or not s.strip()


def most_similar(word, n, model):
    return [word for word, _ in model.wv.most_similar(positive=[word], topn=n)]


def distance_matrix(tokens, model):
    word_vectors = model.wv
    # vectors = np.array([word_vectors.get_vector(token) for token in tokens])
    matrix = np.zeros((len(tokens), len(tokens)))
    for i, token in enumerate(tokens):
        # calculates distance of one certain token to every other one
        matrix[i, :] = word_vectors.distances(token, tokens)
    # delete edges between the token and itself
    np.fill_diagonal(matrix, 0.0)
    cond = np.median(matrix)
    zero = np.zeros_like(matrix)
    matrix = np.where(matrix > cond, zero, matrix)
    return matrix


def immediate_subdirectories(a_dir):
    print(os.getcwd())
    return [
        name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))
    ]


def get_graph(seeds, model, k, m):
    types = []
    for seed in seeds:
        types.extend(most_similar(seed, k, model))
        types.append(seed)
    types = distinct(types)
    tokens = []
    for source in types:
        tokens.extend(most_similar(source, m, model))
        tokens.append(source)
    tokens = distinct(tokens)
    labels = [token.upper() if token in types else token.lower() for token in tokens]
    delta = distance_matrix(tokens, model)
    connections = np.sum(delta != 0, axis=1)
    delta = delta * 10  # scale
    dt = [("len", float)]
    delta = delta.view(dt)
    G = nx.from_numpy_matrix(delta)
    pos = spring_layout(nx.from_numpy_matrix(delta))
    parts = community_louvain.best_partition(G)
    colors = list(parts.values())
    edges = np.array(G.edges())
    return {
        "labels": labels,
        "edges": edges,
        "pos": pos,
        "colors": colors,
        "connections": connections,
    }


def get_edge_pos(edges, x_y):
    a = x_y[edges]
    # 2nas = np.empty_like(a.shape[1])
    # nas.fill(np.nan)
    a.shape
    b = np.zeros((a.shape[0], a.shape[1] + 1))
    b[:, :-1] = a
    b[:, -1] = np.nan
    return b.flatten()


def build_plot(graph, style):
    x, y = zip(*graph["pos"].values())
    x, y = np.array(x), np.array(y)
    edges_x = get_edge_pos(graph["edges"], x)
    edges_y = get_edge_pos(graph["edges"], y)
    sum_connections = np.sum(graph["connections"])
    graph["connections"] = np.array(graph["connections"])
    # tens = np.full_like(graph["connections"], 10)
    indices = list(range(len(x)))
    size = 100 * graph["connections"] / sum_connections
    if style == "sej":
        annotations = [
            dict(
                text=node,
                x=x[i],
                y=y[i],
                showarrow=False,
                xanchor="center",
                bgcolor="rgba(255,255,255,0.5)",
                bordercolor="rgba(0,0,0,0.5)",
                font={
                    "family": "Helvetica",
                    "size": max(size[i], 10),
                    "color": "black",
                },
            )
            for i, node in enumerate(graph["labels"])
        ]
    if style == "sej":
        node_trace = go.Scatter(
            x=x,
            y=y,
            mode="markers",
            hoverinfo="text",
            text=graph["labels"],
            marker={
                "colorscale": "sunsetdark",
                "reversescale": True,
                "color": graph["colors"],
                "size": 10 * size,
                "line_width": 2,
            },
            customdata=indices,
        )
    else:
        node_trace = go.Scatter(
            x=x,
            y=y,
            mode="markers+text",
            hoverinfo="text",
            text=graph["labels"],
            textposition="top center",
            marker={
                "colorscale": "sunsetdark",
                "reversescale": True,
                "color": graph["colors"],
                "size": 10,
                "line_width": 2,
            },
            customdata=indices,
        )
    edge_trace = go.Scatter(
        x=edges_x,
        y=edges_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            clickmode="event",
            annotations=annotations if style == "sej" else [],
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            titlefont_size=16,
            showlegend=False,
            # showscale = False,
            # hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )
    return fig


def get_neighbours(graph, node):
    labels = graph["labels"]
    edges = graph["edges"]
    neighbours = [j if i == node else i for i, j in edges if (i == node) or (j == node)]
    return [labels[i] for i in neighbours]
