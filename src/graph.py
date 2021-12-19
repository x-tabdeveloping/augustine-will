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
#from networkx.drawing.nx_agraph import graphviz_layout
import itertools
import gensim
import re
from networkx.drawing.layout import spring_layout

def distinct(l):
    return list(set(l)) 

def empty(s):
    return not s or not s.strip()

def most_similar(word, n, model):
    return [word for word, _ in model.wv.most_similar(positive = [word], topn = n)]

def similarity_matrix(tokens, model):
    word_vectors = model.wv
    vectors = np.array([word_vectors.get_vector(token) for token in tokens])
    matrix = np.zeros((len(tokens), len(tokens)))
    for i, token in enumerate(tokens):
        #calculates distance of one certain token to every other one
        matrix[i, :] = word_vectors.cosine_similarities(vectors[i], vectors)
    #delete edges between the token and itself
    np.fill_diagonal(matrix, 0.)
    return matrix

def nodes(seeds, model, k = 3, m = 10):
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
    return (types, tokens)

def dist_prune(d, prune=True):
    d = 1-d
    np.fill_diagonal(d, 0.)
    if prune:
        cond = np.mean(d)
        zero = np.zeros_like(d)
        return np.where(d > cond, zero, d)
    else:
        return d

def get_networx_graph(seeds, model, k = 3, m = 3):
    print("Getting tokens")
    types, tokens = nodes(seeds, model, k=k, m = m)
    delta, labels = graph(types, tokens, model)
    print(f"Building matrix with labels: {labels}")
    delta = dist_prune(delta)
    delta = delta * 10  # scale
    dt = [("len", float)]
    delta = delta.view(dt)

    #  Graphviz
    G = nx.from_numpy_matrix(delta)
    G = nx.relabel_nodes(G, dict(enumerate(labels)))
    pos = spring_layout(G) #graphviz_layout(G)
    print("Giving back G and pos")
    return G, pos

def graph(types, tokens, model):
    delta = similarity_matrix(tokens, model)
    labels = [token.upper() if token in types else token.lower() for token in tokens]
    return (delta, labels)

def immediate_subdirectories(a_dir):
    print(os.getcwd())
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def read_graphs():
    graphs = {}
    for koncept in immediate_subdirectories("dat/koncepter"):
        with open(f"dat/koncepter/{koncept}/G.obj", "rb") as f:
            G = pickle.load(f)
        with open(f"dat/koncepter/{koncept}/pos.txt") as f:
            text = f.read()
        lines = text.split("\n")
        pos_dict = {}
        for line in lines:
            [node, pos_text] = line.split(":")
            pos_text = pos_text[1: -1]
            [x,y] = pos_text.split(",")
            x,y = float(x), float(y)
            pos_dict[node] = (x,y)
        graphs[koncept] = (G, pos_dict)
    return graphs

def plotly_graph(G, pos):
    print("constructing plotly graph")
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
    np.random.seed(seed=1234)
    parts = community_louvain.best_partition(G)
    colors = [parts.get(node) for node in G.nodes()]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers + text',
        hoverinfo='text',
        text = list(G.nodes()),
        textposition='top center',
        marker=dict(
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='Rainbow',
            reversescale=True,
            color=colors,
            size=10,
            line_width=2)
    )

    #node_trace.text = list(G.nodes())
    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                 paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                titlefont_size=16,
                showlegend=False,
                #showscale = False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
    return fig