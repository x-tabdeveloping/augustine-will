import numpy as np
from gensim.models import Word2Vec
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from community import community_louvain

def distinct(l):
    return list(set(l)) 

def empty(s):
    return not s or not s.strip()

from gensim.models.callbacks import CallbackAny2Vec
from gensim.test.utils import common_texts, get_tmpfile

class callback(CallbackAny2Vec):
    '''Callback to print loss after each epoch.'''

    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        print('Loss after epoch {}: {}'.format(self.epoch, loss))
        self.epoch += 1

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

def graph(types, tokens, model):
    delta = similarity_matrix(tokens, model)
    labels = [token.upper() if token in types else token.lower() for token in tokens]
    return (delta, labels)

def identity(x):
    return x

def word_split(s):
    return s.split()
            
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
def dist_prune(d, prune=True):
    #minimum = np.min(d)
    #if minimum == 0:
    #    d = d + 0.1
    #else if minimum < 0.1:
    #    d = d * (0.1/minimum)
    d = 1-d
    np.fill_diagonal(d, 0.)
    if prune:
        cond = np.mean(d)
        zero = np.zeros_like(d)
        return np.where(d > cond, zero, d)
    else:
        return d
    
    
def get_networx_graph(seeds, model, k = 3, m = 3):
    types, tokens = nodes(seeds, model, m = m)
    delta, labels = graph(types, tokens, model)
    delta = dist_prune(delta)
    delta = delta * 10  # scale
    dt = [("len", float)]
    delta = delta.view(dt)

    #  Graphviz
    G = nx.from_numpy_matrix(delta)
    G = nx.relabel_nodes(G, dict(enumerate(labels)))
    pos = graphviz_layout(G)
    return G, pos

    
import plotly.graph_objects as go
def plotly_graph(G, pos):
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
        mode='markers',
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
                #title='<br>Network graph made with Python',
                titlefont_size=16,
                showlegend=False,
                #showscale = False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
    fig.show()