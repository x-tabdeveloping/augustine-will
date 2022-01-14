import pandas as pd
import plotly.express as px

def filter_tokens(token_table, tokens, genres, works):
    if not genres:
        genres = ["Writing", "Letter", "Sermon"]
    df = token_table[token_table["Genre"].isin(genres)]
    df = df[df["tokens"].isin(tokens)]
    df = df[df["Forkortelse"].isin(works)]
    return df

def plot_word_use(df, word_use, plot_type):
    df = df\
        .groupby(["Årstal", "tokens"])\
        .sum()\
        .rename({"counts": "Ordbrug"}, axis = "columns")\
        .reset_index()
    df = df.merge(word_use, how = "left", on = "Årstal")\
        .rename({"counts": "Årlig ordbrug", "tokens": "Ord"}, axis = "columns")
    df["Ordbrug %"] = (df["Ordbrug"] / df["Årlig ordbrug"])*100
    if plot_type == "absolute":
        y = "Ordbrug"
    else:
        y = "Ordbrug %"
    fig = px.line(data_frame = df, x = "Årstal", y = y, color = "Ord")
    return fig

def plot_word_occurance(df):
    df = df\
        .drop("Årstal", "columns")\
        .groupby("tokens")\
        .sum()\
        .reset_index()#\
        #.rename({"counts": "Antal af forekomster i alt", "tokens": "Ord"}, axis = "columns")
    fig = px.bar(
        df,
        "tokens",
        "counts",
        labels={
            "tokens": "Ord",
            "counts": "Antal af forekomster i alt"
        }
    )
    return fig