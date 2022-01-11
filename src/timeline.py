import pandas as pd
import plotly.express as px

def plot_word_use(word_use, token_table, words, genres = []):
    if not genres:
        genres = ["Writing", "Letter", "Sermon"]
    df = token_table[token_table["tokens"].isin(words)]
    df = df[df["Genre"].isin(genres)]\
        .groupby(["Årstal", "tokens"])\
        .sum()\
        .rename({"counts": "Ordbrug"}, axis = "columns")\
        .reset_index()
    df = df.merge(word_use, how = "left", on = "Årstal")\
        .rename({"counts": "Årlig ordbrug", "tokens": "Ord"}, axis = "columns")
    df["Ordbrug %"] = (df["Ordbrug"] / df["Årlig ordbrug"])*100
    fig = px.line(data_frame = df, x = "Årstal", y = "Ordbrug", color = "Ord")
    return fig

def plot_word_occurance(token_table, words, genres = []):
    if not genres:
        genres = ["Writing", "Letter", "Sermon"]
    df = token_table[token_table["tokens"].isin(words)]
    df = df[df["Genre"].isin(genres)]\
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