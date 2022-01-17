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
    df = (
        df.groupby(["Årstal", "tokens"])
        .sum()
        .rename({"counts": "Ordbrug"}, axis="columns")
        .reset_index()
    )
    df = df.merge(word_use, how="left", on="Årstal").rename(
        {
            "counts": "Yearly word use",
            "tokens": "Word",
            "Ordbrug": "Word use",
            "Årstal": "Year",
        },
        axis="columns",
    )
    df["Word use %"] = (df["Word use"] / df["Yearly word use"]) * 100
    if plot_type == "absolute":
        y = "Word use"
    else:
        y = "Word use %"
    fig = px.line(data_frame=df, x="Year", y=y, color="Word")
    return fig


def plot_word_occurance(df):
    df = df.drop("Årstal", "columns").groupby("tokens").sum().reset_index()  # \
    # .rename({"counts": "Antal af forekomster i alt", "tokens": "Ord"}, axis = "columns")
    fig = px.bar(
        df,
        "tokens",
        "counts",
        labels={"tokens": "Word", "counts": "Number of occurences"},
    )
    return fig
