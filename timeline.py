import pandas as pd


def load_timeline(csv_path):
    df = pd.read_csv(csv_path, sep=";", encoding="utf-8")
    df["tid"] = df["tid"].astype(float)
    return df, df["tid"].values, df["ord"].values, df["plats"].values, df["ratt"].values
