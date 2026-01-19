import json
import os
import pandas as pd


PLACE_MAP = {1: "mitten", 2: "Vhörn", 3: "Hhörn"}


def _place_from_index(idx):
    return PLACE_MAP.get(int(idx), "mitten")


def _parse_timestamp(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_csv(path):
    df = pd.read_csv(path, sep=";", encoding="utf-8")
    df["tid"] = df["tid"].astype(float)
    return df


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("lar-i-takt-content", [])
    rows = []
    for item in items:
        base_timestamp = _parse_timestamp(item.get("timestamp")) or 0.0
        question = item.get("question", "")
        question_place = _place_from_index(item.get("place_index", 1))
        rows.append({"tid": base_timestamp, "ord": question, "plats": question_place, "ratt": 0})

        for resp in item.get("responses", []) or []:
            correct_flag = str(resp.get("correct", "")).lower() == "true"
            element = resp.get("element", "")
            place_index = resp.get("place_index", item.get("place_index", 1))
            place = _place_from_index(place_index)
            resp_ts = _parse_timestamp(resp.get("timestamp"))
            default_ts = base_timestamp + (0.4 if int(place_index) in (2, 3) else 0.0)
            ts = resp_ts if resp_ts is not None else default_ts
            rows.append({"tid": ts, "ord": element, "plats": place, "ratt": 1 if correct_flag else 0})

    df = pd.DataFrame(rows, columns=["tid", "ord", "plats", "ratt"])
    return df


def load_timeline(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        df = _load_json(path)
    else:
        df = _load_csv(path)

    return df, df["tid"].values, df["ord"].values, df["plats"].values, df["ratt"].values
