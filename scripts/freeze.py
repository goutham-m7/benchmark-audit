# /// script
# requires-python = ">=3.11"
# dependencies = ["pyarrow>=18,<25"]
# ///
"""Regenerate the normalized fixtures. Abort if any upstream bytes changed."""
import ast
import csv
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen

import pyarrow.parquet as pq

OUT = Path(__file__).resolve().parents[1] / "data"
MMLU = "https://huggingface.co/datasets/cais/mmlu/resolve/c30699e8356da336a370243923dbaf21066bb9fe/high_school_statistics/"
REDUX = "https://huggingface.co/datasets/edinburgh-dawg/mmlu-redux/resolve/3720db6aeb3d019de48bf37916c1a54074ff4997/high_school_statistics/test.csv"
BOARD = "https://openbenchmarks.com/api/benchmarks/lookalikes"
HASHES = {
    "dev": "3b7c5d8b0c6c682e922ed199113f1ceef5bc4906ad75eb283d2eba5a4205d8e3",
    "validation": "cb74547d5d9603438b19122af969fee81e2cdc641e23a4c3d1305f182ed06452",
    "test": "4922828bda3ae4c4470e95d8fa809e3abe18f6239047b21d675267ebefb1c691",
    "redux": "1c64757bca5acade8c6984eddb5165343ac3cf283b51bb5ce4dfdbb656caed8d",
    "board": "09328dd05f0a60dcccb57462208e2736b900bcbb873492e3ba6905ed25128cb9",
}


def fetch(url, key):
    with urlopen(url, timeout=45) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != HASHES[key]:
        raise ValueError(f"upstream changed: {key}; review before freezing a new release")
    return data


def text(s):
    # Redux CSV re-encoded UTF-8 bytes as Latin-1 and added whitespace.
    try:
        s = s.encode("latin1").decode("utf8")
    except UnicodeError:
        pass
    return " ".join(s.split())


def identity(q, choices):
    return (text(q), *(text(v) for v in choices))


def freeze():
    splits = {}
    for split in ("dev", "validation", "test"):
        buf = fetch(MMLU + split + "-00000-of-00001.parquet", split)
        splits[split] = [dict(id=f"{split}/{i}", text=r["question"], choices=r["choices"], key=r["answer"])
                         for i, r in enumerate(pq.read_table(io.BytesIO(buf)).to_pylist())]
    items = splits["test"]
    lookup = {identity(r["text"], r["choices"]): r for r in items}
    if len(lookup) != len(items):
        raise ValueError("ambiguous test identities")
    redux, reviews = [], {}
    for i, row in enumerate(csv.DictReader(io.StringIO(fetch(REDUX, "redux").decode()))):
        stem, *choices = identity(row["question"], ast.literal_eval(row["choices"]))
        original = lookup[(stem, *choices)]
        if original["id"] in reviews or original["key"] != int(row["answer"]):
            raise ValueError(f"duplicate or conflicting join at Redux row {i}")
        redux.append(dict(id=f"redux/{i}", text=stem))
        error = row["error_type"]
        if error not in ("ok", "wrong_groundtruth"):
            raise ValueError("unexpected taxonomy in pinned statistics slice")
        suggested = original["key"] if error == "ok" else int(float(row["correct_answer"]))
        if suggested not in range(4):
            raise ValueError("invalid annotation")
        reviews[original["id"]] = dict(source=f"redux/{i}", status=error, key=suggested)
    public = json.loads(fetch(BOARD, "board"))
    providers = {}
    for row in public["results"]:
        cells = {c["seed_slug"]: c["relevant_count_at_10"] for c in row["cells"]}
        if len(cells) != len(row["cells"]) or set(cells) != {s["seed_slug"] for s in public["seeds"]}:
            raise ValueError("incomplete seed roster")
        providers[row["provider_slug"]] = {"cells": cells, "published_pct": row["avg_precision_at_10"]}
    return {
        "benchmark": {"items": items, "reviews": reviews, "corpora": {
            "development": [{"id": r["id"], "text": r["text"]} for r in splits["dev"] + splits["validation"]],
            "republication": redux}},
        "board": {"k": 10, "snapshot": public["last_updated"], "providers": providers,
                  "contrasts": [["companyenrich", "predictleads"], ["companyenrich", "exa"]]},
    }


if __name__ == "__main__":
    snapshots = freeze()
    OUT.mkdir(exist_ok=True)
    for name, payload in snapshots.items():
        (OUT / f"{name}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
