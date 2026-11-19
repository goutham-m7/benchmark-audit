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
from urllib.parse import quote
from urllib.request import urlopen

import pyarrow.parquet as pq

OUT = Path(__file__).resolve().parents[1] / "data"
MMLU = "https://huggingface.co/datasets/cais/mmlu/resolve/c30699e8356da336a370243923dbaf21066bb9fe/high_school_statistics/"
REDUX = "https://huggingface.co/datasets/edinburgh-dawg/mmlu-redux/resolve/3720db6aeb3d019de48bf37916c1a54074ff4997/high_school_statistics/test.csv"
BOARD = "https://openbenchmarks.com/api/benchmarks/lookalikes"
TASK = "harness|hendrycksTest-high_school_statistics|5"
RUNS = {
    "Meta-Llama-3-8B": (
        "db0a9a321c769ebd745b28f3509a6f14ee9a3eb5",
        "2024-04-19T01-26-07.544774",
    ),
    "Meta-Llama-3-8B-Instruct": (
        "0bb67ddda727931a9e7428fdab9f3654a737b1a8",
        "2024-04-19T09-19-13.454877",
    ),
}
HASHES = {
    "dev": "3b7c5d8b0c6c682e922ed199113f1ceef5bc4906ad75eb283d2eba5a4205d8e3",
    "validation": "cb74547d5d9603438b19122af969fee81e2cdc641e23a4c3d1305f182ed06452",
    "test": "4922828bda3ae4c4470e95d8fa809e3abe18f6239047b21d675267ebefb1c691",
    "redux": "1c64757bca5acade8c6984eddb5165343ac3cf283b51bb5ce4dfdbb656caed8d",
    "board": "09328dd05f0a60dcccb57462208e2736b900bcbb873492e3ba6905ed25128cb9",
    "Meta-Llama-3-8B": "dd6a634896eb269a8fdd5888cbe97ff53a8ffa7d976cd6af1d2f62e6ac9765f0",
    "Meta-Llama-3-8B.summary": "3ea65d1835eaf17573557fed1ac71735b0d5532a3d107cf3d822194ef35a7155",
    "Meta-Llama-3-8B-Instruct": "d6fcba92eda5b92c079a7cbc0b9603756c9c4da75edc5874b98ac859d8a4985f",
    "Meta-Llama-3-8B-Instruct.summary": "7050e693c97569fc7d930476c71e68b0845efc2010224e46b14cd4954f8b9f47",
}


def fetch(url, key):
    with urlopen(url, timeout=45) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != HASHES[key]:
        raise ValueError(
            f"upstream changed: {key}; review before freezing a new release"
        )
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


def model_predictions(items):
    by_example = {
        r["text"]
        + "\n"
        + "\n".join(
            f"{letter}. {s}" for letter, s in zip("ABCD", r["choices"], strict=True)
        )
        + "\nAnswer:": r
        for r in items
    }
    models = {}
    for model, (revision, date) in RUNS.items():
        base = f"https://huggingface.co/datasets/open-llm-leaderboard-old/details_meta-llama__{model}/resolve/{revision}"
        source = f"{base}/{date}/details_{quote(TASK)}_{date}.parquet"
        summary_url = f"{base}/results_{date}.json"
        accuracy = json.loads(fetch(summary_url, model + ".summary"))["results"][TASK][
            "acc"
        ]
        columns = ["example", "predictions", "metrics", "num_effective_few_shots"]
        rows = pq.read_table(
            io.BytesIO(fetch(source, model)), columns=columns
        ).to_pylist()
        picks = {}
        for row in rows:
            r = by_example[row["example"].strip()]
            logits = row["predictions"]
            if len(logits) != 4 or row["num_effective_few_shots"] != 5:
                raise ValueError("unexpected leaderboard task or option count")
            pick = max(range(4), key=logits.__getitem__)
            if r["id"] in picks or (pick == r["key"]) != row["metrics"]["acc"]:
                raise ValueError(f"per-item leaderboard receipt mismatch: {r['id']}")
            picks[r["id"]] = pick
        if set(picks) != {r["id"] for r in items}:
            raise ValueError("incomplete model cohort")
        models[model] = {
            "choices": picks,
            "published_accuracy": accuracy,
            "source": source,
            "summary": summary_url,
        }
    return models


def freeze():
    splits = {}
    for split in ("dev", "validation", "test"):
        buf = fetch(MMLU + split + "-00000-of-00001.parquet", split)
        splits[split] = [
            {
                "id": f"{split}/{i}",
                "text": r["question"],
                "choices": r["choices"],
                "key": r["answer"],
            }
            for i, r in enumerate(pq.read_table(io.BytesIO(buf)).to_pylist())
        ]
    items = splits["test"]
    lookup = {identity(r["text"], r["choices"]): r for r in items}
    if len(lookup) != len(items):
        raise ValueError("ambiguous test identities")
    redux, reviews = [], {}
    for i, row in enumerate(
        csv.DictReader(io.StringIO(fetch(REDUX, "redux").decode()))
    ):
        stem, *choices = identity(row["question"], ast.literal_eval(row["choices"]))
        original = lookup[(stem, *choices)]
        if original["id"] in reviews or original["key"] != int(row["answer"]):
            raise ValueError(f"duplicate or conflicting join at Redux row {i}")
        redux.append({"id": f"redux/{i}", "text": stem, "choices": choices})
        error = row["error_type"]
        if error not in ("ok", "wrong_groundtruth"):
            raise ValueError("unexpected taxonomy in pinned statistics slice")
        suggested = (
            original["key"] if error == "ok" else int(float(row["correct_answer"]))
        )
        if suggested not in range(4):
            raise ValueError("invalid annotation")
        reviews[original["id"]] = {
            "source": f"redux/{i}",
            "status": error,
            "key": suggested,
        }
    public = json.loads(fetch(BOARD, "board"))
    providers = {}
    for row in public["results"]:
        cells = {c["seed_slug"]: c["relevant_count_at_10"] for c in row["cells"]}
        if len(cells) != len(row["cells"]) or set(cells) != {
            s["seed_slug"] for s in public["seeds"]
        }:
            raise ValueError("incomplete seed roster")
        providers[row["provider_slug"]] = {
            "cells": cells,
            "published_pct": row["avg_precision_at_10"],
        }
    return {
        "benchmark": {
            "items": items,
            "reviews": reviews,
            "models": model_predictions(items),
            "corpora": {
                "development": [
                    {"id": r["id"], "text": r["text"], "choices": r["choices"]}
                    for r in splits["dev"] + splits["validation"]
                ],
                "republication": redux,
            },
        },
        "board": {
            "k": 10,
            "snapshot": public["last_updated"],
            "providers": providers,
            "contrasts": [["companyenrich", "predictleads"], ["companyenrich", "exa"]],
        },
    }


if __name__ == "__main__":
    snapshots = freeze()
    OUT.mkdir(exist_ok=True)
    for name, payload in snapshots.items():
        (OUT / f"{name}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        )
