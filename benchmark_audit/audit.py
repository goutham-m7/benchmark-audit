"""Assemble human-review queues without modifying an observed label."""

from hashlib import sha256

from .board import leaderboard
from .judges import lexical, repeat
from .overlap import nearest


def validate(benchmark):
    items = benchmark["items"]
    ids = {r["id"] for r in items}
    if not items or len(ids) != len(items):
        raise ValueError("items must be nonempty and have unique IDs")
    for r in items:
        if (
            not isinstance(r["text"], str)
            or not r["text"].strip()
            or len(r["choices"]) != 4
            or not all(isinstance(x, str) for x in r["choices"])
        ):
            raise ValueError(f"invalid four-choice question: {r['id']}")
        if type(r["key"]) is not int or not 0 <= r["key"] < 4:
            raise ValueError(f"invalid key: {r['id']}")
    if benchmark["reviews"].keys() - ids:
        raise ValueError("review without a matching item")
    for row in benchmark["reviews"].values():
        if type(row["key"]) is not int or not 0 <= row["key"] < 4:
            raise ValueError("invalid review key")


def labels(items, reviews):
    disputes = [
        {
            "id": r["id"],
            "source": reviews[r["id"]]["source"],
            "original": r["key"],
            "proposed": reviews[r["id"]]["key"],
        }
        for r in items
        if r["id"] in reviews and r["key"] != reviews[r["id"]]["key"]
    ]
    return {
        "reviewed": len(reviews),
        "unreviewed": len(items) - len(reviews),
        "agree": len(reviews) - len(disputes),
        "disputes": disputes,
    }


def full_item(r):
    return tuple(" ".join(s.split()) for s in (r["text"], *r["choices"]))


def audit(benchmark, board):
    validate(benchmark)
    items = benchmark["items"]
    overlap = {}
    for name, corpus in benchmark["corpora"].items():
        rows = nearest(items, corpus)
        reference_items = {full_item(r) for r in corpus if "choices" in r}
        exact = [r["id"] for r in items if full_item(r) in reference_items]
        flags = {
            key: {r["id"] for r in rows if r[key]["score"] >= threshold}
            for key, threshold in [("lexical", 0.8), ("embedding", 0.95)]
        }
        overlap[name] = {
            "documents": len(corpus),
            "lexical_flags": len(flags["lexical"]),
            "exact_stem_and_choices_ids": exact,
            "embedding_flags": len(flags["embedding"]),
            "either_flags": len(set.union(*flags.values())),
            "by_item": rows,
        }
    result = {
        "items": len(items),
        "overlap_config": {
            "n": 5,
            "dimensions": 16,
            "lexical_threshold": 0.8,
            "embedding_threshold": 0.95,
        },
        "overlap": overlap,
        "labels": labels(items, benchmark["reviews"]),
        "lexical_judge": repeat(items, lexical, repeats=3, first_seed=17),
        "board": leaderboard(board),
    }
    if "models" in benchmark:
        from .models import inspect_models

        result["model_board"] = inspect_models(
            items, benchmark["reviews"], benchmark["models"]
        )
    return result


def reviewed_sample(benchmark, limit):
    """Enrich for known disputes, then take fixed hash order; not a prevalence sample."""
    reviews = benchmark["reviews"]
    population = [r for r in benchmark["items"] if r["id"] in reviews]
    if not 1 <= limit <= len(population):
        raise ValueError(
            f"judge limit must be between 1 and {len(population)} reviewed items"
        )
    return sorted(
        population,
        key=lambda r: (
            reviews[r["id"]]["key"] == r["key"],
            sha256(r["id"].encode()).digest(),
        ),
    )[:limit]
