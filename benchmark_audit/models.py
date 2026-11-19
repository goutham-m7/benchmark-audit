"""Cross-check archived answers with human reviews; never promote a vote to truth."""

from collections import Counter
from math import comb, isclose
from statistics import mean, stdev

import numpy as np

from .board import compare


def inspect_models(items, reviews, models):
    if len(models) != 2:
        raise ValueError("model contrast needs exactly two frozen runs")
    keys = {r["id"]: r["key"] for r in items}
    original, sensitivity = {}, {}
    for name, run in models.items():
        picks = run["choices"]
        if picks.keys() != keys.keys():
            raise ValueError(f"model item roster differs: {name}")
        if any(type(x) is not int or not 0 <= x < 4 for x in picks.values()):
            raise ValueError(f"invalid model choice: {name}")
        correct = {i: int(picks[i] == key) for i, key in keys.items()}
        if not isclose(
            mean(correct.values()), run["published_accuracy"], abs_tol=1e-10
        ):
            raise ValueError(f"replay differs from published accuracy: {name}")
        original[name] = correct
        sensitivity[name] = {
            "original_correct": sum(correct.values()),
            "with_suggestions": sum(
                picks[i] == (reviews[i]["key"] if i in reviews else key)
                for i, key in keys.items()
            ),
        }
    a, b = models
    queue = []
    for i, key in keys.items():
        other = models[a]["choices"][i]
        if other == key or other != models[b]["choices"][i]:
            continue
        if i not in reviews:
            status = "unreviewed"
        else:
            status = {key: "supports_original", other: "supports_alternative"}.get(
                reviews[i]["key"], "supports_neither"
            )
        queue.append({"id": i, "original": key, "alternative": other, "review": status})
    return {
        "a": a,
        "b": b,
        "original": compare(original[a], original[b], 1, comparisons=1),
        "review_sensitivity": sensitivity,
        "consensus_queue": queue,
        "queue_review_counts": dict(Counter(row["review"] for row in queue)),
    }


def rescore(vote_rows, models, strata):
    scores = {}
    for name, run in models.items():
        grades = [
            [int(v == run["choices"][r["id"]]) for v in r["votes"]] for r in vote_rows
        ]
        rates = [mean(col) for col in zip(*grades, strict=True)]
        split = [sum(row) * (len(row) - sum(row)) / comb(len(row), 2) for row in grades]
        scores[name] = {
            "accuracy_by_run": rates,
            "score_sd": stdev(rates),
            "flipped_items": sum(x > 0 for x in split),
            "verdict_pair_disagreement": mean(split),
            "mean_within_item_variance": mean(split) / 2,
            "variance_ci95": stratified_interval([v / 2 for v in split], strata),
        }
    return scores


def stratified_interval(values, strata):
    """Percentile bootstrap within strata, keeping each question's votes together."""
    if not values or len(values) != len(strata):
        raise ValueError("need a stratum for each judgment row")
    rng = np.random.default_rng(17)
    means = np.zeros(10_000)
    for group in sorted(set(strata)):
        sample = [v for v, label in zip(values, strata, strict=True) if label == group]
        for start in range(0, len(means), 128):
            block = means[start : start + 128]
            block += rng.choice(sample, size=(len(block), len(sample))).sum(
                axis=1
            ) / len(values)
    return np.quantile(means, [0.025, 0.975]).tolist()
