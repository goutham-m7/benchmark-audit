"""Paired seed bootstrap on per-task top-K counts, never aggregate percentages."""
from statistics import mean

import numpy as np


DRAWS = 20_000
SEED = 73


def valid_count(x, k):
    return x is None or (type(x) is int and 0 <= x <= k)


def compare(a, b, k, *, comparisons):
    if k < 1 or comparisons < 1 or any(not valid_count(x, k) for r in (a, b) for x in r.values()):
        raise ValueError("invalid top-K counts or comparison parameters")
    union = set(a) | set(b)
    pairs = sorted(s for s in union if a.get(s) is not None and b.get(s) is not None)
    if len(pairs) < 2:
        raise ValueError("need at least two matched seeds")
    delta = np.array([(a[s] - b[s]) * 100/k for s in pairs])
    rng = np.random.default_rng(SEED)
    # Chunk draws so workspace does not grow as all draws × number of tasks.
    means = np.empty(DRAWS)
    for start in range(0, DRAWS, 128):
        block = means[start:start+128]
        block[:] = rng.choice(delta, size=(len(block), len(delta))).mean(axis=1)
    tail = .025/comparisons
    lo, hi = np.quantile(means, [tail, 1-tail])
    return {"paired_seeds": len(pairs), "excluded": sorted(union - set(pairs)),
            "gap_pp": float(delta.mean()), "interval_level": 1-2*tail,
            "ci_pp": [float(lo), float(hi)], "distinguishable": bool(lo > 0 or hi < 0),
            "wins_ties_losses": [int((delta>0).sum()), int((delta==0).sum()), int((delta<0).sum())]}


def leaderboard(board):
    k, providers, contrasts = board['k'], board['providers'], board['contrasts']
    if type(k) is not int or k < 1 or not contrasts:
        raise ValueError("positive integer K and at least one contrast required")
    scores = {}
    for name, row in providers.items():
        cells = row['cells']
        if not all(valid_count(x,k) for x in cells.values()):
            raise ValueError(f"invalid cell: {name}")
        counts = [x for x in cells.values() if x is not None]
        pct = mean(counts)*100/k
        if abs(pct-row['published_pct']) > .00500001:
            raise ValueError(f"recomputed score disagrees with published value: {name}")
        scores[name] = {"pct": pct, "judged": len(counts), "total": len(cells)}
    return {"snapshot": board['snapshot'], "k": k, "scores": scores,
            "resamples": DRAWS, "seed": SEED, "family_level": .95,
            "contrasts": [dict(a=a,b=b,**compare(providers[a]['cells'], providers[b]['cells'],
                                                k, comparisons=len(contrasts))) for a,b in contrasts]}
