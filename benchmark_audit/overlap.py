"""Rank nearest reference documents with independent lexical and LSA signals."""

import re
from collections import Counter, defaultdict

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

WORDS = re.compile(r"\w+", re.UNICODE)


def tokens(text):
    return WORDS.findall(text.casefold())


def shingles(text, n):
    words = tokens(text)
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def nearest(queries, corpus, *, n=5, dimensions=16):
    if min(n, dimensions) < 1 or not corpus:
        raise ValueError("need a nonempty corpus and positive dimensions and n")
    refs = [r["text"] for r in corpus]
    ids = [r["id"] for r in corpus]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate corpus IDs")
    postings = defaultdict(list)
    for index, text in enumerate(refs):
        for gram in shingles(text, n):
            postings[gram].append(index)
    vectorizer = TfidfVectorizer(analyzer=tokens, sublinear_tf=True)
    mat = vectorizer.fit_transform(refs)
    # A one-term vocabulary already is one-dimensional. Never fit on the test set.
    if mat.shape[1] == 1:
        refs_z = normalize(mat.toarray())
        proj = None
    else:
        proj = TruncatedSVD(n_components=min(dimensions, *mat.shape), random_state=17)
        with threadpool_limits(limits=1):
            refs_z = normalize(proj.fit_transform(mat))
    results = []
    for q in queries:
        grams = shingles(q["text"], n)
        counts = Counter(i for g in grams for i in postings.get(g, ()))
        top = min(counts, key=lambda i: (-counts[i], i)) if counts else None
        vec = vectorizer.transform([q["text"]])
        z = vec.toarray() if proj is None else proj.transform(vec)
        scores = refs_z @ normalize(z)[0]
        idx = int(np.argmax(scores))
        similarity = float(np.clip(scores[idx], 0, 1))
        results.append(
            {
                "id": q["id"],
                "lexical": {
                    "id": ids[top] if top is not None else None,
                    "score": round(counts[top] / len(grams), 8)
                    if top is not None
                    else 0.0,
                },
                "embedding": {
                    "id": ids[idx] if similarity > 1e-10 else None,
                    "score": round(similarity, 8) if similarity > 1e-10 else 0.0,
                },
            }
        )
    return results
