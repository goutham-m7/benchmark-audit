"""All inline scenarios are synthetic, not benchmark measurements."""

import json
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

import pytest


def doc(id, text):
    return {"id": id, "text": text}


def item(id, text="which dog", key=1):
    return dict(doc(id, text), choices=["cats", "a dog", "fish", "trees"], key=key)


def test_overlap_returns_document_local_evidence_and_ignores_oov():
    from benchmark_audit.overlap import nearest

    first = "the sample contains several coins"
    corpus = [
        doc("c1", first),
        doc("c2", "several coins rolled far away"),
        doc("c3", "red roses bloom"),
    ]
    exact, split, oov = nearest(
        [doc("q", first), doc("q2", first + " rolled far away"), doc("q3", "zxq")],
        corpus,
        n=3,
        dimensions=2,
    )
    assert exact["lexical"] == {"id": "c1", "score": 1.0}
    assert exact["embedding"]["id"] == "c1" and exact["embedding"][
        "score"
    ] == pytest.approx(1)
    assert 0 < split["lexical"]["score"] < 1  # do not union shingles across documents
    assert oov["embedding"] == oov["lexical"] == {"id": None, "score": 0}


def test_labels_separate_unreviewed_from_reviewed_agreement():
    from benchmark_audit.audit import labels

    out = labels(
        [item("a"), item("b"), item("c")],
        {"a": {"source": "r1", "key": 0}, "b": {"source": "r2", "key": 1}},
    )
    assert (out["reviewed"], out["agree"], out["unreviewed"]) == (2, 1, 1)
    assert out["disputes"] == [
        {"id": "a", "source": "r1", "original": 1, "proposed": 0}
    ]


def test_judge_scatter_uses_within_item_pairs_and_repeated_calls():
    from benchmark_audit.judges import lexical, repeat

    def variable(stem, choices, seed):
        return int(stem == "vary" and seed == 1)

    out = repeat(
        [item("a", "vary"), item("b", "stable")], variable, repeats=3, first_seed=0
    )
    assert out["split_items"] == 1 and out["disagreement"] == pytest.approx(1 / 3)
    assert out["by_item"][0]["votes"] == [0, 1, 0]
    assert repeat([item("a")], lexical, repeats=3, first_seed=0)["disagreement"] == 0
    with pytest.raises(ValueError):
        repeat([item("a")], lexical, repeats=1, first_seed=0)


def test_http_judge_sends_only_choices_and_rejects_invalid_verdict():
    from benchmark_audit.judges import local_judge

    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append(
                json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            )
            content = json.dumps({"answer": "B" if len(seen) == 1 else "E"})
            self.send_response(200)
            self.end_headers()
            self.wfile.write(
                json.dumps({"choices": [{"message": {"content": content}}]}).encode()
            )

    with HTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            judge = local_judge(server.server_port, "test-model")
            assert judge("pick", ["one", "two", "three", "four"], 7) == 1
            with pytest.raises(ValueError):
                judge("pick", ["one", "two", "three", "four"], 8)
        finally:
            server.shutdown()
            thread.join()
    envelope = seen[0]
    assert envelope["model"] == "test-model" and envelope["seed"] == 7
    assert json.loads(envelope["messages"][-1]["content"]) == {
        "question": "pick",
        "choices": dict(zip("ABCD", ["one", "two", "three", "four"])),
    }


def test_board_compares_same_seeds_not_unpaired_scores():
    from benchmark_audit.board import compare

    a = {"s": 10, "t": 8, "missing": None}
    b = {"s": 8, "t": 6, "missing": 10}
    out = compare(a, b, 10, comparisons=2)
    assert out["paired_seeds"] == 2 and out["excluded"] == ["missing"]
    assert out["gap_pp"] == pytest.approx(20) and out["ci_pp"] == pytest.approx(
        [20, 20]
    )
    assert out["interval_level"] == 0.975
    equal = compare({"x": 1, "y": 0}, {"x": 0, "y": 1}, 1, comparisons=1)
    assert not equal["distinguishable"] and equal["ci_pp"][0] < 0 < equal["ci_pp"][1]
    with pytest.raises(ValueError):
        compare({"s": 1}, {"t": 1}, 1, comparisons=1)


def test_cli_replays_public_snapshot(tmp_path):
    target = tmp_path / "report.json"
    run = subprocess.run(
        [sys.executable, "-m", "benchmark_audit", "audit", "--output", str(target)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    out = json.loads(target.read_text())
    assert out["labels"]["reviewed"] == 100 and out["labels"]["unreviewed"] == 116
    assert len(out["labels"]["disputes"]) == 2
    assert out["overlap"]["development"]["documents"] == 28
    assert out["board"]["scores"]["companyenrich"]["judged"] == 48
    assert out["board"]["scores"]["ocean"]["judged"] == 47
    assert len(out["board"]["contrasts"]) == 2
