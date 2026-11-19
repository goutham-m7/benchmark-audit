import json
import subprocess
import sys
from pathlib import Path


def test_replay_ignores_precomputed_metrics_and_rejects_wrong_cohort(tmp_path):
    # Raw votes are real; the tampered summaries below are synthetic.
    source = json.loads(Path("reports/smollm2.json").read_text())
    source["disagreement"] = -1
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(source))
    command = [
        sys.executable,
        "-m",
        "benchmark_audit",
        "judge",
        "--replay",
        str(receipt),
    ]
    run = subprocess.run(command, check=False, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    data = json.loads(run.stdout)
    assert 0 < data["disagreement"] <= 1 and data["split_items"] == 11
    assert data["judge"] == source["judge"]
    source["benchmark_sha256"] = "wrong"
    receipt.write_text(json.dumps(source))
    bad = subprocess.run(command, check=False, capture_output=True, text=True)
    assert bad.returncode != 0 and "snapshot" in bad.stderr
