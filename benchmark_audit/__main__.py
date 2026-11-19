import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

from .audit import audit, reviewed_sample, validate
from .judges import RUBRIC, TEMPERATURE, lexical, local_judge, repeat


ROOT = Path(__file__).resolve().parents[1]


def load(path):
    data = path.read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Audit benchmark evidence without rerunning a vendor.")
    sub = parser.add_subparsers(dest="command", required=True)
    full = sub.add_parser("audit", help="replay the complete deterministic audit")
    full.add_argument("--board", type=Path, default=ROOT/"data/board.json")
    judge = sub.add_parser("judge", help="repeat blind adjudications on the reviewed cohort")
    judge.add_argument("--model", help="opt in to a local llama.cpp model (otherwise lexical control)")
    judge.add_argument("--port", type=int, default=8080)
    judge.add_argument("--limit", type=int, default=12)
    judge.add_argument("--repeats", type=int, default=3)
    for p in (full,judge):
        p.add_argument("--benchmark", type=Path, default=ROOT/"data/benchmark.json")
        p.add_argument("--output", type=Path, help="save all per-item evidence; default stdout")
    args = parser.parse_args()
    try:
        data, digest = load(args.benchmark)
        if args.command == "audit":
            board, board_digest = load(args.board)
            output = audit(data, board)
            output['board_sha256'] = board_digest
        else:
            validate(data)
            sample = reviewed_sample(data, args.limit)
            judge_fn = local_judge(args.port, args.model) if args.model else lexical
            output = repeat(sample, judge_fn, repeats=args.repeats, first_seed=17)
            output['judge'] = {"name":args.model, "rubric":RUBRIC, "temperature":TEMPERATURE} if args.model else {"name":"lexical-jaccard"}
            output['selection'] = 'reviewed disputes first, then SHA-256 order by item ID'
            output['review_key_accuracy_by_run'] = [mean(row['votes'][i] == data['reviews'][row['id']]['key']
                                                        for row in output['by_item']) for i in range(args.repeats)]
        output['benchmark_sha256'] = digest
        encoded = json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
        if args.output:
            args.output.write_text(encoded)
        else:
            print(encoded,end='')
    except (ValueError,KeyError,OSError) as e:
        parser.exit(2, f"audit failed: {e}\n")


if __name__ == "__main__":
    main()
