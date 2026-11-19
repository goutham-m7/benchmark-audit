# Benchmark Audit

**If two models choose the same “wrong” answer, should you change the key?** On 216 MMLU statistics questions, Llama 3 8B and its Instruct sibling agreed against the published key 63 times. Human reviewers had examined 29 of those cases. They backed the models on **2**, not 29. Both corrections increased both models’ scores, leaving their ranking gap unchanged.

This CLI finds overlap, queues disputed keys, regrades frozen answers and measures uncertainty on leaderboard gaps. Everything below uses **real public inputs and observed runs**, not invented model outputs. The two leaderboards measure different things: one ranks language models answering questions; the other ranks search APIs judged by an LLM panel. No vendor was rerun. Local inference and data downloads cost **$0 in API fees**.

## Reproduce without credentials

From the checkout, with [uv](https://docs.astral.sh/uv/):

```sh
uv sync --locked                       # first install needs network or a primed cache
uv --offline run pytest
uv --offline run python -m benchmark_audit audit --output /tmp/audit.json
uv --offline run python -m benchmark_audit judge --replay reports/smollm2.json --output /tmp/judge.json
jq '.model_board | del(.consensus_queue)' /tmp/audit.json
jq '{split_items, disagreement, disagreement_ci95, frozen_response_grades}' /tmp/judge.json
```

The second command-line audit reads raw votes, **not the receipt’s precomputed metrics**. It rejects a changed benchmark hash or item roster. Start with the checked-in [audit report](reports/audit.json), [local model votes](reports/smollm2.json), [lexical control](reports/lexical-sample.json) or [source links, attribution and changes](data/NOTICE.md). The data refresh command, `uv run scripts/freeze.py`, is separate and online: it checks every source’s SHA-256, matches questions **and ordered options**, validates both Llama runs’ 432 original correctness bits, and refuses changed upstream bytes. The snapshot still works if a live API changes.

## 1. Contamination signals: test the detector against known copies

All 216 high-school statistics test questions from [MMLU](https://huggingface.co/datasets/cais/mmlu) were compared independently with two corpora. The query was the stem alone: adding A/B/C/D choices could artificially inflate similarity. A second, stricter check compared normalized stems **plus the ordered option texts**.

| Search corpus | Documents | 5-gram flags | Embedding flags | Either | Exact stem + choices |
|---|---:|---:|---:|---:|---:|
| Same subject’s dev + validation splits | 28 | 3 | 7 | 7 | 0 |
| MMLU-Redux, a known republication | 100 | 101 | 104 | 104 | 100 |

The lexical score is the fraction of a query’s **distinct contiguous five-word shingles** found in a single reference document (flag ≥0.80). Separately, TF-IDF is fitted **only on the reference corpus**, then reduced with truncated SVD to at most 16 dense dimensions (LSA); cosine ≥0.95 flags the closest document. The JSON retains both independent nearest IDs and scores. Matches cannot be stitched across documents. There is no pretrained embedding model to download; this corpus-learned representation is sensitive to vocabulary, small corpora and templates, not a general paraphrase detector.

**What an investigator sees:** `test/139` and `validation/15` have the identical stem “Which of the following is a true statement?” and different options. Conversely, `test/204` vs `dev/1` is worth examining: both ask for the expectation and variance of a sum, with changed numbers (cosine 0.9981, 5-gram containment 0.50). The test even lists E(X) twice. Neither pair is an identical full question. On the republication control both detectors retrieved a hit for all 100 known copies; the union also flagged 4 other questions. **Overlap is observable; proprietary training-set membership and memorization are not.** These thresholds are review triggers, not calibrated error probabilities.

## 2. Answer-key disputes: consensus is a queue, not a correction

Freeze the archived five-shot [base](https://huggingface.co/datasets/open-llm-leaderboard-old/details_meta-llama__Meta-Llama-3-8B) and [Instruct](https://huggingface.co/datasets/open-llm-leaderboard-old/details_meta-llama__Meta-Llama-3-8B-Instruct) answer choices. Flag an item if both chose the **same alternative** to the original key. Only then cross-check the independently produced [MMLU-Redux review](https://huggingface.co/datasets/edinburgh-dawg/mmlu-redux). “OK” means not flagged by that review, not a second blinded vote.

- Of 100 reviewed items, 98 retained their original key; 2 had an explicit alternative. **116/216 had no review** in this release.
- Both reviewer suggestions appeared in the 63-item consensus queue. Of its 29 reviewed cases, 2 supported the suggested alternative (6.9%) and 27 retained the original. The other 34 queue entries remain unresolved. The models share a family and many errors; do not infer the unreviewed error rate from this subset.
- `test/38`: original **A** claims experiments only use human subjects; the models and Redux choose **B**, manipulation versus observation. `test/181`: original **D** excludes a superintendent because their salary is negotiable; Redux and both models choose **B**, a small sample from a skewed salary distribution. We expose the disagreements; we do not silently alter MMLU.

## 3. Which ranking gaps survive uncertainty?

| Public score and contrast (A − B) | Matched units | A vs B | Gap, pp | Paired interval, pp | Distinguishable? |
|---|---:|---:|---:|---:|---|
| MMLU statistics: Llama 3 8B base vs Instruct | 216 questions | 55.56% vs 53.70% | +1.85 | [−3.70, +7.41] | No |
| Lookalikes P@10: CompanyEnrich vs PredictLeads | 48 company seeds | 91.875% vs 89.792% | +2.08 | [−8.33, +13.54] | No |
| Lookalikes P@10: CompanyEnrich vs Exa | 48 company seeds | 91.875% vs 80.833% | +11.04 | [+2.50, +19.58] | Yes, under this sampling model |

For MMLU, base alone got 20 right, Instruct alone 16, and they tied on correctness 180 times. Substituting **only** the two reviewer proposals changes the numerators from 120/116 to 122/118, still a four-question gap. For [Openbenchmarks Lookalikes](https://openbenchmarks.com/lookalikes), we reproduce **all ten** published P@10 scores from the October 16 snapshot: divide each judged seed’s relevant count by **10**, never by returned count. A missing judgment is not zero (four providers have 47/48 scores). Contrasts take the intersection and name exclusions. CompanyEnrich beat PredictLeads on just 8 seeds, tied on 24 and lost on 16: the narrow mean lead is about win size, not frequency.

**Intervals:** 20,000 paired percentile bootstrap draws, seed 73. Resample questions for MMLU, companies for Lookalikes; never treats individual retrieved candidates as independent. MMLU is one 95% contrast. The two named Lookalikes contrasts use Bonferroni-adjusted 97.5% intervals for an approximate 95% family. These are not simultaneous intervals over all 45 provider pairs. The benchmark scores themselves are fixed. The intervals model variation over similar tasks under exchangeability: shared question templates, industry clusters, convenience sampling, serving drift and panel bias limit generalization. Crossing zero means insufficient evidence of separation, **not equivalence**.

## 4. Judge variance: freeze the answer, vary only the adjudicator

On 12 reviewed items (both disputes plus 10 non-disputes selected by fixed ID hash), each judge independently selected the best of the same four options three times. Neither saw the original key, the review, the candidate’s choice or the provider name. Each resulting vote graded both archived Llama answers by exact equality. Candidate outputs never changed. This is a multiple-choice regrading experiment, not a free-text grading benchmark.

| Adjudicator | Items with changing option | Mean pair disagreement (95% interval) | Base response scored correct across 3 runs | Instruct response scored correct across 3 runs |
|---|---:|---:|---|---|
| Deterministic token-Jaccard (highest stem/option similarity, first on ties) | 0/12 | 0% [0, 0] | 1, 1, 1 of 12 | 1, 1, 1 of 12 |
| SmolLM2-360M-Instruct, Q4_K_M, local | 11/12 | 80.6% [63.9%, 94.4%] | 2, 4, 1 of 12 | 1, 5, 1 of 12 |

The deterministic control was perfectly repeatable but agreed with the published keys on just **54/216 (25%)**. For the local model, 29/36 **within-question vote pairs** disagreed on the best option. Final binary grades changed on 7/12 items for each frozen candidate. The mean within-question *sample variance* of those binary grades was **0.194 [0.111, 0.278]** for each model: grade-pair disagreement divided by two. The judge-induced score SD across three runs was **12.7 pp** for base and **19.2 pp** for Instruct. This is instability in the judge, not model performance variance.

Judge intervals use 10,000 percentile bootstrap draws (seed 17), resampling questions within the two review-status strata and keeping each question’s three votes together. They describe the **selected 2:10 mix**, conditional on this prompt, quantization, temperature and observed passes. Twelve selected questions and three seeds give weak information about future judge runs; these are not intervals for a production judge or the whole MMLU dataset. Structured decoding guarantees an A–D letter, not a sound choice. A bad judge can reverse the apparent ordering without any vendor drift.

### Run the judge locally (optional)

The measured run used [llama.cpp b12532](https://github.com/ggml-org/llama.cpp/releases/tag/b12532), commit `08c252e22`, on Apple Silicon; [quantized SmolLM2 artifact](https://huggingface.co/bartowski/SmolLM2-360M-Instruct-GGUF/blob/7be6f65f1db715fe5dc5a4634c0d459b4eed42ec/SmolLM2-360M-Instruct-Q4_K_M.gguf), SHA-256 `2fa3f013dcdd7b99f9b237717fa0b12d75bbb89984cc1274be1471a465bac9c2`. Download that file and a llama.cpp build for your platform; neither is in git. For example, with the executable on PATH:

```sh
llama-server --model SmolLM2-360M-Instruct-Q4_K_M.gguf --alias smollm2-360m-q4 \
  --host 127.0.0.1 --port 18473 -c 2048 -np 1
# In another terminal:
uv --offline run python -m benchmark_audit judge --model smollm2-360m-q4 --port 18473 \
  --limit 12 --repeats 3 --output /tmp/new-judge.json
```

The adapter posts only to numeric loopback. Temperature 0.8, seeds 17–19, fixed option order, JSON-schema constrained output. A non-200 response or invalid vote aborts the run; missing answers are not converted into zeros. Hardware and runtime versions can change sampled votes. Use `--replay` for the exact historical observation.

## Use with your own data

- **Overlap only:** call `benchmark_audit.overlap.nearest(queries, corpus, n=5, dimensions=16)`. Each document needs a unique `id` and `text`. The function returns both nearest IDs/scores for every query. Short stems without five tokens have no lexical hit.
- **Paired gaps only:** `benchmark_audit.board.compare(a, b, k, comparisons=1)` takes two dictionaries of stable task IDs to integer top-K correct counts (or null/missing). Use K=1 for binary correctness. At least two matched tasks are required.
- **End-to-end:** `audit --benchmark PATH --board PATH` accepts the JSON shapes in [benchmark.json](data/benchmark.json) and [board.json](data/board.json). Four-choice keys and predictions are 0-based; keep review evidence separate. With a new source, create new receipts instead of reusing the supplied ones. No hidden-split discovery, training-corpus access, or automatic relabeling.

## Two-minute walkthrough

**0:00–0:30** Open `/tmp/audit.json` after the replay. Show `test/139` and `validation/15`. Exact stem, different choices: a strong overlap score tells us where to look, not what a model trained on.

**0:30–1:00** Open the review queue and `test/38`. Two models correctly challenge A. Now show the other 27 reviewed flags that kept the original key. Agreement is useful for triage but unsafe for automatic correction.

**1:00–1:30** Show the paired intervals: Llama base’s four extra correct answers and CompanyEnrich’s lead over PredictLeads both cross zero; the CompanyEnrich–Exa interval does not. Explain why the unit is the whole question or seed, not a result or judge vote.

**1:30–2:00** Run the offline SmolLM2 receipt replay. Show the same archived Instruct answers scoring 1, 5 and 1 out of 12. No vendor reran. This is a measured warning about one small judge, not a claim about all LLM judges.

Original code: MIT. Upstream rights and attribution: [data/NOTICE.md](data/NOTICE.md).
