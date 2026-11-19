"""Synthetic contracts for real archived predictions and disputed labels."""

import pytest


def test_disagreement_queue_needs_human_review_not_majority_rule():
    from benchmark_audit.models import inspect_models

    items = [{"id": str(i), "key": 0} for i in range(4)]
    base = {"choices": dict(zip("0123", [1, 1, 2, 0])), "published_accuracy": 0.25}
    tuned = {"choices": dict(zip("0123", [1, 1, 2, 1])), "published_accuracy": 0}
    reviews = {"0": {"key": 1}, "1": {"key": 0}}
    out = inspect_models(items, reviews, {"base": base, "tuned": tuned})
    assert out["consensus_queue"] == [
        {"id": "0", "original": 0, "alternative": 1, "review": "supports_alternative"},
        {"id": "1", "original": 0, "alternative": 1, "review": "supports_original"},
        {"id": "2", "original": 0, "alternative": 2, "review": "unreviewed"},
    ]
    assert out["original"]["gap_pp"] == pytest.approx(25)
    assert out["review_sensitivity"]["base"] == {
        "original_correct": 1,
        "with_suggestions": 2,
    }
    with pytest.raises(ValueError, match="published"):
        inspect_models(
            items,
            reviews,
            {"base": dict(base, published_accuracy=0.75), "tuned": tuned},
        )
    with pytest.raises(ValueError, match="roster"):
        inspect_models(
            items, reviews, {"base": dict(base, choices={"0": 1}), "tuned": tuned}
        )


def test_regrade_holds_the_response_fixed_between_judge_runs():
    from benchmark_audit.models import rescore

    votes = [{"id": "0", "votes": [1, 2, 2]}, {"id": "1", "votes": [1, 1, 1]}]
    out = rescore(votes, {"frozen": {"choices": {"0": 1, "1": 2}}}, [True, False])[
        "frozen"
    ]
    assert out["accuracy_by_run"] == [0.5, 0, 0]
    assert out["flipped_items"] == 1
    assert out["verdict_pair_disagreement"] == pytest.approx(1 / 3)
    # Both singleton strata are fixed in this synthetic bootstrap.
    assert out["variance_ci95"] == pytest.approx([1 / 6, 1 / 6])
