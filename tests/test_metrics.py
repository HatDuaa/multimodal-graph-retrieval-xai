import json

import pytest

from src.eval.metrics import aggregate_seeds, evaluate, format_row, gold_rank, write_metrics


def test_gold_rank():
    assert gold_rank(["a", "b", "c"], "a") == 1
    assert gold_rank(["a", "b", "c"], "c") == 3
    assert gold_rank(["a", "b", "c"], "z") is None
    assert gold_rank([7, 8, 9], {9, 8}) == 2           # several golds: best rank counts
    assert gold_rank(["ab", "a"], "a") == 2            # a string gold is one id, not a set of characters


def test_evaluate_hand_computed():
    # gold ranks: 1, 3, absent, 6
    ranked = [["g1", "x", "y"], ["x", "y", "g2"], ["x", "y", "z"], list("abcde") + ["g4"]]
    m = evaluate(ranked, ["g1", "g2", "g3", "g4"], ks=(1, 5, 10))
    assert m["n_queries"] == 4
    assert m["recall@1"] == pytest.approx(1 / 4)
    assert m["recall@5"] == pytest.approx(2 / 4)
    assert m["recall@10"] == pytest.approx(3 / 4)
    assert m["mrr"] == pytest.approx((1 + 1 / 3 + 0 + 1 / 6) / 4)


def test_perfect_and_empty_rankings():
    assert evaluate([["g"]], ["g"])["mrr"] == 1.0
    m = evaluate([["x"]], ["g"])
    assert m["recall@10"] == 0.0 and m["mrr"] == 0.0


def test_evaluate_rejects_mismatched_lengths():
    with pytest.raises(AssertionError):
        evaluate([["a"]], ["a", "b"])


def test_aggregate_mean_and_sample_std():
    runs = [{"n_queries": 10, "recall@1": 0.2, "mrr": 0.5},
            {"n_queries": 10, "recall@1": 0.4, "mrr": 0.5},
            {"n_queries": 10, "recall@1": 0.6, "mrr": 0.5}]
    agg = aggregate_seeds(runs)
    assert agg["n_runs"] == 3 and agg["n_queries"] == 10
    assert agg["recall@1"]["mean"] == pytest.approx(0.4)
    assert agg["recall@1"]["std"] == pytest.approx(0.2)   # sample std of 0.2, 0.4, 0.6
    assert agg["mrr"]["std"] == 0.0


def test_aggregate_single_run_has_zero_std():
    assert aggregate_seeds([{"n_queries": 3, "mrr": 0.7}])["mrr"] == {"mean": 0.7, "std": 0.0}


def test_aggregate_rejects_different_query_sets():
    with pytest.raises(AssertionError):
        aggregate_seeds([{"n_queries": 3, "mrr": 0.7}, {"n_queries": 4, "mrr": 0.7}])


def test_write_metrics_schema(tmp_path):
    run = evaluate([["g"], ["x", "g"]], ["g", "g"])
    out = tmp_path / "level1" / "demo" / "metrics.json"
    write_metrics(out, method="clip", dataset="vg_coco", split="test", per_seed={0: run}, config={"top_k": 50})
    data = json.loads(out.read_text(encoding="utf-8"))
    assert set(data) == {"method", "dataset", "split", "per_seed", "aggregate", "config"}
    assert data["per_seed"]["0"]["mrr"] == pytest.approx(0.75)
    assert data["aggregate"]["recall@1"]["mean"] == pytest.approx(0.5)
    assert format_row(data["aggregate"]) == "50.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 75.00 ± 0.00"
