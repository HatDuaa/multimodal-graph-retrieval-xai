import pytest

from src.explain.score_breakdown import split_final_score


def test_parts_add_up_and_graph_part_is_shared_by_contribution():
    a, z_clip, graph = 0.5, 0.2, 1.1
    fused = a * z_clip + (1 - a) * graph
    out = split_final_score(fused, a, z_clip, graph, {"objects": [0.3, 0.1], "triples": [0.1]},
                            {"objects": True, "triples": True})
    assert out["clip_part"] + out["graph_part"] == pytest.approx(fused)
    assert out["clip_share"] + out["graph_share"] == pytest.approx(1.0)
    assert out["row_parts"]["objects"] == pytest.approx([0.33, 0.11])
    assert out["row_parts"]["triples"] == pytest.approx([0.11])
    shares = out["row_shares"]["objects"] + out["row_shares"]["triples"]
    assert sum(shares) == pytest.approx(out["graph_share"])


def test_negative_clip_part_keeps_its_sign():
    a, z_clip, graph = 0.4, -0.5, 1.0
    fused = a * z_clip + (1 - a) * graph                 # 0.4
    out = split_final_score(fused, a, z_clip, graph, {"objects": [0.2], "triples": []},
                            {"objects": True, "triples": False})
    assert out["clip_share"] == pytest.approx(-0.5)
    assert out["graph_share"] == pytest.approx(1.5)
    assert out["row_shares"]["objects"] == pytest.approx([1.5])


def test_inactive_channel_rows_get_no_share():
    out = split_final_score(0.6, 0.4, 0.3, 0.8, {"objects": [0.2, 0.2], "triples": [0.5]},
                            {"objects": True, "triples": False})
    assert out["row_parts"]["triples"] == [None] and out["row_shares"]["triples"] == [None]
    assert out["row_parts"]["objects"] == pytest.approx([0.24, 0.24])  # inactive 0.5 is left out of the total


def test_non_positive_final_score_has_parts_but_no_percentages():
    out = split_final_score(-0.2, 0.5, -0.6, 0.2, {"objects": [0.1], "triples": [0.1]},
                            {"objects": True, "triples": True})
    assert out["clip_share"] is None and out["graph_share"] is None
    assert out["row_shares"] == {"objects": [None], "triples": [None]}
    assert out["clip_part"] == pytest.approx(-0.3) and out["row_parts"]["objects"] == pytest.approx([0.05])


def test_no_active_positive_contribution_leaves_rows_unassigned():
    out = split_final_score(0.3, 0.6, 0.5, 0.0, {"objects": [], "triples": [0.2]},
                            {"objects": False, "triples": False})
    assert out["row_parts"] == {"objects": [], "triples": [None]}
