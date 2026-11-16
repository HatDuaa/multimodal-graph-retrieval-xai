"""Demo node matching: top-3 image parts per explanation row, boxes over the image, caption ↔ image match graph."""
import re
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from src.explain.graph_explainer import matched_pairs
from src.explain.match_view import (has_boxes, image_frame, image_part, match_graph_svg, ordered_rows, overlay_svg,
                                    valid_box)
from src.explain.score_breakdown import split_raw_score
from src.service.search_service import Hit
from tests.test_demo_countonly import count_model_and_batch


def all_pairs():
    store, model, batch = count_model_and_batch()
    with torch.no_grad():
        fused, detail = model(batch, return_encoded=True)
        query_parts = store.graph_parts(store.queries["q"], query=True)
        ids = batch["candidate_ids"][0].tolist()
        return store, [(i, store.graph_parts(store.graphs[i]),
                        matched_pairs(model, detail, query_parts, store.graph_parts(store.graphs[i]),
                                      int(batch["candidate_index"][0, k]))) for k, i in enumerate(ids)]


def test_top_matches_are_ordered_by_a_times_sim_with_shares_of_the_row_score():
    store, pairs = all_pairs()
    for image_id, image_parts, result in pairs:
        for channel, key in (("objects", "node_indices"), ("triples", "relation_indices")):
            for row in result[channel]:
                matches = row["top_matches"]
                assert len(matches) == min(3, len(image_parts[key]))          # image 2 has a single object
                terms = [m["a_ij"] * m["sim_ij"] for m in matches]
                assert terms == sorted(terms, reverse=True)
                assert all(m["image_index"] in image_parts[key] for m in matches)
                assert len({m["image_index"] for m in matches}) == len(matches)
                assert matches[0]["image_index"] == row["image_index"]       # positive sims: best by sim = best by a × sim
                for m, term in zip(matches, terms):
                    assert m["share"] == pytest.approx(term / row["row_score"])
                    assert 0 <= m["share"] <= 1 + 1e-6
                assert sum(m["share"] for m in matches) <= 1 + 1e-6
                if channel == "objects":                                     # count-table model: each match has its g
                    assert all({"image_count", "count_bias"} <= set(m) for m in matches)


def test_top_k_is_capped_by_the_parts_of_the_image():
    store, model, batch = count_model_and_batch()
    with torch.no_grad():
        _, detail = model(batch, return_encoded=True)
        query_parts = store.graph_parts(store.queries["q"], query=True)
        one = matched_pairs(model, detail, query_parts, store.graph_parts(store.graphs[1]),
                            int(batch["candidate_index"][0, 0]), top_k=1)
    assert all(len(row["top_matches"]) == 1 for channel in ("objects", "triples") for row in one[channel])


GRAPH = {"width": 200, "height": 100,
         "objects": [{"id": 0, "name": "dog", "box": [10, 10, 50, 40]},
                     {"id": 1, "name": "frisbee", "box": [80, 5, 20, 10]},
                     {"id": 2, "name": "window", "box": [100, 0, 90, 90], "boxes": 9},       # merged node
                     {"id": 3, "name": "grass"},                                              # no box
                     {"id": 4, "name": "sky", "box": [0, 0, 0, 30]}],                         # empty box
         "relations": [{"subject": 0, "predicate": "catching", "object": 1},
                       {"subject": 0, "predicate": "on", "object": 3},
                       {"subject": 0, "predicate": "[none]", "object": 1}]}


def test_image_part_labels_and_boxes():
    assert image_part(GRAPH, "objects", 0) == {"part": "dog", "box": (10., 10., 50., 40.), "merged": 1}
    assert image_part(GRAPH, "objects", 2)["merged"] == 9
    assert image_part(GRAPH, "objects", 3)["box"] is None and image_part(GRAPH, "objects", 4)["box"] is None
    assert image_part(GRAPH, "objects", 5) is None                         # the global node is not an object
    assert image_part(GRAPH, "triples", 0) == {"part": ("dog", "catching", "frisbee"),
                                               "subject_box": (10., 10., 50., 40.), "object_box": (80., 5., 20., 10.)}
    assert image_part(GRAPH, "triples", 1)["object_box"] is None
    assert image_part(GRAPH, "triples", 2) is None                          # kNN [none] edges are never matched
    assert image_part({"objects": [{"id": 0, "name": "dog"}], "relations": []}, "objects", 0)["box"] is None
    assert valid_box(None) is None and valid_box([1, 2, 3]) is None and valid_box([0, 0, "a", 1]) is None


def match(index, a, sim, channel="objects", graph=GRAPH):
    return {"image_index": index, "a_ij": a, "sim_ij": sim, "share": None, **(image_part(graph, channel, index) or {})}


def rows_info(triples=True):
    info = {"objects": [{"query_part": "dog", "w_i": .6, "channel_term": .2, "contribution": .2,
                         "top_matches": [match(0, .7, .9), match(2, .2, .5), match(3, .1, .4)]},
                        {"query_part": "frisbee", "w_i": .4, "channel_term": .3, "contribution": .3,
                         "top_matches": [match(5, .9, .8), match(4, .1, .2)]}],          # global node, empty box
            "triples": []}
    if triples:
        info["triples"] = [{"query_part": ("dog", "catching", "frisbee"), "w_i": 1., "channel_term": .1,
                            "contribution": .1,
                            "top_matches": [match(0, .5, .9, "triples"), match(1, .3, .7, "triples")]}]
    return info


@pytest.fixture
def frame(tmp_path):
    path = tmp_path / "7.jpg"
    Image.new("RGB", (400, 200), "white").save(path)                       # file at twice the graph's size
    return image_frame(path, (200, 100))


def test_image_frame_scales_boxes_to_the_file_and_needs_a_size(tmp_path, frame):
    assert frame["size"] == (400, 200) and frame["scale"] == (2., 2.)
    assert frame["href"].startswith("data:image/jpeg;base64,")
    assert image_frame(tmp_path / "7.jpg", [None, None]) is None             # level-1 graphs have no size
    assert image_frame(tmp_path / "missing.jpg", (200, 100)) is None


def test_rows_are_numbered_in_table_order():
    numbered = ordered_rows(rows_info(), raw=True)
    assert [(n, channel, index) for n, channel, index, _ in numbered] == [(1, "objects", 1), (2, "objects", 0),
                                                                          (3, "triples", 0)]


def test_overlay_skips_parts_without_box_and_draws_merged_nodes_and_triples(frame):
    svg = overlay_svg(ordered_rows(rows_info(), raw=True), frame, "7")
    assert svg.count('class="mk alt"') == 2                                # window, dog –on→ grass; grass and sky have no box
    assert svg.count('data-row="1"') == 0                                  # frisbee row: global node and empty box only
    assert svg.count('<g class="mk" data-row="2">') == 2                   # dog: best box + badge
    assert '<rect x="20.0" y="20.0" width="100.0" height="80.0"' in svg     # dog box scaled ×2 to the file
    assert "node gộp 9 hộp" in svg
    assert svg.count("<line ") == 1 and 'marker-end="url(#mx-arrow-7)"' in svg   # grass has no box: no arrow to it
    assert 'id="mx-img-7"' in svg


def test_overlay_and_graph_without_triples_or_boxes(frame):
    numbered = ordered_rows(rows_info(triples=False), raw=True)
    assert "<line " not in overlay_svg(numbered, frame, "7")
    plain = match_graph_svg(numbered)                                      # level 1: no frame, no crops
    assert plain and "<use " not in plain
    with_crops = match_graph_svg(numbered, frame, "7")
    assert with_crops.count('<use href="#mx-img-7"/>') == 2                # dog, window (grass, sky: no usable box)
    assert match_graph_svg([]) == "" and not has_boxes([])


def test_match_graph_joins_each_row_to_its_top_matches():
    svg = match_graph_svg(ordered_rows(rows_info(), raw=True))
    edges = re.findall(r'<path class="mk" data-row="(\d+)"', svg)
    assert sorted(edges) == ["1", "1", "2", "2", "2", "3", "3"]
    # one node per image part per channel: dog (objects) and dog –catching→ frisbee (triples) are different nodes
    assert len(re.findall(r'data-rows="', svg)) == 7
    widths = [float(w) for w in re.findall(r'stroke-width="([\d.]+)" stroke-linecap', svg)]
    assert max(widths) == pytest.approx(8.) and min(widths) >= .8


def fake_mode(store, model):
    from src.service.graph_mode import GraphRerankMode
    mode = GraphRerankMode.__new__(GraphRerankMode)
    mode.encoder = SimpleNamespace(encode_texts=lambda texts: F.normalize(torch.randn(len(texts), 512), dim=-1).numpy())
    mode.device, mode.fusion, mode.model = torch.device("cpu"), "raw", model
    mode.structure = {"knn_edges": 2, "global_node": True}
    mode.graphs, mode.image_parts, mode.extra = store.graphs, {}, {}
    mode.table, mode.row = store.part_vectors, store.part_row
    mode.cfg = {"clip": {"concept_prompt": "a photo of {}"}}
    mode.parse = lambda query: store.queries["q"]
    return mode


def test_demo_rows_keep_exact_totals_and_draw_boxes(tmp_path):
    pytest.importorskip("gradio")
    from app.demo_app import explanation_view
    store, model, batch = count_model_and_batch()
    for image_id, graph in store.graphs.items():
        Image.new("RGB", (graph["width"], graph["height"])).save(tmp_path / f"{image_id}.jpg")
    hits = [Hit(rank=k + 1, image_id=i, score=0., clip_score=float(batch["clip"][0, k]),
                image_path=tmp_path / f"{i}.jpg", captions=[]) for k, i in enumerate(batch["candidate_ids"][0].tolist())]
    with torch.no_grad():
        hits = fake_mode(store, model)("q", hits)
    for hit in hits:
        info = hit.explanation[0]
        a = info["weights_abc"][0]
        numbered = ordered_rows(info, raw=True)
        split = split_raw_score(hit.score, a, hit.clip_score,
                                {key: [row["channel_term"] for row in info[key]] for key in ("objects", "triples")},
                                info["fills"])
        # the numbered rows pick the same parts the table sums: CLIP + rows + filled channels = final score
        parts = [split["row_parts"][channel][index] for _, channel, index, _ in numbered]
        assert parts == [row["channel_term"] for _, _, _, row in numbered]
        fills = [v for v in split["fill_parts"].values() if v is not None]
        assert split["clip_part"] + sum(parts) + sum(fills) == pytest.approx(hit.score, abs=1e-6)
        deltas = [row["delta"] for _, _, _, row in numbered] + [v for v in info["fill_deltas"].values() if v is not None]
        assert info["clip_delta"] + sum(deltas) == pytest.approx(info["delta_total"], abs=1e-6)
        assert info["image_size"] == [store.graphs[hit.image_id]["width"], store.graphs[hit.image_id]["height"]]
        html = explanation_view(info, hit)
        assert 'class="mx-image"' in html and 'class="mx-graph"' in html
        for number, channel, _, row in numbered:
            assert f'<tr class="ch-{"obj" if channel == "objects" else "tri"}' in html
            assert f'data-row="{number}"' in html and row["top_matches"][0]["part"] is not None
    image_1 = next(hit for hit in hits if hit.image_id == 1).explanation[0]
    assert image_1["triples"][0]["top_matches"][0]["subject_box"] == (0., 0., 10., 10.)
    assert all(m["box"] is not None for row in image_1["objects"] for m in row["top_matches"])
    image_2 = next(hit for hit in hits if hit.image_id == 2)
    assert "<line " not in explanation_view(image_2.explanation[0], image_2)  # no triple in image 2: no arrow


def test_level1_graph_without_boxes_gets_a_note_instead_of_the_overlay(tmp_path):
    pytest.importorskip("gradio")
    from app.demo_app import NO_BOXES, explanation_view
    level1 = {"objects": [{"id": 0, "name": "dog"}], "relations": []}
    info = {"weights_abc": [.5, .3, .2], "fusion": "zscore", "z_clip": .4, "channel_active": [True, False],
            "image_size": [None, None], "triples": [], "unmatched": {"objects": [], "triples": []},
            "objects": [{"query_part": "dog", "image_part": "dog", "image_index": 0, "w_i": 1., "a_ij": 1., "sim_ij": .9,
                         "contribution": .27, "row_score": .9, "channel_term": .27,
                         "top_matches": [match(0, 1., .9, graph=level1)]}]}
    Image.new("RGB", (20, 20)).save(tmp_path / "3.jpg")
    hit = Hit(rank=1, image_id=3, score=.5, clip_score=.3, image_path=tmp_path / "3.jpg", captions=[], graph_score=.2)
    html = explanation_view(info, hit)
    assert NO_BOXES in html and "mx-image" not in html and 'class="mx-graph"' in html
    assert 'data-row="1"' in html
