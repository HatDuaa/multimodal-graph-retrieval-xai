"""Demo with the raw-fusion count-table model: exact score parts, count notes, FACTUAL parsing of one caption."""
from types import SimpleNamespace

import pytest
import torch

from src.explain.graph_explainer import matched_pairs
from src.explain.score_breakdown import split_raw_score
from src.graph.counts import CAPTION_CLASSES, IMAGE_BUCKETS
from src.models.graph_reranker import GraphReranker, fill_known_mean
from tests.test_full_graph import fake_store


def test_raw_parts_add_up_to_the_final_score():
    a, clip = .5, .3
    terms = {"objects": [.06, .02], "triples": []}
    fills = {"objects": None, "triples": .01}             # the image has no triple: filled channel part
    fused = a * clip + .06 + .02 + .01
    out = split_raw_score(fused, a, clip, terms, fills)
    assert out["clip_part"] + out["graph_part"] == pytest.approx(fused)
    assert out["graph_part"] == pytest.approx(.09)
    shares = [out["clip_share"], *out["row_shares"]["objects"], out["fill_shares"]["triples"]]
    assert sum(shares) == pytest.approx(1.)
    assert out["fill_shares"]["objects"] is None
    negative = split_raw_score(-.1, a, -.4, {"objects": [.1], "triples": []}, {"objects": None, "triples": None})
    assert negative["clip_share"] is None and negative["row_shares"]["objects"] == [None]


def count_model_and_batch():
    store = fake_store({"knn_edges": 2, "global_node": True})
    store.queries["q"]["objects"][1]["count_class"] = CAPTION_CLASSES.index(">=2")
    for obj in store.graphs[1]["objects"]:
        obj["count_bucket"] = IMAGE_BUCKETS.index("2") if obj["name"] == "ball" else 0
    store.graphs[2]["objects"][0]["count_bucket"] = 0
    torch.manual_seed(5)
    model = GraphReranker(dropout=0., fusion="raw", count_table=True, completion_edges=True, global_node=True,
                          edge_geometry=True).eval()
    with torch.no_grad():
        model.count_table.normal_(0, .1)
    return store, model, store.batch(["q"])


def test_channel_terms_and_filled_channels_add_up_to_the_fused_score():
    store, model, batch = count_model_and_batch()
    with torch.no_grad():
        fused, detail = model(batch, return_encoded=True)
    weights = detail["weights"][0].tolist()
    query_parts = store.graph_parts(store.queries["q"], query=True)
    filled = {key: fill_known_mean(detail[key])[0] for key in ("objects", "triples")}
    for k, image_id in enumerate(batch["candidate_ids"][0].tolist()):
        image_parts = store.graph_parts(store.graphs[image_id])
        with torch.no_grad():
            pairs = matched_pairs(model, detail, query_parts, image_parts, int(batch["candidate_index"][0, k]))
        fills = {key: weights[c] * float(filled[key][k]) if not torch.isfinite(detail[key][0, k]) else None
                 for c, key in ((1, "objects"), (2, "triples"))}
        out = split_raw_score(float(fused[0, k]), weights[0], float(batch["clip"][0, k]),
                              {key: [row["channel_term"] for row in pairs[key]] for key in ("objects", "triples")}, fills)
        total = out["clip_part"] + sum(sum(v) for v in out["row_parts"].values()) + sum(
            v for v in out["fill_parts"].values() if v is not None)
        assert total == pytest.approx(float(fused[0, k]), abs=1e-6)
        # the global node is never matched; only labelled relations are triple rows
        assert len(pairs["objects"]) == 2 and all(row["image_index"] < len(store.graphs[image_id]["objects"])
                                                  for row in pairs["objects"])
        assert len(pairs["triples"]) == (1 if image_id == 1 else 0)
    assert fills["triples"] is not None                     # image 2 has no relation: its triple part is filled
    ball = next(row for row in pairs["objects"] if row["query_count"] == ">=2")
    assert ball["count_bias"] == pytest.approx(float(model.counts_table()[CAPTION_CLASSES.index(">=2"), 0]))


def test_count_notes_only_for_stated_counts():
    pytest.importorskip("gradio")
    from app.demo_app import count_notes
    assert count_notes({"query_count": "unknown", "image_count": "1", "count_bias": 0.}) == ("", "")
    query, image = count_notes({"query_count": "2", "image_count": "3-4", "count_bias": -.088})
    assert "số lượng 2" in query and "số lượng 3-4" in image and "g = −0.088" in image
    assert count_notes({}) == ("", "")


def test_one_caption_is_not_truncated_by_its_own_length():
    pytest.importorskip("factual_scene_graph")
    from src.graph.caption_parsers import FactualParser
    calls = {}

    class Tokenizer:
        pad_token_id = 0

        def encode(self, text, add_special_tokens=True):
            return text.split() + (["</s>"] if add_special_tokens else [])

        def __call__(self, texts, max_length, truncation, return_tensors):
            calls["max_length"], calls["tokens"] = max_length, len(self.encode(texts[0]))
            ids = torch.ones(1, min(max_length, calls["tokens"]), dtype=torch.long)
            return SimpleNamespace(to=lambda device: {"input_ids": ids, "attention_mask": ids})

        def batch_decode(self, ids, skip_special_tokens, clean_up_tokenization_spaces):
            return ["( dog , on , grass )"]

    model = SimpleNamespace(generate=lambda *args, **kwargs: torch.zeros(1, 3, dtype=torch.long))
    parser = FactualParser.__new__(FactualParser)
    parser.parser, parser.beam_size = SimpleNamespace(tokenizer=Tokenizer(), model=model, device="cpu"), 5
    assert parser.generate_one("a dog on grass") == "( dog , on , grass )"
    assert calls["max_length"] == calls["tokens"] == 8      # "Generate Scene Graph:" + 4 words + end token
    parser.generate_one(" ".join(["word"] * 80))
    assert calls["max_length"] == 64
