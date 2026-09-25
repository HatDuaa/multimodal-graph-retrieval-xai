import json

from src.data.build_mkgw_split import commons_url, mask_name, split_entities, to_image_entry


def test_mask_name_removes_label_and_aliases():
    assert mask_name("Linda Darnell was an American actress", ["Linda Darnell"]) == \
        "was an American actress"
    assert mask_name("film by John Ford", ["John Ford", "Ford"]) == "film by"


def test_mask_name_is_case_insensitive_and_word_bounded():
    assert mask_name("PARIS is the capital", ["Paris"]) == "is the capital"
    # "Ford" must not eat a substring of another word
    assert mask_name("afford a car", ["Ford"]) == "afford a car"


def test_mask_name_handles_missing_and_empty():
    assert mask_name("American actress", [None]) == "American actress"
    assert mask_name("Solo", ["Solo"]) == ""


def test_mask_name_cleans_leftover_punctuation():
    assert mask_name("Berlin, capital of Germany", ["Berlin"]) == "capital of Germany"


def test_commons_url_quotes_spaces():
    url = commons_url("Linda Darnell - publicity.JPG", 640)
    assert " " not in url and url.endswith("?width=640")


def _fake_rows(n):
    return [{"qid": f"Q{i}", "id": i, "label_en": f"Name{i}", "aliases_en": [],
             "desc_en": f"Name{i} thing number {i}", "image": f"F{i}.jpg"} for i in range(n)]


def test_split_entities_fractions_and_determinism():
    rows = _fake_rows(100)
    fr = {"train": 0.8, "val": 0.1, "test": 0.1}
    a = split_entities(rows, fr, seed=0)
    b = split_entities(rows, fr, seed=0)
    assert [r["qid"] for r in a["train"]] == [r["qid"] for r in b["train"]]
    assert len(a["train"]) == 80 and len(a["val"]) == 10 and len(a["test"]) == 10
    ids = [r["qid"] for s in a.values() for r in s]
    assert len(set(ids)) == 100                       # no entity in two splits
    assert split_entities(rows, fr, seed=1)["train"] != a["train"]


def test_to_image_entry_schema_and_masked_query():
    entry = to_image_entry(_fake_rows(3)[2], "en", 640)
    assert entry["image_id"] == 2 and entry["qid"] == "Q2"
    assert entry["captions"] == [{"caption_id": "Q2_0", "text": "thing number 2"}]
    json.dumps(entry)                                 # serialisable


def test_export_projection_matches_torch_sequential():
    import torch

    from scripts.export_native_embeddings import project

    torch.manual_seed(0)
    seq = torch.nn.Sequential(torch.nn.Linear(8, 6), torch.nn.ReLU(), torch.nn.Linear(6, 6))
    sd = {f"p.{k}": v for k, v in seq.state_dict().items()}
    x = torch.randn(5, 8)
    assert torch.allclose(project(x, sd, "p"), seq(x), atol=1e-6)


def test_to_image_entry_url_is_none_without_p18():
    row = _fake_rows(1)[0]
    row["image"] = None
    assert to_image_entry(row, "en", 640)["url"] is None
