import unicodedata

from src.data.vnmmkg_text import (contains_name, make_auto_queries, mask_names, name_variants, split_sentences,
                                  strip_parentheticals)

PREFIXES = ["Nhà thờ chính tòa", "Nhà thờ", "Chùa", "Đền"]


def test_name_variants_prefix_and_disambiguation():
    names = name_variants("Chùa Một Cột", ["Diên Hựu tự", "chùa một cột"], "Chùa Một Cột (Hà Nội)", PREFIXES)
    assert names == ["Chùa Một Cột", "Diên Hựu tự", "Một Cột"]       # longest first, case-insensitive dedupe
    assert name_variants(None, [], None, PREFIXES) == []


def test_name_variants_longest_prefix_wins():
    assert "Đức Bà" in name_variants("Nhà thờ chính tòa Đức Bà", [], None, PREFIXES)


def test_mask_longest_first_and_cleanup():
    names = ["Chùa Một Cột", "Một Cột"]
    assert mask_names("Chùa Một Cột, hay Một Cột, là một ngôi chùa.", names) == "hay, là một ngôi chùa."


def test_mask_respects_word_boundaries():
    assert mask_names("Chùa Keo ở Thái Bình, không phải Keoxx", ["Keo"]) == "Chùa ở Thái Bình, không phải Keoxx"


def test_mask_handles_nfd_and_upper_case():
    text = unicodedata.normalize("NFD", "ĐỀN HÙNG nằm trên núi Nghĩa Lĩnh")
    out = mask_names(text, ["Đền Hùng"])
    assert out == "nằm trên núi Nghĩa Lĩnh"
    assert not contains_name(out, ["Đền Hùng"])
    assert contains_name(unicodedata.normalize("NFD", "thăm đền hùng"), ["Đền Hùng"])


def test_strip_parentheticals_nested():
    assert strip_parentheticals("Chùa Một Cột (chữ Hán: 一柱寺 (Nhất Trụ tự)) là chùa") == "Chùa Một Cột là chùa"


def test_split_sentences_vietnamese():
    text = "Chùa ở TP. Hà Nội. Chùa được xây năm 1049. Diện tích 1.000 m². Nó đẹp! Do ông Nguyễn V. A dựng."
    assert split_sentences(text) == ["Chùa ở TP. Hà Nội.", "Chùa được xây năm 1049.", "Diện tích 1.000 m².",
                                     "Nó đẹp!", "Do ông Nguyễn V. A dựng."]
    assert split_sentences("Năm 1049 vua cho dựng. sau đó sửa lại.") == ["Năm 1049 vua cho dựng. sau đó sửa lại."]
    assert split_sentences("") == []


def test_auto_queries_strip_parenthetical_names():
    extract = ("Chùa Một Cột (chữ Hán: 一柱寺, còn gọi là Diên Hựu tự hay Liên Hoa Đài) là một ngôi chùa nằm "
               "giữa lòng thủ đô Hà Nội.\nChùa có kiến trúc độc đáo hình bông sen mọc lên. Ngắn quá. "
               "Câu thứ tư không được lấy dù đủ dài.")
    names = name_variants("Chùa Một Cột", ["Diên Hựu tự"], "Chùa Một Cột", PREFIXES)
    queries, drops = make_auto_queries("Q1", extract, 77, names, max_sentences=3, min_words=6)
    assert [q["sentence_index"] for q in queries] == [0, 1]
    assert queries[0] == {"query_id": "Q1_auto_0", "qid": "Q1",
                          "text": "là một ngôi chùa nằm giữa lòng thủ đô Hà Nội.",
                          "sentence_index": 0, "viwiki_revid": 77}
    assert drops == {"too_short": 1}
    assert all(not contains_name(q["text"], names) for q in queries)
    assert all("Liên Hoa Đài" not in q["text"] for q in queries)
