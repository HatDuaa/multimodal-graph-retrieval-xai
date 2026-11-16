"""Markdown tables for the parser comparison (parts 1b-1c) and the geometry channel (part 2c)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import REPO_ROOT  # noqa: E402

PARSERS = ["sng", "factual-base", "factual-large", "qwen2.5-3b"]
P = REPO_ROOT / "experiments" / "improve" / "parser-compare"
G = REPO_ROOT / "experiments" / "improve" / "geometry-channel"
KEYS = ("recall@1", "recall@5", "recall@10", "mrr")


def pct(x):
    return f"{100 * x:.2f}"


def ms(agg, key):
    return f"{100 * agg[key]['mean']:.2f} ± {100 * agg[key]['std']:.2f}"


def parsers():
    intrinsic = json.loads((P / "intrinsic.json").read_text(encoding="utf-8"))["parsers"]
    lines = ["## 1b. Đo nội tại (val, 10 633 câu)", "",
             "| Parser | câu có ≥1 bộ ba | vật thể/câu | bộ ba/câu | quan hệ bị bỏ | node đại từ | node rỗng | node OOV | "
             "vật thể có trong ảnh đúng | câu có ≥1 bộ ba khớp ảnh đúng | bộ ba khớp ảnh đúng | giây / 1 000 câu |",
             "|---" * 13 + "|"]
    for name in PARSERS:
        r, c = intrinsic[name], intrinsic[name]["coverage"]
        lines.append(f"| {name} | {pct(r['share_with_triple'])}% | {r['objects_mean']:.2f} | {r['triples_mean']:.2f} | "
                     f"{pct(r['relation_drop_share'])}% | {pct(r['pronoun_node_share'])}% | {pct(r['empty_node_share'])}% | "
                     f"{pct(r['oov_node_share'])}% | {pct(c['entity_item_share'])}% | {pct(c['captions_with_triple_hit'])}% | "
                     f"{pct(c['triple_item_share'])}% | {r['seconds_per_1000']:.1f} ({r['device']}) |")
    lines += ["", "## 1c. Probe 3 kênh không train (val, top-50 CLIP)", "",
              "| Parser | R@1 tốt nhất | R@5 | R@10 | MRR | (τ, α, β) tốt nhất | R@1 tại (0,05; 0,6; 0,7) | MRR tại điểm đó | "
              "chỉ vật thể R@1 | chỉ bộ ba R@1 | Δ R@1 tốt nhất so với sng |", "|---" * 11 + "|"]
    probes = {n: json.loads((P / f"probe_{n}.json").read_text(encoding="utf-8")) for n in PARSERS}
    sng_best = probes["sng"]["best"]["recall@1"]
    for name in PARSERS:
        b, ref = probes[name]["best"], probes[name]["reference_point"]
        lines.append(f"| {name} | {pct(b['recall@1'])} | {pct(b['recall@5'])} | {pct(b['recall@10'])} | {pct(b['mrr'])} | "
                     f"({b['temperature']}, {b['alpha']}, {b['beta']}) | {pct(ref['recall@1'])} | {pct(ref['mrr'])} | "
                     f"{pct(probes[name]['objects_only']['recall@1'])} | {pct(probes[name]['triples_only']['recall@1'])} | "
                     f"{100 * (b['recall@1'] - sng_best):+.2f} |")
    lines.append(f"\nChỉ CLIP: R@1 {pct(probes['sng']['clip_only']['recall@1'])}, MRR {pct(probes['sng']['clip_only']['mrr'])}.")
    lines += ["", "## 1c. main_r3 đã train (3 seed), chỉ thay đồ thị câu", "",
              "| Parser | R@1 toàn val | MRR toàn val | R@1 câu có bộ ba (parser đó) | n | R@1 câu có bộ ba theo sng | n |",
              "|---" * 7 + "|"]
    for name in PARSERS:
        r = json.loads((P / f"rerank_main_r3_{name}.json").read_text(encoding="utf-8"))
        a = r["aggregate"]
        lines.append(f"| {name} | {ms(a['all'], 'recall@1')} | {ms(a['all'], 'mrr')} | {ms(a['with_triple'], 'recall@1')} | "
                     f"{r['n_with_triple']} | {ms(a['with_sng_triple'], 'recall@1')} | {r['n_with_sng_triple']} |")
    return lines


def geometry():
    r = json.loads((G / "geometry_channel.json").read_text(encoding="utf-8"))
    lines = ["## 2c. Kênh số lượng + vị trí cộng vào main_r3 (val, 3 seed)", "",
             f"Câu có số đếm: {r['n']['with_count']} ({r['count_items']} mục). Câu có vị từ không gian: "
             f"{r['n']['with_spatial']} ({r['spatial_items']} bộ ba; theo loại {r['spatial_kinds']}). Tỉ lệ cặp "
             f"(câu, ứng viên) kiểm được: " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in r["candidates_checkable"].items()) + ".", ""]
    for feature, entry in r["features"].items():
        best = str(entry["best_weight_on_val"])
        lines += [f"### Kênh `{feature}`", "", "| w | R@1 toàn val | MRR toàn val | R@1 câu có số đếm | MRR | R@1 câu có vị từ không gian | MRR |",
                  "|---" * 7 + "|"]
        for w in ("0.0", best) if best != "0.0" else ("0.0",):
            s = entry["sweep"][w]
            lines.append(f"| {w}{' (tốt nhất trên val)' if w == best and w != '0.0' else ''} | {ms(s['all'], 'recall@1')} | {ms(s['all'], 'mrr')} | "
                         f"{ms(s['with_count'], 'recall@1')} | {ms(s['with_count'], 'mrr')} | {ms(s['with_spatial'], 'recall@1')} | "
                         f"{ms(s['with_spatial'], 'mrr')} |")
        t = entry["two_fold"]
        lines.append(f"| chọn w trên nửa kia (2-fold) | {ms(t['all'], 'recall@1')} | {ms(t['all'], 'mrr')} | {ms(t['with_count'], 'recall@1')} | "
                     f"{ms(t['with_count'], 'mrr')} | {ms(t['with_spatial'], 'recall@1')} | {ms(t['with_spatial'], 'mrr')} |")
        lines += ["", "R@1 toàn val theo w: " + ", ".join(f"{w}: {100 * entry['sweep'][w]['all']['recall@1']['mean']:.2f}"
                                                      for w in entry["sweep"]) + ".", ""]
    return lines


def main() -> None:
    (P / "summary.md").write_text("\n".join(parsers()) + "\n", encoding="utf-8")
    (G / "summary.md").write_text("\n".join(geometry()) + "\n", encoding="utf-8")
    print((P / "summary.md").read_text(encoding="utf-8"))
    print((G / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
