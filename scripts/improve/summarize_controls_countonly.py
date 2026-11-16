"""Summarize the eval-time graph controls of countonly_factual_r3 next to level 1's (main_r3), on val.

Reads experiments/improve/controls/<control>_countonly_factual_r3_seed<k>.json and
experiments/level1/controls/<control>_main_r3_seed<k>.json; per run seed the corrupted metric is the mean over the
three control seeds, then mean +- std (ddof 1) over the three run seeds. Writes experiments/improve/controls/summary.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import REPO_ROOT  # noqa: E402

MODELS = {"countonly_factual_r3": REPO_ROOT / "experiments" / "improve" / "controls",
          "main_r3": REPO_ROOT / "experiments" / "level1" / "controls"}
METRICS = ("recall@1", "recall@5", "recall@10", "mrr")


def main() -> None:
    out = {}
    for model, folder in MODELS.items():
        for control in ("other_image", "rewire"):
            runs = [json.loads((folder / f"{control}_{model}_seed{k}.json").read_text(encoding="utf-8")) for k in range(3)]
            row = {}
            for key in METRICS:
                clean = np.array([100 * r["clean_val"][key] for r in runs])
                corrupted = np.array([100 * np.mean([v[key] for v in r["corrupted_val"].values()]) for r in runs])
                row[key] = {"clean": [float(clean.mean()), float(clean.std(ddof=1))],
                            "corrupted": [float(corrupted.mean()), float(corrupted.std(ddof=1))],
                            "delta": [float((corrupted - clean).mean()), float((corrupted - clean).std(ddof=1))],
                            "corrupted_per_run": corrupted.tolist()}
            out[f"{model}/{control}"] = row
            print(model, control, json.dumps({k: [round(x, 2) for x in row[k]["corrupted"] + row[k]["delta"]]
                                              for k in ("recall@1", "mrr")}), flush=True)
    clip = json.loads((REPO_ROOT / "experiments" / "level1" / "clip_baseline" / "metrics_val.json").read_text(encoding="utf-8"))
    out["clip_only_val"] = {k: 100 * clip["per_seed"]["0"][k] for k in METRICS}
    (MODELS["countonly_factual_r3"] / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
