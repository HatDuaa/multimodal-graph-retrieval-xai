"""Verify the fresh graph reranker reproduces the frozen three-channel reference on val."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.graph_store import GraphStore  # noqa: E402
from src.eval.metrics import evaluate  # noqa: E402
from src.models.graph_reranker import GraphReranker  # noqa: E402


def main() -> None:
    cfg = json.loads((Path("configs") / "default.yaml").read_text()) if False else None
    store = GraphStore()
    model = GraphReranker(adaptive_weights=False).eval()
    reference = np.load("experiments/checks/three_channel_probe_reference_scores.npz")
    candidate = {str(caption): ids.tolist() for caption, ids in zip(reference["caption_ids"], reference["candidate_ids"])}
    max_diff = {key: 0.0 for key in ("objects", "triples", "z_objects", "z_triples", "graph", "fused")}
    for row, caption_id in enumerate(reference["caption_ids"]):
        caption_id = str(caption_id)
        ids = candidate[caption_id]
        query = store.query(caption_id)
        images = [store.image(int(image_id)) for image_id in ids]
        clip = torch.from_numpy(reference["clip"][row])
        with torch.no_grad():
            batch = {'queries': store.batch_graphs([query]), 'images': store.batch_graphs(images),
                     'candidate_index': torch.arange(len(ids), dtype=torch.long)[None], 'clip': clip[None],
                     'sentence': store.sentence(caption_id)[None]}
            fused, details = model(batch)
            fused = fused.squeeze(0)
        checks = {"objects": details["objects"].numpy(), "triples": details["triples"].numpy(),
                  "z_objects": details["z_objects"].numpy(), "z_triples": details["z_triples"].numpy(),
                  "graph": details["graph"].numpy(), "fused": fused.numpy()}
        for key, actual in checks.items():
            expected = reference[key][row]
            max_diff[key] = max(max_diff[key], float(np.nanmax(np.abs(actual - expected))))
            if not np.allclose(actual, expected, atol=1e-4, equal_nan=True):
                raise SystemExit(f"first-200 mismatch: caption={caption_id}, row={row}, channel={key}, "
                                 f"max_abs={np.nanmax(np.abs(actual - expected))}")
        if not np.array_equal(np.argsort(-checks["fused"], kind="stable"),
                              np.argsort(-reference["fused"][row], kind="stable")):
            raise SystemExit(f"first-200 order mismatch: caption={caption_id}, row={row}")
    print("first_200_max_abs_diff", json.dumps(max_diff))
    print("reference_first_200_order", "identical")
    arrays = store.candidates("val")
    ranked, golds = [], []
    for row in range(len(arrays["caption_ids"])):
        caption_id = str(arrays["caption_ids"][row])
        ids = [int(image_id) for image_id in arrays["candidate_ids"][row]]
        images = [store.image(image_id) for image_id in ids]
        with torch.no_grad():
            batch = {'queries': store.batch_graphs([store.query(caption_id)]), 'images': store.batch_graphs(images),
                     'candidate_index': torch.arange(len(ids), dtype=torch.long)[None],
                     'clip': torch.from_numpy(arrays["clip_scores"][row])[None],
                     'sentence': store.sentence(caption_id)[None]}
            scores, _ = model(batch)
            scores = scores.squeeze(0)
        order = torch.argsort(scores, descending=True, stable=True).tolist()
        ranked.append([ids[index] for index in order])
        golds.append(int(arrays["gold"][row]))
        if (row + 1) % 1000 == 0:
            print(f"scored {row + 1}/{len(arrays['caption_ids'])}", flush=True)
    print("full_val_metrics", json.dumps(evaluate(ranked, golds)))


if __name__ == "__main__":
    main()
