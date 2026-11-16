import json, sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / "multimodal-graph-retrieval-xai"))
import torch
from src.data.graph_store import GraphStore
from src.train_graph_reranker import build_model, evaluate_val, structure_options
from src.utils.config import load_config
run = Path("experiments/improve/full-graph/fullgraph_factual_r3_seed0")
options = json.loads((run / "config.json").read_text())["args"]
state = torch.load(run / "checkpoint_best.pt", map_location="cuda", weights_only=False)
plain = build_model(options, "cuda"); plain.load_state_dict(state["model"]); plain.eval()
counted = build_model({**options, "count_table": True}, "cuda"); counted.load_state_dict(state["model"], strict=False); counted.eval()
for config in sys.argv[1:]:
    store = GraphStore(load_config(config), splits=("val",), structure=structure_options(options))
    for name, model in (("plain", plain), ("zero", counted)) * 3:
        m = evaluate_val(model, store, store.candidates("val"), "cuda")
        print(config, name, repr(m["recall@1"]), repr(m["mrr"]), flush=True)
