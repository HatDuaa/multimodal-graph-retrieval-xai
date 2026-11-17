"""Read the link-prediction metrics that NativE's tester prints at the end of a training log.

NativE (zjukg/NativE c72291c, mmkgc/base/Test.h) prints only the filtered setting, as fractions:

    metric:            MRR       MR       hit@10    hit@3    hit@1
    l(filter):         ...       (head prediction)
    r(filter):         ...       (tail prediction)
    averaged(filter):  ...       (mean of head and tail; the number papers report)

The unfiltered ("raw") printf lines are commented out in the authors' code, so they are not in the log.

Usage:  python scripts/parse_native_log.py --log <training log> [...] --seed 42 [...] --out <metrics.json>
        (one --seed per --log; with several runs the file also holds mean and std over them, ddof 1)
"""
import argparse
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import REPO_ROOT  # noqa: E402

COLUMNS = ("mrr", "mr", "hits@10", "hits@3", "hits@1")
ROWS = {"l(filter)": "head", "r(filter)": "tail", "averaged(filter)": "averaged"}

# Zhang et al., NativE: Multi-modal Knowledge Graph Completion in the Wild, SIGIR 2024 (arXiv 2406.17605v1),
# Table 2, MKG-W columns, row NativE (S+I+T); the row NativE (All) has the same numbers. The paper gives only MRR and
# Hit@1 for MKG-W, in percent.
PAPER_MKGW = {"source": "arXiv 2406.17605v1, Table 2, MKG-W, NativE (S+I+T)", "mrr": 0.3658, "hits@1": 0.2956,
              "mr": None, "hits@3": None, "hits@10": None, "note": "null = not reported in the paper"}

HYPER_PARAMETERS = {"batch_size": 1024, "margin": 4, "epoch": 1000, "dim": 250, "neg_num": 128, "mu": 0.0001,
                    "learning_rate": 1e-4, "lrg": 1e-4}


def parse_link_prediction(text: str) -> dict:
    """The last block of filtered head, tail and averaged metrics in a NativE log."""
    out = {}
    for line in text.splitlines():
        parts = line.strip().split()
        name = parts[0].rstrip(":") if parts else ""
        if name in ROWS:
            values = [float(v) for v in parts[1:]]
            if len(values) != len(COLUMNS):
                raise ValueError(f"unexpected metric line: {line!r}")
            out[ROWS[name]] = dict(zip(COLUMNS, values))
    missing = set(ROWS.values()) - set(out)
    if missing:
        raise ValueError(f"log has no {sorted(missing)} metrics (run unfinished?)")
    return out


def parse_command(text: str) -> str:
    """The training command the driver writes as the first line of the log ('$ ...')."""
    first = text.splitlines()[0] if text else ""
    match = re.match(r"^\$ (.*)$", first)
    return match.group(1) if match else ""


def aggregate(runs: list[dict]) -> dict:
    """Mean and sample std (ddof 1) of every metric over runs."""
    return {k: {"mean": statistics.mean(r[k] for r in runs), "std": statistics.stdev(r[k] for r in runs),
                "n_runs": len(runs)} for k in COLUMNS}


def relative(path: Path) -> str:
    path = path.resolve()
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, nargs="+", required=True)
    ap.add_argument("--seed", type=int, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dataset", default="MKG-W")
    args = ap.parse_args()

    if len(args.log) != len(args.seed):
        ap.error("give one --seed per --log")
    texts = [log.read_text(encoding="utf-8", errors="replace") for log in args.log]
    per_seed = {str(seed): parse_link_prediction(text) for seed, text in zip(args.seed, texts)}
    report = {
        "method": "NativE (zjukg/NativE c72291c, run_adv_wgan_gp.py)",
        "dataset": args.dataset,
        "split": "kgc_test (NativE test2id.txt, 4274 triples)",
        "setting": "filtered link prediction, no type constraint, as printed by NativE's tester",
        "per_seed": per_seed,
        "paper": PAPER_MKGW,
        "hyper_parameters": HYPER_PARAMETERS,
        "commands": [parse_command(text) for text in texts],
        "logs": [relative(log) for log in args.log],
    }
    if len(per_seed) > 1:
        report["aggregate"] = aggregate([m["averaged"] for m in per_seed.values()])
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for seed, metrics in per_seed.items():
        print(seed, json.dumps(metrics["averaged"]))


if __name__ == "__main__":
    main()
