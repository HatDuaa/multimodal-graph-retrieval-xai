"""Upload files to the private HF dataset repo of the MKG-W outputs.

Used by notebooks/mkgw_molab.py right after each NativE run ends, from the same detached
shell as the training, so the checkpoint reaches HF even when the notebook is closed.
The token comes from the HF_TOKEN environment variable (never written to disk).

Usage:  python scripts/push_to_hf.py --repo <user/name> <local> <path_in_repo> [<local> <path_in_repo> ...]
"""
import argparse
import os
import sys
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("pairs", nargs="+", help="local path, path in repo, ...")
    args = ap.parse_args()
    if len(args.pairs) % 2:
        sys.exit("pairs must come as <local> <path_in_repo>")
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
    for local, remote in zip(args.pairs[::2], args.pairs[1::2]):
        for attempt in range(5):             # transient network errors: retry with backoff
            try:
                api.upload_file(repo_id=args.repo, repo_type="dataset", path_or_fileobj=local, path_in_repo=remote)
                print("pushed", local, "->", remote, flush=True)
                break
            except Exception as e:  # noqa: BLE001
                print(f"push failed ({e}); retry {attempt + 1}/5", flush=True)
                time.sleep(30 * (attempt + 1))


if __name__ == "__main__":
    main()
