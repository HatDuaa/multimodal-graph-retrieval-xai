import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    mo.md(
        r"""
        # MKG-W on molab — data, CLIP features, NativE training (level 2)

        Runs the GPU-heavy part of level 2 on molab. **Every stage pushes its outputs to a
        private Hugging Face dataset repo as soon as it finishes**, so a disconnect or the
        molab shutdown (12 h per session, 90 min idle) loses nothing. Export of embeddings and
        all evaluations run on the laptop from the HF copy (`scripts/pull_mkgw_from_hf.py`).
        Data steps follow Đạt's scripts on `dat/mkgw-data` (rows 20, 22, 23 of the task sheet);
        this notebook and the leakage mask are Khang's additions.

        **Before you start:** turn on the GPU (notebook specs button) and fill in a Hugging Face
        token with write access. Stages are idempotent: re-running skips finished work.

        | Stage | What | Time (rough) |
        |---|---|---|
        | 1 Setup | clone repo branch + NativE, install, build `Base.so` | 3 min |
        | Restore | pull finished outputs back from HF after a restart | 2–5 min |
        | 2 Data | NativE tables, MMRNS image zip (6.4 GB), Wikidata labels, images, KGC triples → HF | 30–60 min |
        | 3 CLIP | frozen CLIP features for 8 920 images + queries → HF | 5 min |
        | 3b Probe | multilingual CLIP on 84 vi/en MKG-W val queries (row 9) → HF | 10 min |
        | 4 Train | start NativE runs **in the background** (cell returns at once); each run pushes its own checkpoint + log to HF when it ends, even with the notebook closed | hours |
        | 4b Status | progress of each run (optional); also pushes anything a run could not push itself | — |
        | Files | download buttons for every log / JSON produced here | — |
        """
    )
    return (mo,)


@app.cell
def _(mo):
    hf_repo = mo.ui.text(value="duongkhang2100/graphml-mkgw-retrieval", label="HF dataset repo", full_width=True)
    hf_token = mo.ui.text(kind="password", label="HF token (write)", full_width=True)
    branch = mo.ui.text(value="khang/mkgw-level2", label="git branch")
    epochs = mo.ui.number(start=1, stop=2000, value=1000, label="NativE epochs")
    runs = mo.ui.multiselect(
        options=["repro:MKG-W:42", "rmask:MKG-W-rmask:0", "rmask:MKG-W-rmask:1", "rmask:MKG-W-rmask:2"],
        value=["repro:MKG-W:42", "rmask:MKG-W-rmask:0", "rmask:MKG-W-rmask:1", "rmask:MKG-W-rmask:2"],
        label="training runs (tag:dataset:seed), launched in parallel",
    )
    mo.vstack([hf_repo, hf_token, mo.hstack([branch, epochs]), runs])
    return branch, epochs, hf_repo, hf_token, runs


@app.cell
def _():
    import hashlib
    import json
    import shutil
    import subprocess
    import sys
    import zipfile
    from pathlib import Path

    WORK = Path.home() / "work"
    REPO = WORK / "multimodal-graph-retrieval-xai"
    NATIVE = WORK / "NativE"             # matches configs/default.yaml (mkgw.native_repo)
    NATIVE_COMMIT = "c72291c3d571b36c754897935d21718aa4568fe7"
    PY = sys.executable
    LOG_DIR = REPO / "experiments" / "level2" / "native-train"

    def sh(cmd, cwd=None, check=True):
        """Run a shell command, print its output, raise on failure."""
        print(f"$ {cmd}")
        r = subprocess.run(cmd, shell=True, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(r.stdout[-6000:])
        if check and r.returncode != 0:
            raise RuntimeError(f"exit {r.returncode}: {cmd}")
        return r.stdout

    def pip(*pkgs):
        r = subprocess.run([PY, "-m", "pip", "install", "-q", *pkgs], text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if r.returncode != 0:            # uv-managed venv without pip
            sh(f"uv pip install --python {PY} -q " + " ".join(pkgs))

    def sha256(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def run_name(spec, n_epochs):
        tag, dataset, seed = spec.split(":")
        return f"{tag}_seed{seed}_e{n_epochs}", dataset, seed

    return LOG_DIR, NATIVE, NATIVE_COMMIT, PY, Path, REPO, WORK, json, pip, run_name, sh, sha256, shutil, subprocess, zipfile


@app.cell
def _(Path, hf_repo, hf_token):
    def push(local, path_in_repo):
        """Upload one file or folder to the HF dataset repo right away (private repo, created if missing)."""
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token.value)
        api.create_repo(hf_repo.value, repo_type="dataset", private=True, exist_ok=True)
        local = Path(local)
        if local.is_dir():
            api.upload_folder(repo_id=hf_repo.value, repo_type="dataset", folder_path=str(local),
                              path_in_repo=path_in_repo)
        else:
            api.upload_file(repo_id=hf_repo.value, repo_type="dataset", path_or_fileobj=str(local),
                            path_in_repo=path_in_repo)
        print("pushed", local, "->", path_in_repo)

    def remote_files():
        from huggingface_hub import HfApi
        try:
            return set(HfApi(token=hf_token.value).list_repo_files(hf_repo.value, repo_type="dataset"))
        except Exception:
            return set()

    return push, remote_files


@app.cell
def _(mo):
    b_setup = mo.ui.run_button(label="1 · Setup")
    b_restore = mo.ui.run_button(label="Restore from HF")
    b_data = mo.ui.run_button(label="2 · Data")
    b_clip = mo.ui.run_button(label="3 · CLIP features")
    b_probe = mo.ui.run_button(label="3b · Multilingual CLIP probe (row 9)")
    b_train = mo.ui.run_button(label="4 · Start training (background)")
    b_status = mo.ui.run_button(label="4b · Status + push finished runs")
    b_eval_val = mo.ui.run_button(label="5 · Level-2 eval on val")
    b_eval_test = mo.ui.run_button(label="6 · Level-2 test (once, after val)")
    auto = mo.ui.refresh(options=["10m"], default_interval="10m", label="auto status")
    mo.hstack([b_setup, b_restore, b_data, b_clip, b_probe, b_train, b_status, b_eval_val, b_eval_test, auto],
              wrap=True)
    return auto, b_clip, b_data, b_eval_test, b_eval_val, b_probe, b_restore, b_setup, b_status, b_train


@app.cell
def _(NATIVE, NATIVE_COMMIT, REPO, WORK, b_setup, branch, mo, pip, sh):
    mo.stop(not b_setup.value, mo.md("Stage 1 not run yet."))
    WORK.mkdir(parents=True, exist_ok=True)
    if not REPO.exists():
        sh(f"git clone -b {branch.value} https://github.com/HatDuaa/multimodal-graph-retrieval-xai.git {REPO}")
    else:
        sh(f"git fetch origin && git checkout {branch.value} && git pull --ff-only", cwd=REPO)
    if not NATIVE.exists():
        sh(f"git clone https://github.com/zjukg/NativE.git {NATIVE}")
    sh(f"git checkout -q {NATIVE_COMMIT}", cwd=NATIVE)
    pip("open_clip_torch==3.3.0", "faiss-cpu", "gdown", "huggingface_hub", "PyYAML", "requests", "tqdm",
        "pillow", "scikit-learn")
    # the shipped Base.so segfaults on some machines (experiments/level2/native-repro/README.md): always rebuild
    sh("bash make.sh", cwd=NATIVE / "mmkgc", check=False)
    sh("nvidia-smi --query-gpu=name,memory.total --format=csv; nproc; df -h ~", check=False)
    mo.md("Setup done.")
    return


@app.cell
def _(NATIVE, Path, REPO, b_restore, hf_repo, hf_token, mo, shutil, zipfile):
    mo.stop(not b_restore.value, mo.md("Restore not run."))
    from huggingface_hub import snapshot_download as _snapshot_download
    _local = Path(_snapshot_download(hf_repo.value, repo_type="dataset", token=hf_token.value or None,
                                     local_dir=str(Path.home() / "work" / "hf_mkgw")))
    for _sub in ("data", "experiments"):
        if (_local / _sub).exists():
            shutil.copytree(_local / _sub, REPO / _sub, dirs_exist_ok=True)
    if (_local / "images" / "mkgw_images_640.zip").exists():
        zipfile.ZipFile(_local / "images" / "mkgw_images_640.zip").extractall(REPO / "data" / "raw" / "images")
    for _sub in ("embeddings", "checkpoint"):
        if (_local / "native" / _sub).exists():
            shutil.copytree(_local / "native" / _sub, NATIVE / _sub, dirs_exist_ok=True)
    mo.md(f"Restored from `{hf_repo.value}`.")
    return


@app.cell
def _(NATIVE, PY, REPO, b_data, json, mo, push, sh, sha256, zipfile):
    mo.stop(not b_data.value, mo.md("Stage 2 not run yet."))
    import gdown as _gdown

    # (a) NativE pre-extracted modality tables (Drive of the NativE authors)
    _emb_dir = NATIVE / "embeddings"
    _emb_dir.mkdir(exist_ok=True)
    if not (_emb_dir / "MKG-W-visual.pth").exists():
        _z = _gdown.download(id="16qQg1NCyHSRaAWjRPpPEFajW2wrhC4aX", output=str(_emb_dir / "MKG.zip"), quiet=False)
        with zipfile.ZipFile(_z) as _zf:
            for _m in _zf.namelist():
                _name = _m.rsplit("/", 1)[-1]
                if _name.startswith("MKG-W-") and _name.endswith(".pth"):
                    (_emb_dir / _name).write_bytes(_zf.read(_m))

    # (b) MMRNS author images + DBpedia<->Wikidata links (Drive of the MKG-W authors)
    _raw = REPO / "data" / "raw" / "mkgw"
    _raw.mkdir(parents=True, exist_ok=True)
    if not (_raw / "ent_links.tsv").exists():
        _gdown.download(id="1GALz2zKIkMWMZ6thPyGuW02_j6EMGKVf", output=str(_raw / "ent_links.tsv"), quiet=False)
    if not (_raw / "mkgw_img.zip").exists():
        _gdown.download(id="1UA3qsboZ5pvt0nbfsMNDgxjlG30PZjde", output=str(_raw / "mkgw_img.zip"), quiet=False)

    # (c) Wikidata labels/descriptions/aliases + relation labels (resumable)
    sh(f"{PY} scripts/fetch_mkgw_wikidata.py", cwd=REPO)
    sh(f"{PY} scripts/fetch_mkgw_relations.py", cwd=REPO)

    # (d) one 640px image per entity, then the KGC triples with their NativE split.
    #     The retrieval split files are committed (Đạt, seed 0) and are NOT rebuilt here.
    sh(f"{PY} scripts/extract_mkgw_images.py", cwd=REPO)
    sh(f"{PY} -c \"from pathlib import Path; from src.data.build_mkgw_split import copy_kgc_triples; "
       f"from src.utils.config import load_config, resolve; c=load_config(); "
       f"print(copy_kgc_triples(Path(c['mkgw']['native_repo']).expanduser(), resolve(c,'processed')/'mkgw_triples.jsonl'))\"",
       cwd=REPO)

    # (e) every split entity must have its image; compare checksums with data/MANIFEST.md (Đạt, 2026-09-25)
    _missing = []
    for _split in ("train", "val", "test"):
        for _im in json.loads((REPO / "data" / "splits" / f"mkgw_{_split}.json").read_text(encoding="utf-8"))["images"]:
            if not (REPO / "data" / "raw" / "images" / _im["file_name"]).exists():
                _missing.append(_im["qid"])
    _expected = {
        "data/raw/mkgw/ent_links.tsv": "4aa7e9f5ff3bbeea7dee3124669edba7903e65b22844529999325c6bec624511",
        "data/raw/mkgw/mkgw_img.zip": "0178f7bb3b6199309a56fad23c595c252f79506deb5dbecece58d814d5f0aede",
        "data/raw/mkgw/author_images_index.json": "0727dafebdf3d9c3fd988778e34f574d324eb3bf9dd9e144ce15c19631dd45bd",
        "data/processed/mkgw_triples.jsonl": "40ccfa72ce54d9d2c6a23e9ed50ea571255425e20cf1b0cf6c2c5e2e3cecbedb",
    }
    _rows = [f"| `{p}` | {'match' if sha256(REPO / p) == h else 'DIFFERENT'} |" for p, h in _expected.items()]

    # (f) push right away: small raw files, triples, the 640px images, and the NativE tables
    for _f in ("wikidata_entities.jsonl", "wikidata_relations.json", "ent_links.tsv", "author_images_index.json"):
        push(_raw / _f, f"data/raw/mkgw/{_f}")
    push(REPO / "data" / "processed" / "mkgw_triples.jsonl", "data/processed/mkgw_triples.jsonl")
    _img_zip = REPO.parent / "mkgw_images_640.zip"
    with zipfile.ZipFile(_img_zip, "w", zipfile.ZIP_STORED) as _zf:
        for _p in sorted((REPO / "data" / "raw" / "images").glob("Q*.jpg")):
            _zf.write(_p, _p.name)
    push(_img_zip, "images/mkgw_images_640.zip")
    for _p in _emb_dir.glob("MKG-W-*.pth"):
        push(_p, f"native/embeddings/{_p.name}")
    mo.md(f"Split entities without an image: **{len(_missing)}** {_missing[:10]}\n\n"
          "| File | vs MANIFEST |\n|---|---|\n" + "\n".join(_rows) + "\n\nPushed to HF.")
    return


@app.cell
def _(PY, REPO, b_clip, mo, push, sh, sha256):
    mo.stop(not b_clip.value, mo.md("Stage 3 not run yet."))
    sh(f"{PY} scripts/extract_features.py --dataset mkgw", cwd=REPO)
    _exp = {"mkgw_image.npy": "fe9739e40478cf58e87e6a720ceba9bf3ec4deee9bef9e92107a99f7ba2da827",
            "mkgw_caption.npy": "59de0cf337a14ae3c468131bb9ca6952d85d3f4be577f971303a2c042de17d2d"}
    _rows = [f"| `{n}` | {'match' if sha256(REPO / 'data' / 'features' / n) == h else 'different (GPU numerics)'} |"
             for n, h in _exp.items()]
    for _n in ("mkgw_image", "mkgw_caption"):
        for _ext in (".npy", ".ids.json"):
            push(REPO / "data" / "features" / f"{_n}{_ext}", f"data/features/{_n}{_ext}")
    mo.md("| File | vs MANIFEST |\n|---|---|\n" + "\n".join(_rows) + "\n\nPushed to HF.")
    return


@app.cell
def _(PY, REPO, b_probe, mo, pip, push, sh):
    mo.stop(not b_probe.value, mo.md("Stage 3b not run yet."))
    pip("sentence-transformers")
    sh(f"{PY} scripts/checks/multilingual_clip_probe.py --large", cwd=REPO)
    _out = REPO / "experiments" / "checks" / "multilingual_clip_probe.json"
    push(_out, "experiments/checks/multilingual_clip_probe.json")
    mo.md("```json\n" + _out.read_text(encoding="utf-8")[:5000] + "\n```")
    return


@app.cell
def _(LOG_DIR, NATIVE, PY, REPO, b_train, epochs, hf_repo, hf_token, mo, remote_files, run_name, runs, sh, subprocess):
    mo.stop(not b_train.value, mo.md("Stage 4 not run yet."))
    import os as _os_env
    # leakage-safe copy: modality rows of retrieval val/test entities set to zero
    sh(f"{PY} scripts/mask_native_modalities.py --name MKG-W-rmask", cwd=REPO)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _done_remote = remote_files()
    _started = []
    for _spec in runs.value:
        _name, _dataset, _seed = run_name(_spec, epochs.value)
        if (NATIVE / "checkpoint" / f"{_name}.ckpt").exists() or f"native/checkpoint/{_name}.ckpt" in _done_remote:
            print("skip (checkpoint exists):", _name)
            continue
        # hyper-parameters = the authors' scripts/run_mkgw.sh; only dataset, seed, epoch vary.
        # setsid + nohup: the run survives the notebook cell, a UI disconnect and a kernel restart.
        _cmd = (f"{PY} run_adv_wgan_gp.py -dataset={_dataset} -batch_size=1024 -margin=4 -epoch={epochs.value} "
                f"-dim=250 -save=./checkpoint/{_name}.ckpt -neg_num=128 -mu=0.0001 -learning_rate=1e-4 "
                f"-lrg=1e-4 -seed={_seed}")
        _log = LOG_DIR / f"{_name}.log"
        _ckpt = NATIVE / "checkpoint" / f"{_name}.ckpt"
        # after training, the same detached shell pushes checkpoint + log to HF (no notebook needed)
        _push = (f"{PY} {REPO}/scripts/push_to_hf.py --repo {hf_repo.value} {_ckpt} native/checkpoint/{_name}.ckpt "
                 f"{_log} experiments/level2/native-train/{_name}.log")
        _script = LOG_DIR / f"{_name}.sh"
        _script.write_text(f"#!/bin/bash\ncd {NATIVE}\n{_cmd}\n{_push}\n")
        _log.write_text(f"$ {_cmd}\n")
        # token goes through the environment only: not printed, not written to the script or the log
        _p = subprocess.Popen(["setsid", "nohup", "bash", str(_script)], cwd=NATIVE, start_new_session=True,
                              stdin=subprocess.DEVNULL, stdout=open(_log, "a"), stderr=subprocess.STDOUT,
                              env={**_os_env.environ, "HF_TOKEN": hf_token.value})
        (LOG_DIR / f"{_name}.pid").write_text(str(_p.pid))
        _started.append(_name)
    mo.md(f"Started in the background: {_started or 'nothing'}. Each run pushes its checkpoint and log to HF "
          "by itself when it ends; **4b · Status** only shows progress.")
    return


@app.cell
def _(LOG_DIR, NATIVE, auto, b_status, epochs, mo, push, remote_files, run_name, runs):
    auto.value  # re-run on every auto refresh tick
    mo.stop(not (b_status.value or auto.value), mo.md("Status not checked yet."))
    import os as _os
    _remote = remote_files()
    _rows = []
    for _spec in runs.value:
        _name, _, _ = run_name(_spec, epochs.value)
        _log, _pid = LOG_DIR / f"{_name}.log", LOG_DIR / f"{_name}.pid"
        _ckpt = NATIVE / "checkpoint" / f"{_name}.ckpt"
        _text = _log.read_text(errors="replace") if _log.exists() else ""
        _alive = False
        if _pid.exists():
            try:
                _os.kill(int(_pid.read_text().strip()), 0)
                _alive = True
            except (OSError, ValueError):
                pass
        _finished = "averaged(filter)" in _text and _ckpt.exists()
        _progress = _text.replace("\r", "\n").strip().splitlines()[-1][:160] if _text else "(no log)"
        if _finished and f"native/checkpoint/{_name}.ckpt" not in _remote:
            push(_ckpt, f"native/checkpoint/{_name}.ckpt")
        if _log.exists():
            push(_log, f"experiments/level2/native-train/{_name}.log")
        _state = "finished, on HF" if _finished else ("running" if _alive else "not running")
        _rows.append(f"| `{_name}` | {_state} | `{_progress}` |")
    mo.md("| Run | State | Last log line |\n|---|---|---|\n" + "\n".join(_rows))
    return


@app.cell
def _(NATIVE, PY, REPO, b_eval_val, branch, hf_repo, hf_token, mo, push, sh):
    # Level 2 on val: export the NativE entity vectors of the 3 rmask seeds, then alpha/beta on val.
    # Runs here because the checkpoints are on this machine (or one fast HF download away);
    # only the small JSON results go back to HF.
    mo.stop(not b_eval_val.value, mo.md("Level-2 val not run yet."))
    sh(f"git fetch origin && git checkout {branch.value} && git pull --ff-only", cwd=REPO)
    from huggingface_hub import hf_hub_download as _hf_hub_download
    for _s in (0, 1, 2):
        _ckpt = NATIVE / "checkpoint" / f"rmask_seed{_s}_e1000.ckpt"
        if not _ckpt.exists():
            _hf_hub_download(hf_repo.value, f"native/checkpoint/{_ckpt.name}", repo_type="dataset",
                             token=hf_token.value or None, local_dir=str(NATIVE.parent / "hf_mkgw"))
            _ckpt.parent.mkdir(parents=True, exist_ok=True)
            (NATIVE.parent / "hf_mkgw" / "native" / "checkpoint" / _ckpt.name).replace(_ckpt)
        sh(f"{PY} scripts/export_native_embeddings.py --checkpoint {_ckpt} --what ent --suffix _seed{_s}", cwd=REPO)
    _out = sh(f"{PY} scripts/eval_mkgw_level2.py --split val", cwd=REPO)
    (REPO / "experiments" / "level2" / "val_run.log").write_text(_out, encoding="utf-8")
    push(REPO / "experiments" / "level2" / "val", "experiments/level2/val")
    push(REPO / "experiments" / "level2" / "val_run.log", "experiments/level2/val_run.log")
    for _s in (0, 1, 2):
        for _ext in (".npy", ".ids.json"):
            push(REPO / "data" / "features" / f"mkgw_native_ent_seed{_s}{_ext}",
                 f"data/features/mkgw_native_ent_seed{_s}{_ext}")
    mo.md("Level-2 val done; results in `experiments/level2/val` (also on HF).")
    return


@app.cell
def _(PY, REPO, b_eval_test, mo, push, sh):
    # Test, once: alpha/beta are read from experiments/level2/val/choice.json, nothing is tuned.
    mo.stop(not b_eval_test.value, mo.md("Level-2 test not run yet."))
    mo.stop(not (REPO / "experiments" / "level2" / "val" / "choice.json").exists(), mo.md("Run step 5 first."))
    _out = sh(f"{PY} scripts/eval_mkgw_level2.py --split test --hand", cwd=REPO)
    (REPO / "experiments" / "level2" / "test_run.log").write_text(_out, encoding="utf-8")
    push(REPO / "experiments" / "level2" / "test", "experiments/level2/test")
    push(REPO / "experiments" / "level2" / "test_run.log", "experiments/level2/test_run.log")
    mo.md("Level-2 test done; results in `experiments/level2/test` (also on HF).")
    return


@app.cell
def _(REPO, mo):
    # download buttons for everything small this notebook writes
    _files = sorted([*(REPO / "experiments" / "level2").rglob("*.log"),
                     *(REPO / "experiments" / "level2").rglob("*.json"),
                     *(REPO / "experiments" / "checks").glob("multilingual_clip_probe.json")]) if REPO.exists() else []
    _items = [mo.download(data=p.read_bytes(), filename=p.name, label=str(p.relative_to(REPO)))
              for p in _files if p.stat().st_size < 50_000_000]
    mo.vstack([mo.md("### Files (re-run this cell to refresh)"), *_items] if _items
              else [mo.md("### Files\nNothing produced yet.")])
    return


if __name__ == "__main__":
    app.run()
