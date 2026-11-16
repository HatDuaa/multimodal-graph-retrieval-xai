import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    mo.md(
        r"""
        # MKG-W on molab — data, CLIP features, NativE training (level 2)

        Runs the heavy part of level 2 on a molab GPU and keeps every output on a
        Hugging Face dataset repo, so the laptop only downloads small files.
        Data steps follow Đạt's scripts on `dat/mkgw-data` (rows 20, 22, 23 of the task sheet);
        this notebook and the leakage mask are Khang's additions.

        **Before you start:** turn on the GPU (notebook specs button), then fill in the
        Hugging Face token below. Each stage has its own button; stages are idempotent
        and **Restore** pulls finished outputs back after a session restart
        (molab stops after 12 h, or after 90 min idle).

        | Stage | What | Time (rough) |
        |---|---|---|
        | 1 Setup | clone repo branch + NativE, install, build `Base.so` | 3 min |
        | 2 Data | NativE embeddings, MMRNS image zip (6.4 GB), Wikidata labels, images, KGC triples | 30–60 min |
        | 3 CLIP | frozen CLIP features for 8 920 images + queries | 5 min |
        | 3b Probe | multilingual CLIP on 84 vi/en MKG-W val queries (row 9) | 10 min |
        | 4 Train | NativE: paper reproduction (original MKG-W) + 3 seeds on the leakage-safe copy | hours |
        | 5 Export | entity embeddings per run | 1 min |
        | 6 Upload | push everything to the HF dataset repo | 5–10 min |
        | 7 Eval | level-2 evaluation on val (3 seeds, ablations); upload again after it | 5 min |
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
    import os
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

    def sh(cmd, cwd=None, check=True, env=None):
        """Run a shell command, print its output, raise on failure."""
        print(f"$ {cmd}")
        r = subprocess.run(cmd, shell=True, cwd=cwd, env=env, text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
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

    return NATIVE, NATIVE_COMMIT, PY, Path, REPO, WORK, os, pip, sh, sha256, shutil, subprocess, zipfile


@app.cell
def _(mo):
    b_setup = mo.ui.run_button(label="1 · Setup")
    b_restore = mo.ui.run_button(label="Restore from HF")
    b_data = mo.ui.run_button(label="2 · Data")
    b_clip = mo.ui.run_button(label="3 · CLIP features")
    b_train = mo.ui.run_button(label="4 · Train NativE")
    b_export = mo.ui.run_button(label="5 · Export embeddings")
    b_upload = mo.ui.run_button(label="6 · Upload to HF")
    b_probe = mo.ui.run_button(label="3b · Multilingual CLIP probe (row 9)")
    b_eval = mo.ui.run_button(label="7 · Level-2 eval on val")
    mo.hstack([b_setup, b_restore, b_data, b_clip, b_probe, b_train, b_export, b_eval, b_upload], wrap=True)
    return b_clip, b_data, b_eval, b_export, b_probe, b_restore, b_setup, b_train, b_upload


@app.cell
def _(PY, REPO, b_probe, mo, pip, sh):
    mo.stop(not b_probe.value, mo.md("Stage 3b not run yet."))
    pip("sentence-transformers")
    sh(f"{PY} scripts/checks/multilingual_clip_probe.py --large", cwd=REPO)
    mo.md((REPO / "experiments" / "checks" / "multilingual_clip_probe.json").read_text(encoding="utf-8")[:4000])
    return


@app.cell
def _(PY, REPO, b_eval, mo, sh):
    mo.stop(not b_eval.value, mo.md("Stage 7 not run yet."))
    # val only: chooses alpha/beta per seed. Test runs once, later, after the method is locked.
    sh(f"{PY} scripts/eval_mkgw_level2.py --split val", cwd=REPO)
    mo.md("Wrote `experiments/level2/val/` (summary.json, choice.json, metrics_*.json).")
    return


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
    sh("ls -la release && nvidia-smi --query-gpu=name,memory.total --format=csv && nproc && df -h ~",
       cwd=NATIVE / "mmkgc", check=False)
    mo.md("Setup done.")
    return


@app.cell
def _(NATIVE, Path, REPO, b_restore, hf_repo, hf_token, mo, shutil, zipfile):
    mo.stop(not b_restore.value, mo.md("Restore not run."))
    from huggingface_hub import snapshot_download as _snapshot_download
    _local = Path(_snapshot_download(hf_repo.value, repo_type="dataset", token=hf_token.value or None,
                                   local_dir=str(Path.home() / "work" / "hf_mkgw")))
    for _sub in ("data/raw/mkgw", "data/processed", "data/features", "experiments/level2"):
        if (_local / _sub).exists():
            shutil.copytree(_local / _sub, REPO / _sub, dirs_exist_ok=True)
    if (_local / "images" / "mkgw_images_640.zip").exists():
        zipfile.ZipFile(_local / "images" / "mkgw_images_640.zip").extractall(REPO / "data" / "raw" / "images")
    if (_local / "native").exists():
        shutil.copytree(_local / "native" / "embeddings", NATIVE / "embeddings", dirs_exist_ok=True)
        shutil.copytree(_local / "native" / "checkpoint", NATIVE / "checkpoint", dirs_exist_ok=True)
    mo.md(f"Restored from `{hf_repo.value}`.")
    return


@app.cell
def _(NATIVE, PY, REPO, b_data, mo, sh, sha256, zipfile):
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
    #     The retrieval split files are committed (Đạt, seed 0) and are NOT rebuilt here:
    #     Wikidata descriptions drift, and every model must use the same split file.
    sh(f"{PY} scripts/extract_mkgw_images.py", cwd=REPO)
    sh(f"{PY} -c \"from pathlib import Path; from src.data.build_mkgw_split import copy_kgc_triples; "
       f"from src.utils.config import load_config, resolve; c=load_config(); "
       f"print(copy_kgc_triples(Path(c['mkgw']['native_repo']).expanduser(), resolve(c,'processed')/'mkgw_triples.jsonl'))\"",
       cwd=REPO)

    # (e) every split entity must have its image; compare checksums with data/MANIFEST.md (Đạt, 2026-09-25)
    import json as _json
    _missing = []
    for _split in ("train", "val", "test"):
        for _im in _json.loads((REPO / "data" / "splits" / f"mkgw_{_split}.json").read_text(encoding="utf-8"))["images"]:
            if not (REPO / "data" / "raw" / "images" / _im["file_name"]).exists():
                _missing.append(_im["qid"])
    _expected = {
        "data/raw/mkgw/ent_links.tsv": "4aa7e9f5ff3bbeea7dee3124669edba7903e65b22844529999325c6bec624511",
        "data/raw/mkgw/mkgw_img.zip": "0178f7bb3b6199309a56fad23c595c252f79506deb5dbecece58d814d5f0aede",
        "data/raw/mkgw/author_images_index.json": "0727dafebdf3d9c3fd988778e34f574d324eb3bf9dd9e144ce15c19631dd45bd",
        "data/processed/mkgw_triples.jsonl": "40ccfa72ce54d9d2c6a23e9ed50ea571255425e20cf1b0cf6c2c5e2e3cecbedb",
        "data/raw/mkgw/wikidata_entities.jsonl": "8e1aa6387e9a21a5067b577b1d2deae1651232cba54d30f8232685908c689daf",
    }
    _rows = [f"| `{p}` | {'match' if sha256(REPO / p) == h else 'DIFFERENT'} |" for p, h in _expected.items()]
    mo.md(f"Split entities without an image: **{len(_missing)}** {_missing[:10]}\n\n"
          "| File | vs MANIFEST |\n|---|---|\n" + "\n".join(_rows) +
          "\n\n`wikidata_entities.jsonl` may differ (Wikidata changes over time); the split files do not depend on it.")
    return


@app.cell
def _(PY, REPO, b_clip, mo, sh, sha256):
    mo.stop(not b_clip.value, mo.md("Stage 3 not run yet."))
    sh(f"{PY} scripts/extract_features.py --dataset mkgw", cwd=REPO)
    _exp = {"mkgw_image.npy": "fe9739e40478cf58e87e6a720ceba9bf3ec4deee9bef9e92107a99f7ba2da827",
            "mkgw_caption.npy": "59de0cf337a14ae3c468131bb9ca6952d85d3f4be577f971303a2c042de17d2d"}
    _rows = [f"| `{n}` | {'match' if sha256(REPO / 'data' / 'features' / n) == h else 'different (GPU numerics; compare metrics instead)'} |"
             for n, h in _exp.items()]
    mo.md("| File | vs MANIFEST |\n|---|---|\n" + "\n".join(_rows))
    return


@app.cell
def _(NATIVE, PY, REPO, b_train, epochs, mo, runs, sh, subprocess):
    mo.stop(not b_train.value, mo.md("Stage 4 not run yet."))
    # leakage-safe copy: modality rows of retrieval val/test entities set to zero
    sh(f"{PY} scripts/mask_native_modalities.py --name MKG-W-rmask", cwd=REPO)

    _log_dir = REPO / "experiments" / "level2" / "native-train"
    _log_dir.mkdir(parents=True, exist_ok=True)
    _procs = {}
    for _spec in runs.value:
        _tag, _dataset, _seed = _spec.split(":")
        _name = f"{_tag}_seed{_seed}_e{epochs.value}"
        _ckpt = NATIVE / "checkpoint" / f"{_name}.ckpt"
        if _ckpt.exists():
            print("skip (checkpoint exists):", _name)
            continue
        # hyper-parameters = the authors' scripts/run_mkgw.sh; only dataset, seed, epoch vary
        _cmd = (f"{PY} run_adv_wgan_gp.py -dataset={_dataset} -batch_size=1024 -margin=4 -epoch={epochs.value} "
               f"-dim=250 -save=./checkpoint/{_name}.ckpt -neg_num=128 -mu=0.0001 -learning_rate=1e-4 "
               f"-lrg=1e-4 -seed={_seed}")
        _log = open(_log_dir / f"{_name}.log", "w")
        _log.write(f"$ {_cmd}\n")
        _log.flush()
        _procs[_name] = subprocess.Popen(_cmd, shell=True, cwd=NATIVE, stdout=_log, stderr=subprocess.STDOUT)
    for _name, _p in _procs.items():
        _p.wait()
        print(_name, "exit", _p.returncode)
    sh(f"tail -n 12 {_log_dir}/*.log", check=False)
    mo.md("Training finished; link-prediction results are at the end of each _log.")
    return


@app.cell
def _(NATIVE, PY, REPO, b_export, epochs, mo, runs, sh):
    mo.stop(not b_export.value, mo.md("Stage 5 not run yet."))
    for _spec in runs.value:
        _tag, _dataset, _seed = _spec.split(":")
        _name = f"{_tag}_seed{_seed}_e{epochs.value}"
        _suffix = "_repro" if _tag == "repro" else f"_seed{_seed}"
        sh(f"{PY} scripts/export_native_embeddings.py --checkpoint {NATIVE}/checkpoint/{_name}.ckpt "
           f"--what ent img_proj text_proj --suffix {_suffix}", cwd=REPO)
    mo.md("Exported to `data/features/mkgw_native_*`.")
    return


@app.cell
def _(NATIVE, REPO, b_upload, hf_repo, hf_token, mo, shutil, zipfile):
    mo.stop(not b_upload.value, mo.md("Stage 6 not run yet."))
    from huggingface_hub import HfApi as _HfApi
    _api = _HfApi(token=hf_token.value)
    # private: the images are the MKG-W authors' crawl, redistributed for course use only
    _api.create_repo(hf_repo.value, repo_type="dataset", private=True, exist_ok=True)
    _stage = REPO.parent / "hf_upload"
    shutil.rmtree(_stage, ignore_errors=True)
    for _sub in ("data/raw/mkgw", "data/processed", "data/features", "experiments/level2"):
        if (REPO / _sub).exists():
            shutil.copytree(REPO / _sub, _stage / _sub,
                            ignore=shutil.ignore_patterns("mkgw_img.zip", "vg_coco*", "vnmmkg*"))
    (_stage / "images").mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(_stage / "images" / "mkgw_images_640.zip", "w", zipfile.ZIP_STORED) as _zf:
        for _p in sorted((REPO / "data" / "raw" / "images").glob("Q*.jpg")):
            _zf.write(_p, _p.name)
    (_stage / "native" / "embeddings").mkdir(parents=True, exist_ok=True)
    for _p in (NATIVE / "embeddings").glob("MKG-W*.pth"):
        shutil.copy2(_p, _stage / "native" / "embeddings" / _p.name)
    shutil.copytree(NATIVE / "checkpoint", _stage / "native" / "checkpoint",
                    ignore=shutil.ignore_patterns("*.py"), dirs_exist_ok=True)
    _api.upload_large_folder(repo_id=hf_repo.value, repo_type="dataset", folder_path=str(_stage))
    mo.md(f"Uploaded to https://huggingface.co/datasets/{hf_repo.value}")
    return


if __name__ == "__main__":
    app.run()
