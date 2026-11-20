"""Download FracAtlas, read its metadata and build one fixed, stratified image-level split."""
from __future__ import annotations

import hashlib
import shutil
import socket
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

URL = "https://figshare.com/ndownloader/files/43283628"  # FracAtlas.zip, figshare 10.6084/m9.figshare.22363012
REGIONS = ("hand", "leg", "hip", "shoulder")
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}


ARTICLE_API = "https://api.figshare.com/v2/articles/22363012"
FALLBACK_URLS = ("https://ndownloader.figshare.com/files/43283628", URL)
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) fracatlas-fracture-benchmark", "Accept": "*/*"}


def _candidate_urls() -> list:
    """Archive URLs from the figshare API (largest zip first), then the fixed fallbacks."""
    import json
    urls = []
    try:
        req = urllib.request.Request(ARTICLE_API, headers=HEADERS)
        with urllib.request.urlopen(req) as resp:
            files = json.loads(resp.read().decode())["files"]
        files = sorted((f for f in files if f["name"].lower().endswith(".zip")), key=lambda f: -f["size"])
        urls += [f["download_url"] for f in files]
        print("figshare API:", [(f["name"], round(f["size"] / 1e6, 1)) for f in files], flush=True)
    except Exception as err:  # noqa: BLE001
        print("figshare API lookup failed:", repr(err), flush=True)
    return urls + [u for u in FALLBACK_URLS if u not in urls]


def download(root: str | Path, url: str | None = None, retries: int = 4) -> Path:
    """Return the folder that holds dataset.csv: an attached copy under /kaggle/input if present,
    otherwise the figshare archive downloaded and unpacked into ``root``."""
    root = Path(root)
    for candidate in (root, Path("/kaggle/input")):
        found = find_root(candidate)
        if found is not None:
            return found
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "FracAtlas.zip"
    socket.setdefaulttimeout(120)
    for link in ([url] if url else _candidate_urls()):
        for attempt in range(retries):
            try:
                req = urllib.request.Request(link, headers=HEADERS)
                with urllib.request.urlopen(req) as resp, open(archive, "wb") as fh:
                    shutil.copyfileobj(resp, fh, 1 << 20)
                    status = resp.status
                head = archive.read_bytes()[:4] if archive.exists() else b""
                size = archive.stat().st_size
                print(f"{link}: HTTP {status}, {size / 1e6:.1f} MB", flush=True)
                if head.startswith(b"PK") and size > 1e6:
                    _unzip(archive, root)
                    found = find_root(root)
                    if found is not None:
                        return found
                break  # a response that is not the archive: try the next URL
            except Exception as err:  # noqa: BLE001
                print(f"{link}: attempt {attempt + 1} failed: {err!r}", flush=True)
                time.sleep(2 ** attempt * 5)
    raise RuntimeError("Could not download FracAtlas from figshare. Download FracAtlas.zip in a browser from "
                       "https://doi.org/10.6084/m9.figshare.22363012, add it to this notebook as a private Kaggle "
                       "dataset (Add Input) and run again; the attached copy is used automatically.")


def _unzip(archive: Path, root: Path) -> None:
    """Unpack, including zip files nested inside the archive."""
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(root)
    archive.unlink()
    for inner in list(root.rglob("*.zip")):
        _unzip(inner, inner.parent)


def find_root(root: str | Path) -> Path | None:
    hits = sorted(Path(root).glob("**/dataset.csv")) if Path(root).exists() else []
    hits = [h for h in hits if "__MACOSX" not in str(h)]
    return hits[0].parent if hits else None


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_metadata(root: str | Path) -> pd.DataFrame:
    """One row per image with id, path, label, region flags, primary region and view."""
    root = Path(root)
    df = pd.read_csv(root / "dataset.csv")
    df.columns = [c.strip().lower() for c in df.columns]
    id_col = "image_id" if "image_id" in df.columns else df.columns[0]
    df = df.rename(columns={id_col: "image_id"})
    for col in (*REGIONS, "fractured"):
        if col not in df.columns:
            raise KeyError(f"dataset.csv has no column '{col}'; columns: {list(df.columns)}")
        df[col] = df[col].astype(int)
    images = {p.name: p for p in root.glob("images/**/*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    if not images:
        images = {p.name: p for p in root.glob("**/*.jpg")}
    df["path"] = df["image_id"].map(lambda i: str(images.get(i, "")))
    missing = (df["path"] == "").sum()
    if missing:
        raise FileNotFoundError(f"{missing} images listed in dataset.csv were not found under {root}")
    df["region"] = primary_region(df)
    df["view"] = view(df)
    return df.reset_index(drop=True)


def primary_region(df: pd.DataFrame) -> pd.Series:
    """The body region of an image; images flagged with more than one region are 'multiple'."""
    flags = df[list(REGIONS)].to_numpy()
    out = np.where(flags.sum(axis=1) == 1, np.array(REGIONS)[flags.argmax(axis=1)], "multiple")
    out = np.where(flags.sum(axis=1) == 0, "unknown", out)
    return pd.Series(out, index=df.index)


def view(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in ("frontal", "lateral", "oblique") if c in df.columns]
    if not cols:
        return pd.Series("unknown", index=df.index)
    flags = df[cols].fillna(0).astype(int).to_numpy()
    out = np.where(flags.sum(axis=1) >= 1, np.array(cols)[flags.argmax(axis=1)], "unknown")
    return pd.Series(out, index=df.index)


def make_split(df: pd.DataFrame, seed: int = 2023) -> pd.Series:
    """Stratified by fracture label x primary region, 70/15/15. FracAtlas has no patient ids, so the split is by image."""
    rng = np.random.default_rng(seed)
    split = pd.Series("train", index=df.index)
    for _, idx in df.groupby(["fractured", "region"]).groups.items():
        idx = np.array(sorted(idx))
        rng.shuffle(idx)
        n = len(idx)
        n_test = int(round(n * SPLIT_FRACTIONS["test"]))
        n_val = int(round(n * SPLIT_FRACTIONS["val"]))
        split.loc[idx[:n_test]] = "test"
        split.loc[idx[n_test:n_test + n_val]] = "val"
    return split


def yolo_dataset(df: pd.DataFrame, root: str | Path, out: str | Path) -> Path:
    """Write an Ultralytics dataset (images symlinked, YOLO labels copied, empty labels for images without fracture)."""
    root, out = Path(root), Path(out)
    labels_src = {p.stem: p for p in root.glob("Annotations/YOLO/*.txt")}
    for part in ("train", "val", "test"):
        (out / "images" / part).mkdir(parents=True, exist_ok=True)
        (out / "labels" / part).mkdir(parents=True, exist_ok=True)
    for _, row in df.iterrows():
        src = Path(row["path"])
        dst = out / "images" / row["split"] / src.name
        if not dst.exists():
            dst.symlink_to(src.resolve())
        label = out / "labels" / row["split"] / (src.stem + ".txt")
        text = labels_src[src.stem].read_text() if src.stem in labels_src and row["fractured"] == 1 else ""
        label.write_text(text)
    yaml = out / "fracatlas.yaml"
    yaml.write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: fractured\n")
    return yaml
