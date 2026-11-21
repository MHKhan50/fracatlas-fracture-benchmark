"""Near-duplicate check between splits with perceptual hashes.

FracAtlas releases no patient identifiers, so the split is by image. This step looks for test (and validation)
images that are near-copies of a training image: a 64-bit DCT perceptual hash per image, Hamming distance to the
nearest training image, and test AUROC of every seed ensemble recomputed without the test images whose nearest
training image is within a distance threshold. It finds re-exported or slightly cropped copies of the same
radiograph; it cannot find different views or different visits of the same patient.

    python -m fracatlas_bench.neardup --images /path/to/FracAtlas --reports /path/to/reports --out neardup
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.fft import dctn
from sklearn.metrics import roc_auc_score

from .metrics import bootstrap_ci

THRESHOLDS = (0, 4, 8, 12)


def phash(img) -> np.ndarray:
    """64-bit perceptual hash of a PIL image: 32x32 grey, 2D DCT, top-left 8x8 block against its median."""
    from PIL import Image
    a = np.asarray(img.convert("L").resize((32, 32), Image.LANCZOS), dtype=np.float64)
    block = dctn(a, norm="ortho")[:8, :8].ravel()
    return block > np.median(block[1:])


def hamming_to_nearest(query: np.ndarray, ref: np.ndarray) -> tuple:
    """For each row of ``query`` (bool hashes), distance and index of the nearest row of ``ref``."""
    q, r = query.astype(np.uint8), ref.astype(np.uint8)
    dist = np.empty(len(q), dtype=int)
    idx = np.empty(len(q), dtype=int)
    for i in range(len(q)):
        d = (q[i] != r).sum(axis=1)
        idx[i] = int(d.argmin())
        dist[i] = int(d[idx[i]])
    return dist, idx


def ensemble_logits(runs: Path) -> dict:
    out = {}
    for f in sorted(runs.glob("*_logits.npz")):
        m = re.match(r"(.+)_s(\d+)_logits", f.stem)
        out.setdefault(m.group(1), []).append(np.load(f)["test"])
    return {k: np.mean(v, axis=0) for k, v in out.items()}


def auroc_without(y, z, drop, bootstrap=1000) -> dict:
    keep = ~drop
    return {"images": int(keep.sum()), "fractured": int(y[keep].sum()),
            "auroc": float(roc_auc_score(y[keep], z[keep])),
            "auroc_ci95": bootstrap_ci(y[keep], z[keep], roc_auc_score, bootstrap, 0)}


def _montage(top, files, open_gray, path):
    """The closest pairs side by side, for a visual check (kept in the run output, not in the repository)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(top), 2, figsize=(6, 2.6 * len(top)))
    for row, (_, r) in zip(np.atleast_2d(axes), top.iterrows()):
        for ax, key in zip(row, ("image_id", "train_image_id")):
            ax.imshow(open_gray(str(files[r[key]])), cmap="gray"); ax.axis("off")
            ax.set_title(f"{r[key]} (d={r['distance']})" if key == "image_id" else r[key], fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main(argv=None):
    from .train import _open_gray
    p = argparse.ArgumentParser(prog="fracatlas-neardup")
    p.add_argument("--images", required=True)
    p.add_argument("--reports", required=True)
    p.add_argument("--out", default="neardup")
    a = p.parse_args(argv)
    rep, out = Path(a.reports), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    split = pd.read_csv(rep / "split.csv")
    files = {f.name: f for f in Path(a.images).rglob("*") if f.suffix.lower() in (".jpg", ".jpeg", ".png")}
    hashes = np.stack([phash(_open_gray(str(files[i]))) for i in split["image_id"]])
    part = split["split"].to_numpy()
    tr = np.where(part == "train")[0]
    report = {"hash": "64-bit DCT perceptual hash", "thresholds": list(THRESHOLDS), "splits": {}}
    pairs = []
    for name in ("val", "test"):
        q = np.where(part == name)[0]
        dist, nn = hamming_to_nearest(hashes[q], hashes[tr])
        report["splits"][name] = {"images": int(len(q)),
                                  "distance_quantiles": {str(k): float(np.percentile(dist, k)) for k in (0, 1, 5, 25, 50)},
                                  "with_train_neighbour_within": {str(t): int((dist <= t).sum()) for t in THRESHOLDS}}
        for i, d, j in zip(q, dist, nn):
            pairs.append({"split": name, "image_id": split["image_id"][i], "train_image_id": split["image_id"][tr[j]],
                          "distance": int(d), "fractured": int(split["fractured"][i]),
                          "train_fractured": int(split["fractured"][tr[j]]), "region": split["region"][i]})
        if name == "test":
            test_dist = dist
    y = split.loc[part == "test", "fractured"].to_numpy()
    ens = ensemble_logits(rep / "runs")
    report["test_auroc"] = {}
    for model, z in ens.items():
        if len(z) != len(y):
            raise ValueError(f"{model}: logits do not match split.csv")
        report["test_auroc"][model] = {"all": auroc_without(y, z, np.zeros(len(y), bool))}
        for t in THRESHOLDS:
            report["test_auroc"][model][f"without_within_{t}"] = auroc_without(y, z, test_dist <= t)
    pd.DataFrame(pairs).sort_values("distance").to_csv(out / "nearest_train_neighbours.csv", index=False)
    (out / "neardup.json").write_text(json.dumps(report, indent=2))
    _montage(pd.DataFrame(pairs).sort_values("distance").head(8), files, _open_gray, out / "closest_pairs.png")
    print(json.dumps(report, indent=2))
    print(pd.DataFrame(pairs).sort_values("distance").head(20).to_string(index=False))


if __name__ == "__main__":
    main()
