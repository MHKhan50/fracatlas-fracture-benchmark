"""Choose the three sample images for the browser app from the FracAtlas test split, using the released ONNX models.

Runs the same preprocessing as training (grayscale, letterbox) and the same decision rule as the app, then picks:
  fractured_clear.jpg   - fractured test image with the highest probability that also gets a detector box
  fractured_subtle.jpg  - fractured test image the model refers (confidence below the referral cut-off), closest to 0.5
  not_fractured.jpg     - non-fractured test image with the lowest probability and no detector box
The choice and every test score are written to samples/samples.json and samples/test_scores.csv, so the selection
is transparent. With --upload (and HF_TOKEN set) the images are pushed to the Space.

    python scripts/pick_samples.py --split results/kaggle-2026-11-21/split.csv --images /kaggle/input/<fracatlas> --upload
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

MODEL_REPO = "MHkhan50/fracatlas-convnext-tiny"
SPACE_REPO = "MHkhan50/fracture-xray-check"


def gray(path: Path):
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True  # a few FracAtlas JPEGs end early, as in training
    return Image.open(path).convert("L")


def classifier_input(img, size: int, mean, std) -> np.ndarray:
    from PIL import Image
    w, h = img.size
    s = size / max(w, h)
    r = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
    out = np.zeros((size, size), dtype=np.float32)
    top, left = (size - r.size[1]) // 2, (size - r.size[0]) // 2
    out[top:top + r.size[1], left:left + r.size[0]] = np.asarray(r, dtype=np.float32) / 255.0
    x = np.stack([(out - mean[c]) / std[c] for c in range(3)])
    return x[None].astype(np.float32)


def detector_input(img, size: int) -> np.ndarray:
    from PIL import Image
    w, h = img.size
    r = min(size / w, size / h)
    nw, nh = max(1, round(w * r)), max(1, round(h * r))
    left, top = max(0, round((size - nw) / 2 - 0.1)), max(0, round((size - nh) / 2 - 0.1))
    canvas = np.full((size, size), 114, dtype=np.float32)
    canvas[top:top + nh, left:left + nw] = np.asarray(img.resize((nw, nh), Image.BILINEAR), dtype=np.float32)
    x = np.repeat((canvas / 255.0)[None], 3, axis=0)
    return x[None].astype(np.float32)


def best_box_score(out: np.ndarray, min_score: float) -> float:
    """YOLO11 ONNX output [1, 5, N]; returns the highest box score at or above min_score (0 if none)."""
    scores = out[0, 4]
    s = float(scores.max()) if scores.size else 0.0
    return s if s >= min_score else 0.0


def verdict(logit: float, c: dict) -> tuple[float, str]:
    p = 1 / (1 + np.exp(-logit / c["temperature"]))
    conf = max(p, 1 - p)
    if conf < c["refer_below_confidence"]:
        return float(p), "refer"
    return float(p), "fracture" if p >= c["threshold"] else "none"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--images", required=True, help="FracAtlas folder (searched recursively for the .jpg files)")
    ap.add_argument("--out", default="space_samples")
    ap.add_argument("--upload", action="store_true", help="push the samples to the Space (needs HF_TOKEN)")
    a = ap.parse_args()

    import onnxruntime as ort
    import pandas as pd
    from huggingface_hub import hf_hub_download

    cfg = json.loads(Path(hf_hub_download(MODEL_REPO, "config.json")).read_text())
    cc, dc = cfg["classifier"], cfg["detector"]
    clf = ort.InferenceSession(hf_hub_download(MODEL_REPO, cc["file"]), providers=["CPUExecutionProvider"])
    det = ort.InferenceSession(hf_hub_download(MODEL_REPO, dc["file"]), providers=["CPUExecutionProvider"])

    split = pd.read_csv(a.split)
    test = split[split["split"] == "test"]
    files = {p.name: p for p in Path(a.images).rglob("*.jpg")}
    rows = []
    for i, r in enumerate(test.itertuples(), 1):
        img = gray(files[r.image_id])
        z = float(clf.run(None, {clf.get_inputs()[0].name: classifier_input(img, cc["size"], cc["mean"], cc["std"])})[0].ravel()[0])
        box = best_box_score(det.run(None, {det.get_inputs()[0].name: detector_input(img, dc["imgsz"])})[0], dc["box_score_min"])
        p, kind = verdict(z, cc)
        rows.append({"image_id": r.image_id, "region": r.region, "fractured": int(r.fractured), "p": p, "verdict": kind, "box": box})
        if i % 50 == 0:
            print(f"{i}/{len(test)}", flush=True)
    df = pd.DataFrame(rows)

    frac, norm = df[df.fractured == 1], df[df.fractured == 0]
    clear = frac[(frac.verdict == "fracture") & (frac.box > 0)].sort_values(["p", "box"], ascending=False)
    if clear.empty:
        clear = frac.sort_values("p", ascending=False)
    subtle = frac[frac.verdict == "refer"].assign(d=lambda x: (x.p - 0.5).abs()).sort_values("d")
    if subtle.empty:
        subtle = frac.assign(d=lambda x: (x.p - 0.5).abs()).sort_values("d")
    normal = norm[(norm.verdict == "none") & (norm.box == 0)].sort_values("p")
    if normal.empty:
        normal = norm.sort_values("p")
    pick = {"fractured_clear": clear.iloc[0], "fractured_subtle": subtle.iloc[0], "not_fractured": normal.iloc[0]}

    out = Path(a.out) / "samples"
    out.mkdir(parents=True, exist_ok=True)
    for name, r in pick.items():
        img = gray(files[r.image_id])
        img.thumbnail((1024, 1024))
        img.save(out / f"{name}.jpg", quality=90)
        print(f"{name}.jpg <- {r.image_id}  p={r.p:.3f}  {r.verdict}  box={r.box:.2f}")
    df.to_csv(out / "test_scores.csv", index=False)
    (out / "samples.json").write_text(json.dumps(
        {k: {"image_id": v.image_id, "region": v.region, "probability": round(v.p, 4), "verdict": v.verdict,
             "best_box_score": round(v.box, 3)} for k, v in pick.items()} |
        {"rule": "clear = highest-probability fractured test image with a detector box; subtle = referred fractured image "
                 "closest to 0.5; normal = lowest-probability non-fractured image with no box",
         "test_images": len(df)}, indent=1))

    if a.upload:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ["HF_TOKEN"])
        api.upload_folder(repo_id=SPACE_REPO, repo_type="space", folder_path=str(Path(a.out)),
                          commit_message="Samples: clear fracture, subtle fracture, no fracture (chosen by script)")
        api.upload_file(path_or_fileobj="space/index.html", path_in_repo="index.html", repo_id=SPACE_REPO,
                        repo_type="space", commit_message="Safe local-file loading (Safari) and three samples")
        print(f"https://huggingface.co/spaces/{SPACE_REPO}")


if __name__ == "__main__":
    main()
