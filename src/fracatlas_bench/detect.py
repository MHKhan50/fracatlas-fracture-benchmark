"""Fracture localisation with Ultralytics YOLO on the same split, plus an image-level score from the boxes."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np


def run_yolo(df, root, out_dir, weights="yolo11s.pt", epochs=60, imgsz=640, seed=0, batch=16) -> dict:
    from sklearn.metrics import roc_auc_score
    from ultralytics import YOLO

    from .data import yolo_dataset
    from .metrics import sens_at_spec
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    yaml = yolo_dataset(df, root, out / "yolo_data")
    model = YOLO(weights)
    model.train(data=str(yaml), epochs=epochs, imgsz=imgsz, batch=batch, seed=seed, deterministic=True,
                patience=15, project=str(out / "yolo_runs"), name="train", exist_ok=True, verbose=False, plots=True)
    best = out / "yolo_runs" / "train" / "weights" / "best.pt"
    model = YOLO(str(best))
    res = {}
    for part in ("val", "test"):
        m = model.val(data=str(yaml), split=part, imgsz=imgsz, batch=batch, project=str(out / "yolo_runs"),
                      name=f"eval_{part}", exist_ok=True, verbose=False, plots=False)
        res[part] = {"map50": float(m.box.map50), "map50_95": float(m.box.map), "precision": float(m.box.mp),
                     "recall": float(m.box.mr)}
    test = df[df["split"] == "test"]
    scores = []
    for path in test["path"]:
        r = model.predict(path, imgsz=imgsz, conf=0.001, verbose=False)[0]
        scores.append(float(r.boxes.conf.max()) if len(r.boxes) else 0.0)
    scores, y = np.array(scores), test["fractured"].to_numpy()
    res["test"]["image_level_auroc"] = float(roc_auc_score(y, scores))
    res["test"]["image_level_sens_at_90_spec"] = sens_at_spec(y, scores, 0.90)
    res["weights"] = weights
    res["epochs"] = epochs
    res["imgsz"] = imgsz
    np.savez(out / "yolo_test_scores.npz", scores=scores, y=y, image_id=test["image_id"].to_numpy())
    shutil.copy(best, out / "yolo_best.pt")
    (out / "yolo_metrics.json").write_text(json.dumps(res, indent=2))
    return res
