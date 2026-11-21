"""Release step: export the chosen ConvNeXt-Tiny and the YOLO detector to ONNX, check them against PyTorch, and
write the calibration used by the browser app (temperature and threshold fitted on validation data only).

Usage (on a machine with the ``train`` extras and ``onnx``/``onnxruntime`` installed):

    python -m fracatlas_bench.release --reports /path/to/reports --out release
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .metrics import ece, fit_temperature, sens_at_spec, sigmoid, youden_threshold

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
REFER_BELOW = 0.8


def calibration_for_seed(split: pd.DataFrame, logits: dict) -> dict:
    """Temperature and Youden threshold of one released seed from its validation logits; test scores for the card."""
    y_val = split.loc[split["split"] == "val", "fractured"].to_numpy()
    y_test = split.loc[split["split"] == "test", "fractured"].to_numpy()
    z_val, z_test = np.asarray(logits["val"], float), np.asarray(logits["test"], float)
    if len(z_val) != len(y_val) or len(z_test) != len(y_test):
        raise ValueError("logits do not match the split: check that split.csv comes from the same run")
    t = fit_temperature(y_val, z_val)
    thr = youden_threshold(y_val, sigmoid(z_val / t))
    p_test = sigmoid(z_test / t)
    pred = p_test >= thr
    conf = np.maximum(p_test, 1 - p_test)
    return {
        "temperature": t,
        "threshold": thr,
        "refer_below_confidence": REFER_BELOW,
        "test": {
            "auroc": float(roc_auc_score(y_test, z_test)),
            "sens_at_90_spec": sens_at_spec(y_test, z_test, 0.90),
            "sensitivity_at_threshold": float(pred[y_test == 1].mean()),
            "specificity_at_threshold": float((~pred[y_test == 0]).mean()),
            "ece_calibrated": ece(y_test, p_test),
            "referred_share": float((conf < REFER_BELOW).mean()),
            "images": int(len(y_test)),
            "fractured": int(y_test.sum()),
        },
    }


def export_classifier(weights: Path, out: Path, size: int = 384) -> dict:
    import onnxruntime as ort
    import timm
    import torch
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    model = timm.create_model("convnext_tiny.fb_in22k_ft_in1k", pretrained=False, num_classes=1)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    x = torch.randn(2, 3, size, size)
    torch.onnx.export(model, x, str(out), input_names=["image"], output_names=["logit"], opset_version=17,
                      dynamic_axes={"image": {0: "batch"}, "logit": {0: "batch"}}, dynamo=False)
    with torch.no_grad():
        ref = model(x).numpy()
    got = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"]).run(None, {"image": x.numpy()})[0]
    diff = float(np.abs(ref - got).max())
    if diff > 1e-3:
        raise RuntimeError(f"ONNX output differs from PyTorch by {diff}")
    return {"meta": ckpt.get("meta", {}), "max_abs_diff_vs_torch": diff}


def export_detector(weights: Path, out: Path, imgsz: int = 640) -> dict:
    from ultralytics import YOLO
    path = YOLO(str(weights)).export(format="onnx", imgsz=imgsz, opset=17, dynamic=False, simplify=False)
    shutil.move(path, out)
    return {"imgsz": imgsz, "output": "[1, 5, N]: cx, cy, w, h, fracture score (no NMS inside)"}


def main(argv=None):
    p = argparse.ArgumentParser(prog="fracatlas-release")
    p.add_argument("--reports", required=True, help="reports folder of the Kaggle run (analysis.json, split.csv, runs/, weights/)")
    p.add_argument("--out", default="release")
    a = p.parse_args(argv)
    rep, out = Path(a.reports), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    analysis = json.loads((rep / "analysis.json").read_text())
    seed = analysis["released_weights"]["convnext_tiny"]["seed"]
    split = pd.read_csv(rep / "split.csv")
    cal = calibration_for_seed(split, np.load(rep / "runs" / f"convnext_tiny_s{seed}_logits.npz"))
    clf = export_classifier(rep / "weights" / "convnext_tiny_best.pt", out / "convnext_tiny_fracatlas.onnx")
    det = export_detector(rep / "weights" / "yolo_best.pt", out / "yolo11s_fracatlas.onnx")
    shutil.copy(rep / "weights" / "convnext_tiny_best.pt", out / "convnext_tiny_fracatlas.pt")
    config = {
        "classifier": {"file": "convnext_tiny_fracatlas.onnx", "timm_id": "convnext_tiny.fb_in22k_ft_in1k",
                       "seed": seed, "best_val_auroc": analysis["released_weights"]["convnext_tiny"]["best_val_auroc"],
                       "input": "grayscale, letterboxed to 384x384 (aspect kept, black padding), repeated to 3 channels",
                       "size": 384, "mean": MEAN, "std": STD, **cal, "onnx_check": clf["max_abs_diff_vs_torch"]},
        "detector": {"file": "yolo11s_fracatlas.onnx", "box_score_min": 0.25, "nms_iou": 0.7,
                     "test_map50": analysis["yolo"]["test"]["map50"], **det},
        "ensemble_results": {k: {"auroc": v["ensemble"]["auroc"], "auroc_ci95": v["ensemble"]["auroc_ci95"]}
                             for k, v in analysis["models"].items()},
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
