"""Publish the released model and the browser app to Hugging Face.

    python scripts/publish_hf.py --release release --split reports/split.csv --images /path/to/FracAtlas

Needs a write token in the environment variable HF_TOKEN. Creates (or updates):
  - model  MHkhan50/fracatlas-convnext-tiny  (ONNX + PyTorch weights, config.json, model card)
  - space  MHkhan50/fracture-xray-check      (static page that runs both ONNX models in the browser)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

MODEL_REPO = "MHkhan50/fracatlas-convnext-tiny"
SPACE_REPO = "MHkhan50/fracture-xray-check"
GITHUB = "https://github.com/MHKhan50/fracatlas-fracture-benchmark"

NAMES = {"cnn": "Small CNN", "convnext_tiny": "ConvNeXt-Tiny", "dinov2_probe": "DINOv2-S frozen probe",
         "dinov2_small": "DINOv2-S fine-tuned"}


def model_card(cfg: dict) -> str:
    c, t, d = cfg["classifier"], cfg["classifier"]["test"], cfg["detector"]
    rows = "\n".join(f"| {NAMES.get(k, k)} | {v['auroc']:.3f} [{v['auroc_ci95'][0]:.3f}, {v['auroc_ci95'][1]:.3f}] |"
                     for k, v in cfg["ensemble_results"].items())
    return f"""---
license: mit
library_name: timm
pipeline_tag: image-classification
tags: [medical-imaging, x-ray, fracture-detection, convnext, yolo, onnx, fracatlas]
datasets: [FracAtlas]
metrics: [roc_auc]
---

# FracAtlas fracture classifier (ConvNeXt-Tiny) and detector (YOLO11s)

**Research use only. Not a medical device.** The outputs are scores against the labels of one dataset. They are not a
diagnosis and must not be used for patient care.

Image-level fracture classifier for musculoskeletal X-rays (hand, leg, hip, shoulder), fine-tuned from
`{c['timm_id']}` on FracAtlas, plus a YOLO11s fracture-box detector trained on the same split. Both are exported to
ONNX. Code, split, paper and the full comparison: [{GITHUB.split('/')[-1]}]({GITHUB}). Try it in the browser:
[fracture-xray-check](https://huggingface.co/spaces/{SPACE_REPO}).

## Files

| File | What it is |
|---|---|
| `convnext_tiny_fracatlas.onnx` | classifier, input `image` [N, 3, 384, 384], output `logit` [N, 1] |
| `convnext_tiny_fracatlas.pt` | the same weights for PyTorch/timm (`state_dict`, `meta`) |
| `yolo11s_fracatlas.onnx` | detector, input [1, 3, 640, 640], output [1, 5, N] = cx, cy, w, h, score (apply NMS yourself) |
| `config.json` | preprocessing, temperature, threshold and test results |

## Preprocessing

Convert to grayscale; resize so the longer side is 384 px (aspect ratio kept) and pad to 384 x 384 with black,
centred; scale to [0, 1]; repeat to three channels; normalise with ImageNet mean {c['mean']} and std {c['std']}.
Probability = sigmoid(logit / T) with **T = {c['temperature']:.3f}**. Decision threshold (Youden, on validation)
= **{c['threshold']:.3f}**. If max(p, 1 - p) < {c['refer_below_confidence']}, refer the image instead of answering.

## Which weights, and how they were chosen

Seed {c['seed']} of three, chosen by **validation** AUROC ({c['best_val_auroc']:.3f}); the test set was not used for
any choice. The ONNX export matches PyTorch to {c['onnx_check']:.1e} (largest absolute difference on random inputs).

## Results on the held-out test split ({t['images']} images, {t['fractured']} fractured)

This released seed: AUROC {t['auroc']:.3f}; sensitivity {t['sensitivity_at_threshold']:.2f} and specificity
{t['specificity_at_threshold']:.2f} at the threshold; sensitivity {t['sens_at_90_spec']:.2f} at 90% specificity;
ECE after temperature scaling {t['ece_calibrated']:.3f}; {100 * t['referred_share']:.0f}% of images referred at
confidence < 0.8. Detector box mAP@0.5 on the test images: {d['test_map50']:.3f}.

Seed ensembles (three seeds, logits averaged), test AUROC with 95% bootstrap intervals:

| Model | AUROC [95% CI] |
|---|---|
{rows}

## Limitations

- FracAtlas has no patient ids, so the split is by image; images of one patient may be in both training and test
  data, which can inflate the scores.
- One dataset from three hospitals in Bangladesh; no external validation. Expect lower performance on other
  hospitals, devices, children, or body parts not in FracAtlas.
- Hip and shoulder have few test fractures; per-region numbers are uncertain.
- The detector's boxes are much less reliable than the classifier's image score.

## Data and licence

Trained on FracAtlas (Abedeen et al., 2023, *Scientific Data* 10:521,
[doi:10.1038/s41597-023-02432-4](https://doi.org/10.1038/s41597-023-02432-4)), CC BY 4.0. Weights: MIT.
Author: Muzammal Hussain ([ORCID 0009-0001-2570-5686](https://orcid.org/0009-0001-2570-5686)).
"""


SPACE_README = f"""---
title: Fracture X-ray Check
emoji: 🦴
colorFrom: blue
colorTo: red
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Fracture classifier and detector for X-rays, in the browser
models: [{MODEL_REPO}]
---

Research demo of a ConvNeXt-Tiny fracture classifier and a YOLO11s detector trained on FracAtlas. Both models run
in the browser with onnxruntime-web; images are not uploaded. **Not a medical device.**

Code and paper: {GITHUB}. Model card: https://huggingface.co/{MODEL_REPO}.

The two sample images are from the FracAtlas test split (Abedeen et al., 2023, CC BY 4.0).
"""


def samples(split_csv: Path, images_root: Path, out: Path) -> None:
    import pandas as pd
    from PIL import Image
    df = pd.read_csv(split_csv)
    test = df[df["split"] == "test"]
    files = {p.name: p for p in images_root.rglob("*.jpg")}
    out.mkdir(parents=True, exist_ok=True)
    for label, name in ((1, "fractured.jpg"), (0, "not_fractured.jpg")):
        image_id = test[test["fractured"] == label]["image_id"].iloc[0]
        img = Image.open(files[image_id]).convert("L")
        img.thumbnail((1024, 1024))
        img.save(out / name, quality=90)
        print(name, "<-", image_id)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--release", default="release")
    p.add_argument("--split", required=True)
    p.add_argument("--images", required=True, help="FracAtlas folder (for the two sample images)")
    p.add_argument("--space-dir", default="space")
    a = p.parse_args()
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    rel = Path(a.release)
    cfg = json.loads((rel / "config.json").read_text())
    (rel / "README.md").write_text(model_card(cfg))
    api.create_repo(MODEL_REPO, repo_type="model", exist_ok=True)
    api.upload_folder(repo_id=MODEL_REPO, repo_type="model", folder_path=str(rel),
                      commit_message="FracAtlas ConvNeXt-Tiny and YOLO11s: ONNX, weights, config and model card")
    stage = Path("space_upload")
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir()
    for f in ("index.html", "core.mjs"):
        shutil.copy(Path(a.space_dir) / f, stage / f)
    (stage / "README.md").write_text(SPACE_README)
    samples(Path(a.split), Path(a.images), stage / "samples")
    api.create_repo(SPACE_REPO, repo_type="space", space_sdk="static", exist_ok=True)
    api.upload_folder(repo_id=SPACE_REPO, repo_type="space", folder_path=str(stage),
                      commit_message="Fracture X-ray Check: browser app")
    print(f"https://huggingface.co/{MODEL_REPO}\nhttps://huggingface.co/spaces/{SPACE_REPO}")


if __name__ == "__main__":
    main()
