# FracAtlas Fracture Benchmark

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24429477.svg)](https://doi.org/10.5281/zenodo.24429477) · [Model](https://huggingface.co/MHkhan50/fracatlas-convnext-tiny) · [Browser demo](https://huggingface.co/spaces/MHkhan50/fracture-xray-check) · [Paper](paper/main.pdf)

**Fracture classification, localisation, calibration and abstention on musculoskeletal X-rays: a small CNN, ConvNeXt-Tiny and DINOv2 (frozen and fine-tuned), three seeds each, and YOLO boxes, on one fixed split of FracAtlas.**

> **Research use only. Not a medical device.** Outputs are model scores against dataset labels. They are not diagnoses and must not be used for patient care.

## Questions

1. How much do ImageNet-pretrained (ConvNeXt-Tiny) and self-supervised (DINOv2) backbones gain over a small CNN trained from scratch on 4,083 X-rays, once seed variation and test-sampling uncertainty are reported?
2. Where do the models fail: which body regions (hand, leg, hip, shoulder) lose sensitivity?
3. After temperature scaling, how many images must a model refer to a clinician to reach a given sensitivity on the images it keeps?
4. Does a box detector (YOLO) rank images as well as a classifier when its highest box confidence is used as the image score?

## Data

- **FracAtlas** (Abedeen et al., 2023, *Scientific Data* 10:521): 4,083 X-rays of hand, leg, hip and shoulder from three hospitals in Bangladesh, 717 with fractures (922 fracture instances), with classification labels and expert boxes and masks. Figshare: [10.6084/m9.figshare.22363012](https://doi.org/10.6084/m9.figshare.22363012). Licence CC BY 4.0.
- Not redistributed: `fracatlas download` fetches the archive from figshare.
- **Split:** 70/15/15, stratified by fracture label and body region, fixed seed (`fracatlas split` writes `split.csv`). FracAtlas releases no patient ids, so the split is by image; images of the same patient may fall in different partitions (see Limitations).

## Method

| Model | Pretraining | Input | Training |
|---|---|---|---|
| Small CNN | none | 384 px | 25 epochs max, lr 1e-3 |
| ConvNeXt-Tiny | ImageNet-22k → 1k | 384 px | 15 epochs max, lr 1e-4 |
| DINOv2-S/14 fine-tuned | self-supervised (LVD-142M) | 378 px | 15 epochs max, lr 5e-5 |
| DINOv2-S/14 frozen probe | self-supervised | 378 px | logistic regression on the CLS token, C on validation |
| YOLO11s | COCO | 640 px | 60 epochs max, boxes from FracAtlas |

- Images are letterboxed (aspect ratio kept). AdamW, one-cycle schedule, class-weighted binary cross-entropy, flips, small rotations and contrast jitter, mixed precision; early stopping on validation AUROC (patience 5). Seeds 0, 1, 2.
- **Test metrics:** AUROC and AUPRC with 1,000-sample bootstrap intervals, sensitivity at 90% specificity, and sensitivity, specificity and F1 at a threshold chosen on validation (Youden). Seed ensembles average the logits.
- **Calibration:** expected calibration error before and after temperature scaling fitted on validation.
- **Abstention:** images are ranked by confidence max(p, 1 − p); sensitivity and accuracy on the kept images at 100% to 50% coverage, and the share referred at confidence below 0.8.
- **By region:** AUROC, sensitivity and missed fractures per body region.
- **Released weights:** for each model the seed with the best **validation** AUROC (never chosen on test).

## Results

Kaggle run of `notebooks/kaggle_fracatlas.ipynb` (one T4 GPU). Test split: 614 images, 109 fractured. Three seeds per
fine-tuned model; the seed ensemble averages the logits; 95% intervals from 1,000 bootstrap resamples of test images.
Sensitivity at 90% specificity, ECE and the referral rate are for the seed ensemble.

| Model | AUROC, mean ± SD | Seed ensemble AUROC [95% CI] | Sens. at 90% spec. | ECE before → after temperature scaling | Referred (confidence < 0.8) |
|---|---|---|---|---|---|
| Small CNN | 0.791 ± 0.039 | 0.809 [0.763, 0.852] | 0.440 | 0.182 → 0.151 | 45% |
| **ConvNeXt-Tiny** | **0.922 ± 0.006** | **0.928 [0.891, 0.957]** | **0.807** | 0.057 → **0.036** | **10%** |
| DINOv2-S frozen probe | 0.873 | 0.873 [0.833, 0.909] | 0.642 | 0.139 → 0.129 | 36% |
| DINOv2-S fine-tuned | 0.872 ± 0.043 | 0.902 [0.865, 0.934] | 0.743 | 0.092 → 0.085 | 25% |

Paired differences of ensemble AUROC (same bootstrap resamples), ConvNeXt-Tiny minus:
small CNN +0.118 [+0.086, +0.151]; DINOv2 probe +0.054 [+0.026, +0.082]; DINOv2 fine-tuned +0.026 [+0.005, +0.045].

**YOLO11s detector (test):** box precision 0.438, recall 0.346, mAP@0.5 0.321, mAP@0.5:0.95 0.143; as an image
classifier (highest box confidence) AUROC 0.882 and sensitivity 0.752 at 90% specificity.

**Released weights (best validation AUROC, never chosen on test):** ConvNeXt-Tiny seed 1 (0.941), DINOv2-S seed 1
(0.933), small CNN seed 1 (0.854).

**By region and referral (ConvNeXt-Tiny ensemble):** AUROC 0.918 on hand, 0.880 on leg and 0.983 on
multiple-region images. At the validation threshold it finds 78% of fractures with 93% specificity, but only 17 of 32
leg fractures; leg images are 15 of its 24 misses. Referring the least confident 20% raises accuracy on the kept
images from 0.906 to 0.961 (sensitivity 0.78 to 0.82); referring half gives 0.984 (sensitivity 0.90). Full numbers:
`results/kaggle-2026-11-21/analysis.json`.

**Duplicates between splits (`results/kaggle-2026-11-21/neardup.json`):** 20 test and 23 validation images are exact
copies of training images (perceptual-hash distance 0; at least one with the opposite label), 54 test images lie within
4 bits. Without them, ensemble AUROC is 0.926 (exact copies removed) and 0.924 (within 4 bits) for ConvNeXt-Tiny, and
no model moves by more than 0.008; the ranking is unchanged.

**Released ConvNeXt-Tiny (seed 1) alone on test:** AUROC 0.917; at its validation threshold (0.37, temperature 1.75)
sensitivity 0.75 and specificity 0.93; ECE 0.050; 14% of images referred at confidence below 0.8. The ONNX export
matches PyTorch to 3e-6.

In short: pretraining matters (every pretrained model beats the CNN trained from scratch), ConvNeXt-Tiny is both the
best and the most stable across seeds, DINOv2 features are strong frozen but fine-tuning them varies a lot between
seeds, and the detector ranks images well but localises fractures only moderately.

## Release: ONNX models and browser app

`notebooks/kaggle_fracatlas_release.ipynb` (CPU) takes the output of the training notebook, exports the released
ConvNeXt-Tiny and the YOLO11s detector to ONNX (checked against PyTorch), fits that seed's temperature and threshold
on validation data, and, only when `PUBLISH = True`, uploads the model card and the static browser app in `space/`
(`node space/test_core.mjs` tests its pre- and post-processing).

## Paper

`paper/main.pdf` (LaTeX source in `paper/`; figure from `scripts/paper_figures.py`).

## Quick start

```bash
pip install -e ".[train,dev]"
pytest -q
fracatlas download --data-dir data
fracatlas split --data-dir data --out reports
fracatlas probe --data-dir data --out runs
fracatlas train --model convnext_tiny --seeds 0,1,2 --data-dir data --out runs
fracatlas detect --epochs 60 --data-dir data --out runs
fracatlas analyse --data-dir data --out runs --reports reports
```

On Kaggle: import `notebooks/kaggle_fracatlas.ipynb` from this repository (GPU T4, Internet on) and run all; about 3 hours.

## Limitations

- No patient ids: the image-level split may place images of one patient in train and test, which can inflate test scores. FracAtlas also holds exact duplicate radiographs (some with conflicting labels); removing test images that copy a training image leaves the results unchanged (see above), but different views or visits of the same patient cannot be detected.
- One dataset from one country; no external validation. Results do not transfer to other hospitals, devices or populations without testing.
- Most fractured hip and shoulder images are also flagged with another region (the "multiple" group); hip-only and shoulder-only fractures number 10 each, 2 each in the test split, so those per-region values are uninformative.
- Image-level labels only for the classifiers; YOLO uses the released boxes.

## Citation

Please cite FracAtlas: Abedeen, I., Rahman, Md. A., Prottyasha, F. Z., Ahmed, T., Chowdhury, T. M. and Shatabda, S. (2023). FracAtlas: A Dataset for Fracture Classification, Localization and Segmentation of Musculoskeletal Radiographs. *Scientific Data* 10, 521. https://doi.org/10.1038/s41597-023-02432-4

Software: see `CITATION.cff`.

## How to cite

```bibtex
@software{hussain2026fracatlas,
  author    = {Hussain, Muzammal},
  title     = {Pretrained Backbones, Calibration and Referral for Fracture Detection on {FracAtlas}:
               A Three-Seed Comparison of a Small {CNN}, {ConvNeXt}, {DINOv2} and {YOLO}},
  year      = {2026},
  version   = {0.2.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.24429478},
  url       = {https://doi.org/10.5281/zenodo.24429478}
}
```

Please also cite FracAtlas: Abedeen et al. (2023), *Scientific Data* 10, 521, doi:10.1038/s41597-023-02432-4.

