# FracAtlas Fracture Benchmark

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

Results will be added from the Kaggle run (`notebooks/kaggle_fracatlas.ipynb`).

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

- No patient ids: the image-level split may place images of one patient in train and test, which can inflate test scores.
- One dataset from one country; no external validation. Results do not transfer to other hospitals, devices or populations without testing.
- Hip and shoulder have 63 fractured images each; their per-region estimates are uncertain.
- Image-level labels only for the classifiers; YOLO uses the released boxes.

## Citation

Please cite FracAtlas: Abedeen, I., Rahman, Md. A., Prottyasha, F. Z., Ahmed, T., Chowdhury, T. M. and Shatabda, S. (2023). FracAtlas: A Dataset for Fracture Classification, Localization and Segmentation of Musculoskeletal Radiographs. *Scientific Data* 10, 521. https://doi.org/10.1038/s41597-023-02432-4

Software: see `CITATION.cff`.
