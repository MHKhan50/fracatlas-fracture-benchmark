# Changelog

## 0.1.0 (2026-11-20)

- FracAtlas download from figshare, metadata reader and a fixed 70/15/15 split stratified by fracture label and body region.
- Classifiers: small CNN, ConvNeXt-Tiny, DINOv2-S fine-tuned and DINOv2-S frozen probe; three seeds; early stopping on validation AUROC.
- YOLO11s localisation on the same split, with an image-level score from the highest box confidence.
- Metrics with bootstrap intervals, temperature scaling, abstention curves and per-region results; released weights chosen on validation.
- Kaggle notebook with a smoke test before the full run; tests and CI.
