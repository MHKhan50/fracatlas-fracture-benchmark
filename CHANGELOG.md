# Changelog

## 0.2.0 (2026-11-21)

- Near-duplicate results of the Kaggle release run in `results/kaggle-2026-11-21/neardup.json`, the paper and the README; detector export from a copy of the weights.
- Near-duplicate check between splits (`fracatlas_bench.neardup`): perceptual hashes, nearest training image for every validation and test image, test AUROC without near-copies; run in the release notebook.
- Paper: shorter abstract; earlier FracAtlas classification results cited.
- Paper (`paper/`), per-region and referral results, analysis files of the Kaggle run in `results/kaggle-2026-11-21/`.
- Results of the Kaggle run in the README.
- Release step (`fracatlas_bench.release`): ONNX export of ConvNeXt-Tiny and YOLO11s checked against PyTorch, temperature and threshold of the released seed fitted on validation data; release notebook; Hugging Face publishing script with model card.
- Static browser app (`space/`): both ONNX models run in the browser with onnxruntime-web; calibrated probability, validation threshold, referral below confidence 0.8, detector boxes; tested pre- and post-processing.
- Detector scoring decodes test images with Pillow, so truncated JPEGs no longer stop the run.
- Finished steps are skipped: seeds whose outputs exist, the probe and a trained detector; the notebook restores them from an attached earlier version.

- Images: a JPEG that ends a few bytes early is decoded as far as its data goes instead of stopping the run; such files are listed in `truncated_images.json` next to the image cache.

## 0.1.0 (2026-11-20)

- FracAtlas download from figshare, metadata reader and a fixed 70/15/15 split stratified by fracture label and body region.
- Classifiers: small CNN, ConvNeXt-Tiny, DINOv2-S fine-tuned and DINOv2-S frozen probe; three seeds; early stopping on validation AUROC.
- YOLO11s localisation on the same split, with an image-level score from the highest box confidence.
- Metrics with bootstrap intervals, temperature scaling, abstention curves and per-region results; released weights chosen on validation.
- Kaggle notebook with a smoke test before the full run; tests and CI.
