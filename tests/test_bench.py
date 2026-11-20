import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from fracatlas_bench.data import load_metadata, make_split, primary_region, yolo_dataset
from fracatlas_bench.metrics import (at_threshold, ece, fit_temperature, risk_coverage, sens_at_spec, sigmoid,
                                     summarise, youden_threshold)


@pytest.fixture
def fake_root(tmp_path):
    root = tmp_path / "FracAtlas"
    rng = np.random.default_rng(0)
    rows = []
    for i in range(120):
        frac = int(i % 4 == 0)
        region = ["hand", "leg", "hip", "shoulder"][i % 4 if i % 5 else 0]
        name = f"IMG{i:07d}.jpg"
        folder = root / "images" / ("Fractured" if frac else "Non_fractured")
        folder.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rng.integers(0, 255, (40, 30), dtype=np.uint8)).save(folder / name)
        if frac:
            (root / "Annotations" / "YOLO").mkdir(parents=True, exist_ok=True)
            (root / "Annotations" / "YOLO" / f"IMG{i:07d}.txt").write_text("0 0.5 0.5 0.1 0.1\n")
        rows.append({"image_id": name, "hand": int(region == "hand"), "leg": int(region == "leg"),
                     "hip": int(region == "hip"), "shoulder": int(region == "shoulder"), "mixed": 0,
                     "hardware": 0, "multiscan": 0, "fractured": frac, "fracture_count": frac,
                     "frontal": 1, "lateral": 0, "oblique": 0})
    pd.DataFrame(rows).to_csv(root / "dataset.csv", index=False)
    return root


def test_metadata_and_split(fake_root):
    df = load_metadata(fake_root)
    assert len(df) == 120 and df["path"].str.endswith(".jpg").all()
    split = make_split(df)
    assert split.equals(make_split(df))
    assert set(split) == {"train", "val", "test"}
    for part in ("val", "test"):
        assert df.loc[split == part, "fractured"].sum() > 0
    assert abs((split == "train").mean() - 0.70) < 0.06


def test_primary_region_multiple():
    df = pd.DataFrame({"hand": [1, 1, 0], "leg": [0, 1, 0], "hip": [0, 0, 0], "shoulder": [0, 0, 0]})
    assert primary_region(df).tolist() == ["hand", "multiple", "unknown"]


def test_yolo_dataset(fake_root, tmp_path):
    df = load_metadata(fake_root)
    df["split"] = make_split(df)
    yaml = yolo_dataset(df, fake_root, tmp_path / "yolo")
    assert "names" in yaml.read_text()
    labels = list((tmp_path / "yolo" / "labels").glob("*/*.txt"))
    assert len(labels) == 120
    assert sum(1 for f in labels if f.read_text().strip()) == int(df["fractured"].sum())


def test_metrics():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    z = (y * 2 - 1) * 1.5 + rng.normal(0, 1, 2000)
    t = youden_threshold(y, sigmoid(z))
    assert 0.2 < t < 0.8
    assert 0.5 < sens_at_spec(y, sigmoid(z)) <= 1
    logit = 3 * z  # calibrated log-odds for class means +-1.5 and unit variance
    over = 3 * logit
    temp = fit_temperature(y, over)
    assert abs(temp - 3) < 0.6
    assert ece(y, sigmoid(over)) > ece(y, sigmoid(over / temp))
    m = at_threshold([1, 0, 1, 0], [0.9, 0.2, 0.3, 0.7], 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    rc = risk_coverage(y, sigmoid(z), 0.5)
    assert rc["0.5"]["accuracy"] >= rc["1.0"]["accuracy"]


def test_summarise_and_analysis(tmp_path, fake_root):
    from fracatlas_bench.analysis import analyse
    df = load_metadata(fake_root)
    df["split"] = make_split(df)
    rng = np.random.default_rng(2)
    yv = df.loc[df["split"] == "val", "fractured"].to_numpy()
    yt = df.loc[df["split"] == "test", "fractured"].to_numpy()
    runs = tmp_path / "runs"; runs.mkdir()
    for name, strength in (("cnn", 0.5), ("convnext_tiny", 2.0)):
        for s in (0, 1):
            np.savez(runs / f"{name}_s{s}_logits.npz", val=strength * (2 * yv - 1) + rng.normal(0, 1, len(yv)),
                     test=strength * (2 * yt - 1) + rng.normal(0, 1, len(yt)))
            (runs / f"{name}_s{s}_meta.json").write_text(json.dumps({"model": name, "seed": s, "best_val_auroc": 0.5 + s / 10}))
    res = summarise(yv, rng.normal(size=len(yv)), yt, rng.normal(size=len(yt)), df.loc[df["split"] == "test", "region"])
    assert set(res["by_region"]) <= {"hand", "leg", "hip", "shoulder"}
    rep = analyse(df, runs, tmp_path / "reports", bootstrap=50)
    assert rep["models"]["convnext_tiny"]["ensemble"]["auroc"] > rep["models"]["cnn"]["ensemble"]["auroc"]
    assert "convnext_tiny_minus_cnn" in rep["comparison"]
    assert (tmp_path / "reports" / "fig_region_selective.png").exists()
