"""Combine runs: per-seed metrics, seed ensembles with bootstrap intervals, paired differences, figures."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .metrics import COVERAGES, bootstrap_ci, sigmoid, summarise

COLOURS = {"cnn": "#E3A800", "convnext_tiny": "#1F4E9C", "dinov2_small": "#C8102E", "dinov2_probe": "#5FA65A"}
LABELS = {"cnn": "Small CNN", "convnext_tiny": "ConvNeXt-Tiny", "dinov2_small": "DINOv2-S fine-tuned",
          "dinov2_probe": "DINOv2-S frozen probe"}


def analyse(df, run_dir, out_dir, bootstrap=1000) -> dict:
    run_dir, out = Path(run_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    val, test = df[df["split"] == "val"], df[df["split"] == "test"]
    y_val, y_test, regions = val["fractured"].to_numpy(), test["fractured"].to_numpy(), test["region"].to_numpy()
    runs = {}
    for f in sorted(run_dir.glob("*_logits.npz")):
        m = re.match(r"(.+)_s(\d+)_logits", f.stem)
        runs.setdefault(m.group(1), {})[int(m.group(2))] = np.load(f)
    report = {"test_images": int(len(test)), "test_fractured": int(y_test.sum()), "models": {}}
    ens_test = {}
    for name, seeds in runs.items():
        per_seed = {s: summarise(y_val, d["val"], y_test, d["test"], regions, bootstrap=200, seed=s)
                    for s, d in seeds.items()}
        z_val = np.mean([d["val"] for d in seeds.values()], axis=0)
        z_test = np.mean([d["test"] for d in seeds.values()], axis=0)
        ens = summarise(y_val, z_val, y_test, z_test, regions, bootstrap=bootstrap, seed=0)
        ens_test[name] = z_test
        aucs = [r["auroc"] for r in per_seed.values()]
        report["models"][name] = {
            "seeds": sorted(seeds), "auroc_per_seed": {str(s): r["auroc"] for s, r in per_seed.items()},
            "auroc_mean": float(np.mean(aucs)), "auroc_sd": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
            "sens_at_90_spec_mean": float(np.mean([r["sens_at_90_spec"] for r in per_seed.values()])),
            "ece_before_mean": float(np.mean([r["ece_before"] for r in per_seed.values()])),
            "ece_after_mean": float(np.mean([r["ece_after"] for r in per_seed.values()])),
            "ensemble": ens}
    best = max(report["models"], key=lambda k: report["models"][k]["ensemble"]["auroc"])
    report["comparison"] = {}
    for name in ens_test:
        if name == best:
            continue
        d = _paired_diff(y_test, sigmoid(ens_test[best]), sigmoid(ens_test[name]), bootstrap)
        report["comparison"][f"{best}_minus_{name}"] = d
    yolo = run_dir / "yolo_metrics.json"
    if yolo.exists():
        report["yolo"] = json.loads(yolo.read_text())
    _copy_best_weights(run_dir, out, report)
    (out / "analysis.json").write_text(json.dumps(report, indent=2))
    (out / "analysis.md").write_text(_markdown(report))
    _figures(report, out)
    return report


def _paired_diff(y, p_a, p_b, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            vals.append(roc_auc_score(y[i], p_a[i]) - roc_auc_score(y[i], p_b[i]))
    return {"auroc_difference": float(roc_auc_score(y, p_a) - roc_auc_score(y, p_b)),
            "ci95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]}


def _copy_best_weights(run_dir, out, report):
    """For each fine-tuned model keep the seed with the best validation AUROC (never chosen on test)."""
    report["released_weights"] = {}
    for meta_file in run_dir.glob("*_meta.json"):
        meta = json.loads(meta_file.read_text())
        name = meta["model"]
        cur = report["released_weights"].get(name)
        if (run_dir / f"{name}_s{meta['seed']}_best.pt").exists() and (cur is None or meta["best_val_auroc"] > cur["best_val_auroc"]):
            report["released_weights"][name] = {"seed": meta["seed"], "best_val_auroc": meta["best_val_auroc"]}
    (out / "weights").mkdir(exist_ok=True)
    for name, info in report["released_weights"].items():
        shutil.copy(run_dir / f"{name}_s{info['seed']}_best.pt", out / "weights" / f"{name}_best.pt")
    if (run_dir / "yolo_best.pt").exists():
        shutil.copy(run_dir / "yolo_best.pt", out / "weights" / "yolo_best.pt")


def _markdown(r):
    lines = [f"Test images: {r['test_images']} ({r['test_fractured']} fractured).", "",
             "| Model | AUROC, mean ± SD | Seed ensemble AUROC [95% CI] | Sens. at 90% spec. | ECE before → after | Refer rate (conf < 0.8) |",
             "|---|---|---|---|---|---|"]
    for name, m in r["models"].items():
        e = m["ensemble"]
        lines.append(f"| {LABELS.get(name, name)} | {m['auroc_mean']:.3f} ± {m['auroc_sd']:.3f} | {e['auroc']:.3f} "
                     f"[{e['auroc_ci95'][0]:.3f}, {e['auroc_ci95'][1]:.3f}] | {e['sens_at_90_spec']:.3f} | "
                     f"{e['ece_before']:.3f} → {e['ece_after']:.3f} | {e['risk_coverage']['refer_rate_at_0.8_confidence']:.2f} |")
    for k, v in r["comparison"].items():
        lines.append(f"\n{k}: {v['auroc_difference']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")
    if "yolo" in r:
        t = r["yolo"]["test"]
        lines.append(f"\nYOLO test: mAP50 {t['map50']:.3f}, mAP50-95 {t['map50_95']:.3f}, image-level AUROC {t['image_level_auroc']:.3f}")
    return "\n".join(lines) + "\n"


def _figures(r, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    names = list(r["models"])
    regions = sorted({k for m in r["models"].values() for k in m["ensemble"].get("by_region", {})
                      if m["ensemble"]["by_region"][k].get("auroc") is not None})
    w = 0.8 / max(1, len(names))
    for i, n in enumerate(names):
        br = r["models"][n]["ensemble"]["by_region"]
        a.bar(np.arange(len(regions)) + i * w, [br[g].get("auroc", np.nan) for g in regions], w,
              label=LABELS.get(n, n), color=COLOURS.get(n, "#1F4E9C"))
        rc = r["models"][n]["ensemble"]["risk_coverage"]
        b.plot([100 * c for c in COVERAGES], [rc[f"{c:.1f}"]["sensitivity"] for c in COVERAGES], "o-",
               color=COLOURS.get(n, "#1F4E9C"), label=LABELS.get(n, n))
    a.set_xticks(np.arange(len(regions)) + w * (len(names) - 1) / 2, regions)
    a.set_ylabel("Test AUROC (seed ensemble)"); a.set_ylim(0.5, 1.0); a.legend(frameon=False, fontsize=8)
    b.set_xlabel("Images kept, most confident first (%)"); b.set_ylabel("Sensitivity on kept images")
    b.invert_xaxis(); b.legend(frameon=False, fontsize=8)
    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig_region_selective.png", dpi=200); fig.savefig(out / "fig_region_selective.pdf")
    plt.close(fig)
