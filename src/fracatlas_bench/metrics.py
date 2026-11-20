"""Binary metrics, calibration, abstention and bootstrap intervals (numpy and scikit-learn only)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

COVERAGES = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def youden_threshold(y, p) -> float:
    """Validation operating point: the threshold that maximises sensitivity + specificity - 1."""
    fpr, tpr, thr = roc_curve(y, p)
    return float(np.clip(thr[np.argmax(tpr - fpr)], 0.0, 1.0))


def sens_at_spec(y, p, spec=0.90) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    ok = fpr <= 1 - spec
    return float(tpr[ok].max()) if ok.any() else 0.0


def spec_at_sens(y, p, sens=0.90) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    ok = tpr >= sens
    return float(1 - fpr[ok].min()) if ok.any() else 0.0


def at_threshold(y, p, t) -> dict:
    y, pred = np.asarray(y).astype(int), (np.asarray(p) >= t).astype(int)
    tp, fp = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())
    fn, tn = int(((pred == 0) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * sens / (prec + sens) if prec + sens else 0.0
    return {"threshold": float(t), "sensitivity": sens, "specificity": spec, "precision": prec, "f1": f1,
            "accuracy": (tp + tn) / len(y), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def ece(y, p, bins=15) -> float:
    """Expected calibration error of the fracture probability (equal-width bins)."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def fit_temperature(y, logits) -> float:
    """Temperature that minimises the validation negative log-likelihood."""
    y, z = np.asarray(y, dtype=float), np.asarray(logits, dtype=float)

    def nll(log_t):
        q = np.clip(sigmoid(z / np.exp(log_t)), 1e-7, 1 - 1e-7)
        return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))

    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


def risk_coverage(y, p, t, coverages=COVERAGES) -> dict:
    """Keep the most confident records (confidence = max(p, 1 - p)); report accuracy and sensitivity on them."""
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    order = np.argsort(-np.maximum(p, 1 - p), kind="stable")
    out = {}
    for c in coverages:
        keep = order[: max(1, int(round(c * len(y))))]
        m = at_threshold(y[keep], p[keep], t)
        out[f"{c:.1f}"] = {"accuracy": m["accuracy"], "sensitivity": m["sensitivity"],
                           "specificity": m["specificity"], "kept": int(len(keep))}
    conf = np.maximum(p, 1 - p)
    out["aurc"] = float(np.mean([1 - at_threshold(y[order[:k]], p[order[:k]], t)["accuracy"]
                                 for k in range(1, len(y) + 1)]))
    out["refer_rate_at_0.8_confidence"] = float((conf < 0.8).mean())
    return out


def bootstrap_ci(y, p, fn, n=1000, seed=0) -> list:
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            vals.append(fn(y[i], p[i]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def summarise(y_val, z_val, y_test, z_test, regions=None, bootstrap=1000, seed=0) -> dict:
    """All test metrics from validation and test logits: threshold and temperature are fitted on validation only."""
    t_scale = fit_temperature(y_val, z_val)
    p_val, p_test = sigmoid(z_val), sigmoid(z_test)
    pc_val, pc_test = sigmoid(np.asarray(z_val) / t_scale), sigmoid(np.asarray(z_test) / t_scale)
    thr = youden_threshold(y_val, pc_val)
    res = {
        "auroc": float(roc_auc_score(y_test, p_test)),
        "auroc_ci95": bootstrap_ci(y_test, p_test, roc_auc_score, bootstrap, seed),
        "auprc": float(average_precision_score(y_test, p_test)),
        "auprc_ci95": bootstrap_ci(y_test, p_test, average_precision_score, bootstrap, seed),
        "sens_at_90_spec": sens_at_spec(y_test, p_test, 0.90),
        "spec_at_90_sens": spec_at_sens(y_test, p_test, 0.90),
        "operating_point": at_threshold(y_test, pc_test, thr),
        "temperature": t_scale,
        "ece_before": ece(y_test, p_test),
        "ece_after": ece(y_test, pc_test),
        "risk_coverage": risk_coverage(y_test, pc_test, thr),
        "prevalence_test": float(np.mean(y_test)),
    }
    if regions is not None:
        regions = np.asarray(regions)
        res["by_region"] = {}
        for r in sorted(set(regions)):
            m = regions == r
            yy, pp = np.asarray(y_test)[m], pc_test[m]
            entry = {"n": int(m.sum()), "fractured": int(yy.sum())}
            if 0 < yy.sum() < len(yy):
                entry["auroc"] = float(roc_auc_score(yy, pp))
            entry.update({k: v for k, v in at_threshold(yy, pp, thr).items() if k in ("sensitivity", "specificity", "fn", "fp")})
            res["by_region"][r] = entry
    return res
