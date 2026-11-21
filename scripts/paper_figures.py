"""Figure and table rows for the paper from results/.../analysis.json. Run from paper/: python ../scripts/paper_figures.py ../results/kaggle-2026-11-21/analysis.json"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

r = json.load(open(sys.argv[1]))
ORDER = ["cnn", "convnext_tiny", "dinov2_probe", "dinov2_small"]
LAB = {"cnn": "Small CNN", "convnext_tiny": "ConvNeXt-Tiny", "dinov2_probe": "DINOv2-S probe", "dinov2_small": "DINOv2-S fine-tuned"}
COL = {"cnn": "#E3A800", "convnext_tiny": "#1F4E9C", "dinov2_probe": "#5FA65A", "dinov2_small": "#C8102E"}
M = r["models"]
names = [n for n in ORDER if n in M]
REG = ["hand", "leg", "hip", "shoulder", "multiple"]

for n in names:
    m = M[n]; e = m["ensemble"]
    print(n, "seeds", m["auroc_per_seed"], "auprc", round(e["auprc"], 3), e["auprc_ci95"], "temp", round(e["temperature"], 2))
    print("  op", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in e["operating_point"].items()})
    print("  spec@90sens", round(e["spec_at_90_sens"], 3), "aurc", round(e["risk_coverage"]["aurc"], 4))
    print("  rc", {k: (round(v["sensitivity"], 3), round(v["accuracy"], 3), v["kept"]) for k, v in e["risk_coverage"].items() if isinstance(v, dict)})
    print("  region", {g: e["by_region"].get(g) for g in REG})

# region table
rows = []
ex = M[names[0]]["ensemble"]["by_region"]
for g in REG:
    if g not in ex: continue
    cells = []
    for n in names:
        b = M[n]["ensemble"]["by_region"][g]
        a = b.get("auroc")
        cells.append(f"{a:.3f} / {b['sensitivity']:.2f}" if a is not None else "--")
    rows.append(f"{g.capitalize()} & {ex[g]['n']} ({ex[g]['fractured']}) & " + " & ".join(cells) + r" \\")
print("\n".join(rows))

# referral table
covs = ["1.0", "0.9", "0.8", "0.7", "0.6", "0.5"]
rows = []
for n in names:
    rc = M[n]["ensemble"]["risk_coverage"]
    rows.append(LAB[n] + " & " + " & ".join(f"{rc[c]['sensitivity']:.2f} / {rc[c]['accuracy']:.2f}" for c in covs) + r" \\")
print("\n".join(rows))

fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
regs = [g for g in ("hand", "leg", "multiple") if g in ex]  # hip and shoulder: 2 test fractures each
w = 0.8 / len(names)
for i, n in enumerate(names):
    br = M[n]["ensemble"]["by_region"]
    a.bar(np.arange(len(regs)) + i * w, [br[g].get("auroc", np.nan) for g in regs], w * 0.92, label=LAB[n], color=COL[n])
    rc = M[n]["ensemble"]["risk_coverage"]
    b.plot([100 * float(c) for c in covs], [rc[c]["sensitivity"] for c in covs], "o-", lw=2, ms=6, color=COL[n], label=LAB[n])
a.set_xticks(np.arange(len(regs)) + w * (len(names) - 1) / 2, [f"{g}\n(n={ex[g]['n']}, {ex[g]['fractured']} fx)" for g in regs], fontsize=8)
a.set_ylabel("Test AUROC (seed ensemble)"); a.set_ylim(0.5, 1.0)
b.set_xlabel("Images kept, most confident first (%)"); b.set_ylabel("Sensitivity on kept images"); b.invert_xaxis()
for ax in (a, b):
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=0.3)
h, l = a.get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=4, frameon=False, fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig("figures/fig_region_referral.pdf"); fig.savefig("figures/fig_region_referral.png", dpi=200)
