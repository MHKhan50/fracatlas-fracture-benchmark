"""Command line: download, split, train, probe, detect, analyse."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import download, find_root, load_metadata, make_split


def _data(args):
    root = find_root(args.data_dir) or download(args.data_dir)
    df = load_metadata(root)
    df["split"] = make_split(df)
    if getattr(args, "limit", None):
        df = (df.groupby(["split", "fractured"], group_keys=False)
                .apply(lambda g: g.head(max(2, args.limit // 6))).reset_index(drop=True))
    return root, df


def main(argv=None):
    p = argparse.ArgumentParser(prog="fracatlas")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("download", "split", "train", "probe", "detect", "analyse"):
        s = sub.add_parser(name)
        s.add_argument("--data-dir", default="data")
        s.add_argument("--out", default="runs")
        s.add_argument("--limit", type=int, default=None, help="small balanced subset for a smoke test")
    sub.choices["train"].add_argument("--model", required=True, choices=["cnn", "convnext_tiny", "dinov2_small"])
    sub.choices["train"].add_argument("--seeds", default="0,1,2")
    sub.choices["train"].add_argument("--epochs", type=int, default=None)
    sub.choices["train"].add_argument("--batch-size", type=int, default=32)
    sub.choices["detect"].add_argument("--epochs", type=int, default=60)
    sub.choices["detect"].add_argument("--weights", default="yolo11s.pt")
    sub.choices["analyse"].add_argument("--reports", default="reports")
    args = p.parse_args(argv)

    root, df = _data(args)
    if args.cmd == "download":
        print(root, len(df), "images,", int(df["fractured"].sum()), "fractured")
    elif args.cmd == "split":
        Path(args.out).mkdir(parents=True, exist_ok=True)
        df[["image_id", "region", "view", "fractured", "split"]].to_csv(Path(args.out) / "split.csv", index=False)
        print(df.groupby(["split", "fractured"]).size().unstack().to_string())
        print(df.groupby(["split", "region"]).size().unstack().fillna(0).astype(int).to_string())
    elif args.cmd in ("train", "probe"):
        from .train import MODELS, cache_images, probe_dinov2, train_one
        model = args.model if args.cmd == "train" else "dinov2_small"
        size = MODELS[model][1]
        cache = None if args.limit else Path(args.data_dir) / f"cache_{size}.npy"
        images = cache_images(df["path"].tolist(), size, cache)
        if args.cmd == "probe":
            print(json.dumps(probe_dinov2(df, images, args.out)))
        else:
            for seed in [int(s) for s in args.seeds.split(",")]:
                print(json.dumps(train_one(args.model, df, images, seed, args.out, epochs=args.epochs,
                                           batch_size=args.batch_size)), flush=True)
    elif args.cmd == "detect":
        from .detect import run_yolo
        print(json.dumps(run_yolo(df, root, args.out, weights=args.weights, epochs=args.epochs), indent=2))
    elif args.cmd == "analyse":
        from .analysis import analyse
        analyse(df, args.out, args.reports)
        print((Path(args.reports) / "analysis.md").read_text())


if __name__ == "__main__":
    main()
