"""Train and evaluate image-level fracture classifiers. Every run saves validation and test logits."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = {
    # name: (timm id or None, image size, learning rate, epochs)
    "cnn": (None, 384, 1e-3, 25),
    "convnext_tiny": ("convnext_tiny.fb_in22k_ft_in1k", 384, 1e-4, 15),
    "dinov2_small": ("vit_small_patch14_dinov2.lvd142m", 378, 5e-5, 15),
}


def build(name: str):
    import timm
    import torch.nn as nn
    if name == "cnn":
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                 nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                 nn.MaxPool2d(2))
        return nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 256), block(256, 256),
                             nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, 1))
    timm_id, size, _, _ = MODELS[name]
    kwargs = {"img_size": size} if "dinov2" in timm_id else {}
    return timm.create_model(timm_id, pretrained=True, num_classes=1, **kwargs)


def letterbox(path: str, size: int) -> np.ndarray:
    """Grayscale image resized so its longer side is ``size`` and padded to a square (uint8)."""
    from PIL import Image
    img = Image.open(path).convert("L")
    w, h = img.size
    s = size / max(w, h)
    img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
    out = np.zeros((size, size), dtype=np.uint8)
    out[(size - img.size[1]) // 2:(size - img.size[1]) // 2 + img.size[1],
        (size - img.size[0]) // 2:(size - img.size[0]) // 2 + img.size[0]] = np.asarray(img)
    return out


def cache_images(paths, size: int, cache: str | Path | None = None) -> np.ndarray:
    if cache is not None and Path(cache).exists():
        return np.load(cache)
    arr = np.stack([letterbox(p, size) for p in paths])
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, arr)
    return arr


class _Images:
    def __init__(self, images, labels, train):
        import torch
        from torchvision.transforms import v2
        self.x, self.y, self.train = images, labels.astype(np.float32), train
        self.aug = v2.Compose([v2.RandomHorizontalFlip(), v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
                               v2.ColorJitter(brightness=0.2, contrast=0.2)])
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        import torch
        x = torch.from_numpy(self.x[i]).float().div(255).unsqueeze(0).expand(3, -1, -1).contiguous()
        if self.train:
            x = self.aug(x)
        return (x - self.mean) / self.std, torch.tensor(self.y[i])


def _logits(model, loader, device):
    import torch
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        for x, _ in loader:
            out.append(model(x.to(device, non_blocking=True)).float().squeeze(1).cpu().numpy())
    return np.concatenate(out)


def seed_everything(seed: int):
    import random
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def train_one(name, df, images, seed, out_dir, epochs=None, batch_size=32, workers=2, save_weights=True) -> dict:
    """Fine-tune one model with one seed; early stopping on validation AUROC; returns logits and metadata."""
    import torch
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, size, lr, default_epochs = MODELS[name]
    epochs = epochs or default_epochs
    parts = {s: np.where(df["split"].to_numpy() == s)[0] for s in ("train", "val", "test")}
    y = df["fractured"].to_numpy()
    loaders = {s: DataLoader(_Images(images[idx], y[idx], s == "train"), batch_size=batch_size, shuffle=s == "train",
                             num_workers=workers, pin_memory=device.type == "cuda", drop_last=s == "train")
               for s, idx in parts.items()}
    model = build(name).to(device)
    pos = y[parts["train"]].mean()
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float((1 - pos) / pos), dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    steps = epochs * len(loaders["train"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    best, best_state, best_epoch, patience, stale = -1.0, None, 0, 5, 0
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        for x, t in loaders["train"]:
            x, t = x.to(device, non_blocking=True), t.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = loss_fn(model(x).squeeze(1), t)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
        z_val = _logits(model, loaders["val"], device)
        auc = roc_auc_score(y[parts["val"]], z_val)
        print(f"{name} seed={seed} epoch={epoch}/{epochs} val_auroc={auc:.4f}", flush=True)
        if auc > best:
            best, best_epoch, stale = auc, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    z_val, z_test = _logits(model, loaders["val"], device), _logits(model, loaders["test"], device)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{name}_s{seed}_logits.npz", val=z_val, test=z_test)
    meta = {"model": name, "seed": seed, "image_size": size, "lr": lr, "epochs_max": epochs, "best_epoch": best_epoch,
            "best_val_auroc": float(best), "minutes": round((time.time() - start) / 60, 2), "device": str(device),
            "torch": torch.__version__}
    if save_weights:
        torch.save({"state_dict": best_state, "meta": meta}, out / f"{name}_s{seed}_best.pt")
    (out / f"{name}_s{seed}_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def probe_dinov2(df, images, out_dir, seed=0, batch_size=64) -> dict:
    """Frozen DINOv2 features (CLS token) and a class-balanced logistic regression; C chosen on validation AUROC."""
    import timm
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timm_id, size, _, _ = MODELS["dinov2_small"]
    model = timm.create_model(timm_id, pretrained=True, num_classes=0, img_size=size).to(device).eval()
    y = df["fractured"].to_numpy()
    loader = DataLoader(_Images(images, y, False), batch_size=batch_size, shuffle=False, num_workers=2)
    feats = []
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        for x, _ in loader:
            feats.append(model(x.to(device)).float().cpu().numpy())
    f = np.concatenate(feats)
    f = f / np.linalg.norm(f, axis=1, keepdims=True)
    s = df["split"].to_numpy()
    best = None
    for c in (0.1, 0.3, 1, 3, 10, 30):
        clf = LogisticRegression(C=c, class_weight="balanced", max_iter=5000, random_state=seed).fit(f[s == "train"], y[s == "train"])
        auc = roc_auc_score(y[s == "val"], clf.decision_function(f[s == "val"]))
        if best is None or auc > best[0]:
            best = (auc, c, clf)
    auc, c, clf = best
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"dinov2_probe_s{seed}_logits.npz", val=clf.decision_function(f[s == "val"]),
             test=clf.decision_function(f[s == "test"]))
    meta = {"model": "dinov2_probe", "seed": seed, "C": c, "best_val_auroc": float(auc), "features": int(f.shape[1])}
    (out / f"dinov2_probe_s{seed}_meta.json").write_text(json.dumps(meta, indent=2))
    return meta
