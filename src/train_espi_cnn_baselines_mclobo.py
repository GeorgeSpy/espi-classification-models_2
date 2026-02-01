#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESPI CNN Baselines — MC-LOBO (per-class gaps) + LODO + focal/weights/AMP
"""

import os
import json
import random
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import torchvision
from torchvision import models
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from torch.amp import autocast, GradScaler

# --------------------- utils ---------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --------------------- dataset ---------------------
class ESPICNNDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_size: int = 256, augment: str = "none") -> None:
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, path: str) -> np.ndarray:
        if path.lower().endswith('.npy'):
            arr = np.load(path)
            if arr.ndim == 3 and arr.shape[-1] == 1:
                arr = arr[..., 0]
            arr = arr.astype(np.float32)
            if arr.size and arr.max() > 1.0:
                arr = arr / (arr.max() + 1e-12)
            return arr
        with Image.open(path) as im:
            im = im.convert('I;16')
            arr = np.array(im, dtype=np.uint16).astype(np.float32) / 65535.0
            return arr

    def _resize(self, arr: np.ndarray) -> np.ndarray:
        if arr.shape[0] == self.img_size and arr.shape[1] == self.img_size:
            return arr
        pil = Image.fromarray((np.clip(arr, 0, 1) * 65535).astype(np.uint16), mode='I;16')
        pil = pil.resize((self.img_size, self.img_size), resample=Image.BILINEAR)
        return (np.array(pil, dtype=np.uint16).astype(np.float32) / 65535.0)

    def _augment(self, img: torch.Tensor) -> torch.Tensor:
        if self.augment == 'none':
            return img
        if random.random() < 0.5:
            img = torch.flip(img, dims=[2])  # horizontal flip
        if random.random() < 0.3:
            img = torch.flip(img, dims=[1])  # vertical flip
        if random.random() < 0.5:
            img = torchvision.transforms.functional.rotate(
                img,
                angle=random.uniform(-8, 8),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                fill=0.0,
            )
        if random.random() < 0.5:
            brightness = 1.0 + random.uniform(-0.1, 0.1)
            contrast = 1.0 + random.uniform(-0.1, 0.1)
            img = (img - 0.5) * contrast + 0.5
            img = torch.clamp(img * brightness, 0.0, 1.0)
        return img

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        arr = self._load_image(row['path'])
        arr = self._resize(arr)
        img = torch.from_numpy(arr).unsqueeze(0)
        img = self._augment(img)
        return img, int(row['label'])

# --------------------- models ---------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)

class ResNet18Gray(nn.Module):
    def __init__(self, num_classes: int = 6, freeze_until: Optional[str] = None):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        w = self.backbone.conv1.weight.data
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.backbone.conv1.weight.copy_(w.mean(dim=1, keepdim=True))
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_feats, num_classes)
        if freeze_until:
            self._freeze_until(freeze_until)

    def _freeze_until(self, stage: str) -> None:
        freeze = True
        for name, param in self.backbone.named_parameters():
            if name.startswith('fc.'):
                freeze = False
            elif stage == 'conv1' and name.startswith('layer1.'):
                freeze = False
            elif stage == 'layer1' and name.startswith('layer2.'):
                freeze = False
            elif stage == 'layer2' and name.startswith('layer3.'):
                freeze = False
            elif stage == 'layer3' and name.startswith('layer4.'):
                freeze = False
            param.requires_grad = not freeze

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
# --------------------- class weights & focal ---------------------
def make_class_weights(counts: np.ndarray, mode: str = 'inverse') -> np.ndarray:
    counts = counts.astype(float)
    if mode == 'none':
        w = np.ones_like(counts, dtype=np.float32)
    elif mode == 'inverse':
        w = 1.0 / np.clip(counts, 1.0, None)
    elif mode == 'sqrt_inv':
        w = 1.0 / np.sqrt(np.clip(counts, 1.0, None))
    elif mode == 'effective':
        beta = 0.999
        eff = (1.0 - np.power(beta, counts)) / (1.0 - beta)
        w = 1.0 / np.clip(eff, 1e-6, None)
    else:
        raise ValueError(f"Unknown loss_weights mode: {mode}")
    w_mean = w.mean()
    if w_mean > 0:
        w = w / w_mean
    return w.astype(np.float32)

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.register_buffer('weight', weight if weight is not None else None)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=1)
        logpt = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = logpt.exp()
        loss = -((1 - pt) ** self.gamma) * logpt
        if self.weight is not None:
            at = self.weight.gather(0, target)
            loss = loss * at
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss

# --------------------- metrics ---------------------
def compute_metrics(y_true, y_pred, labels):
    rep = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = rep.get('accuracy', 0.0)
    mf1 = rep['macro avg']['f1-score']
    wf1 = rep['weighted avg']['f1-score']
    return acc, mf1, wf1, rep, cm

def plot_confusion_matrix(cm: np.ndarray, labels: List[str], out_path: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        ylabel='True label',
        xlabel='Predicted label',
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    thr = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha='center', va='center', color='white' if cm[i, j] > thr else 'black')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

# --------------------- splits ---------------------
def stratified_split(df: pd.DataFrame, seed: int, val_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    rng = np.random.RandomState(seed)
    train_idx, val_idx = [], []
    for label in sorted(df['label'].unique()):
        idx = df.index[df['label'] == label].tolist()
        rng.shuffle(idx)
        n = len(idx)
        n_val = max(1, int(round(n * val_ratio)))
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])
    return df.loc[train_idx].reset_index(drop=True), df.loc[val_idx].reset_index(drop=True)

def stratified_split_full(df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df_train, df_temp = stratified_split(df, seed=seed, val_ratio=0.30)
    df_val, df_test = stratified_split(df_temp, seed=seed + 1, val_ratio=0.50)
    return df_train, df_val, df_test

def make_splits_singleband(df: pd.DataFrame, lobo_band: Tuple[float, float], seed: int):
    lo, hi = lobo_band
    mask = df['freq_hz'].between(lo, hi)
    df_test = df[mask].copy()
    df_trainval = df[~mask].copy()
    df_train, df_val = stratified_split(df_trainval, seed=seed, val_ratio=0.1765)
    return df_train, df_val, df_test

def _rand_split_single_label(sub: pd.DataFrame, seed: int, val_ratio: float = 0.1765, test_ratio: float = 0.15):
    rng = np.random.RandomState(seed)
    idx = sub.sample(frac=1.0, random_state=seed).index.tolist()
    n = len(idx)
    n_test = max(1, int(round(n * test_ratio)))
    n_val = max(1, int(round((n - n_test) * val_ratio)))
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]
    return sub.loc[train_idx], sub.loc[val_idx], sub.loc[test_idx]

def make_splits_per_class_gap(df: pd.DataFrame, pct: float, seed: int):
    assert 'freq_hz' in df.columns, 'freq_hz required for MC-LOBO'
    train_parts, val_parts, test_parts = [], [], []
    for label in sorted(df['label'].unique()):
        sub = df[df['label'] == label].copy()
        usable = sub['freq_hz'].notna().sum() >= 5
        if usable:
            q_lo = sub['freq_hz'].quantile(0.5 - pct / 2.0)
            q_hi = sub['freq_hz'].quantile(0.5 + pct / 2.0)
            mask_test = sub['freq_hz'].between(q_lo, q_hi)
            sub_test = sub[mask_test]
            sub_trainval = sub[~mask_test]
            if len(sub_test) < 2 or len(sub_trainval) < 5:
                t, v, s = _rand_split_single_label(sub, seed=seed)
            else:
                t, v = stratified_split(sub_trainval.assign(label=label), seed=seed, val_ratio=0.1765)
                s = sub_test
        else:
            t, v, s = _rand_split_single_label(sub, seed=seed)
        train_parts.append(t)
        val_parts.append(v)
        test_parts.append(s)
    df_train = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df_val = pd.concat(val_parts).sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)
    df_test = pd.concat(test_parts).sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)
    return df_train, df_val, df_test

def make_splits_lodo(df: pd.DataFrame, holdout: str, seed: int):
    assert 'dataset_id' in df.columns, 'dataset_id required for LODO'
    df_test = df[df['dataset_id'] == holdout].copy()
    df_rest = df[df['dataset_id'] != holdout].copy()
    df_train, df_val = stratified_split(df_rest, seed=seed, val_ratio=0.1765)
    return df_train, df_val, df_test
# --------------------- loaders ---------------------
def make_loaders(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    img_size: int,
    batch_size: int,
    augment: str,
    sampler: str,
    device: str,
    loss_weights_mode: str,
):
    ds_train = ESPICNNDataset(df_train, img_size=img_size, augment=augment)
    ds_val = ESPICNNDataset(df_val, img_size=img_size, augment='none')
    ds_test = ESPICNNDataset(df_test, img_size=img_size, augment='none')

    class_counts = df_train['label'].value_counts().reindex(range(6), fill_value=0).values
    cw_np = make_class_weights(class_counts, mode=loss_weights_mode)

    pin = device.startswith('cuda')
    if sampler == 'weighted':
        sample_weights = cw_np[df_train['label'].values]
        sampler_obj = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).double(),
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(ds_train, batch_size=batch_size, sampler=sampler_obj, num_workers=4, pin_memory=pin)
    else:
        train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=pin)

    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=pin)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=pin)

    return train_loader, val_loader, test_loader, torch.tensor(cw_np, dtype=torch.float32)

# --------------------- training ---------------------
@dataclass
class TrainConfig:
    model: str
    img_size: int
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    augment: str
    run_dir: str
    device: str
    freeze_until: Optional[str]
    patience: int
    sampler: str
    loss_weights_mode: str
    label_smoothing: float
    focal_gamma: Optional[float]
    seed: int

def build_model(name: str, num_classes: int, freeze_until: Optional[str]):
    if name == 'simple':
        return SimpleCNN(num_classes)
    if name == 'resnet18':
        return ResNet18Gray(num_classes=num_classes, freeze_until=freeze_until)
    raise ValueError(f"Unknown model: {name}")

def train_and_eval(cfg: TrainConfig, df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame, class_names: List[str]):
    os.makedirs(cfg.run_dir, exist_ok=True)

    train_loader, val_loader, test_loader, class_weights = make_loaders(
        df_train, df_val, df_test,
        img_size=cfg.img_size,
        batch_size=cfg.batch_size,
        augment=cfg.augment,
        sampler=cfg.sampler,
        device=cfg.device,
        loss_weights_mode=cfg.loss_weights_mode,
    )

    model = build_model(cfg.model, len(class_names), cfg.freeze_until).to(cfg.device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(10, cfg.epochs))

    if cfg.focal_gamma is not None and cfg.focal_gamma > 0:
        weight = None if cfg.loss_weights_mode == 'none' else class_weights.to(cfg.device)
        criterion = FocalLoss(gamma=float(cfg.focal_gamma), weight=weight)
    else:
        weight = None if cfg.loss_weights_mode == 'none' else class_weights.to(cfg.device)
        criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=float(cfg.label_smoothing))

    is_cuda = cfg.device.startswith('cuda')
    scaler = GradScaler(enabled=is_cuda)

    best_macro = -1.0
    best_epoch = -1
    patience_left = cfg.patience
    history = []

    def run_epoch(loader, train: bool):
        model.train(train)
        running = 0.0
        ys, ps = [], []
        for imgs, labels in loader:
            imgs = imgs.to(cfg.device, non_blocking=is_cuda)
            labels = torch.as_tensor(labels, device=cfg.device)
            with autocast(device_type='cuda', enabled=is_cuda):
                logits = model(imgs)
                loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if is_cuda:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            running += loss.item() * imgs.size(0)
            preds = torch.argmax(logits, dim=1)
            ys.append(labels.detach().cpu().numpy())
            ps.append(preds.detach().cpu().numpy())
        y = np.concatenate(ys) if ys else np.array([])
        p = np.concatenate(ps) if ps else np.array([])
        loss = running / max(len(loader.dataset), 1)
        acc, mf1, wf1, rep, cm = compute_metrics(y, p, labels=list(range(len(class_names))))
        return loss, acc, mf1, wf1, rep, cm

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc, tr_mf1, _, _, _ = run_epoch(train_loader, True)
        va_loss, va_acc, va_mf1, _, va_rep, va_cm = run_epoch(val_loader, False)
        scheduler.step()
        history.append({'epoch': epoch, 'train_loss': tr_loss, 'val_loss': va_loss, 'val_acc': va_acc, 'val_macro_f1': va_mf1})
        print(f"Epoch {epoch:03d} | train_loss={tr_loss:.4f} val_loss={va_loss:.4f} val_acc={va_acc*100:.2f}% val_mF1={va_mf1*100:.2f}%")
        if va_mf1 > best_macro:
            best_macro = va_mf1
            best_epoch = epoch
            patience_left = cfg.patience
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'val_macro_f1': float(va_mf1)}, os.path.join(cfg.run_dir, 'best.pt'))
            with open(os.path.join(cfg.run_dir, 'val_classification_report.json'), 'w') as f:
                json.dump(va_rep, f, indent=2)
            plot_confusion_matrix(va_cm, class_names, os.path.join(cfg.run_dir, 'cm_val_best.png'), f'Val CM (epoch {epoch})')
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch} (best {best_epoch})")
                break

    ckpt = torch.load(os.path.join(cfg.run_dir, 'best.pt'), map_location=cfg.device)
    model.load_state_dict(ckpt['model'])

    te_loss, te_acc, te_mf1, te_wf1, te_rep, te_cm = run_epoch(test_loader, False)
    plot_confusion_matrix(te_cm, class_names, os.path.join(cfg.run_dir, 'cm_test.png'), 'Test CM')

    metrics = {
        'best_epoch': best_epoch,
        'val_best_macro_f1': float(best_macro),
        'test_loss': float(te_loss),
        'test_acc': float(te_acc),
        'test_macro_f1': float(te_mf1),
        'test_weighted_f1': float(te_wf1),
        'class_names': class_names,
        'config': cfg.__dict__,
    }
    with open(os.path.join(cfg.run_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(cfg.run_dir, 'test_classification_report.json'), 'w') as f:
        json.dump(te_rep, f, indent=2)
    with open(os.path.join(cfg.run_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    print("Saved:", os.path.join(cfg.run_dir, 'metrics.json'))
# --------------------- main ---------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--labels_csv', type=str, required=True)
    p.add_argument('--run_dir', type=str, required=True)
    p.add_argument('--model', type=str, choices=['simple', 'resnet18'], default='resnet18')
    p.add_argument('--img_size', type=int, default=256)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--augment', type=str, choices=['none', 'light', 'strong'], default='light')
    p.add_argument('--freeze_until', type=str, default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    # splitting options
    p.add_argument('--lobo_band', type=float, nargs=2, default=None)
    p.add_argument('--lodo_holdout', type=str, default=None)
    p.add_argument('--lobo_per_class_pct', type=float, default=None)
    # imbalance / loss
    p.add_argument('--sampler', type=str, choices=['weighted', 'plain'], default='weighted')
    p.add_argument('--loss_weights', type=str, choices=['inverse', 'sqrt_inv', 'effective', 'none'], default='inverse')
    p.add_argument('--label_smoothing', type=float, default=0.0)
    p.add_argument('--focal_gamma', type=float, default=None)
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.run_dir, exist_ok=True)

    df = pd.read_csv(args.labels_csv)
    if not {'path', 'label'}.issubset(df.columns):
        raise ValueError("CSV must contain columns: 'path', 'label'")
    df['path'] = df['path'].apply(os.path.normpath)

    if (args.lobo_band is not None or args.lobo_per_class_pct is not None) and 'freq_hz' not in df.columns:
        raise ValueError('freq_hz column required for LOBO/MC-LOBO')
    if args.lodo_holdout is not None and 'dataset_id' not in df.columns:
        raise ValueError('dataset_id column required for LODO')

    class_names = [
        'mode_(1,1)H',
        'mode_(1,1)T',
        'mode_(1,2)',
        'mode_(2,1)',
        'mode_higher',
        'other_unknown',
    ]

    if args.lobo_per_class_pct is not None:
        df_train, df_val, df_test = make_splits_per_class_gap(df, pct=float(args.lobo_per_class_pct), seed=args.seed)
        split_kind = {'type': 'MC-LOBO_pct', 'pct': args.lobo_per_class_pct}
    elif args.lobo_band is not None:
        df_train, df_val, df_test = make_splits_singleband(df, lobo_band=tuple(args.lobo_band), seed=args.seed)
        split_kind = {'type': 'LOBO_band', 'band': list(map(float, args.lobo_band))}
    elif args.lodo_holdout is not None:
        df_train, df_val, df_test = make_splits_lodo(df, holdout=args.lodo_holdout, seed=args.seed)
        split_kind = {'type': 'LODO', 'holdout': args.lodo_holdout}
    else:
        df_train, df_val, df_test = stratified_split_full(df, seed=args.seed)
        split_kind = {'type': 'stratified_70_15_15'}

    split_info = {
        'train': len(df_train),
        'val': len(df_val),
        'test': len(df_test),
        'split': split_kind,
        'label_counts_train': df_train['label'].value_counts().sort_index().to_dict(),
        'label_counts_val': df_val['label'].value_counts().sort_index().to_dict(),
        'label_counts_test': df_test['label'].value_counts().sort_index().to_dict(),
    }
    with open(os.path.join(args.run_dir, 'split.json'), 'w') as f:
        json.dump(split_info, f, indent=2)

    cfg = TrainConfig(
        model=args.model,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        augment=args.augment,
        run_dir=args.run_dir,
        device=args.device,
        freeze_until=args.freeze_until,
        patience=10,
        sampler=args.sampler,
        loss_weights_mode=args.loss_weights,
        label_smoothing=args.label_smoothing,
        focal_gamma=args.focal_gamma,
        seed=args.seed,
    )
    with open(os.path.join(args.run_dir, 'config.json'), 'w') as f:
        json.dump(cfg.__dict__, f, indent=2)

    train_and_eval(cfg, df_train, df_val, df_test, class_names)

if __name__ == '__main__':
    main()
