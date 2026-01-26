#!/usr/bin/env python3
"""
ESPI CNN Baselines (SimpleCNN & ResNet18)
=========================================

Purpose
-------
Train two image-based baselines for ESPI modal classification to compare against your
feature-engineering RF models:
  1) SimpleCNN (≈1.2M params) from scratch
  2) ResNet18 transfer learning (ImageNet) adapted to 1-channel input

Highlights
---------
- Works with **16-bit grayscale PNG** or **.npy** arrays
- Flexible **imbalance handling** (weighted sampler, class-weighted loss, focal loss)
- ESPI-safe augmentations (small rotations, flips, slight intensity jitter)
- Supports **LOBO** (leave-one-band-out) and **LODO** (leave-one-dataset-out)
- Mixed precision (AMP), early stopping, best-by macro-F1, confusion matrix plots
- Reproducible seeds and clean run directory with JSON metrics

Inputs
------
A CSV with at least the columns:
  - path: absolute or relative path to 16-bit PNG or .npy image
  - label: integer in {0..5} per your label_map
Optional columns (recommended for special splits):
  - freq_hz: float (center frequency of this sample)
  - dataset_id: string (e.g., W01/W02/W03) for LODO

Example CSV row:
  path,freq_hz,dataset_id,label
  D:/ESPI/frames/W02/0520Hz/img_0001.png,520.1,W02,2

Usage
-----
python train_espi_cnn_baselines.py \
  --labels_csv FINAL_labels_images.csv \
  --run_dir runs/cnn_simple \
  --model simple \
  --img_size 256 \
  --epochs 60 \
  --batch_size 64 \
  --lr 1e-3 \
  --augment strong

python train_espi_cnn_baselines.py \
  --labels_csv FINAL_labels_images.csv \
  --run_dir runs/resnet18 \
  --model resnet18 \
  --img_size 256 \
  --epochs 50 \
  --batch_size 64 \
  --lr 3e-4 \
  --freeze_until layer2 \
  --augment strong

# LOBO (exclude band [500,525] Hz from train, test only on that band)
python train_espi_cnn_baselines.py \
  --labels_csv FINAL_labels_images.csv \
  --run_dir runs/resnet18_LOBO_500_525 \
  --model resnet18 \
  --lobo_band 500 525 \
  --epochs 50

# LODO (train on W01+W03, test on W02)
python train_espi_cnn_baselines.py \
  --labels_csv FINAL_labels_images.csv \
  --run_dir runs/simple_LODO_W02 \
  --model simple \
  --lodo_holdout W02

Notes
-----
- **Avoid SMOTE for images**: we use sampling weights & class-weighted loss instead.
- For reproducibility, seeds are fixed; still, cuDNN has nondeterministic ops. We set flags to reduce variance.

"""
import os
import json
import math
import random
import argparse
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import GradScaler

import torchvision
from torchvision import models
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# ---------------------
# Reproducibility utils
# ---------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------------
# Dataset
# ---------------------
class ESPICNNDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        img_size: int = 256,
        augment: str = "none",
        allow_vertical_flip: bool = True,
        allow_horizontal_flip: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.augment = augment
        self.allow_v = allow_vertical_flip
        self.allow_h = allow_horizontal_flip

    def __len__(self):
        return len(self.df)

    def _load_image(self, path: str) -> np.ndarray:
        if path.lower().endswith('.npy'):
            arr = np.load(path)
            if arr.ndim == 3 and arr.shape[-1] == 1:
                arr = arr[..., 0]
            arr = arr.astype(np.float32)
            # Robust normalization for unwrapped phase data (may have negative values)
            mn, mx = np.percentile(arr, 1), np.percentile(arr, 99)
            if mx > mn:
                arr = np.clip((arr - mn) / (mx - mn), 0.0, 1.0)
            else:
                arr = np.zeros_like(arr, dtype=np.float32)
            return arr
        # 16-bit PNG
        with Image.open(path) as im:
            im = im.convert('I;16')  # force 16-bit
            arr = np.array(im, dtype=np.uint16)
            arr = arr.astype(np.float32) / 65535.0
            return arr

    def _resize(self, arr: np.ndarray) -> np.ndarray:
        if arr.shape[0] == self.img_size and arr.shape[1] == self.img_size:
            return arr
        pil = Image.fromarray((arr * 65535.0).astype(np.uint16), mode='I;16')
        pil = pil.resize((self.img_size, self.img_size), resample=Image.BILINEAR)
        arr = np.array(pil, dtype=np.uint16).astype(np.float32) / 65535.0
        return arr

    def _augment(self, img: torch.Tensor) -> torch.Tensor:
        # img: (1, H, W) in [0,1]
        if self.augment == 'none':
            return img
        # ESPI-safe light augments: small rotations, flips, slight brightness/contrast
        # Avoid warping or heavy elastic transforms that may alter modal topology
        if self.allow_h and random.random() < 0.5:
            img = torch.flip(img, dims=[2])  # horizontal
        if self.allow_v and random.random() < 0.3:
            img = torch.flip(img, dims=[1])  # vertical
        # small rotation
        if random.random() < 0.5:
            angle = random.uniform(-8, 8)
            img = torchvision.transforms.functional.rotate(img, angle=angle, interpolation=torchvision.transforms.InterpolationMode.BILINEAR, fill=0.0)
        # intensity jitter (simulate exposure changes)
        if random.random() < 0.5:
            # brightness and contrast factors
            b = 1.0 + random.uniform(-0.1, 0.1)
            c = 1.0 + random.uniform(-0.1, 0.1)
            img = (img - 0.5) * c + 0.5
            img = img * b
            img = torch.clamp(img, 0.0, 1.0)
        
        # Strong augmentation: additional effects for more robust training
        if self.augment == 'strong':
            # Gaussian blur (simulate slight defocus)
            if random.random() < 0.5:
                img = torchvision.transforms.functional.gaussian_blur(img, kernel_size=3, sigma=(0.4, 0.8))
            # Gamma correction (simulate different exposure levels)
            if random.random() < 0.5:
                g = 1.0 + random.uniform(-0.07, 0.07)  # gamma-ish
                img = torch.clamp(img**g, 0.0, 1.0)
        
        return img

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = row['path']
        label = int(row['label'])
        arr = self._load_image(path)
        arr = self._resize(arr)
        # to tensor 1xHxW
        img = torch.from_numpy(arr).unsqueeze(0)  # [1,H,W]
        img = self._augment(img)
        return img, label

# ---------------------
# Models
# ---------------------
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

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class ResNet18Gray(nn.Module):
    def __init__(self, num_classes: int = 6, freeze_until: Optional[str] = None):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # adapt first conv to 1-channel by averaging pretrained weights
        w = self.backbone.conv1.weight.data  # [64,3,7,7]
        w_mean = w.mean(dim=1, keepdim=True)  # [64,1,7,7]
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.backbone.conv1.weight.copy_(w_mean)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_feats, num_classes)

        if freeze_until is not None:
            self._freeze_until(freeze_until)

    def _freeze_until(self, stage: str):
        # stage in {"conv1","layer1","layer2","layer3"}; everything before is frozen
        freeze = True
        for name, param in self.backbone.named_parameters():
            if name.startswith('fc.'):
                freeze = False  # always train fc
            elif stage == 'conv1' and name.startswith('layer1.'):
                freeze = False
            elif stage == 'layer1' and name.startswith('layer2.'):
                freeze = False
            elif stage == 'layer2' and name.startswith('layer3.'):
                freeze = False
            elif stage == 'layer3' and name.startswith('layer4.'):
                freeze = False
            param.requires_grad = not freeze

    def forward(self, x):
        return self.backbone(x)

class FocalLoss(nn.Module):
    """Multi-class focal loss with optional class balancing."""
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            alpha = torch.as_tensor(alpha, dtype=torch.float32)
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.dim() != 1:
            raise ValueError('FocalLoss expects targets with shape [N]')
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        ce = F.nll_loss(log_probs, targets, reduction='none')
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_factor = (1.0 - pt).clamp(min=0).pow(self.gamma)
        if self.alpha is not None:
            alpha_factor = self.alpha[targets]
        else:
            alpha_factor = 1.0
        loss = focal_factor * ce * alpha_factor
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss

# ---------------------
# Metrics & plots
# ---------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: List[int]):
    rep = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # Handle case where only one class is present (LOBO mono-class)
    if 'accuracy' in rep:
        acc = rep['accuracy']
    else:
        acc = (y_true == y_pred).mean() if len(y_true) > 0 else 0.0
    macro_f1 = rep['macro avg']['f1-score']
    weighted_f1 = rep['weighted avg']['f1-score']
    return acc, macro_f1, weighted_f1, rep, cm


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], out_path: str, title: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=labels, yticklabels=labels, ylabel='True label', xlabel='Predicted label', title=title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center', color='white' if cm[i, j] > thresh else 'black')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

# ---------------------
# Training / Eval loops
# ---------------------
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
    loss_weights: str
    label_smoothing: float
    focal_gamma: float
    num_workers: int
    present_only_metrics: bool


def make_class_weights(counts: np.ndarray, mode: str):
    counts = counts.astype(float)
    if mode == 'none':
        return np.ones_like(counts, dtype=np.float32)
    if mode == 'inverse':
        w = 1.0 / np.clip(counts, 1, None)
    elif mode == 'sqrt_inv':
        w = 1.0 / np.sqrt(np.clip(counts, 1, None))
    elif mode == 'effective':
        # Cui et al. (CVPR'19): Effective number of samples
        beta = 0.999
        eff = (1.0 - np.power(beta, counts)) / (1.0 - beta)
        w = 1.0 / np.clip(eff, 1e-6, None)
    else:
        raise ValueError(f"Unknown class weight mode: {mode}")
    w = w / w.mean()
    return w.astype(np.float32)


def make_loaders(
    df_train,
    df_val,
    df_test,
    img_size,
    batch_size,
    augment,
    device,
    sampler_mode,
    class_weight_mode,
    num_classes,
    num_workers,
):
    ds_train = ESPICNNDataset(df_train, img_size=img_size, augment=augment)
    ds_val = ESPICNNDataset(df_val, img_size=img_size, augment='none')
    ds_test = ESPICNNDataset(df_test, img_size=img_size, augment='none')

    class_counts = df_train['label'].value_counts().reindex(range(num_classes), fill_value=0).values
    class_weights_np = make_class_weights(class_counts, mode=class_weight_mode)

    pin_memory = device.startswith('cuda')

    pw = (num_workers > 0)
    if sampler_mode == 'weighted':
        sample_weights = class_weights_np[df_train['label'].values]
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).double(),
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            ds_train,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=pw,
        )
    else:
        train_loader = DataLoader(
            ds_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=pw,
        )

    val_loader = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False,
    )
    test_loader = DataLoader(
        ds_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False,
    )

    cw = torch.tensor(class_weights_np, dtype=torch.float32)
    return train_loader, val_loader, test_loader, cw


def build_model(name: str, num_classes: int, freeze_until: Optional[str]):
    if name == 'simple':
        return SimpleCNN(num_classes)
    elif name == 'resnet18':
        return ResNet18Gray(num_classes=num_classes, freeze_until=freeze_until)
    else:
        raise ValueError(f"Unknown model: {name}")


def train_and_eval(cfg: TrainConfig, df_train, df_val, df_test, class_names):
    os.makedirs(cfg.run_dir, exist_ok=True)

    train_loader, val_loader, test_loader, class_weights = make_loaders(
        df_train,
        df_val,
        df_test,
        img_size=cfg.img_size,
        batch_size=cfg.batch_size,
        augment=cfg.augment,
        device=cfg.device,
        sampler_mode=cfg.sampler,
        class_weight_mode=cfg.loss_weights,
        num_classes=len(class_names),
        num_workers=cfg.num_workers,
    )
    class_weights = class_weights.to(cfg.device)
    if cfg.loss_weights == 'none':
        class_weights_for_loss = None
    elif cfg.sampler == 'weighted':
        print('[info] Weighted sampler active; disabling class-weighted loss to avoid double correction.')
        class_weights_for_loss = None
    else:
        class_weights_for_loss = class_weights

    if cfg.focal_gamma > 0 and cfg.label_smoothing > 0:
        print('[warn] label smoothing is ignored when focal loss is enabled; continuing without smoothing.')

    def make_criterion():
        if cfg.focal_gamma > 0:
            alpha = None
            if class_weights_for_loss is not None:
                alpha = (class_weights_for_loss / class_weights_for_loss.sum()).detach()
            return FocalLoss(gamma=cfg.focal_gamma, alpha=alpha)
        return nn.CrossEntropyLoss(weight=class_weights_for_loss, label_smoothing=cfg.label_smoothing)

    model = build_model(cfg.model, num_classes=len(class_names), freeze_until=cfg.freeze_until)
    model.to(cfg.device)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(10, cfg.epochs))
    criterion = make_criterion().to(cfg.device)

    scaler = GradScaler(device='cuda', enabled=cfg.device.startswith('cuda'))

    best_val_macro_f1 = -1.0
    best_epoch = -1
    patience_left = cfg.patience

    def run_epoch(loader, train: bool):
        if train:
            model.train()
        else:
            model.eval()
        running_loss = 0.0
        y_true_all, y_pred_all = [], []
        for imgs, labels in loader:
            imgs = imgs.to(cfg.device, non_blocking=True)
            labels = torch.as_tensor(labels, device=cfg.device)
            with torch.amp.autocast('cuda', enabled=cfg.device.startswith('cuda')):
                logits = model(imgs)
                loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            running_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(logits, dim=1)
            y_true_all.append(labels.detach().cpu().numpy())
            y_pred_all.append(preds.detach().cpu().numpy())
        epoch_loss = running_loss / len(loader.dataset)
        y_true = np.concatenate(y_true_all)
        y_pred = np.concatenate(y_pred_all)
        labels_list = (sorted(np.unique(y_true).tolist())
                       if getattr(cfg, 'present_only_metrics', False)
                       else list(range(len(class_names))))
        acc, macro_f1, weighted_f1, rep, cm = compute_metrics(y_true, y_pred, labels=labels_list)
        return epoch_loss, acc, macro_f1, weighted_f1, rep, cm, labels_list

    history = []
    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc, train_mf1, train_wf1, _, _, _ = run_epoch(train_loader, train=True)
        val_loss, val_acc, val_mf1, val_wf1, val_rep, val_cm, val_labels = run_epoch(val_loader, train=False)
        scheduler.step()

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_macro_f1': train_mf1,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_macro_f1': val_mf1,
        })

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}% val_mF1={val_mf1*100:.2f}%")

        # save best by val macro-F1
        if val_mf1 > best_val_macro_f1:
            best_val_macro_f1 = val_mf1
            best_epoch = epoch
            patience_left = cfg.patience
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'val_macro_f1': val_mf1}, os.path.join(cfg.run_dir, 'best.pt'))
            # also save confusion matrix plot
            val_names = [class_names[i] for i in val_labels]
            plot_confusion_matrix(val_cm, val_names, os.path.join(cfg.run_dir, 'cm_val_best.png'), title=f'Val CM (epoch {epoch})')
            with open(os.path.join(cfg.run_dir, 'val_classification_report.json'), 'w') as f:
                json.dump(val_rep, f, indent=2)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch} (best epoch {best_epoch})")
                break

    # Load best checkpoint
    ckpt = torch.load(os.path.join(cfg.run_dir, 'best.pt'), map_location=cfg.device)
    model.load_state_dict(ckpt['model'])

    # Final test
    test_loss, test_acc, test_mf1, test_wf1, test_rep, test_cm, test_labels = run_epoch(test_loader, train=False)
    test_names = [class_names[i] for i in test_labels]
    plot_confusion_matrix(test_cm, test_names, os.path.join(cfg.run_dir, 'cm_test.png'), title='Test Confusion Matrix')

    # Save test predictions for McNemar/analysis
    try:
        y_true = []
        y_pred = []
        model.eval()
        is_cuda = cfg.device.startswith('cuda')
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=is_cuda):
            for imgs, labels in test_loader:
                imgs = imgs.to(cfg.device, non_blocking=is_cuda)
                logits = model(imgs)
                preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
                y_pred += preds
                y_true += labels.numpy().tolist()
        import csv
        with open(os.path.join(cfg.run_dir, 'test_preds.csv'),'w',newline='',encoding='utf-8') as f:
            w=csv.writer(f); w.writerow(['y_true','y_pred'])
            w.writerows(zip(y_true,y_pred))
        print("Saved test predictions:", os.path.join(cfg.run_dir, 'test_preds.csv'))
    except Exception as e:
        print('WARN: could not save test predictions:', e)

    # Save metrics
    out = {
        'best_epoch': best_epoch,
        'val_best_macro_f1': best_val_macro_f1,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'test_macro_f1': test_mf1,
        'test_weighted_f1': test_wf1,
        'class_names': class_names,
    }
    with open(os.path.join(cfg.run_dir, 'metrics.json'), 'w') as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(cfg.run_dir, 'test_classification_report.json'), 'w') as f:
        json.dump(test_rep, f, indent=2)
    print("Saved:", os.path.join(cfg.run_dir, 'metrics.json'))

# ---------------------
# Splitting strategies (Std / LOBO / LODO)
# ---------------------
def make_splits(df: pd.DataFrame, seed: int, lobo_band: Optional[Tuple[float, float]] = None, lodo_holdout: Optional[str] = None):
    # stratified split 70/15/15 OR special cases
    if lobo_band is not None:
        lo, hi = lobo_band
        in_band = (df['freq_hz'].between(lo, hi))
        df_test = df[in_band].copy()
        df_trainval = df[~in_band].copy()
        # train/val split stratified
        df_train, df_val = stratified_split(df_trainval, seed=seed, val_ratio=0.1765)  # so that overall ~70/15/15
        return df_train, df_val, df_test
    if lodo_holdout is not None:
        assert 'dataset_id' in df.columns, "dataset_id required for LODO"
        df_test = df[df['dataset_id'] == lodo_holdout].copy()
        df_trainval = df[df['dataset_id'] != lodo_holdout].copy()
        df_train, df_val = stratified_split(df_trainval, seed=seed, val_ratio=0.1765)
        return df_train, df_val, df_test
    # standard stratified split
    return stratified_split_full(df, seed=seed)


def stratified_split(df: pd.DataFrame, seed: int, val_ratio: float = 0.2):
    # split df into train/val maintaining label distribution
    labels = df['label'].values
    rng = np.random.RandomState(seed)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    # group by label
    train_idx, val_idx = [], []
    for y in sorted(df['label'].unique()):
        idx = df.index[df['label'] == y].tolist()
        n = len(idx)
        n_val = max(1, int(round(n * val_ratio)))
        rng.shuffle(idx)
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])
    df_train = df.loc[train_idx].reset_index(drop=True)
    df_val = df.loc[val_idx].reset_index(drop=True)
    return df_train, df_val


def stratified_split_full(df: pd.DataFrame, seed: int):
    # 70/15/15 stratified
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df_train, df_temp = stratified_split(df, seed=seed, val_ratio=0.30)  # 70/30
    df_val, df_test = stratified_split(df_temp, seed=seed+1, val_ratio=0.5)  # 15/15
    return df_train, df_val, df_test

# ---------------------
# Main
# ---------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--labels_csv', type=str, required=True)
    p.add_argument('--run_dir', type=str, required=True)
    p.add_argument('--model', type=str, choices=['simple','resnet18'], default='simple')
    p.add_argument('--img_size', type=int, default=256)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--augment', type=str, choices=['none','light','strong'], default='light')
    p.add_argument('--sampler', type=str, choices=['weighted','plain'], default='weighted')
    p.add_argument('--loss_weights', type=str, choices=['inverse','sqrt_inv','effective','none'], default='inverse')
    p.add_argument('--label_smoothing', type=float, default=0.0)
    p.add_argument('--focal_gamma', type=float, default=0.0, help='Set >0 to enable focal loss (e.g., 2.0).')
    p.add_argument('--freeze_until', type=str, default=None, help='resnet18: conv1|layer1|layer2|layer3')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--present_only_metrics', action='store_true',
                   help='Σε LOBO/μονοκλασικά sets, υπολόγισε macro/weighted μόνο στις παρούσες κλάσεις')
    p.add_argument('--lobo_band', type=float, nargs=2, default=None, help='e.g., 500 525')
    p.add_argument('--lodo_holdout', type=str, default=None, help='e.g., W02')
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.run_dir, exist_ok=True)

    df = pd.read_csv(args.labels_csv)
    required = {'path','label'}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required}")
    if args.lobo_band is not None and 'freq_hz' not in df.columns:
        raise ValueError('freq_hz column required for LOBO')
    if args.lodo_holdout is not None and 'dataset_id' not in df.columns:
        raise ValueError('dataset_id column required for LODO')

    # Normalize paths
    df['path'] = df['path'].apply(lambda p: os.path.normpath(p))

    # Class names: fixed to your label_map order
    class_names = [
        'mode_(1,1)H',
        'mode_(1,1)T',
        'mode_(1,2)',
        'mode_(2,1)',
        'mode_higher',
        'other_unknown',
    ]
    # Ensure labels are in 0..5
    assert set(df['label'].unique()).issubset(set(range(len(class_names)))), 'Unexpected label ids in CSV.'

    # Splits
    if args.lobo_band is not None:
        df_train, df_val, df_test = make_splits(df, seed=args.seed, lobo_band=tuple(args.lobo_band))
        split_kind = {'type':'LOBO', 'band': args.lobo_band}
    elif args.lodo_holdout is not None:
        df_train, df_val, df_test = make_splits(df, seed=args.seed, lodo_holdout=args.lodo_holdout)
        split_kind = {'type':'LODO', 'holdout': args.lodo_holdout}
    else:
        df_train, df_val, df_test = stratified_split_full(df, seed=args.seed)
        split_kind = {'type':'stratified_70_15_15'}

    # Save split sizes
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
        loss_weights=args.loss_weights,
        label_smoothing=args.label_smoothing,
        focal_gamma=args.focal_gamma,
        num_workers=args.num_workers,
        present_only_metrics=getattr(args, 'present_only_metrics', False),
    )
    with open(os.path.join(args.run_dir, 'config.json'), 'w') as f:
        json.dump(cfg.__dict__, f, indent=2)

    train_and_eval(cfg, df_train, df_val, df_test, class_names)

if __name__ == '__main__':
    main()

















