
# train_cats.py
# Tiny cat vs not-cat classifier with on-the-fly augmentation (PyTorch).

import argparse
import os
import random
from pathlib import Path
from typing import Tuple, Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_dataloaders(
    data_dir: str,
    img_size: int = 224,
    batch_size: int = 8,
    val_ratio: float = 0.2,
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=12),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.02),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.2),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(int(img_size*1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = datasets.ImageFolder(data_dir, transform=train_transform)
    class_to_idx = full_dataset.class_to_idx

    full_dataset_val = datasets.ImageFolder(data_dir, transform=val_transform)

    indices_by_class: Dict[int, List[int]] = {v: [] for v in class_to_idx.values()}
    for idx, (_, y) in enumerate(full_dataset.samples):
        indices_by_class[y].append(idx)

    train_indices, val_indices = [], []
    for cls, idxs in indices_by_class.items():
        random.shuffle(idxs)
        cut = max(1, int(len(idxs) * (1 - val_ratio))) if len(idxs) > 1 else len(idxs)
        train_indices.extend(idxs[:cut])
        val_indices.extend(idxs[cut:])

    train_indices = train_indices if len(train_indices) > 0 else list(range(len(full_dataset)))
    if len(val_indices) == 0 and len(full_dataset) > 1:
        val_indices = [train_indices.pop()]

    train_ds = Subset(full_dataset, train_indices)
    val_ds = Subset(full_dataset_val, val_indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, class_to_idx

def build_model(num_classes: int = 2, pretrained: bool = False) -> nn.Module:
    if pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    avg_loss = total_loss / max(1, total)
    acc = correct / max(1, total)
    return avg_loss, acc

def train(
    data_dir: str = "dataset",
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 1e-3,
    img_size: int = 224,
    pretrained: bool = False,
    val_ratio: float = 0.2,
    num_workers: int = 2,
    seed: int = 42,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    train_loader, val_loader, class_to_idx = build_dataloaders(
        data_dir=data_dir,
        img_size=img_size,
        batch_size=batch_size,
        val_ratio=val_ratio,
        num_workers=num_workers
    )
    print("[Classes]", class_to_idx)

    model = build_model(num_classes=len(class_to_idx), pretrained=pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    best_val_acc = 0.0
    best_path = "models/cats_resnet18.pt"

    import json
    with open("models/class_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, ensure_ascii=False, indent=2)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            running_correct += (preds == y).sum().item()
            running_total += x.size(0)

        train_loss = running_loss / max(1, running_total)
        train_acc = running_correct / max(1, running_total)

        val_loss, val_acc = evaluate(model, val_loader, device)

        log_line = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
        }
        print(f"[Epoch {epoch:02d}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        with open("logs/train_log.jsonl", "a", encoding="utf-8") as f:
            f.write(str(log_line) + "\n")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)

    print(f"[Best] val_acc={best_val_acc:.4f} -> saved to {best_path}")

def parse_args():
    p = argparse.ArgumentParser(description="Tiny Cat vs Not-Cat Trainer (with on-the-fly augmentation)")
    p.add_argument("--data_dir", type=str, default="dataset", help="Root directory with class subfolders")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--pretrained", action="store_true", help="Use ImageNet-pretrained ResNet18 (requires internet/cached weights)")
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        img_size=args.img_size,
        pretrained=args.pretrained,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
        seed=args.seed,
    )
