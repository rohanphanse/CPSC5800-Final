import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights, ResNet50_Weights
from tqdm import tqdm


def get_transforms(img_size, pretrained):
    default_mean, default_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    if pretrained:
        try:
            norm_mean, norm_std = ResNet18_Weights.IMAGENET1K_V1.transforms().mean, ResNet18_Weights.IMAGENET1K_V1.transforms().std
        except Exception:
            norm_mean, norm_std = default_mean, default_std
    else:
        norm_mean, norm_std = default_mean, default_std

    train_tfms = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ]
    )

    val_tfms = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.15)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ]
    )
    return train_tfms, val_tfms


def load_data(data_root, batch_size, num_workers, img_size, pretrained):
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    train_tfms, val_tfms = get_transforms(img_size, pretrained)

    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tfms)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, train_ds.classes


def build_model(model_name, num_classes, pretrained):
    if model_name == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
    elif model_name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet50(weights=weights)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    with torch.no_grad():
        preds = output.argmax(dim=1)
        correct = (preds == target).sum().item()
        return correct / target.size(0)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0
    for images, targets in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            outputs = model(images)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_acc += accuracy(outputs, targets)
        total_batches += 1
    return total_loss / total_batches, total_acc / total_batches


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="val", leave=False):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            total_acc += accuracy(outputs, targets)
            total_batches += 1
    return total_loss / total_batches, total_acc / total_batches


def save_checkpoint(model, class_names, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "classes": class_names}, out_path)
    print(f"Saved best checkpoint to {out_path}")


def parse_args():
    ap = argparse.ArgumentParser(description="Finetune a ResNet on an image classification dataset.")
    ap.add_argument("--data", type=Path, required=True, help="Dataset root containing train/ and val/ subfolders")
    ap.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet50"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--pretrained", action="store_true", help="Start from ImageNet weights")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--output", type=Path, default=Path("checkpoints/resnet_best.pt"))
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, class_names = load_data(
        args.data, args.batch, args.num_workers, args.img_size, args.pretrained
    )
    model = build_model(args.model, num_classes=len(class_names), pretrained=args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch:03d}/{args.epochs}: "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, class_names, args.output)

    print(f"Training done. Best val acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
