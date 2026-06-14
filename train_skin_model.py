from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "DermNet"
DEFAULT_MODEL_PATH = BASE_DIR / "skin_model.pth"

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transform, eval_transform


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def train(data_dir: Path, model_path: Path, epochs: int = 10, batch_size: int = 32, lr: float = 1e-4):
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            f"Không tìm thấy dữ liệu train/test trong {data_dir}. "
            f"Kỳ vọng có các thư mục: {train_dir} và {test_dir}"
        )

    print("Train path:", train_dir)
    print("Test path:", test_dir)
    print("Device:", DEVICE)

    train_transform, eval_transform = build_transforms()

    train_dataset = datasets.ImageFolder(str(train_dir), transform=train_transform)
    test_dataset = datasets.ImageFolder(str(test_dir), transform=eval_transform)

    if train_dataset.classes != test_dataset.classes:
        raise ValueError("Class train/test không khớp. Vui lòng kiểm tra lại thư mục dữ liệu.")

    print("Số lớp:", len(train_dataset.classes))
    print("Ví dụ lớp:", train_dataset.classes[:5])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=(DEVICE == "cuda"))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=(DEVICE == "cuda"))

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_acc = -1.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)

        train_loss = running_loss / max(len(train_dataset), 1)
        val_loss, val_acc = evaluate(model, test_loader, criterion)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc * 100:.2f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint = {
        "model": model.state_dict(),
        "classes": train_dataset.classes,
        "class_to_idx": train_dataset.class_to_idx,
        "image_size": IMAGE_SIZE,
        "normalize_mean": IMAGENET_MEAN,
        "normalize_std": IMAGENET_STD,
        "best_val_acc": best_acc,
    }
    torch.save(checkpoint, str(model_path))

    print(f"Đã lưu model tại: {model_path}")
    print(f"Best validation accuracy: {best_acc * 100:.2f}%")


def parse_args():
    parser = argparse.ArgumentParser(description="Train model nhận diện bệnh da từ ảnh (DermNet).")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.data_dir, args.output, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)