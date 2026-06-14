from __future__ import annotations

from typing import List, Tuple
from pathlib import Path

from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms, models
import torch.nn as nn

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "skin_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model = None
_classes = None
_transform = None


def _build_transform(checkpoint: dict):
    image_size = int(checkpoint.get("image_size", 224))
    mean = checkpoint.get("normalize_mean", [0.485, 0.456, 0.406])
    std = checkpoint.get("normalize_std", [0.229, 0.224, 0.225])

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def _load_model():
    global _model, _classes, _transform

    if _model is not None:
        return _model, _classes, _transform

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Skin model not found at {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    classes = checkpoint.get("classes")
    if not classes:
        raise ValueError("Checkpoint không có thông tin classes")

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["model"])
    model.to(DEVICE)
    model.eval()

    preprocess = _build_transform(checkpoint)

    _model = model
    _classes = classes
    _transform = preprocess
    return _model, _classes, _transform


def predict_image(path: str | Path, top_k: int = 3) -> List[Tuple[str, float]]:
    model, classes, preprocess = _load_model()

    image = Image.open(path).convert("RGB")
    image = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image)
        probs = F.softmax(output, dim=1)
        top_k = max(1, min(int(top_k), len(classes)))
        top_probs, top_idx = torch.topk(probs, top_k)

    results = []
    for prob, index in zip(top_probs[0], top_idx[0]):
        class_index = int(index.item())
        results.append((str(classes[class_index]), float(prob.item())))

    return results
