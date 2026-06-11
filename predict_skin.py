from __future__ import annotations

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
_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def _load_model():
    global _model, _classes

    if _model is not None:
        return _model, _classes

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Skin model not found at {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    classes = checkpoint["classes"]

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["model"])
    model.to(DEVICE)
    model.eval()

    _model = model
    _classes = classes
    return _model, _classes


def predict_image(path):
    model, classes = _load_model()

    image = Image.open(path).convert("RGB")
    image = _transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image)
        probs = F.softmax(output, dim=1)
        top_probs, top_idx = torch.topk(probs, 3)

    results = []
    for prob, index in zip(top_probs[0], top_idx[0]):
        results.append((classes[index], float(prob)))

    return results
