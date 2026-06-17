"""
predict_skin.py — Skin disease inference with optional TTA
==========================================================
• Loads EfficientNet-B3 checkpoint (backward-compatible with ResNet18 format)
• Supports top-k prediction with calibrated confidence
• Optional Test-Time Augmentation for higher accuracy at the cost of speed
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

BASE_DIR   = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "skin_model.pth"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Module-level cache so the model is loaded only once per process ──
_model     = None
_classes   = None
_transform = None
_tta_transform = None


def _build_eval_transform(checkpoint: dict) -> transforms.Compose:
    image_size = int(checkpoint.get("image_size", 300))
    mean = checkpoint.get("normalize_mean", IMAGENET_MEAN)
    std  = checkpoint.get("normalize_std",  IMAGENET_STD)
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def _build_tta_transform(checkpoint: dict) -> transforms.Compose:
    """Random-crop transform for Test-Time Augmentation passes."""
    image_size = int(checkpoint.get("image_size", 300))
    mean = checkpoint.get("normalize_mean", IMAGENET_MEAN)
    std  = checkpoint.get("normalize_std",  IMAGENET_STD)
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def _build_model(checkpoint: dict) -> nn.Module:
    """Auto-detect architecture from checkpoint (defaults to efficientnet_b3)."""
    arch     = checkpoint.get("architecture", "efficientnet_b3")
    classes  = checkpoint["classes"]
    n        = len(classes)

    if arch == "efficientnet_b3":
        model = models.efficientnet_b3(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, n),
        )
    elif arch in ("resnet18", None):
        # Backward compatibility with old checkpoints
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, n)
    else:
        raise ValueError(f"Unknown architecture in checkpoint: {arch!r}")

    model.load_state_dict(checkpoint["model"])
    model.to(DEVICE)
    model.eval()
    return model


def _load_model():
    global _model, _classes, _transform, _tta_transform

    if _model is not None:
        return _model, _classes, _transform, _tta_transform

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}.\n"
            "Run train_skin_model.py first to generate it."
        )

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if not checkpoint.get("classes"):
        raise ValueError("Checkpoint is missing 'classes' key — retrain the model.")

    _classes       = checkpoint["classes"]
    _model         = _build_model(checkpoint)
    _transform     = _build_eval_transform(checkpoint)
    _tta_transform = _build_tta_transform(checkpoint)

    return _model, _classes, _transform, _tta_transform


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def predict_image(
    path: str | Path,
    top_k: int  = 3,
    tta: bool   = False,
    tta_n: int  = 5,
) -> List[Tuple[str, float]]:
    """
    Predict the top-k most likely skin conditions for an image.

    Parameters
    ----------
    path  : path to the image file (any PIL-readable format)
    top_k : number of top predictions to return
    tta   : enable Test-Time Augmentation (slower but more accurate)
    tta_n : number of TTA passes (only used when tta=True)

    Returns
    -------
    List of (class_name, confidence) tuples, descending by confidence.
    """
    model, classes, transform, tta_transform = _load_model()

    image = Image.open(path).convert("RGB")
    top_k = max(1, min(int(top_k), len(classes)))

    if tta:
        # Average softmax over N augmented views
        probs_sum = torch.zeros(len(classes), device=DEVICE)
        with torch.no_grad():
            for _ in range(tta_n):
                tensor = tta_transform(image).unsqueeze(0).to(DEVICE)
                logits = model(tensor)
                probs_sum += F.softmax(logits[0], dim=0)
        probs = probs_sum / tta_n
    else:
        tensor = transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(tensor)
        probs = F.softmax(logits[0], dim=0)

    top_probs, top_idx = torch.topk(probs, top_k)
    return [
        (str(classes[int(idx.item())]), float(p.item()))
        for p, idx in zip(top_probs, top_idx)
    ]


def predict_bytes(
    image_bytes: bytes,
    top_k: int  = 3,
    tta: bool   = False,
    tta_n: int  = 5,
) -> List[Tuple[str, float]]:
    """
    Same as predict_image but accepts raw image bytes (convenient for web APIs).
    """
    import io
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Write to temp file and reuse predict_image (keeps code DRY)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.save(tmp.name)
        tmp_path = tmp.name

    try:
        return predict_image(tmp_path, top_k=top_k, tta=tta, tta_n=tta_n)
    finally:
        Path(tmp_path).unlink(missing_ok=True)