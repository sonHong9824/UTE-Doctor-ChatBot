import os
import torch
import torch.nn as nn

from torchvision import datasets
from torchvision import transforms
from torchvision import models

from torch.utils.data import DataLoader

# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = r"D:\TLCN\Chatbot\UTE-Doctor-ChatBot\DermNet"

BATCH_SIZE = 16
EPOCHS = 3

# =========================
# IMAGE TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =========================
# LOAD DATA
# =========================
train_path = os.path.join(DATA_DIR, "train")
test_path = os.path.join(DATA_DIR, "test")

print("Train path:", train_path)
print("Test path:", test_path)

train_dataset = datasets.ImageFolder(
    train_path,
    transform=transform
)

test_dataset = datasets.ImageFolder(
    test_path,
    transform=transform
)

print("Số lớp:", len(train_dataset.classes))
print("Ví dụ lớp:", train_dataset.classes[:5])

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# =========================
# MODEL
# =========================
model = models.resnet18(pretrained=True)

model.fc = nn.Linear(
    model.fc.in_features,
    len(train_dataset.classes)
)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)

# =========================
# TRAIN
# =========================
for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}"
    )

# =========================
# SAVE
# =========================
torch.save(
    {
        "model": model.state_dict(),
        "classes": train_dataset.classes
    },
    "skin_model.pth"
)

print("Đã lưu skin_model.pth")