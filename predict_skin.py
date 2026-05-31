from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms, models
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = torch.load(
    "skin_model.pth",
    map_location=DEVICE
)

classes = checkpoint["classes"]

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    len(classes)
)

model.load_state_dict(
    checkpoint["model"]
)

model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def predict_image(path):

    image = Image.open(path).convert("RGB")

    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        output = model(image)

        probs = F.softmax(output, dim=1)

        top_probs, top_idx = torch.topk(
            probs,
            3
        )

    results = []

    for p, i in zip(
        top_probs[0],
        top_idx[0]
    ):

        results.append(
            (
                classes[i],
                float(p)
            )
        )

    return results