import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG_SIZE = 224
NAMES = ["Normal", "COVID-19", "CAP"]

tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model(path="pulmovision_best.pt"):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


class GradCAM:
    """Highlights image regions that influenced the model's prediction."""

    def __init__(self, model):
        self.model = model
        self.acts = None
        self.grads = None
        model.layer4[-1].register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        self.acts = output.detach()
        output.register_hook(self._save_grad)

    def _save_grad(self, grad):
        self.grads = grad.detach()

    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        out = self.model(x)
        probs = torch.softmax(out, dim=1)[0].detach().numpy()
        if class_idx is None:
            class_idx = int(out.argmax(1))
        out[0, class_idx].backward()
        weights = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE),
                            mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.numpy(), probs, class_idx


def overlay(pil_img, cam, alpha=0.4):
    base = np.array(pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))) / 255.0
    heat = matplotlib.colormaps["jet"](cam)[..., :3]
    return np.clip((1 - alpha) * base + alpha * heat, 0, 1)


if __name__ == "__main__":
    df = pd.read_csv("data/test_predictions.csv")
    model = load_model()
    gc = GradCAM(model)

    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for row, label in enumerate(NAMES):
        sample = df[df["label"] == label].sample(2, random_state=3)
        for k, (_, r) in enumerate(sample.iterrows()):
            img = Image.open(r["path"]).convert("RGB")
            x = tfm(img).unsqueeze(0)
            cam, probs, idx = gc(x)
            axes[row, 2 * k].imshow(img.resize((IMG_SIZE, IMG_SIZE)))
            axes[row, 2 * k].set_title(f"True: {label}", fontsize=9)
            axes[row, 2 * k + 1].imshow(overlay(img, cam))
            axes[row, 2 * k + 1].set_title(
                f"Pred: {NAMES[idx]} ({probs[idx] * 100:.0f}%)", fontsize=9)
            axes[row, 2 * k].axis("off")
            axes[row, 2 * k + 1].axis("off")

    plt.tight_layout()
    plt.savefig("data/gradcam_demo.png", dpi=100)
    print("Saved data/gradcam_demo.png")