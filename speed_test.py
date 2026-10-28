import time
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

LABELS = {"Normal": 0, "COVID-19": 1, "CAP": 2}
IMG_SIZE = 224

tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

class CTDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(r["path"]).convert("RGB")
        return tfm(img), LABELS[r["label"]]

if __name__ == "__main__":
    ds = CTDataset("data/train.csv")
    loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)
    print("Training images:", len(ds))

    t0 = time.time()
    x, y = next(iter(loader))
    t_load = time.time() - t0
    print("Batch shape:", tuple(x.shape), "| first labels:", y[:8].tolist())
    print(f"Time to load one batch: {t_load:.1f} s")

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    model.train()

    times = []
    for _ in range(3):
        t0 = time.time()
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        times.append(time.time() - t0)
    print("Seconds per training step:", [round(t, 1) for t in times])

    batches = len(ds) / 32
    est = (min(times) + t_load) * batches / 60
    print(f"Rough time for one full epoch: {est:.0f} minutes")