import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import f1_score, recall_score

SEED = 42
IMG_SIZE = 224
PER_CLASS = 1800
EPOCHS = 4
BATCH = 32
LABELS = {"Normal": 0, "COVID-19": 1, "CAP": 2}
NAMES = ["Normal", "COVID-19", "CAP"]

torch.manual_seed(SEED)
np.random.seed(SEED)

norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(8),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(), norm])
eval_tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(), norm])

class CTDataset(Dataset):
    def __init__(self, df, tfm):
        self.df = df.reset_index(drop=True)
        self.tfm = tfm
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(r["path"]).convert("RGB")
        return self.tfm(img), LABELS[r["label"]]

def evaluate(model, df):
    loader = DataLoader(CTDataset(df, eval_tfm), batch_size=64, shuffle=False)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            preds += model(x).argmax(1).tolist()
            trues += y.tolist()
    f1 = f1_score(trues, preds, average="macro")
    rec = recall_score(trues, preds, average=None, labels=[0, 1, 2])
    return f1, rec

if __name__ == "__main__":
    train_df = pd.read_csv("data/train.csv")
    val_df = pd.read_csv("data/val.csv")

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    best_f1 = 0.0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        sub = pd.concat([g.sample(min(len(g), PER_CLASS), random_state=epoch)
                         for _, g in train_df.groupby("label")])
        loader = DataLoader(CTDataset(sub, train_tfm), batch_size=BATCH, shuffle=True)
        model.train()
        total, n = 0.0, 0
        for i, (x, y) in enumerate(loader, 1):
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            n += len(y)
            if i % 25 == 0:
                print(f"  epoch {epoch} batch {i}/{len(loader)} loss {total / n:.3f}", flush=True)

        f1, rec = evaluate(model, val_df)
        mins = (time.time() - t0) / 60
        print(f"EPOCH {epoch} done in {mins:.1f} min | train loss {total / n:.3f} | "
              f"val macro-F1 {f1:.3f} | recall " +
              ", ".join(f"{NAMES[k]} {rec[k]:.2f}" for k in range(3)), flush=True)
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "pulmovision_best.pt")
            print("  saved best model -> pulmovision_best.pt", flush=True)

    print("Training finished. Best validation macro-F1:", round(best_f1, 3))