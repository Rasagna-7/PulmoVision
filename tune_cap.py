import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, recall_score, accuracy_score

from train import CTDataset, eval_tfm, NAMES
from gradcam import load_model


def get_probs(model, csv_path):
    df = pd.read_csv(csv_path)
    loader = DataLoader(CTDataset(df, eval_tfm), batch_size=64, shuffle=False)
    probs, trues = [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader, 1):
            probs.append(torch.softmax(model(x), dim=1).numpy())
            trues += y.tolist()
            if i % 10 == 0:
                print(f"  {csv_path}: batch {i}/{len(loader)}", flush=True)
    return np.concatenate(probs), np.array(trues)


def score(probs, y, k):
    p = probs.copy()
    p[:, 2] *= k  # boost the CAP score by factor k
    pred = p.argmax(1)
    f1 = f1_score(y, pred, average="macro")
    acc = accuracy_score(y, pred)
    rec = recall_score(y, pred, average=None, labels=[0, 1, 2])
    return f1, acc, rec


if __name__ == "__main__":
    model = load_model()
    pv, yv = get_probs(model, "data/val.csv")
    pt, yt = get_probs(model, "data/test.csv")

    ks = [1, 1.5, 2, 3, 4, 6, 8, 12]
    print("\nVALIDATION (used to choose k)")
    print(f"{'k':>5} | macro-F1 | accuracy | recall Normal / COVID-19 / CAP")
    best_k, best_f1 = 1, -1
    for k in ks:
        f1, acc, rec = score(pv, yv, k)
        print(f"{k:>5} | {f1:8.3f} | {acc:8.3f} | {rec[0]:.2f} / {rec[1]:.2f} / {rec[2]:.2f}")
        if f1 > best_f1:
            best_f1, best_k = f1, k

    print(f"\nChosen k (best validation macro-F1): {best_k}")
    print("\nTEST (unseen patients, checked once)")
    for k in sorted({1, best_k}):
        f1, acc, rec = score(pt, yt, k)
        tag = "original model" if k == 1 else "with CAP boost"
        print(f"k={k:<4} ({tag}): macro-F1 {f1:.3f} | accuracy {acc:.3f} | "
              f"recall Normal {rec[0]:.2f}, COVID-19 {rec[1]:.2f}, CAP {rec[2]:.2f}")

    with open("data/cap_boost.json", "w") as f:
        json.dump({"k": best_k}, f)
    print("\nSaved data/cap_boost.json")