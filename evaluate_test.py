import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score, roc_auc_score)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train import CTDataset, eval_tfm, NAMES

if __name__ == "__main__":
    test_df = pd.read_csv("data/test.csv")
    print("Test images:", len(test_df))

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model.load_state_dict(torch.load("pulmovision_best.pt", map_location="cpu"))
    model.eval()

    loader = DataLoader(CTDataset(test_df, eval_tfm), batch_size=64, shuffle=False)
    probs, trues = [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader, 1):
            probs.append(torch.softmax(model(x), dim=1).numpy())
            trues += y.tolist()
            if i % 5 == 0:
                print(f"  batch {i}/{len(loader)}", flush=True)

    probs = np.concatenate(probs)
    with open("data/cap_boost.json") as f:
        k = json.load(f)["k"]
    probs[:, 2] *= k
    probs = probs / probs.sum(axis=1, keepdims=True)
    y_true = np.array(trues)
    y_pred = probs.argmax(1)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    auc = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    report = classification_report(y_true, y_pred, target_names=NAMES,
                                   output_dict=True, digits=3)

    print("\n===== TEST RESULTS (unseen patients) =====")
    print(f"Accuracy : {acc:.3f}")
    print(f"Macro-F1 : {macro_f1:.3f}")
    print(f"ROC-AUC  : {auc:.3f} (macro, one-vs-rest)")
    print("\n" + classification_report(y_true, y_pred, target_names=NAMES, digits=3))
    print("Confusion matrix (rows = true, columns = predicted):")
    print("            " + "  ".join(f"{n:>9}" for n in NAMES))
    for n, row in zip(NAMES, cm):
        print(f"{n:>10}  " + "  ".join(f"{v:>9}" for v in row))

    # save for the website
    results = {
        "n_images": int(len(test_df)), "cap_boost": float(k),
        "patients_per_class": {k: int(v) for k, v in
                               test_df.groupby("label")["patient_id"].nunique().items()},
        "accuracy": acc, "macro_f1": macro_f1, "roc_auc_macro": auc,
        "per_class": {n: {"precision": report[n]["precision"],
                          "recall": report[n]["recall"],
                          "f1": report[n]["f1-score"],
                          "support": int(report[n]["support"])} for n in NAMES},
        "confusion_matrix": cm.tolist(), "class_names": NAMES,
    }
    with open("data/test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    out = test_df.copy()
    out["pred"] = [NAMES[i] for i in y_pred]
    out.to_csv("data/test_predictions.csv", index=False)

    cmn = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3)); ax.set_xticklabels(NAMES)
    ax.set_yticks(range(3)); ax.set_yticklabels(NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]}\n({cmn[i, j]*100:.0f}%)", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=9)
    ax.set_title("Test set confusion matrix")
    plt.tight_layout()
    plt.savefig("data/confusion_matrix.png", dpi=120)
    print("\nSaved: data/test_results.json, data/test_predictions.csv, data/confusion_matrix.png")