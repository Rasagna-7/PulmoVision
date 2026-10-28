import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

df = pd.read_csv("data/train.csv")
labels = ["Normal", "COVID-19", "CAP"]

fig, axes = plt.subplots(3, 4, figsize=(12, 9))
for row, label in enumerate(labels):
    sample = df[df["label"] == label].sample(4, random_state=1)
    for col, (_, r) in enumerate(sample.iterrows()):
        axes[row, col].imshow(Image.open(r["path"]), cmap="gray")
        axes[row, col].set_title(f"{label}\n{r['patient_id']}", fontsize=8)
        axes[row, col].axis("off")

plt.tight_layout()
plt.savefig("data/samples.png", dpi=100)
print("Saved data/samples.png")