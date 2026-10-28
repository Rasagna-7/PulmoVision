import shutil
from pathlib import Path
import pandas as pd

df = pd.read_csv("data/test_predictions.csv")
out = Path("demo_images")
out.mkdir(exist_ok=True)

n = 0
for label in ["Normal", "COVID-19", "CAP"]:
    g = df[df["label"] == label]
    for kind, sub, want in [("correct", g[g["pred"] == label], 3),
                            ("wrong", g[g["pred"] != label], 1)]:
        # one image per patient so the demo shows different people
        sub = sub.sample(frac=1, random_state=5).drop_duplicates("patient_id").head(want)
        for _, r in sub.iterrows():
            src = Path(r["path"])
            name = f"{label}_{kind}_pred-{r['pred']}_{src.name}"
            shutil.copy(src, out / name)
            n += 1

print("Copied", n, "images to", out.resolve())