from pathlib import Path
import pandas as pd

DATA_DIR = Path(r"C:\Users\rasag\Downloads\archive (1)\curated_data\curated_data")
CLASS_FOLDERS = {"1NonCOVID": "Normal", "2COVID": "COVID-19", "3CAP": "CAP"}

if not DATA_DIR.exists():
    raise SystemExit(f"Folder not found: {DATA_DIR}")

rows = []
for folder, label in CLASS_FOLDERS.items():
    for img in sorted((DATA_DIR / folder).glob("*.png")):
        patient = img.stem.rsplit("_", 1)[0]  # e.g. "P142_8" -> "P142"
        rows.append({"path": str(img), "label": label, "patient_id": patient})

df = pd.DataFrame(rows)
Path("data").mkdir(exist_ok=True)
df.to_csv("data/image_index.csv", index=False)

print("Total images:", len(df))
print("\nImages per class:")
print(df["label"].value_counts())
print("\nPatients per class:")
print(df.groupby("label")["patient_id"].nunique())
print("\nExample patient IDs per class:")
for label, g in df.groupby("label"):
    print(label, g["patient_id"].unique()[:5].tolist())

shared = df.groupby("patient_id")["label"].nunique()
print("\nPatient IDs appearing in more than one class:", int((shared > 1).sum()))