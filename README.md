\# PulmoVision



AI-assisted chest CT slice analysis prototype (educational, not a medical device).

It classifies a CT slice as \*\*COVID-19\*\*, \*\*community-acquired pneumonia (CAP)\*\* or \*\*Normal\*\*,

shows a Grad-CAM heatmap of the regions that influenced the prediction, and explains the result

in simple words with safe next steps (no medicine advice).



> This is a research/education prototype. Its output is not a diagnosis. Always consult a qualified doctor.



\## What is in this repo

\- `train.py` - fine-tunes a pretrained ResNet18 (CPU friendly, balanced sampling per epoch)

\- `evaluate\_test.py` - evaluates on the patient-level test set and saves results for the website

\- `tune\_cap.py` - picks a CAP score boost on the validation set

\- `gradcam.py` - Grad-CAM heatmaps

\- `app.py` - Streamlit website (Home, Analyze scan, Model results)

\- `build\_index.py`, `split\_data.py` - build the image index and the patient-level train/val/test split

\- `pulmovision\_best.pt` - trained model (about 45 MB)



\## Dataset (not included)

Search Kaggle for \*\*"Large COVID-19 CT scan slice dataset"\*\* (about 2 GB, three folders:

`1NonCOVID`, `2COVID`, `3CAP`). Check the dataset page for its license and terms before reuse.

Extract it, then open `build\_index.py` and set `DATA\_DIR` to the folder that contains those three folders.



\## Setup (tested with Python 3.14 on Windows)

```

python -m venv venv

venv\\Scripts\\activate

pip install numpy pandas pillow matplotlib scikit-learn streamlit

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

```



\## Run order

```

python build\_index.py

python split\_data.py

python view\_samples.py

python evaluate\_test.py

python -m streamlit run app.py

```

`pulmovision\_best.pt`, `data/cap\_boost.json`, `data/test\_results.json` and `data/confusion\_matrix.png`

are already in the repo, so training is optional (`python train.py` takes about 40 minutes on a CPU).



\## Results (2,402 test slices from patients never seen in training)

| Metric | Value |

|---|---|

| Accuracy | 85.3% |

| Macro-F1 | 0.817 |

| ROC-AUC (macro) | 0.923 |

| Recall Normal / COVID-19 / CAP | 97.8% / 84.8% / 56.2% |



A CAP score boost (x12) chosen on the validation set raised CAP recall from 41% to 56%.



\## Limitations

\- CAP is still under-detected; the test set has only 8 CAP patients, so numbers can change a lot.

\- The model sees one 2D slice at a time, not a full 3D scan.

\- The classes come from different sources and scanners, so the model may partly learn scanner differences.

\- Heatmaps show where the model looked, not where disease is.

\- Patient IDs come from file names and were not independently verified.

\- Not clinically validated. Not for diagnosis or treatment decisions.

