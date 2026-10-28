import json

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from gradcam import GradCAM, load_model, tfm, overlay, NAMES

st.set_page_config(page_title="PulmoVision", page_icon="🫁", layout="wide")

with open("data/cap_boost.json") as _f:
    CAP_BOOST = json.load(_f)["k"]

DISCLAIMER = (
    "PulmoVision is an educational research prototype. It is not a medical device "
    "and its output is not a diagnosis. Always consult a qualified doctor."
)

EXPLANATIONS = {
    "Normal": (
        "The model did not find the patterns it links to COVID-19 or pneumonia in this slice. "
        "This does not rule out disease: one slice shows only a small part of the lungs, "
        "and this model misses a number of real cases, especially pneumonia (CAP)."
    ),
    "COVID-19": (
        "The model found patterns similar to those it saw in COVID-19 scans during training, "
        "such as patchy, cloudy areas in the lungs. The coloured areas on the heatmap show "
        "where the model looked most. Many other conditions can look similar on a CT scan."
    ),
    "CAP": (
        "The model found patterns similar to those it saw in community-acquired pneumonia (CAP), "
        "a lung infection that is not COVID-19. The coloured areas on the heatmap show where the "
        "model looked most. Many other conditions can look similar on a CT scan."
    ),
}

NEXT_STEPS = [
    "Show this scan and the full CT study to a doctor (a pulmonologist or radiologist).",
    "Do not start or stop any medicine because of this result. Only a doctor can prescribe treatment.",
    "Get urgent medical care if there is trouble breathing, chest pain, blue lips, or confusion.",
]


@st.cache_resource
def get_gradcam():
    return GradCAM(load_model())


@st.cache_data
def load_index():
    return pd.read_csv("data/image_index.csv")


@st.cache_data
def load_split(name):
    return pd.read_csv(f"data/{name}.csv")


@st.cache_data
def load_results():
    with open("data/test_results.json") as f:
        return json.load(f)


def page_home():
    st.title("🫁 PulmoVision")
    st.caption("AI-assisted chest CT analysis prototype: COVID-19, pneumonia (CAP) or normal")
    st.warning(DISCLAIMER)

    st.markdown(
        "**What it does:** you give it a chest CT slice. It suggests which of three patterns the "
        "slice looks most like, draws a heatmap of the regions that influenced that suggestion, "
        "and explains the result in simple words."
    )

    idx = load_index()
    counts = idx["label"].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CT images", f"{len(idx):,}")
    c2.metric("COVID-19", f"{counts['COVID-19']:,}")
    c3.metric("Normal", f"{counts['Normal']:,}")
    c4.metric("CAP (pneumonia)", f"{counts['CAP']:,}")

    st.subheader("Images and patients per class")
    table = idx.groupby("label").agg(images=("path", "count"), patients=("patient_id", "nunique"))
    st.dataframe(table, use_container_width=True)

    st.subheader("Patient-level train / validation / test split")
    rows, sets = [], {}
    for name in ["train", "val", "test"]:
        d = load_split(name)
        sets[name] = set(d["patient_id"])
        rows.append({"split": name, "images": len(d), "patients": d["patient_id"].nunique()})
    st.dataframe(pd.DataFrame(rows).set_index("split"), use_container_width=True)
    shared = (sets["train"] & sets["val"]) | (sets["train"] & sets["test"]) | (sets["val"] & sets["test"])
    if shared:
        st.error(f"{len(shared)} patients appear in more than one split.")
    else:
        st.success("No patient appears in more than one split, so test results are not inflated by leakage.")

    st.subheader("Sample CT slices from the training data")
    st.image("data/samples.png", use_container_width=True)


def page_analyze():
    st.title("Analyze a CT scan")
    st.warning(DISCLAIMER)

    source = st.radio("Choose input", ["Upload my own slice", "Use a sample from the test set"], horizontal=True)
    img, truth = None, None

    if source.startswith("Upload"):
        up = st.file_uploader("Upload one chest CT slice (PNG or JPG)", type=["png", "jpg", "jpeg"])
        if up is not None:
            try:
                img = Image.open(up).convert("RGB")
            except Exception:
                st.error("Could not read this file. Please upload a PNG or JPG image.")
                return
    else:
        df = load_split("test")
        cls = st.selectbox("Class of the sample", NAMES)
        n = st.selectbox("Sample number", [1, 2, 3, 4, 5])
        row = df[df["label"] == cls].sample(5, random_state=7).reset_index(drop=True).iloc[n - 1]
        img, truth = Image.open(row["path"]).convert("RGB"), row["label"]

    if img is None:
        st.info("Upload a scan to begin.")
        return
    if min(img.size) < 64:
        st.error("This image is too small to analyze. Please upload a larger image.")
        return

    arr = np.asarray(img.resize((224, 224))).astype(float)
    colour = np.abs(arr[..., 0] - arr[..., 1]).mean() + np.abs(arr[..., 1] - arr[..., 2]).mean()
    if colour > 12:
        st.warning("This looks like a colour image, not a grayscale CT slice. Results may be meaningless.")

    with st.spinner("Analyzing the scan..."):
        gc = get_gradcam()
        x = tfm(img).unsqueeze(0)
        cam, probs, idx = gc(x)
        # CAP score boost chosen on the validation set (see tune_cap.py)
        adj = probs.copy()
        adj[2] *= CAP_BOOST
        adj = adj / adj.sum()
        if int(adj.argmax()) != idx:
            idx = int(adj.argmax())
            cam, _, _ = gc(x, class_idx=idx)
        probs = adj

    pred, conf = NAMES[idx], float(probs[idx])
    left, right = st.columns(2)
    left.image(img, caption="Original scan", use_container_width=True)
    right.image(overlay(img, cam),
                caption="Heatmap: regions that influenced the prediction (not a diagnosis)",
                use_container_width=True)

    st.subheader(f"Most likely pattern: {pred}")
    if truth is not None:
        st.caption(f"(This sample's recorded label in the dataset: {truth})")
    for name, p in zip(NAMES, probs):
        st.progress(float(p), text=f"{name}: {p * 100:.1f}%")
    st.caption("These percentages are the model's scores. They are not the chance that you have a "
               "disease, and a score can be high even when the model is wrong.")
    if conf < 0.6:
        st.warning("The model is not confident about this slice. Treat the result as unreliable.")

    st.info(EXPLANATIONS[pred])
    st.markdown("**Suggested next steps**")
    for s in NEXT_STEPS:
        st.markdown(f"- {s}")


def page_results():
    st.title("Model results")
    r = load_results()
    st.caption(f"Evaluated on {r['n_images']:,} slices from patients the model never saw during training.")
    st.caption(f"The CAP score is multiplied by {r.get('cap_boost', 1):g} before the final class is chosen. This factor was picked on the validation set, not the test set.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{r['accuracy'] * 100:.1f}%")
    c2.metric("Macro F1-score", f"{r['macro_f1']:.3f}")
    c3.metric("ROC-AUC (macro)", f"{r['roc_auc_macro']:.3f}")

    st.subheader("Per-class performance")
    pc = pd.DataFrame(r["per_class"]).T
    pc.columns = ["Precision", "Recall", "F1-score", "Test slices"]
    st.dataframe(pc.style.format({"Precision": "{:.3f}", "Recall": "{:.3f}",
                                  "F1-score": "{:.3f}", "Test slices": "{:.0f}"}),
                 use_container_width=True)
    st.caption("Recall = how many real cases of that class the model found. "
               "Precision = how often its call of that class was right.")

    st.subheader("Confusion matrix")
    st.image("data/confusion_matrix.png", width=520)

    st.subheader("Limitations")
    st.markdown(
        "- Pneumonia (CAP) is still under-detected: even after boosting its score, only about 56% of CAP slices were found (41% before).\n"
        "- The test set has only a few patients per class (CAP: 8), so numbers can change a lot with a different split.\n"
        "- The model looks at one 2D slice at a time, not a full 3D scan.\n"
        "- The classes come from different sources and scanners, so the model may partly learn scanner differences.\n"
        "- Heatmaps show where the model looked, not where disease is.\n"
        "- Not clinically validated. Not for diagnosis or treatment decisions."
    )


page = st.sidebar.radio("Go to", ["Home", "Analyze scan", "Model results"])
st.sidebar.caption("Research prototype. Not for medical use.")
if page == "Home":
    page_home()
elif page == "Analyze scan":
    page_analyze()
else:
    page_results()
