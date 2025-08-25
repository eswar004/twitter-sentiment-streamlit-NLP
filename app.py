import streamlit as st
import joblib
import re
import pandas as pd

MODEL_PATH = "models/sentiment140_sgd_hashing.pkl"

# ----- keep inference cleaning identical to training -----
def basic_clean(s: str) -> str:
    s = s.lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)   # URLs
    s = re.sub(r"@\w+", " ", s)               # @handles
    s = re.sub(r"#", " ", s)                  # remove # symbol, keep words
    s = re.sub(r"\s+", " ", s).strip()
    return s

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

clf = load_model()
CLASSES = list(clf.classes_)  # ["negative","positive"]

st.set_page_config(page_title="Sentiment140 (Binary) — Streamlit", page_icon="📝")
st.title("📝 Sentiment Analysis (Positive / Negative)")
st.caption("Model: HashingVectorizer + TF-IDF + SGD (logistic) trained on Sentiment140")

# -------- Single text ----------
st.subheader("Single Prediction")
txt = st.text_area("Enter a tweet-like text", height=120, placeholder="e.g., absolutely love this update!!!")
if st.button("Predict sentiment"):
    if txt.strip():
        clean = basic_clean(txt)
        pred = clf.predict([clean])[0]
        proba = clf.predict_proba([clean])[0]
        st.markdown(f"**Prediction:** `{pred}`")
        # show probabilities
        prob_df = pd.DataFrame({"class": CLASSES, "probability": proba})
        prob_df = prob_df.set_index("class")
        st.bar_chart(prob_df)
    else:
        st.warning("Please enter some text.")

st.divider()

# -------- Batch mode ----------
st.subheader("Batch Prediction")
st.caption("Paste multiple lines. One text per line.")
batch = st.text_area("Batch input", height=160, placeholder="line 1\nline 2\nline 3")
if st.button("Run batch"):
    rows = [r.strip() for r in batch.splitlines() if r.strip()]
    if not rows:
        st.warning("No valid lines.")
    else:
        clean_rows = [basic_clean(r) for r in rows]
        preds = clf.predict(clean_rows)
        probas = clf.predict_proba(clean_rows)
        out = []
        for orig, pred, proba in zip(rows, preds, probas):
            out.append({"text": orig, "prediction": pred, **{f"p_{c}": p for c, p in zip(CLASSES, proba)}})
        df = pd.DataFrame(out)
        st.dataframe(df, use_container_width=True)
        st.download_button("Download results CSV", df.to_csv(index=False).encode("utf-8"), "predictions.csv", "text/csv")

st.divider()
st.caption("Tip: This is a binary model (no neutral). It expects tweet-style English text.")
