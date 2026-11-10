"""Web demo (Streamlit): classify a news article with the best model.

    streamlit run app.py

On first start it downloads Reuters-21578 if needed and trains Naive Bayes +
TF-IDF (a few seconds); the trained model is cached for later visits.
"""

from pathlib import Path

import streamlit as st

from scripts.download_data import download
from src.data_loader import load_reuters_dataset
from src.metrics import accuracy, macro_f1
from src.model import TextClassificationModel

# Short descriptions of the most common Reuters topics.
TOPICS = {
    "earn": "Earnings and company results", "acq": "Mergers and acquisitions",
    "crude": "Crude oil", "money-fx": "Currency markets", "grain": "Grain",
    "trade": "International trade", "interest": "Interest rates", "ship": "Shipping",
    "sugar": "Sugar", "coffee": "Coffee", "gold": "Gold", "money-supply": "Money supply",
    "gnp": "GNP / economic growth", "cpi": "Consumer prices / inflation", "cocoa": "Cocoa",
    "jobs": "Employment", "copper": "Copper", "reserves": "Foreign reserves",
    "rubber": "Rubber", "ipi": "Industrial production", "alum": "Aluminium",
    "nat-gas": "Natural gas", "iron-steel": "Iron and steel", "veg-oil": "Vegetable oil",
}

EXAMPLES = {
    "Acquisition": "The technology giant announced today that it has agreed to acquire the smaller "
                   "software startup for 500 million dollars in cash. The merger is expected to close next month.",
    "Oil": "OPEC ministers agreed to cut crude output by one million barrels per day to support "
           "oil prices, which fell to their lowest level this year.",
    "Earnings": "The company reported first quarter net profit of 12.5 mln dlrs, or 45 cts a share, "
                "up from 9.8 mln dlrs a year earlier. Revenues rose 14 pct.",
    "Interest rates": "The central bank raised its key lending rate by half a point to 7.5 pct, "
                      "citing inflation pressure, and said further increases were possible.",
}


@st.cache_resource(show_spinner="Training the model on Reuters-21578 (first visit only)...")
def get_model():
    data = download(Path("data/reuters21578"))
    X_train, y_train, X_test, y_test = load_reuters_dataset(str(data))
    model = TextClassificationModel("tfidf", "naive_bayes").fit(X_train, y_train)
    pred = model.predict(X_test)
    return model, accuracy(y_test, pred), macro_f1(y_test, pred), len(X_train), len(X_test)


st.set_page_config(page_title="Reuters news classifier", page_icon="📰")
st.title("Reuters news classifier")
st.caption(
    "Naive Bayes + TF-IDF written from scratch in Python (no ML libraries), trained on "
    "the Reuters-21578 corpus. Seminar project, Artificial Intelligence, University of West Bohemia."
)

model, acc, f1, n_train, n_test = get_model()

if "text" not in st.session_state:
    st.session_state.text = EXAMPLES["Acquisition"]

st.write("Try an example or paste any English news article:")
cols = st.columns(len(EXAMPLES))
for col, (name, text) in zip(cols, EXAMPLES.items()):
    if col.button(name, use_container_width=True):
        st.session_state.text = text

text = st.text_area("News article", key="text", height=170, label_visibility="collapsed")

if text.strip():
    top = model.explain(text, top=3)
    label, prob = top[0]
    st.subheader(f"{TOPICS.get(label, label)}  ·  `{label}`")
    for lab, p in top:
        st.progress(p, text=f"{TOPICS.get(lab, lab)} (`{lab}`): {p:.0%}")
else:
    st.info("Type or paste some text to classify it.")

with st.expander("How good is the model?"):
    st.markdown(
        f"Evaluated on the {n_test:,} Reuters test articles (trained on {n_train:,}): "
        f"**accuracy {acc:.1%}**, **macro F1 {f1:.3f}**. Macro F1 is lower because many of the "
        "80 topics have only a handful of articles. Full results for every combination "
        "of features and classifiers are in `results/RESULTS.md` in the repository.\n\n"
        "The percentages are Naive Bayes posterior probabilities. Naive Bayes treats every "
        "word as independent evidence, so on a long text it is usually close to 100 % sure "
        "of its first choice; read them as a ranking rather than a calibrated confidence."
    )
