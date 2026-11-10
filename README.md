# Reuters news classifier

Text classification of news articles into topics (`earn`, `acq`, `crude`, `money-fx`, …) on the
**Reuters-21578** corpus, with the feature extraction, three classifiers and the evaluation
metrics **written from scratch in Python**. scikit-learn is used only to check the results.

Seminar project for *Artificial Intelligence* (KIV/UIR-E) at the University of West Bohemia
(Plzeň, Czech Republic), Erasmus+ 2025-26, later reworked and extended.

**[▶ Live demo](https://reuters-classifier-rodrigo.streamlit.app/)** · [Results](results/RESULTS.md) · [Original report](docs/User_Doc.pdf)

## What it does

- **3 document representations:** Binary, Term Frequency (TF) and TF-IDF, on sparse vectors.
- **3 classifiers:**
  - **Multinomial Naive Bayes** with Laplace smoothing and log-probabilities (no underflow).
  - **Minimum Distance** (nearest centroid). It ranks classes by ‖μ‖² − 2·x·μ, which gives the
    same answer as the Euclidean distance with less work.
  - **K-Nearest Neighbours** (k = 5, cosine similarity) with an **inverted index**. It only
    compares a document with training documents that share a word with it: **6× faster than
    brute force, same predictions** (300/300 checked).
- **Evaluation:** accuracy, per-class precision / recall / F1, macro F1 and a confusion matrix.
- **3 ways to use it:** command line (train, evaluate, save), a Tkinter desktop GUI, and a
  Streamlit web demo.

## Results

7,733 training and 3,008 test articles (LEWISSPLIT), 80 topics, first topic of each article.
The corpus is very imbalanced: `earn` and `acq` are 60 % of the test set, so macro F1 is
reported next to accuracy.

| Accuracy | Binary | TF | TF-IDF |
|---|---:|---:|---:|
| Minimum Distance | 80.22 % | 70.55 % | 77.79 % |
| Naive Bayes | 78.49 % | 82.15 % | **83.94 %** |
| KNN (k=5) | 82.98 % | 82.15 % | 77.53 % |

| Macro F1 | Binary | TF | TF-IDF |
|---|---:|---:|---:|
| Minimum Distance | 0.389 | 0.373 | 0.448 |
| Naive Bayes | 0.160 | 0.253 | 0.456 |
| KNN (k=5) | 0.403 | 0.447 | **0.510** |

**What the numbers show**

- **TF-IDF + Naive Bayes** has the best accuracy. **TF-IDF + KNN** has the best macro F1: it
  does better on the small topics.
- **Naive Bayes on Binary/TF features** reaches ~80 % accuracy but a very low macro F1: it
  mostly predicts the big classes. Accuracy alone would hide this.
- **Raw term frequency hurts Minimum Distance** the most: very frequent words dominate the
  centroids.
- The most common mistake is **`interest` predicted as `money-fx`** (38 of 100 articles):
  interest-rate news is usually about currency markets as well. See the
  [confusion matrix](results/RESULTS.md#best-model-naive-bayes--tfidf).

**Checked against scikit-learn** with the same data and tokenizer: Minimum Distance and
Naive Bayes give **exactly the same accuracy** on Binary and TF features. On TF-IDF they differ
by under 1.5 points, because scikit-learn adds 1 to the IDF. Full tables in
[results/RESULTS.md](results/RESULTS.md).

## Changes since the original submission

The first commit is the code as submitted in May 2026. With the original tokenizer, the new
code reproduces the original report exactly (e.g. 82.91 % for Naive Bayes + TF-IDF), so the
improvements below are measured against the same baseline.

| | Submitted | Now |
|---|---|---|
| Best accuracy | 82.91 % | **83.94 %** |
| Tokenization | `lower().split()`: "oil", "oil." and "oil," are different words | letters only, stop words removed |
| Vectors | dense list per document (~30,000 entries) | sparse `dict` (only the words present) |
| Saved model (NB + TF-IDF) | 56 MB | **1.7 MB** |
| KNN search | compares with every training document | inverted index, 6× faster |
| Metrics | accuracy | + per-class P/R/F1, macro F1, confusion matrix |
| scikit-learn comparison | "excellent results", no numbers | full table, same tokenizer |
| Tests | none | 17 unit tests (`pytest`) |
| Dead code | 5 empty or broken files | removed |

The cleaner tokenization helps most combinations, up to +20 points for Minimum Distance + TF.
It lowers KNN + TF-IDF by 1.4 points.

## Usage

Python 3.11+.

```bash
pip install -r requirements.txt
python scripts/download_data.py          # downloads Reuters-21578 into data/reuters21578
```

**Train, evaluate and save a model** (interface required by the assignment):

```bash
python main.py data/reuters21578 data/reuters21578 tfidf naive_bayes nb_tfidf
#              <train set>       <test set>        <binary|tf|tfidf> <min_dist|naive_bayes|knn> <model name>
```

**Classify your own text** with a saved model, in the desktop GUI:

```bash
python main.py nb_tfidf
```

**Web demo:** `streamlit run app.py`

**All experiments** (writes `results/RESULTS.md`, about 5 minutes): `python -m experiments.run_experiments`

**Tests:** `pytest`

## Project structure

```
src/
  data_loader.py       Reuters .sgm parser (LEWISSPLIT, first topic)
  preprocessing.py     tokenizers: "basic" (original) and "clean"
  vectorizers.py       Binary / TF / TF-IDF on sparse vectors
  classifiers/         naive_bayes.py, min_distance.py, knn.py
  metrics.py           accuracy, precision/recall/F1, macro F1, confusion matrix
  model.py             vectorizer + classifier, save / load
main.py                command line (train / GUI)
gui.py                 Tkinter desktop app
app.py                 Streamlit web demo
bonus_comparison.py    same experiments with scikit-learn
experiments/           runs every combination and writes results/
tests/                 unit tests
docs/                  assignment and original report
```

## Author

**Rodrigo Portugal**, Computer Engineering student, University of Burgos ·
[Portfolio](https://rodrigoportugal-dev.github.io) ·
[LinkedIn](https://www.linkedin.com/in/rodrigo-portugal-dev/)

Dataset: Reuters-21578, Distribution 1.0, from the UCI KDD Archive (free for research use).
