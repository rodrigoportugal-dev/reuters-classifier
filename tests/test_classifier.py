"""Unit tests on small hand-made data (no dataset needed).  Run: python -m pytest"""

import math

import pytest

from src.classifiers.knn import KNNClassifier
from src.classifiers.min_distance import MinimumDistanceClassifier
from src.classifiers.naive_bayes import NaiveBayesClassifier
from src.metrics import accuracy, confusion_matrix, macro_f1, per_class_report
from src.model import TextClassificationModel
from src.preprocessing import tokenize_basic, tokenize_clean
from src.vectorizers import TextVectorizer

TRAIN = [
    ("oil prices rise as opec cuts crude output", "crude"),
    ("crude oil barrel price falls on opec news", "crude"),
    ("opec ministers discuss oil production quotas", "crude"),
    ("wheat and corn exports grow, grain harvest strong", "grain"),
    ("grain prices: wheat harvest hit by drought", "grain"),
    ("corn and wheat shipments to china rise", "grain"),
]
TEST = [("opec raises crude oil output", "crude"), ("wheat harvest and corn exports", "grain")]


# --- preprocessing ---------------------------------------------------------

def test_clean_tokenizer_strips_punctuation_and_stop_words():
    assert tokenize_basic("Oil, oil. OIL!") == ["oil,", "oil.", "oil!"]
    assert tokenize_clean("Oil, oil. OIL! The price of oil, Reuter") == ["oil", "oil", "oil", "price", "oil"]


def test_clean_tokenizer_drops_numbers_and_single_letters():
    assert tokenize_clean("a 1987 profit of 5 mln dlrs") == ["profit", "mln", "dlrs"]


# --- vectorizers -----------------------------------------------------------

def test_binary_tf_tfidf_values():
    docs = ["oil oil price", "oil wheat"]
    for method, expected_oil in (("binary", 1.0), ("tf", 2.0), ("tfidf", 0.0)):
        v = TextVectorizer(method)
        X = v.fit_transform(docs)
        oil = v.vocabulary["oil"]
        assert X[0].get(oil, 0.0) == pytest.approx(expected_oil)  # oil is in every doc -> idf 0

    v = TextVectorizer("tfidf")
    X = v.fit_transform(docs)
    assert X[0][v.vocabulary["price"]] == pytest.approx(math.log(2 / 1))


def test_unknown_words_are_ignored_and_vectors_are_sparse():
    v = TextVectorizer("tf").fit(["oil price"])
    assert v.transform(["completely unrelated words"]) == [{}]


# --- classifiers -----------------------------------------------------------

@pytest.mark.parametrize("make", [NaiveBayesClassifier, MinimumDistanceClassifier, lambda: KNNClassifier(k=3)])
@pytest.mark.parametrize("method", ["binary", "tf", "tfidf"])
def test_every_classifier_learns_the_toy_problem(make, method):
    v = TextVectorizer(method)
    X = v.fit_transform([t for t, _ in TRAIN])
    clf = make().fit(X, [l for _, l in TRAIN])
    assert clf.predict(v.transform([t for t, _ in TEST])) == [l for _, l in TEST]


def test_naive_bayes_probabilities_sum_to_one():
    model = TextClassificationModel("tfidf", "naive_bayes").fit(*zip(*TRAIN))
    probs = model.explain("crude oil", top=10)
    assert probs[0][0] == "crude"
    assert sum(p for _, p in probs) == pytest.approx(1.0)


def test_knn_matches_brute_force_cosine():
    v = TextVectorizer("tfidf")
    X = v.fit_transform([t for t, _ in TRAIN])
    knn = KNNClassifier(k=2).fit(X, [l for _, l in TRAIN])
    q = v.transform(["opec oil price"])[0]

    def cos(a, b):
        dot = sum(a[i] * b.get(i, 0) for i in a)
        na, nb = math.sqrt(sum(x * x for x in a.values())), math.sqrt(sum(x * x for x in b.values()))
        return dot / (na * nb) if na and nb else 0.0

    brute = sorted((cos(q, x) for x in X), reverse=True)[:2]
    assert [s for s, _ in knn.neighbours(q)] == pytest.approx(brute)


def test_model_save_and_load(tmp_path):
    model = TextClassificationModel("tf", "min_dist").fit(*zip(*TRAIN))
    path = tmp_path / "m.pkl"
    model.save(path)
    assert TextClassificationModel.load(path).predict(["wheat corn"]) == ["grain"]


# --- metrics ---------------------------------------------------------------

def test_metrics():
    y_true = ["a", "a", "a", "b"]
    y_pred = ["a", "a", "b", "b"]
    assert accuracy(y_true, y_pred) == 0.75
    rep = per_class_report(y_true, y_pred)
    assert rep["a"]["precision"] == 1.0 and rep["a"]["recall"] == pytest.approx(2 / 3)
    assert rep["b"]["precision"] == 0.5 and rep["b"]["recall"] == 1.0
    assert macro_f1(y_true, y_pred) == pytest.approx((0.8 + 2 / 3) / 2)
    cols, m = confusion_matrix(y_true, y_pred, ["a"])
    assert cols == ["a", "other"] and m == [[2, 1]]
