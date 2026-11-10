"""Bonus: compare the from-scratch classifiers with scikit-learn on the same data.

Both sides use the same documents, the same split and the same tokenizer, so the
only difference is the implementation.

    python bonus_comparison.py [data_path]
"""

import sys
import warnings

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid

from src.data_loader import load_reuters_dataset
from src.preprocessing import tokenize_clean


def run(data_path="data/reuters21578"):
    warnings.filterwarnings("ignore", category=UserWarning)
    X_train, y_train, X_test, y_test = load_reuters_dataset(data_path)

    # Same tokenization as our TextVectorizer(tokenizer="clean").
    common = dict(tokenizer=tokenize_clean, lowercase=False, token_pattern=None)
    vectorizers = {
        "binary": CountVectorizer(binary=True, **common),
        "tf": CountVectorizer(**common),
        "tfidf": TfidfVectorizer(norm=None, smooth_idf=False, **common),
    }
    classifiers = {
        "min_dist": NearestCentroid,
        "naive_bayes": MultinomialNB,
        "knn": lambda: KNeighborsClassifier(n_neighbors=5, metric="cosine", algorithm="brute"),
    }

    results = {}
    for vname, vec in vectorizers.items():
        Xtr = vec.fit_transform(X_train)
        Xte = vec.transform(X_test)
        for cname, make in classifiers.items():
            pred = make().fit(Xtr, y_train).predict(Xte)
            acc = accuracy_score(y_test, pred)
            f1 = f1_score(y_test, pred, average="macro", labels=sorted(set(y_test)), zero_division=0)
            results[(vname, cname)] = (acc, f1)
            print(f"scikit-learn  {vname:<6} + {cname:<11}  accuracy {acc:.2%}  macro F1 {f1:.3f}")
    return results


if __name__ == "__main__":
    run(*sys.argv[1:2])
