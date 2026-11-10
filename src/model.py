"""A trained model = vectorizer + classifier, with save/load helpers."""

import pickle
from pathlib import Path

from src.classifiers.knn import KNNClassifier
from src.classifiers.min_distance import MinimumDistanceClassifier
from src.classifiers.naive_bayes import NaiveBayesClassifier
from src.vectorizers import METHODS, TextVectorizer

CLASSIFIERS = {
    "min_dist": MinimumDistanceClassifier,
    "naive_bayes": NaiveBayesClassifier,
    "knn": lambda: KNNClassifier(k=5),
}
PARAMETERIZATIONS = METHODS


class TextClassificationModel:
    def __init__(self, parameterization="tfidf", classifier="naive_bayes", tokenizer="clean"):
        if classifier not in CLASSIFIERS:
            raise ValueError(f"Classifier must be one of {list(CLASSIFIERS)}, got '{classifier}'")
        self.parameterization = parameterization
        self.classifier_name = classifier
        self.vectorizer = TextVectorizer(parameterization, tokenizer)
        self.classifier = CLASSIFIERS[classifier]()

    def fit(self, texts, labels):
        X = self.vectorizer.fit_transform(texts)
        if isinstance(self.classifier, NaiveBayesClassifier):
            self.classifier.fit(X, labels, vocab_size=self.vectorizer.vocab_size)
        else:
            self.classifier.fit(X, labels)
        return self

    def predict(self, texts):
        return self.classifier.predict(self.vectorizer.transform(texts))

    def explain(self, text, top=3):
        """Top predictions for one text, as (label, score) pairs.

        Naive Bayes returns posterior probabilities; KNN returns the share of the
        k neighbours that voted for each label; Minimum Distance only gives the
        winning class.
        """
        vec = self.vectorizer.transform([text])[0]
        clf = self.classifier
        if isinstance(clf, NaiveBayesClassifier):
            return clf.predict_proba(vec)[:top]
        if isinstance(clf, KNNClassifier):
            votes = [label for _, label in clf.neighbours(vec)]
            counts = {}
            for label in votes:
                counts[label] = counts.get(label, 0) + 1
            return [(lab, n / len(votes)) for lab, n in sorted(counts.items(), key=lambda kv: -kv[1])][:top]
        return [(clf.predict([vec])[0], None)]

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        # Only load models you trained yourself: pickle files can run code.
        with open(path, "rb") as f:
            return pickle.load(f)
