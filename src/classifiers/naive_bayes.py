"""Multinomial Naive Bayes with Laplace smoothing, on sparse vectors."""

import math
from collections import Counter, defaultdict


class NaiveBayesClassifier:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Laplace smoothing (+1)
        self.classes = []
        self.log_priors = {}
        self.log_likelihood = {}  # class -> {feature: log P(x_i | C)}
        self.log_unseen = {}      # class -> log P(x_i | C) for features never seen in C

    def fit(self, X, y, vocab_size=None):
        self.classes = sorted(set(y))
        vocab = set()
        feature_sums = defaultdict(Counter)
        for vec, label in zip(X, y):
            feature_sums[label].update(vec)
            vocab.update(vec)
        if vocab_size is None:
            vocab_size = len(vocab)

        class_counts = Counter(y)
        for c in self.classes:
            self.log_priors[c] = math.log(class_counts[c] / len(y))
            denominator = sum(feature_sums[c].values()) + self.alpha * vocab_size
            self.log_likelihood[c] = {
                i: math.log((value + self.alpha) / denominator)
                for i, value in feature_sums[c].items()
            }
            self.log_unseen[c] = math.log(self.alpha / denominator)
        return self

    def decision_scores(self, vec):
        """log P(C) + sum_i x_i * log P(x_i | C) for every class."""
        scores = {}
        for c in self.classes:
            ll, unseen = self.log_likelihood[c], self.log_unseen[c]
            scores[c] = self.log_priors[c] + sum(v * ll.get(i, unseen) for i, v in vec.items())
        return scores

    def predict_proba(self, vec):
        """Posterior probabilities (softmax of the log scores), highest first."""
        scores = self.decision_scores(vec)
        top = max(scores.values())
        exp = {c: math.exp(s - top) for c, s in scores.items()}
        total = sum(exp.values())
        return sorted(((c, e / total) for c, e in exp.items()), key=lambda p: p[1], reverse=True)

    def predict(self, X):
        out = []
        for vec in X:
            scores = self.decision_scores(vec)
            out.append(max(scores, key=scores.get))
        return out
