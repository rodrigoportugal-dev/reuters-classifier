"""Minimum Distance (nearest centroid) classifier with Euclidean distance.

||x - mu||^2 = ||x||^2 - 2 x.mu + ||mu||^2, and ||x||^2 is the same for every
class, so ranking by ||mu||^2 - 2 x.mu gives the same answer without computing
the full distance. Only the words present in x take part in the dot product.
"""

from collections import Counter, defaultdict


class MinimumDistanceClassifier:
    def __init__(self):
        self.classes = []
        self.centroids = {}
        self.centroid_sq_norms = {}

    def fit(self, X, y):
        self.classes = sorted(set(y))
        sums = defaultdict(Counter)
        counts = Counter(y)
        for vec, label in zip(X, y):
            sums[label].update(vec)
        for c in self.classes:
            centroid = {i: v / counts[c] for i, v in sums[c].items()}
            self.centroids[c] = centroid
            self.centroid_sq_norms[c] = sum(v * v for v in centroid.values())
        return self

    def predict(self, X):
        out = []
        for vec in X:
            best, best_score = None, float("inf")
            for c in self.classes:
                mu = self.centroids[c]
                dot = sum(v * mu.get(i, 0.0) for i, v in vec.items())
                score = self.centroid_sq_norms[c] - 2 * dot
                if score < best_score:
                    best, best_score = c, score
            out.append(best)
        return out
