"""K-Nearest Neighbours with cosine similarity, using an inverted index.

Instead of comparing a test document with all ~7,000 training documents, an
inverted index (word -> documents that contain it) only visits the training
documents that share at least one word with it. The result is the same as the
brute-force version, but much faster on sparse text.
"""

import heapq
import math
from collections import Counter, defaultdict


def _normalize(vec):
    norm = math.sqrt(sum(v * v for v in vec.values()))
    return {i: v / norm for i, v in vec.items()} if norm else {}


class KNNClassifier:
    def __init__(self, k=5):
        self.k = k
        self.labels = []
        self.index = {}  # feature -> [(doc_id, normalized value)]

    def fit(self, X, y):
        self.labels = list(y)
        index = defaultdict(list)
        for doc_id, vec in enumerate(X):
            for i, v in _normalize(vec).items():
                index[i].append((doc_id, v))
        self.index = dict(index)
        return self

    def neighbours(self, vec):
        """The k most similar training documents as (similarity, label), best first."""
        sims = defaultdict(float)
        for i, v in _normalize(vec).items():
            for doc_id, w in self.index.get(i, ()):
                sims[doc_id] += v * w
        top = heapq.nlargest(self.k, sims.items(), key=lambda kv: kv[1])
        return [(s, self.labels[d]) for d, s in top]

    def predict(self, X):
        out = []
        for vec in X:
            votes = [label for _, label in self.neighbours(vec)]
            # Ties go to the label of the closest neighbour (Counter keeps insertion order).
            out.append(Counter(votes).most_common(1)[0][0] if votes else None)
        return out
