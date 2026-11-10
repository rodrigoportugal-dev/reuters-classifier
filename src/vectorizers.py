"""Document parameterization: Binary, TF and TF-IDF.

Vectors are sparse: each document is a ``dict {feature_index: value}`` that only
stores the words it contains. The original version stored a dense list with one
entry per vocabulary word (~30,000), which made every saved model 40-55 MB.
"""

import math
from collections import Counter

from src.preprocessing import get_tokenizer

METHODS = ("binary", "tf", "tfidf")


class TextVectorizer:
    def __init__(self, method="tfidf", tokenizer="clean"):
        if method not in METHODS:
            raise ValueError(f"Method must be one of {METHODS}, got '{method}'")
        self.method = method
        self.tokenizer = tokenizer
        self.vocabulary = {}
        self.idf = {}

    @property
    def vocab_size(self):
        return len(self.vocabulary)

    def _tokens(self, text):
        return get_tokenizer(self.tokenizer)(text)

    def fit(self, raw_documents):
        doc_freq = Counter()
        for doc in raw_documents:
            doc_freq.update(set(self._tokens(doc)))

        self.vocabulary = {word: i for i, word in enumerate(sorted(doc_freq))}

        # IDF(t) = log(N / df(t)), indexed by feature number
        n_docs = len(raw_documents)
        self.idf = {self.vocabulary[w]: math.log(n_docs / df) for w, df in doc_freq.items()}
        return self

    def transform(self, raw_documents):
        vectors = []
        for doc in raw_documents:
            counts = Counter(self._tokens(doc))
            vec = {}
            for word, count in counts.items():
                idx = self.vocabulary.get(word)
                if idx is None:  # word not seen during training
                    continue
                if self.method == "binary":
                    value = 1.0
                elif self.method == "tf":
                    value = float(count)
                else:  # tfidf
                    value = count * self.idf[idx]
                if value > 0:
                    vec[idx] = value
            vectors.append(vec)
        return vectors

    def fit_transform(self, raw_documents):
        return self.fit(raw_documents).transform(raw_documents)
