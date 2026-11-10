import math
from collections import defaultdict

class NaiveBayesClassifier:
    def __init__(self):
        self.classes = []
        self.class_priors = {}
        # Dictionary to store P(x_i | C)
        self.feature_probs = defaultdict(dict)
        self.vocab_size = 0

    def fit(self, X, y):
        """
        Trains the model by calculating prior and conditional probabilities.
        :param X: List of numeric vectors (training)
        :param y: List of labels
        """
        print("    -> Calculating probabilities (Naive Bayes training)...")
        self.classes = list(set(y))
        total_docs = len(y)
        self.vocab_size = len(X[0]) if len(X) > 0 else 0
        
        class_doc_counts = defaultdict(int)
        feature_counts = defaultdict(lambda: [0.0] * self.vocab_size)
        total_features_per_class = defaultdict(float)

        # 1. Count frequencies
        for features, label in zip(X, y):
            class_doc_counts[label] += 1
            for i, value in enumerate(features):
                if value > 0: # Small optimization: only add if the value is greater than 0
                    feature_counts[label][i] += value
                    total_features_per_class[label] += value

        # 2. Calculate probabilities with Laplace Smoothing (+1)
        for c in self.classes:
            # P(C): Prior probability of the class
            self.class_priors[c] = class_doc_counts[c] / total_docs
            
            # P(x_i|C): Conditional probability of each feature
            denominator = total_features_per_class[c] + self.vocab_size # + vocab_size for smoothing
            for i in range(self.vocab_size):
                numerator = feature_counts[c][i] + 1.0 # Laplace smoothing
                self.feature_probs[c][i] = numerator / denominator

    def predict(self, X):
        """
        Predicts the class for a list of new vectors (OPTIMIZED VERSION).
        :param X: List of numeric vectors (test)
        """
        predictions = []
        total_docs = len(X)
        
        for doc_index, features in enumerate(X):
            # Print progress every 500 documents to show it's advancing
            if (doc_index + 1) % 500 == 0 or doc_index == 0:
                print(f"    -> Predicting document {doc_index + 1}/{total_docs}...")

            best_class = None
            max_log_prob = float('-inf')

            active_features = [(i, value) for i, value in enumerate(features) if value > 0]

            # Evaluate the probability for each possible class
            for c in self.classes:
                # Start with the log of the prior probability: log P(C)
                log_prob = math.log(self.class_priors[c])
                
                # Add the log of the conditional probabilities only for existing words
                for i, value in active_features:
                    log_prob += value * math.log(self.feature_probs[c][i])
                
                # Keep the class that yields the highest probability
                if log_prob > max_log_prob:
                    max_log_prob = log_prob
                    best_class = c
            
            predictions.append(best_class)
            
        return predictions