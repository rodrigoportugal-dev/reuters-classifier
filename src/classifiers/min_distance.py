from collections import defaultdict

class MinimumDistanceClassifier:
    def __init__(self):
        self.classes = []
        self.centroids = {}
        # We save the sum of the squared centroid values for the speed trick
        self.centroid_sq_norms = {} 

    def fit(self, X, y):
        """
        Calculates the centroid (mean vector) for each class.
        """
        print("    -> Calculating centroids (Minimum Distance training)...")
        self.classes = list(set(y))
        vocab_size = len(X[0]) if len(X) > 0 else 0
        
        # Dictionaries to sum the vectors and count how many documents there are per class
        class_sums = defaultdict(lambda: [0.0] * vocab_size)
        class_counts = defaultdict(int)
        
        # 1. Sum all vectors of the same class
        for features, label in zip(X, y):
            class_counts[label] += 1
            for i, val in enumerate(features):
                if val > 0:
                    class_sums[label][i] += val
                    
        # 2. Divide by the total to get the mean (the centroid)
        for c in self.classes:
            centroid = [0.0] * vocab_size
            sq_norm = 0.0
            
            for i in range(vocab_size):
                centroid[i] = class_sums[c][i] / class_counts[c]
                # Precalculate the square of each centroid component
                sq_norm += centroid[i] ** 2
            
            self.centroids[c] = centroid
            self.centroid_sq_norms[c] = sq_norm

    def predict(self, X):
        """
        Assigns each vector to the class of the nearest centroid.
        """
        predictions = []
        total_docs = len(X)
        
        for doc_index, features in enumerate(X):
            if (doc_index + 1) % 500 == 0 or doc_index == 0:
                print(f"    -> Predicting document {doc_index + 1}/{total_docs}...")
                
            best_class = None
            min_distance_score = float('inf')
            
            
            active_features = [(i, val) for i, val in enumerate(features) if val > 0]
            
            for c in self.classes:
                
                dot_product = 0.0
                for i, val in active_features:
                    dot_product += val * self.centroids[c][i]
                    
                distance_score = self.centroid_sq_norms[c] - 2 * dot_product
                
                if distance_score < min_distance_score:
                    min_distance_score = distance_score
                    best_class = c
                    
            predictions.append(best_class)
            
        return predictions