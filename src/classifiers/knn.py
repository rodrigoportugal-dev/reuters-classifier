import math
from collections import Counter

class KNNClassifier:
    def __init__(self, k=5):
        """
        Initializes the KNN classifier.
        :param k: Number of nearest neighbors to consider.
        """
        self.k = k
        self.X_train = []
        self.y_train = []
        self.train_norms = []

    def fit(self, X, y):
        """
        In KNN, the training phase simply "memorizes" the data.
        """
        print(f"    -> Memorizing data (KNN training with k={self.k})...")
        self.y_train = y
        self.X_train = []
        self.train_norms = []
        
        for features in X:
            active_features = {i: val for i, val in enumerate(features) if val > 0}
            self.X_train.append(active_features)
            
            # Precalculate the magnitude of the vector (needed for Cosine Similarity)
            norm = math.sqrt(sum(val**2 for val in active_features.values()))
            self.train_norms.append(norm)

    def predict(self, X):
        """
        Predicts the class by finding the K most similar documents.
        """
        predictions = []
        total_docs = len(X)
        
        for doc_index, features in enumerate(X):
            if (doc_index + 1) % 500 == 0 or doc_index == 0:
                print(f"    -> Predicting document {doc_index + 1}/{total_docs}...")
                
            active_features = {i: val for i, val in enumerate(features) if val > 0}
            test_norm = math.sqrt(sum(val**2 for val in active_features.values()))
            
            similarities = []
            
            # We compare the Test document with ALL Training documents
            for train_idx, train_doc in enumerate(self.X_train):
                train_norm = self.train_norms[train_idx]
                
                # If any document is empty, similarity is 0
                if test_norm == 0 or train_norm == 0:
                    sim = 0.0
                else:
                    # Calculate the dot product super fast (only where words match)
                    dot_product = 0.0
                    # Iterate over the smaller dictionary to go even faster
                    if len(active_features) < len(train_doc):
                        for i, val in active_features.items():
                            if i in train_doc:
                                dot_product += val * train_doc[i]
                    else:
                        for i, val in train_doc.items():
                            if i in active_features:
                                dot_product += val * active_features[i]
                                
                    # Cosine Similarity formula
                    sim = dot_product / (test_norm * train_norm) 
                    
                similarities.append((sim, self.y_train[train_idx]))
            
            # Sort the results from highest to lowest similarity and take the top 'K'
            similarities.sort(key=lambda x: x[0], reverse=True)
            top_k = similarities[:self.k]
            
            # Extract the labels of those K neighbors and count the votes
            votes = [label for sim, label in top_k]
            if votes:
                # The most common wins
                best_class = Counter(votes).most_common(1)[0][0]
            else:
                best_class = None
                
            predictions.append(best_class)
            
        return predictions