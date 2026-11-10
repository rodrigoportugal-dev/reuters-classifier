from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from src.classifiers.min_distance import MinimumDistanceClassifier

def get_classifier(name):
    if name == "nb":
        return MultinomialNB()
    elif name == "knn":
        return KNeighborsClassifier()
    elif name == "min_distance":
        return MinimumDistanceClassifier()
    else:
        raise ValueError("Unknown classifier")