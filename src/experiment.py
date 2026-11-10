from src.vectorizers import get_vectorizer
from src.utils import get_classifier
from src.train import train_model
from src.data_loader import load_data

vectorizers = ["bow", "tfidf", "ngrams"]
classifiers = ["nb", "knn", "min_distance"]

def run_experiments():
    X_train, y_train = load_data("data/train")
    X_test, y_test = load_data("data/test")

    for v in vectorizers:
        for c in classifiers:
            print(f"\n=== {v} + {c} ===")

            vectorizer = get_vectorizer(v)
            classifier = get_classifier(c)

            train_model(
                X_train, y_train,
                X_test, y_test,
                vectorizer,
                classifier,
                f"{v}_{c}"
            )