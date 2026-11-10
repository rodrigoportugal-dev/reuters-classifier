"""Command-line entry point, with the two modes required by the specification.

Train, evaluate and save a model (5 parameters):
    python main.py <train_path> <test_path> <binary|tf|tfidf> <min_dist|naive_bayes|knn> <model_name>

Open the GUI with a saved model (1 parameter):
    python main.py <model_name>
"""

import sys
import time

from src.data_loader import load_reuters_dataset
from src.metrics import accuracy, macro_f1
from src.model import CLASSIFIERS, PARAMETERIZATIONS, TextClassificationModel

USAGE = __doc__


def train(train_path, test_path, param, clf, model_name):
    if param not in PARAMETERIZATIONS:
        sys.exit(f"Unknown parameterization '{param}'. Use one of: {', '.join(PARAMETERIZATIONS)}")
    if clf not in CLASSIFIERS:
        sys.exit(f"Unknown classifier '{clf}'. Use one of: {', '.join(CLASSIFIERS)}")

    print("1. Loading Reuters-21578...")
    X_train, y_train, _, _ = load_reuters_dataset(train_path)
    _, _, X_test, y_test = load_reuters_dataset(test_path)
    print(f"   {len(X_train)} training documents, {len(X_test)} test documents")

    print(f"2. Training {clf} with {param} features...")
    start = time.perf_counter()
    model = TextClassificationModel(param, clf).fit(X_train, y_train)
    print(f"   Vocabulary: {model.vectorizer.vocab_size} words")

    print("3. Evaluating on the test set...")
    y_pred = model.predict(X_test)
    print(f"   Accuracy: {accuracy(y_test, y_pred):.2%}   Macro F1: {macro_f1(y_test, y_pred):.3f}")
    print(f"   ({time.perf_counter() - start:.1f} s)")

    path = f"models/{model_name}.pkl"
    model.save(path)
    print(f"4. Model saved to {path}")


def main(argv):
    if len(argv) == 6:
        train(*argv[1:])
    elif len(argv) == 2:
        import gui
        gui.start_gui(argv[1])
    else:
        print(USAGE)


if __name__ == "__main__":
    main(sys.argv)
