import sys
import os
import pickle
from src.data_loader import load_reuters_dataset
from src.vectorizers import TextVectorizer
from src.classifiers.naive_bayes import NaiveBayesClassifier
from src.classifiers.min_distance import MinimumDistanceClassifier
from src.classifiers.knn import KNNClassifier


def calculate_accuracy(y_true, y_pred):
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return (correct / len(y_true)) * 100

def main():
    # --- MODE 2: EXECUTION WITH ONE PARAMETER (GUI) ---
    if len(sys.argv) == 2:
        model_name = sys.argv[1]
        print(f"Loading model '{model_name}' and opening graphical interface...")
        
        try:
            import gui
            gui.start_gui(model_name)
        except ImportError:
            print("Error: The file gui.py or the function start_gui could not be found.")
        return

    # --- MODE 1: TRAINING (5 PARAMETERS) ---
    # Parameters: training_set, testing_set, parameterization, classification, model_name
    if len(sys.argv) == 6:
        train_path = sys.argv[1]
        test_path = sys.argv[2] # In Reuters it is usually the same folder with *.sgm
        param_algo = sys.argv[3] # 'binary', 'tf', 'tfidf'
        class_algo = sys.argv[4] # 'min_dist', 'naive_bayes', etc.
        model_name = sys.argv[5]

        print(f"--- Starting training process ---")
        
        # 1. Load data
        print("1. Loading Reuters-21578 corpus...")
        X_train_raw, y_train, X_test_raw, y_test = load_reuters_dataset(train_path)

        # 2. Parameterization (Vectorization)
        print(f"2. Applying parameterization: {param_algo}...")
        vectorizer = TextVectorizer(method=param_algo)
        X_train = vectorizer.fit_transform(X_train_raw)
        X_test = vectorizer.transform(X_test_raw)

        # 3. Classification
        print(f"3. Training classifier: {class_algo}...")
        
        classifier = None
        if class_algo == "min_dist":
            classifier = MinimumDistanceClassifier()
        elif class_algo == "naive_bayes":
             classifier = NaiveBayesClassifier()
        elif class_algo == "knn":
            classifier = KNNClassifier(k=5)
        else:
            print(f"Error: The algorithm '{class_algo}' is not implemented.")
            return

        classifier.fit(X_train, y_train)

        # 4. Evaluation
        print("4. Evaluating accuracy...")
        y_pred = classifier.predict(X_test)
        acc = calculate_accuracy(y_test, y_pred)
        print(f"RESULT: Accuracy = {acc:.2f}%")
        

        # 5. Save model
        print(f"5. Saving model to models/{model_name}.pkl...")
        if not os.path.exists('models'):
            os.makedirs('models')
            
        model_data = {
            'vectorizer': vectorizer,
            'classifier': classifier,
            'param_algo': param_algo,
            'class_algo': class_algo
        }
        
        with open(f"models/{model_name}.pkl", "wb") as f:
            pickle.dump(model_data, f)
            
        print("Training finished successfully.")
        
    else:
        print("\nProgram usage:")
        print("To train: python main.py <train_path> <test_path> <parameterization> <classification> <model_name>")
        print("For GUI:    python main.py <model_name>")

if __name__ == "__main__":
    main()