import sys
import warnings 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.metrics import accuracy_score
from src.data_loader import load_reuters_dataset

def main():
  
    warnings.filterwarnings("ignore", category=UserWarning)

    if len(sys.argv) != 2:
        print("Usage: python bonus_comparison.py <data_path>")
        return

    data_path = sys.argv[1]

    print("--- SCIKIT-LEARN BONUS COMPARISON ---")
    
    # 1. Load data using YOUR function
    print("1. Loading Reuters dataset...")
    X_train_raw, y_train, X_test_raw, y_test = load_reuters_dataset(data_path)
    
    # 2. Vectorization (TF-IDF)
    print("2. Vectorizing text using scikit-learn's TfidfVectorizer...")
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)

    # 3. Classifiers
    print("\n3. Training and evaluating scikit-learn models...\n")

    # A. Naive Bayes
    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train, y_train)
    nb_pred = nb_classifier.predict(X_test)
    nb_acc = accuracy_score(y_test, nb_pred) * 100
    print(f"-> Scikit-learn Naive Bayes Accuracy: {nb_acc:.2f}%")

    # B. Minimum Distance (Nearest Centroid)
    md_classifier = NearestCentroid()
    md_classifier.fit(X_train, y_train)
    md_pred = md_classifier.predict(X_test)
    md_acc = accuracy_score(y_test, md_pred) * 100
    print(f"-> Scikit-learn Minimum Distance (Nearest Centroid) Accuracy: {md_acc:.2f}%")

    # C. KNN
    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    knn_classifier.fit(X_train, y_train)
    knn_pred = knn_classifier.predict(X_test)
    knn_acc = accuracy_score(y_test, knn_pred) * 100
    print(f"-> Scikit-learn KNN (k=5) Accuracy: {knn_acc:.2f}%")

if __name__ == "__main__":
    main()