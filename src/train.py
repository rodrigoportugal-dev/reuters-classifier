from sklearn.metrics import accuracy_score
import joblib

def train_model(X_train, y_train, X_test, y_test, vectorizer, classifier, model_name):
    X_train_vec = vectorizer.fit_transform(X_train).toarray()
    X_test_vec = vectorizer.transform(X_test).toarray()

    classifier.fit(X_train_vec, y_train)
    preds = classifier.predict(X_test_vec)

    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc}")

    joblib.dump((vectorizer, classifier), f"models/{model_name}.pkl")