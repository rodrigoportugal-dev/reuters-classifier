import joblib

def predict_text(model_path, text):
    vectorizer, classifier = joblib.load(model_path)
    X = vectorizer.transform([text]).toarray()
    return classifier.predict(X)[0]