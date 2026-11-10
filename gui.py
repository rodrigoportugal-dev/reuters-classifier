import tkinter as tk
from tkinter import messagebox
import pickle
import os

def start_gui(model_name):
    model_path = f"models/{model_name}.pkl"
    
    # 1. Check if the model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}.")
        print("Make sure you typed the name correctly or trained it first.")
        return

    # 2. Load the saved model (.pkl)
    print(f"Loading model '{model_name}'...")
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
        
    vectorizer = model_data['vectorizer']
    classifier = model_data['classifier']
    param_algo = model_data['param_algo']
    class_algo = model_data['class_algo']

    # 3. Configure the main window (Tkinter)
    root = tk.Tk()
    root.title(f"Reuters Classifier - Model: {model_name}")
    root.geometry("600x550")
    
    # Top information labels
    info_text = f"Vectorizer: {param_algo.upper()} | Classifier: {class_algo.upper()}"
    info_label = tk.Label(root, text=info_text, font=("Helvetica", 10, "italic"), fg="gray")
    info_label.pack(pady=10)

    instruction_label = tk.Label(root, text="Type or paste a news article in English:", font=("Helvetica", 12, "bold"))
    instruction_label.pack(pady=5)

    # Text box for the user to type
    text_input = tk.Text(root, height=15, width=65, font=("Helvetica", 10))
    text_input.pack(pady=10)

    # Label where the result will be displayed
    result_label = tk.Label(root, text="Prediction: ---", font=("Helvetica", 16, "bold"), fg="blue")

    # Function executed when the button is clicked
    def classify_text():
        user_text = text_input.get("1.0", tk.END).strip()
        
        if not user_text:
            messagebox.showwarning("Empty text", "Please enter some text to classify.")
            return
            
        try:
            # A. Convert text to numbers using the learned vocabulary
            # (We put it in a list because transform expects a list of documents)
            vectorized_text = vectorizer.transform([user_text])
            
            # B. Predict using the trained classifier
            prediction = classifier.predict(vectorized_text)
            predicted_class = prediction[0]
            
            # C. Show the result in the interface
            result_label.config(text=f"Prediction: {predicted_class}", fg="green")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while predicting:\n{str(e)}")

    # Classification button
    classify_button = tk.Button(root, text="Classify News", font=("Helvetica", 12, "bold"), bg="black", fg="white", command=classify_text)
    classify_button.pack(pady=15)

    result_label.pack(pady=10)

    # Start the application
    root.mainloop()