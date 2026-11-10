"""Desktop GUI (Tkinter) to classify text with a saved model."""

import os
import tkinter as tk

from src.model import TextClassificationModel

EXAMPLE = (
    "The technology giant announced today that it has agreed to acquire the smaller "
    "software startup for 500 million dollars in cash. The merger and takeover process "
    "is expected to close next month."
)


def start_gui(model_name):
    path = f"models/{model_name}.pkl"
    if not os.path.exists(path):
        print(f"Model not found: {path}. Train it first with main.py (see README).")
        return

    model = TextClassificationModel.load(path)

    root = tk.Tk()
    root.title(f"Reuters classifier · {model_name}")
    root.geometry("640x560")

    tk.Label(
        root,
        text=f"Features: {model.parameterization.upper()}  ·  Classifier: {model.classifier_name}",
        font=("Helvetica", 10, "italic"), fg="gray",
    ).pack(pady=(12, 4))
    tk.Label(root, text="Type or paste a news article in English:", font=("Helvetica", 12, "bold")).pack()

    text_input = tk.Text(root, height=14, width=70, font=("Helvetica", 10), wrap="word")
    text_input.insert("1.0", EXAMPLE)
    text_input.pack(pady=10, padx=12)

    result = tk.Label(root, text="", font=("Helvetica", 15, "bold"))
    detail = tk.Label(root, text="", font=("Helvetica", 10), fg="gray", justify="left")

    def classify():
        text = text_input.get("1.0", tk.END).strip()
        if not text:
            result.config(text="Type some text first.", fg="#b45309")
            detail.config(text="")
            return
        top = model.explain(text)
        result.config(text=f"Prediction: {top[0][0]}", fg="#15803d")
        lines = [f"{label}: {score:.0%}" for label, score in top if score is not None]
        detail.config(text="   ".join(lines))

    tk.Button(root, text="Classify", font=("Helvetica", 12, "bold"), command=classify).pack(pady=6)
    result.pack(pady=(10, 2))
    detail.pack()
    root.mainloop()
