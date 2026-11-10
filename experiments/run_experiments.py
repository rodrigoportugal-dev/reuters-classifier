"""Run every parameterization x classifier combination and write results/RESULTS.md.

    python -m experiments.run_experiments [--skip-sklearn]

It evaluates both tokenizers ("basic" = original submission, "clean" = improved)
so the effect of preprocessing is visible, and compares with scikit-learn.
"""

import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

from src.data_loader import load_reuters_dataset
from src.metrics import accuracy, confusion_matrix, macro_f1, per_class_report
from src.model import CLASSIFIERS, PARAMETERIZATIONS, TextClassificationModel

DATA = "data/reuters21578"
OUT = Path("results")
NAMES = {"min_dist": "Minimum Distance", "naive_bayes": "Naive Bayes", "knn": "KNN (k=5)"}


def table(title, rows, value):
    lines = [f"**{title}**", "", "| Classifier | Binary | TF | TF-IDF |", "|---|---:|---:|---:|"]
    for clf in CLASSIFIERS:
        cells = [value(rows[(p, clf)]) for p in PARAMETERIZATIONS]
        lines.append(f"| {NAMES[clf]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(skip_sklearn=False):
    X_train, y_train, X_test, y_test = load_reuters_dataset(DATA)
    n_classes = len(set(y_train) | set(y_test))
    print(f"{len(X_train)} train / {len(X_test)} test documents, {n_classes} classes")

    results = {}
    best = None
    for tok in ("basic", "clean"):
        for param in PARAMETERIZATIONS:
            for clf in CLASSIFIERS:
                start = time.perf_counter()
                model = TextClassificationModel(param, clf, tokenizer=tok).fit(X_train, y_train)
                pred = model.predict(X_test)
                secs = time.perf_counter() - start
                r = {"accuracy": accuracy(y_test, pred), "macro_f1": macro_f1(y_test, pred), "seconds": secs}
                results[(tok, param, clf)] = r
                print(f"{tok:<5} {param:<6} {clf:<11} acc {r['accuracy']:.2%}  F1 {r['macro_f1']:.3f}  {secs:5.1f}s")
                if tok == "clean" and (best is None or r["accuracy"] > best[0]["accuracy"]):
                    best = (r, param, clf, model, pred)

    sk = {}
    if not skip_sklearn:
        import bonus_comparison
        sk = bonus_comparison.run(DATA)

    OUT.mkdir(exist_ok=True)
    pct = lambda r: f"{r['accuracy']:.2%}"
    f1 = lambda r: f"{r['macro_f1']:.3f}"
    pick = lambda tok: {(p, c): results[(tok, p, c)] for p in PARAMETERIZATIONS for c in CLASSIFIERS}

    best_r, best_p, best_c, best_model, best_pred = best
    size_mb = len(pickle.dumps(best_model)) / 1e6
    test_dist = Counter(y_test).most_common()
    top_share = (test_dist[0][1] + test_dist[1][1]) / len(y_test)

    md = [
        "# Results",
        "",
        f"Reuters-21578, LEWISSPLIT train/test, first topic only: **{len(X_train)} training "
        f"and {len(X_test)} test documents, {n_classes} classes**. "
        f"The two largest classes (`{test_dist[0][0]}`, `{test_dist[1][0]}`) are "
        f"{top_share:.0%} of the test set, so macro F1 is reported next to accuracy.",
        "",
        "## Improved preprocessing (`clean` tokenizer)",
        "",
        table("Accuracy", pick("clean"), pct),
        "",
        table("Macro F1", pick("clean"), f1),
        "",
        "## Original preprocessing (`basic` tokenizer, as submitted)",
        "",
        table("Accuracy", pick("basic"), pct),
        "",
        table("Macro F1", pick("basic"), f1),
    ]
    if sk:
        md += ["", "## scikit-learn, same data and tokenizer", "",
               table("Accuracy", {k: {"accuracy": v[0]} for k, v in sk.items()}, pct), "",
               table("Macro F1", {k: {"macro_f1": v[1]} for k, v in sk.items()}, f1)]

    labels = [lab for lab, _ in test_dist[:8]]
    cols, matrix = confusion_matrix(y_test, best_pred, labels)
    md += ["", f"## Best model: {NAMES[best_c]} + {best_p.upper()}", "",
           f"Accuracy {best_r['accuracy']:.2%}, macro F1 {best_r['macro_f1']:.3f}, "
           f"saved model {size_mb:.1f} MB.", "",
           "Confusion matrix for the 8 largest classes (rows = true, columns = predicted):", "",
           "| true \\ pred | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
    for lab, row in zip(labels, matrix):
        md.append(f"| **{lab}** | " + " | ".join(str(v) if v else "·" for v in row) + " |")

    report = per_class_report(y_test, best_pred)
    md += ["", "Per-class F1 for the same classes:", "", "| class | support | precision | recall | F1 |", "|---|---:|---:|---:|---:|"]
    for lab in labels:
        r = report[lab]
        md.append(f"| {lab} | {r['support']} | {r['precision']:.2f} | {r['recall']:.2f} | {r['f1']:.2f} |")

    (OUT / "RESULTS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (OUT / "results.json").write_text(json.dumps(
        {"ours": {"|".join(k): v for k, v in results.items()},
         "sklearn": {"|".join(k): {"accuracy": v[0], "macro_f1": v[1]} for k, v in sk.items()}},
        indent=2), encoding="utf-8")
    print(f"\nWritten {OUT / 'RESULTS.md'}")


if __name__ == "__main__":
    main(skip_sklearn="--skip-sklearn" in sys.argv)
