"""Evaluation metrics written from scratch (accuracy, per-class P/R/F1, macro F1).

Reuters-21578 is very imbalanced: "earn" and "acq" are about 60 % of the test
set, so accuracy alone hides how the classifier does on small classes. Macro F1
gives every class the same weight.
"""

from collections import Counter


def accuracy(y_true, y_pred):
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


def per_class_report(y_true, y_pred):
    """{label: {"precision", "recall", "f1", "support"}} for every label in y_true."""
    tp, fp, fn = Counter(), Counter(), Counter()
    for t, p in zip(y_true, y_pred):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1
    support = Counter(y_true)
    report = {}
    for label in support:
        prec = tp[label] / (tp[label] + fp[label]) if tp[label] + fp[label] else 0.0
        rec = tp[label] / (tp[label] + fn[label]) if tp[label] + fn[label] else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        report[label] = {"precision": prec, "recall": rec, "f1": f1, "support": support[label]}
    return report


def macro_f1(y_true, y_pred):
    report = per_class_report(y_true, y_pred)
    return sum(r["f1"] for r in report.values()) / len(report)


def confusion_matrix(y_true, y_pred, labels):
    """Rows = true label, columns = predicted label, restricted to `labels` (+ 'other')."""
    keep = set(labels)
    cols = list(labels) + ["other"]
    matrix = {t: Counter() for t in labels}
    for t, p in zip(y_true, y_pred):
        if t in keep:
            matrix[t][p if p in keep else "other"] += 1
    return cols, [[matrix[t][c] for c in cols] for t in labels]
