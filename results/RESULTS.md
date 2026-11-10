# Results

Reuters-21578, LEWISSPLIT train/test, first topic only: **7733 training and 3008 test documents, 80 classes**. The two largest classes (`earn`, `acq`) are 60% of the test set, so macro F1 is reported next to accuracy.

## Improved preprocessing (`clean` tokenizer)

**Accuracy**

| Classifier | Binary | TF | TF-IDF |
|---|---:|---:|---:|
| Minimum Distance | 80.22% | 70.55% | 77.79% |
| Naive Bayes | 78.49% | 82.15% | 83.94% |
| KNN (k=5) | 82.98% | 82.15% | 77.53% |

**Macro F1**

| Classifier | Binary | TF | TF-IDF |
|---|---:|---:|---:|
| Minimum Distance | 0.389 | 0.373 | 0.448 |
| Naive Bayes | 0.160 | 0.253 | 0.456 |
| KNN (k=5) | 0.403 | 0.447 | 0.510 |

## Original preprocessing (`basic` tokenizer, as submitted)

**Accuracy**

| Classifier | Binary | TF | TF-IDF |
|---|---:|---:|---:|
| Minimum Distance | 71.74% | 50.33% | 76.06% |
| Naive Bayes | 75.20% | 77.76% | 82.91% |
| KNN (k=5) | 82.45% | 75.43% | 78.96% |

**Macro F1**

| Classifier | Binary | TF | TF-IDF |
|---|---:|---:|---:|
| Minimum Distance | 0.352 | 0.203 | 0.435 |
| Naive Bayes | 0.091 | 0.124 | 0.411 |
| KNN (k=5) | 0.391 | 0.377 | 0.508 |

## scikit-learn, same data and tokenizer

**Accuracy**

| Classifier | Binary | TF | TF-IDF |
|---|---:|---:|---:|
| Minimum Distance | 80.22% | 70.55% | 76.46% |
| Naive Bayes | 78.49% | 82.15% | 84.57% |
| KNN (k=5) | 82.28% | 81.52% | 79.39% |

**Macro F1**

| Classifier | Binary | TF | TF-IDF |
|---|---:|---:|---:|
| Minimum Distance | 0.389 | 0.373 | 0.443 |
| Naive Bayes | 0.160 | 0.253 | 0.459 |
| KNN (k=5) | 0.329 | 0.412 | 0.492 |

## Best model: Naive Bayes + TFIDF

Accuracy 83.94%, macro F1 0.456, saved model 1.7 MB.

Confusion matrix for the 8 largest classes (rows = true, columns = predicted):

| true \ pred | earn | acq | crude | money-fx | grain | trade | interest | ship | other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **earn** | 1054 | 16 | 2 | · | · | 2 | 2 | · | 1 |
| **acq** | 46 | 627 | 6 | 2 | 2 | 4 | 1 | · | 26 |
| **crude** | · | 1 | 139 | · | · | 1 | · | 12 | 5 |
| **money-fx** | · | · | 1 | 125 | · | 4 | 12 | · | 10 |
| **grain** | · | · | · | 1 | 108 | 5 | · | 6 | 6 |
| **trade** | · | · | 2 | 6 | 3 | 77 | · | · | 16 |
| **interest** | · | · | · | 38 | · | 2 | 57 | · | 3 |
| **ship** | · | · | 12 | · | · | 6 | · | 34 | 2 |

Per-class F1 for the same classes:

| class | support | precision | recall | F1 |
|---|---:|---:|---:|---:|
| earn | 1077 | 0.96 | 0.98 | 0.97 |
| acq | 714 | 0.96 | 0.88 | 0.92 |
| crude | 158 | 0.76 | 0.88 | 0.82 |
| money-fx | 152 | 0.60 | 0.82 | 0.69 |
| grain | 126 | 0.72 | 0.86 | 0.78 |
| trade | 104 | 0.61 | 0.74 | 0.67 |
| interest | 100 | 0.72 | 0.57 | 0.64 |
| ship | 54 | 0.60 | 0.63 | 0.61 |
