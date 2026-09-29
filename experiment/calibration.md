# Removing the shared direction (anisotropy) - results

73 questions, same 5-fold CV protocol as evaluate.py.

| method | doc #1 | AUC | gap (min pos − max neg) | answerable ok | unanswerable ok | overall (CV) |
|---|---|---|---|---|---|---|
| raw cosine (current) | 100.0% | 0.973 | -0.038 | 91.2% | 93.8% | **91.8%** |
| centered | 98.2% | 0.977 | -0.088 | 89.5% | 87.5% | **89.0%** |
| centered + drop 1 PC | 94.7% | 0.962 | -0.106 | 87.7% | 87.5% | **87.7%** |
| centered + drop 3 PCs | 96.5% | 0.938 | -0.094 | 75.4% | 81.2% | **76.7%** |
| whitened k=64 | 86.0% | 0.922 | -0.114 | 68.4% | 87.5% | **72.6%** |
| whitened k=128 | 82.5% | 0.821 | -0.143 | 59.6% | 68.8% | **61.6%** |
| BM25 only (lexical) | 93.0% | 0.888 | -9.802 | 70.2% | 81.2% | **72.6%** |
| raw cosine (current) + BM25 (logistic) | 100.0% | 0.973 | -0.038 | 84.2% | 93.8% | **86.3%** |
| centered + BM25 (logistic) | 98.2% | 0.977 | -0.088 | 89.5% | 87.5% | **89.0%** |

## Scores per question type - `centered` vs raw

| group | raw min / median / max | treated min / median / max |
|---|---|---|
| answerable | 0.782 / 0.843 / 0.899 | 0.161 / 0.348 / 0.577 |
| easy negatives | 0.736 / 0.773 / 0.788 | 0.107 / 0.112 / 0.203 |
| hard negatives | 0.774 / 0.801 / 0.819 | 0.119 / 0.186 / 0.248 |
