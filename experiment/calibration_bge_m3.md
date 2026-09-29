# Removing the shared direction (anisotropy) - results, `bge_m3`

73 questions, same 5-fold CV protocol as evaluate.py.

| method | doc #1 | AUC | gap (min pos − max neg) | answerable ok | unanswerable ok | overall (CV) |
|---|---|---|---|---|---|---|
| raw cosine (current) | 100.0% | 0.984 | -0.059 | 93.0% | 93.8% | **93.2%** |
| centered | 98.2% | 0.991 | -0.044 | 94.7% | 93.8% | **94.5%** |
| centered + drop 1 PC | 96.5% | 0.988 | -0.048 | 94.7% | 87.5% | **93.2%** |
| centered + drop 3 PCs | 91.2% | 0.929 | -0.123 | 73.7% | 81.2% | **75.3%** |
| whitened k=64 | 80.7% | 0.955 | -0.114 | 75.4% | 93.8% | **79.5%** |
| whitened k=128 | 80.7% | 0.863 | -0.105 | 61.4% | 81.2% | **65.8%** |
| BM25 only (lexical) | 93.0% | 0.888 | -9.802 | 70.2% | 81.2% | **72.6%** |
| raw cosine (current) + BM25 (logistic) | 100.0% | 0.984 | -0.059 | 91.2% | 93.8% | **91.8%** |
| centered + BM25 (logistic) | 98.2% | 0.991 | -0.044 | 91.2% | 93.8% | **91.8%** |

## Scores per question type - `centered` vs raw

| group | raw min / median / max | treated min / median / max |
|---|---|---|
| answerable | 0.452 / 0.617 / 0.752 | 0.213 / 0.409 / 0.609 |
| easy negatives | 0.332 / 0.383 / 0.463 | 0.105 / 0.191 / 0.216 |
| hard negatives | 0.375 / 0.463 / 0.510 | 0.114 / 0.204 / 0.257 |
