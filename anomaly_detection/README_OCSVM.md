# OC-SVM Implementation for Text Difficulty Anomaly Detection

## Overview

This implementation uses **One-Class SVM (OC-SVM)** instead of Isolation Forest to distinguish between easy texts (Simple Wikipedia) and hard texts (Normal Wikipedia).

Based on the **AD-NLP paper** (Bejan et al., 2023), OC-SVM showed competitive performance (AUROC 64.9-84.9%) for semantic anomaly detection.

## Key Differences from Isolation Forest

| Aspect | Isolation Forest | OC-SVM |
|--------|------------------|--------|
| **Algorithm** | Tree-based (random paths) | Support Vector Machine (margin-based) |
| **Decision** | Path length in trees | Distance from hyperplane boundary |
| **Kernel** | N/A | RBF / Polynomial / Linear |
| **Feature Normalization** | Not required | **CRITICAL - Required!** |
| **Hyperparameters** | `n_estimators`, `contamination` | `kernel`, `nu`, `gamma` |
| **Training Speed** | Fast | Slower (especially with many features) |
| **Memory** | Low | Higher (stores support vectors) |

## Critical: Feature Normalization

⚠️ **OC-SVM REQUIRES StandardScaler normalization!**

```python
# ❌ WRONG - Will fail!
ocsvm.fit(X_train)

# ✅ CORRECT
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
ocsvm.fit(X_train_scaled)
```

**Why?** OC-SVM uses distance-based decision boundaries. Features with different scales will dominate the distance calculation:
- BERT embeddings: [-1, +1]
- `perplexity`: [1, 1000]
- `sentence_length`: [5, 50]

Without normalization, perplexity dominates everything else!

## Hyperparameters (from AD-NLP Paper)

### Nu (`nu`)
Upper bound on fraction of outliers, lower bound on fraction of support vectors.

- `nu=0.05`: Very tight boundary (expect 5% outliers)
- `nu=0.1`: Tight boundary (expect 10% outliers) ← **Good starting point**
- `nu=0.2`: Moderate boundary (expect 20% outliers)
- `nu=0.5`: Loose boundary (expect 50% outliers)

### Kernel
- `'rbf'`: Radial Basis Function (default, works best for most cases) ← **Recommended**
- `'poly'`: Polynomial kernel
- `'linear'`: Linear kernel (faster, good if data is linearly separable)

### Gamma (for RBF/Poly kernels)
- `'scale'`: 1 / (n_features * X.var()) ← **Recommended**
- `'auto'`: 1 / n_features

## Usage

### Quick Start

```bash
# Run with BERT embeddings (recommended)
python main_ocsvm.py

# Quick test (no hyperparameter tuning)
python main_ocsvm.py --quick

# Force reload data and embeddings
python main_ocsvm.py --reload-data --reload-embeddings
```

### Configuration

Edit `config.py` to switch embedding type:

```python
# BERT embeddings (RECOMMENDED - best performance)
EMBEDDING_TYPE = "bert"
BERT_MODEL = "all-MiniLM-L6-v2"  # 384-dim, fast

# OR GloVe embeddings (faster, but lower performance)
EMBEDDING_TYPE = "glove"
GLOVE_DIM = 300
```

## Expected Results

Based on AD-NLP paper performance on semantic anomaly detection:

### With BERT Embeddings (Recommended)
- **AUROC**: 70-85% ✅
- **F1 Score**: 70-80%
- **Why?** BERT captures syntax, context, and complexity

### With GloVe Embeddings
- **AUROC**: 60-70% ⚠️
- **F1 Score**: 60-70%
- **Why?** GloVe only captures semantic similarity, not difficulty

### With Hand-Crafted Features (6D)
- **AUROC**: 65-75%
- **F1 Score**: 65-75%
- **Why?** Direct difficulty metrics, but low-dimensional

## Files Overview

```
anomaly_detection/
├── model_training_ocsvm.py          # OC-SVM training with StandardScaler
├── main_ocsvm.py                    # Main pipeline for OC-SVM
├── evaluation.py                    # Updated to support both IsoForest and OC-SVM
├── config.py                        # Added OCSVM_PARAM_GRID
├── README_OCSVM.md                  # This file
└── outputs/
    ├── best_ocsvm.pkl               # Trained OC-SVM model
    ├── best_ocsvm_scaler.pkl        # StandardScaler (IMPORTANT!)
    ├── evaluation_results_ocsvm.json
    └── hyperparameter_tuning_results_ocsvm.json
```

## Comparison with Isolation Forest

After running both, compare results:

```bash
# Run Isolation Forest
python main.py

# Run OC-SVM
python main_ocsvm.py

# Results are automatically compared at the end
```

Example output:
```
COMPARISON: OC-SVM vs Isolation Forest
AUROC:              0.4815    0.7234      +0.2419  ✅
F1 Score:           0.9031    0.7856      -0.1175
Score Separation:   0.0023    0.1245      +0.1222  ✅
```

## Why OC-SVM Might Outperform Isolation Forest

1. **Non-linear decision boundaries**: RBF kernel can capture complex patterns
2. **Margin maximization**: OC-SVM finds the best boundary, not random trees
3. **Kernel trick**: Can implicitly work in high-dimensional space
4. **Robust to outliers in training**: Nu parameter controls sensitivity

## Troubleshooting

### Low AUROC (<60%)

**Likely cause**: Using GloVe embeddings

**Solution**: Switch to BERT embeddings in `config.py`:
```python
EMBEDDING_TYPE = "bert"
```

### Training very slow

**Likely cause**: Too many support vectors (high `nu` value)

**Solution**: Reduce `nu` to 0.05 or 0.1, or use linear kernel:
```python
OCSVM_DEFAULT_PARAMS = {
    'nu': 0.05,
    'kernel': 'linear',  # Much faster than RBF
    'gamma': 'scale'
}
```

### Memory error

**Likely cause**: Large dataset + RBF kernel

**Solution**: Use linear kernel or subsample training data

### Poor separation (inlier_mean ≈ outlier_mean)

**Likely cause**: Features don't capture difficulty well

**Solution**:
1. Switch to BERT embeddings
2. Or add hand-crafted difficulty features (length, complexity, perplexity)

## Implementation Notes

### Training Setup
- **Train**: 80% of simple texts (easy, label=0) ← **All inliers**
- **Val**: 10% of simple texts (easy, label=0) ← **All inliers**
- **Test**: 10% simple + 100% normal (label=0 + label=1) ← **Mix**

OC-SVM learns: "What does easy text look like?"
At test time: Hard texts are detected as anomalies (deviations from easy boundary)

### Hyperparameter Tuning
Uses 20% of normal texts + val set for tuning to avoid data leakage.

## References

- **AD-NLP Paper**: Bejan et al. (2023), "AD-NLP: A Benchmark for Anomaly Detection in Natural Language Processing"
- **OC-SVM Original**: Schölkopf et al. (1999), "Support Vector Method for Novelty Detection"

## Questions?

Check the implementation in:
- `model_training_ocsvm.py` - Training logic
- `main_ocsvm.py` - Full pipeline
- `config.py` - Hyperparameters

Happy anomaly detecting! 🎯
