# Anomaly Detection: Isolation Forest + GloVe

Text difficulty classification using Isolation Forest on GloVe embeddings for SimpleWikipedia dataset.

## Overview

This pipeline implements anomaly detection to distinguish between simple and normal (difficult) texts using:
- **GloVe embeddings** (300-dimensional, from HuggingFace)
- **Isolation Forest** (sklearn implementation)
- **Mean pooling** for sentence-level embeddings
- **OOV strategy**: Skip out-of-vocabulary words

## Project Structure

```
anomaly_detection/
├── config.py                  # Configuration parameters
├── data_preparation.py        # Data loading and splitting
├── embedding_extraction.py    # GloVe loading and text-to-embedding
├── model_training.py          # Isolation Forest training
├── evaluation.py              # Evaluation metrics and plots
├── main.py                    # Main pipeline orchestrator
├── requirements.txt           # Python dependencies
├── run_pipeline.sh            # SLURM job script (HPC)
└── outputs/                   # Generated outputs
    ├── data_splits.pkl
    ├── sentence_embeddings.pkl
    ├── best_isolation_forest.pkl
    ├── evaluation_results.json
    ├── hyperparameter_tuning_results.json
    └── *.png                  # Plots
```

## Installation

```bash
cd /home/user/lab/anomaly_detection
pip install -r requirements.txt
```

## Usage

### Quick Start (Local)

```bash
# Full pipeline with hyperparameter tuning
python main.py

# Quick test (skip tuning and plots)
python main.py --quick

# Force reload data and embeddings
python main.py --reload-data --reload-embeddings

# Skip hyperparameter tuning
python main.py --no-tuning

# Skip plot generation
python main.py --no-plots
```

### HPC Cluster (SLURM)

```bash
# Submit job to SLURM
sbatch run_pipeline.sh

# Check job status
squeue -u $USER

# View output
tail -f slurm-<job_id>.out
```

## Configuration

Edit `config.py` to modify:
- **Data paths**: `DATA_DIR`, `OUTPUT_DIR`, `EMBEDDINGS_DIR`
- **Data splits**: `TRAIN_RATIO`, `VAL_RATIO`, `TEST_RATIO`
- **GloVe settings**: `GLOVE_DIM`, `GLOVE_HF_DATASET`
- **Preprocessing**: `OOV_STRATEGY`, `SPECIAL_TOKENS`
- **Hyperparameters**: `IFOREST_PARAM_GRID`, `IFOREST_DEFAULT_PARAMS`
- **Evaluation**: Metrics to compute, plot settings

## Pipeline Steps

### 1. Data Preparation (`data_preparation.py`)
- Loads SimpleWikipedia simple.aligned and normal.aligned
- Creates 80/10/10 split (train/val/test) for simple texts
- All normal texts go to test set (anomaly detection setup)

### 2. Embedding Extraction (`embedding_extraction.py`)
- Downloads GloVe 6B 300d from HuggingFace (cached)
- Tokenizes text (lowercase, remove punctuation/special tokens)
- Skips OOV words, uses zero vector if all words OOV
- Mean pooling to create sentence embeddings (300-dim)

### 3. Model Training (`model_training.py`)
- Trains Isolation Forest on simple text embeddings only
- Hyperparameter grid search (from AD-NLP paper):
  - `n_estimators`: [64, 100, 128, 256]
  - `contamination`: [0.001, 0.01, 0.05]
- Validates on val set + sample of normal texts
- Saves best model

### 4. Evaluation (`evaluation.py`)
- Computes metrics:
  - AUROC (primary metric)
  - AUPR (precision-recall)
  - F1, Precision, Recall at optimal threshold
  - Confusion matrix
- Score distribution statistics
- Saves results to JSON

### 5. Visualization
- Score distribution histogram
- ROC curve
- Precision-Recall curve
- Confusion matrix heatmap

## Output Files

### Models & Data
- `data_splits.pkl`: Train/val/test splits
- `sentence_embeddings.pkl`: Cached embeddings (fast rerun)
- `best_isolation_forest.pkl`: Trained model

### Results
- `evaluation_results.json`: All metrics
- `hyperparameter_tuning_results.json`: Grid search results
- `pipeline.log`: Detailed execution log

### Plots
- `score_distribution.png`: Anomaly score histograms
- `roc_curve.png`: ROC curve
- `precision_recall_curve.png`: PR curve
- `confusion_matrix.png`: Confusion matrix heatmap

## Expected Results

Based on dataset analysis:
- **AUROC**: 0.65 - 0.75 (good separation)
- **F1**: 0.70+ (solid performance)
- **GloVe coverage**: ~88% of tokens (12-14% OOV)

## Troubleshooting

### GloVe Download Fails
```bash
# Manually download GloVe if HuggingFace fails
# Alternative: Use gensim or torchtext
```

### Memory Issues
```bash
# Reduce batch size for embeddings
# Use smaller GloVe (100d instead of 300d)
```

### Low Performance (AUROC < 0.60)
- Check OOV rate (should be ~12-14%)
- Try FastText instead of GloVe (handles OOV better)
- Augment with handcrafted features (length, word complexity)

## Citation

If using this code, please reference:
- AD-NLP paper (for Isolation Forest baseline)
- GloVe: Pennington et al., 2014
- SimpleWikipedia dataset

## Contact

For questions about this implementation, contact your supervisor.
