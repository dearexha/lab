"""
Configuration for Anomaly Detection Pipeline
"""
import os
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================
PROJECT_ROOT = Path("/home/user/lab")
DATA_DIR = PROJECT_ROOT / "datasets" / "SimpleWikipedia_v2"
OUTPUT_DIR = PROJECT_ROOT / "anomaly_detection" / "outputs"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"

# Create output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA CONFIGURATION
# ============================================================================
SIMPLE_FILE = DATA_DIR / "simple.aligned"
NORMAL_FILE = DATA_DIR / "normal.aligned"

# Data splits (80% train, 10% val, 10% test for simple texts)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# GLOVE CONFIGURATION
# ============================================================================
GLOVE_MODEL = "glove.6B.300d"  # 300-dimensional GloVe embeddings
GLOVE_DIM = 300
GLOVE_CACHE_FILE = EMBEDDINGS_DIR / "glove_6B_300d.pkl"

# HuggingFace dataset for GloVe
GLOVE_HF_DATASET = "stanfordnlp/glove"
GLOVE_HF_CONFIG = "6B"  # Wikipedia 2014 + Gigaword 5

# ============================================================================
# PREPROCESSING CONFIGURATION
# ============================================================================
# Special tokens to remove (parsing artifacts from Wikipedia)
SPECIAL_TOKENS = ['lrb', 'rrb', 'nbsp', 'quot', 'amp', 'lt', 'gt']

# OOV handling
OOV_STRATEGY = "skip"  # Options: "skip", "zero", "mean"
USE_ZERO_VECTOR_FOR_EMPTY = True  # Use zero vector if all words are OOV

# Text preprocessing
LOWERCASE = True
REMOVE_PUNCTUATION = True
MIN_WORDS_PER_SENTENCE = 1  # Minimum words after OOV filtering (0 = allow all)

# ============================================================================
# ISOLATION FOREST CONFIGURATION
# ============================================================================
# Hyperparameter grid (from AD-NLP paper)
IFOREST_PARAM_GRID = {
    'n_estimators': [64, 100, 128, 256],
    'contamination': [0.001, 0.01, 0.05],
    'max_samples': ['auto'],  # Default: min(256, n_samples)
    'random_state': [RANDOM_SEED]
}

# Default parameters (if not tuning)
IFOREST_DEFAULT_PARAMS = {
    'n_estimators': 100,
    'contamination': 0.01,
    'max_samples': 'auto',
    'random_state': RANDOM_SEED,
    'n_jobs': -1  # Use all available cores
}

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================
# Metrics to compute
COMPUTE_AUROC = True
COMPUTE_AUPR = True
COMPUTE_F1 = True
COMPUTE_CONFUSION_MATRIX = True

# Visualization
SAVE_PLOTS = True
PLOT_FORMAT = 'png'
DPI = 300

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR
LOG_FILE = OUTPUT_DIR / "pipeline.log"

# ============================================================================
# OUTPUT FILES
# ============================================================================
# Data splits
SPLITS_FILE = OUTPUT_DIR / "data_splits.pkl"

# Embeddings (cached for faster rerun)
EMBEDDINGS_CACHE = OUTPUT_DIR / "sentence_embeddings.pkl"

# Models
BEST_MODEL_FILE = OUTPUT_DIR / "best_isolation_forest.pkl"

# Results
RESULTS_JSON = OUTPUT_DIR / "evaluation_results.json"
HYPERPARAMETER_RESULTS_JSON = OUTPUT_DIR / "hyperparameter_tuning_results.json"

# Plots
SCORE_DISTRIBUTION_PLOT = OUTPUT_DIR / "score_distribution.png"
ROC_CURVE_PLOT = OUTPUT_DIR / "roc_curve.png"
PR_CURVE_PLOT = OUTPUT_DIR / "precision_recall_curve.png"
CONFUSION_MATRIX_PLOT = OUTPUT_DIR / "confusion_matrix.png"

print(f"Configuration loaded. Output directory: {OUTPUT_DIR}")
