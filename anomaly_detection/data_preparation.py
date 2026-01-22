"""
Data Preparation Module
Loads and splits SimpleWikipedia dataset
"""
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

import config

# Set up logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_aligned_file(filepath: Path) -> List[str]:
    """
    Load texts from aligned file format.

    Format: article_title<TAB>paragraph_number<TAB>sentence

    Args:
        filepath: Path to .aligned file

    Returns:
        List of sentence strings
    """
    texts = []

    logger.info(f"Loading texts from {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t', maxsplit=2)
            if len(parts) >= 3:
                text = parts[2].strip()
                if text:
                    texts.append(text)
            else:
                logger.warning(f"Line {line_num} has incorrect format: {line[:50]}...")

    logger.info(f"Loaded {len(texts)} texts")
    return texts


def create_data_splits(
    simple_texts: List[str],
    normal_texts: List[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42
) -> Dict[str, Dict[str, any]]:
    """
    Create train/val/test splits.

    Strategy:
    - Simple texts: Split into train (80%), val (10%), test (10%)
    - Normal texts: All go to test set (anomaly detection setup)

    Args:
        simple_texts: List of simple text strings
        normal_texts: List of normal text strings
        train_ratio: Proportion of simple texts for training
        val_ratio: Proportion of simple texts for validation
        test_ratio: Proportion of simple texts for testing
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary with splits:
        {
            'train': {'texts': [...], 'labels': [0, 0, ...]},
            'val': {'texts': [...], 'labels': [0, 0, ...]},
            'test': {'texts': [...], 'labels': [0, 0, ..., 1, 1, ...]}
        }
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"

    np.random.seed(random_seed)

    # Shuffle simple texts
    simple_indices = np.random.permutation(len(simple_texts))

    # Calculate split points
    n_simple = len(simple_texts)
    n_train = int(n_simple * train_ratio)
    n_val = int(n_simple * val_ratio)

    train_indices = simple_indices[:n_train]
    val_indices = simple_indices[n_train:n_train + n_val]
    test_indices = simple_indices[n_train + n_val:]

    # Create splits
    train_texts = [simple_texts[i] for i in train_indices]
    val_texts = [simple_texts[i] for i in val_indices]
    test_simple_texts = [simple_texts[i] for i in test_indices]

    # Test set: simple + normal
    test_texts = test_simple_texts + normal_texts
    test_labels = [0] * len(test_simple_texts) + [1] * len(normal_texts)

    splits = {
        'train': {
            'texts': train_texts,
            'labels': [0] * len(train_texts)  # All simple = inliers
        },
        'val': {
            'texts': val_texts,
            'labels': [0] * len(val_texts)  # All simple = inliers
        },
        'test': {
            'texts': test_texts,
            'labels': test_labels  # 0=simple (inlier), 1=normal (outlier)
        }
    }

    # Log statistics
    logger.info("=" * 60)
    logger.info("DATA SPLITS CREATED")
    logger.info("=" * 60)
    logger.info(f"Train set: {len(train_texts):,} samples (100% simple)")
    logger.info(f"Val set:   {len(val_texts):,} samples (100% simple)")
    logger.info(f"Test set:  {len(test_texts):,} samples "
                f"({len(test_simple_texts):,} simple + {len(normal_texts):,} normal)")
    logger.info(f"Test set label distribution: "
                f"{test_labels.count(0):,} simple (inliers), "
                f"{test_labels.count(1):,} normal (outliers)")
    logger.info("=" * 60)

    return splits


def prepare_data(force_reload: bool = False) -> Dict[str, Dict[str, any]]:
    """
    Load and prepare data splits.

    Args:
        force_reload: If True, reload from raw files even if cache exists

    Returns:
        Data splits dictionary
    """
    # Check if cached splits exist
    if config.SPLITS_FILE.exists() and not force_reload:
        logger.info(f"Loading cached data splits from {config.SPLITS_FILE}")
        with open(config.SPLITS_FILE, 'rb') as f:
            splits = pickle.load(f)
        logger.info("Cached splits loaded successfully")
        return splits

    # Load raw data
    logger.info("Loading raw data files...")
    simple_texts = load_aligned_file(config.SIMPLE_FILE)
    normal_texts = load_aligned_file(config.NORMAL_FILE)

    # Create splits
    splits = create_data_splits(
        simple_texts=simple_texts,
        normal_texts=normal_texts,
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        test_ratio=config.TEST_RATIO,
        random_seed=config.RANDOM_SEED
    )

    # Save splits to cache
    logger.info(f"Saving data splits to {config.SPLITS_FILE}")
    with open(config.SPLITS_FILE, 'wb') as f:
        pickle.dump(splits, f)

    return splits


if __name__ == "__main__":
    # Test data preparation
    splits = prepare_data(force_reload=True)

    print("\nSample texts from each split:")
    print("\nTrain (first 2):")
    for i, text in enumerate(splits['train']['texts'][:2]):
        print(f"  [{i}] {text[:100]}...")

    print("\nVal (first 2):")
    for i, text in enumerate(splits['val']['texts'][:2]):
        print(f"  [{i}] {text[:100]}...")

    print("\nTest - Simple (first 2):")
    test_simple = [t for t, l in zip(splits['test']['texts'], splits['test']['labels']) if l == 0]
    for i, text in enumerate(test_simple[:2]):
        print(f"  [{i}] {text[:100]}...")

    print("\nTest - Normal (first 2):")
    test_normal = [t for t, l in zip(splits['test']['texts'], splits['test']['labels']) if l == 1]
    for i, text in enumerate(test_normal[:2]):
        print(f"  [{i}] {text[:100]}...")
