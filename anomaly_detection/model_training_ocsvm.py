"""
OC-SVM Model Training Module for Anomaly Detection
Alternative to Isolation Forest
"""
import pickle
import logging
import numpy as np
from typing import Dict, Tuple
from sklearn.svm import OneClassSVM
from tqdm import tqdm

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


# Hyperparameter grid for OC-SVM (based on AD-NLP paper)
OCSVM_PARAM_GRID = {
    'nu': [0.001, 0.01, 0.05, 0.1],  # Contamination levels
    'kernel': ['rbf'],  # Can also try 'linear' for comparison
    'gamma': ['scale', 'auto']  # RBF kernel width
}


def train_ocsvm(
    X_train: np.ndarray,
    nu: float = 0.01,
    kernel: str = 'rbf',
    gamma: str = 'scale'
) -> OneClassSVM:
    """
    Train One-Class SVM on training data (easy texts only).

    Args:
        X_train: Training embeddings (n_samples, n_features)
        nu: Upper bound on fraction of outliers (0 < nu <= 1)
        kernel: Kernel type ('rbf', 'linear', 'poly', 'sigmoid')
        gamma: Kernel coefficient for 'rbf', 'poly', 'sigmoid'

    Returns:
        Trained OC-SVM model
    """
    logger.info(f"Training One-Class SVM with nu={nu}, kernel={kernel}, gamma={gamma}")

    model = OneClassSVM(
        kernel=kernel,
        nu=nu,
        gamma=gamma,
        cache_size=1000,  # MB - increase for faster training
        verbose=False
    )

    model.fit(X_train)
    logger.info("Training complete")

    # Log support vector info
    n_support = len(model.support_)
    support_ratio = n_support / len(X_train) * 100
    logger.info(f"  Support vectors: {n_support} ({support_ratio:.1f}% of training data)")

    return model


def tune_ocsvm_hyperparameters(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    param_grid: Dict = None
) -> Tuple[OneClassSVM, Dict]:
    """
    Tune OC-SVM hyperparameters using validation set.

    Args:
        X_train: Training embeddings (easy texts only)
        X_val: Validation embeddings (mix of easy + normal)
        y_val: Validation labels (0=easy/inlier, 1=normal/outlier)
        param_grid: Grid of hyperparameters to try

    Returns:
        best_model: OC-SVM with best hyperparameters
        results: Dictionary of all results
    """
    if param_grid is None:
        param_grid = OCSVM_PARAM_GRID

    logger.info("="*60)
    logger.info("HYPERPARAMETER TUNING - OC-SVM")
    logger.info("="*60)
    logger.info(f"Parameter grid: {param_grid}")

    # Generate all combinations
    from itertools import product
    param_combinations = list(product(
        param_grid['nu'],
        param_grid['kernel'],
        param_grid['gamma']
    ))

    logger.info(f"Total combinations to try: {len(param_combinations)}")

    results = []
    best_auroc = 0
    best_model = None
    best_params = None

    # Try each combination
    for nu, kernel, gamma in tqdm(param_combinations, desc="Hyperparameter search"):
        logger.info(f"Training OC-SVM with nu={nu}, kernel={kernel}, gamma={gamma}")

        # Train model
        model = train_ocsvm(X_train, nu=nu, kernel=kernel, gamma=gamma)

        # Evaluate on validation set
        # OC-SVM decision_function: positive = inlier, negative = outlier
        # We negate to get: positive = outlier (for AUROC calculation)
        scores = -model.decision_function(X_val)

        # Compute AUROC
        from sklearn.metrics import roc_auc_score
        auroc = roc_auc_score(y_val, scores)

        logger.info(f"Params: nu={nu}, kernel={kernel}, gamma={gamma} -> AUROC: {auroc:.4f}")

        results.append({
            'nu': nu,
            'kernel': kernel,
            'gamma': gamma,
            'auroc': auroc,
            'n_support_vectors': len(model.support_)
        })

        # Track best
        if auroc > best_auroc:
            best_auroc = auroc
            best_model = model
            best_params = {'nu': nu, 'kernel': kernel, 'gamma': gamma}

    logger.info("="*60)
    logger.info(f"Best AUROC: {best_auroc:.4f}")
    logger.info(f"Best params: {best_params}")
    logger.info("="*60)

    # Save results
    results_dict = {
        'best_params': best_params,
        'best_auroc': best_auroc,
        'all_results': results
    }

    logger.info(f"Saving hyperparameter results to {config.HYPERPARAMETER_RESULTS_JSON}")
    with open(config.HYPERPARAMETER_RESULTS_JSON, 'w') as f:
        import json
        json.dump(results_dict, f, indent=2)

    return best_model, results_dict


def train_model(
    embeddings: Dict[str, np.ndarray],
    tune_hyperparameters: bool = True
) -> OneClassSVM:
    """
    Main training function for OC-SVM.

    Args:
        embeddings: Dictionary with 'train', 'val', 'test' embeddings
        tune_hyperparameters: If True, run hyperparameter tuning

    Returns:
        Trained OC-SVM model
    """
    X_train = embeddings['train']

    logger.info(f"Training data shape: {X_train.shape}")

    if tune_hyperparameters:
        # Prepare validation set (mix of simple and normal)
        # Sample 20% of normal texts to add to val set
        from data_preparation import load_aligned_file
        import numpy as np

        normal_texts_count = len(load_aligned_file(config.NORMAL_FILE))
        n_normal_sample = int(normal_texts_count * 0.2)

        # Create labels: 0=simple (inlier), 1=normal (outlier)
        X_val = embeddings['val']
        y_val_simple = np.zeros(len(X_val))

        # Add sample of normal text embeddings
        normal_indices = np.random.RandomState(config.RANDOM_SEED).choice(
            len(embeddings['test']) - len(embeddings['val']),
            size=n_normal_sample,
            replace=False
        )
        X_val_normal = embeddings['test'][len(embeddings['val']):][normal_indices]
        y_val_normal = np.ones(len(X_val_normal))

        # Combine
        X_eval = np.vstack([X_val, X_val_normal])
        y_eval = np.concatenate([y_val_simple, y_val_normal])

        logger.info(f"Using {len(X_eval):,} samples for validation")
        logger.info(f"  Simple (inlier): {len(y_val_simple):,}")
        logger.info(f"  Normal (outlier): {len(y_val_normal):,}")

        # Tune hyperparameters
        model, tuning_results = tune_ocsvm_hyperparameters(
            X_train, X_eval, y_eval, OCSVM_PARAM_GRID
        )

        # Train final model with best parameters
        logger.info("Training final model with best parameters...")
        best_params = tuning_results['best_params']
        model = train_ocsvm(
            X_train,
            nu=best_params['nu'],
            kernel=best_params['kernel'],
            gamma=best_params['gamma']
        )
    else:
        # Use default parameters
        logger.info("Training with default parameters (no tuning)")
        model = train_ocsvm(X_train, **config.OCSVM_DEFAULT_PARAMS)

    # Save model
    logger.info(f"Saving model to {config.BEST_MODEL_FILE}")
    with open(config.BEST_MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)

    return model


if __name__ == "__main__":
    # Test training
    from data_preparation import prepare_data
    from embedding_extraction import prepare_embeddings

    # Prepare data
    splits = prepare_data()

    # Extract embeddings
    embeddings = prepare_embeddings(splits)

    # Train model
    model = train_model(embeddings, tune_hyperparameters=True)

    print("OC-SVM training complete!")
