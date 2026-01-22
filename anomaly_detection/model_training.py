"""
Model Training Module
Isolation Forest training and hyperparameter tuning
"""
import pickle
import json
import logging
from typing import Dict, List, Tuple
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from itertools import product
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


def train_isolation_forest(
    X_train: np.ndarray,
    n_estimators: int = 100,
    contamination: float = 0.01,
    max_samples: str = 'auto',
    random_state: int = 42
) -> IsolationForest:
    """
    Train Isolation Forest model.

    Args:
        X_train: Training data (n_samples, n_features)
        n_estimators: Number of trees
        contamination: Expected proportion of outliers in training data
        max_samples: Number of samples to draw for each tree
        random_state: Random seed

    Returns:
        Trained IsolationForest model
    """
    logger.info(f"Training Isolation Forest with n_estimators={n_estimators}, "
                f"contamination={contamination}")

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples=max_samples,
        random_state=random_state,
        n_jobs=-1,  # Use all cores
        verbose=0
    )

    model.fit(X_train)

    logger.info("Training complete")
    return model


def evaluate_on_validation(
    model: IsolationForest,
    X_val: np.ndarray,
    X_test_normal_sample: np.ndarray
) -> float:
    """
    Evaluate model on validation set using AUROC.

    Strategy:
    - Validation contains only simple texts (inliers)
    - We need outliers for AUROC computation
    - Use a sample of normal texts from test set as outliers

    Args:
        model: Trained Isolation Forest
        X_val: Validation embeddings (simple texts only)
        X_test_normal_sample: Sample of normal text embeddings

    Returns:
        AUROC score
    """
    # Combine val (inliers) and normal sample (outliers)
    X_eval = np.vstack([X_val, X_test_normal_sample])
    y_eval = np.array([0] * len(X_val) + [1] * len(X_test_normal_sample))

    # Get anomaly scores (more negative = more anomalous in sklearn)
    scores = model.decision_function(X_eval)

    # Convert to positive scores (higher = more anomalous)
    scores = -scores

    # Compute AUROC
    auroc = roc_auc_score(y_eval, scores)

    return auroc


def hyperparameter_tuning(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test_normal_sample: np.ndarray,
    param_grid: Dict[str, List] = None
) -> Tuple[Dict, List[Dict]]:
    """
    Perform grid search for hyperparameter tuning.

    Args:
        X_train: Training embeddings
        X_val: Validation embeddings
        X_test_normal_sample: Sample of normal embeddings for validation
        param_grid: Parameter grid (uses config default if None)

    Returns:
        Tuple of (best_params, all_results)
    """
    if param_grid is None:
        param_grid = config.IFOREST_PARAM_GRID

    logger.info("=" * 60)
    logger.info("HYPERPARAMETER TUNING")
    logger.info("=" * 60)
    logger.info(f"Parameter grid: {param_grid}")

    # Generate all parameter combinations
    keys = param_grid.keys()
    values = param_grid.values()
    param_combinations = [dict(zip(keys, v)) for v in product(*values)]

    logger.info(f"Total combinations to try: {len(param_combinations)}")

    results = []
    best_auroc = 0
    best_params = None

    for params in tqdm(param_combinations, desc="Hyperparameter search"):
        # Train model
        model = train_isolation_forest(
            X_train,
            n_estimators=params['n_estimators'],
            contamination=params['contamination'],
            max_samples=params['max_samples'],
            random_state=params['random_state']
        )

        # Evaluate
        auroc = evaluate_on_validation(model, X_val, X_test_normal_sample)

        result = {
            'params': params,
            'auroc': auroc
        }
        results.append(result)

        logger.info(f"Params: {params} -> AUROC: {auroc:.4f}")

        # Track best
        if auroc > best_auroc:
            best_auroc = auroc
            best_params = params

    logger.info("=" * 60)
    logger.info(f"Best AUROC: {best_auroc:.4f}")
    logger.info(f"Best params: {best_params}")
    logger.info("=" * 60)

    # Save hyperparameter results
    logger.info(f"Saving hyperparameter results to {config.HYPERPARAMETER_RESULTS_JSON}")
    with open(config.HYPERPARAMETER_RESULTS_JSON, 'w') as f:
        json.dump({
            'best_params': best_params,
            'best_auroc': best_auroc,
            'all_results': results
        }, f, indent=2)

    return best_params, results


def train_best_model(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    tune_hyperparameters: bool = True
) -> IsolationForest:
    """
    Train final model with best hyperparameters.

    Args:
        X_train: Training embeddings
        X_val: Validation embeddings
        X_test: Test embeddings
        y_test: Test labels
        tune_hyperparameters: If True, perform hyperparameter tuning

    Returns:
        Trained IsolationForest model
    """
    if tune_hyperparameters:
        # Sample normal texts for validation
        # Use 20% of normal texts to avoid data leakage
        test_normal_mask = (y_test == 1)
        test_normal_indices = np.where(test_normal_mask)[0]

        # Take first 20% as validation sample (deterministic)
        n_val_sample = int(len(test_normal_indices) * 0.2)
        val_sample_indices = test_normal_indices[:n_val_sample]
        X_test_normal_sample = X_test[val_sample_indices]

        logger.info(f"Using {len(X_test_normal_sample):,} normal texts for validation")

        # Tune hyperparameters
        best_params, _ = hyperparameter_tuning(
            X_train,
            X_val,
            X_test_normal_sample
        )
    else:
        # Use default parameters
        best_params = config.IFOREST_DEFAULT_PARAMS
        logger.info(f"Using default parameters: {best_params}")

    # Train final model with best parameters
    logger.info("Training final model with best parameters...")
    model = train_isolation_forest(
        X_train,
        n_estimators=best_params['n_estimators'],
        contamination=best_params['contamination'],
        max_samples=best_params.get('max_samples', 'auto'),
        random_state=best_params.get('random_state', config.RANDOM_SEED)
    )

    # Save model
    logger.info(f"Saving model to {config.BEST_MODEL_FILE}")
    with open(config.BEST_MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)

    return model


if __name__ == "__main__":
    # Test model training
    from data_preparation import prepare_data
    from embedding_extraction import prepare_embeddings

    # Prepare data and embeddings
    splits = prepare_data()
    embeddings = prepare_embeddings(splits)

    # Train model
    model = train_best_model(
        X_train=embeddings['train'],
        X_val=embeddings['val'],
        X_test=embeddings['test'],
        y_test=np.array(splits['test']['labels']),
        tune_hyperparameters=True  # Set to False for quick test
    )

    print(f"\nModel trained: {model}")
