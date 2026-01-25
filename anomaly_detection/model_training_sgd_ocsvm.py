"""
SGD One-Class SVM Model Training Module
Fast linear kernel implementation for large datasets

Uses SGDOneClassSVM which has O(n) complexity vs O(n²) for standard OneClassSVM.
Only supports linear kernel, but much faster for our 134k training samples.
"""
import pickle
import logging
import numpy as np
from typing import Dict, Tuple
from sklearn.linear_model import SGDOneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

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


# Hyperparameter grid for SGD-OC-SVM
SGDOCSVM_PARAM_GRID = {
    'nu': [0.05, 0.1, 0.2, 0.5],
    'learning_rate': ['optimal', 'constant', 'adaptive'],
    'eta0': [0.0, 0.01, 0.1]  # Only used if learning_rate != 'optimal'
}


def train_sgd_ocsvm(
    X_train: np.ndarray,
    nu: float = 0.1,
    learning_rate: str = 'optimal',
    eta0: float = 0.0,
    max_iter: int = 1000,
    tol: float = 1e-3
) -> Tuple[SGDOneClassSVM, StandardScaler]:
    """
    Train SGD One-Class SVM on training data (linear kernel only).

    Much faster than standard OC-SVM: O(n) vs O(n²) complexity.

    Args:
        X_train: Training embeddings (n_samples, n_features)
        nu: Upper bound on fraction of outliers (0 < nu <= 1)
        learning_rate: Learning rate schedule
            - 'optimal': eta = 1.0 / (alpha * (t + t0)) [recommended]
            - 'constant': eta = eta0
            - 'adaptive': eta = eta0, decreases if no improvement
        eta0: Initial learning rate (for constant/adaptive)
        max_iter: Maximum number of passes over training data
        tol: Stopping criterion

    Returns:
        Tuple of (trained SGD-OC-SVM model, fitted StandardScaler)
    """
    logger.info(f"Training SGD One-Class SVM with nu={nu}, learning_rate={learning_rate}, eta0={eta0}")

    # CRITICAL: Normalize features first!
    logger.info("Normalizing features with StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    logger.info(f"Feature statistics after scaling:")
    logger.info(f"  Mean: {X_train_scaled.mean(axis=0)[:5]}... (should be ~0)")
    logger.info(f"  Std:  {X_train_scaled.std(axis=0)[:5]}... (should be ~1)")

    # Train SGD-OC-SVM (linear kernel only)
    model = SGDOneClassSVM(
        nu=nu,
        learning_rate=learning_rate,
        eta0=eta0,
        max_iter=max_iter,
        tol=tol,
        shuffle=True,
        random_state=config.RANDOM_SEED,
        verbose=0
    )

    model.fit(X_train_scaled)
    logger.info("Training complete")

    # Log convergence info
    logger.info(f"  Converged: {model.t_}")
    logger.info(f"  Final offset: {model.offset_[0]:.4f}")

    return model, scaler


def tune_sgd_ocsvm_hyperparameters(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    param_grid: Dict = None
) -> Tuple[SGDOneClassSVM, StandardScaler, Dict]:
    """
    Tune SGD-OC-SVM hyperparameters using validation set.

    Args:
        X_train: Training embeddings (easy texts only)
        X_val: Validation embeddings (mix of easy + normal)
        y_val: Validation labels (0=easy/inlier, 1=normal/outlier)
        param_grid: Grid of hyperparameters to try

    Returns:
        best_model: SGD-OC-SVM with best hyperparameters
        best_scaler: StandardScaler fitted on training data
        results: Dictionary of all results
    """
    if param_grid is None:
        param_grid = SGDOCSVM_PARAM_GRID

    logger.info("="*60)
    logger.info("HYPERPARAMETER TUNING - SGD-OC-SVM (Linear Kernel)")
    logger.info("="*60)
    logger.info(f"Parameter grid: {param_grid}")

    # Generate all combinations
    from itertools import product
    param_combinations = list(product(
        param_grid['nu'],
        param_grid['learning_rate'],
        param_grid['eta0']
    ))

    logger.info(f"Total combinations to try: {len(param_combinations)}")

    # Define helper function for parallel execution
    def train_and_evaluate(nu, learning_rate, eta0):
        """Train single SGD-OC-SVM and return results"""
        try:
            # Train model (returns model AND scaler)
            model, scaler = train_sgd_ocsvm(
                X_train,
                nu=nu,
                learning_rate=learning_rate,
                eta0=eta0
            )

            # Evaluate on validation set (MUST use same scaler!)
            X_val_scaled = scaler.transform(X_val)

            # SGD-OC-SVM decision_function: positive = inlier, negative = outlier
            # We negate to get: positive = outlier (for AUROC calculation)
            scores = -model.decision_function(X_val_scaled)

            # Compute AUROC
            auroc = roc_auc_score(y_val, scores)

            logger.info(f"Params: nu={nu}, lr={learning_rate}, eta0={eta0} -> AUROC: {auroc:.4f}")

            return {
                'nu': nu,
                'learning_rate': learning_rate,
                'eta0': eta0,
                'auroc': auroc,
                'n_iter': model.t_,
                'model': model,
                'scaler': scaler
            }
        except Exception as e:
            logger.error(f"Failed with nu={nu}, lr={learning_rate}, eta0={eta0}: {e}")
            return {
                'nu': nu,
                'learning_rate': learning_rate,
                'eta0': eta0,
                'auroc': 0.0,
                'error': str(e),
                'model': None,
                'scaler': None
            }

    # Run in parallel using all available CPUs
    import os
    n_jobs = int(os.environ.get('SLURM_CPUS_PER_TASK', -1))
    logger.info(f"Running hyperparameter search with {n_jobs} parallel jobs")

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(train_and_evaluate)(nu, learning_rate, eta0)
        for nu, learning_rate, eta0 in param_combinations
    )

    # Find best result
    best_result = max(results, key=lambda x: x.get('auroc', 0))
    best_auroc = best_result['auroc']
    best_model = best_result['model']
    best_scaler = best_result['scaler']
    best_params = {
        'nu': best_result['nu'],
        'learning_rate': best_result['learning_rate'],
        'eta0': best_result['eta0']
    }

    logger.info("="*60)
    logger.info(f"Best AUROC: {best_auroc:.4f}")
    logger.info(f"Best params: {best_params}")
    logger.info("="*60)

    # Save results (filter out model/scaler for JSON serialization)
    results_for_json = [
        {k: v for k, v in r.items() if k not in ['model', 'scaler']}
        for r in results
    ]
    results_dict = {
        'best_params': best_params,
        'best_auroc': best_auroc,
        'all_results': results_for_json
    }

    output_file = config.OUTPUT_DIR / "hyperparameter_tuning_results_sgd_ocsvm.json"
    logger.info(f"Saving hyperparameter results to {output_file}")
    with open(output_file, 'w') as f:
        import json
        json.dump(results_dict, f, indent=2)

    return best_model, best_scaler, results_dict


def train_model(
    embeddings: Dict[str, np.ndarray],
    labels: Dict[str, np.ndarray],
    tune_hyperparameters: bool = True
) -> Tuple[SGDOneClassSVM, StandardScaler]:
    """
    Main training function for SGD-OC-SVM.

    Args:
        embeddings: Dictionary with 'train', 'val', 'test' embeddings
        labels: Dictionary with 'train', 'val', 'test' labels
        tune_hyperparameters: If True, run hyperparameter tuning

    Returns:
        Tuple of (trained SGD-OC-SVM model, fitted StandardScaler)
    """
    X_train = embeddings['train']
    X_test = embeddings['test']
    y_test = np.array(labels['test']['labels'])

    logger.info(f"Training data shape: {X_train.shape}")

    if tune_hyperparameters:
        # Prepare validation set (mix of simple and normal)
        # Sample 20% of normal texts for validation
        test_normal_mask = (y_test == 1)
        test_normal_indices = np.where(test_normal_mask)[0]

        # Take first 20% as validation sample (deterministic)
        n_val_sample = int(len(test_normal_indices) * 0.2)
        val_sample_indices = test_normal_indices[:n_val_sample]
        X_test_normal_sample = X_test[val_sample_indices]

        # Create labels: 0=simple (inlier), 1=normal (outlier)
        X_val = embeddings['val']
        y_val_simple = np.zeros(len(X_val))
        y_val_normal = np.ones(len(X_test_normal_sample))

        # Combine
        X_eval = np.vstack([X_val, X_test_normal_sample])
        y_eval = np.concatenate([y_val_simple, y_val_normal])

        logger.info(f"Using {len(X_eval):,} samples for validation")
        logger.info(f"  Simple (inlier): {len(y_val_simple):,}")
        logger.info(f"  Normal (outlier): {len(y_val_normal):,}")

        # Tune hyperparameters
        model, scaler, tuning_results = tune_sgd_ocsvm_hyperparameters(
            X_train, X_eval, y_eval, SGDOCSVM_PARAM_GRID
        )

        # Train final model with best parameters
        logger.info("Training final model with best parameters...")
        best_params = tuning_results['best_params']
        model, scaler = train_sgd_ocsvm(
            X_train,
            nu=best_params['nu'],
            learning_rate=best_params['learning_rate'],
            eta0=best_params['eta0']
        )
    else:
        # Use default parameters
        logger.info("Training with default parameters (no tuning)")
        model, scaler = train_sgd_ocsvm(X_train, nu=0.1, learning_rate='optimal', eta0=0.0)

    # Save model and scaler
    model_file = config.OUTPUT_DIR / "best_sgd_ocsvm.pkl"
    scaler_file = config.OUTPUT_DIR / "best_sgd_ocsvm_scaler.pkl"

    logger.info(f"Saving model to {model_file}")
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)

    logger.info(f"Saving scaler to {scaler_file}")
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)

    return model, scaler


if __name__ == "__main__":
    # Test training
    from data_preparation import prepare_data
    from embedding_extraction import prepare_embeddings

    # Prepare data
    splits = prepare_data()

    # Extract embeddings
    embeddings = prepare_embeddings(splits)

    # Train model
    model, scaler = train_model(embeddings, splits, tune_hyperparameters=True)

    print("SGD-OC-SVM training complete!")
