"""
Evaluation Module
Comprehensive evaluation metrics and visualization
"""
import json
import logging
from typing import Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

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

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11


def compute_anomaly_scores(
    model,
    X_test: np.ndarray,
    scaler: StandardScaler = None
) -> np.ndarray:
    """
    Compute anomaly scores for test data.

    Works with both Isolation Forest and OC-SVM.

    Args:
        model: Trained model (Isolation Forest or OC-SVM)
        X_test: Test embeddings
        scaler: StandardScaler (required for OC-SVM, optional for IsolationForest)

    Returns:
        Anomaly scores (higher = more anomalous)
    """
    # If scaler provided, normalize features (for OC-SVM)
    if scaler is not None:
        X_test = scaler.transform(X_test)

    # Get decision scores
    # Both IsolationForest and OC-SVM return negative scores (more negative = more anomalous)
    scores = model.decision_function(X_test)

    # Convert to positive (higher = more anomalous)
    scores = -scores

    return scores


def find_optimal_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    metric: str = 'f1'
) -> Tuple[float, float]:
    """
    Find optimal threshold for binary classification.

    Args:
        y_true: True labels (0=inlier, 1=outlier)
        scores: Anomaly scores
        metric: Metric to optimize ('f1', 'precision', 'recall')

    Returns:
        Tuple of (optimal_threshold, best_metric_value)
    """
    # Try percentile-based thresholds
    thresholds = np.percentile(scores, np.arange(10, 91, 5))

    best_value = 0
    best_threshold = None

    for threshold in thresholds:
        y_pred = (scores > threshold).astype(int)

        if metric == 'f1':
            value = f1_score(y_true, y_pred)
        elif metric == 'precision':
            value = precision_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            value = recall_score(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if value > best_value:
            best_value = value
            best_threshold = threshold

    return best_threshold, best_value


def evaluate_model(
    model: IsolationForest,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler: StandardScaler = None,
    save_plots: bool = True
) -> Dict:
    """
    Comprehensive model evaluation.

    Args:
        model: Trained Isolation Forest or OC-SVM
        X_test: Test embeddings
        y_test: Test labels (0=inlier, 1=outlier)
        scaler: Optional StandardScaler for OC-SVM
        save_plots: Whether to generate and save plots

    Returns:
        Dictionary with all evaluation metrics
    """
    logger.info("=" * 60)
    logger.info("MODEL EVALUATION")
    logger.info("=" * 60)

    # Compute anomaly scores
    scores = compute_anomaly_scores(model, X_test, scaler=scaler)

    results = {}

    # 1. AUROC
    if config.COMPUTE_AUROC:
        auroc = roc_auc_score(y_test, scores)
        results['auroc'] = auroc
        logger.info(f"AUROC: {auroc:.4f}")

    # 2. AUPR (Area Under Precision-Recall)
    if config.COMPUTE_AUPR:
        # AUPR for detecting outliers (normal texts)
        aupr_outlier = average_precision_score(y_test, scores)
        # AUPR for detecting inliers (simple texts)
        aupr_inlier = average_precision_score(1 - y_test, -scores)

        results['aupr_outlier'] = aupr_outlier
        results['aupr_inlier'] = aupr_inlier
        logger.info(f"AUPR (Outlier): {aupr_outlier:.4f}")
        logger.info(f"AUPR (Inlier): {aupr_inlier:.4f}")

    # 3. Optimal F1 threshold
    if config.COMPUTE_F1:
        optimal_threshold, best_f1 = find_optimal_threshold(y_test, scores, metric='f1')
        results['optimal_threshold'] = optimal_threshold
        results['best_f1'] = best_f1

        # Predictions at optimal threshold
        y_pred = (scores > optimal_threshold).astype(int)

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        results['precision_at_optimal'] = precision
        results['recall_at_optimal'] = recall

        logger.info(f"Optimal threshold: {optimal_threshold:.4f}")
        logger.info(f"Best F1: {best_f1:.4f}")
        logger.info(f"Precision at optimal: {precision:.4f}")
        logger.info(f"Recall at optimal: {recall:.4f}")

    # 4. Confusion Matrix
    if config.COMPUTE_CONFUSION_MATRIX:
        y_pred = (scores > optimal_threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred)

        results['confusion_matrix'] = cm.tolist()

        logger.info("Confusion Matrix:")
        logger.info(f"  TN={cm[0,0]}, FP={cm[0,1]}")
        logger.info(f"  FN={cm[1,0]}, TP={cm[1,1]}")

        # Rates
        tn, fp, fn, tp = cm.ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        results['true_positive_rate'] = tpr
        results['false_positive_rate'] = fpr
        results['true_negative_rate'] = tnr
        results['false_negative_rate'] = fnr

        logger.info(f"TPR (Recall): {tpr:.4f}")
        logger.info(f"FPR: {fpr:.4f}")

    # 5. Score distribution statistics
    scores_inlier = scores[y_test == 0]
    scores_outlier = scores[y_test == 1]

    results['score_stats'] = {
        'inlier_mean': float(np.mean(scores_inlier)),
        'inlier_std': float(np.std(scores_inlier)),
        'inlier_median': float(np.median(scores_inlier)),
        'outlier_mean': float(np.mean(scores_outlier)),
        'outlier_std': float(np.std(scores_outlier)),
        'outlier_median': float(np.median(scores_outlier)),
        'separation': float(abs(np.mean(scores_outlier) - np.mean(scores_inlier)))
    }

    logger.info("Score Distribution:")
    logger.info(f"  Inlier (simple):  mean={results['score_stats']['inlier_mean']:.4f}, "
                f"std={results['score_stats']['inlier_std']:.4f}")
    logger.info(f"  Outlier (normal): mean={results['score_stats']['outlier_mean']:.4f}, "
                f"std={results['score_stats']['outlier_std']:.4f}")
    logger.info(f"  Separation: {results['score_stats']['separation']:.4f}")

    logger.info("=" * 60)

    # Save results
    logger.info(f"Saving evaluation results to {config.RESULTS_JSON}")
    with open(config.RESULTS_JSON, 'w') as f:
        json.dump(results, f, indent=2)

    # Store scores for plotting
    results['_scores'] = scores
    results['_y_test'] = y_test

    return results


def plot_score_distribution(results: Dict):
    """
    Plot distribution of anomaly scores.

    Args:
        results: Results dictionary from evaluate_model()
    """
    scores = results['_scores']
    y_test = results['_y_test']

    scores_inlier = scores[y_test == 0]
    scores_outlier = scores[y_test == 1]

    plt.figure(figsize=(10, 6))
    plt.hist(scores_inlier, bins=50, alpha=0.6, label='Simple (Inlier)', color='blue', density=True)
    plt.hist(scores_outlier, bins=50, alpha=0.6, label='Normal (Outlier)', color='red', density=True)

    # Add optimal threshold line
    if 'optimal_threshold' in results:
        plt.axvline(results['optimal_threshold'], color='green', linestyle='--',
                    linewidth=2, label=f"Optimal Threshold ({results['optimal_threshold']:.3f})")

    plt.xlabel('Anomaly Score', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Distribution of Anomaly Scores', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(config.SCORE_DISTRIBUTION_PLOT, dpi=config.DPI, bbox_inches='tight')
    logger.info(f"Saved score distribution plot to {config.SCORE_DISTRIBUTION_PLOT}")
    plt.close()


def plot_roc_curve(results: Dict):
    """
    Plot ROC curve.

    Args:
        results: Results dictionary from evaluate_model()
    """
    scores = results['_scores']
    y_test = results['_y_test']

    fpr, tpr, thresholds = roc_curve(y_test, scores)
    auroc = results['auroc']

    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, linewidth=2, label=f'Isolation Forest (AUROC = {auroc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier (AUROC = 0.5)')

    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(config.ROC_CURVE_PLOT, dpi=config.DPI, bbox_inches='tight')
    logger.info(f"Saved ROC curve to {config.ROC_CURVE_PLOT}")
    plt.close()


def plot_precision_recall_curve(results: Dict):
    """
    Plot Precision-Recall curve.

    Args:
        results: Results dictionary from evaluate_model()
    """
    scores = results['_scores']
    y_test = results['_y_test']

    precision, recall, thresholds = precision_recall_curve(y_test, scores)
    aupr = results['aupr_outlier']

    plt.figure(figsize=(8, 8))
    plt.plot(recall, precision, linewidth=2, label=f'Isolation Forest (AUPR = {aupr:.4f})')

    # Baseline (random classifier)
    baseline = np.sum(y_test) / len(y_test)
    plt.axhline(baseline, color='k', linestyle='--', linewidth=1,
                label=f'Random Classifier (AUPR = {baseline:.4f})')

    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve (Outlier Detection)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(config.PR_CURVE_PLOT, dpi=config.DPI, bbox_inches='tight')
    logger.info(f"Saved Precision-Recall curve to {config.PR_CURVE_PLOT}")
    plt.close()


def plot_confusion_matrix(results: Dict):
    """
    Plot confusion matrix heatmap.

    Args:
        results: Results dictionary from evaluate_model()
    """
    cm = np.array(results['confusion_matrix'])

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Inlier', 'Outlier'],
                yticklabels=['Inlier', 'Outlier'],
                annot_kws={'fontsize': 14})

    plt.xlabel('Predicted', fontsize=12, fontweight='bold')
    plt.ylabel('Actual', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()

    plt.savefig(config.CONFUSION_MATRIX_PLOT, dpi=config.DPI, bbox_inches='tight')
    logger.info(f"Saved confusion matrix to {config.CONFUSION_MATRIX_PLOT}")
    plt.close()


def create_all_plots(results: Dict):
    """
    Create all evaluation plots.

    Args:
        results: Results dictionary from evaluate_model()
    """
    if config.SAVE_PLOTS:
        logger.info("Creating evaluation plots...")
        plot_score_distribution(results)
        plot_roc_curve(results)
        plot_precision_recall_curve(results)
        plot_confusion_matrix(results)
        logger.info("All plots saved")


if __name__ == "__main__":
    # Test evaluation
    import pickle
    from data_preparation import prepare_data

    # Load model and data
    splits = prepare_data()
    y_test = np.array(splits['test']['labels'])

    with open(config.BEST_MODEL_FILE, 'rb') as f:
        model = pickle.load(f)

    with open(config.EMBEDDINGS_CACHE, 'rb') as f:
        embeddings = pickle.load(f)

    X_test = embeddings['test']

    # Evaluate
    results = evaluate_model(model, X_test, y_test)

    # Create plots
    create_all_plots(results)

    print("\nEvaluation complete!")
