"""
Main Pipeline for SGD One-Class SVM Anomaly Detection
Fast linear kernel implementation using stochastic gradient descent

This is a faster alternative to standard OC-SVM for linear kernels:
- Standard OC-SVM linear: O(n²) complexity, ~hours
- SGD-OC-SVM linear: O(n) complexity, ~minutes

Usage:
    python main_sgd_ocsvm.py
    python main_sgd_ocsvm.py --no-tuning      # Skip hyperparameter tuning
    python main_sgd_ocsvm.py --reload-data    # Force reload data from raw files
"""
import argparse
import logging
import numpy as np
from pathlib import Path

import config
from data_preparation import prepare_data
from embedding_extraction import prepare_embeddings
from model_training_sgd_ocsvm import train_model
from evaluation import evaluate_model

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


def main(args):
    """Main pipeline for SGD-OC-SVM anomaly detection."""

    logger.info("="*80)
    logger.info("SGD ONE-CLASS SVM ANOMALY DETECTION PIPELINE")
    logger.info("="*80)
    logger.info(f"Embedding type: {config.EMBEDDING_TYPE}")
    logger.info(f"Output directory: {config.OUTPUT_DIR}")
    logger.info("")

    # ========================================================================
    # STEP 1: Data Preparation
    # ========================================================================
    logger.info("STEP 1/5: Data Preparation")
    logger.info("-" * 80)

    splits = prepare_data(force_reload=args.reload_data)

    logger.info(f"Train set: {len(splits['train']['texts']):,} texts (100% simple)")
    logger.info(f"Val set:   {len(splits['val']['texts']):,} texts (100% simple)")
    logger.info(f"Test set:  {len(splits['test']['texts']):,} texts")
    logger.info(f"  Simple (inliers):  {splits['test']['labels'].count(0):,}")
    logger.info(f"  Normal (outliers): {splits['test']['labels'].count(1):,}")
    logger.info("")

    # ========================================================================
    # STEP 2: Embedding Extraction
    # ========================================================================
    logger.info("STEP 2/5: Embedding Extraction")
    logger.info("-" * 80)

    embeddings = prepare_embeddings(splits, force_reload=args.reload_embeddings)

    logger.info(f"Train embeddings shape: {embeddings['train'].shape}")
    logger.info(f"Val embeddings shape:   {embeddings['val'].shape}")
    logger.info(f"Test embeddings shape:  {embeddings['test'].shape}")
    logger.info("")

    # ========================================================================
    # STEP 3: Model Training (SGD-OC-SVM)
    # ========================================================================
    logger.info("STEP 3/5: Model Training (SGD-OC-SVM)")
    logger.info("-" * 80)

    model, scaler = train_model(
        embeddings=embeddings,
        labels=splits,
        tune_hyperparameters=not args.no_tuning
    )

    logger.info("SGD-OC-SVM training complete")
    logger.info(f"Nu parameter: {model.nu}")
    logger.info(f"Learning rate: {model.learning_rate}")
    logger.info(f"Iterations: {model.t_}")
    logger.info("")

    # ========================================================================
    # STEP 4: Evaluation
    # ========================================================================
    logger.info("STEP 4/5: Evaluation")
    logger.info("-" * 80)

    # Get test data
    X_test = embeddings['test']
    y_test = np.array(splits['test']['labels'])

    logger.info(f"Test set: {len(y_test):,} samples")
    logger.info(f"  Simple (inliers):  {(y_test == 0).sum():,}")
    logger.info(f"  Normal (outliers): {(y_test == 1).sum():,}")

    # Evaluate (scores computed internally)
    results = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        scaler=scaler,
        save_plots=not args.no_plots
    )

    # Print results
    logger.info("")
    logger.info("=" * 80)
    logger.info("EVALUATION RESULTS (SGD-OC-SVM)")
    logger.info("=" * 80)
    logger.info(f"AUROC: {results['auroc']:.4f}")
    logger.info(f"Best F1 Score: {results['best_f1']:.4f}")
    logger.info(f"  Threshold: {results['best_threshold']:.4f}")
    logger.info(f"  Precision: {results['precision_at_optimal']:.4f}")
    logger.info(f"  Recall: {results['recall_at_optimal']:.4f}")
    logger.info("")
    logger.info("Score Statistics:")
    logger.info(f"  Inlier mean:  {results['score_stats']['inlier_mean']:.4f}")
    logger.info(f"  Outlier mean: {results['score_stats']['outlier_mean']:.4f}")
    logger.info(f"  Separation:   {results['score_stats']['separation']:.4f}")
    logger.info("=" * 80)

    # ========================================================================
    # STEP 5: Save Results
    # ========================================================================
    logger.info("STEP 5/5: Saving Results")
    logger.info("-" * 80)

    import json
    results_file = config.OUTPUT_DIR / "evaluation_results_sgd_ocsvm.json"

    logger.info(f"Saving evaluation results to {results_file}")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    # ========================================================================
    # Comparison with Standard OC-SVM (if available)
    # ========================================================================
    ocsvm_results_file = config.OUTPUT_DIR / "evaluation_results_ocsvm.json"
    if ocsvm_results_file.exists():
        logger.info("")
        logger.info("=" * 80)
        logger.info("COMPARISON: SGD-OC-SVM vs Standard OC-SVM")
        logger.info("=" * 80)

        try:
            with open(ocsvm_results_file, 'r') as f:
                ocsvm_results = json.load(f)

            logger.info(f"{'Metric':<20} {'Standard OC-SVM':>15} {'SGD-OC-SVM':>15} {'Difference':>15}")
            logger.info("-" * 70)

            auroc_diff = results['auroc'] - ocsvm_results.get('auroc', 0)
            logger.info(f"{'AUROC':<20} {ocsvm_results.get('auroc', 0):>15.4f} {results['auroc']:>15.4f} {auroc_diff:>+15.4f}")

            f1_diff = results['best_f1'] - ocsvm_results.get('best_f1', 0)
            logger.info(f"{'F1 Score':<20} {ocsvm_results.get('best_f1', 0):>15.4f} {results['best_f1']:>15.4f} {f1_diff:>+15.4f}")

            sep_diff = results['score_stats']['separation'] - ocsvm_results.get('score_stats', {}).get('separation', 0)
            logger.info(f"{'Score Separation':<20} {ocsvm_results.get('score_stats', {}).get('separation', 0):>15.4f} {results['score_stats']['separation']:>15.4f} {sep_diff:>+15.4f}")

            logger.info("=" * 80)

        except Exception as e:
            logger.warning(f"Could not load OC-SVM results for comparison: {e}")

    # ========================================================================
    # Comparison with Isolation Forest (if available)
    # ========================================================================
    iforest_results_file = config.OUTPUT_DIR.parent / "outputs" / "evaluation_results.json"
    if iforest_results_file.exists():
        logger.info("")
        logger.info("=" * 80)
        logger.info("COMPARISON: SGD-OC-SVM vs Isolation Forest")
        logger.info("=" * 80)

        try:
            with open(iforest_results_file, 'r') as f:
                iforest_results = json.load(f)

            logger.info(f"{'Metric':<20} {'Isolation Forest':>18} {'SGD-OC-SVM':>15} {'Difference':>15}")
            logger.info("-" * 73)

            auroc_diff = results['auroc'] - iforest_results.get('auroc', 0)
            logger.info(f"{'AUROC':<20} {iforest_results.get('auroc', 0):>18.4f} {results['auroc']:>15.4f} {auroc_diff:>+15.4f}")

            sep_diff = results['score_stats']['separation'] - iforest_results.get('score_stats', {}).get('separation', 0)
            logger.info(f"{'Score Separation':<20} {iforest_results.get('score_stats', {}).get('separation', 0):>18.4f} {results['score_stats']['separation']:>15.4f} {sep_diff:>+15.4f}")

            logger.info("=" * 80)

        except Exception as e:
            logger.warning(f"Could not load Isolation Forest results for comparison: {e}")

    logger.info("")
    logger.info("Pipeline complete!")
    logger.info(f"All results saved to: {config.OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SGD One-Class SVM Anomaly Detection Pipeline")

    parser.add_argument(
        '--no-tuning',
        action='store_true',
        help='Skip hyperparameter tuning (use default parameters)'
    )

    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip plot generation'
    )

    parser.add_argument(
        '--reload-data',
        action='store_true',
        help='Force reload data from raw files (ignore cache)'
    )

    parser.add_argument(
        '--reload-embeddings',
        action='store_true',
        help='Force recompute embeddings (ignore cache)'
    )

    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        raise
