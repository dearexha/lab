"""
Main Pipeline for OC-SVM Anomaly Detection
Separates easy texts (Simple Wikipedia) from hard texts (Normal Wikipedia)

Based on AD-NLP paper methodology
"""
import argparse
import logging
import numpy as np
import sys
from pathlib import Path

import config
from data_preparation import prepare_data
from embedding_extraction import prepare_embeddings
from model_training_ocsvm import train_model
from evaluation import (
    compute_anomaly_scores,
    find_optimal_threshold,
    evaluate_model
)

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
    """
    Main pipeline for OC-SVM anomaly detection.

    Steps:
    1. Load and split data (train/val/test)
    2. Extract embeddings (BERT or GloVe)
    3. Train OC-SVM on easy texts only
    4. Evaluate on easy + hard texts
    5. Compute metrics and generate plots
    """
    logger.info("=" * 80)
    logger.info("OC-SVM ANOMALY DETECTION PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Embedding type: {config.EMBEDDING_TYPE}")
    logger.info(f"Output directory: {config.OUTPUT_DIR}")
    logger.info("")

    # ========================================================================
    # STEP 1: Data Preparation
    # ========================================================================
    logger.info("STEP 1/5: Data Preparation")
    logger.info("-" * 80)

    splits = prepare_data(force_reload=args.reload_data)

    logger.info(f"Train set: {len(splits['train']['texts']):,} samples (all simple/easy)")
    logger.info(f"Val set:   {len(splits['val']['texts']):,} samples (all simple/easy)")
    logger.info(f"Test set:  {len(splits['test']['texts']):,} samples "
                f"(simple + normal)")
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
    # STEP 3: Model Training (OC-SVM)
    # ========================================================================
    logger.info("STEP 3/5: Model Training (OC-SVM)")
    logger.info("-" * 80)

    model, scaler = train_model(
        embeddings=embeddings,
        labels=splits,
        tune_hyperparameters=not args.no_tuning
    )

    logger.info("OC-SVM training complete")
    logger.info(f"Support vectors: {model.n_support_}")
    logger.info(f"Nu parameter: {model.nu}")
    logger.info(f"Kernel: {model.kernel}")
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

    # Compute anomaly scores
    scores = compute_anomaly_scores(model, X_test, scaler=scaler)

    # Evaluate
    results = evaluate_model(
        y_true=y_test,
        scores=scores,
        save_plots=not args.no_plots
    )

    # Print results
    logger.info("")
    logger.info("=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)
    logger.info(f"AUROC: {results['auroc']:.4f}")
    logger.info(f"AUPR (Outlier): {results['aupr_outlier']:.4f}")
    logger.info(f"AUPR (Inlier):  {results['aupr_inlier']:.4f}")
    logger.info(f"")
    logger.info(f"Optimal threshold: {results['optimal_threshold']:.4f}")
    logger.info(f"Best F1:           {results['best_f1']:.4f}")
    logger.info(f"Precision:         {results['precision_at_optimal']:.4f}")
    logger.info(f"Recall:            {results['recall_at_optimal']:.4f}")
    logger.info(f"")
    logger.info("Confusion Matrix:")
    cm = results['confusion_matrix']
    logger.info(f"  TN={cm[0][0]:,}  FP={cm[0][1]:,}")
    logger.info(f"  FN={cm[1][0]:,}  TP={cm[1][1]:,}")
    logger.info(f"")
    logger.info(f"Score separation: {results['score_stats']['separation']:.4f}")
    logger.info(f"  Inlier mean:  {results['score_stats']['inlier_mean']:.4f}")
    logger.info(f"  Outlier mean: {results['score_stats']['outlier_mean']:.4f}")
    logger.info("=" * 80)

    # ========================================================================
    # STEP 5: Save Results
    # ========================================================================
    logger.info("")
    logger.info("STEP 5/5: Saving Results")
    logger.info("-" * 80)

    results_file = config.OUTPUT_DIR / "evaluation_results_ocsvm.json"
    logger.info(f"Saving evaluation results to {results_file}")

    import json
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info("")
    logger.info("=" * 80)
    logger.info("PIPELINE COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {config.OUTPUT_DIR}")
    logger.info("")

    # Comparison with Isolation Forest (if results exist)
    iforest_results_file = config.OUTPUT_DIR / "evaluation_results.json"
    if iforest_results_file.exists():
        logger.info("=" * 80)
        logger.info("COMPARISON: OC-SVM vs Isolation Forest")
        logger.info("=" * 80)

        with open(iforest_results_file, 'r') as f:
            iforest_results = json.load(f)

        logger.info(f"                    Isolation Forest    OC-SVM      Improvement")
        logger.info(f"AUROC:              {iforest_results['auroc']:.4f}             {results['auroc']:.4f}      "
                    f"{(results['auroc'] - iforest_results['auroc']):.4f}")
        logger.info(f"F1 Score:           {iforest_results['best_f1']:.4f}             {results['best_f1']:.4f}      "
                    f"{(results['best_f1'] - iforest_results['best_f1']):.4f}")
        logger.info(f"Score Separation:   {iforest_results['score_stats']['separation']:.4f}             "
                    f"{results['score_stats']['separation']:.4f}      "
                    f"{(results['score_stats']['separation'] - iforest_results['score_stats']['separation']):.4f}")
        logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='OC-SVM Anomaly Detection for Text Difficulty'
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
        '--quick',
        action='store_true',
        help='Quick run (skip tuning and plots)'
    )

    args = parser.parse_args()

    # Quick mode enables no-tuning and no-plots
    if args.quick:
        args.no_tuning = True
        args.no_plots = True

    try:
        main(args)
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        sys.exit(1)
