"""
Main Pipeline Script
Orchestrates the complete Isolation Forest + GloVe pipeline
"""
import logging
import argparse
import sys
import numpy as np

import config
from data_preparation import prepare_data
from embedding_extraction import prepare_embeddings
from model_training import train_best_model
from evaluation import evaluate_model, create_all_plots

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


def run_pipeline(
    force_reload_data: bool = False,
    force_reload_embeddings: bool = False,
    tune_hyperparameters: bool = True,
    create_plots: bool = True
):
    """
    Run complete anomaly detection pipeline.

    Args:
        force_reload_data: If True, reload data from raw files
        force_reload_embeddings: If True, recompute embeddings
        tune_hyperparameters: If True, perform hyperparameter tuning
        create_plots: If True, create evaluation plots
    """
    logger.info("=" * 70)
    logger.info("ANOMALY DETECTION PIPELINE - ISOLATION FOREST + GLOVE")
    logger.info("=" * 70)
    logger.info(f"Output directory: {config.OUTPUT_DIR}")
    logger.info("")

    # Step 1: Data Preparation
    logger.info("STEP 1/5: Data Preparation")
    logger.info("-" * 70)
    splits = prepare_data(force_reload=force_reload_data)
    logger.info("")

    # Step 2: Embedding Extraction
    logger.info("STEP 2/5: Embedding Extraction")
    logger.info("-" * 70)
    embeddings = prepare_embeddings(splits, force_reload=force_reload_embeddings)
    logger.info("")

    # Step 3: Model Training
    logger.info("STEP 3/5: Model Training")
    logger.info("-" * 70)
    model = train_best_model(
        X_train=embeddings['train'],
        X_val=embeddings['val'],
        X_test=embeddings['test'],
        y_test=np.array(splits['test']['labels']),
        tune_hyperparameters=tune_hyperparameters
    )
    logger.info("")

    # Step 4: Evaluation
    logger.info("STEP 4/5: Evaluation")
    logger.info("-" * 70)
    results = evaluate_model(
        model=model,
        X_test=embeddings['test'],
        y_test=np.array(splits['test']['labels'])
    )
    logger.info("")

    # Step 5: Visualization
    if create_plots:
        logger.info("STEP 5/5: Visualization")
        logger.info("-" * 70)
        create_all_plots(results)
        logger.info("")

    # Summary
    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info("Key Results:")
    logger.info(f"  AUROC: {results['auroc']:.4f}")
    logger.info(f"  Best F1: {results['best_f1']:.4f}")
    logger.info(f"  Precision: {results['precision_at_optimal']:.4f}")
    logger.info(f"  Recall: {results['recall_at_optimal']:.4f}")
    logger.info("")
    logger.info("Output files:")
    logger.info(f"  - Model: {config.BEST_MODEL_FILE}")
    logger.info(f"  - Results: {config.RESULTS_JSON}")
    logger.info(f"  - Hyperparameters: {config.HYPERPARAMETER_RESULTS_JSON}")
    if create_plots:
        logger.info(f"  - Plots: {config.OUTPUT_DIR}")
    logger.info("=" * 70)

    return results


def main():
    """
    Main entry point with command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Anomaly Detection Pipeline: Isolation Forest + GloVe"
    )

    parser.add_argument(
        '--reload-data',
        action='store_true',
        help='Force reload data from raw files'
    )

    parser.add_argument(
        '--reload-embeddings',
        action='store_true',
        help='Force recompute embeddings'
    )

    parser.add_argument(
        '--no-tuning',
        action='store_true',
        help='Skip hyperparameter tuning (use default params)'
    )

    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip plot generation'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode: skip tuning and plots'
    )

    args = parser.parse_args()

    # Quick mode overrides
    if args.quick:
        args.no_tuning = True
        args.no_plots = True
        logger.info("Quick mode enabled: skipping hyperparameter tuning and plots")

    try:
        results = run_pipeline(
            force_reload_data=args.reload_data,
            force_reload_embeddings=args.reload_embeddings,
            tune_hyperparameters=not args.no_tuning,
            create_plots=not args.no_plots
        )
        return 0
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
