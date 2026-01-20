"""
Example: Embedding-Based Outlier Detection for Curriculum Learning

This script demonstrates how to use the embedding-based outlier detection approach
as an alternative to autoencoders for detecting hard texts in your curriculum.

Workflow:
1. Train BERT on easy texts only (you already do this)
2. Extract BERT embeddings from all texts
3. Fit Isolation Forest / OC-SVM on easy text embeddings
4. Score medium/hard texts as outliers
5. Use outlier scores for curriculum progression
"""

import torch
import yaml
import logging
from transformers import BertTokenizer, BertForMaskedLM, BertConfig
from datasets import load_from_disk

from training.EmbeddingExtractor import (
    BERTEmbeddingExtractor,
    EmbeddingOutlierDetector,
    add_embedding_outlier_score_to_dataset,
    evaluate_outlier_detection
)
from training.DataPreprocessing import prepare_data
import utils

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # ========================================================================
    # 1. Load Configuration
    # ========================================================================
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Set random seeds
    utils.set_random_seed(config["random_seed"])

    # ========================================================================
    # 2. Load Data
    # ========================================================================
    logger.info("Loading datasets...")

    # Option A: Load your pre-processed datasets
    # dataset_dict = load_from_disk("hf_datasets/OneStopEnglish")
    # full_dataset = dataset_dict['train']

    # Option B: Use your existing data preprocessing
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_dataset, val_dataset, test_dataset = prepare_data(
        config["dataset_name"],
        tokenizer,
        config["max_length"]
    )

    # For this example, we'll use the full training set
    full_dataset = train_dataset

    # Separate by difficulty label
    # Assuming labels: 0=easy, 1=medium, 2=hard (adjust for your data)
    easy_dataset = full_dataset.filter(lambda x: x['label'] == 0)
    medium_dataset = full_dataset.filter(lambda x: x['label'] == 1)
    hard_dataset = full_dataset.filter(lambda x: x['label'] == 2)

    logger.info(f"Dataset sizes: Easy={len(easy_dataset)}, "
                f"Medium={len(medium_dataset)}, Hard={len(hard_dataset)}")

    # ========================================================================
    # 3. Load Your Easy-Trained BERT Model
    # ========================================================================
    # Scenario: You've already trained BERT on easy texts using your curriculum
    # For this example, we'll use a fresh model (in practice, load your checkpoint)

    bert_config = BertConfig(
        vocab_size=config["vocab_size"],
        hidden_size=config["hidden_size"],
        num_hidden_layers=config["num_hidden_layers"],
        num_attention_heads=config["num_attention_heads"],
        intermediate_size=config["intermediate_size"],
        max_position_embeddings=config["max_position_embeddings"]
    )

    # Option 1: Load your saved easy-trained model
    model_path = "path/to/your/easy_trained_bert_checkpoint.pt"
    # model = BertForMaskedLM.from_pretrained(model_path)

    # Option 2: For demo, use a fresh model
    model = BertForMaskedLM(bert_config)

    logger.info("Model loaded successfully")

    # ========================================================================
    # 4. Extract Embeddings
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 1: Extracting BERT Embeddings")
    logger.info("="*80)

    # Initialize extractor
    extractor = BERTEmbeddingExtractor(
        model=model,
        tokenizer=tokenizer,
        device=device,
        pooling_strategy='cls'  # Options: 'cls' or 'mean_pool'
    )

    # Extract embeddings from each difficulty level
    logger.info("Extracting easy text embeddings...")
    easy_embeddings, easy_labels = extractor.extract_dataset_embeddings(
        easy_dataset, batch_size=32
    )

    logger.info("Extracting medium text embeddings...")
    medium_embeddings, medium_labels = extractor.extract_dataset_embeddings(
        medium_dataset, batch_size=32
    )

    logger.info("Extracting hard text embeddings...")
    hard_embeddings, hard_labels = extractor.extract_dataset_embeddings(
        hard_dataset, batch_size=32
    )

    # Combine all embeddings for full dataset scoring
    import numpy as np
    full_embeddings = np.vstack([easy_embeddings, medium_embeddings, hard_embeddings])
    full_labels = np.concatenate([easy_labels, medium_labels, hard_labels])

    logger.info(f"Total embeddings extracted: {full_embeddings.shape[0]}")

    # ========================================================================
    # 5. Fit Outlier Detector on Easy Texts
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 2: Fitting Outlier Detector on Easy Texts Only")
    logger.info("="*80)

    # Option A: Isolation Forest (Recommended - fast, robust)
    detector_if = EmbeddingOutlierDetector(
        method='isolation_forest',
        contamination=0.1,  # Expect ~10% outliers in the data
        n_estimators=100,
        random_state=42
    )
    detector_if.fit(easy_embeddings)

    # Option B: One-Class SVM (Alternative - slower but powerful)
    detector_svm = EmbeddingOutlierDetector(
        method='ocsvm',
        contamination=0.1,
        kernel='rbf',  # Options: 'rbf', 'linear', 'poly'
        gamma='scale'
    )
    detector_svm.fit(easy_embeddings)

    # ========================================================================
    # 6. Score All Texts (Including Medium/Hard)
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 3: Computing Outlier Scores for All Texts")
    logger.info("="*80)

    # Get outlier scores
    scores_if = detector_if.predict_outlier_scores(full_embeddings)
    scores_svm = detector_svm.predict_outlier_scores(full_embeddings)

    # Higher score = more outlier-like = harder text
    logger.info(f"\nIsolation Forest Scores:")
    logger.info(f"  Min: {scores_if.min():.3f}, Max: {scores_if.max():.3f}, "
                f"Mean: {scores_if.mean():.3f}, Std: {scores_if.std():.3f}")

    logger.info(f"\nOC-SVM Scores:")
    logger.info(f"  Min: {scores_svm.min():.3f}, Max: {scores_svm.max():.3f}, "
                f"Mean: {scores_svm.mean():.3f}, Std: {scores_svm.std():.3f}")

    # ========================================================================
    # 7. Evaluate: Do Outlier Scores Correlate with Difficulty?
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 4: Evaluating Outlier Detection Quality")
    logger.info("="*80)

    label_mapping = {0: 'easy', 1: 'medium', 2: 'hard'}

    logger.info("\n--- Isolation Forest Evaluation ---")
    metrics_if = evaluate_outlier_detection(scores_if, full_labels, label_mapping)

    logger.info("\n--- OC-SVM Evaluation ---")
    metrics_svm = evaluate_outlier_detection(scores_svm, full_labels, label_mapping)

    # ========================================================================
    # 8. Visualize Embedding Space (Optional)
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 5: Visualizing Embedding Space (Optional)")
    logger.info("="*80)

    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt

        # Reduce embeddings to 2D using t-SNE
        logger.info("Running t-SNE dimensionality reduction (this may take a while)...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)

        # Sample subset for faster visualization (t-SNE is slow)
        n_samples = min(5000, full_embeddings.shape[0])
        indices = np.random.choice(full_embeddings.shape[0], n_samples, replace=False)

        embeddings_2d = tsne.fit_transform(full_embeddings[indices])

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Color by true difficulty label
        scatter1 = axes[0].scatter(
            embeddings_2d[:, 0], embeddings_2d[:, 1],
            c=full_labels[indices], cmap='viridis', alpha=0.6, s=10
        )
        axes[0].set_title('BERT Embeddings colored by True Difficulty')
        axes[0].set_xlabel('t-SNE Dimension 1')
        axes[0].set_ylabel('t-SNE Dimension 2')
        cbar1 = plt.colorbar(scatter1, ax=axes[0])
        cbar1.set_label('Difficulty (0=easy, 1=medium, 2=hard)')

        # Plot 2: Color by outlier score
        scatter2 = axes[1].scatter(
            embeddings_2d[:, 0], embeddings_2d[:, 1],
            c=scores_if[indices], cmap='coolwarm', alpha=0.6, s=10
        )
        axes[1].set_title('BERT Embeddings colored by Outlier Score (Isolation Forest)')
        axes[1].set_xlabel('t-SNE Dimension 1')
        axes[1].set_ylabel('t-SNE Dimension 2')
        cbar2 = plt.colorbar(scatter2, ax=axes[1])
        cbar2.set_label('Outlier Score (higher = harder)')

        plt.tight_layout()
        plt.savefig('embedding_visualization.png', dpi=150)
        logger.info("Visualization saved to 'embedding_visualization.png'")

    except ImportError:
        logger.warning("sklearn or matplotlib not available, skipping visualization")

    # ========================================================================
    # 9. Integration with Your Curriculum Learning
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 6: Integrating with Curriculum Learning")
    logger.info("="*80)

    # You can now use outlier scores as a difficulty metric in your curriculum
    # Two approaches:

    # Approach A: Add as a new column to your dataset
    full_dataset_with_scores = full_dataset.add_column(
        'embedding_outlier_score',
        scores_if.tolist()
    )

    # Approach B: Use it to filter/sort data for curriculum progression
    # Example: Get top 20% hardest texts according to outlier score
    threshold = np.percentile(scores_if, 80)
    hard_text_indices = np.where(scores_if > threshold)[0]

    logger.info(f"Number of texts above 80th percentile (hardest): {len(hard_text_indices)}")

    # Example: In your CurriculumController, you could use this to decide
    # when to advance to harder data:
    # if mean_outlier_score_on_validation < threshold:
    #     advance_to_next_difficulty_level()

    # ========================================================================
    # 10. Save Detector for Later Use
    # ========================================================================
    detector_if.save('outlier_detector_isolation_forest.pkl')
    logger.info("\nOutlier detector saved. You can reload it later with:")
    logger.info("  detector.load('outlier_detector_isolation_forest.pkl')")

    logger.info("\n" + "="*80)
    logger.info("COMPLETE!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
