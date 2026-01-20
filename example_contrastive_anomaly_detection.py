"""
Example: Contrastive Learning for Curriculum Text Difficulty Detection

This demonstrates a NOVEL approach: using contrastive learning to detect hard texts
as anomalies by training on easy texts only.

Research Contribution:
- First application of contrastive learning to curriculum difficulty estimation
- Combines SupCon loss with anomaly detection for NLP
- More principled than perplexity-based approaches

Workflow:
1. Train contrastive BERT on easy texts (SupCon loss pushes easy texts together)
2. Extract contrastive embeddings from all texts
3. Compute anomaly scores (distance from easy cluster)
4. Hard texts = high anomaly scores = curriculum progression signal
"""

import torch
import yaml
import logging
import numpy as np
from transformers import BertTokenizer, BertModel, BertConfig
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from training.ContrastiveLearning import (
    ContrastiveBERT,
    SupConLoss,
    InfoNCELoss,
    train_contrastive_on_easy_texts,
    extract_contrastive_embeddings,
    compute_anomaly_scores_contrastive
)
from training.DataPreprocessing import prepare_data
import utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # ========================================================================
    # 1. Setup
    # ========================================================================
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    utils.set_random_seed(config["random_seed"])

    # ========================================================================
    # 2. Load Data
    # ========================================================================
    logger.info("Loading datasets...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_dataset, val_dataset, test_dataset = prepare_data(
        config["dataset_name"],
        tokenizer,
        config["max_length"]
    )

    # Separate by difficulty
    easy_dataset = train_dataset.filter(lambda x: x['label'] == 0)
    medium_dataset = train_dataset.filter(lambda x: x['label'] == 1)
    hard_dataset = train_dataset.filter(lambda x: x['label'] == 2)

    logger.info(f"Easy: {len(easy_dataset)}, Medium: {len(medium_dataset)}, "
                f"Hard: {len(hard_dataset)}")

    # ========================================================================
    # 3. Initialize Contrastive BERT
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 1: Initializing Contrastive BERT")
    logger.info("="*80)

    # Option A: Use your existing BERT config
    bert_config = BertConfig(
        vocab_size=config["vocab_size"],
        hidden_size=config["hidden_size"],
        num_hidden_layers=config["num_hidden_layers"],
        num_attention_heads=config["num_attention_heads"],
        intermediate_size=config["intermediate_size"],
        max_position_embeddings=config["max_position_embeddings"]
    )
    bert_base = BertModel(bert_config)

    # Option B: Use pre-trained BERT (transfer learning)
    # bert_base = BertModel.from_pretrained("bert-base-uncased")

    # Wrap with contrastive head
    projection_dim = 128  # Dimensionality of contrastive space
    model = ContrastiveBERT(bert_base, projection_dim=projection_dim)

    logger.info(f"Model initialized: BERT hidden_dim={config['hidden_size']}, "
                f"Projection dim={projection_dim}")

    # ========================================================================
    # 4. Choose Contrastive Loss
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 2: Selecting Contrastive Loss Function")
    logger.info("="*80)

    # Option A: Supervised Contrastive Loss (RECOMMENDED for labeled data)
    # This pulls all easy texts together in embedding space
    loss_fn = SupConLoss(temperature=0.07)
    logger.info("Using Supervised Contrastive Loss (SupCon)")
    logger.info("  - Pulls together all samples with same difficulty label")
    logger.info("  - Creates tight cluster for easy texts")

    # Option B: InfoNCE (if you want unsupervised approach with augmentation)
    # loss_fn = InfoNCELoss(temperature=0.07)
    # logger.info("Using InfoNCE Loss")
    # logger.info("  - Requires data augmentation to create positive pairs")

    # ========================================================================
    # 5. Train on Easy Texts Only
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 3: Training Contrastive Model on Easy Texts ONLY")
    logger.info("="*80)

    training_config = {
        'lr': 1e-4,
        'batch_size': 32,
        'num_epochs': 10,
    }

    logger.info(f"Training config: {training_config}")
    logger.info(f"Training ONLY on {len(easy_dataset)} easy texts...")

    model = train_contrastive_on_easy_texts(
        model=model,
        device=device,
        easy_dataset=easy_dataset,
        tokenizer=tokenizer,
        loss_fn=loss_fn,
        config=training_config
    )

    # Save model
    torch.save(model.state_dict(), 'contrastive_bert_easy_trained.pt')
    logger.info("Model saved to 'contrastive_bert_easy_trained.pt'")

    # ========================================================================
    # 6. Extract Embeddings from All Texts
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 4: Extracting Contrastive Embeddings")
    logger.info("="*80)

    logger.info("Extracting from easy texts...")
    easy_embeddings, easy_labels = extract_contrastive_embeddings(
        model, device, easy_dataset, batch_size=32
    )

    logger.info("Extracting from medium texts...")
    medium_embeddings, medium_labels = extract_contrastive_embeddings(
        model, device, medium_dataset, batch_size=32
    )

    logger.info("Extracting from hard texts...")
    hard_embeddings, hard_labels = extract_contrastive_embeddings(
        model, device, hard_dataset, batch_size=32
    )

    # Combine
    all_embeddings = np.vstack([easy_embeddings, medium_embeddings, hard_embeddings])
    all_labels = np.concatenate([easy_labels, medium_labels, hard_labels])

    logger.info(f"Total embeddings: {all_embeddings.shape}")

    # ========================================================================
    # 7. Compute Anomaly Scores
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 5: Computing Anomaly Scores")
    logger.info("="*80)

    # Test all three methods
    methods = ['distance_to_centroid', 'knn_distance', 'density']

    results = {}
    for method in methods:
        logger.info(f"\nMethod: {method}")

        scores = compute_anomaly_scores_contrastive(
            easy_embeddings=easy_embeddings,
            test_embeddings=all_embeddings,
            method=method
        )

        # Analyze scores by difficulty level
        easy_scores = scores[:len(easy_embeddings)]
        medium_scores = scores[len(easy_embeddings):len(easy_embeddings)+len(medium_embeddings)]
        hard_scores = scores[len(easy_embeddings)+len(medium_embeddings):]

        logger.info(f"  Easy:   mean={easy_scores.mean():.4f}, std={easy_scores.std():.4f}")
        logger.info(f"  Medium: mean={medium_scores.mean():.4f}, std={medium_scores.std():.4f}")
        logger.info(f"  Hard:   mean={hard_scores.mean():.4f}, std={hard_scores.std():.4f}")

        results[method] = {
            'scores': scores,
            'easy': easy_scores,
            'medium': medium_scores,
            'hard': hard_scores
        }

    # ========================================================================
    # 8. Evaluate Performance
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 6: Evaluating Anomaly Detection Performance")
    logger.info("="*80)

    from sklearn.metrics import roc_auc_score, average_precision_score

    for method, res in results.items():
        logger.info(f"\n--- {method} ---")

        scores = res['scores']

        # Binary classification: easy vs non-easy
        binary_labels = (all_labels > 0).astype(int)

        # Normalize scores
        scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

        auroc = roc_auc_score(binary_labels, scores_norm)
        avg_prec = average_precision_score(binary_labels, scores_norm)

        logger.info(f"  AUROC (easy vs medium/hard): {auroc:.4f}")
        logger.info(f"  Average Precision: {avg_prec:.4f}")

        # Separation: check if medium/hard have higher scores than easy
        easy_median = np.median(res['easy'])
        medium_median = np.median(res['medium'])
        hard_median = np.median(res['hard'])

        logger.info(f"  Median scores: Easy={easy_median:.4f}, "
                    f"Medium={medium_median:.4f}, Hard={hard_median:.4f}")

        # Good separation if: easy < medium < hard
        if easy_median < medium_median < hard_median:
            logger.info("  ✓ Perfect ordering: easy < medium < hard")
        elif easy_median < hard_median:
            logger.info("  ✓ Good separation: easy < hard")
        else:
            logger.info("  ✗ Poor separation")

    # ========================================================================
    # 9. Visualization
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 7: Visualizing Embedding Space")
    logger.info("="*80)

    try:
        from sklearn.manifold import TSNE

        logger.info("Running t-SNE (this may take a while)...")

        # Sample for faster visualization
        n_samples = min(3000, all_embeddings.shape[0])
        indices = np.random.choice(all_embeddings.shape[0], n_samples, replace=False)

        embeddings_2d = TSNE(
            n_components=2,
            random_state=42,
            perplexity=30
        ).fit_transform(all_embeddings[indices])

        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        # Plot 1: Color by true difficulty
        scatter1 = axes[0, 0].scatter(
            embeddings_2d[:, 0], embeddings_2d[:, 1],
            c=all_labels[indices], cmap='viridis', alpha=0.6, s=20
        )
        axes[0, 0].set_title('Contrastive Embeddings: True Difficulty Labels', fontsize=14)
        axes[0, 0].set_xlabel('t-SNE 1')
        axes[0, 0].set_ylabel('t-SNE 2')
        cbar1 = plt.colorbar(scatter1, ax=axes[0, 0])
        cbar1.set_label('Difficulty (0=easy, 1=medium, 2=hard)')

        # Plot 2: Color by distance to centroid
        method = 'distance_to_centroid'
        scatter2 = axes[0, 1].scatter(
            embeddings_2d[:, 0], embeddings_2d[:, 1],
            c=results[method]['scores'][indices],
            cmap='coolwarm', alpha=0.6, s=20
        )
        axes[0, 1].set_title('Anomaly Scores: Distance to Centroid', fontsize=14)
        axes[0, 1].set_xlabel('t-SNE 1')
        axes[0, 1].set_ylabel('t-SNE 2')
        cbar2 = plt.colorbar(scatter2, ax=axes[0, 1])
        cbar2.set_label('Anomaly Score (higher = harder)')

        # Plot 3: Score distributions by difficulty
        axes[1, 0].hist(results[method]['easy'], bins=30, alpha=0.5, label='Easy', color='green')
        axes[1, 0].hist(results[method]['medium'], bins=30, alpha=0.5, label='Medium', color='orange')
        axes[1, 0].hist(results[method]['hard'], bins=30, alpha=0.5, label='Hard', color='red')
        axes[1, 0].set_xlabel('Anomaly Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Score Distribution by Difficulty', fontsize=14)
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

        # Plot 4: KNN distance scores
        method = 'knn_distance'
        axes[1, 1].hist(results[method]['easy'], bins=30, alpha=0.5, label='Easy', color='green')
        axes[1, 1].hist(results[method]['medium'], bins=30, alpha=0.5, label='Medium', color='orange')
        axes[1, 1].hist(results[method]['hard'], bins=30, alpha=0.5, label='Hard', color='red')
        axes[1, 1].set_xlabel('KNN Distance Score')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('KNN Distance Score Distribution', fontsize=14)
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('contrastive_anomaly_detection_results.png', dpi=150)
        logger.info("Visualization saved to 'contrastive_anomaly_detection_results.png'")

    except ImportError as e:
        logger.warning(f"Visualization skipped: {e}")

    # ========================================================================
    # 10. Integration with Curriculum Learning
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 8: How to Integrate with Your Curriculum Learning")
    logger.info("="*80)

    logger.info("""
    Integration Strategy:

    1. TRAINING PHASE:
       - Train BERT with contrastive loss on easy texts only
       - This creates tight cluster representation of "easy text characteristics"

    2. DIFFICULTY SCORING:
       - Extract contrastive embeddings for all texts (easy, medium, hard)
       - Compute anomaly scores (distance to easy cluster centroid)
       - Use scores as difficulty metric in CL_DifficultyMeasurer

    3. CURRICULUM PROGRESSION:
       - In CurriculumController, track anomaly scores on validation set
       - When model performance improves, anomaly scores on medium texts ↓
       - This signals model is ready for harder curriculum level

    4. ADVANTAGES:
       - Direct optimization for discrimination (contrastive loss)
       - More principled than perplexity (which measures reconstruction)
       - Can combine with existing metrics (FRE, word_rarity, etc.)

    Example in your CurriculumController:

        def should_advance_curriculum(self):
            # Get anomaly scores on current validation set
            val_anomaly_score = self.compute_mean_anomaly_score(val_set)

            # If anomaly score drops below threshold, advance
            if val_anomaly_score < self.anomaly_threshold:
                return True
            return False
    """)

    # ========================================================================
    # 11. Save Results
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 9: Saving Results")
    logger.info("="*80)

    # Save embeddings and scores for later analysis
    np.savez(
        'contrastive_anomaly_results.npz',
        easy_embeddings=easy_embeddings,
        medium_embeddings=medium_embeddings,
        hard_embeddings=hard_embeddings,
        easy_labels=easy_labels,
        medium_labels=medium_labels,
        hard_labels=hard_labels,
        scores_centroid=results['distance_to_centroid']['scores'],
        scores_knn=results['knn_distance']['scores'],
        scores_density=results['density']['scores']
    )

    logger.info("Results saved to 'contrastive_anomaly_results.npz'")

    logger.info("\n" + "="*80)
    logger.info("COMPLETE! Contrastive Anomaly Detection Pipeline Finished")
    logger.info("="*80)


if __name__ == "__main__":
    main()
