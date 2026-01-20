"""
Complete Example: Autoencoder-Based Outlier Detection for Text Difficulty

This demonstrates how to use autoencoders for your specific use case:
- Train on easy texts only
- Test on all texts (easy, medium, hard)
- Use reconstruction error as difficulty score

Three approaches demonstrated:
1. Embedding-Space Autoencoder (RECOMMENDED - simplest, most reliable)
2. Variational Autoencoder (VAE) (More sophisticated, probabilistic)
3. Comparison with baseline approaches
"""

import torch
import yaml
import logging
import numpy as np
from transformers import BertTokenizer, BertModel, BertConfig

from training.TextAutoencoders import (
    EmbeddingAutoencoder,
    EmbeddingVAE,
    train_embedding_autoencoder,
    train_vae,
    compute_reconstruction_error,
    compute_vae_anomaly_score,
    plot_reconstruction_errors,
    evaluate_autoencoder_anomaly_detection
)
from training.EmbeddingExtractor import BERTEmbeddingExtractor
from training.DataPreprocessing import prepare_data
import utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # ========================================================================
    # 1. Setup
    # ========================================================================
    logger.info("="*80)
    logger.info("AUTOENCODER OUTLIER DETECTION FOR TEXT DIFFICULTY")
    logger.info("="*80)

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    utils.set_random_seed(config["random_seed"])

    # ========================================================================
    # 2. Load Data
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 1: Loading and Preparing Data")
    logger.info("="*80)

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

    logger.info(f"Dataset sizes:")
    logger.info(f"  Easy:   {len(easy_dataset)}")
    logger.info(f"  Medium: {len(medium_dataset)}")
    logger.info(f"  Hard:   {len(hard_dataset)}")

    # ========================================================================
    # 3. Load BERT Model (for embedding extraction)
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 2: Loading BERT Model")
    logger.info("="*80)

    # Option A: Use your existing BERT model trained on easy texts
    # model_path = "path/to/your/easy_trained_bert.pt"
    # bert_model = BertModel.from_pretrained(model_path)

    # Option B: For demo, create a fresh model
    bert_config = BertConfig(
        vocab_size=config["vocab_size"],
        hidden_size=config["hidden_size"],
        num_hidden_layers=config["num_hidden_layers"],
        num_attention_heads=config["num_attention_heads"],
        intermediate_size=config["intermediate_size"],
        max_position_embeddings=config["max_position_embeddings"]
    )
    bert_model = BertModel(bert_config)

    logger.info(f"BERT model loaded (hidden_size={config['hidden_size']})")

    # ========================================================================
    # 4. Extract BERT Embeddings from All Texts
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 3: Extracting BERT Embeddings")
    logger.info("="*80)

    # Use [CLS] token embeddings
    extractor = BERTEmbeddingExtractor(
        model=bert_model,
        tokenizer=tokenizer,
        device=device,
        pooling_strategy='cls'
    )

    logger.info("Extracting embeddings from easy texts...")
    easy_embeddings, _ = extractor.extract_dataset_embeddings(
        easy_dataset, batch_size=32
    )

    logger.info("Extracting embeddings from medium texts...")
    medium_embeddings, _ = extractor.extract_dataset_embeddings(
        medium_dataset, batch_size=32
    )

    logger.info("Extracting embeddings from hard texts...")
    hard_embeddings, _ = extractor.extract_dataset_embeddings(
        hard_dataset, batch_size=32
    )

    logger.info(f"Embedding shape: {easy_embeddings.shape}")

    # ========================================================================
    # 5. Approach 1: Standard Autoencoder
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("APPROACH 1: EMBEDDING-SPACE AUTOENCODER")
    logger.info("="*80)

    # Initialize autoencoder
    embedding_dim = easy_embeddings.shape[1]
    hidden_dim = 128
    latent_dim = 32  # Bottleneck - forces compression

    autoencoder = EmbeddingAutoencoder(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim
    )

    logger.info(f"Architecture: {embedding_dim} → {hidden_dim} → {latent_dim} "
                f"→ {hidden_dim} → {embedding_dim}")

    # Train ONLY on easy text embeddings
    ae_config = {
        'lr': 1e-3,
        'batch_size': 64,
        'epochs': 50
    }

    logger.info(f"\nTraining on EASY texts only ({len(easy_embeddings)} samples)...")
    autoencoder, ae_losses = train_embedding_autoencoder(
        autoencoder,
        easy_embeddings,
        ae_config,
        device
    )

    # Save model
    torch.save(autoencoder.state_dict(), 'autoencoder_easy_trained.pt')
    logger.info("Model saved to 'autoencoder_easy_trained.pt'")

    # Compute reconstruction errors on ALL texts
    logger.info("\nComputing reconstruction errors...")

    easy_errors_ae = compute_reconstruction_error(
        autoencoder, easy_embeddings, device
    )
    medium_errors_ae = compute_reconstruction_error(
        autoencoder, medium_embeddings, device
    )
    hard_errors_ae = compute_reconstruction_error(
        autoencoder, hard_embeddings, device
    )

    logger.info(f"\nReconstruction Error Statistics (Autoencoder):")
    logger.info(f"  Easy:   mean={easy_errors_ae.mean():.6f}, std={easy_errors_ae.std():.6f}")
    logger.info(f"  Medium: mean={medium_errors_ae.mean():.6f}, std={medium_errors_ae.std():.6f}")
    logger.info(f"  Hard:   mean={hard_errors_ae.mean():.6f}, std={hard_errors_ae.std():.6f}")

    # Check if errors increase with difficulty
    if easy_errors_ae.mean() < medium_errors_ae.mean() < hard_errors_ae.mean():
        logger.info("  ✓ Perfect ordering: easy < medium < hard")
    elif easy_errors_ae.mean() < hard_errors_ae.mean():
        logger.info("  ✓ Good separation: easy < hard")
    else:
        logger.info("  ✗ Poor separation - autoencoder may not be working well")

    # Evaluate
    metrics_ae = evaluate_autoencoder_anomaly_detection(
        easy_errors_ae,
        medium_errors_ae,
        hard_errors_ae
    )

    # ========================================================================
    # 6. Approach 2: Variational Autoencoder (VAE)
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("APPROACH 2: VARIATIONAL AUTOENCODER (VAE)")
    logger.info("="*80)

    # Initialize VAE
    vae = EmbeddingVAE(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim
    )

    logger.info(f"VAE Architecture: {embedding_dim} → {hidden_dim} → "
                f"μ,σ ({latent_dim}) → {hidden_dim} → {embedding_dim}")

    # Train on easy texts
    vae_config = {
        'lr': 1e-3,
        'batch_size': 64,
        'epochs': 50,
        'beta': 1.0  # Weight for KL divergence
    }

    logger.info(f"\nTraining VAE on EASY texts only...")
    vae, vae_losses = train_vae(
        vae,
        easy_embeddings,
        vae_config,
        device
    )

    # Save model
    torch.save(vae.state_dict(), 'vae_easy_trained.pt')
    logger.info("VAE saved to 'vae_easy_trained.pt'")

    # Compute anomaly scores
    logger.info("\nComputing VAE anomaly scores...")

    # Try different score types
    for score_type in ['total', 'recon', 'kl']:
        logger.info(f"\nScore type: {score_type}")

        easy_scores_vae = compute_vae_anomaly_score(
            vae, easy_embeddings, device, score_type=score_type
        )
        medium_scores_vae = compute_vae_anomaly_score(
            vae, medium_embeddings, device, score_type=score_type
        )
        hard_scores_vae = compute_vae_anomaly_score(
            vae, hard_embeddings, device, score_type=score_type
        )

        logger.info(f"  Easy:   mean={easy_scores_vae.mean():.6f}")
        logger.info(f"  Medium: mean={medium_scores_vae.mean():.6f}")
        logger.info(f"  Hard:   mean={hard_scores_vae.mean():.6f}")

        if score_type == 'total':
            # Save these for final comparison
            easy_errors_vae = easy_scores_vae
            medium_errors_vae = medium_scores_vae
            hard_errors_vae = hard_scores_vae

    # Evaluate
    metrics_vae = evaluate_autoencoder_anomaly_detection(
        easy_errors_vae,
        medium_errors_vae,
        hard_errors_vae
    )

    # ========================================================================
    # 7. Visualize Results
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 4: Visualizing Results")
    logger.info("="*80)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Autoencoder reconstruction errors
    axes[0, 0].hist(easy_errors_ae, bins=50, alpha=0.6, label='Easy', color='green')
    axes[0, 0].hist(medium_errors_ae, bins=50, alpha=0.6, label='Medium', color='orange')
    axes[0, 0].hist(hard_errors_ae, bins=50, alpha=0.6, label='Hard', color='red')
    axes[0, 0].set_xlabel('Reconstruction Error')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Autoencoder: Reconstruction Error Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Plot 2: VAE anomaly scores
    axes[0, 1].hist(easy_errors_vae, bins=50, alpha=0.6, label='Easy', color='green')
    axes[0, 1].hist(medium_errors_vae, bins=50, alpha=0.6, label='Medium', color='orange')
    axes[0, 1].hist(hard_errors_vae, bins=50, alpha=0.6, label='Hard', color='red')
    axes[0, 1].set_xlabel('VAE Anomaly Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('VAE: Anomaly Score Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Plot 3: Box plots for autoencoder
    axes[1, 0].boxplot(
        [easy_errors_ae, medium_errors_ae, hard_errors_ae],
        labels=['Easy', 'Medium', 'Hard']
    )
    axes[1, 0].set_ylabel('Reconstruction Error')
    axes[1, 0].set_title('Autoencoder: Error by Difficulty')
    axes[1, 0].grid(alpha=0.3)

    # Plot 4: Box plots for VAE
    axes[1, 1].boxplot(
        [easy_errors_vae, medium_errors_vae, hard_errors_vae],
        labels=['Easy', 'Medium', 'Hard']
    )
    axes[1, 1].set_ylabel('VAE Anomaly Score')
    axes[1, 1].set_title('VAE: Score by Difficulty')
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('autoencoder_outlier_detection_results.png', dpi=150)
    logger.info("Visualization saved to 'autoencoder_outlier_detection_results.png'")

    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Autoencoder loss
    axes[0].plot(ae_losses, label='Reconstruction Loss', color='blue')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Autoencoder Training Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # VAE losses
    axes[1].plot(vae_losses['total'], label='Total Loss', color='blue')
    axes[1].plot(vae_losses['recon'], label='Reconstruction', color='green')
    axes[1].plot(vae_losses['kl'], label='KL Divergence', color='red')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('VAE Training Losses')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('autoencoder_training_curves.png', dpi=150)
    logger.info("Training curves saved to 'autoencoder_training_curves.png'")

    # ========================================================================
    # 8. Compare with Baselines
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 5: Comparison with Baselines")
    logger.info("="*80)

    # Baseline 1: Simple distance to centroid (no autoencoder)
    easy_centroid = np.mean(easy_embeddings, axis=0, keepdims=True)

    easy_dist = np.linalg.norm(easy_embeddings - easy_centroid, axis=1)
    medium_dist = np.linalg.norm(medium_embeddings - easy_centroid, axis=1)
    hard_dist = np.linalg.norm(hard_embeddings - easy_centroid, axis=1)

    logger.info("\nBaseline: Distance to Easy Centroid")
    logger.info(f"  Easy:   mean={easy_dist.mean():.6f}")
    logger.info(f"  Medium: mean={medium_dist.mean():.6f}")
    logger.info(f"  Hard:   mean={hard_dist.mean():.6f}")

    metrics_baseline = evaluate_autoencoder_anomaly_detection(
        easy_dist, medium_dist, hard_dist
    )

    # Summary comparison
    logger.info("\n" + "="*80)
    logger.info("FINAL COMPARISON")
    logger.info("="*80)

    comparison_table = f"""
    Method                          | AUROC  | Mean Easy | Mean Hard | Ordering
    --------------------------------|--------|-----------|-----------|----------
    Autoencoder (AE)                | {metrics_ae['auroc_easy_vs_rest']:.4f} | {metrics_ae['mean_easy']:.6f} | {metrics_ae['mean_hard']:.6f} | {'✓' if metrics_ae['correct_ordering'] else '✗'}
    Variational AE (VAE)            | {metrics_vae['auroc_easy_vs_rest']:.4f} | {metrics_vae['mean_easy']:.6f} | {metrics_vae['mean_hard']:.6f} | {'✓' if metrics_vae['correct_ordering'] else '✗'}
    Baseline (Distance to Centroid) | {metrics_baseline['auroc_easy_vs_rest']:.4f} | {metrics_baseline['mean_easy']:.6f} | {metrics_baseline['mean_hard']:.6f} | {'✓' if metrics_baseline['correct_ordering'] else '✗'}
    """

    logger.info(comparison_table)

    # ========================================================================
    # 9. Integration with Your Curriculum Learning
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 6: Integration Guide")
    logger.info("="*80)

    logger.info("""
    How to use autoencoders in your curriculum learning:

    1. TRAINING PHASE:
       a) Train BERT on easy texts (you already do this)
       b) Extract BERT embeddings from easy training set
       c) Train autoencoder on easy embeddings
       d) Save autoencoder checkpoint

    2. CURRICULUM PROGRESSION:
       a) Extract embeddings from validation set
       b) Compute reconstruction error using trained autoencoder
       c) Use error as difficulty metric in CL_DifficultyMeasurer

    3. DECISION MAKING:
       a) Track mean reconstruction error on validation set
       b) When error decreases below threshold → model adapted → advance curriculum
       c) Combine with other metrics (perplexity, FRE, etc.)

    Example in CurriculumController:

        def should_advance_curriculum(self):
            # Get current validation embeddings
            val_embeddings = self.extract_embeddings(val_set)

            # Compute reconstruction error
            errors = compute_reconstruction_error(
                self.autoencoder, val_embeddings
            )

            # If model reconstructs well, it has adapted to current difficulty
            if np.mean(errors) < self.reconstruction_threshold:
                return True  # Advance to harder data
            return False
    """)

    # ========================================================================
    # 10. Save Results
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 7: Saving Results")
    logger.info("="*80)

    np.savez(
        'autoencoder_outlier_results.npz',
        easy_errors_ae=easy_errors_ae,
        medium_errors_ae=medium_errors_ae,
        hard_errors_ae=hard_errors_ae,
        easy_errors_vae=easy_errors_vae,
        medium_errors_vae=medium_errors_vae,
        hard_errors_vae=hard_errors_vae,
        easy_dist=easy_dist,
        medium_dist=medium_dist,
        hard_dist=hard_dist,
        metrics_ae=metrics_ae,
        metrics_vae=metrics_vae,
        metrics_baseline=metrics_baseline
    )

    logger.info("Results saved to 'autoencoder_outlier_results.npz'")

    logger.info("\n" + "="*80)
    logger.info("COMPLETE! Autoencoder Outlier Detection Pipeline Finished")
    logger.info("="*80)

    # Final recommendations
    logger.info("\n" + "="*80)
    logger.info("RECOMMENDATIONS")
    logger.info("="*80)

    best_auroc = max(
        metrics_ae['auroc_easy_vs_rest'],
        metrics_vae['auroc_easy_vs_rest'],
        metrics_baseline['auroc_easy_vs_rest']
    )

    if metrics_ae['auroc_easy_vs_rest'] == best_auroc:
        logger.info("✓ Standard Autoencoder performed best")
        logger.info("  Recommendation: Use EmbeddingAutoencoder for your curriculum")
    elif metrics_vae['auroc_easy_vs_rest'] == best_auroc:
        logger.info("✓ VAE performed best")
        logger.info("  Recommendation: Use EmbeddingVAE for your curriculum")
    else:
        logger.info("✗ Simple baseline performed best")
        logger.info("  Recommendation: Autoencoders may not add value over distance-to-centroid")
        logger.info("  Consider using Isolation Forest (from EmbeddingExtractor.py) instead")


if __name__ == "__main__":
    main()
