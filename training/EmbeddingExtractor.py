"""
Embedding Extractor for Outlier Detection in Curriculum Learning

This module extracts BERT embeddings from text samples and uses them for
outlier detection to determine text difficulty.

Key Idea:
- Train BERT on easy texts only
- Extract embeddings from all texts (easy, medium, hard)
- Apply outlier detection: texts far from easy distribution = harder texts
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import pickle

logger = logging.getLogger(__name__)


class BERTEmbeddingExtractor:
    """
    Extracts embeddings from a BERT model for outlier detection.

    Supports two extraction methods:
    1. [CLS] token: Use the first token embedding (common for classification)
    2. Mean pooling: Average all token embeddings (captures full sequence)
    """

    def __init__(self, model, tokenizer, device, pooling_strategy='cls'):
        """
        Args:
            model: Pre-trained BERT model (trained on easy texts only)
            tokenizer: BERT tokenizer
            device: torch device (cuda/cpu)
            pooling_strategy: 'cls' or 'mean_pool'
        """
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        self.pooling_strategy = pooling_strategy

        if pooling_strategy not in ['cls', 'mean_pool']:
            raise ValueError("pooling_strategy must be 'cls' or 'mean_pool'")

    def extract_embedding(self, input_ids, attention_mask):
        """
        Extract a single embedding vector from BERT hidden states.

        Args:
            input_ids: Tensor [batch_size, seq_len]
            attention_mask: Tensor [batch_size, seq_len]

        Returns:
            embeddings: Tensor [batch_size, hidden_dim]
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

            # Get last hidden state [batch_size, seq_len, hidden_dim]
            hidden_states = outputs.hidden_states[-1]

            if self.pooling_strategy == 'cls':
                # Use [CLS] token (first token) embedding
                embeddings = hidden_states[:, 0, :]  # [batch_size, hidden_dim]

            elif self.pooling_strategy == 'mean_pool':
                # Mean pooling over sequence (excluding padding)
                # Expand attention_mask to match hidden_states shape
                attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()

                # Sum embeddings (masked)
                sum_embeddings = torch.sum(hidden_states * attention_mask_expanded, dim=1)

                # Count non-padding tokens
                sum_mask = attention_mask_expanded.sum(dim=1)
                sum_mask = torch.clamp(sum_mask, min=1e-9)  # Avoid division by zero

                # Average
                embeddings = sum_embeddings / sum_mask  # [batch_size, hidden_dim]

            return embeddings

    def extract_dataset_embeddings(self, dataset, batch_size=32, show_progress=True):
        """
        Extract embeddings for an entire dataset.

        Args:
            dataset: HuggingFace Dataset with 'input_ids' and 'attention_mask'
            batch_size: Batch size for extraction
            show_progress: Show tqdm progress bar

        Returns:
            embeddings: numpy array [N_samples, hidden_dim]
            labels: numpy array [N_samples] (difficulty labels if available)
        """
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_embeddings = []
        all_labels = []

        iterator = tqdm(dataloader, desc="Extracting embeddings") if show_progress else dataloader

        for batch in iterator:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            embeddings = self.extract_embedding(input_ids, attention_mask)
            all_embeddings.append(embeddings.cpu().numpy())

            # Store labels if available (for evaluation)
            if 'label' in batch:
                all_labels.append(batch['label'].cpu().numpy())

        embeddings = np.vstack(all_embeddings)
        labels = np.concatenate(all_labels) if all_labels else None

        logger.info(f"Extracted {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")

        return embeddings, labels


class EmbeddingOutlierDetector:
    """
    Outlier detection on BERT embeddings for curriculum learning.

    Training: Fit on easy texts only
    Inference: Score all texts; high scores = outliers = harder texts
    """

    def __init__(self, method='isolation_forest', contamination=0.1, **kwargs):
        """
        Args:
            method: 'isolation_forest' or 'ocsvm'
            contamination: Expected proportion of outliers (0.1 = 10%)
            **kwargs: Additional parameters for the detector
        """
        self.method = method
        self.contamination = contamination

        if method == 'isolation_forest':
            self.detector = IsolationForest(
                contamination=contamination,
                n_estimators=kwargs.get('n_estimators', 100),
                max_samples=kwargs.get('max_samples', 'auto'),
                random_state=kwargs.get('random_state', 42),
                n_jobs=-1  # Use all CPUs
            )
        elif method == 'ocsvm':
            self.detector = OneClassSVM(
                kernel=kwargs.get('kernel', 'rbf'),
                nu=contamination,  # Upper bound on fraction of outliers
                gamma=kwargs.get('gamma', 'scale')
            )
        else:
            raise ValueError("method must be 'isolation_forest' or 'ocsvm'")

    def fit(self, easy_embeddings):
        """
        Fit the outlier detector on easy text embeddings.

        Args:
            easy_embeddings: numpy array [N_easy, hidden_dim]
        """
        logger.info(f"Fitting {self.method} on {easy_embeddings.shape[0]} easy text embeddings...")
        self.detector.fit(easy_embeddings)
        logger.info("Outlier detector fitted successfully")

    def predict_outlier_scores(self, embeddings):
        """
        Compute outlier scores for embeddings.

        Args:
            embeddings: numpy array [N, hidden_dim]

        Returns:
            scores: numpy array [N]
                - Isolation Forest: negative score = outlier (we flip sign)
                - OC-SVM: negative score = outlier (we flip sign)
                Higher score = more outlier-like = harder text
        """
        if self.method == 'isolation_forest':
            # decision_function: negative score = outlier
            # We flip sign so higher = harder
            scores = -self.detector.decision_function(embeddings)

        elif self.method == 'ocsvm':
            # decision_function: negative = outside margin (outlier)
            # We flip sign so higher = harder
            scores = -self.detector.decision_function(embeddings)

        return scores

    def predict_labels(self, embeddings):
        """
        Binary classification: inlier (easy) vs outlier (hard).

        Args:
            embeddings: numpy array [N, hidden_dim]

        Returns:
            labels: numpy array [N], 1 = inlier (easy), -1 = outlier (hard)
        """
        return self.detector.predict(embeddings)

    def save(self, filepath):
        """Save fitted detector to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.detector, f)
        logger.info(f"Outlier detector saved to {filepath}")

    def load(self, filepath):
        """Load fitted detector from disk."""
        with open(filepath, 'rb') as f:
            self.detector = pickle.load(f)
        logger.info(f"Outlier detector loaded from {filepath}")


# ========================================================================
# Integration with Existing CL_DifficultyMeasurer
# ========================================================================

def add_embedding_outlier_score_to_dataset(dataset, model, tokenizer, device,
                                           easy_dataset, outlier_detector_config):
    """
    Complete pipeline: Extract embeddings → Fit outlier detector → Score all texts.

    This can be used as a new difficulty metric in your CL_DifficultyMeasurer.

    Args:
        dataset: Full dataset (easy + medium + hard)
        model: BERT model trained on easy texts
        tokenizer: BERT tokenizer
        device: torch device
        easy_dataset: Subset of easy texts for fitting outlier detector
        outlier_detector_config: dict with 'method', 'contamination', etc.

    Returns:
        dataset: Updated dataset with new column 'embedding_outlier_score'
    """
    # Step 1: Extract embeddings
    extractor = BERTEmbeddingExtractor(
        model, tokenizer, device,
        pooling_strategy=outlier_detector_config.get('pooling', 'cls')
    )

    logger.info("Extracting embeddings from easy texts...")
    easy_embeddings, _ = extractor.extract_dataset_embeddings(easy_dataset)

    logger.info("Extracting embeddings from full dataset...")
    full_embeddings, labels = extractor.extract_dataset_embeddings(dataset)

    # Step 2: Fit outlier detector on easy embeddings
    detector = EmbeddingOutlierDetector(
        method=outlier_detector_config.get('method', 'isolation_forest'),
        contamination=outlier_detector_config.get('contamination', 0.1),
        **outlier_detector_config.get('kwargs', {})
    )
    detector.fit(easy_embeddings)

    # Step 3: Score all texts
    logger.info("Computing outlier scores for full dataset...")
    outlier_scores = detector.predict_outlier_scores(full_embeddings)

    # Step 4: Add scores to dataset
    dataset = dataset.add_column('embedding_outlier_score', outlier_scores.tolist())

    logger.info(f"Outlier score statistics: min={outlier_scores.min():.3f}, "
                f"max={outlier_scores.max():.3f}, mean={outlier_scores.mean():.3f}")

    return dataset, detector


# ========================================================================
# Standalone Evaluation Script
# ========================================================================

def evaluate_outlier_detection(outlier_scores, true_labels, label_mapping):
    """
    Evaluate how well outlier scores correlate with actual difficulty labels.

    Args:
        outlier_scores: numpy array [N]
        true_labels: numpy array [N] (0=easy, 1=medium, 2=hard)
        label_mapping: dict {0: 'easy', 1: 'medium', 2: 'hard'}

    Returns:
        metrics: dict with evaluation results
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    # Binary: easy (0) vs not-easy (1,2)
    binary_labels = (true_labels > 0).astype(int)

    # Normalize scores to [0, 1]
    scores_normalized = (outlier_scores - outlier_scores.min()) / (outlier_scores.max() - outlier_scores.min())

    metrics = {
        'auroc': roc_auc_score(binary_labels, scores_normalized),
        'avg_precision': average_precision_score(binary_labels, scores_normalized),
    }

    # Score distribution by label
    for label, name in label_mapping.items():
        mask = true_labels == label
        metrics[f'mean_score_{name}'] = outlier_scores[mask].mean()
        metrics[f'std_score_{name}'] = outlier_scores[mask].std()

    logger.info("Outlier Detection Evaluation:")
    logger.info(f"  AUROC (easy vs not-easy): {metrics['auroc']:.3f}")
    logger.info(f"  Avg Precision: {metrics['avg_precision']:.3f}")
    for label, name in label_mapping.items():
        logger.info(f"  {name}: mean={metrics[f'mean_score_{name}']:.3f}, "
                    f"std={metrics[f'std_score_{name}']:.3f}")

    return metrics
