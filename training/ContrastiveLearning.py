"""
Contrastive Learning for Text Difficulty Anomaly Detection

This module implements contrastive learning approaches for detecting hard texts
in a curriculum learning setting. The key idea: train on easy texts with contrastive
loss to create tight embedding clusters, then detect hard texts as outliers.

Three approaches:
1. SimCLR-style: Unsupervised, uses data augmentation
2. SupCon: Supervised, uses difficulty labels
3. Sentence-pair: Uses sentences from same document as positives

Novel research contribution: First application of contrastive learning to
curriculum learning difficulty estimation via anomaly detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Contrastive Loss Functions
# ============================================================================

class InfoNCELoss(nn.Module):
    """
    InfoNCE loss for contrastive learning.

    Reference: "A Simple Framework for Contrastive Learning of Visual Representations"
    (SimCLR, Chen et al., 2020)

    Formula:
        ℒ = -log [ exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ) ]

    where z_i and z_j are embeddings of positive pairs (similar samples).
    """

    def __init__(self, temperature=0.07):
        """
        Args:
            temperature: Controls concentration of distribution (lower = tighter clusters)
                        Typical values: 0.07 (SimCLR), 0.1-0.5 (NLP)
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, positive_mask=None):
        """
        Compute InfoNCE loss.

        Args:
            embeddings: [batch_size, embedding_dim]
            positive_mask: [batch_size, batch_size], 1 if samples are positives, 0 otherwise
                          If None, assumes first half and second half are positive pairs

        Returns:
            loss: scalar
        """
        batch_size = embeddings.shape[0]

        # Normalize embeddings to unit sphere (cosine similarity)
        embeddings = F.normalize(embeddings, dim=1)

        # Compute similarity matrix [batch_size, batch_size]
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Create positive mask if not provided
        if positive_mask is None:
            # Assume batch structured as [z1, z2, z3, ..., z1', z2', z3', ...]
            # where zi and zi' are positive pairs
            assert batch_size % 2 == 0, "Batch size must be even for automatic pairing"
            positive_mask = torch.zeros(batch_size, batch_size, device=embeddings.device)
            for i in range(batch_size // 2):
                positive_mask[i, i + batch_size // 2] = 1
                positive_mask[i + batch_size // 2, i] = 1

        # Mask out self-similarity (diagonal)
        self_mask = torch.eye(batch_size, device=embeddings.device).bool()
        similarity_matrix = similarity_matrix.masked_fill(self_mask, -float('inf'))

        # For each sample, positive samples are those with positive_mask = 1
        # Negatives are all others (positive_mask = 0)

        # Compute log-sum-exp of all similarities (denominator)
        denominator = torch.logsumexp(similarity_matrix, dim=1)

        # Compute log of positive similarities (numerator)
        positive_similarities = similarity_matrix * positive_mask

        # For each sample, sum over all positives
        num_positives = positive_mask.sum(dim=1)
        num_positives = torch.clamp(num_positives, min=1)  # Avoid division by zero

        # InfoNCE loss for each sample
        losses = []
        for i in range(batch_size):
            if num_positives[i] > 0:
                # Get positive similarities for sample i
                pos_sims = similarity_matrix[i][positive_mask[i] == 1]
                # log(exp(pos) / sum(exp(all))) = pos - log(sum(exp(all)))
                loss_i = -torch.mean(pos_sims - denominator[i])
                losses.append(loss_i)

        if len(losses) == 0:
            return torch.tensor(0.0, device=embeddings.device)

        return torch.mean(torch.stack(losses))


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.

    Reference: "Supervised Contrastive Learning" (Khosla et al., NeurIPS 2020)

    Key idea: Pull together all samples with same label, push apart different labels.
    Perfect for our use case: all easy texts should cluster together.
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: [batch_size, embedding_dim]
            labels: [batch_size], integer labels (0=easy, 1=medium, 2=hard)

        Returns:
            loss: scalar
        """
        batch_size = embeddings.shape[0]

        # Normalize
        embeddings = F.normalize(embeddings, dim=1)

        # Similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Create label mask: 1 if same label, 0 otherwise
        labels = labels.unsqueeze(1)
        label_mask = torch.eq(labels, labels.T).float()

        # Mask out self-similarity
        self_mask = torch.eye(batch_size, device=embeddings.device)
        label_mask = label_mask * (1 - self_mask)

        # For each anchor, compute SupCon loss
        # Numerator: sum of exp(sim) for all positives (same label)
        # Denominator: sum of exp(sim) for all samples (except self)

        # Mask diagonal
        similarity_matrix_masked = similarity_matrix.masked_fill(
            self_mask.bool(), -float('inf')
        )

        # Denominator: log-sum-exp over all negatives
        denominator = torch.logsumexp(similarity_matrix_masked, dim=1, keepdim=True)

        # Numerator: for each positive, compute log(exp(sim_pos) / denominator)
        log_probs = similarity_matrix - denominator

        # Mask to keep only positives
        log_probs_pos = log_probs * label_mask

        # Count number of positives per sample
        num_positives = label_mask.sum(dim=1)

        # Average over positives (avoid division by zero)
        loss = -torch.sum(log_probs_pos, dim=1) / torch.clamp(num_positives, min=1)

        # Average over batch
        return loss.mean()


# ============================================================================
# Contrastive BERT Model
# ============================================================================

class ContrastiveBERT(nn.Module):
    """
    BERT model with contrastive learning head for anomaly detection.

    Architecture:
        BERT encoder → [CLS] embedding → Projection head → Normalized embedding

    Projection head: Small MLP that maps BERT embeddings to contrastive space.
    This is standard practice in contrastive learning.
    """

    def __init__(self, bert_model, projection_dim=128):
        """
        Args:
            bert_model: Pre-trained or from-scratch BERT model
            projection_dim: Dimension of contrastive embedding space
        """
        super().__init__()
        self.bert = bert_model
        self.hidden_dim = bert_model.config.hidden_size

        # Projection head: BERT hidden → projection_dim
        # Standard in SimCLR: 2-layer MLP with ReLU
        self.projection_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, projection_dim)
        )

    def forward(self, input_ids, attention_mask, return_bert_embedding=False):
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            return_bert_embedding: If True, return BERT [CLS] before projection

        Returns:
            projected_embedding: [batch_size, projection_dim], normalized
            (optional) bert_embedding: [batch_size, hidden_dim]
        """
        # BERT forward pass
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # Get [CLS] token embedding (first token)
        bert_embedding = outputs.hidden_states[-1][:, 0, :]  # [batch_size, hidden_dim]

        # Project to contrastive space
        projected = self.projection_head(bert_embedding)  # [batch_size, projection_dim]

        # Normalize (standard for cosine similarity)
        projected = F.normalize(projected, dim=1)

        if return_bert_embedding:
            return projected, bert_embedding
        return projected


# ============================================================================
# Training Functions
# ============================================================================

def train_contrastive_on_easy_texts(
    model,
    device,
    easy_dataset,
    tokenizer,
    loss_fn,
    config
):
    """
    Train contrastive BERT on easy texts only.

    This creates a tight embedding cluster for easy texts.
    Later, medium/hard texts will be detected as outliers.

    Args:
        model: ContrastiveBERT model
        device: torch device
        easy_dataset: Dataset containing only easy texts
        tokenizer: BERT tokenizer
        loss_fn: InfoNCELoss or SupConLoss
        config: Training config (lr, batch_size, epochs, etc.)

    Returns:
        trained_model: ContrastiveBERT model
    """
    model = model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get('lr', 1e-4))

    # DataLoader
    dataloader = DataLoader(
        easy_dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=True
    )

    num_epochs = config.get('num_epochs', 10)

    logger.info(f"Training contrastive model on {len(easy_dataset)} easy texts...")
    logger.info(f"Epochs: {num_epochs}, Batch size: {config.get('batch_size', 32)}")

    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Get embeddings
            embeddings = model(input_ids, attention_mask)

            # Compute loss
            if isinstance(loss_fn, SupConLoss):
                # Need labels for supervised contrastive loss
                labels = batch['label'].to(device)
                loss = loss_fn(embeddings, labels)
            else:
                # InfoNCE: assume positive pairs are created by augmentation
                # (implementation depends on your data augmentation strategy)
                loss = loss_fn(embeddings)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f}")

    logger.info("Contrastive training complete!")
    return model


# ============================================================================
# Anomaly Detection Functions
# ============================================================================

def extract_contrastive_embeddings(model, device, dataset, batch_size=32):
    """
    Extract normalized contrastive embeddings from trained model.

    Args:
        model: Trained ContrastiveBERT
        device: torch device
        dataset: Dataset to extract embeddings from
        batch_size: Batch size

    Returns:
        embeddings: [N, projection_dim], normalized
        labels: [N], difficulty labels (if available)
    """
    model = model.to(device).eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            embeddings = model(input_ids, attention_mask)
            all_embeddings.append(embeddings.cpu().numpy())

            if 'label' in batch:
                all_labels.append(batch['label'].cpu().numpy())

    embeddings = np.vstack(all_embeddings)
    labels = np.concatenate(all_labels) if all_labels else None

    return embeddings, labels


def compute_anomaly_scores_contrastive(
    easy_embeddings,
    test_embeddings,
    method='distance_to_centroid'
):
    """
    Compute anomaly scores based on contrastive embeddings.

    Three methods:
    1. 'distance_to_centroid': Distance from test sample to easy cluster centroid
    2. 'knn_distance': Average distance to K nearest easy neighbors
    3. 'density': Local density estimation (fewer neighbors = outlier)

    Args:
        easy_embeddings: [N_easy, dim], embeddings from easy texts
        test_embeddings: [N_test, dim], embeddings to score
        method: Anomaly scoring method

    Returns:
        scores: [N_test], higher = more anomalous = harder text
    """
    if method == 'distance_to_centroid':
        # Compute centroid of easy embeddings
        centroid = np.mean(easy_embeddings, axis=0, keepdims=True)  # [1, dim]

        # Distance from each test sample to centroid
        # Using Euclidean distance (cosine would be 1 - dot product)
        scores = np.linalg.norm(test_embeddings - centroid, axis=1)

    elif method == 'knn_distance':
        # Average distance to K nearest easy neighbors
        from sklearn.neighbors import NearestNeighbors

        k = min(5, len(easy_embeddings))  # Use 5 nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(easy_embeddings)
        distances, _ = nbrs.kneighbors(test_embeddings)

        # Average distance to K neighbors
        scores = np.mean(distances, axis=1)

    elif method == 'density':
        # Local Outlier Factor (LOF) style
        from sklearn.neighbors import LocalOutlierFactor

        lof = LocalOutlierFactor(n_neighbors=5, novelty=True)
        lof.fit(easy_embeddings)

        # Negative LOF score (we flip so higher = more anomalous)
        scores = -lof.score_samples(test_embeddings)

    else:
        raise ValueError(f"Unknown method: {method}")

    return scores


# ============================================================================
# Data Augmentation for Contrastive Learning (Text-specific)
# ============================================================================

class TextAugmentation:
    """
    Text augmentation strategies for contrastive learning.

    Create positive pairs by augmenting same text in different ways.
    Standard augmentations:
    1. Random token masking (dropout)
    2. Random token shuffling (within short windows)
    3. Synonym replacement
    4. Back-translation (expensive, but powerful)
    """

    @staticmethod
    def random_token_mask(input_ids, mask_prob=0.15, mask_token_id=103):
        """
        Randomly mask tokens (similar to MLM, but for augmentation).

        Args:
            input_ids: [seq_len]
            mask_prob: Probability of masking each token
            mask_token_id: ID for [MASK] token

        Returns:
            masked_input_ids: [seq_len]
        """
        input_ids = input_ids.clone()
        mask = torch.rand(input_ids.shape) < mask_prob

        # Don't mask special tokens [CLS], [SEP], [PAD]
        special_tokens_mask = (input_ids == 101) | (input_ids == 102) | (input_ids == 0)
        mask = mask & ~special_tokens_mask

        input_ids[mask] = mask_token_id
        return input_ids

    @staticmethod
    def random_token_shuffle(input_ids, window_size=3):
        """
        Shuffle tokens within small windows (preserves some order).

        Args:
            input_ids: [seq_len]
            window_size: Size of shuffling window

        Returns:
            shuffled_input_ids: [seq_len]
        """
        input_ids = input_ids.clone()
        seq_len = input_ids.shape[0]

        for i in range(1, seq_len - 1, window_size):  # Skip [CLS] and [SEP]
            end = min(i + window_size, seq_len - 1)
            if end - i > 1:
                perm = torch.randperm(end - i)
                input_ids[i:end] = input_ids[i:end][perm]

        return input_ids


# ============================================================================
# Contrastive Dataset Wrapper
# ============================================================================

class ContrastiveDataset(Dataset):
    """
    Wraps a dataset to return positive pairs for contrastive learning.

    Two strategies:
    1. Augmentation-based: Apply augmentations to create pairs
    2. Sentence-pair: Sample two sentences from same document
    """

    def __init__(self, base_dataset, strategy='augmentation', tokenizer=None):
        """
        Args:
            base_dataset: Original dataset
            strategy: 'augmentation' or 'sentence_pair'
            tokenizer: Tokenizer (needed for augmentation)
        """
        self.base_dataset = base_dataset
        self.strategy = strategy
        self.tokenizer = tokenizer
        self.augmenter = TextAugmentation()

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]

        if self.strategy == 'augmentation':
            # Create two augmented views of same text
            input_ids_1 = self.augmenter.random_token_mask(
                torch.tensor(item['input_ids'])
            )
            input_ids_2 = self.augmenter.random_token_mask(
                torch.tensor(item['input_ids'])
            )

            return {
                'input_ids_1': input_ids_1,
                'attention_mask_1': item['attention_mask'],
                'input_ids_2': input_ids_2,
                'attention_mask_2': item['attention_mask'],
                'label': item.get('label', 0)
            }

        elif self.strategy == 'sentence_pair':
            # This would require splitting documents into sentences
            # and sampling pairs - implementation depends on your data format
            pass

        return item
