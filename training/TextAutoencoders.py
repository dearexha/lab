"""
Autoencoders for Text Difficulty Outlier Detection

Three practical approaches for using autoencoders to detect hard texts:

1. Embedding-Space Autoencoder (RECOMMENDED)
   - Train on BERT embeddings of easy texts
   - Reconstruct embeddings (continuous space)
   - Reconstruction error = difficulty score

2. Variational Autoencoder (VAE)
   - Learn probabilistic distribution of easy text embeddings
   - Hard texts have low probability under learned distribution
   - More theoretically grounded

3. Sequence Autoencoder
   - Encode/decode token sequences directly
   - Uses cross-entropy loss on reconstructed tokens
   - More complex but end-to-end

For your use case (train on easy, detect hard), Approach 1 is most practical.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Approach 1: Embedding-Space Autoencoder (RECOMMENDED)
# ============================================================================

class EmbeddingAutoencoder(nn.Module):
    """
    Autoencoder that works in BERT embedding space.

    Architecture:
        Encoder: embedding (768) → hidden (128) → latent (32)
        Decoder: latent (32) → hidden (128) → embedding (768)

    The bottleneck (latent dimension) forces compression, creating
    a low-dimensional representation of easy text characteristics.

    Training: Only on easy texts
    Testing: Reconstruction error on all texts
        - Easy texts: low error (fits learned distribution)
        - Hard texts: high error (doesn't fit)
    """

    def __init__(self, embedding_dim=768, hidden_dim=128, latent_dim=32):
        """
        Args:
            embedding_dim: BERT embedding dimension (768 for base, 128 for yours)
            hidden_dim: Hidden layer dimension
            latent_dim: Bottleneck dimension (compression)
        """
        super().__init__()

        # Encoder: embedding → latent
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )

        # Decoder: latent → embedding
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, x):
        """
        Args:
            x: [batch_size, embedding_dim]

        Returns:
            reconstructed: [batch_size, embedding_dim]
            latent: [batch_size, latent_dim]
        """
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

    def encode(self, x):
        """Get latent representation."""
        return self.encoder(x)

    def decode(self, z):
        """Reconstruct from latent."""
        return self.decoder(z)


def train_embedding_autoencoder(
    model,
    easy_embeddings,
    config,
    device='cuda'
):
    """
    Train autoencoder on easy text embeddings.

    Args:
        model: EmbeddingAutoencoder
        easy_embeddings: numpy array [N_easy, embedding_dim]
        config: dict with 'lr', 'batch_size', 'epochs'
        device: torch device

    Returns:
        trained_model: Trained autoencoder
        training_losses: List of losses per epoch
    """
    model = model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('lr', 1e-3))

    # Convert to tensor dataset
    easy_tensor = torch.FloatTensor(easy_embeddings)
    dataset = torch.utils.data.TensorDataset(easy_tensor)
    dataloader = DataLoader(
        dataset,
        batch_size=config.get('batch_size', 64),
        shuffle=True
    )

    num_epochs = config.get('epochs', 50)
    training_losses = []

    logger.info(f"Training autoencoder on {len(easy_embeddings)} easy embeddings...")
    logger.info(f"Architecture: {easy_embeddings.shape[1]} → {model.encoder[0].out_features} → {model.encoder[3].out_features}")

    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0

        for batch in dataloader:
            embeddings = batch[0].to(device)

            # Forward pass
            reconstructed, latent = model(embeddings)

            # Reconstruction loss (MSE in embedding space)
            loss = F.mse_loss(reconstructed, embeddings)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        training_losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.6f}")

    logger.info("Training complete!")
    return model, training_losses


def compute_reconstruction_error(
    model,
    embeddings,
    device='cuda',
    batch_size=64
):
    """
    Compute reconstruction error for embeddings.

    Args:
        model: Trained EmbeddingAutoencoder
        embeddings: numpy array [N, embedding_dim]
        device: torch device
        batch_size: Batch size for inference

    Returns:
        errors: numpy array [N], reconstruction error per sample
                Higher error = more anomalous = harder text
    """
    model = model.to(device).eval()

    embeddings_tensor = torch.FloatTensor(embeddings)
    dataset = torch.utils.data.TensorDataset(embeddings_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_errors = []

    with torch.no_grad():
        for batch in dataloader:
            emb = batch[0].to(device)
            reconstructed, _ = model(emb)

            # Compute per-sample MSE
            errors = torch.mean((reconstructed - emb) ** 2, dim=1)
            all_errors.append(errors.cpu().numpy())

    return np.concatenate(all_errors)


# ============================================================================
# Approach 2: Variational Autoencoder (VAE)
# ============================================================================

class EmbeddingVAE(nn.Module):
    """
    Variational Autoencoder for text embeddings.

    Instead of deterministic encoding, VAE learns a distribution.

    Encoder: embedding → μ (mean), σ (std)
    Sample: z ~ N(μ, σ)
    Decoder: z → reconstructed embedding

    Loss = Reconstruction Loss + KL Divergence

    Anomaly detection:
    - Easy texts fit learned distribution (low loss)
    - Hard texts don't fit (high loss, especially high KL divergence)
    """

    def __init__(self, embedding_dim=768, hidden_dim=128, latent_dim=32):
        super().__init__()

        # Encoder: embedding → mean and log_variance
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Mean and log-variance heads
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder: latent → embedding
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def encode(self, x):
        """
        Encode to latent distribution parameters.

        Returns:
            mu: [batch_size, latent_dim]
            logvar: [batch_size, latent_dim]
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = μ + σ * ε, where ε ~ N(0,1)

        This allows backpropagation through stochastic sampling.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode from latent."""
        return self.decoder(z)

    def forward(self, x):
        """
        Full forward pass.

        Returns:
            reconstructed: [batch_size, embedding_dim]
            mu: [batch_size, latent_dim]
            logvar: [batch_size, latent_dim]
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar


def vae_loss(reconstructed, original, mu, logvar, beta=1.0):
    """
    VAE loss = Reconstruction Loss + β * KL Divergence

    Args:
        reconstructed: [batch_size, embedding_dim]
        original: [batch_size, embedding_dim]
        mu: [batch_size, latent_dim]
        logvar: [batch_size, latent_dim]
        beta: Weight for KL term (beta-VAE, typically 1.0)

    Returns:
        total_loss: scalar
        recon_loss: scalar
        kl_loss: scalar
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(reconstructed, original, reduction='mean')

    # KL divergence: KL(N(μ,σ²) || N(0,1))
    # Formula: -0.5 * Σ(1 + log(σ²) - μ² - σ²)
    kl_loss = -0.5 * torch.mean(
        torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    )

    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss


def train_vae(
    model,
    easy_embeddings,
    config,
    device='cuda'
):
    """
    Train VAE on easy text embeddings.

    Args:
        model: EmbeddingVAE
        easy_embeddings: numpy array [N_easy, embedding_dim]
        config: dict with 'lr', 'batch_size', 'epochs', 'beta'
        device: torch device

    Returns:
        trained_model: Trained VAE
        losses_dict: Dict with training losses
    """
    model = model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('lr', 1e-3))

    easy_tensor = torch.FloatTensor(easy_embeddings)
    dataset = torch.utils.data.TensorDataset(easy_tensor)
    dataloader = DataLoader(
        dataset,
        batch_size=config.get('batch_size', 64),
        shuffle=True
    )

    num_epochs = config.get('epochs', 50)
    beta = config.get('beta', 1.0)

    losses_dict = {
        'total': [],
        'recon': [],
        'kl': []
    }

    logger.info(f"Training VAE on {len(easy_embeddings)} easy embeddings...")
    logger.info(f"Beta (KL weight): {beta}")

    for epoch in range(num_epochs):
        epoch_total = 0
        epoch_recon = 0
        epoch_kl = 0
        num_batches = 0

        for batch in dataloader:
            embeddings = batch[0].to(device)

            # Forward pass
            reconstructed, mu, logvar = model(embeddings)

            # Compute loss
            total_loss, recon_loss, kl_loss = vae_loss(
                reconstructed, embeddings, mu, logvar, beta
            )

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_total += total_loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()
            num_batches += 1

        # Store losses
        losses_dict['total'].append(epoch_total / num_batches)
        losses_dict['recon'].append(epoch_recon / num_batches)
        losses_dict['kl'].append(epoch_kl / num_batches)

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                       f"Total: {losses_dict['total'][-1]:.6f}, "
                       f"Recon: {losses_dict['recon'][-1]:.6f}, "
                       f"KL: {losses_dict['kl'][-1]:.6f}")

    logger.info("VAE training complete!")
    return model, losses_dict


def compute_vae_anomaly_score(
    model,
    embeddings,
    device='cuda',
    batch_size=64,
    score_type='total'
):
    """
    Compute VAE anomaly scores.

    Args:
        model: Trained EmbeddingVAE
        embeddings: numpy array [N, embedding_dim]
        device: torch device
        batch_size: Batch size
        score_type: 'total' (recon+KL), 'recon' (only reconstruction), 'kl' (only KL)

    Returns:
        scores: numpy array [N], anomaly score per sample
    """
    model = model.to(device).eval()

    embeddings_tensor = torch.FloatTensor(embeddings)
    dataset = torch.utils.data.TensorDataset(embeddings_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_scores = []

    with torch.no_grad():
        for batch in dataloader:
            emb = batch[0].to(device)
            reconstructed, mu, logvar = model(emb)

            if score_type == 'recon':
                # Only reconstruction error
                scores = torch.mean((reconstructed - emb) ** 2, dim=1)

            elif score_type == 'kl':
                # Only KL divergence (per sample)
                kl_per_sample = -0.5 * torch.sum(
                    1 + logvar - mu.pow(2) - logvar.exp(), dim=1
                )
                scores = kl_per_sample

            elif score_type == 'total':
                # Combined score
                recon_error = torch.mean((reconstructed - emb) ** 2, dim=1)
                kl_per_sample = -0.5 * torch.sum(
                    1 + logvar - mu.pow(2) - logvar.exp(), dim=1
                )
                scores = recon_error + kl_per_sample

            all_scores.append(scores.cpu().numpy())

    return np.concatenate(all_scores)


# ============================================================================
# Approach 3: Sequence Autoencoder (Advanced)
# ============================================================================

class SequenceAutoencoder(nn.Module):
    """
    Sequence-to-sequence autoencoder for token sequences.

    Encoder: LSTM/Transformer → context vector
    Decoder: context vector → reconstructed sequence

    Loss: Cross-entropy on reconstructed tokens

    This is the most end-to-end approach but also most complex.
    For your use case, Approach 1 or 2 is more practical.
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.1
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Encoder LSTM
        self.encoder = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Decoder LSTM
        self.decoder = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Output projection
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, target_ids=None):
        """
        Args:
            input_ids: [batch_size, seq_len]
            target_ids: [batch_size, seq_len] (for training)

        Returns:
            logits: [batch_size, seq_len, vocab_size]
            hidden: Encoder hidden state
        """
        # Embed
        embedded = self.embedding(input_ids)  # [batch, seq_len, emb_dim]

        # Encode
        encoder_output, (hidden, cell) = self.encoder(embedded)

        # Decode (teacher forcing during training)
        if target_ids is not None:
            target_embedded = self.embedding(target_ids)
        else:
            target_embedded = embedded  # Autoencoding: decode same sequence

        decoder_output, _ = self.decoder(target_embedded, (hidden, cell))

        # Project to vocabulary
        logits = self.output_projection(decoder_output)

        return logits, hidden


# ============================================================================
# Utilities
# ============================================================================

def plot_reconstruction_errors(
    easy_errors,
    medium_errors,
    hard_errors,
    save_path='reconstruction_errors.png'
):
    """
    Visualize reconstruction error distributions by difficulty.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(easy_errors, bins=50, alpha=0.6, label='Easy', color='green')
    axes[0].hist(medium_errors, bins=50, alpha=0.6, label='Medium', color='orange')
    axes[0].hist(hard_errors, bins=50, alpha=0.6, label='Hard', color='red')
    axes[0].set_xlabel('Reconstruction Error')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Reconstruction Errors')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Box plot
    axes[1].boxplot(
        [easy_errors, medium_errors, hard_errors],
        labels=['Easy', 'Medium', 'Hard']
    )
    axes[1].set_ylabel('Reconstruction Error')
    axes[1].set_title('Reconstruction Error by Difficulty Level')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    logger.info(f"Plot saved to {save_path}")


def evaluate_autoencoder_anomaly_detection(
    easy_scores,
    medium_scores,
    hard_scores
):
    """
    Evaluate how well reconstruction errors separate difficulty levels.

    Returns:
        metrics: dict with evaluation metrics
    """
    from sklearn.metrics import roc_auc_score

    # Combine all scores and labels
    all_scores = np.concatenate([easy_scores, medium_scores, hard_scores])
    all_labels = np.concatenate([
        np.zeros(len(easy_scores)),
        np.ones(len(medium_scores)),
        np.ones(len(hard_scores)) * 2
    ])

    # Binary: easy vs not-easy
    binary_labels = (all_labels > 0).astype(int)
    auroc = roc_auc_score(binary_labels, all_scores)

    # Mean scores by difficulty
    mean_easy = np.mean(easy_scores)
    mean_medium = np.mean(medium_scores)
    mean_hard = np.mean(hard_scores)

    # Check ordering
    correct_ordering = (mean_easy < mean_medium < mean_hard)

    metrics = {
        'auroc_easy_vs_rest': auroc,
        'mean_easy': mean_easy,
        'mean_medium': mean_medium,
        'mean_hard': mean_hard,
        'correct_ordering': correct_ordering
    }

    logger.info("=== Autoencoder Anomaly Detection Evaluation ===")
    logger.info(f"AUROC (easy vs medium/hard): {auroc:.4f}")
    logger.info(f"Mean reconstruction errors:")
    logger.info(f"  Easy:   {mean_easy:.6f}")
    logger.info(f"  Medium: {mean_medium:.6f}")
    logger.info(f"  Hard:   {mean_hard:.6f}")
    logger.info(f"Correct ordering (easy < medium < hard): {correct_ordering}")

    return metrics
