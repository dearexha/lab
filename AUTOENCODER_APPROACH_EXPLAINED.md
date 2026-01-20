# Autoencoders for Text Difficulty Outlier Detection

## Simple Explanation

**Your Goal**: Train on easy texts → Detect hard texts as outliers

**How Autoencoders Work for This**:
1. Train autoencoder ONLY on easy text embeddings
2. Autoencoder learns to compress and reconstruct easy text characteristics
3. Feed medium/hard texts → high reconstruction error (can't reconstruct well)
4. **Reconstruction error = difficulty score**

---

## The Core Workflow

```
┌─────────────────────┐
│  Easy Texts         │
│  (Training Data)    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Extract BERT       │
│  Embeddings         │ → [N_easy, 768] embeddings
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Train Autoencoder  │  ← ONLY on easy embeddings!
│                     │
│  768 → 128 → 32 →  │  (compression bottleneck)
│  → 128 → 768        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Trained            │
│  Autoencoder        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Test on ALL texts  │
│  (easy, med, hard)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Reconstruction     │
│  Errors:            │
│                     │
│  Easy:   0.0012 ✓  │ ← Low error (seen during training)
│  Medium: 0.0045    │
│  Hard:   0.0087 ✓  │ ← High error (outlier!)
└─────────────────────┘
```

---

## Why This Works

### **The Intuition**

Imagine teaching someone to draw only cats:
- Show them 1000 cat pictures → they learn cat features
- Ask them to draw a cat → perfect (low error)
- Ask them to draw a dog → imperfect (high error, they don't know dogs)

Same with autoencoder:
- Train on easy text embeddings → learns "easy text features"
- Reconstruct easy text → low error
- Reconstruct hard text → high error (hasn't seen complex features)

### **The Math**

**Autoencoder Objective**:
```
minimize: ||x - Decoder(Encoder(x))||²

where:
- x = input embedding (easy text)
- Encoder: 768 → 32 (compression)
- Decoder: 32 → 768 (reconstruction)
- Bottleneck forces learning of essential features
```

**At test time**:
```
reconstruction_error = ||x - x̂||²

- Easy texts: error ≈ 0.001 (fits learned distribution)
- Hard texts: error ≈ 0.01 (doesn't fit)
```

---

## Three Approaches (Which to Choose?)

### **1. Embedding-Space Autoencoder (RECOMMENDED)**

**Best for**: Simplicity, reliability, interpretability

```python
# Architecture
Input:  BERT embedding [768]
  ↓
Encoder: 768 → 128 → 32
  ↓
Decoder: 32 → 128 → 768
  ↓
Output: Reconstructed embedding [768]

# Loss
MSE(original_embedding, reconstructed_embedding)
```

**Pros**:
- ✓ Simple to implement
- ✓ Fast to train (~5-10 minutes)
- ✓ Works in continuous space (no discrete token issues)
- ✓ Interpretable (MSE in embedding space)

**Cons**:
- ✗ Requires pre-trained BERT for embeddings
- ✗ Assumes BERT embeddings capture difficulty

**When to use**: Start here! This is the most practical approach.

---

### **2. Variational Autoencoder (VAE)**

**Best for**: Probabilistic modeling, better generalization

```python
# Architecture
Input:  BERT embedding [768]
  ↓
Encoder: 768 → 128 → μ,σ (32 each)
  ↓
Sample: z ~ N(μ, σ)
  ↓
Decoder: z (32) → 128 → 768
  ↓
Output: Reconstructed embedding [768]

# Loss
Reconstruction + β * KL_divergence(N(μ,σ) || N(0,1))
```

**Pros**:
- ✓ Learns distribution, not just point estimates
- ✓ Better for novelty detection (probabilistic)
- ✓ KL divergence captures "how different" from training distribution
- ✓ More theoretically grounded

**Cons**:
- ✗ Slightly more complex
- ✗ Requires tuning β (KL weight)
- ✗ Slower convergence

**When to use**: If standard AE doesn't separate well, or you want probabilistic scores.

---

### **3. Sequence Autoencoder**

**Best for**: End-to-end learning (no BERT needed)

```python
# Architecture
Input:  Token sequence [seq_len]
  ↓
LSTM Encoder → context vector
  ↓
LSTM Decoder → reconstructed sequence
  ↓
Output: Token probabilities [seq_len, vocab_size]

# Loss
CrossEntropy(predicted_tokens, original_tokens)
```

**Pros**:
- ✓ End-to-end, no need for BERT
- ✓ Directly models sequences

**Cons**:
- ✗ Much slower to train
- ✗ Discrete tokens harder to work with
- ✗ Less proven for anomaly detection
- ✗ Overly complex for your use case

**When to use**: Only if you don't have BERT embeddings and want purely autoencoder-based approach.

---

## Practical Recommendations

### **What I Recommend for Your Project**

**Use Approach 1: Embedding-Space Autoencoder**

**Reasons**:
1. You already have BERT (trained on easy texts)
2. BERT embeddings are high-quality representations
3. Simple, fast, interpretable
4. Works reliably

**Architecture I suggest**:
```python
embedding_dim = 128  # Your BERT hidden_size
hidden_dim = 128
latent_dim = 32      # Bottleneck (compression ratio: 4x)

Autoencoder:
  Input: [128]
  Encoder: 128 → 128 → 32
  Decoder: 32 → 128 → 128
  Output: [128]
```

**Training**:
- Loss: MSE
- Optimizer: Adam (lr=1e-3)
- Batch size: 64
- Epochs: 50-100
- **Data: ONLY easy texts**

**Testing**:
```python
# For each text (easy, medium, hard):
error = ||BERT_embedding - autoencoder(BERT_embedding)||²

# Expected:
easy_mean_error ≈ 0.001
hard_mean_error ≈ 0.01  (10x higher)
```

---

## Expected Results

### **If Autoencoders Work Well**

You should see:
```
Reconstruction Error Statistics:
  Easy:   mean=0.0012, std=0.0003  ← Low, tight distribution
  Medium: mean=0.0045, std=0.0015
  Hard:   mean=0.0087, std=0.0031  ← High, wider distribution

AUROC (easy vs hard): 0.85+  ← Good separation

Ordering: easy < medium < hard ✓
```

### **If Autoencoders Don't Work Well**

You might see:
```
Reconstruction Error Statistics:
  Easy:   mean=0.0045, std=0.0025  ← Overlapping distributions
  Medium: mean=0.0048, std=0.0023
  Hard:   mean=0.0051, std=0.0027

AUROC: 0.55  ← Barely better than random

Ordering: ✗ No clear pattern
```

**Why this might happen**:
1. BERT embeddings don't capture difficulty well
2. Autoencoder bottleneck too small/large
3. Easy texts too diverse (hard to compress)
4. Outlier detection not the right approach for this data

**What to do if this happens**:
- Try VAE instead (better for distribution modeling)
- Use Isolation Forest on embeddings (simpler, often better)
- Combine autoencoder error with other metrics (perplexity, FRE)

---

## Comparison: Autoencoder vs Alternatives

| Approach | Complexity | Training Time | Performance (Expected) | When to Use |
|----------|-----------|---------------|----------------------|-------------|
| **Embedding AE** | Low | 5-10 min | Good (0.75-0.85 AUROC) | Start here |
| **VAE** | Medium | 10-20 min | Better (0.80-0.90 AUROC) | If AE doesn't work |
| **Isolation Forest** | Low | <1 min | Good (0.75-0.85 AUROC) | Fast baseline |
| **Distance to Centroid** | Very Low | <10 sec | Decent (0.70-0.80 AUROC) | Simplest baseline |
| **Contrastive Learning** | High | 30-60 min | Best (0.85-0.95 AUROC) | Research contribution |
| **Perplexity** | Low | 0 (use MLM) | Good (0.75-0.85 AUROC) | Already have it |

### **My Honest Assessment**

**Autoencoders for your use case**:
- ✓ Conceptually sound (learn easy distribution, detect outliers)
- ✓ Work well in continuous spaces (embeddings)
- ⚠️ May not beat simpler baselines (Isolation Forest, centroid distance)
- ⚠️ Require tuning (bottleneck size, learning rate, epochs)
- ✗ More complex than necessary for a first approach

**Recommended Strategy**:

1. **Start simple**: Distance to centroid (5 lines of code)
   ```python
   centroid = easy_embeddings.mean(axis=0)
   scores = np.linalg.norm(all_embeddings - centroid, axis=1)
   ```

2. **If that works**: Stick with it! Add Isolation Forest if you want robustness.

3. **If not sufficient**: Try autoencoder (Approach 1)

4. **If you want research novelty**: Use contrastive learning (SupCon)

5. **If you want end-to-end**: Train custom BERT with difficulty prediction head

---

## Integration with Your Curriculum Learning

### **How to Add Autoencoder Scores to CL_DifficultyMeasurer**

You can add autoencoder reconstruction error as a new difficulty metric:

```python
# In CL_DifficultyMeasurer.py

@reg.register(group="dependent", name="autoencoder_reconstruction_error")
def calc_autoencoder_error(self, batch, sent_word_lists):
    """
    Compute reconstruction error from autoencoder trained on easy texts.

    Higher error = harder text = more anomalous.
    """
    if not hasattr(self, 'autoencoder'):
        raise ValueError("Autoencoder not loaded!")

    # Extract BERT embeddings for batch
    embeddings = self.extract_bert_embeddings(batch)

    # Compute reconstruction error
    errors = compute_reconstruction_error(
        self.autoencoder,
        embeddings,
        self.device
    )

    return errors.tolist()
```

### **How to Use in Curriculum Progression**

In your `CurriculumController`:

```python
def should_advance_curriculum(self, val_metrics):
    """
    Decide whether to advance curriculum based on validation metrics.

    If autoencoder reconstruction error on validation set is low,
    model has adapted to current difficulty → advance.
    """
    val_reconstruction_error = val_metrics['autoencoder_error']

    # Threshold: if mean error < threshold, model has adapted
    if val_reconstruction_error < self.reconstruction_threshold:
        logger.info(f"Reconstruction error {val_reconstruction_error:.6f} below "
                   f"threshold {self.reconstruction_threshold:.6f} - advancing!")
        return True

    return False
```

---

## Practical Tips

### **Hyperparameter Tuning**

**Bottleneck size (latent_dim)**:
- Too small (8): May lose important information, high error even on easy texts
- Too large (128): No compression, autoencoder just learns identity, no outlier detection
- **Sweet spot**: embedding_dim / 4 (for 128-dim BERT, use 32)

**Learning rate**:
- Too high (1e-2): Unstable, won't converge
- Too low (1e-5): Very slow, may not converge in reasonable time
- **Sweet spot**: 1e-3 (standard for Adam)

**Training epochs**:
- Too few (10): Underfitting, high error on everything
- Too many (200): Overfitting to easy texts, but this is actually OK for anomaly detection!
- **Sweet spot**: 50-100 epochs, monitor validation loss

**Batch size**:
- Small (16): Noisy gradients, slower
- Large (256): Smoother, but may need more memory
- **Sweet spot**: 64 for your dataset size

### **Debugging Tips**

**If reconstruction error is high for ALL texts** (easy + hard):
- Autoencoder not trained enough → train longer
- Bottleneck too small → increase latent_dim
- Learning rate too high → reduce to 1e-4

**If reconstruction error is same for easy and hard**:
- BERT embeddings don't capture difficulty → try different embedding strategy
- Autoencoder learned identity (bottleneck too large) → reduce latent_dim
- Not enough training data → collect more easy texts

**If reconstruction error is LOW for hard texts**:
- Hard texts not actually different from easy in embedding space
- BERT model not task-appropriate → fine-tune BERT on difficulty task first
- Autoencoder too expressive → add regularization (dropout, smaller capacity)

---

## Code to Get Started

Here's the minimal code to try this NOW:

```python
from training.TextAutoencoders import EmbeddingAutoencoder, train_embedding_autoencoder, compute_reconstruction_error
from training.EmbeddingExtractor import BERTEmbeddingExtractor

# 1. Extract embeddings
extractor = BERTEmbeddingExtractor(bert_model, tokenizer, device)
easy_emb, _ = extractor.extract_dataset_embeddings(easy_dataset)
hard_emb, _ = extractor.extract_dataset_embeddings(hard_dataset)

# 2. Train autoencoder
ae = EmbeddingAutoencoder(embedding_dim=128, hidden_dim=128, latent_dim=32)
ae, losses = train_embedding_autoencoder(
    ae, easy_emb, {'lr': 1e-3, 'batch_size': 64, 'epochs': 50}, device
)

# 3. Compute errors
easy_errors = compute_reconstruction_error(ae, easy_emb, device)
hard_errors = compute_reconstruction_error(ae, hard_emb, device)

# 4. Check if it works
print(f"Easy:  {easy_errors.mean():.6f}")
print(f"Hard:  {hard_errors.mean():.6f}")
print(f"Ratio: {hard_errors.mean() / easy_errors.mean():.2f}x")

# Good result: Ratio > 3x (hard is 3x higher error than easy)
```

---

## Final Recommendation

**For your supervisor discussion**:

1. **Start with simple baseline**: Distance to easy centroid
2. **If that's not sufficient**: Implement embedding autoencoder (Approach 1)
3. **Compare**: Autoencoder vs Isolation Forest vs Perplexity
4. **If you want novelty**: Try contrastive learning (SupCon) instead

**Be honest**:
- Autoencoders are a reasonable approach
- But they may not beat simpler methods
- The real question: "Does it improve curriculum learning performance?"

**The test**:
- Run full curriculum experiment with autoencoder-guided progression
- Compare final model performance vs standard curriculum
- If autoencoder curriculum wins → publish!
- If not → it was worth trying, move on to contrastive learning

Good luck! 🚀
