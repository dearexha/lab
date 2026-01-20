# Contrastive Learning for Text Difficulty Anomaly Detection

## Executive Summary

This document explains **Approach 3: Contrastive Anomaly Detection**, a novel method for detecting hard texts in curriculum learning. This approach has NOT been tested in the AD-NLP paper you read, making it a potential research contribution.

**Key Idea**: Train BERT with contrastive loss on easy texts only → Creates tight embedding cluster → Medium/hard texts fall outside cluster → Distance = difficulty score.

---

## 🎯 Why Contrastive Learning?

### **Fundamental Difference from MLM**

| Aspect | Masked Language Modeling (MLM) | Contrastive Learning |
|--------|-------------------------------|---------------------|
| **Objective** | Predict masked tokens | Make similar samples close, dissimilar far |
| **What it learns** | Token distributions | Discriminative representations |
| **Embedding quality** | By-product of prediction | Primary optimization target |
| **Anomaly detection** | Indirect (via perplexity) | Direct (via distance metrics) |
| **Training signal** | Reconstruction error | Similarity/dissimilarity |

### **The Mathematical Intuition**

**Contrastive loss optimizes two properties**:

1. **Alignment**: Minimize distance between similar samples
   ```
   𝔼[(z_i - z_j)²] → 0  for similar samples
   ```

2. **Uniformity**: Distribute embeddings uniformly on unit sphere
   ```
   log 𝔼[e^(-||z_i - z_j||²)] → maximize
   ```

**Result**: Easy texts form a **compact, dense cluster** in embedding space.

**Why this matters for anomaly detection**:
- Easy texts trained with contrastive loss → **tight, well-defined cluster**
- Hard texts never seen during training → **far from cluster**
- Distance from cluster = **natural anomaly score**

This is more principled than perplexity because:
- Perplexity measures "how well can I predict masked tokens"
- Distance measures "how different is this from what I learned as 'easy'"

The second is a **direct measure of difficulty deviation**.

---

## 📊 Three Contrastive Approaches Explained

### **Approach 1: SimCLR-Style (Unsupervised Augmentation)**

**Concept**: Create positive pairs by augmenting the same text.

```
Original text: "The cat sat on the mat."

Augmentation 1: "[MASK] cat [MASK] on the mat."  (random masking)
Augmentation 2: "The cat sat on [MASK] [MASK]."  (different masking)

Loss: Pull these two augmented views together in embedding space
```

**How it works**:
1. For each easy text, create 2 augmented versions
2. Feed both through BERT → get embeddings z₁, z₂
3. Loss: Make z₁ and z₂ close (positives), far from other texts (negatives)
4. Result: Model learns to ignore augmentation noise, focus on core content

**Augmentation strategies for text**:
- Random token masking (15% like MLM)
- Random token deletion
- Random token shuffling (within small windows)
- Synonym replacement
- Back-translation (expensive but powerful)

**Pros**:
- Unsupervised (doesn't need difficulty labels)
- Learns robust representations invariant to noise

**Cons**:
- Requires careful augmentation design
- Text augmentations are tricky (unlike images)
- May not capture difficulty differences well

---

### **Approach 2: Supervised Contrastive (SupCon) - RECOMMENDED**

**Concept**: Use difficulty labels to define positive/negative pairs.

```
Batch of texts:
- Text A (easy)
- Text B (easy)    ← A and B are positives (same label)
- Text C (medium)  ← C is negative to A,B (different label)

Loss: Pull A and B together, push A and C apart
```

**How it works**:
1. Sample batch of easy texts (all have label=0)
2. Feed through BERT → get embeddings
3. Loss: Make ALL easy text embeddings close to each other
4. Result: All easy texts form tight cluster

**Mathematical formulation**:
```
ℒ_SupCon = -1/|P(i)| Σ log[ exp(z_i·z_p/τ) / Σ exp(z_i·z_a/τ) ]

where:
- P(i) = all samples with same label as sample i
- z_i·z_p = cosine similarity between sample i and positive p
- τ = temperature (0.07 typical)
```

**Why this is perfect for your case**:
- You HAVE difficulty labels (easy/medium/hard)
- Directly optimizes for "all easy texts are similar"
- No augmentation needed
- State-of-the-art performance (Khosla et al., NeurIPS 2020)

**Pros**:
- Leverages labels (more supervision = better)
- Creates very tight clusters for each difficulty level
- No complex augmentation required
- Proven SOTA on many tasks

**Cons**:
- Requires labels (but you have them!)

---

### **Approach 3: Sentence-Pair Contrastive**

**Concept**: Use document structure to create positive pairs.

```
Document D1 (easy):
- Sentence 1: "The cat sat on the mat."
- Sentence 2: "The dog ran in the park."

Positive pair: (S1, S2) from same document
Negative pair: (S1, S3) where S3 from different document
```

**How it works**:
1. Split each document into sentences
2. Sample pairs of sentences from same document (positives)
3. Sample sentences from different documents (negatives)
4. Loss: Pull intra-document sentences together

**Why this works**:
- Sentences in same document share topic, vocabulary, difficulty level
- Model learns to cluster sentences by document characteristics
- Easy documents → tight sentence clusters
- Hard documents → sentences far from easy clusters

**Pros**:
- Exploits document structure
- More fine-grained than document-level
- Can work with long documents

**Cons**:
- Requires sentence segmentation
- More complex data pipeline
- May not work if documents are very short

---

## 🔬 Novel Research Contributions

### **Why This is a Research Contribution**

1. **Gap in Literature**:
   - AD-NLP paper tests: CVDD, DATE, Isolation Forest, OC-SVM
   - **None of these use contrastive learning**
   - Contrastive learning has been SOTA in CV (SimCLR, MoCo, BYOL) but underexplored for text anomaly detection

2. **Natural Fit**:
   - Contrastive learning explicitly optimizes for clustering
   - Curriculum learning has natural clusters (easy/medium/hard)
   - This is more aligned than reconstruction-based methods (autoencoders, MLM)

3. **Better Theoretical Foundation**:
   - Autoencoders: minimize reconstruction error (indirect for anomaly detection)
   - Perplexity: measures prediction quality (indirect for difficulty)
   - **Contrastive**: directly optimizes for "similar = close, different = far"

### **Potential Paper Title**

> **"Contrastive Representation Learning for Text Difficulty Estimation via Anomaly Detection in Curriculum Learning"**

### **Key Claims**:
1. Contrastive learning creates more discriminative embeddings than MLM
2. Distance-based anomaly scores better correlate with text difficulty than perplexity
3. SupCon trained on easy texts only can predict medium/hard difficulty
4. This approach is more robust than autoencoder-based methods

### **Experiments to Validate**:
1. **RQ1**: Do contrastive embeddings cluster better than MLM embeddings?
   - Metric: Silhouette score, Davies-Bouldin index
   - Compare: MLM [CLS], SupCon embeddings

2. **RQ2**: Do anomaly scores correlate with difficulty better than perplexity?
   - Metric: Spearman correlation with ground-truth difficulty
   - Compare: Distance to centroid vs perplexity

3. **RQ3**: Can this improve curriculum learning performance?
   - Metric: Final BERT MLM performance after curriculum
   - Compare: Standard curriculum vs contrastive-guided curriculum

---

## 🆚 Comparison: All Approaches

| Approach | Training Cost | Inference Cost | Interpretability | Performance (Expected) | Novelty |
|----------|--------------|----------------|------------------|----------------------|---------|
| **Perplexity** | Low (use MLM) | Low | High (token-level) | Good baseline | ❌ Standard |
| **Autoencoder** | High (train AE) | Medium | Medium | Unclear (not tested in paper) | ⚠️ Not validated |
| **Embedding + IsolationForest** | Low (use MLM) | Low | High | Good (validated in paper) | ❌ Standard |
| **Contrastive (SupCon)** | Medium (contrastive training) | Low | High (distance-based) | **Very Good** (hypothesis) | ✅ **NOVEL** |

### **Detailed Comparison**

#### **vs. Autoencoders**
| Criterion | Autoencoders | Contrastive Learning |
|-----------|-------------|---------------------|
| **Text compatibility** | Poor (discrete tokens) | Excellent (embeddings) |
| **Loss function** | Reconstruction (MSE) | Similarity (cosine) |
| **Anomaly score** | Reconstruction error | Distance to cluster |
| **Training stability** | Can be unstable | Very stable |
| **Validated in NLP?** | Limited | Yes (SupCon, SimCLR) |

**Verdict**: Contrastive learning is **better suited for text** than autoencoders.

#### **vs. Perplexity**
| Criterion | Perplexity | Contrastive Distance |
|-----------|-----------|---------------------|
| **What it measures** | Prediction quality | Similarity to training distribution |
| **Direct difficulty?** | No (indirect) | Yes (direct) |
| **Optimization** | MLM objective | Clustering objective |
| **Robustness** | Can be noisy | More stable (uses all neighbors) |

**Verdict**: Contrastive is **more principled** for anomaly detection.

#### **vs. Embedding + Isolation Forest**
| Criterion | MLM [CLS] + IsolationForest | SupCon + Distance |
|-----------|---------------------------|------------------|
| **Embedding quality** | Not optimized for discrimination | Optimized for discrimination |
| **Requires external model** | Yes (Isolation Forest) | No (just distance metric) |
| **Training** | MLM (general purpose) | Contrastive (task-specific) |
| **Performance** | Good (validated) | **Better** (hypothesis) |

**Verdict**: Contrastive **should outperform** because embeddings are optimized for the task.

---

## 🚀 Implementation Roadmap

### **Phase 1: Baseline (Week 1)**
1. Implement SupCon loss
2. Train on easy texts only
3. Extract embeddings from easy/medium/hard
4. Compute distance-to-centroid scores
5. Evaluate AUROC vs ground truth labels

**Success metric**: AUROC > 0.7 on easy vs hard classification

### **Phase 2: Optimization (Week 2)**
1. Tune temperature parameter (try 0.05, 0.07, 0.1, 0.5)
2. Experiment with projection head dimensions (64, 128, 256)
3. Try different anomaly scoring methods (centroid, KNN, LOF)
4. Compare with perplexity and Isolation Forest

**Success metric**: Beat perplexity baseline

### **Phase 3: Integration (Week 3)**
1. Add contrastive anomaly score to `CL_DifficultyMeasurer`
2. Modify `CurriculumController` to use scores for progression
3. Run full curriculum learning experiment
4. Compare final model performance: standard vs contrastive-guided

**Success metric**: Improved downstream task performance

### **Phase 4: Analysis & Writing (Week 4)**
1. Visualize embedding space (t-SNE, UMAP)
2. Analyze what model learns (attention patterns, feature importance)
3. Write up results
4. Compare with AD-NLP paper baselines

---

## 📈 Expected Results

### **Hypothesis 1: Better Clustering**
- **Claim**: SupCon embeddings will have tighter clusters than MLM [CLS]
- **Metric**: Silhouette score
- **Expected**: SupCon > 0.6, MLM [CLS] ~ 0.4

### **Hypothesis 2: Better Anomaly Detection**
- **Claim**: Distance scores correlate better with difficulty than perplexity
- **Metric**: Spearman correlation
- **Expected**: Contrastive ρ > 0.8, Perplexity ρ ~ 0.6

### **Hypothesis 3: Score Ordering**
- **Claim**: Anomaly scores will order: easy < medium < hard
- **Metric**: Median scores by difficulty
- **Expected**: Monotonic ordering in >90% of subsets

---

## 🔧 Practical Tips

### **Temperature Selection**
- **τ = 0.05**: Very tight clusters (may overfit)
- **τ = 0.07**: Standard (SimCLR, SupCon default)
- **τ = 0.5**: Looser clusters (more robust to noise)

**Recommendation**: Start with 0.07, tune if needed.

### **Projection Head Dimensionality**
- **64**: Faster, may underfit
- **128**: Standard, good balance
- **256**: More expressive, slower

**Recommendation**: 128 for your hidden_size=128 BERT.

### **Batch Size Importance**
- Contrastive learning benefits from **large batches**
- More negatives per sample = better discrimination
- Minimum: 32, Recommended: 64-256

**If memory limited**: Use gradient accumulation.

### **Training Duration**
- Contrastive learning converges slower than MLM
- Expect: 10-20 epochs vs 3-5 for MLM
- Monitor: Embedding distance between easy texts (should decrease)

---

## 🎓 Theoretical Insight: Why This Works

### **The Geometry of Difficulty**

Easy texts share properties:
- Simple vocabulary → overlapping token distributions
- Short sentences → similar positional encodings
- Common topics → similar semantic content

In BERT embedding space (without contrastive training):
- These similarities exist but are **not emphasized**
- Embeddings spread across space

With contrastive training:
- Loss explicitly **pulls easy texts together**
- Creates **high-density region** for easy characteristics
- Hard texts (complex vocab, long sentences, rare topics) → **low-density outliers**

### **Manifold Hypothesis**

**Claim**: Natural data lies on low-dimensional manifolds in high-dimensional space.

For text difficulty:
- Easy texts → dense manifold region
- Medium texts → sparse region nearby
- Hard texts → far from easy manifold

Contrastive learning **learns this manifold structure** better than MLM because:
1. MLM optimizes for token prediction (tangent to manifold)
2. Contrastive optimizes for manifold geometry directly

---

## 📚 Key References

1. **Supervised Contrastive Learning** (Khosla et al., NeurIPS 2020)
   - Foundation for SupCon loss
   - Shows contrastive > cross-entropy for ImageNet

2. **SimCLR** (Chen et al., ICML 2020)
   - Unsupervised contrastive learning framework
   - Shows importance of: large batches, strong augmentation, projection head

3. **AD-NLP Benchmark** (Bejan et al., EMNLP 2023)
   - Shows Isolation Forest + embeddings works well
   - **Missing**: Contrastive learning approaches

4. **SupCon for NLP** (Gunel et al., NAACL 2021)
   - Applies SupCon to text classification
   - Shows contrastive pre-training improves downstream tasks

---

## 💡 Final Recommendation

### **Go with Supervised Contrastive Learning (SupCon)**

**Reasons**:
1. ✅ You have labels → use them
2. ✅ No complex augmentation needed
3. ✅ State-of-the-art performance
4. ✅ Novel for anomaly detection in curriculum learning
5. ✅ More principled than autoencoders
6. ✅ Better than standard embedding + outlier detector

**Implementation**: Use the code I provided in `ContrastiveLearning.py`

**Timeline**: 2-4 weeks to validate, integrate, and publish

**Impact**: Could be a strong research contribution to both:
- Curriculum learning literature
- Text anomaly detection literature

---

## ❓ Discussion with Supervisor

**Key points to raise**:
1. Contrastive learning is more principled than autoencoders for text
2. Not tested in AD-NLP paper → research gap
3. Natural fit for curriculum learning (explicit clustering)
4. Can combine with existing metrics (perplexity, FRE, etc.)

**Questions to ask**:
1. Should we compare all three (autoencoder, embedding+IF, contrastive)?
2. Is novelty important, or just performance?
3. What evaluation metrics matter most?
4. Target venue: curriculum learning or anomaly detection conference?

Good luck! This is a promising research direction. 🚀
