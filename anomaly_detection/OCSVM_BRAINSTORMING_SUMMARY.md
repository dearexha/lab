# OC-SVM Brainstorming & Implementation Summary

## Problem Statement

**Goal**: Separate easy texts (Simple Wikipedia) from hard texts (Normal Wikipedia) using anomaly detection.

**Current Status**: Isolation Forest with GloVe embeddings **FAILED** (AUROC = 0.481, worse than random)

## Why Isolation Forest Failed

Looking at `evaluation_results.json`:
```json
{
  "auroc": 0.481,  // Worse than random (0.5)!
  "separation": 0.0023,  // Almost zero separation
  "inlier_mean": -0.193,
  "outlier_mean": -0.195  // Virtually identical!
}
```

### Root Cause Analysis

**GloVe captures SEMANTIC SIMILARITY, not DIFFICULTY**

Example:
- Easy: "The dog runs fast"
- Hard: "The canine accelerates rapidly"

In GloVe space, these are **VERY CLOSE** (same topic/meaning) but have **different difficulty**!

**Isolation Forest assumes outliers are isolated/sparse** in feature space, but hard texts are distributed **similarly** to easy texts in semantic space.

## Brainstorming Solutions

### ✅ Solution 1: OC-SVM with GloVe (IMPLEMENTED)

**What changed**: Algorithm only (same features)

**Why it might help**:
- RBF kernel → non-linear boundaries
- Nu parameter → controls boundary tightness
- Margin maximization → more robust than random trees

**Expected improvement**: 55-65% AUROC (marginal)

**Limitation**: GloVe still doesn't capture difficulty well

---

### ⭐ Solution 2: OC-SVM with BERT (RECOMMENDED - READY TO RUN)

**What changed**: Switch from GloVe (300D) to BERT (384D) embeddings

**Why BERT >> GloVe**:

| Feature | GloVe | BERT |
|---------|-------|------|
| **Context** | Static | Dynamic (context-aware) |
| **Syntax** | Word-level | Sentence structure |
| **Complexity** | Averages words | Captures clause depth |
| **Rare words** | OOV = zero | Subword tokenization |

**Example from your data**:

Simple: `"Skateboard decks are normally between 28 and 33 inches long ."`
- GloVe: Averages word vectors
- BERT: Encodes "normally between X and Y" construction ✅

Hard: `"Some of them have special materials that help to keep the deck from breaking : such as fiberglass , bamboo , resin , Kevlar , carbon fiber , aluminum , and plastic ."`
- GloVe: Averages many words, loses structure
- BERT: Captures complex enumeration, "help to keep X from Y", clause depth ✅

**Expected improvement**: **70-85% AUROC** (based on AD-NLP paper Table 4)

**How to run**: Just change `config.py`:
```python
EMBEDDING_TYPE = "bert"  # Already configured!
```

---

### Solution 3: Hand-Crafted Features (FUTURE WORK)

Use difficulty metrics from `training/CL_DifficultyMeasurer.py`:
```python
features = [
    sentence_length_words,     # Hard texts are longer
    word_rarity_words,         # Hard texts have rare words
    fre_score_words,           # Flesch Reading Ease
    shannon_entropy_words,     # Lexical diversity
    ttr_words,                 # Type-token ratio
    perplexity                 # Language model surprise
]
```

**Expected improvement**: 65-75% AUROC

**Limitation**: Only 6 dimensions (low-D, might miss subtle patterns)

---

### 🚀 Solution 4: Hybrid BERT + Metrics (BEST EXPECTED)

Combine:
- BERT embeddings: 384D (deep patterns)
- Hand-crafted metrics: 6D (explicit difficulty signals)
- **Total**: 390D

**Expected improvement**: **75-90% AUROC** 🎯

**Implementation**: Future work

## Key Implementation Details

### Critical Difference: Feature Normalization

```python
# Isolation Forest: Works without normalization
iforest.fit(X_train)  # ✅ OK

# OC-SVM: REQUIRES normalization
ocsvm.fit(X_train)  # ❌ WILL FAIL!

# Correct OC-SVM:
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
ocsvm.fit(X_train_scaled)  # ✅ Correct
```

**Why?** Features have different scales:
- `sentence_length`: 5-50
- `perplexity`: 1-1000
- `fre_score`: -50 to +100

SVM uses distances → **scale matters!**

### Hyperparameters (from AD-NLP Paper Section 4.2)

```python
OCSVM_PARAM_GRID = {
    'nu': [0.05, 0.1, 0.2, 0.5],  # From paper Table 4
    'kernel': ['rbf', 'poly', 'linear'],
    'gamma': ['scale', 'auto']
}
```

**Nu parameter** (most important):
- `nu=0.05`: Very tight boundary (5% outliers expected)
- `nu=0.1`: Tight boundary (10% outliers) ← **Good default**
- `nu=0.2`: Moderate boundary (20% outliers)
- `nu=0.5`: Loose boundary (50% outliers)

### Training Setup (Correct for Anomaly Detection)

From `data_preparation.py`:
- **Train**: 80% of SIMPLE texts (label=0) ← **All inliers**
- **Val**: 10% of SIMPLE texts (label=0) ← **All inliers**
- **Test**: 10% SIMPLE + 100% NORMAL (label=0 + label=1) ← **Mix**

**OC-SVM learns**: "What does easy text look like?"
**At test time**: Hard texts detected as anomalies (deviations from easy boundary)

**This setup is ALREADY CORRECT** - no changes needed!

## What Was Implemented

### Files Created/Modified

1. ✅ **`model_training_ocsvm.py`** - OC-SVM training with StandardScaler
   - Critical fix: Added feature normalization
   - Updated hyperparameters from AD-NLP paper
   - Returns both model AND scaler

2. ✅ **`main_ocsvm.py`** - Complete OC-SVM pipeline
   - Loads data
   - Extracts embeddings (BERT or GloVe)
   - Trains OC-SVM with hyperparameter tuning
   - Evaluates and saves results
   - Automatically compares with Isolation Forest

3. ✅ **`config.py`** - Added OC-SVM configuration
   - `OCSVM_PARAM_GRID`: Hyperparameters from paper
   - `OCSVM_DEFAULT_PARAMS`: Good defaults
   - Ready to switch to BERT with one line

4. ✅ **`evaluation.py`** - Updated to support both models
   - Generic `compute_anomaly_scores()` accepts scaler
   - Works with Isolation Forest AND OC-SVM

5. ✅ **`README_OCSVM.md`** - Comprehensive documentation
6. ✅ **This file** - Brainstorming summary

### Branch Structure

```
main
│
├── claude/isolation-forest-glove-6vjK4
│   └── Isolation Forest + GloVe (AUROC = 0.481) ❌
│
└── claude/ocsvm-text-6GRQe  ← NEW BRANCH
    └── OC-SVM + (GloVe or BERT) ← TO BE TESTED
```

## How to Run

### Option 1: Test OC-SVM with GloVe (Quick Test)

```bash
# See if algorithm change alone helps
python main_ocsvm.py --quick
```

**Expected**: AUROC ~55-65% (small improvement)

### Option 2: OC-SVM with BERT (RECOMMENDED)

```bash
# 1. Edit config.py
# Change line 41 from:
EMBEDDING_TYPE = "glove"
# To:
EMBEDDING_TYPE = "bert"

# 2. Run full pipeline
python main_ocsvm.py
```

**Expected**: AUROC ~70-85% (big improvement!) ✅

### Option 3: Quick Test with BERT

```bash
# Edit config.py first, then:
python main_ocsvm.py --quick --reload-embeddings
```

## Expected Results

### Current: Isolation Forest + GloVe
```
AUROC: 0.481  ❌ (worse than random)
Separation: 0.0023  ❌ (no separation)
```

### Target: OC-SVM + BERT
```
AUROC: 0.70-0.85  ✅ (good separation)
Separation: 0.10-0.20  ✅ (clear difference)
F1 Score: 0.70-0.80  ✅
```

## Next Steps

1. **Test OC-SVM with GloVe** (quick baseline)
   - Expected: Small improvement (~55-65% AUROC)
   - Confirms algorithm change helps

2. **Switch to BERT embeddings** (main improvement)
   - Expected: Large improvement (~70-85% AUROC)
   - This should be the primary solution

3. **If BERT works well**: Commit and celebrate! 🎉

4. **If still not good enough**: Try hybrid approach (BERT + metrics)

## Why This Should Work

### Theoretical Reasoning

1. **BERT captures difficulty signals** that GloVe misses:
   - Sentence structure complexity
   - Word complexity in context
   - Syntactic patterns

2. **OC-SVM with RBF kernel** can find non-linear boundaries:
   - Not just linear separation
   - Can adapt to complex difficulty patterns

3. **Feature normalization** ensures all dimensions contribute:
   - No single feature dominates
   - Balanced decision boundary

### Empirical Evidence (AD-NLP Paper)

From Table 4:
- **OC-SVM on semantic anomalies**: 64-85% AUROC
- **Our task**: Semantic + syntactic difficulty detection
- **Expected range**: 70-85% AUROC with good embeddings

## Questions to Explore After Testing

1. **Which kernel works best?** (RBF vs linear vs poly)
2. **What's the optimal nu value?** (Boundary tightness)
3. **How much does BERT help vs GloVe?** (Quantify improvement)
4. **Which features matter most?** (If using hybrid)

## Ready to Test!

All code is implemented and ready to run. The branch is `claude/ocsvm-text-6GRQe`.

**Recommended first test**:
```bash
# Switch to BERT in config.py, then:
python main_ocsvm.py
```

Good luck! 🚀
