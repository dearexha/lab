# BERT Implementation for Anomaly Detection

## Overview

This document describes the BERT embedding implementation added to test whether contextual embeddings can better distinguish text difficulty compared to static GloVe embeddings.

## Why BERT Instead of GloVe?

### GloVe Results (Initial Implementation)

The initial pipeline used GloVe embeddings with mean pooling:
- **AUROC: 0.4815** (worse than random 0.5)
- **Score separation: 0.0023** (essentially zero)
- **Conclusion**: GloVe + mean pooling cannot distinguish SimpleWikipedia from NormalWikipedia

### Why GloVe Failed

1. **Static embeddings**: GloVe captures semantic meaning but not syntactic complexity
2. **Mean pooling**: Averaging word vectors loses sentence structure information
3. **Similar vocabulary**: Both simple and normal texts use similar words (98.5% GloVe coverage)
4. **Lost signals**: Word order, sentence structure, and complexity are washed out

### Why BERT Might Work Better

BERT (Bidirectional Encoder Representations from Transformers) has advantages:

1. **Contextual embeddings**: Word representations depend on context
2. **Sentence-level encoding**: Captures sentence structure, not just word meanings
3. **Syntactic awareness**: Trained to understand grammar and sentence complexity
4. **Pre-trained on diverse texts**: Includes both simple and complex language

## Implementation Details

### Model Selection

**Chosen model**: `all-MiniLM-L6-v2`
- **Dimensions**: 384
- **Speed**: Fast (6-layer MiniLM)
- **Quality**: Good performance on semantic similarity tasks
- **Size**: ~80MB (reasonable for HPC)

**Alternative models**:
- `all-mpnet-base-v2`: Better quality (768-dim) but slower
- `paraphrase-MiniLM-L6-v2`: Optimized for paraphrase detection

### Configuration

In `config.py`:
```python
EMBEDDING_TYPE = "bert"  # Switch between "glove" and "bert"
BERT_MODEL = "all-MiniLM-L6-v2"
BERT_DIM = 384
BERT_BATCH_SIZE = 32
BERT_MAX_SEQ_LENGTH = 128
```

### Key Changes

1. **config.py**: Added BERT configuration parameters
2. **embedding_extraction.py**: Added BERT loading and extraction functions
3. **requirements.txt**: Added `sentence-transformers` and `torch`
4. **run_pipeline.sh**: Added GPU support (`--gres=gpu:1`)

### Encoding Process

```python
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode texts (batch processing)
embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
)
# Output: (n_texts, 384) numpy array
```

## Running the Pipeline

### Quick Test (Local)

```bash
cd anomaly_detection
python test_bert.py  # Verify BERT works
python test_setup.py  # Verify full setup
```

### Full Pipeline (HPC)

```bash
cd anomaly_detection
sbatch run_pipeline.sh
```

The SLURM script will:
- Request 1 GPU (`--gres=gpu:1`)
- Install `torch` and `sentence-transformers`
- Download BERT model on first run (~80MB)
- Extract embeddings for all 335k texts
- Train Isolation Forest
- Evaluate and generate plots

### Expected Runtime

- **GloVe**: ~20 minutes (CPU only)
- **BERT**: ~30-45 minutes (with GPU, longer on CPU)
  - Model download: ~1 minute
  - Embedding extraction: ~15-30 minutes (depends on GPU)
  - Training/evaluation: ~5-10 minutes

## Switching Between GloVe and BERT

To switch embedding types, edit `config.py`:

```python
# Use GloVe
EMBEDDING_TYPE = "glove"

# Use BERT
EMBEDDING_TYPE = "bert"
```

Then run:
```bash
python main.py --reload-embeddings  # Force recompute embeddings
```

## Expected Results

### If BERT Works Better

Expected improvements:
- **AUROC > 0.6** (ideally 0.7-0.8)
- **Score separation > 0.1** (meaningful gap)
- Normal texts should score significantly higher than simple texts

### If BERT Also Fails

If AUROC ≈ 0.5:
- Text difficulty may not be captured by semantic content alone
- Consider adding handcrafted features:
  - Sentence length
  - Word frequency/rarity
  - Syntactic complexity metrics
- Or switch to supervised learning

## Comparison: GloVe vs BERT

| Aspect | GloVe | BERT |
|--------|-------|------|
| **Type** | Static word embeddings | Contextual sentence embeddings |
| **Dimensions** | 300 | 384 |
| **Encoding** | Mean pooling of word vectors | Transformer-based sentence encoding |
| **Captures** | Semantic meaning | Semantics + syntax + structure |
| **Speed** | Fast (dictionary lookup) | Slower (neural network inference) |
| **Size** | ~350MB (400k words) | ~80MB (model weights) |
| **AUROC Result** | 0.4815 (failed) | TBD |

## Troubleshooting

### Out of Memory

If GPU runs out of memory:
```python
# In config.py
BERT_BATCH_SIZE = 16  # Reduce from 32
```

### Slow Encoding

If encoding is very slow:
1. Check GPU is being used: `nvidia-smi` during run
2. Verify CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
3. Consider smaller model or CPU-only mode

### Model Download Fails

If model download fails on HPC:
1. Download locally: `python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"`
2. Model cached in `~/.cache/torch/sentence_transformers/`
3. Transfer to HPC if needed

## Next Steps After Results

### If BERT Works (AUROC > 0.6)
1. Document results comparison with GloVe
2. Use BERT embeddings for curriculum learning
3. Consider fine-tuning BERT on your specific task

### If BERT Fails (AUROC ≈ 0.5)
1. Add handcrafted features (length, word frequency)
2. Try supervised learning (you have balanced labels!)
3. Explore contrastive learning approaches
4. Consider that semantic content alone may be insufficient

## Files Modified

- `config.py`: Added BERT configuration
- `embedding_extraction.py`: Added BERT loading and extraction
- `requirements.txt`: Added sentence-transformers, torch
- `run_pipeline.sh`: Added GPU support, updated packages
- `test_setup.py`: Added torch/sentence-transformers checks
- `test_bert.py`: New test script for BERT functionality

## References

- Sentence-Transformers: https://www.sbert.net/
- all-MiniLM-L6-v2: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- BERT paper: https://arxiv.org/abs/1810.04805
