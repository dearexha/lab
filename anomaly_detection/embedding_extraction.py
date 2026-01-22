"""
Embedding Extraction Module
Loads GloVe embeddings and converts texts to sentence embeddings
"""
import pickle
import logging
import re
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm

import config

# Set up logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_glove_from_huggingface() -> Dict[str, np.ndarray]:
    """
    Load GloVe embeddings from HuggingFace.

    Returns:
        Dictionary mapping word -> embedding vector (300-dim)
    """
    if config.GLOVE_CACHE_FILE.exists():
        logger.info(f"Loading cached GloVe embeddings from {config.GLOVE_CACHE_FILE}")
        with open(config.GLOVE_CACHE_FILE, 'rb') as f:
            glove_dict = pickle.load(f)
        logger.info(f"Loaded {len(glove_dict):,} word embeddings")
        return glove_dict

    logger.info("Downloading GloVe embeddings from HuggingFace...")

    try:
        from datasets import load_dataset

        # Load GloVe dataset from HuggingFace
        dataset = load_dataset(
            config.GLOVE_HF_DATASET,
            config.GLOVE_HF_CONFIG,
            split='train'
        )

        logger.info(f"Downloaded dataset with {len(dataset):,} entries")

        # Extract embeddings for 300-dimensional version
        glove_dict = {}

        for entry in tqdm(dataset, desc="Building GloVe dictionary"):
            word = entry['word']
            # GloVe 6B has multiple dimensions: 50, 100, 200, 300
            # We want the 300-dimensional version
            if '300' in entry and entry['300'] is not None:
                embedding = np.array(entry['300'], dtype=np.float32)
                glove_dict[word] = embedding

        logger.info(f"Extracted {len(glove_dict):,} word embeddings (300-dim)")

        # Cache for future use
        logger.info(f"Caching GloVe embeddings to {config.GLOVE_CACHE_FILE}")
        with open(config.GLOVE_CACHE_FILE, 'wb') as f:
            pickle.dump(glove_dict, f)

    except ImportError:
        logger.error("HuggingFace 'datasets' library not installed. Install with: pip install datasets")
        raise
    except Exception as e:
        logger.error(f"Failed to load GloVe from HuggingFace: {e}")
        logger.info("Attempting alternative: manual download...")
        raise

    return glove_dict


def tokenize(text: str) -> List[str]:
    """
    Tokenize text into words.

    Steps:
    1. Lowercase (if configured)
    2. Extract word tokens (alphanumeric sequences)
    3. Remove special tokens

    Args:
        text: Input text string

    Returns:
        List of word tokens
    """
    # Lowercase
    if config.LOWERCASE:
        text = text.lower()

    # Extract words (alphanumeric sequences)
    if config.REMOVE_PUNCTUATION:
        # Only keep alphanumeric words
        tokens = re.findall(r'\b\w+\b', text)
    else:
        # Keep punctuation attached to words
        tokens = text.split()

    # Remove special tokens
    tokens = [t for t in tokens if t not in config.SPECIAL_TOKENS]

    return tokens


def text_to_embedding(
    text: str,
    glove_dict: Dict[str, np.ndarray],
    oov_strategy: str = 'skip'
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Convert text to sentence embedding via mean pooling.

    Args:
        text: Input text string
        glove_dict: Dictionary of word -> embedding
        oov_strategy: How to handle OOV words ('skip', 'zero', 'mean')

    Returns:
        Tuple of:
        - sentence_embedding: 300-dimensional vector
        - stats: Dictionary with statistics (n_words, n_oov, etc.)
    """
    tokens = tokenize(text)

    # Collect valid embeddings
    valid_embeddings = []
    n_oov = 0

    for word in tokens:
        if word in glove_dict:
            valid_embeddings.append(glove_dict[word])
        else:
            n_oov += 1

    # Statistics
    stats = {
        'n_tokens': len(tokens),
        'n_valid': len(valid_embeddings),
        'n_oov': n_oov,
        'oov_rate': n_oov / len(tokens) if len(tokens) > 0 else 0.0
    }

    # Compute sentence embedding
    if len(valid_embeddings) > 0:
        # Mean pooling over valid embeddings
        sentence_embedding = np.mean(valid_embeddings, axis=0)
    else:
        # All words are OOV
        if config.USE_ZERO_VECTOR_FOR_EMPTY:
            sentence_embedding = np.zeros(config.GLOVE_DIM, dtype=np.float32)
            stats['used_zero_vector'] = True
        else:
            # This shouldn't happen if we filter properly, but handle it
            sentence_embedding = np.zeros(config.GLOVE_DIM, dtype=np.float32)
            stats['used_zero_vector'] = True
            logger.warning(f"All words OOV for text: {text[:50]}...")

    return sentence_embedding, stats


def extract_embeddings_for_split(
    texts: List[str],
    glove_dict: Dict[str, np.ndarray],
    split_name: str = "data"
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Extract embeddings for a list of texts.

    Args:
        texts: List of text strings
        glove_dict: GloVe dictionary
        split_name: Name of split (for logging)

    Returns:
        Tuple of:
        - embeddings: numpy array of shape (n_texts, 300)
        - stats_list: List of statistics dictionaries
    """
    logger.info(f"Extracting embeddings for {split_name} ({len(texts):,} samples)...")

    embeddings = []
    stats_list = []

    for text in tqdm(texts, desc=f"Processing {split_name}"):
        embedding, stats = text_to_embedding(text, glove_dict, config.OOV_STRATEGY)
        embeddings.append(embedding)
        stats_list.append(stats)

    embeddings = np.array(embeddings, dtype=np.float32)

    # Aggregate statistics
    total_tokens = sum(s['n_tokens'] for s in stats_list)
    total_valid = sum(s['n_valid'] for s in stats_list)
    total_oov = sum(s['n_oov'] for s in stats_list)
    n_zero_vectors = sum(1 for s in stats_list if s.get('used_zero_vector', False))

    logger.info(f"{split_name} embedding statistics:")
    logger.info(f"  Total tokens: {total_tokens:,}")
    logger.info(f"  Valid tokens (in GloVe): {total_valid:,} ({total_valid/total_tokens*100:.2f}%)")
    logger.info(f"  OOV tokens: {total_oov:,} ({total_oov/total_tokens*100:.2f}%)")
    logger.info(f"  Sentences with all OOV: {n_zero_vectors} ({n_zero_vectors/len(texts)*100:.2f}%)")
    logger.info(f"  Embedding shape: {embeddings.shape}")

    return embeddings, stats_list


def prepare_embeddings(
    splits: Dict[str, Dict],
    force_reload: bool = False
) -> Dict[str, np.ndarray]:
    """
    Prepare embeddings for all data splits.

    Args:
        splits: Data splits dictionary
        force_reload: If True, recompute embeddings even if cache exists

    Returns:
        Dictionary with embeddings for each split:
        {
            'train': np.array (n_train, 300),
            'val': np.array (n_val, 300),
            'test': np.array (n_test, 300)
        }
    """
    # Check cache
    if config.EMBEDDINGS_CACHE.exists() and not force_reload:
        logger.info(f"Loading cached embeddings from {config.EMBEDDINGS_CACHE}")
        with open(config.EMBEDDINGS_CACHE, 'rb') as f:
            embeddings = pickle.load(f)
        logger.info("Cached embeddings loaded successfully")
        return embeddings

    # Load GloVe
    glove_dict = load_glove_from_huggingface()

    # Extract embeddings for each split
    embeddings = {}

    for split_name in ['train', 'val', 'test']:
        texts = splits[split_name]['texts']
        emb, stats = extract_embeddings_for_split(texts, glove_dict, split_name)
        embeddings[split_name] = emb

    # Cache embeddings
    logger.info(f"Caching embeddings to {config.EMBEDDINGS_CACHE}")
    with open(config.EMBEDDINGS_CACHE, 'wb') as f:
        pickle.dump(embeddings, f)

    logger.info("=" * 60)
    logger.info("EMBEDDINGS PREPARED")
    logger.info("=" * 60)
    for split_name, emb in embeddings.items():
        logger.info(f"{split_name}: {emb.shape}")
    logger.info("=" * 60)

    return embeddings


if __name__ == "__main__":
    # Test embedding extraction
    from data_preparation import prepare_data

    # Prepare data
    splits = prepare_data()

    # Extract embeddings
    embeddings = prepare_embeddings(splits, force_reload=True)

    # Show sample
    print("\nSample embeddings:")
    print(f"Train[0]: {embeddings['train'][0][:10]}... (shape: {embeddings['train'][0].shape})")
    print(f"Val[0]: {embeddings['val'][0][:10]}... (shape: {embeddings['val'][0].shape})")
    print(f"Test[0]: {embeddings['test'][0][:10]}... (shape: {embeddings['test'][0].shape})")
