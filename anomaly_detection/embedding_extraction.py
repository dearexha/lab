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


def load_glove_from_file(filepath: str) -> Dict[str, np.ndarray]:
    """
    Load GloVe embeddings from text file.

    Args:
        filepath: Path to GloVe text file (e.g., glove.6B.300d.txt)

    Returns:
        Dictionary mapping word -> embedding vector
    """
    glove_dict = {}

    logger.info(f"Loading GloVe from file: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading GloVe embeddings"):
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype=np.float32)
            glove_dict[word] = vector

    logger.info(f"Loaded {len(glove_dict):,} word embeddings ({len(vector)}-dim)")
    return glove_dict


def download_glove_direct() -> Dict[str, np.ndarray]:
    """
    Download GloVe embeddings directly from Stanford's server.

    Returns:
        Dictionary mapping word -> embedding vector (300-dim)
    """
    import urllib.request
    import zipfile
    import tempfile

    logger.info("Downloading GloVe 6B from Stanford NLP...")

    # Download URL
    url = "https://nlp.stanford.edu/data/glove.6B.zip"

    # Download to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
        tmp_path = tmp_file.name
        logger.info(f"Downloading {url} ...")

        # Download with progress
        def reporthook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded / total_size * 100, 100)
            if block_num % 100 == 0:  # Update every 100 blocks
                logger.info(f"  Downloaded: {downloaded / 1024 / 1024:.1f} MB / {total_size / 1024 / 1024:.1f} MB ({percent:.1f}%)")

        urllib.request.urlretrieve(url, tmp_path, reporthook)
        logger.info("Download complete!")

    # Extract 300d file
    logger.info("Extracting glove.6B.300d.txt...")
    with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
        # Extract only the 300d file to embeddings directory
        target_file = 'glove.6B.300d.txt'
        zip_ref.extract(target_file, config.EMBEDDINGS_DIR)

    # Load the extracted file
    glove_file_path = config.EMBEDDINGS_DIR / target_file
    glove_dict = load_glove_from_file(glove_file_path)

    # Clean up temp file
    import os
    os.remove(tmp_path)

    return glove_dict


def load_glove_from_huggingface() -> Dict[str, np.ndarray]:
    """
    Load GloVe embeddings (tries multiple methods).

    Returns:
        Dictionary mapping word -> embedding vector (300-dim)
    """
    # Check cache first
    if config.GLOVE_CACHE_FILE.exists():
        logger.info(f"Loading cached GloVe embeddings from {config.GLOVE_CACHE_FILE}")
        with open(config.GLOVE_CACHE_FILE, 'rb') as f:
            glove_dict = pickle.load(f)
        logger.info(f"Loaded {len(glove_dict):,} word embeddings")
        return glove_dict

    # Check if already downloaded as text file
    glove_text_file = config.EMBEDDINGS_DIR / 'glove.6B.300d.txt'
    if glove_text_file.exists():
        logger.info(f"Found existing GloVe file: {glove_text_file}")
        glove_dict = load_glove_from_file(str(glove_text_file))

        # Cache for faster future loading
        logger.info(f"Caching GloVe embeddings to {config.GLOVE_CACHE_FILE}")
        with open(config.GLOVE_CACHE_FILE, 'wb') as f:
            pickle.dump(glove_dict, f)

        return glove_dict

    # Try direct download from Stanford
    logger.info("GloVe not found locally. Downloading from Stanford NLP...")
    try:
        glove_dict = download_glove_direct()

        # Cache for faster future loading
        logger.info(f"Caching GloVe embeddings to {config.GLOVE_CACHE_FILE}")
        with open(config.GLOVE_CACHE_FILE, 'wb') as f:
            pickle.dump(glove_dict, f)

        return glove_dict

    except Exception as e:
        logger.error(f"Failed to download GloVe: {e}")
        logger.error("Please manually download GloVe from:")
        logger.error("  https://nlp.stanford.edu/data/glove.6B.zip")
        logger.error(f"Extract glove.6B.300d.txt to: {config.EMBEDDINGS_DIR}/")
        raise


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


def load_bert_model():
    """
    Load BERT model from sentence-transformers.

    Returns:
        SentenceTransformer model
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
        raise

    logger.info(f"Loading BERT model: {config.BERT_MODEL}")
    model = SentenceTransformer(config.BERT_MODEL)
    logger.info(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")

    return model


def extract_bert_embeddings_for_split(
    texts: List[str],
    model,
    split_name: str = "data"
) -> np.ndarray:
    """
    Extract BERT embeddings for a list of texts.

    Args:
        texts: List of text strings
        model: SentenceTransformer model
        split_name: Name of split (for logging)

    Returns:
        embeddings: numpy array of shape (n_texts, embedding_dim)
    """
    logger.info(f"Extracting BERT embeddings for {split_name} ({len(texts):,} samples)...")

    # Encode texts in batches
    embeddings = model.encode(
        texts,
        batch_size=config.BERT_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False  # Keep raw embeddings
    )

    logger.info(f"{split_name} embedding shape: {embeddings.shape}")

    return embeddings


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
            'train': np.array (n_train, embedding_dim),
            'val': np.array (n_val, embedding_dim),
            'test': np.array (n_test, embedding_dim)
        }
    """
    # Check cache
    if config.EMBEDDINGS_CACHE.exists() and not force_reload:
        logger.info(f"Loading cached embeddings from {config.EMBEDDINGS_CACHE}")
        with open(config.EMBEDDINGS_CACHE, 'rb') as f:
            embeddings = pickle.load(f)
        logger.info("Cached embeddings loaded successfully")
        return embeddings

    # Extract embeddings based on config
    embeddings = {}

    if config.EMBEDDING_TYPE == "glove":
        logger.info("Using GloVe embeddings")
        # Load GloVe
        glove_dict = load_glove_from_huggingface()

        # Extract embeddings for each split
        for split_name in ['train', 'val', 'test']:
            texts = splits[split_name]['texts']
            emb, stats = extract_embeddings_for_split(texts, glove_dict, split_name)
            embeddings[split_name] = emb

    elif config.EMBEDDING_TYPE == "bert":
        logger.info("Using BERT embeddings")
        # Load BERT model
        model = load_bert_model()

        # Extract embeddings for each split
        for split_name in ['train', 'val', 'test']:
            texts = splits[split_name]['texts']
            emb = extract_bert_embeddings_for_split(texts, model, split_name)
            embeddings[split_name] = emb
    else:
        raise ValueError(f"Unknown embedding type: {config.EMBEDDING_TYPE}. Use 'glove' or 'bert'")

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
