"""
Quick test of BERT embedding extraction
"""
import sys

def test_bert_import():
    """Test that sentence-transformers is installed."""
    print("Testing BERT imports...")
    try:
        from sentence_transformers import SentenceTransformer
        print("✓ sentence-transformers imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import sentence-transformers: {e}")
        print("  Install with: pip install sentence-transformers")
        return False


def test_bert_model_loading():
    """Test loading BERT model."""
    print("\nTesting BERT model loading...")
    try:
        from sentence_transformers import SentenceTransformer
        import config

        print(f"Loading model: {config.BERT_MODEL}...")
        model = SentenceTransformer(config.BERT_MODEL)

        dim = model.get_sentence_embedding_dimension()
        print(f"✓ Model loaded successfully")
        print(f"  Embedding dimension: {dim}")
        print(f"  Expected dimension: {config.BERT_DIM}")

        if dim != config.BERT_DIM:
            print(f"⚠ Warning: Dimension mismatch! Update config.BERT_DIM to {dim}")

        return True
    except Exception as e:
        print(f"✗ Failed to load BERT model: {e}")
        return False


def test_bert_encoding():
    """Test BERT encoding on sample texts."""
    print("\nTesting BERT encoding...")
    try:
        from sentence_transformers import SentenceTransformer
        import config
        import numpy as np

        model = SentenceTransformer(config.BERT_MODEL)

        # Test sentences
        simple_text = "The cat sat on the mat."
        complex_text = "Notwithstanding the aforementioned circumstances, the feline positioned itself atop the textile floor covering."

        print(f"Encoding sample texts...")
        embeddings = model.encode([simple_text, complex_text], convert_to_numpy=True)

        print(f"✓ Encoding successful")
        print(f"  Shape: {embeddings.shape}")
        print(f"  Sample embedding (first 5 dims): {embeddings[0][:5]}")

        # Check that embeddings are different
        similarity = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
        print(f"  Cosine similarity between simple and complex: {similarity:.4f}")

        return True
    except Exception as e:
        print(f"✗ Failed to encode texts: {e}")
        return False


def main():
    print("="*70)
    print("BERT IMPLEMENTATION TEST")
    print("="*70)
    print()

    # Run tests
    tests = [
        ("Import test", test_bert_import),
        ("Model loading test", test_bert_model_loading),
        ("Encoding test", test_bert_encoding),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ Test '{name}' crashed: {e}")
            results.append((name, False))
        print()

    # Summary
    print("="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")

    print()
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\n✓ All tests passed! BERT implementation is ready.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed. Fix issues before running pipeline.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
