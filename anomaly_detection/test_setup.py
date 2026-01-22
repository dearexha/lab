"""
Quick setup test script
Verifies that basic imports and data loading work
"""
import sys

def test_imports():
    """Test that all required packages are installed."""
    print("Testing imports...")

    try:
        import numpy as np
        print(f"  ✓ numpy {np.__version__}")
    except ImportError as e:
        print(f"  ✗ numpy: {e}")
        return False

    try:
        import sklearn
        print(f"  ✓ scikit-learn {sklearn.__version__}")
    except ImportError as e:
        print(f"  ✗ scikit-learn: {e}")
        return False

    try:
        import matplotlib
        print(f"  ✓ matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"  ✗ matplotlib: {e}")
        return False

    try:
        import seaborn
        print(f"  ✓ seaborn {seaborn.__version__}")
    except ImportError as e:
        print(f"  ✗ seaborn: {e}")
        return False

    try:
        import tqdm
        print(f"  ✓ tqdm {tqdm.__version__}")
    except ImportError as e:
        print(f"  ✗ tqdm: {e}")
        return False

    try:
        import datasets
        print(f"  ✓ datasets {datasets.__version__}")
    except ImportError as e:
        print(f"  ✗ datasets (HuggingFace): {e}")
        print("    Install with: pip install datasets")
        return False

    print("All imports successful!\n")
    return True


def test_config():
    """Test that config loads correctly."""
    print("Testing configuration...")

    try:
        import config
        print(f"  ✓ Project root: {config.PROJECT_ROOT}")
        print(f"  ✓ Data directory: {config.DATA_DIR}")
        print(f"  ✓ Output directory: {config.OUTPUT_DIR}")
        print(f"  ✓ Embeddings directory: {config.EMBEDDINGS_DIR}")

        # Check if data files exist
        if config.SIMPLE_FILE.exists():
            print(f"  ✓ Simple texts file found: {config.SIMPLE_FILE}")
        else:
            print(f"  ✗ Simple texts file NOT found: {config.SIMPLE_FILE}")
            return False

        if config.NORMAL_FILE.exists():
            print(f"  ✓ Normal texts file found: {config.NORMAL_FILE}")
        else:
            print(f"  ✗ Normal texts file NOT found: {config.NORMAL_FILE}")
            return False

        print("Configuration OK!\n")
        return True

    except Exception as e:
        print(f"  ✗ Config error: {e}")
        return False


def test_data_loading():
    """Test that data can be loaded."""
    print("Testing data loading (first 10 samples)...")

    try:
        from data_preparation import load_aligned_file
        import config

        simple_texts = load_aligned_file(config.SIMPLE_FILE)
        print(f"  ✓ Loaded {len(simple_texts):,} simple texts")
        print(f"    Sample: {simple_texts[0][:80]}...")

        normal_texts = load_aligned_file(config.NORMAL_FILE)
        print(f"  ✓ Loaded {len(normal_texts):,} normal texts")
        print(f"    Sample: {normal_texts[0][:80]}...")

        print("Data loading OK!\n")
        return True

    except Exception as e:
        print(f"  ✗ Data loading error: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("SETUP TEST - Isolation Forest + GloVe Pipeline")
    print("=" * 70)
    print()

    all_ok = True

    all_ok = test_imports() and all_ok
    all_ok = test_config() and all_ok
    all_ok = test_data_loading() and all_ok

    print("=" * 70)
    if all_ok:
        print("✓ ALL TESTS PASSED - Setup is ready!")
        print()
        print("Next steps:")
        print("  1. Run quick test: python main.py --quick")
        print("  2. Run full pipeline: python main.py")
        print("  3. Submit to SLURM: sbatch run_pipeline.sh")
    else:
        print("✗ SOME TESTS FAILED - Fix errors above before running pipeline")
        return 1
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
