#!/bin/bash
#SBATCH --job-name=cl_training          # Job name
#SBATCH --output=slurm_%j.out           # Standard output log (%j = job ID)
#SBATCH --error=slurm_%j.err            # Standard error log (%j = job ID)
#SBATCH --partition=A40short            # GPU partition
#SBATCH --time=8:00:00                  # Time limit (8 hours)
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=8                # CPUs per task
#SBATCH --gpus=1                         # Request 1 GPU
#SBATCH --mem=64GB                       # Memory allocation

# --- Print job info ---
echo "=================================================="
echo "Starting SLURM Job: $SLURM_JOB_NAME (ID: $SLURM_JOB_ID)"
echo "Run on host: $(hostname)"
echo "Node: $SLURM_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Job started at: $(date)"
echo "Working Directory: $SLURM_SUBMIT_DIR"
echo "=================================================="

# --- Module setup ---
module purge
module load Python
module list

# --- Python environment setup ---
VENV_DIR="$HOME/cl_training_env"

# Create venv if it does not exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python -m venv "$VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support first (check CUDA version on cluster)
# Check available CUDA version: module avail CUDA
# For CUDA 11.8, use: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# For CUDA 12.1, use: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# For CUDA 12.6 (as in requirements.txt), use: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

echo "Installing PyTorch with CUDA support..."
# Try to detect CUDA version, or use the one from requirements.txt
# Uncomment the appropriate line based on your cluster's CUDA version:
pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu126
# Alternative if cu126 doesn't work:
# pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu121
# Or for CPU-only (not recommended for training):
# pip install torch==2.9.0 torchvision==0.24.0

# Install all other required packages from requirements.txt
echo "Installing remaining packages from requirements.txt..."
# Install packages that don't depend on torch first
pip install --no-cache-dir \
    numpy==2.3.4 \
    matplotlib==3.10.7 \
    pandas==2.3.3 \
    scikit-learn \
    PyYAML==6.0.3 \
    tqdm==4.67.1 \
    nltk==3.9.2 \
    pyphen==0.17.2 \
    datasets==4.4.1 \
    transformers==4.57.1 \
    accelerate \
    huggingface-hub==0.36.0 \
    tokenizers==0.22.1 \
    safetensors==0.6.2

# Install remaining dependencies (these may have version conflicts, so install without strict versions)
echo "Installing additional dependencies..."
pip install --no-cache-dir \
    seaborn \
    pyarrow \
    dill \
    filelock \
    requests \
    regex \
    packaging \
    pillow \
    python-dateutil \
    pytz \
    urllib3 \
    idna \
    certifi \
    charset-normalizer \
    MarkupSafe \
    Jinja2 \
    joblib

# Verify critical packages are installed
echo "=================================================="
echo "Verifying package installations..."
echo "=================================================="
python -c "import torch; print(f'✓ PyTorch {torch.__version__}')" || echo "✗ PyTorch not installed"
python -c "import transformers; print(f'✓ Transformers {transformers.__version__}')" || echo "✗ Transformers not installed"
python -c "import datasets; print(f'✓ Datasets {datasets.__version__}')" || echo "✗ Datasets not installed"
python -c "import matplotlib; print('✓ Matplotlib')" || echo "✗ Matplotlib not installed"
python -c "import numpy; print(f'✓ NumPy {numpy.__version__}')" || echo "✗ NumPy not installed"
python -c "import nltk; print('✓ NLTK')" || echo "✗ NLTK not installed"
python -c "import pyphen; print('✓ Pyphen')" || echo "✗ Pyphen not installed"
python -c "import yaml; print('✓ PyYAML')" || echo "✗ PyYAML not installed"
python -c "import sklearn; print('✓ scikit-learn')" || echo "✗ scikit-learn not installed"
python -c "import tqdm; print('✓ tqdm')" || echo "✗ tqdm not installed"

# Download NLTK data (if needed)
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)" || echo "NLTK data download failed (may already be present)"

# --- Environment variables ---
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# --- Print environment info ---
echo "=================================================="
echo "Environment Information:"
echo "=================================================="
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'PyTorch not available')"
if command -v nvidia-smi &> /dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    echo "GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)"
fi

# --- Change to project directory ---
cd $SLURM_SUBMIT_DIR

# --- Create results directory structure if it doesn't exist ---
echo "=================================================="
echo "Setting up results directory structure..."
echo "=================================================="
mkdir -p results/hf_datasets
mkdir -p results/runs
mkdir -p results/difficulty_classifier_normalized_input
echo "Results directory structure ready"
echo ""

# --- Check for hf_datasets folder in various locations and move/copy if needed ---
echo "=================================================="
echo "Checking for hf_datasets folder..."
echo "=================================================="

# Check if hf_datasets exists in current directory or as results/hf_datasets
if [ -d "hf_datasets" ] && [ ! -d "results/hf_datasets" ]; then
    echo "Found 'hf_datasets' folder in current directory. Moving to results/..."
    mv hf_datasets results/hf_datasets
    echo "Moved hf_datasets to results/hf_datasets"
elif [ -d "hf_datasets" ] && [ -d "results/hf_datasets" ]; then
    echo "Found both 'hf_datasets' and 'results/hf_datasets'. Copying contents..."
    cp -r hf_datasets/* results/hf_datasets/ 2>/dev/null || true
    echo "Copied contents from hf_datasets to results/hf_datasets"
fi

# Also check if results/hf_datasets might be in a different location (e.g., if colleague provided it)
# Check common alternative locations
if [ ! -d "results/hf_datasets" ]; then
    if [ -d "../hf_datasets" ]; then
        echo "Found hf_datasets in parent directory. Creating symlink..."
        ln -s ../hf_datasets results/hf_datasets
    elif [ -d "$HOME/hf_datasets" ]; then
        echo "Found hf_datasets in home directory. Creating symlink..."
        ln -s "$HOME/hf_datasets" results/hf_datasets
    fi
fi

# Check what already exists ---
echo "=================================================="
echo "Checking existing datasets and models..."
echo "=================================================="

INITIAL_DATASET="results/hf_datasets/SimpleWikipedia_tokenized_and_measured"
CLASSIFIER_DIR="results/difficulty_classifier_normalized_input"
FULL_DATASET="results/hf_datasets/SimpleWikipedia"

HAS_INITIAL=$(test -d "$INITIAL_DATASET" && echo "yes" || echo "no")
HAS_CLASSIFIER=$(test -f "$CLASSIFIER_DIR/log.json" && echo "yes" || echo "no")
HAS_FULL_DATASET=$(test -d "$FULL_DATASET" && echo "yes" || echo "no")

echo "Initial dataset (tokenized_and_measured) exists: $HAS_INITIAL"
echo "Classifier exists: $HAS_CLASSIFIER"
echo "Full dataset (SimpleWikipedia) exists: $HAS_FULL_DATASET"
echo ""

# List what's in hf_datasets for debugging
if [ -d "results/hf_datasets" ]; then
    echo "Contents of results/hf_datasets:"
    ls -la results/hf_datasets/ | head -15
    echo ""
fi

# Backup main.py before modifications
cp main.py main.py.backup

# --- OPTIMIZED PATH: If full dataset exists, skip everything and go straight to training ---
if [ "$HAS_FULL_DATASET" = "yes" ]; then
    echo "=================================================="
    echo "✓ Full dataset 'SimpleWikipedia' found!"
    echo "Skipping dataset creation steps - going straight to training."
    echo "=================================================="
    echo ""
    # Make sure main.py has dataset creation commented (it should be by default)
    # Just verify and proceed to training
    echo "Starting training directly..."
    echo ""
else
    # --- Stage 1: Train difficulty classifier (needs initial dataset) ---
    if [ "$HAS_CLASSIFIER" = "no" ]; then
        if [ "$HAS_INITIAL" = "no" ]; then
            echo "=================================================="
            echo "WARNING: Cannot proceed - missing required datasets."
            echo "=================================================="
            echo "To train the classifier, you need:"
            echo "  results/hf_datasets/SimpleWikipedia_tokenized_and_measured"
            echo ""
            echo "To run training directly, you need:"
            echo "  results/hf_datasets/SimpleWikipedia"
            echo ""
            echo "Current contents of results/hf_datasets:"
            if [ -d "results/hf_datasets" ]; then
                ls -la results/hf_datasets/
            else
                echo "  (directory does not exist)"
            fi
            echo ""
            echo "If your colleague provided datasets, make sure the 'hf_datasets' folder"
            echo "is placed in the project root or in results/ directory."
            echo ""
            mv main.py.backup main.py 2>/dev/null || true
            exit 1
        fi
        echo "=================================================="
        echo "Stage 1: Training difficulty classifier"
        echo "Started at $(date)"
        echo "=================================================="
        python composite_difficulty_metric.py
        if [ $? -ne 0 ]; then
            echo "ERROR: Failed to train classifier. Exiting."
            mv main.py.backup main.py 2>/dev/null || true
            exit 1
        fi
        echo "Stage 1 completed at $(date)"
        echo ""
    else
        echo "Skipping Stage 1: Classifier already exists"
        echo ""
    fi

    # --- Stage 2: Create full dataset with metrics (uncomment main.py dataset creation) ---
    if [ "$HAS_FULL_DATASET" = "no" ]; then
        if [ "$HAS_CLASSIFIER" = "no" ]; then
            echo "ERROR: Cannot create full dataset - classifier does not exist."
            echo "Please run Stage 1 first."
            mv main.py.backup main.py 2>/dev/null || true
            exit 1
        fi
        echo "=================================================="
        echo "Stage 2: Creating full dataset with difficulty metrics"
        echo "Started at $(date)"
        echo "=================================================="
    # Uncomment the dataset creation section (lines 64-112) and comment the load section
    echo "Temporarily modifying main.py to uncomment dataset creation..."
    python3 << 'PYTHON_SCRIPT'
with open('main.py', 'r') as f:
    lines = f.readlines()

# Uncomment lines 64-112 (indices 63-111)
for i in range(63, 112):  # 0-indexed, so 64-112 becomes 63-111
    if i < len(lines):
        line = lines[i]
        # Remove leading # and space if it's a comment
        if line.strip().startswith('#'):
            # Remove # and following space, but preserve indentation
            stripped = line.lstrip()
            if stripped.startswith('# '):
                lines[i] = line[:len(line) - len(stripped)] + stripped[2:]
            elif stripped.startswith('#'):
                lines[i] = line[:len(line) - len(stripped)] + stripped[1:]

# Comment out line 115 (index 114) - the load_from_disk line
if len(lines) > 114:
    line = lines[114]
    if 'sw_dataset_dict = load_from_disk' in line and not line.strip().startswith('#'):
        # Add comment, preserving indentation
        indent = len(line) - len(line.lstrip())
        lines[114] = ' ' * indent + '# ' + line.lstrip()

with open('main.py', 'w') as f:
    f.writelines(lines)

print("Uncommented dataset creation section (lines 64-112) and commented load section (line 115)")
PYTHON_SCRIPT

    python main.py
    DATASET_CREATION_EXIT=$?
    
    # Restore main.py to original (commented) state immediately
    mv main.py.backup main.py
    
        if [ $DATASET_CREATION_EXIT -ne 0 ]; then
            echo "ERROR: Failed to create full dataset. Exiting."
            exit 1
        fi
        echo "Stage 2 completed at $(date)"
        echo ""
    else
        echo "Skipping Stage 2: Full dataset already exists"
        echo ""
    fi
fi

# --- Stage 3: Run training (main.py should have dataset creation commented) ---
echo "=================================================="
echo "Stage 3: Starting training"
echo "Started at $(date)"
echo "=================================================="
python main.py
TRAINING_EXIT_CODE=$?
echo "Training finished at $(date)"
if [ $TRAINING_EXIT_CODE -ne 0 ]; then
    echo "WARNING: Training exited with code $TRAINING_EXIT_CODE"
fi

# --- Completion info ---
echo "=================================================="
echo "Job completed at $(date)"
echo "Exit code: $?"
echo "=================================================="

# --- Deactivate virtual environment ---
deactivate

