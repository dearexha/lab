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

# Install PyTorch with CUDA support first
# Adjust CUDA version based on your cluster: cu118, cu121, or cu126
echo "Installing PyTorch with CUDA support..."
pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu126
# Alternative if cu126 doesn't work:
# pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu121

# Install other required packages from requirements.txt
echo "Installing remaining packages from requirements.txt..."
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

# Install remaining dependencies
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
mkdir -p results/runs
mkdir -p results/difficulty_classifier_normalized_input
echo "Results directory structure ready"
echo ""

# --- Check for required pre-computed datasets ---
echo "=================================================="
echo "Checking for pre-computed datasets..."
echo "=================================================="

SW_DATASET="hf_datasets/SimpleWikipedia"
OSE_DATASET="hf_datasets/OneStopEnglish"

HAS_SW=$(test -d "$SW_DATASET" && echo "yes" || echo "no")
HAS_OSE=$(test -d "$OSE_DATASET" && echo "yes" || echo "no")

echo "SimpleWikipedia dataset exists: $HAS_SW"
echo "OneStopEnglish dataset exists: $HAS_OSE"
echo ""

if [ -d "hf_datasets" ]; then
    echo "Contents of hf_datasets/:"
    ls -la hf_datasets/ 2>/dev/null | head -10
    echo ""
fi

# Verify SimpleWikipedia dataset (required for current config)
if [ "$HAS_SW" = "no" ]; then
    echo "=================================================="
    echo "ERROR: Required dataset not found!"
    echo "=================================================="
    echo "The training script requires: hf_datasets/SimpleWikipedia/"
    echo ""
    echo "This should contain pre-computed datasets from your colleague with:"
    echo "  - train_class_0/"
    echo "  - train_class_1/"
    echo "  - validation/"
    echo "  - test/"
    echo ""
    echo "Please ensure the 'hf_datasets' folder is in the project root directory."
    echo ""
    exit 1
fi

# Check dataset structure
echo "Verifying dataset structure..."
if [ -d "$SW_DATASET/train_class_0" ] && [ -d "$SW_DATASET/train_class_1" ]; then
    echo "✓ Dataset structure looks correct"

    # Show dataset info
    if [ -f "$SW_DATASET/train_class_0/dataset_info.json" ]; then
        echo ""
        echo "Dataset features (from train_class_0/dataset_info.json):"
        python3 -c "
import json
with open('$SW_DATASET/train_class_0/dataset_info.json', 'r') as f:
    info = json.load(f)
    features = list(info.get('features', {}).keys())
    print('  ' + ', '.join(features[:10]))
    if len(features) > 10:
        print('  ... and', len(features) - 10, 'more')
" 2>/dev/null || echo "  (could not read dataset info)"
    fi
else
    echo "⚠ WARNING: Dataset structure may be incomplete"
    echo "Expected subdirectories: train_class_0, train_class_1, validation, test"
fi
echo ""

# --- Optional: Dataset creation for advanced users ---
# If you need to recreate datasets from scratch, uncomment this section
# and ensure you have:
#   1. Raw data in datasets/ folder
#   2. Difficulty classifier trained (run composite_difficulty_metric.py first)
#
# RECREATE_DATASETS=false  # Set to true if you want to recreate datasets
#
# if [ "$RECREATE_DATASETS" = "true" ]; then
#     echo "=================================================="
#     echo "Recreating datasets from scratch..."
#     echo "=================================================="
#
#     # Backup main.py
#     cp main.py main.py.backup
#
#     # Uncomment dataset creation section (lines 64-112) and comment load section (line 115)
#     python3 << 'PYTHON_SCRIPT'
# with open('main.py', 'r') as f:
#     lines = f.readlines()
#
# # Uncomment lines 64-112 (indices 63-111)
# for i in range(63, 112):
#     if i < len(lines):
#         line = lines[i]
#         if line.strip().startswith('#'):
#             stripped = line.lstrip()
#             if stripped.startswith('# '):
#                 lines[i] = line[:len(line) - len(stripped)] + stripped[2:]
#             elif stripped.startswith('#'):
#                 lines[i] = line[:len(line) - len(stripped)] + stripped[1:]
#
# # Comment out line 115
# if len(lines) > 114:
#     line = lines[114]
#     if 'sw_dataset_dict = load_from_disk' in line and not line.strip().startswith('#'):
#         indent = len(line) - len(line.lstrip())
#         lines[114] = ' ' * indent + '# ' + line.lstrip()
#
# with open('main.py', 'w') as f:
#     f.writelines(lines)
# PYTHON_SCRIPT
#
#     # Run dataset creation
#     python main.py
#     DATASET_EXIT=$?
#
#     # Restore main.py
#     mv main.py.backup main.py
#
#     if [ $DATASET_EXIT -ne 0 ]; then
#         echo "ERROR: Dataset creation failed"
#         exit 1
#     fi
#     echo "Dataset creation completed"
#     echo ""
# fi

# --- Run Training ---
echo "=================================================="
echo "Starting BERT Curriculum Learning Training"
echo "Started at: $(date)"
echo "=================================================="
echo ""
echo "Training configuration (from config.yaml):"
python3 -c "
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    print(f\"  Seed: {config.get('seed', 'N/A')}\")
    print(f\"  Batch size: {config.get('batch_size', 'N/A')}\")
    print(f\"  Learning rate: {config.get('learning_rate', 'N/A')}\")
    print(f\"  Max steps: {config.get('max_steps', 'N/A')}\")
    print(f\"  Label-based CL: {config.get('label_based', 'N/A')}\")
    if config.get('label_based'):
        print(f\"  Label schedule: {config.get('label_schedule', 'N/A')}\")
    else:
        print(f\"  Difficulty metric: {config.get('difficulty_metric', 'N/A')}\")
" 2>/dev/null || echo "  (could not read config.yaml)"
echo ""

python main.py
TRAINING_EXIT_CODE=$?

echo ""
echo "=================================================="
echo "Training completed at: $(date)"
echo "Exit code: $TRAINING_EXIT_CODE"
echo "=================================================="

# Show results location
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Training artifacts saved to:"
    LATEST_RUN=$(ls -td results/runs/*/ 2>/dev/null | head -1)
    if [ -n "$LATEST_RUN" ]; then
        echo "  $LATEST_RUN"
        echo ""
        echo "Files created:"
        ls -lh "$LATEST_RUN" 2>/dev/null | grep -v "^total" | awk '{print "  " $9 " (" $5 ")"}'
    fi
else
    echo ""
    echo "⚠ WARNING: Training exited with error code $TRAINING_EXIT_CODE"
    echo "Check the log files for details:"
    echo "  - slurm_${SLURM_JOB_ID}.out"
    echo "  - slurm_${SLURM_JOB_ID}.err"
fi

echo ""
echo "=================================================="
echo "Job finished at: $(date)"
echo "=================================================="

# --- Deactivate virtual environment ---
deactivate
