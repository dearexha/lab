#!/bin/bash
#SBATCH --job-name=ad_ocsvm_bert          # Job name (OC-SVM)
#SBATCH --output=slurm_%j.out             # Standard output log (%j = job ID)
#SBATCH --error=slurm_%j.err              # Standard error log (%j = job ID)
#SBATCH --partition=A40short              # Partition with GPU access
#SBATCH --gres=gpu:1                      # Request 1 GPU (BERT benefits from GPU)
#SBATCH --time=4:00:00                    # Time limit (4 hours)
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks=1                        # Number of tasks
#SBATCH --cpus-per-task=8                 # CPUs per task
#SBATCH --mem=32GB                        # Memory allocation

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
VENV_DIR="$HOME/ad_pipeline_env"

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

# Install required packages for Anomaly Detection pipeline
echo "Installing packages from requirements.txt..."
pip install --no-cache-dir \
    numpy \
    scikit-learn \
    datasets \
    tqdm \
    matplotlib \
    seaborn \
    pandas \
    torch \
    sentence-transformers

# Install additional dependencies
echo "Installing additional dependencies..."
pip install --no-cache-dir \
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
    joblib \
    xxhash \
    multiprocess \
    fsspec

# Verify critical packages are installed
echo "=================================================="
echo "Verifying package installations..."
echo "=================================================="
python -c "import numpy; print(f'✓ NumPy {numpy.__version__}')" || echo "✗ NumPy not installed"
python -c "import sklearn; print(f'✓ scikit-learn {sklearn.__version__}')" || echo "✗ scikit-learn not installed"
python -c "import datasets; print(f'✓ HuggingFace Datasets {datasets.__version__}')" || echo "✗ Datasets not installed"
python -c "import tqdm; print(f'✓ tqdm {tqdm.__version__}')" || echo "✗ tqdm not installed"
python -c "import matplotlib; print(f'✓ Matplotlib {matplotlib.__version__}')" || echo "✗ Matplotlib not installed"
python -c "import seaborn; print(f'✓ Seaborn {seaborn.__version__}')" || echo "✗ Seaborn not installed"
python -c "import pandas; print(f'✓ Pandas {pandas.__version__}')" || echo "✗ Pandas not installed"
python -c "import torch; print(f'✓ PyTorch {torch.__version__}')" || echo "✗ PyTorch not installed"
python -c "import sentence_transformers; print(f'✓ sentence-transformers {sentence_transformers.__version__}')" || echo "✗ sentence-transformers not installed"

# --- Environment variables ---
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# --- Print environment info ---
echo "=================================================="
echo "Environment Information:"
echo "=================================================="
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "CPUs available: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo ""

# --- Change to project directory ---
# Try multiple paths to find anomaly_detection directory
if [ -d "$SLURM_SUBMIT_DIR/anomaly_detection" ]; then
    cd "$SLURM_SUBMIT_DIR/anomaly_detection"
elif [ -d "$SLURM_SUBMIT_DIR" ] && [ "$(basename $SLURM_SUBMIT_DIR)" = "anomaly_detection" ]; then
    cd "$SLURM_SUBMIT_DIR"
elif [ -f "$SLURM_SUBMIT_DIR/run_pipeline_ocsvm.sh" ]; then
    # We're already in anomaly_detection directory
    cd "$SLURM_SUBMIT_DIR"
else
    echo "ERROR: Could not find anomaly_detection directory"
    echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
    echo "Current directory: $(pwd)"
    echo ""
    echo "Please submit the job from one of these locations:"
    echo "  1. From repository root: sbatch anomaly_detection/run_pipeline_ocsvm.sh"
    echo "  2. From anomaly_detection/: sbatch run_pipeline_ocsvm.sh"
    exit 1
fi

echo "Working directory: $(pwd)"

# --- Create outputs directory if it doesn't exist ---
echo "=================================================="
echo "Setting up output directory structure..."
echo "=================================================="
mkdir -p outputs
echo "Output directory ready"
echo ""

# --- Check for required datasets ---
echo "=================================================="
echo "Checking for required datasets..."
echo "=================================================="

SW_SIMPLE="../datasets/SimpleWikipedia_v2/simple.aligned"
SW_NORMAL="../datasets/SimpleWikipedia_v2/normal.aligned"

HAS_SIMPLE=$(test -f "$SW_SIMPLE" && echo "yes" || echo "no")
HAS_NORMAL=$(test -f "$SW_NORMAL" && echo "yes" || echo "no")

echo "SimpleWikipedia simple.aligned exists: $HAS_SIMPLE"
echo "SimpleWikipedia normal.aligned exists: $HAS_NORMAL"
echo ""

if [ "$HAS_SIMPLE" = "no" ] || [ "$HAS_NORMAL" = "no" ]; then
    echo "=================================================="
    echo "ERROR: Required datasets not found!"
    echo "=================================================="
    echo "The pipeline requires:"
    echo "  - ../datasets/SimpleWikipedia_v2/simple.aligned"
    echo "  - ../datasets/SimpleWikipedia_v2/normal.aligned"
    echo ""
    echo "Please ensure the datasets directory is in the project root."
    echo ""
    exit 1
fi

# Check dataset size
echo "Dataset information:"
SIMPLE_SIZE=$(wc -l < "$SW_SIMPLE" 2>/dev/null || echo "unknown")
NORMAL_SIZE=$(wc -l < "$SW_NORMAL" 2>/dev/null || echo "unknown")
echo "  Simple texts: $SIMPLE_SIZE lines"
echo "  Normal texts: $NORMAL_SIZE lines"
echo ""

# --- Run setup test ---
echo "=================================================="
echo "Running setup verification..."
echo "=================================================="
python test_setup.py 2>/dev/null
SETUP_EXIT=$?

if [ $SETUP_EXIT -ne 0 ]; then
    echo "⚠ WARNING: Setup test failed or not found, but continuing anyway..."
    echo ""
fi

# --- Run OC-SVM Anomaly Detection Pipeline ---
echo "=================================================="
echo "Starting OC-SVM Anomaly Detection Pipeline"
echo "Started at: $(date)"
echo "=================================================="
echo ""
echo "Pipeline configuration (from config.py):"
python -c "
import config
print(f'  Random seed: {config.RANDOM_SEED}')
print(f'  Data split: {config.TRAIN_RATIO}/{config.VAL_RATIO}/{config.TEST_RATIO} (train/val/test)')
print(f'  Embedding type: {config.EMBEDDING_TYPE}')
if config.EMBEDDING_TYPE == 'bert':
    print(f'  BERT model: {config.BERT_MODEL}')
    print(f'  BERT dimensions: {config.BERT_DIM}')
    print(f'  BERT batch size: {config.BERT_BATCH_SIZE}')
else:
    print(f'  GloVe model: {config.GLOVE_MODEL}')
    print(f'  GloVe dimensions: {config.GLOVE_DIM}')
    print(f'  OOV strategy: {config.OOV_STRATEGY}')
print(f'  OC-SVM Hyperparameter grid:')
print(f'    kernel: {config.OCSVM_PARAM_GRID[\"kernel\"]}')
print(f'    nu: {config.OCSVM_PARAM_GRID[\"nu\"]}')
print(f'    gamma: {config.OCSVM_PARAM_GRID[\"gamma\"]}')
" 2>/dev/null || echo "  (could not read config.py)"
echo ""

# Run the OC-SVM pipeline
# Options:
#   --quick              : Skip hyperparameter tuning and plots (fast test)
#   --no-tuning          : Skip hyperparameter tuning (use defaults)
#   --no-plots           : Skip plot generation
#   --reload-data        : Force reload data from raw files
#   --reload-embeddings  : Force recompute embeddings

python main_ocsvm.py
PIPELINE_EXIT_CODE=$?

echo ""
echo "=================================================="
echo "Pipeline completed at: $(date)"
echo "Exit code: $PIPELINE_EXIT_CODE"
echo "=================================================="

# Show results location
if [ $PIPELINE_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ OC-SVM Pipeline completed successfully!"
    echo ""
    echo "Output files saved to: $(pwd)/outputs/"
    echo ""

    if [ -d "outputs" ]; then
        echo "Files created:"
        ls -lh outputs/ 2>/dev/null | grep -v "^total" | grep -v "^d" | awk '{print "  " $9 " (" $5 ")"}'
        echo ""

        # Show key results if evaluation_results_ocsvm.json exists
        if [ -f "outputs/evaluation_results_ocsvm.json" ]; then
            echo "=" * 60
            echo "OC-SVM Key Results:"
            echo "=" * 60
            python -c "
import json
try:
    with open('outputs/evaluation_results_ocsvm.json', 'r') as f:
        results = json.load(f)
    print(f\"  AUROC: {results.get('auroc', 'N/A'):.4f}\")
    print(f\"  Best F1: {results.get('best_f1', 'N/A'):.4f}\")
    print(f\"  Precision: {results.get('precision_at_optimal', 'N/A'):.4f}\")
    print(f\"  Recall: {results.get('recall_at_optimal', 'N/A'):.4f}\")
    print(f\"  Score Separation: {results.get('score_stats', {}).get('separation', 'N/A'):.4f}\")
except:
    print('  (results file could not be read)')
" 2>/dev/null
            echo ""

            # Show comparison with Isolation Forest if available
            if [ -f "outputs/evaluation_results.json" ]; then
                echo "=" * 60
                echo "Comparison: OC-SVM vs Isolation Forest"
                echo "=" * 60
                python -c "
import json
try:
    with open('outputs/evaluation_results_ocsvm.json', 'r') as f:
        ocsvm_results = json.load(f)
    with open('outputs/evaluation_results.json', 'r') as f:
        iforest_results = json.load(f)

    ocsvm_auroc = ocsvm_results.get('auroc', 0)
    iforest_auroc = iforest_results.get('auroc', 0)
    improvement = ocsvm_auroc - iforest_auroc

    print(f\"                    Isolation Forest    OC-SVM      Improvement\")
    print(f\"AUROC:              {iforest_auroc:.4f}             {ocsvm_auroc:.4f}      {improvement:+.4f}\")

    ocsvm_f1 = ocsvm_results.get('best_f1', 0)
    iforest_f1 = iforest_results.get('best_f1', 0)
    f1_improvement = ocsvm_f1 - iforest_f1

    print(f\"F1 Score:           {iforest_f1:.4f}             {ocsvm_f1:.4f}      {f1_improvement:+.4f}\")

    ocsvm_sep = ocsvm_results.get('score_stats', {}).get('separation', 0)
    iforest_sep = iforest_results.get('score_stats', {}).get('separation', 0)
    sep_improvement = ocsvm_sep - iforest_sep

    print(f\"Score Separation:   {iforest_sep:.4f}             {ocsvm_sep:.4f}      {sep_improvement:+.4f}\")
except:
    print('  (could not compare results)')
" 2>/dev/null
                echo ""
            fi
        fi
    fi
else
    echo ""
    echo "⚠ WARNING: Pipeline exited with error code $PIPELINE_EXIT_CODE"
    echo "Check the log files for details:"
    echo "  - slurm_${SLURM_JOB_ID}.out"
    echo "  - slurm_${SLURM_JOB_ID}.err"
    echo "  - outputs/pipeline.log"
    echo ""
fi

echo "=================================================="
echo "Job finished at: $(date)"
echo "=================================================="

# --- Deactivate virtual environment ---
deactivate

exit $PIPELINE_EXIT_CODE
