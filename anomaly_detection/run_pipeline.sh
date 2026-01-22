#!/bin/bash
#SBATCH --job-name=isolation_forest_glove
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=batch

# ============================================================================
# SLURM Job Script for Anomaly Detection Pipeline
# Isolation Forest + GloVe Embeddings
# ============================================================================

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Number of CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: $SLURM_MEM_PER_NODE MB"
echo "============================================================================"

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

# Load modules (adjust based on your HPC environment)
# module load python/3.9
# module load gcc/11.2.0

# Activate virtual environment if using one
# source /path/to/venv/bin/activate

# Or use conda environment
# conda activate your_env_name

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ============================================================================
# WORKING DIRECTORY
# ============================================================================

cd /home/user/lab/anomaly_detection || exit 1

echo "Working directory: $(pwd)"
echo "Python version: $(python --version)"
echo "============================================================================"

# ============================================================================
# INSTALL DEPENDENCIES (if needed)
# ============================================================================

# Uncomment if dependencies not installed
# echo "Installing dependencies..."
# pip install -r requirements.txt --user
# echo "Dependencies installed"
# echo "============================================================================"

# ============================================================================
# RUN PIPELINE
# ============================================================================

echo "Starting pipeline..."
echo ""

# Default: Full pipeline with hyperparameter tuning
python main.py

# Alternative options:
# python main.py --quick                    # Quick test (no tuning, no plots)
# python main.py --no-tuning               # Skip hyperparameter tuning
# python main.py --no-plots                # Skip plot generation
# python main.py --reload-data             # Force reload data
# python main.py --reload-embeddings       # Force recompute embeddings

EXIT_CODE=$?

echo ""
echo "============================================================================"
echo "Pipeline finished with exit code: $EXIT_CODE"
echo "Job ended at: $(date)"
echo "============================================================================"

# ============================================================================
# SUMMARY
# ============================================================================

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "SUCCESS! Check outputs in: /home/user/lab/anomaly_detection/outputs/"
    echo ""
    echo "Key files:"
    echo "  - Model: outputs/best_isolation_forest.pkl"
    echo "  - Results: outputs/evaluation_results.json"
    echo "  - Hyperparameters: outputs/hyperparameter_tuning_results.json"
    echo "  - Plots: outputs/*.png"
    echo "  - Log: outputs/pipeline.log"
    echo ""
else
    echo ""
    echo "FAILED! Check the log for errors: outputs/pipeline.log"
    echo ""
fi

exit $EXIT_CODE
