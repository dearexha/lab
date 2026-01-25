#!/bin/bash
#SBATCH --job-name=ad_sgd_ocsvm
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --partition=A40short
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB

echo "Job $SLURM_JOB_ID started at $(date)"

# Load Python
module purge
module load Python

# Activate environment (assume it already exists)
source "$HOME/ad_pipeline_env/bin/activate"

# Set threading environment variables for parallel processing
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Navigate to directory
cd "$SLURM_SUBMIT_DIR" || { cd "${SLURM_SUBMIT_DIR}/anomaly_detection" || exit 1; }

# Run pipeline
echo "Starting SGD-OC-SVM pipeline at $(date)"
python main_sgd_ocsvm.py

echo "Job finished at $(date)"
deactivate
