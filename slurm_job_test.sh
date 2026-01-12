#!/bin/bash
#SBATCH --job-name=cl_test              # Job name
#SBATCH --output=slurm_test_%j.out      # Standard output log
#SBATCH --error=slurm_test_%j.err       # Standard error log
#SBATCH --partition=A40devel            # A40 devel partition (1 hour limit for quick testing)
#SBATCH --time=1:00:00                  # Time limit (1 hour for testing)
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks=1                      # Number of tasks
#SBATCH --cpus-per-task=4               # CPUs per task (reduced for testing)
#SBATCH --gpus=1                        # Request 1 GPU
#SBATCH --mem=32GB                      # Memory allocation (reduced for testing)

# --- Print job info ---
echo "=================================================="
echo "TEST JOB: Starting SLURM Job: $SLURM_JOB_NAME (ID: $SLURM_JOB_ID)"
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

# --- Python virtual environment setup ---
VENV_DIR="$HOME/cl_training_env"  # Path to your virtual environment

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    python -m venv "$VENV_DIR"
fi

# Activate the virtual environment
source "$VENV_DIR/bin/activate"

# Optional: install dependencies (if not already installed)
pip install --upgrade pip
pip install --no-cache-dir torch transformers tqdm datasets pyyaml nltk pyphen scikit-learn numpy

# --- Environment variables ---
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# --- Print environment info ---
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'PyTorch not available')"
if command -v nvidia-smi &> /dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    echo "GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)"
fi

# --- Change to project directory ---
cd $SLURM_SUBMIT_DIR

# --- Create test config override ---
echo "=================================================="
echo "Creating test configuration with reduced steps..."
echo "=================================================="

# Create a Python script to run training with test config
cat > test_training.py << 'TESTSCRIPT'
import sys
import yaml
import logging
import logging.config
from datetime import datetime
import os
from datasets import load_from_disk
from transformers import BertTokenizerFast, BertForMaskedLM, BertConfig
import torch

import utils
from training.CL_Scheduler import CL_Scheduler
from training.CompetenceFunction import CompetenceFunction, sqrt_competence_func
from training.training_loop import train_model

def setup_logging():
    logging_config_path = os.path.join(utils.get_project_dir(), "logging_config.yaml")
    training_config_path = os.path.join(utils.get_project_dir(), "config.yaml")

    with open(logging_config_path, "r") as f:
        logging_config = yaml.safe_load(f.read())

    with open(training_config_path, "r") as f:
        experiment_config = yaml.safe_load(f.read())
    
    # OVERRIDE CONFIG FOR TESTING
    experiment_config["max_steps"] = 200  # Very short test
    experiment_config["update_every_conv"] = 50  # Check convergence more frequently
    if "update_every_competence" in experiment_config:
        experiment_config["update_every_competence"] = 50  # Update competence more frequently
    experiment_config["batch_size"] = 4  # Smaller batch for testing
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(utils.get_project_dir(), "results", "runs", f"test_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    experiment_config["meta"] = {"start_time": timestamp, "test_run": True}

    with open(os.path.join(run_dir, "config.yaml"), 'w') as f:
        yaml.dump(experiment_config, f)

    logging_config["handlers"]["training_log_file"]["filename"] = os.path.join(run_dir, "training.csv")

    logging.config.dictConfig(logging_config)
    return experiment_config, run_dir

if __name__=="__main__":
    config, run_dir = setup_logging()
    logger = logging.getLogger(__name__)
    
    utils.set_seed(config["seed"])

    # tokenizer
    model_id = "bert-base-uncased"
    tokenizer = BertTokenizerFast.from_pretrained(model_id)

    # Load dataset
    SW_SAVE_PATH = os.path.join(utils.get_project_dir(), "results/hf_datasets/SimpleWikipedia")
    sw_dataset_dict = load_from_disk(SW_SAVE_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_config = BertConfig(**config["bert_config"])
    model = BertForMaskedLM(bert_config).to(device)
    logger.info(f"Initialized BERT model for MLM on device {device}")
    logger.info(f"TEST RUN: max_steps={config['max_steps']}, batch_size={config['batch_size']}")

    # Test both controller types
    test_label_based = config.get("label_based", True)
    logger.info(f"Testing {'label-based' if test_label_based else 'competence-based'} controller")
    
    cl_scheduler = CL_Scheduler(
        sw_dataset_dict, 
        label_based=test_label_based, 
        difficulty_metric_name="label" if test_label_based else config.get("difficulty_metric", "fre_score_words"),
        label_schedule=[[0], [1]] if test_label_based else None,
        tokenizer=tokenizer, 
        competence_func=CompetenceFunction(sqrt_competence_func, config["T"], config["c0"])
    )

    logger.info("Starting TEST training...")
    train_model(model=model, device=device, tokenizer=tokenizer, cl_scheduler=cl_scheduler, config=config)

    logger.info("Saving the final model...")
    model.save_pretrained(run_dir)
    logger.info(f"Model saved to {run_dir}.")
    logger.info("TEST COMPLETED SUCCESSFULLY!")
TESTSCRIPT

# --- Run test training script ---
echo "=================================================="
echo "Starting TEST training at $(date)"
echo "This will run a very short training (200 steps) to verify the unified loop works"
echo "=================================================="
python test_training.py
echo "Training finished at $(date)"

# --- Completion info ---
echo "=================================================="
echo "TEST Job completed at $(date)"
echo "Exit code: $?"
echo "=================================================="
echo ""
echo "Check the output files:"
echo "  - slurm_test_${SLURM_JOB_ID}.out (standard output)"
echo "  - slurm_test_${SLURM_JOB_ID}.err (standard error)"
echo "  - results/runs/test_*/training.csv (training metrics)"
echo ""

# Deactivate virtual environment
deactivate

