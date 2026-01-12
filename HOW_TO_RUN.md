# How to Run the Project

## Prerequisites

1. **Python 3.8+** (check with `python --version`)
2. **CUDA-capable GPU** (optional but recommended for faster training)
3. **Preprocessed dataset** at `results/hf_datasets/SimpleWikipedia`

## Running on HPC Cluster (SLURM)

If you're running on an HPC cluster with SLURM, see the dedicated guide:
- **Quick Start**: See `slurm_setup_guide.md` for detailed instructions
- **Main Job Script**: `slurm_job.sh` - Edit and submit with `sbatch slurm_job.sh`
- **Test Job Script**: `slurm_job_test.sh` - For quick validation runs
- **Environment Check**: Run `bash check_environment.sh` before submitting

### Quick SLURM Commands:
```bash
# 1. Check environment
bash check_environment.sh

# 2. Edit slurm_job.sh with your cluster settings (partition, modules, etc.)

# 3. Make executable and submit
chmod +x slurm_job.sh
sbatch slurm_job.sh

# 4. Monitor job
squeue -u $USER

# 5. Check output
tail -f slurm_JOBID.out
```

## Step 1: Install Dependencies

Install all required packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

**Note**: If you have CUDA, PyTorch should automatically use it. The requirements.txt includes `torch==2.9.0+cu126` which requires CUDA 12.6. If you don't have CUDA or have a different version, you may need to install PyTorch separately:

```bash
# For CPU only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# For CUDA (adjust version as needed):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Step 2: Verify Dataset Exists

The project expects a preprocessed dataset at:
```
results/hf_datasets/SimpleWikipedia
```

**Check if dataset exists:**
```bash
# From the project root directory
python -c "from datasets import load_from_disk; import os; print('Dataset exists!' if os.path.exists('results/hf_datasets/SimpleWikipedia') else 'Dataset not found!')"
```

**If the dataset doesn't exist**, you need to create it first by uncommenting the dataset creation section in `main.py` (lines 64-98). This requires:
- A perplexity model (BERT)
- A logistic classifier model for difficulty metrics
- The raw datasets in `datasets/SimpleWikipedia_v2/` and `datasets/OneStopEnglish/`

## Step 3: Configure Training

Edit `config.yaml` to set your training parameters:

```yaml
# Key settings:
label_based: true  # or false for competence-based
batch_size: 8
learning_rate: 1.0e-4
max_steps: 500000  # Reduce this for quick testing (e.g., 1000)
update_every_conv: 25000  # Reduce for quick testing (e.g., 100)
patience: 3
```

**For quick testing**, reduce `max_steps` to a small number (e.g., 1000) and `update_every_conv` to a smaller number (e.g., 100).

## Step 4: Run Training

From the project root directory:

```bash
python main.py
```

Or on Windows PowerShell:
```powershell
python main.py
```

## Step 5: Monitor Training

The training will:
1. Create a timestamped run directory: `results/runs/YYYY-MM-DD_HH-MM-SS/`
2. Save training logs to: `results/runs/YYYY-MM-DD_HH-MM-SS/training.csv`
3. Save the final model to: `results/runs/YYYY-MM-DD_HH-MM-SS/`
4. Display progress in the terminal with a progress bar

**What to look for:**
- Progress bar showing loss and curriculum phase (competence or label subset)
- Log messages indicating validation runs
- CSV file being written with training metrics
- No errors or exceptions

## Step 6: Verify Refactored Code Works

To verify the new controller-based implementation works correctly:

### Check 1: Controller is Created
Look for log messages indicating which controller is being used (though this isn't explicitly logged, the behavior will show it).

### Check 2: Training Behavior Matches Original
- **Label-based**: Should advance through label schedule when patience is exhausted
- **Competence-based**: Should expand dataset based on competence function

### Check 3: Logging Format
Check `results/runs/YYYY-MM-DD_HH-MM-SS/training.csv`:
- **Label-based**: Should have header `schedule_step,step,train_loss,val_loss,val_perplexity,val_accuracy`
- **Competence-based**: Should have header `step,train_loss,val_loss,val_perplexity,val_accuracy`

### Check 4: Progress Bar
- **Label-based**: Should show `Loss: X.XX | label_subset: [0]` (or current label subset)
- **Competence-based**: Should show `Loss: X.XX | Comp: 0.XXXX` (current competence)

## Quick Test Run

For a quick test to verify everything works:

1. **Edit `config.yaml`:**
   ```yaml
   max_steps: 500  # Very short run
   update_every_conv: 100
   update_every_competence: 100  # If using competence-based
   ```

2. **Run:**
   ```bash
   python main.py
   ```

3. **Expected output:**
   - Training starts
   - Progress bar appears
   - Validation runs at specified intervals
   - Training completes or stops early
   - Model saved to results/runs/

## Troubleshooting

### Error: "Dataset not found"
- Make sure `results/hf_datasets/SimpleWikipedia` exists
- If not, you need to create the dataset first (uncomment dataset creation code in main.py)

### Error: "CUDA out of memory"
- Reduce `batch_size` in `config.yaml`
- Use CPU instead: The code will automatically fall back to CPU if CUDA is not available

### Error: Import errors
- Make sure you're in the project root directory
- Make sure all dependencies are installed: `pip install -r requirements.txt`

### Error: "No module named 'training'"
- Make sure you're running from the project root directory
- The project structure should have `training/` as a subdirectory

## Testing Both Curriculum Types

To test both curriculum types:

### Test Label-Based:
```yaml
# config.yaml
label_based: true
label_schedule: [[0], [1]]
```

### Test Competence-Based:
```yaml
# config.yaml
label_based: false
difficulty_metric: "fre_score_words"
update_every_competence: 100
T: 1000
c0: 0.05
```

Then run `python main.py` for each configuration.

## Expected Runtime

- **Quick test** (500 steps): ~1-5 minutes
- **Short run** (10,000 steps): ~30 minutes - 1 hour
- **Full run** (500,000 steps): Several hours to days (depending on GPU)

## Output Files

After training, check:
- `results/runs/YYYY-MM-DD_HH-MM-SS/training.csv` - Training metrics
- `results/runs/YYYY-MM-DD_HH-MM-SS/config.yaml` - Saved configuration
- `results/runs/YYYY-MM-DD_HH-MM-SS/` - Saved model files

