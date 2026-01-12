# Full Setup Workflow - Running from Scratch

This guide explains how to run the project from the very beginning when you have no pre-computed datasets.

## Overview

The project requires several stages to be run in order:

1. **Create initial tokenized dataset** (`create_initial_dataset.py`)
   - Creates `SimpleWikipedia_tokenized_and_measured` with basic metrics
   - Needed by `composite_difficulty_metric.py`

2. **Train difficulty classifier** (`composite_difficulty_metric.py`)
   - Trains a logistic regression classifier on difficulty metrics
   - Creates `results/difficulty_classifier_normalized_input/`

3. **Create full datasets** (`create_full_dataset.py`)
   - Creates final datasets with all difficulty metrics using the trained classifier
   - Creates `results/hf_datasets/SimpleWikipedia` and `OneStopEnglish`

4. **Run training** (`main.py`)
   - Trains the BERT model with curriculum learning
   - Uses the pre-computed datasets

## Quick Start: Using the Updated SLURM Script

The updated `slurm_job_bender.sh` automatically runs all stages in order:

```bash
sbatch slurm_job_bender.sh
```

The script will:
- Check what already exists
- Skip stages that are already complete
- Run only the necessary stages
- Proceed to training once everything is ready

## Manual Execution (If Needed)

If you prefer to run stages manually or need to debug:

### Stage 1: Create Initial Dataset

```bash
python create_initial_dataset.py
```

**Creates:** `results/hf_datasets/SimpleWikipedia_tokenized_and_measured`

**Requirements:**
- Raw datasets in `datasets/SimpleWikipedia_v2/`
- GPU recommended (for perplexity computation)

**Time:** ~30-60 minutes depending on GPU

### Stage 2: Train Difficulty Classifier

```bash
python composite_difficulty_metric.py
```

**Creates:** `results/difficulty_classifier_normalized_input/`
- `log.json` - Configuration and results
- `model_seed_*.pt` - Trained classifier models

**Requirements:**
- `SimpleWikipedia_tokenized_and_measured` from Stage 1

**Time:** ~10-30 minutes

### Stage 3: Create Full Datasets

```bash
python create_full_dataset.py
```

**Creates:**
- `results/hf_datasets/SimpleWikipedia/`
- `results/hf_datasets/OneStopEnglish/`

**Requirements:**
- Classifier from Stage 2
- Raw datasets in `datasets/OneStopEnglish/`

**Time:** ~1-2 hours depending on GPU

### Stage 4: Run Training

```bash
python main.py
```

**Requirements:**
- Full datasets from Stage 3
- `config.yaml` configured

**Time:** Hours to days depending on `max_steps` in config

## File Dependencies

```
Raw Data (datasets/)
    ↓
Stage 1: create_initial_dataset.py
    ↓
SimpleWikipedia_tokenized_and_measured
    ↓
Stage 2: composite_difficulty_metric.py
    ↓
difficulty_classifier_normalized_input/
    ↓
Stage 3: create_full_dataset.py
    ↓
SimpleWikipedia/ + OneStopEnglish/
    ↓
Stage 4: main.py (training)
```

## Checking Progress

After each stage, verify the output:

```bash
# After Stage 1
ls -la results/hf_datasets/SimpleWikipedia_tokenized_and_measured/

# After Stage 2
ls -la results/difficulty_classifier_normalized_input/
cat results/difficulty_classifier_normalized_input/log.json | head -20

# After Stage 3
ls -la results/hf_datasets/SimpleWikipedia/
ls -la results/hf_datasets/OneStopEnglish/

# After Stage 4
ls -la results/runs/
```

## Troubleshooting

### Stage 1 Fails: "Dataset not found"
- Make sure raw datasets exist in `datasets/SimpleWikipedia_v2/`
- Check file permissions

### Stage 2 Fails: "SimpleWikipedia_tokenized_and_measured not found"
- Run Stage 1 first
- Check the path is correct

### Stage 3 Fails: "log.json not found"
- Run Stage 2 first
- Verify classifier training completed successfully

### Stage 4 Fails: "SimpleWikipedia dataset not found"
- Run Stage 3 first
- Make sure `main.py` has lines 64-98 commented out

## Time Estimates

For a full run from scratch on a GPU:

- **Stage 1:** 30-60 minutes
- **Stage 2:** 10-30 minutes  
- **Stage 3:** 1-2 hours
- **Stage 4:** Hours to days (depends on config)

**Total setup time:** ~2-4 hours (before training)

## Resuming After Interruption

The SLURM script automatically checks what exists and skips completed stages. If you need to resume manually:

1. Check which stages are complete (see "Checking Progress" above)
2. Run only the missing stages
3. Continue to the next stage

## Notes

- Each stage can be run independently once prerequisites are met
- Stages are idempotent (safe to re-run, but may overwrite existing data)
- GPU is highly recommended for Stages 1, 3, and 4
- Stage 2 can run on CPU but will be slower

