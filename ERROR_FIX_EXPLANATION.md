# Error Fix Explanation: FileNotFoundError for log.json

## Problem Summary

You're getting this error:
```
FileNotFoundError: [Errno 2] No such file or directory: '/home/s07drexh/mlai-lab/lab_beyond_shallow_heuristics-main/results/difficulty_classifier_normalized_input//log.json'
```

## Root Cause

The error occurs because:

1. **On your server, lines 71-72 in `main.py` are UNCOMMENTED** (even though they're commented in the local file)
2. When uncommented, the code tries to read `results/difficulty_classifier_normalized_input/log.json`
3. This file **only exists** if you've run `composite_difficulty_metric.py` first
4. Since you haven't run that script, the file doesn't exist → FileNotFoundError

## Why This Happens

The `main.py` file has two modes:

### Mode 1: Using Pre-computed Datasets (NORMAL WORKFLOW)
- Lines 64-98 are **commented out**
- Code loads datasets from disk: `load_from_disk(SW_SAVE_PATH)`
- This is the default mode when datasets already exist

### Mode 2: Creating Datasets from Scratch
- Lines 64-98 are **uncommented**
- Requires:
  1. Running `composite_difficulty_metric.py` first to create the difficulty classifier
  2. This creates `results/difficulty_classifier_normalized_input/log.json` and model files
  3. Then `main.py` can use these to compute difficulty metrics

## Solution

You have **two options**:

### Option 1: Keep Using Pre-computed Datasets (RECOMMENDED)

**Make sure lines 64-98 in `main.py` are COMMENTED OUT:**

```python
#================================== HF Dataset Creation (comment out if loading exisiting via load_from_disk) ================================================
# logging.info("HF Dataset Creation ...")
# ... (all lines commented)
#=============================================================================================================================================================
# load if metrics computation above already done
sw_dataset_dict = load_from_disk(SW_SAVE_PATH)  # This line should be active
```

**Verify on your server:**
```bash
# Check if the dataset exists
ls -la results/hf_datasets/SimpleWikipedia

# If it exists, make sure lines 64-98 are commented in main.py
```

### Option 2: Create Datasets from Scratch

If you need to create the datasets:

1. **First, run the difficulty classifier training:**
   ```bash
   python composite_difficulty_metric.py
   ```
   This will create:
   - `results/difficulty_classifier_normalized_input/log.json`
   - `results/difficulty_classifier_normalized_input/model_seed_*.pt` files

2. **Then uncomment lines 64-98 in `main.py`**

3. **Run main.py:**
   ```bash
   python main.py
   ```

## What I Fixed

I've added error checking to the commented code section that will:
- Check if required files exist before trying to read them
- Provide clear error messages explaining what's missing and how to fix it
- This helps prevent confusion if someone accidentally uncomments those lines

## Quick Check Commands

On your server, run these to diagnose:

```bash
# 1. Check if dataset exists (should exist for normal workflow)
ls -la results/hf_datasets/SimpleWikipedia

# 2. Check if difficulty classifier files exist (only needed if creating datasets)
ls -la results/difficulty_classifier_normalized_input/

# 3. Check if lines 64-98 are commented in main.py
sed -n '64,98p' main.py | head -5  # Should show commented lines starting with #
```

## Expected File Structure

### For Normal Workflow (Pre-computed datasets):
```
results/
└── hf_datasets/
    └── SimpleWikipedia/          # ← This should exist
        ├── train/
        ├── validation/
        └── test/
```

### For Dataset Creation:
```
results/
├── difficulty_classifier_normalized_input/  # ← Created by composite_difficulty_metric.py
│   ├── log.json                              # ← Required file
│   ├── model_seed_0.pt
│   ├── model_seed_1.pt
│   └── ...
└── hf_datasets/
    └── SimpleWikipedia/          # ← Created by main.py after uncommenting
```

## Most Likely Fix

**On your server, check `main.py` and make sure lines 64-98 are commented out.** The error suggests they're currently uncommented.

You can verify with:
```bash
grep -n "EXPERIMENT_DIR = os.path.join" main.py
```

If this line is NOT commented (doesn't start with `#`), that's your problem!

