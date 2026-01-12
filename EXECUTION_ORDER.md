# Execution Order Guide

## Quick Answer: What to Run First?

**It depends on what you already have!** Check what exists first, then follow the appropriate path below.

## Step 0: Check What You Have

Run these checks to see what's already set up:

```bash
# Check if pre-computed dataset exists (most common case)
ls -la results/hf_datasets/SimpleWikipedia

# Check if difficulty classifier exists
ls -la results/difficulty_classifier_normalized_input/

# Check if tokenized dataset exists (needed for composite_difficulty_metric.py)
ls -la results/hf_datasets/SimpleWikipedia_tokenized_and_measured
```

## Execution Paths

### Path A: Everything Already Exists (Most Common - Just Run Training)

**If you have:**
- ✅ `results/hf_datasets/SimpleWikipedia` exists

**Then:**
1. **Just run `main.py`** (make sure lines 64-98 are commented out)
   ```bash
   python main.py
   ```

**That's it!** You're done. The training will start.

---

### Path B: Need to Create Everything from Scratch

**If you have:**
- ❌ Nothing in `results/` directory
- ✅ Raw datasets in `datasets/SimpleWikipedia_v2/` and `datasets/OneStopEnglish/`

**Then run in this order:**

#### Step 1: Create Tokenized Dataset (if needed)

**Check if this exists:**
```bash
ls -la results/hf_datasets/SimpleWikipedia_tokenized_and_measured
```

**If it doesn't exist**, you need to create it first. This is typically done by:
- Running a script that tokenizes the raw datasets
- OR it might be created as part of the dataset creation process in `main.py`

**Note:** If `composite_difficulty_metric.py` needs this, but you don't have it, you may need to:
1. Uncomment the dataset creation section in `main.py` (lines 64-98)
2. But first, you need the difficulty classifier... (chicken-and-egg problem)

**Solution:** If you're starting from scratch, you might need to:
- Create a basic tokenized dataset first (without difficulty metrics)
- Then run `composite_difficulty_metric.py`
- Then run `main.py` with dataset creation uncommented

#### Step 2: Train Difficulty Classifier

```bash
python composite_difficulty_metric.py
```

**This creates:**
- `results/difficulty_classifier_normalized_input/log.json`
- `results/difficulty_classifier_normalized_input/model_seed_*.pt` files

**Requirements:**
- Needs `results/hf_datasets/SimpleWikipedia_tokenized_and_measured` to exist
- This dataset should have basic metrics computed (sentence_length_words, word_rarity_words, etc.)

#### Step 3: Create Full Dataset with Difficulty Metrics

1. **Uncomment lines 64-98 in `main.py`**
2. **Run:**
   ```bash
   python main.py
   ```

**This creates:**
- `results/hf_datasets/SimpleWikipedia/` (with all difficulty metrics)
- `results/hf_datasets/OneStopEnglish/` (with all difficulty metrics)

#### Step 4: Run Training (Normal Mode)

1. **Comment out lines 64-98 in `main.py` again**
2. **Run:**
   ```bash
   python main.py
   ```

---

## Recommended Workflow (If Starting Fresh)

If you're starting completely from scratch, here's the recommended order:

### Option 1: If You Have Pre-computed Datasets Available

**Best approach:** Get the pre-computed datasets from your team/cluster and skip to Path A.

### Option 2: If You Must Create Everything

1. **First, check what raw data you have:**
   ```bash
   ls -la datasets/SimpleWikipedia_v2/
   ls -la datasets/OneStopEnglish/
   ```

2. **Create initial tokenized dataset** (you may need to write a script or modify existing code to do this without difficulty metrics first)

3. **Run difficulty classifier training:**
   ```bash
   python composite_difficulty_metric.py
   ```

4. **Create full datasets with metrics:**
   - Uncomment lines 64-98 in `main.py`
   - Run `python main.py`

5. **Run training:**
   - Comment lines 64-98 back
   - Run `python main.py`

---

## For Your Current Error

Based on your error, you're in **Path A** but the code thinks you're in **Path B**.

**Quick Fix:**
1. Check that `results/hf_datasets/SimpleWikipedia` exists
2. Make sure lines 64-98 in `main.py` are **commented out**
3. Run `python main.py`

**If the dataset doesn't exist:**
- You need to follow Path B (create everything from scratch)
- OR get the pre-computed dataset from your team

---

## Summary Table

| What You Have | What to Run | Order |
|--------------|-------------|-------|
| `SimpleWikipedia` dataset exists | `main.py` | 1 |
| Nothing exists, have raw data | `composite_difficulty_metric.py` → `main.py` (uncommented) → `main.py` (commented) | 1, 2, 3 |
| Only `SimpleWikipedia_tokenized_and_measured` exists | `composite_difficulty_metric.py` → `main.py` (uncommented) → `main.py` (commented) | 1, 2, 3 |

---

## File Dependencies

```
Raw Data (datasets/)
    ↓
Tokenized Dataset (results/hf_datasets/SimpleWikipedia_tokenized_and_measured)
    ↓
Difficulty Classifier (composite_difficulty_metric.py)
    ↓
    Creates: results/difficulty_classifier_normalized_input/
    ↓
Full Dataset with Metrics (main.py with lines 64-98 uncommented)
    ↓
    Creates: results/hf_datasets/SimpleWikipedia/
    ↓
Training (main.py with lines 64-98 commented)
```

