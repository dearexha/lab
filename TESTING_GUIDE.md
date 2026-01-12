# Testing Guide for Unified Training Loop

## Quick Test on HPC Cluster

Use the test SLURM script to quickly verify that the unified training loop and controller are working correctly.

### Submit Test Job

```bash
chmod +x slurm_job_test.sh
sbatch slurm_job_test.sh
```

### What the Test Does

1. **Uses A40devel partition** (1 hour limit, faster queue)
2. **Reduced resources**: 4 CPUs, 32GB RAM (vs 8 CPUs, 64GB for full run)
3. **Very short training**: Only 200 steps (vs 500,000 in full config)
4. **Frequent validation**: Checks every 50 steps (vs 25,000 in full config)
5. **Smaller batch size**: 4 (vs 8 in full config)

### What to Check

After the test completes, verify:

1. **Job completed successfully**:
   ```bash
   # Check exit code in slurm_test_JOBID.out
   tail slurm_test_JOBID.out
   ```

2. **Controller was created correctly**:
   - Look for log messages showing which controller type is being used
   - Check CSV header format matches controller type

3. **Training loop executed**:
   - Progress bar appeared
   - Training steps were logged
   - Validation ran at expected intervals

4. **Output files created**:
   ```bash
   # Check training CSV was created
   ls -lh results/runs/test_*/training.csv
   
   # View training metrics
   head results/runs/test_*/training.csv
   ```

5. **Correct CSV format**:
   - **Label-based**: Should have `schedule_step,step,train_loss,val_loss,val_perplexity,val_accuracy`
   - **Competence-based**: Should have `step,train_loss,val_loss,val_perplexity,val_accuracy`

6. **Progress bar format**:
   - **Label-based**: `Loss: X.XX | label_subset: [0]`
   - **Competence-based**: `Loss: X.XX | Comp: 0.XXXX`

### Expected Test Duration

- **Setup**: ~2-5 minutes (module loading, environment setup)
- **Training**: ~5-15 minutes (200 steps with small batch)
- **Total**: ~10-20 minutes

### Test Both Controller Types

To test both label-based and competence-based controllers:

1. **Test label-based** (default):
   ```bash
   # Edit config.yaml: label_based: true
   sbatch slurm_job_test.sh
   ```

2. **Test competence-based**:
   ```bash
   # Edit config.yaml: label_based: false
   sbatch slurm_job_test.sh
   ```

### Troubleshooting

#### Test Fails Immediately

1. **Check error log**: `cat slurm_test_JOBID.err`
2. **Common issues**:
   - Dataset not found → Make sure `results/hf_datasets/SimpleWikipedia` exists
   - Module not found → Check `module avail Python`
   - Import errors → Check virtual environment has all packages

#### Test Runs But No Progress

1. **Check if GPU is being used**: Look for CUDA messages in output
2. **Check batch size**: Very small batches might cause issues
3. **Check dataset**: Verify dataset loads correctly

#### Test Takes Too Long

1. **Reduce steps further**: Edit `test_training.py`, change `max_steps = 200` to `max_steps = 50`
2. **Check GPU availability**: `squeue -j JOBID` to see if job is waiting

### Local Testing (Before HPC)

You can also test locally before submitting to HPC:

```bash
# Create test config
python -c "
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['max_steps'] = 50
config['update_every_conv'] = 20
config['batch_size'] = 2
with open('config_test.yaml', 'w') as f:
    yaml.dump(config, f)
"

# Run locally (if you have GPU)
python main.py  # (after temporarily pointing to config_test.yaml)
```

### Success Criteria

The test is successful if:

✅ Job completes without errors  
✅ Training CSV file is created with correct header format  
✅ Progress bar shows correct format (competence or label subset)  
✅ Validation runs at expected intervals  
✅ Model is saved to results/runs/test_*/  
✅ No curriculum-specific errors (e.g., wrong controller type)  

### After Successful Test

Once the test passes, you can:

1. **Submit full training job**: Use `slurm_job_bender.sh` (or your main script)
2. **Increase test steps**: Edit `test_training.py` to use more steps (e.g., 1000) for longer validation
3. **Test specific scenarios**: Modify test to check edge cases (competence reaching 1.0, label schedule exhaustion, etc.)

