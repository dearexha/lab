# Training Loop Refactoring Analysis

## Current State: High Code Duplication

The training loop has **~95% code duplication** between competence-based and label-based CL training. Only ~5% differs in:
- Initialization variables
- Update triggers/conditions
- Logging format
- Progress bar display

## Similarities (Identical Code)

1. **Core training loop structure** (lines 101-127, 212-238)
   - Same `while global_step < config["max_steps"]` loop
   - Same `try/except StopIteration` pattern for data iteration
   - Same batch-to-device transfer
   - Same forward/backward pass
   - Same NaN loss check

2. **Training loss logging** (lines 147-151, 276-279)
   - Both log every 100 steps
   - Only difference: label-based includes `schedule_step` in CSV

3. **Validation evaluation** (lines 130-137, 241-249)
   - Same evaluation call
   - Same logging pattern
   - Only difference: label-based includes `schedule_step` in CSV

4. **Convergence checking** (lines 154-179, 241-274)
   - Same patience counter logic
   - Same best model state tracking
   - Same validation loss comparison
   - Same model state restoration

## Key Differences

### 1. **Initialization State**
- **Competence-based**: Tracks `current_competence` (float)
- **Label-based**: Tracks `current_schedule_step` (int) and `current_label_subset` (list)

### 2. **Update Triggers**
- **Competence-based**:
  - Updates subset at `update_every_competence` when `current_competence < 1`
  - Convergence check at `update_every_conv` when `current_competence == 1`
- **Label-based**:
  - Convergence check at `update_every_conv` (always)
  - Updates subset only when patience exhausted

### 3. **Update Behavior**
- **Competence-based**: 
  - Updates subset → recomputes competence → recreates dataloader
  - No patience reset on update
- **Label-based**:
  - On patience exhaustion: moves to next schedule step → recreates dataloader
  - Resets patience counter on subset update

### 4. **Logging Format**
- **Competence-based**: `step,train_loss,val_loss,val_perplexity,val_accuracy`
- **Label-based**: `schedule_step,step,train_loss,val_loss,val_perplexity,val_accuracy`

### 5. **Progress Bar Display**
- **Competence-based**: Shows `Loss: {loss:.2f} | Comp: {current_competence:.4f}`
- **Label-based**: Shows `Loss: {loss:.2f} | label_subset: {current_label_subset}`

## Proposed Unified Approach

### Strategy: Extract Common Logic, Parameterize Differences

1. **Unified State Object**
   - Create a state object that can hold either competence or schedule info
   - Use helper methods to get display string and logging values

2. **Unified Update Logic**
   - Create a function `should_update_subset(global_step, state, config)` that returns:
     - `(should_update, reason)` where reason is "competence", "convergence", or None
   - Create a function `update_subset(cl_scheduler, global_step, state, config)` that handles both cases

3. **Unified Logging**
   - Create a function `format_csv_log(global_step, train_loss, val_loss, val_perplexity, val_accuracy, state, config)` 
   - Returns appropriate format based on `label_based` flag

4. **Unified Progress Bar**
   - Create a function `get_progress_bar_string(loss, state, config)` 
   - Returns appropriate string based on `label_based` flag

### Benefits

1. **Single source of truth** for training logic
2. **Easier maintenance** - bug fixes apply to both modes
3. **Easier testing** - test one loop instead of two
4. **Reduced code size** - from ~210 lines to ~120 lines
5. **Better readability** - differences are explicit and localized

### Implementation Structure

```python
def train_model(...):
    # Common initialization
    optimizer = ...
    data_collator = ...
    val_dataloader = ...
    
    # Initialize state based on label_based flag
    state = initialize_training_state(cl_scheduler, config)
    
    # Setup logging header
    setup_csv_header(config)
    
    # Unified training loop
    while global_step < config["max_steps"]:
        # Common training step
        batch = get_next_batch(data_iter, train_dataloader)
        loss = train_step(model, optimizer, batch, device)
        
        if torch.isnan(loss):
            continue
        
        global_step += 1
        
        # Unified update logic
        update_result = check_and_update_subset(
            global_step, state, cl_scheduler, config, 
            model, device, val_dataloader, patience_counter, 
            best_val_loss, best_model_state
        )
        
        if update_result.should_stop:
            break
        if update_result.should_recreate_dataloader:
            train_dataloader, data_iter = recreate_dataloader(...)
        
        # Common logging and progress bar
        log_training_step(global_step, loss, state, config)
        update_progress_bar(pbar, loss, state, config)
```

### Key Abstraction Points

1. **State Management**: `TrainingState` class with methods:
   - `get_display_string()` - for progress bar
   - `get_log_prefix()` - for CSV logging
   - `should_check_convergence(global_step, config)` - when to check
   - `should_update_subset(global_step, config)` - when to update

2. **Update Logic**: Single function that handles:
   - Competence updates (when competence < 1)
   - Convergence checks (when competence == 1 or always for label-based)
   - Label schedule progression (when patience exhausted)

3. **Conditional Execution**: Use `if/elif` blocks only where truly different, otherwise unified code

### Potential Challenges

1. **Complex conditional logic**: Need to ensure all edge cases are handled
2. **State synchronization**: Must keep state in sync with `cl_scheduler` internal state
3. **Testing**: Need to verify both paths work correctly after refactoring

### Recommendation

**Yes, this refactoring is highly beneficial and feasible.** The code duplication is significant enough that maintaining two separate loops is error-prone and inefficient. The differences are well-contained and can be abstracted cleanly.

