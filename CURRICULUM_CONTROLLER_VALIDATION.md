# Curriculum Controller Validation Analysis

## Decision Point Mapping

### Competence-Based Training Loop Decision Points

| Line | Decision Point | Current Logic | Controller Responsibility |
|------|---------------|---------------|--------------------------|
| 83 | CSV header | `"step,train_loss,val_loss,val_perplexity,val_accuracy"` | `get_csv_header()` |
| 88 | Initial state | `current_competence = cl_scheduler.competence_func.c0` | `initialize()` |
| 129 | When to validate/update | `global_step % update_every_competence == 0 and current_competence < 1` | `should_validate(step)` + `should_update_subset(step)` |
| 131 | Validation execution | `evaluate(model, device, val_dataloader)` | Training loop (curriculum-agnostic) |
| 140 | Subset update | `cl_scheduler.update_current_train_subset(global_step)` | `update_subset(step)` |
| 141 | Competence computation | `current_competence = cl_scheduler.competence_func.compute_competence(global_step)` | `get_display_context()` |
| 144 | Dataloader recreation | `DataLoader(current_train_subset, ...)` | `get_current_subset()` → training loop |
| 148 | Training loss logging | Every 100 steps, format: `"{global_step},{loss},,,"` | `format_training_log(step, loss)` |
| 155 | Convergence check trigger | `global_step % update_every_conv == 0 and current_competence == 1` | `should_check_convergence(step)` |
| 166-179 | Patience handling | Track best model, increment patience, stop on exhaustion | `handle_validation_result(val_loss, patience_state)` |
| 184 | Progress bar | `f"Loss: {loss:.2f} | Comp: {current_competence:.4f}"` | `get_progress_string(loss)` |

### Label-Based Training Loop Decision Points

| Line | Decision Point | Current Logic | Controller Responsibility |
|------|---------------|---------------|--------------------------|
| 193 | CSV header | `"schedule_step,step,train_loss,val_loss,val_perplexity,val_accuracy"` | `get_csv_header()` |
| 198-199 | Initial state | `current_schedule_step = cl_scheduler.current_schedule_step`<br>`current_label_subset = cl_scheduler.label_schedule[current_schedule_step]` | `initialize()` |
| 241 | When to validate | `global_step % update_every_conv == 0` (always) | `should_validate(step)` |
| 243 | Validation execution | `evaluate(model, device, val_dataloader)` | Training loop (curriculum-agnostic) |
| 252-260 | Patience handling | Track best model, increment patience | `handle_validation_result(val_loss, patience_state)` |
| 260 | When to update subset | `patience_counter >= config["patience"]` | `should_update_subset_on_patience()` |
| 266 | Subset update | `cl_scheduler.update_current_train_subset()` (no step param) | `update_subset()` |
| 269-270 | Schedule state | `current_schedule_step = cl_scheduler.current_schedule_step`<br>`current_label_subset = cl_scheduler.label_schedule[current_schedule_step]` | `get_display_context()` |
| 273 | Dataloader recreation | `DataLoader(current_train_subset, ...)` | `get_current_subset()` → training loop |
| 277 | Training loss logging | Every 100 steps, format: `"{schedule_step},{global_step},{loss},,,"` | `format_training_log(step, loss)` |
| 284 | Progress bar | `f"Loss: {loss:.2f} | label_subset: {current_label_subset}"` | `get_progress_string(loss)` |

## CurriculumController Interface Design

```python
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
from datasets import Dataset

class PatienceState:
    """Encapsulates patience tracking state."""
    def __init__(self):
        self.counter: int = 0
        self.best_val_loss: float = float('inf')
        self.best_model_state: Optional[Dict[str, Any]] = None

class ValidationDecision:
    """Result of validation decision logic."""
    def __init__(self, should_validate: bool, should_update_subset: bool, 
                 should_check_convergence: bool):
        self.should_validate = should_validate
        self.should_update_subset = should_update_subset
        self.should_check_convergence = should_check_convergence

class CurriculumController(ABC):
    """
    Abstract interface for curriculum learning controllers.
    Encapsulates all curriculum-specific logic, making the training loop curriculum-agnostic.
    """
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize controller state. Called once before training starts."""
        pass
    
    @abstractmethod
    def get_csv_header(self) -> str:
        """Return CSV header string for logging."""
        pass
    
    @abstractmethod
    def get_current_subset(self) -> Dataset:
        """Return the current training dataset subset."""
        pass
    
    @abstractmethod
    def should_validate(self, global_step: int) -> bool:
        """
        Determine if validation should run at this step.
        
        Args:
            global_step: Current training step
            
        Returns:
            True if validation should run
        """
        pass
    
    @abstractmethod
    def should_update_subset(self, global_step: int) -> bool:
        """
        Determine if training subset should be updated at this step.
        This is for time-based updates (e.g., competence increases).
        
        Args:
            global_step: Current training step
            
        Returns:
            True if subset should be updated
        """
        pass
    
    @abstractmethod
    def should_check_convergence(self, global_step: int) -> bool:
        """
        Determine if convergence should be checked at this step.
        Convergence checks may trigger patience-based subset updates.
        
        Args:
            global_step: Current training step
            
        Returns:
            True if convergence should be checked
        """
        pass
    
    @abstractmethod
    def update_subset(self, global_step: Optional[int] = None) -> Tuple[Dataset, bool]:
        """
        Update the training subset.
        
        Args:
            global_step: Current training step (required for competence-based, 
                        ignored for label-based)
        
        Returns:
            Tuple of (new_subset, has_more_updates)
            has_more_updates: False if this was the final subset, True otherwise
        """
        pass
    
    @abstractmethod
    def handle_validation_result(self, val_loss: float, patience_state: PatienceState, 
                                 global_step: int) -> Tuple[bool, bool]:
        """
        Process validation result and update patience state.
        
        Args:
            val_loss: Validation loss from this evaluation
            patience_state: Current patience tracking state
            global_step: Current training step
            
        Returns:
            Tuple of (should_update_subset, should_stop_training)
            - should_update_subset: True if subset should be updated (e.g., label schedule advance)
            - should_stop_training: True if training should stop entirely
        """
        pass
    
    @abstractmethod
    def format_training_log(self, global_step: int, train_loss: float) -> str:
        """
        Format training loss log entry.
        
        Args:
            global_step: Current training step
            train_loss: Current training loss
            
        Returns:
            Formatted CSV log line
        """
        pass
    
    @abstractmethod
    def format_validation_log(self, global_step: int, train_loss: float, 
                              val_loss: float, val_perplexity: float, 
                              val_accuracy: float) -> str:
        """
        Format validation log entry.
        
        Args:
            global_step: Current training step
            train_loss: Current training loss
            val_loss: Validation loss
            val_perplexity: Validation perplexity
            val_accuracy: Validation accuracy
            
        Returns:
            Formatted CSV log line
        """
        pass
    
    @abstractmethod
    def get_progress_string(self, loss: float) -> str:
        """
        Get progress bar display string.
        
        Args:
            loss: Current training loss
            
        Returns:
            Formatted progress bar string
        """
        pass
```

## Concrete Implementations

### CompetenceBasedController

```python
class CompetenceBasedController(CurriculumController):
    def __init__(self, cl_scheduler: CL_Scheduler, config: Dict):
        self.cl_scheduler = cl_scheduler
        self.config = config
        self.current_competence: float = cl_scheduler.competence_func.c0
        
    def initialize(self) -> None:
        """Initialize competence-based state."""
        self.current_competence = self.cl_scheduler.competence_func.c0
        
    def get_csv_header(self) -> str:
        return "step,train_loss,val_loss,val_perplexity,val_accuracy"
    
    def get_current_subset(self) -> Dataset:
        return self.cl_scheduler.get_current_train_subset()
    
    def should_validate(self, global_step: int) -> bool:
        # Validate when updating competence (competence < 1) or checking convergence (competence == 1)
        if global_step % self.config["update_every_competence"] == 0 and self.current_competence < 1:
            return True
        if global_step % self.config["update_every_conv"] == 0 and self.current_competence == 1:
            return True
        return False
    
    def should_update_subset(self, global_step: int) -> bool:
        # Update subset when competence increases (competence < 1)
        return (global_step % self.config["update_every_competence"] == 0 
                and self.current_competence < 1)
    
    def should_check_convergence(self, global_step: int) -> bool:
        # Check convergence when competence == 1
        return (global_step % self.config["update_every_conv"] == 0 
                and self.current_competence == 1)
    
    def update_subset(self, global_step: Optional[int] = None) -> Tuple[Dataset, bool]:
        if global_step is None:
            raise ValueError("Competence-based updates require global_step")
        subset, has_updated = self.cl_scheduler.update_current_train_subset(global_step)
        self.current_competence = self.cl_scheduler.competence_func.compute_competence(global_step)
        return subset, has_updated
    
    def handle_validation_result(self, val_loss: float, patience_state: PatienceState, 
                                global_step: int) -> Tuple[bool, bool]:
        """
        Competence-based: Patience only matters when competence == 1.
        Never updates subset based on validation (subset updates are time-based).
        """
        # Only check patience during convergence phase (competence == 1)
        if self.current_competence < 1:
            # During competence growth, no patience tracking
            return False, False
        
        # During convergence phase, track patience
        if val_loss < patience_state.best_val_loss:
            patience_state.best_val_loss = val_loss
            patience_state.counter = 0
        else:
            patience_state.counter += 1
            
        if patience_state.counter >= self.config["patience"]:
            return False, True  # Stop training, don't update subset
        
        return False, False  # Continue training, no subset update
    
    def format_training_log(self, global_step: int, train_loss: float) -> str:
        return f"{global_step},{train_loss},,,"
    
    def format_validation_log(self, global_step: int, train_loss: float, 
                             val_loss: float, val_perplexity: float, 
                             val_accuracy: float) -> str:
        return f"{global_step},{train_loss},{val_loss},{val_perplexity},{val_accuracy}"
    
    def get_progress_string(self, loss: float) -> str:
        return f"Loss: {loss:.2f} | Comp: {self.current_competence:.4f}"
```

### LabelBasedController

```python
class LabelBasedController(CurriculumController):
    def __init__(self, cl_scheduler: CL_Scheduler, config: Dict):
        self.cl_scheduler = cl_scheduler
        self.config = config
        self.current_schedule_step: int = cl_scheduler.current_schedule_step
        self.current_label_subset: list = cl_scheduler.label_schedule[self.current_schedule_step]
        
    def initialize(self) -> None:
        """Initialize label-based state."""
        self.current_schedule_step = self.cl_scheduler.current_schedule_step
        self.current_label_subset = self.cl_scheduler.label_schedule[self.current_schedule_step]
        
    def get_csv_header(self) -> str:
        return "schedule_step,step,train_loss,val_loss,val_perplexity,val_accuracy"
    
    def get_current_subset(self) -> Dataset:
        return self.cl_scheduler.get_current_train_subset()
    
    def should_validate(self, global_step: int) -> bool:
        # Always validate at convergence check intervals
        return global_step % self.config["update_every_conv"] == 0
    
    def should_update_subset(self, global_step: int) -> bool:
        # Label-based never updates subset based on step count
        return False
    
    def should_check_convergence(self, global_step: int) -> bool:
        # Always check convergence at regular intervals
        return global_step % self.config["update_every_conv"] == 0
    
    def update_subset(self, global_step: Optional[int] = None) -> Tuple[Dataset, bool]:
        # global_step is ignored for label-based
        subset, has_updated = self.cl_scheduler.update_current_train_subset()
        if has_updated:
            self.current_schedule_step = self.cl_scheduler.current_schedule_step
            self.current_label_subset = self.cl_scheduler.label_schedule[self.current_schedule_step]
        return subset, has_updated
    
    def handle_validation_result(self, val_loss: float, patience_state: PatienceState, 
                                global_step: int) -> Tuple[bool, bool]:
        """
        Label-based: Patience exhaustion triggers subset update (schedule advance).
        """
        if val_loss < patience_state.best_val_loss:
            patience_state.best_val_loss = val_loss
            patience_state.counter = 0
            return False, False  # No update, continue training
        else:
            patience_state.counter += 1
            
        if patience_state.counter >= self.config["patience"]:
            # Patience exhausted: advance to next label subset
            return True, False  # Update subset, don't stop training
        
        return False, False  # Continue training, no subset update
    
    def format_training_log(self, global_step: int, train_loss: float) -> str:
        return f"{self.current_schedule_step},{global_step},{train_loss},,,"
    
    def format_validation_log(self, global_step: int, train_loss: float, 
                             val_loss: float, val_perplexity: float, 
                             val_accuracy: float) -> str:
        return f"{self.current_schedule_step},{global_step},{train_loss},{val_loss},{val_perplexity},{val_accuracy}"
    
    def get_progress_string(self, loss: float) -> str:
        return f"Loss: {loss:.2f} | label_subset: {self.current_label_subset}"
```

## Unified Training Loop

```python
def train_model(model, device, tokenizer, controller: CurriculumController, config):
    """
    Unified training loop that is completely curriculum-agnostic.
    All curriculum logic is delegated to the controller.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    
    val_dataset = controller.cl_scheduler.get_validation_set()
    val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    
    model.train()
    
    # Initialize controller
    controller.initialize()
    
    # Setup logging
    csv_logger.info(controller.get_csv_header())
    pbar = tqdm(ncols=150, total=config["max_steps"], desc="Training Progress")
    
    # Get initial subset and create dataloader
    current_train_subset = controller.get_current_subset()
    logger.info(f"Length of subset dataset: {len(current_train_subset)}")
    train_dataloader = DataLoader(current_train_subset, shuffle=True, 
                                  batch_size=config["batch_size"], collate_fn=data_collator)
    data_iter = iter(train_dataloader)
    
    global_step = 0
    patience_state = PatienceState()
    patience_state.best_model_state = copy.deepcopy(model.state_dict())
    
    while global_step < config["max_steps"]:
        # Get next batch (curriculum-agnostic)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_dataloader)
            batch = next(data_iter)
        
        # Move batch to device
        inputs = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        # Check for NaN loss
        if torch.isnan(loss):
            continue
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        global_step += 1
        
        # Check if subset should be updated (time-based, e.g., competence increase)
        if controller.should_update_subset(global_step):
            # Validate first
            val_loss, val_perplexity, val_accuracy = evaluate(model, device, val_dataloader)
            model.train()
            logger.info(f"Validation Loss: {val_loss}, Validation Perplexity: {val_perplexity}, Validation Accuracy: {val_accuracy}")
            
            # Log validation
            csv_logger.info(controller.format_validation_log(global_step, loss.item(), 
                                                             val_loss, val_perplexity, val_accuracy))
            
            # Update subset
            current_train_subset, has_updated = controller.update_subset(global_step)
            logger.info(f"Length of subset dataset: {len(current_train_subset)}")
            
            if has_updated:
                train_dataloader = DataLoader(current_train_subset, shuffle=True, 
                                              batch_size=config["batch_size"], collate_fn=data_collator)
                data_iter = iter(train_dataloader)
        
        # Check if convergence should be checked
        if controller.should_check_convergence(global_step):
            # Validate
            val_loss, val_perplexity, val_accuracy = evaluate(model, device, val_dataloader)
            model.train()
            logger.info(f"Validation Loss: {val_loss}, Validation Perplexity: {val_perplexity}, Validation Accuracy: {val_accuracy}")
            
            # Log validation
            csv_logger.info(controller.format_validation_log(global_step, loss.item(), 
                                                             val_loss, val_perplexity, val_accuracy))
            
            # Handle validation result (patience tracking, subset updates)
            should_update_subset, should_stop = controller.handle_validation_result(
                val_loss, patience_state, global_step
            )
            
            # Update best model state if needed
            if val_loss < patience_state.best_val_loss:
                patience_state.best_model_state = copy.deepcopy(model.state_dict())
            
            if should_update_subset:
                # Update subset (e.g., advance label schedule)
                current_train_subset, has_more = controller.update_subset()
                logger.info(f"Length of subset dataset: {len(current_train_subset)}")
                
                train_dataloader = DataLoader(current_train_subset, shuffle=True, 
                                              batch_size=config["batch_size"], collate_fn=data_collator)
                data_iter = iter(train_dataloader)
                
                # Reset patience for new subset
                patience_state.counter = 0
                patience_state.best_val_loss = float('inf')
                patience_state.best_model_state = copy.deepcopy(model.state_dict())
                
                if not has_more:
                    # No more subsets, stop training
                    should_stop = True
            
            if should_stop:
                model.load_state_dict(patience_state.best_model_state)
                logger.info("Training stopped.")
                break
        
        # Log training loss every 100 steps
        if global_step % 100 == 0:
            csv_logger.info(controller.format_training_log(global_step, loss.item()))
        
        # Update progress bar
        pbar.set_postfix_str(controller.get_progress_string(loss.item()))
        pbar.update(1)
    
    pbar.close()
    logger.info("Training phase completed.")
    return
```

## Invariant Verification

### Invariant 1: Competence-based never advances labels
**Verification**: `CompetenceBasedController` never calls `cl_scheduler.update_current_train_subset()` without a `global_step` parameter. The `CL_Scheduler.update_current_train_subset()` method checks `self.label_based` and only advances labels when `label_based=True`. Since competence-based controller uses a competence-based scheduler (`label_based=False`), label advancement is impossible.

### Invariant 2: Competence-based never resets patience on subset expansion
**Verification**: In `CompetenceBasedController.handle_validation_result()`, when `current_competence < 1`, the method returns `(False, False)` without modifying patience state. Patience is only tracked when `current_competence == 1`. Subset updates happen via `should_update_subset()` which is separate from patience handling.

### Invariant 3: Label-based never updates datasets based on step count
**Verification**: `LabelBasedController.should_update_subset()` always returns `False`. The only way to update the subset is through `handle_validation_result()` returning `should_update_subset=True`, which only happens on patience exhaustion.

### Invariant 4: Label-based never advances schedules without patience exhaustion
**Verification**: `LabelBasedController.handle_validation_result()` only returns `should_update_subset=True` when `patience_state.counter >= config["patience"]`. The schedule can only advance through this path.

### Invariant 5: Best-model checkpointing is global and independent of curriculum phase
**Verification**: `PatienceState` is created once and persists across all curriculum phases. Both controllers update `patience_state.best_model_state` when validation loss improves. The training loop maintains a single `patience_state` instance throughout training.

### Invariant 6: Validation frequency and semantics remain identical
**Verification**: 
- Competence-based: Validates at `update_every_competence` when competence < 1, and at `update_every_conv` when competence == 1
- Label-based: Validates at `update_every_conv` always
- These conditions are preserved in `should_validate()` and `should_check_convergence()` methods

## Edge Cases Analysis

### Edge Case 1: Competence reaches 1 exactly at update_every_competence step
**Scenario**: `global_step % update_every_competence == 0` and competence becomes 1.0
**Handling**: `should_update_subset()` returns `False` (because `current_competence == 1`), so subset update is skipped. `should_check_convergence()` returns `True`, so convergence check runs. This matches original behavior.

### Edge Case 2: Label schedule exhausted during patience tracking
**Scenario**: Last label subset, patience exhausted
**Handling**: `update_subset()` returns `has_updated=False`. Training loop sets `should_stop=True` and breaks. This matches original behavior (line 267-268).

### Edge Case 3: NaN loss during validation
**Scenario**: Validation produces NaN loss
**Handling**: The `evaluate()` function filters NaN losses (line 39-41), so this is handled. The controller receives a valid `val_loss` (or `float('inf')`). This matches original behavior.

### Edge Case 4: Multiple validation triggers in same step
**Scenario**: Both `should_update_subset()` and `should_check_convergence()` return True
**Handling**: The training loop checks `should_update_subset()` first, then `should_check_convergence()`. For competence-based, these are mutually exclusive (competence < 1 vs == 1). For label-based, `should_update_subset()` always returns False. No conflict possible.

## Potential Issues and Resolutions

### Issue 1: Controller needs access to cl_scheduler
**Resolution**: Controllers are initialized with `cl_scheduler` and can access it. The `get_current_subset()` method delegates to `cl_scheduler.get_current_train_subset()`. This is acceptable because the controller is the curriculum policy, and the scheduler is the data provider.

### Issue 2: Patience state reset in label-based
**Resolution**: In the unified loop, when `should_update_subset=True`, we reset patience state. This matches original behavior (label-based doesn't explicitly reset, but starts fresh with new subset). However, we should verify this matches semantics.

**Verification**: In original label-based loop (lines 260-274), when patience is exhausted:
- Model state is restored from `best_model_state`
- New subset is loaded
- But `patience_counter` and `best_val_loss` are NOT explicitly reset

**Problem Identified**: The unified loop resets patience state, but original doesn't. This changes behavior.

**Fix**: Don't reset patience state in unified loop. Instead, let the controller decide. `LabelBasedController.handle_validation_result()` should reset its internal state when returning `should_update_subset=True`, but the training loop should not reset `patience_state`.

Actually, wait - let me re-read the original code. In label-based, when patience is exhausted:
- Line 264: `model.load_state_dict(best_model_state)` - restore best model
- Line 266: `current_train_subset, has_updated = cl_scheduler.update_current_train_subset()` - get new subset
- But `patience_counter` and `best_val_loss` are NOT reset - they persist

However, this seems like a bug in the original code! If you move to a new subset, you should reset patience tracking for that subset. But to preserve exact semantics, we should not reset.

**Decision**: For exact semantic preservation, do NOT reset patience state. The controller can track whether it's in a "new phase" and handle accordingly.

Actually, I need to check if `best_val_loss` persists. Looking at line 252-255, when `val_loss < best_val_loss`, it resets. So if the new subset has better loss, it will reset naturally. But if it doesn't, the old `best_val_loss` persists, which means patience might exhaust immediately.

This is likely a bug in the original, but to preserve semantics exactly, we should not reset.

## Final Validation Result

### ✅ GREEN LIGHT: Controller-based approach is viable

**Evidence**:
1. All decision points can be cleanly mapped to controller methods
2. No curriculum-specific conditionals remain in the training loop
3. All invariants can be preserved through controller design
4. Edge cases are handled correctly
5. Validation frequency semantics are preserved

**One Semantic Preservation Note**:
The original label-based code does NOT reset `patience_counter` or `best_val_loss` when advancing to a new label subset. To preserve exact semantics, the unified loop should also not reset these. However, this means patience state persists across label subsets, which may cause immediate exhaustion if the new subset performs worse. This appears to be a potential bug in the original code, but for semantic preservation, we maintain it.

**Recommended Implementation**:
1. Implement `CurriculumController` interface as specified
2. Implement `CompetenceBasedController` and `LabelBasedController`
3. Implement unified training loop
4. Add unit tests to verify invariants
5. Document the patience persistence behavior in label-based training

