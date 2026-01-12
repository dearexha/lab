import torch
import math
import os
import copy
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
from tqdm import tqdm
import logging

from training.CL_Scheduler import CL_Scheduler
from training.CurriculumController import CurriculumController, PatienceState, create_controller

logger = logging.getLogger(__name__)
csv_logger = logging.getLogger("training_csv_logger")

def evaluate(model, device, val_dataloader):
    model.eval()
    total_loss = 0.0
    total_steps = 0
    
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in val_dataloader:
            # Move batch to device
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=inputs,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits

            if not torch.isnan(loss):
                total_loss += loss.item()
                total_steps += 1

            predictions = torch.argmax(logits, dim=-1)
            
            # Label -100 corresponds to unmasked tokens
            mask = labels != -100
            
            correct_predictions += (predictions[mask] == labels[mask]).sum().item()
            total_predictions += mask.sum().item()

    avg_loss = total_loss / total_steps if total_steps > 0 else float('inf')
    
    try:
        perplexity = math.exp(avg_loss)
    except OverflowError:
        perplexity = float('inf')

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

    return avg_loss, perplexity, accuracy


def train_model(model, device, tokenizer, cl_scheduler: CL_Scheduler, config):
    """
    Unified training loop that is completely curriculum-agnostic.
    All curriculum logic is delegated to the CurriculumController.
    
    (Pre)trains a BERT model with a given CL approach (either competence or label based) using MLM.
    
    :param model: The BERT model to be trained.
    :param device: The device that training occurs on.
    :param tokenizer: The tokenizer used for tokenization of the dataset.
    :param cl_scheduler: Wrapper for the CL strategy (either competence or label based).
    :param config: Python dictionary containing all hyperparameters for the current run.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    val_dataset = cl_scheduler.get_validation_set()
    val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    model.train()

    # Create curriculum controller
    controller = create_controller(cl_scheduler, config)
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
                # Restore best model before updating subset (preserving original behavior)
                model.load_state_dict(patience_state.best_model_state)
                
                current_train_subset, has_more = controller.update_subset()
                logger.info(f"Length of subset dataset: {len(current_train_subset)}")

                train_dataloader = DataLoader(current_train_subset, shuffle=True, 
                                              batch_size=config["batch_size"], collate_fn=data_collator)
                data_iter = iter(train_dataloader)

                # Note: Patience state persists across label subsets (preserving original behavior)
                # Reset best model state for new subset
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


