import torch
import math
import os
import copy
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
from tqdm import tqdm
import logging

from training.CL_Scheduler import CL_Scheduler

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

    if not config["label_based"]: # competence based CL training

        csv_logger.info("step,train_loss,val_loss,val_perplexity,val_accuracy")  # create header for log file

        pbar = tqdm(ncols=150, total=config["max_steps"], desc="Training Progress")

        current_train_subset = cl_scheduler.get_current_train_subset()
        current_competence = cl_scheduler.competence_func.c0
        logger.info(f"Length of subset dataset: {len(current_train_subset)}")

        train_dataloader = DataLoader(current_train_subset, shuffle=True, batch_size=config["batch_size"], collate_fn=data_collator)
        data_iter = iter(train_dataloader)

        global_step = 0

        # for checking convergence after the entire dataset has been added
        patience_counter = 0
        best_val_loss = float('inf')
        best_model_state = copy.deepcopy(model.state_dict())
        
        while global_step < config["max_steps"]:
            try:
                batch = next(data_iter)
            except StopIteration:
                # Re-initialize data_iter
                data_iter = iter(train_dataloader)
                batch = next(data_iter)

            # Move batch to device
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids=inputs,attention_mask=attention_mask,labels=labels)
            loss = outputs.loss

            # Check for NaN loss if no tokens got masked
            if torch.isnan(loss):
                continue  # Skip this batch

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

            if global_step % config["update_every_competence"] == 0 and current_competence < 1: # update competence
                # evaluate on validation set
                val_loss, val_perplexity, val_accuracy = evaluate(model, device, val_dataloader)
                model.train()
                logger.info(f"Validation Loss: {val_loss}, Validation Perplexity: {val_perplexity}, Validation Accuracy: {val_accuracy}")

                # log to file
                current_train_loss = loss.item()
                csv_logger.info(f"{global_step},{current_train_loss},{val_loss},{val_perplexity},{val_accuracy}")

                # update current trainset based on competence
                current_train_subset, has_updated = cl_scheduler.update_current_train_subset(global_step)
                current_competence = cl_scheduler.competence_func.compute_competence(global_step)
                logger.info(f"Length of subset dataset: {len(current_train_subset)}")

                train_dataloader = DataLoader(current_train_subset, shuffle=True, batch_size=config["batch_size"], collate_fn=data_collator)
                data_iter = iter(train_dataloader)
            
            # Monitor train loss every 100 steps
            if global_step % 100 == 0:
                current_train_loss = loss.item()
                # Log only the training loss to the CSV
                csv_logger.info(f"{global_step},{current_train_loss},,,")
            

            # Dont update the competence because it is already 1 and monitor validation loss every 25000 steps
            if global_step % config["update_every_conv"] == 0 and current_competence == 1:
                # evaluate on validation set
                val_loss, val_perplexity, val_accuracy = evaluate(model, device, val_dataloader)
                model.train()
                logger.info(f"Validation Loss: {val_loss}, Validation Perplexity: {val_perplexity}, Validation Accuracy: {val_accuracy}")

                # log to file
                current_train_loss = loss.item()
                csv_logger.info(f"{global_step},{current_train_loss},{val_loss},{val_perplexity},{val_accuracy}")


                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = copy.deepcopy(model.state_dict())
                else:
                    patience_counter += 1
                    logger.info(f"No improvement in validation loss. Patience counter: {patience_counter}/{config["patience"]}")

                    if patience_counter >= config["patience"]:
                        logger.info(
                            f"Validation loss did not improve for {config["patience"]} consecutive checks. Reverting and stopping training.")
                        model.load_state_dict(best_model_state)
                        pbar.close()
                        break



            # Update tqdm progress bar
            pbar.set_postfix_str(f"Loss: {loss:.2f} | Comp: {current_competence:.4f}")
            pbar.update(1)

        # Close the progress bar
        pbar.close()
        logger.info("Training phase completed.")
    
    else: # label based CL training

        csv_logger.info("schedule_step,step,train_loss,val_loss,val_perplexity,val_accuracy")  # create header for log file

        pbar = tqdm(ncols=150, total=config["max_steps"], desc="Training Progress")

        current_train_subset = cl_scheduler.get_current_train_subset()
        current_schedule_step = cl_scheduler.current_schedule_step
        current_label_subset = cl_scheduler.label_schedule[current_schedule_step]
        logger.info(f"Length of subset dataset (labels={current_label_subset}): {len(current_train_subset)}")

        train_dataloader = DataLoader(current_train_subset, shuffle=True, batch_size=config["batch_size"], collate_fn=data_collator)
        data_iter = iter(train_dataloader)

        global_step = 0

        # for checking convergence and updating the current subset of labels according to the schedule
        patience_counter = 0
        best_val_loss = float('inf')
        best_model_state = copy.deepcopy(model.state_dict())
        
        while global_step < config["max_steps"]:
            try:
                batch = next(data_iter)
            except StopIteration:
                # Re-initialize data_iter
                data_iter = iter(train_dataloader)
                batch = next(data_iter)

            # Move batch to device
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids=inputs,attention_mask=attention_mask,labels=labels)
            loss = outputs.loss

            # Check for NaN loss if no tokens got masked
            if torch.isnan(loss):
                continue  # Skip this batch

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

            # Check for convergence
            if global_step % config["update_every_conv"] == 0:
                # evaluate on validation set
                val_loss, val_perplexity, val_accuracy = evaluate(model, device, val_dataloader)
                model.train()
                logger.info(f"Validation Loss: {val_loss}, Validation Perplexity: {val_perplexity}, Validation Accuracy: {val_accuracy}")

                # log to file
                current_train_loss = loss.item()
                csv_logger.info(f"{current_schedule_step},{global_step},{current_train_loss},{val_loss},{val_perplexity},{val_accuracy}")


                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = copy.deepcopy(model.state_dict())
                else:
                    patience_counter += 1
                    logger.info(f"No improvement in validation loss. Patience counter: {patience_counter}/{config["patience"]}")

                    if patience_counter >= config["patience"]:
                        logger.info(
                            f"Validation loss did not improve for {config["patience"]} consecutive checks. Reverting and stopping training for current label subset.")
                        logger.info(f"Completed training on current label subset: {current_label_subset}")
                        model.load_state_dict(best_model_state)
                        # update current trainset based on the label schedule
                        current_train_subset, has_updated = cl_scheduler.update_current_train_subset()
                        if not has_updated:  # training is done
                            break
                        current_schedule_step = cl_scheduler.current_schedule_step
                        current_label_subset = cl_scheduler.label_schedule[current_schedule_step]
                        logger.info(f"Length of subset dataset (labels={current_label_subset}): {len(current_train_subset)}")

                        train_dataloader = DataLoader(current_train_subset, shuffle=True, batch_size=config["batch_size"], collate_fn=data_collator)
                        data_iter = iter(train_dataloader)
            
            # Monitor train loss every 100 steps
            if global_step % 100 == 0:
                current_train_loss = loss.item()
                csv_logger.info(f"{current_schedule_step},{global_step},{current_train_loss},,,")



            # Update tqdm progress bar
            pbar.set_postfix_str(f"Loss: {loss:.2f} | label_subset: {current_label_subset}")
            pbar.update(1)

        # Close the progress bar
        pbar.close()
        logger.info("Training phase completed.")
    
    return


