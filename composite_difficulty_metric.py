import os
import pathlib
import json
from datetime import datetime
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from datasets import load_from_disk
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import utils
import logging

logger = logging.getLogger(__name__)

def compute_train_mean_std(loader, selected_metrics, device):
    running_sum = None
    running_squared_sum = None
    total_count = 0
    
    for batch in loader:
        features = input_from_batch(batch, selected_metrics, device)
        
        features = features.double() 
        if running_sum is None:
            input_dim = features.shape[1]
            running_sum = torch.zeros(input_dim, dtype=torch.float64).to(device)
            running_squared_sum = torch.zeros(input_dim, dtype=torch.float64).to(device)
        
        running_sum += torch.sum(features, dim=0)
        
        running_squared_sum += torch.sum(features ** 2, dim=0)

        total_count += features.size(0)
    
    mean = running_sum / total_count
    mean_of_squares = running_squared_sum / total_count
    variance = mean_of_squares - (mean ** 2)
    std = torch.sqrt(torch.clamp(variance, min=1e-10))

    return mean.float(), std.float()

def normalize(features, mean, std):
    return (features - mean)/std

class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return self.linear(x)
    

def input_from_batch(batch, selected_metrics, device): # helper method that converts the current batch (dict of columns) into a batch of input vectors
    return torch.stack([batch[m] for m in selected_metrics], dim=1).to(torch.float32).to(device)


def eval_model(model, data_loader, selected_metrics, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            raw_input_metrics = input_from_batch(batch, selected_metrics, device)
            input_metrics = normalize(raw_input_metrics, train_mean, train_std)
            labels = batch["label"].cpu().numpy()

            logits = model(input_metrics)
            preds = (torch.sigmoid(logits) > 0.5).to(torch.int64).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="binary")
    return acc, f1

def train_one_run(seed, train_loader, val_loader, test_loader, selected_metrics):
    utils.set_seed(seed)

    model = LogisticRegression(len(selected_metrics)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    # we stop training after the validation loss does not decrease for 5 epochs (max 50 epochs)
    best_val_loss = np.inf
    patience_counter = 0
    best_weights = None
    best_model_num_epochs = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        for batch in tqdm(train_loader):
            raw_input_metrics = input_from_batch(batch, selected_metrics, device)
            input_metrics = normalize(raw_input_metrics, train_mean, train_std)
            labels = batch["label"].to(torch.float32).unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(input_metrics)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # validation loss computation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                raw_input_metrics = input_from_batch(batch, selected_metrics, device)
                input_metrics = normalize(raw_input_metrics, train_mean, train_std)
                labels = batch["label"].to(torch.float32).unsqueeze(1).to(device)
                val_loss = criterion(model(input_metrics), labels).item()
        
        avg_val_loss = val_loss/len(val_loader)

        logger.info(f"Epoch [{epoch+1}/{MAX_EPOCHS}] Train Loss: {loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # check for convergence
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_weights = model.state_dict()
            best_model_num_epochs = (epoch + 1)

        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    model.load_state_dict(best_weights)
    test_acc, test_f1 = eval_model(model, test_loader, selected_metrics, device)

    learned_weights = model.linear.weight.detach().cpu().numpy().flatten()

    return test_acc, test_f1, learned_weights, best_weights, best_model_num_epochs
            
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)





if __name__=="__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logging.info("Classifier training ...")
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    normalize_inputs = True
    EXPERIMENT_DIR = pathlib.Path(os.path.join(utils.get_project_dir(), f"results/difficulty_classifier{"_normalized_input" if normalize_inputs else ""}/"))  # /run_{timestamp} 
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    logging.info(f"Saving all outputs to: {EXPERIMENT_DIR}")

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    logging.info(f"Using {device} device")

    dataset_name = "SimpleWikipedia_tokenized_and_measured"
    dataset_path = pathlib.Path(os.path.join(utils.get_project_dir(), f"results/hf_datasets/{dataset_name}"))
    dataset = load_from_disk(dataset_path)

    # select relevant columns for training the classifier
    selected_metrics = ["sentence_length_words", "word_rarity_words", "fre_score_words", "shannon_entropy_words", "ttr_words", "perplexity"]
    dataset = dataset.select_columns(selected_metrics + ["label"])


    # train, test, validation split
    split_1 = dataset.train_test_split(test_size=0.15, seed=42)
    train_val_set = split_1["train"]
    test_set = split_1["test"]

    split_2 = train_val_set.train_test_split(test_size=0.15/0.85, seed=42) # 15% size of validation and test set
    train_set = split_2["train"]
    val_set = split_2["test"]

    train_set.set_format(type="torch")
    val_set.set_format(type="torch")
    test_set.set_format(type="torch")

    # hyperparams
    BATCH_SIZE = 64
    LR = 0.01
    MAX_EPOCHS = 50
    PATIENCE = 5

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

    if normalize_inputs:
        train_mean, train_std = compute_train_mean_std(train_loader, selected_metrics, device)
    else:
        train_mean = torch.zeros(len(selected_metrics), dtype=torch.float32).to(device)
        train_std = torch.ones(len(selected_metrics), dtype=torch.float32).to(device)

    SEEDS = list(range(10))
    experiment_log = {
        "config": {
            "selected_metrics": selected_metrics,
            "learning_rate": LR,
            "batch_size": BATCH_SIZE,
            "patience": PATIENCE,
            "seeds": SEEDS,
            "train_mean": train_mean.detach().cpu().numpy().flatten(),
            "train_std": train_std.detach().cpu().numpy().flatten()
        },
        "runs": []
    }

    for seed in SEEDS:
        logging.info(f"Running Seed {seed}...")
        acc, f1, learned_weights, best_weights_state_dict, num_epochs = train_one_run(seed, train_loader, val_loader, test_loader, selected_metrics)
        
        # save model for current seed
        model_filename = f"model_seed_{seed}.pt"
        save_path = os.path.join(EXPERIMENT_DIR, model_filename)
        torch.save(best_weights_state_dict, save_path) 

        # log data for current run
        run_data = {
            "seed": seed,
            "accuracy": acc,
            "f1_score": f1,
            "weights": learned_weights,
            "model_file": model_filename,
            "num_epochs": num_epochs
        }
        experiment_log["runs"].append(run_data)


    acc_list = [run["accuracy"] for run in experiment_log["runs"]]
    f1_list = [run["f1_score"] for run in experiment_log["runs"]]

    experiment_log["aggregates"] = {
        "mean_accuracy": np.mean(acc_list),
        "std_accuracy": np.std(acc_list),
        "mean_f1": np.mean(f1_list),
        "std_f1": np.std(f1_list)
    }

    json_path = os.path.join(EXPERIMENT_DIR, "log.json")
    with open(json_path, 'w') as f:
        json.dump(experiment_log, f, indent=4, cls=NumpyEncoder)

    logging.info("Classifier training done")


    # log file and model loading

    # EXPERIMENT_DIR = f"./results/difficulty_classifier/"
    # # Load the data
    # with open(f"{EXPERIMENT_DIR}/log.json", 'r') as f:
    #     data = json.load(f)
    # selected_metrics = data["config"]["selected_metrics"]
    # model_path = f"{EXPERIMENT_DIR}/model_seed_{target_seed}.pt"
    # model = LogisticRegression(len(selected_metrics)).to(device)
    # state_dict = torch.load(model_path)
    # model.load_state_dict(state_dict)
    # model.eval()
    



