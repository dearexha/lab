import logging
import logging.config
import yaml
from datasets import load_from_disk
from transformers import BertTokenizerFast, BertForMaskedLM, BertConfig
import torch
import json
import datasets
import os
from datetime import datetime

import utils
from composite_difficulty_metric import LogisticRegression
from training.DataPreprocessing import load_OSE_raw, load_SW_raw, prepare_dataset, convert_dataset
from training.CL_DifficultyMeasurer import CL_DifficultyMeasurer
from training.CL_Scheduler import CL_Scheduler
from training.CompetenceFunction import CompetenceFunction, sqrt_competence_func
from training.training_loop import train_model


def setup_logging():

    logging_config_path = os.path.join(utils.get_project_dir(), "logging_config.yaml")
    training_config_path = os.path.join(utils.get_project_dir(), "config.yaml")

    with open(logging_config_path, "r") as f:
        logging_config = yaml.safe_load(f.read())

    with open(training_config_path, "r") as f:
        experiment_config = yaml.safe_load(f.read())
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(utils.get_project_dir(), "results", "runs", timestamp)
    os.makedirs(run_dir, exist_ok=True)

    experiment_config["meta"] = {"start_time": timestamp}

    with open(os.path.join(run_dir, "config.yaml"), 'w') as f:
        yaml.dump(experiment_config, f)

    logging_config["handlers"]["training_log_file"]["filename"] = os.path.join(run_dir, "training.csv")

    logging.config.dictConfig(logging_config)
    return experiment_config, run_dir

SW_NAME = "SimpleWikipedia"
OSE_NAME = "OneStopEnglish"
SW_SAVE_PATH = os.path.join(utils.get_project_dir(), "results/hf_datasets/SimpleWikipedia")
OSE_SAVE_PATH = os.path.join(utils.get_project_dir(), "results/hf_datasets/OneStopEnglish")
TRAIN_VAL_TEST_SPLIT_SEED = 42 # seed use for creating the split
TRAIN_VAL_TEST_SPLIT = (0.8, 0.1, 0.1)
MAX_LENGTH = 512  # max context size

if __name__=="__main__":
    config, run_dir = setup_logging()
    logger = logging.getLogger(__name__)
    
    utils.set_seed(config["seed"])

    # tokenizer
    model_id = "bert-base-uncased"
    tokenizer = BertTokenizerFast.from_pretrained(model_id) # 100% coverage on SimpleWikipedia

    #================================== HF Dataset Creation (comment out if loading exisiting via load_from_disk) ================================================
    # logging.info("HF Dataset Creation ...")
    # # model for perplexity metric
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # perplexity_model = BertForMaskedLM.from_pretrained(model_id).to(device)

    # # logistic model as combined difficulty metric
    # EXPERIMENT_DIR = os.path.join(utils.get_project_dir(), "results/difficulty_classifier_normalized_input/")
    # log_json_path = os.path.join(EXPERIMENT_DIR, "log.json")
    # if not os.path.exists(log_json_path):
    #     raise FileNotFoundError(
    #         f"Required file not found: {log_json_path}\n"
    #         f"This file is created by running 'composite_difficulty_metric.py' first.\n"
    #         f"Either:\n"
    #         f"  1. Run 'python composite_difficulty_metric.py' to create the difficulty classifier, OR\n"
    #         f"  2. Comment out this dataset creation section (lines 64-98) if you're using pre-computed datasets."
    #     )
    # with open(log_json_path, 'r') as f:
    #     data = json.load(f)
    # selected_metrics = data["config"]["selected_metrics"]
    # model_path = f"{EXPERIMENT_DIR}/model_seed_4.pt"
    # if not os.path.exists(model_path):
    #     raise FileNotFoundError(
    #         f"Required model file not found: {model_path}\n"
    #         f"This file is created by running 'composite_difficulty_metric.py' first."
    #     )
    # classifier_model = LogisticRegression(len(selected_metrics))
    # state_dict = torch.load(model_path)
    # classifier_model.load_state_dict(state_dict)
    # classifier_config = data["config"]

    # sw_raw = load_SW_raw()
    # ose_raw = load_OSE_raw()

    # sw_dataset_dict = prepare_dataset(sw_raw, train_val_test_split=TRAIN_VAL_TEST_SPLIT, seed=TRAIN_VAL_TEST_SPLIT_SEED, tokenizer=tokenizer, max_length=MAX_LENGTH)
    # ose_dataset_dict = prepare_dataset(ose_raw, train_val_test_split=TRAIN_VAL_TEST_SPLIT, seed=TRAIN_VAL_TEST_SPLIT_SEED, tokenizer=tokenizer, max_length=MAX_LENGTH)

    # for key in sw_dataset_dict.keys():
    #     dm = CL_DifficultyMeasurer(sw_dataset_dict[key], tokenizer, perplexity_model, classifier_model, classifier_config, device)
    #     sw_dataset_dict[key] = dm.compute_all_metrics()
    
    # for key in ose_dataset_dict.keys():
    #     dm = CL_DifficultyMeasurer(ose_dataset_dict[key], tokenizer, perplexity_model, classifier_model, classifier_config, device)
    #     ose_dataset_dict[key] = dm.compute_all_metrics()

    # # save these dataset_dict -> can be used for label_CL, competence_CL and eval
    # sw_dataset_dict.save_to_disk(SW_SAVE_PATH)
    # ose_dataset_dict.save_to_disk(OSE_SAVE_PATH)
    # logging.info("HF Dataset Creation done")
    #=============================================================================================================================================================
    # load if metrics computation above already done
    sw_dataset_dict = load_from_disk(SW_SAVE_PATH)
    # ose_dataset_dict = load_from_disk(OSE_SAVE_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_config = BertConfig(**config["bert_config"])
    model = BertForMaskedLM(bert_config).to(device)
    logger.info(f"Initialized BERT model for MLM on device {device}")

    cl_scheduler = CL_Scheduler(sw_dataset_dict, label_based=True, difficulty_metric_name="label", label_schedule=[[0, 1]], tokenizer=tokenizer, competence_func=CompetenceFunction(sqrt_competence_func, config["T"], config["c0"]))

    logger.info("Starting training...")
    train_model(model=model, device=device, tokenizer=tokenizer, cl_scheduler=cl_scheduler, config=config)

    logger.info("Saving the final model...")
    model.save_pretrained(run_dir)
    logger.info(f"Model saved to {run_dir}.")

