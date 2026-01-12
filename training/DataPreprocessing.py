import logging
import os
import pathlib
import utils
import nltk
from datasets import Dataset, concatenate_datasets, Features, Value, ClassLabel, DatasetDict
from collections import Counter
from tqdm import tqdm
import pyphen
import numpy as np
import torch
import math
from collections.abc import Callable

# nltk.download('punkt')
# nltk.download('punkt_tab')

logger = logging.getLogger(__name__)

sw_default_path = os.path.join(
    utils.get_project_dir(), "datasets/SimpleWikipedia_v2")
ose_default_path = os.path.join(
    utils.get_project_dir(), "datasets/OneStopEnglish")
# use datasets.load_from_disk() for HF dataset filepaths


def load_SW_raw(filepath: str = sw_default_path):
    """
    Loads the SimpleWikipedia dataset as a Huggingface dataset.

    :param filepath: Description
    :type filepath: str
    """
    logger.info("Creating SimpleWikipedia HF dataset...")
    file_label_map = {
        0: os.path.join(filepath, "simple.aligned"),
        1: os.path.join(filepath, "normal.aligned")
    }

    def sample_stream_generator(filepath, label):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                parts = line.strip().split("\t", maxsplit=2)

                # SimpleWikipedia_v2: each line contains multiple sentences, and has the format article_title<TAB>paragraph_number<TAB>sentence
                sample_text = parts[2].strip()
                if sample_text:
                    yield {"text": sample_text, "label": label}

    # create a list of huggingface datasets
    labeled_datasets_list = []
    for label, filepath in file_label_map.items():

        # create HF dataset from generator
        ds = Dataset.from_generator(
            sample_stream_generator,
            gen_kwargs={
                "filepath": filepath,
                "label": label
            }
        )
        labeled_datasets_list.append(ds)
    # concatenate datasets
    raw_dataset = concatenate_datasets(labeled_datasets_list)

    label_names = ["SL", "EL"]
    features = Features({
        "text": Value("string"),
        "label": ClassLabel(names=label_names)
    })
    raw_dataset = raw_dataset.cast(features)

    return raw_dataset


def load_OSE_raw(filepath: str = ose_default_path):
    """
    Loads the OneStopEnglish dataset as a Huggingface dataset.

    :param filepath: Description
    :type filepath: str
    """
    logger.info("Creating OneStopEnglish HF dataset...")
    file_label_map = {
        0: os.path.join(filepath, "Ele-Txt"),
        1: os.path.join(filepath, "Int-Txt"),
        2: os.path.join(filepath, "Adv-Txt")
    }

    def sample_stream_generator(filepath, label):
        scan_dir = pathlib.Path(filepath)
        if scan_dir.is_dir():
            for item in scan_dir.iterdir():
                if item.is_file():
                    try:
                        # each paragraph contains multiple sentences
                        text = item.read_text(encoding="utf-8")

                        if not text.strip():
                            continue

                        yield{
                            "text": text,
                            "label": label
                        }
                    except (OSError, UnicodeDecodeError) as e:
                        logger.error(f"Could not read file {item.name}", exc_info=1)
        else:
            logger.error(f"Given directory not found: {scan_dir}")

    # create a list of huggingface datasets
    labeled_datasets_list = []
    for label, filepath in file_label_map.items():

        # create HF dataset from generator
        ds = Dataset.from_generator(
            sample_stream_generator,
            gen_kwargs={
                "filepath": filepath,
                "label": label
            }
        )
        labeled_datasets_list.append(ds)

    # concatenate datasets
    raw_dataset = concatenate_datasets(labeled_datasets_list)

    label_names = ["Ele", "Int", "Adv"]
    features = Features({
        "text": Value("string"),
        "label": ClassLabel(names=label_names)
    })
    raw_dataset = raw_dataset.cast(features)

    return raw_dataset


def prepare_dataset(raw_dataset: Dataset, *, train_val_test_split: tuple[float] = (0.8, 0.1, 0.1), seed: int, tokenizer, max_length: int):
    """
    Loads the raw dataset as and performs sentence splitting, tokenization, train-val-test split (where the training set is split according to the labels).

    :param raw_dataset: The raw dataset containing paragraphs (or parts) as instances.
    :type raw_dataset: Dataset
    :param train_val_test_split: Tuple of 3 floats determining the train-validation-test split of the dataset.
    :type train_val_test_split: tuple[float]
    :param seed: The seed used for splitting.
    :type seed: int
    :param tokenizer: The tokenizer used for training.
    :param max_length: Integer describing the maximum context size/sentence length.
    :type max_length: int
    """

    logger.info("Splitting into Train-Val-Test...")

    train_testval = raw_dataset.train_test_split(test_size=sum(
        train_val_test_split[1:]), stratify_by_column="label", seed=seed)
    test_val = train_testval["test"].train_test_split(
        test_size=train_val_test_split[1]/sum(train_val_test_split[1:]), stratify_by_column="label", seed=seed)

    split_dataset = DatasetDict({
        "train": train_testval["train"],
        "validation": test_val["test"],
        "test": test_val["train"]
    })

    logger.info("Splitting samples into individual sentences...")

    def split_into_sentences(batch):
        new_texts = []
        new_labels = []

        for text, label in zip(batch["text"], batch["label"]):
            sentences = nltk.sent_tokenize(text)
            new_texts.extend(sentences)
            new_labels.extend([label] * len(sentences))

        return {"text": new_texts, "label": new_labels}

    for key in split_dataset.keys():
        split_dataset[key] = split_dataset[key].map(
            split_into_sentences,
            batched=True,
            remove_columns=split_dataset[key].column_names
        )

    logger.info(
        "Creating separate training datasets for each difficulty_label...")
    df = split_dataset["train"].select_columns(
        ["label"]).to_pandas()
    indices_per_group = df.groupby("label").groups

    class_based_train_split_dataset = {
        f"train_class_{val}": split_dataset["train"].select(indices)
        for val, indices in indices_per_group.items()
    }

    class_based_train_split_dataset["validation"] = split_dataset["validation"]
    class_based_train_split_dataset["test"] = split_dataset["test"]

    split_dataset = DatasetDict(class_based_train_split_dataset)

    logger.info("Tokenizing dataset...")

    def tokenize(examples, tokenizer):
        return tokenizer(examples["text"], truncation=True, padding=False, max_length=max_length, return_special_tokens_mask=True)

    for key in split_dataset.keys():
        split_dataset[key] = split_dataset[key].map(
            tokenize,
            batched=True,
            fn_kwargs={"tokenizer": tokenizer}
        )

    return split_dataset


# e.g. filter_func = lambda x: len(x["text"]) > 4
def filter_dataset(dataset: Dataset | DatasetDict, filter_func: Callable):
    """
    Filters the dataset by removing outliers according to the given filter_func.

    :param dataset: The HF dataset to be filtered.
    :type dataset: Dataset | DatasetDict
    :param filter_func: A function that returns True for valid examples and False for the ones to be filtered.
    :type filter_func: Callable
    """
    # apply outlier filtering after sentence split
    logger.info("Filtering dataset...")

    if isinstance(dataset, Dataset):
        filtered_dataset = dataset.filter(filter_func)
    else:
        filtered_dataset = {}
        for key in dataset.keys():
            filtered_dataset[key] = dataset[key].filter(
                filter_func)

    return filtered_dataset


def convert_dataset(dataset_dict: DatasetDict, purpose: str ="eval", difficulty_metric:str =None):
    """
    Converts the dataset as returned by prepare_dataset to a dataset for one of three purposes: label_CL, competence_CL, eval. Returns a copy.
    
    :param dataset_dict: The dataset to be converted.
    :type dataset_dict: DatasetDict
    :param purpose: The conversion purpose. Must be one of: "label_CL", "competence_CL", "eval".
    :type purpose: str
    :param difficulty_metric: The name of the column to use as difficulty metric. Only relevant for competence_CL.
    :type difficulty_metric: str
    """
    logger.info(f"Converting dataset for purpose: {purpose} ...")
    converted_dataset_dict = {}

    if purpose == "label_CL":
        # returns only the columns input_ids, attention_mask and special_tokens_mask for each dataset in the dictionary
        # leaves the training set split as needed for label based CL
        for key in dataset_dict.keys():
            converted_dataset_dict[key] = dataset_dict[key].select_columns(["input_ids", "attention_mask", "special_tokens_mask"])
        return DatasetDict(converted_dataset_dict)
    
    elif purpose == "competence_CL":
        if not difficulty_metric or difficulty_metric not in dataset_dict.values()[0].column_names:
            raise ValueError(f"Difficulty metric ({difficulty_metric}) either none or not in dataset.")
        # returns only the columns input_ids, attention_mask and special_tokens_mask and the difficulty_metric for each dataset in the dictionary
        # combines the training set into one as needed for competence based CL
        train_subsets = []
        for key in dataset_dict.keys():
            if "train" not in key:
                converted_dataset_dict[key] = dataset_dict[key].select_columns(["input_ids", "attention_mask", "special_tokens_mask", difficulty_metric])
            else:
                train_subsets.append(dataset_dict[key].select_columns(["input_ids", "attention_mask", "special_tokens_mask", difficulty_metric]))
        
        train_set = concatenate_datasets(train_subsets)
        converted_dataset_dict["train"] = train_set
        return DatasetDict(converted_dataset_dict)



    elif purpose == "eval":
        # concatenate all datasets and return all columns as is for evaluation (e.g. histogram plotting)
        converted_dataset = concatenate_datasets(dataset_dict.values())
        return converted_dataset

    else:
        raise ValueError("conversion purpose not defined properly.")


