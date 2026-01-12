from transformers import BertTokenizerFast
from datasets import load_from_disk
from tqdm import tqdm
import os
import utils

model_checkpoint = "bert-base-uncased"
tokenizer = BertTokenizerFast.from_pretrained(model_checkpoint)

dataset_name = "SimpleWikipedia"
# dataset_name = "OneStopEnglish"

print("Loading raw dataset ...")
dataset = load_from_disk(os.path.join(utils.get_project_dir(), f"results/hf_datasets/{dataset_name}_raw"))
print("Loading raw dataset done")

total_tokens = 0
unk_tokens = 0

# get the ID for the [UNK] token
unk_token_id = tokenizer.unk_token_id

print(f"Counting tokens using tokenizer: {model_checkpoint}...")

for text in tqdm(dataset):
    # use add_special_tokens=False to avoid counting [CLS] and [SEP]
    tokens = tokenizer.encode(text["text"][:512], add_special_tokens=False)
    
    # Update counts
    total_tokens += len(tokens)
    unk_tokens += tokens.count(unk_token_id)


if total_tokens > 0:
    unk_percentage = (unk_tokens / total_tokens) * 100
    coverage = 100 - unk_percentage
    
    print("-" * 30)
    print(f"Total Tokens: {total_tokens}")
    print(f"Unknown Tokens: {unk_tokens}")
    print(f"UNK Percentage: {unk_percentage:.2f}%")
    print(f"Vocabulary Coverage: {coverage:.2f}%")
    print("-" * 30)
else:
    print("Dataset is empty.")

# Results from run
# ------------------------------
# Total Tokens: 9817264
# Unknown Tokens: 0
# UNK Percentage: 0.00%
# Vocabulary Coverage: 100.00%
# ------------------------------