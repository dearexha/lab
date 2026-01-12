import logging
import os
import utils
import nltk
from datasets import Dataset, concatenate_datasets, Features, Value, ClassLabel, DatasetDict
from collections import Counter
from tqdm import tqdm
import pyphen
import numpy as np
import torch
import math

# nltk.download('punkt')
# nltk.download('punkt_tab')

logger = logging.getLogger(__name__)
import torch
import nltk
from datasets import Dataset

class DifficultyMetricRegistry:
    def __init__(self):
        self._metrics = {"cpu": {}, "gpu": {}, "dependent": {}}
        self._metric_to_group = {}

    def register(self, group="cpu", name=None):
        def decorator(func):
            key = name if name else func.__name__
            self._metrics[group][key] = func
            self._metric_to_group[key] = group
            return func
        return decorator

    def get_group_metrics(self, group):
        return self._metrics.get(group, {})

    def get_metric_group(self, name):
        return self._metric_to_group.get(name)


class CL_DifficultyMeasurer:
    reg = DifficultyMetricRegistry()

    def __init__(self, dataset, tokenizer, perplexity_model, logistic_classifier, classifier_config, device):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device
        self.classifier_config = classifier_config
        self.unigram_word_counts, self.unigram_token_counts = (None, None)  # initialized when required (i.e. when computing word rarity)
        self.dic = pyphen.Pyphen(lang="en_US")
        
        self.perplexity_model = perplexity_model.to(self.device).eval()
        self.logistic_classifier = logistic_classifier.to(self.device).eval()
        self.unigram_word_counts, self.unigram_token_counts = self._compute_unigram_counts()

        # reflects if instance is the main process or a worker
        self._is_worker = False

    def __getstate__(self):
        state = self.__dict__.copy()
        if "perplexity_model" in state:
            # avoids that the bert model is copied to every worker
            del state["perplexity_model"]
        
        if "logistic_classifier" in state:
            # avoids that the logistic model is copied to every worker
            del state["logistic_classifier"]

        state["_is_worker"] = True
        return state
    

    def _compute_unigram_counts(self):
        logger.info("Computing Unigram counts for dataset...")
        word_counts = Counter()
        token_counts = Counter()

        for example in tqdm(self.dataset):
            sent_word_list = nltk.word_tokenize(example["text"])
            sent_word_counts = Counter(sent_word_list)
            word_counts.update(sent_word_counts)

            sent_token_list = self.tokenizer.convert_ids_to_tokens(
                example["input_ids"])
            sent_token_counts = Counter(sent_token_list)
            token_counts.update(sent_token_counts)
        
        return word_counts, token_counts

    # ------------------------------------------------------------------
    # CPU METRICS 
    # ------------------------------------------------------------------
    @reg.register(group="cpu", name="sentence_length_words")
    def calc_length(self, batch, sent_word_lists):
        return [len(t) for t in sent_word_lists]
    
    @reg.register(group="cpu", name="word_rarity_words")
    def calc_word_rarity(self, batch, sent_word_lists):
        word_rarities = []
        for sent_word_list in sent_word_lists:
            if len(sent_word_list) == 0: # in case the sentence contains no words
                word_rarities.append(float("nan"))
                continue
            counts = np.array([self.unigram_word_counts[word] for word in sent_word_list])
            N_total = np.sum(list(self.unigram_word_counts.values()))
            word_rarity = - 1/len(sent_word_list) * np.sum(np.log(counts/N_total))
            word_rarities.append(word_rarity)
        return word_rarities
    
    @reg.register(group="cpu", name="fre_score_words")
    def calc_fre_score(self, batch, sent_word_lists):
        if not self.dic:
            self.dic = pyphen.Pyphen(lang="en_US")
        fre_scores = []
        for sent_word_list in sent_word_lists:
            if len(sent_word_list) == 0: # in case the sentence contains no words
                fre_scores.append(float("nan"))
                continue
            syllable_count = []
            for word in sent_word_list:
                syllable_list = self.dic.inserted(word).split("-")
                syllable_count.append(len(syllable_list))
            avg_syllables_per_word = np.mean(syllable_count)
            sentence_length = len(sent_word_list)
            fre_score = 206.835 - 1.015 * sentence_length - 84.6 * avg_syllables_per_word
            fre_scores.append(fre_score)
        return fre_scores

    @reg.register(group="cpu", name="shannon_entropy_words")
    def calc_shannon_entropy(self, batch, sent_word_lists):
        shannon_entropies = []
        for sent_word_list in sent_word_lists:
            if len(sent_word_list) == 0:
                shannon_entropies.append(float("nan"))
                continue
            sentence_hist = Counter(sent_word_list)
            p = np.array(list(sentence_hist.values()))/len(sent_word_list)
            shannon_entropy = -np.sum(p * np.log2(p))
            shannon_entropies.append(shannon_entropy)
        return shannon_entropies

    @reg.register(group="cpu", name="ttr_words")
    def calc_ttr(self, batch, sent_word_lists): # #unique_words/#total_words
        ttrs = []
        for sent_word_list in sent_word_lists:
            if len(sent_word_list) == 0:
                ttrs.append(float("nan"))
                continue
            ttr = len(np.unique(sent_word_list))/len(sent_word_list)
            ttrs.append(ttr)
        return ttrs



    def _run_cpu_batch(self, batch, metrics_to_run=None):
        """
        metrics_to_run: list of strings (metric names). If None, runs all.
        """
        # computes word tokens for each sentence for all metrics
        word_tokens = [nltk.word_tokenize(text) for text in batch["text"]]
        
        results = {}
        available_metrics = self.reg.get_group_metrics("cpu")
        
        for name, func in available_metrics.items():
            # filter only chosen metrics
            if metrics_to_run is not None and name not in metrics_to_run:
                continue
                
            results[name] = func(self, batch, word_tokens)
            
        return results

    # ------------------------------------------------------------------
    # GPU METRICS 
    # ------------------------------------------------------------------
    @reg.register(group="gpu", name="perplexity")
    def calc_perplexity(self, batch):
        if not self.perplexity_model:
            raise ValueError("Language model to compute the perplexity is not defined.")
        
        ppls = []
        inference_batch_size = 32

        for input_ids in batch["input_ids"]:
            input_ids = torch.as_tensor(input_ids, device=self.device)

            if self.tokenizer.pad_token_id is not None:
                 input_ids = input_ids[input_ids != self.tokenizer.pad_token_id]
            
            # sequence length (includes [CLS] and [SEP])
            seq_len = input_ids.size(0)
            
            # return None as a value for sentence that only contain CLS and SEP
            if seq_len < 3: 
                ppls.append(float("nan"))

            # to compute perplexity we create a batch of the same sentence, where in each instance a different token is masked (excluding CLS and SEP)
            n_tokens_to_mask = seq_len - 2
            repeat_input_ids = input_ids.repeat(n_tokens_to_mask, 1)
            mask_indices = torch.arange(1, seq_len - 1).to(self.device)
            repeat_input_ids[torch.arange(n_tokens_to_mask), mask_indices] = self.tokenizer.mask_token_id
            
            # create the labels for computation of the NLL/cross entropy (Pytorch convention: ignore_index=-100)
            labels = torch.full(repeat_input_ids.shape, -100).to(self.device)
            labels[torch.arange(n_tokens_to_mask), mask_indices] = input_ids[mask_indices]

            # instead of processing the entire n_tokens many sentences as a single batch, process in smaller batches to avoid memory problems on gpu
            total_nll = 0.0
            with torch.no_grad():
                for i in range(0, n_tokens_to_mask, inference_batch_size):
                    # note: slices beyond the maximum index are ignored
                    batch_input = repeat_input_ids[i : i + inference_batch_size]
                    batch_labels = labels[i : i + inference_batch_size]
                    
                    # forward pass through the bert model
                    outputs = self.perplexity_model(batch_input, labels=batch_labels)
                    
                    # outputs.loss is avg nll, we multiply with the batchsize to obtain the sum
                    total_nll += outputs.loss.item() * batch_input.size(0)

            # PPL = exp(avg_NLL)
            avg_nll = total_nll / n_tokens_to_mask  # normalize sum over NLLs
            ppl = math.exp(avg_nll)
            ppls.append(ppl)
        return ppls

    def _run_gpu_batch(self, batch, metrics_to_run=None):
        results = {}
        available_metrics = self.reg.get_group_metrics("gpu")

        for name, func in available_metrics.items():
            if metrics_to_run is not None and name not in metrics_to_run:
                continue
            results[name] = func(self, batch)
        return results

    # ------------------------------------------------------------------
    # DEPENDENT METRICS 
    # ------------------------------------------------------------------
    @reg.register(group="dependent", name="classifier_score")
    def calc_logistic(self, batch):
        if not self.logistic_classifier:
            raise ValueError("self.logistic_classifier is not defined.")
        if not self.classifier_config:
            raise ValueError("self.classifier_config is not defined.")
        
        required_cols = self.classifier_config["selected_metrics"] # required columns
        missing = [c for c in required_cols if c not in self.dataset.column_names]
        if missing:
            raise ValueError(f"Cannot compute classifier_score. Missing columns: {missing}")
        input_metrics = [batch[m] for m in required_cols] # creates 
        input_tensor_raw = torch.tensor(input_metrics).to(torch.float32).transpose(0, 1).to(self.device)
        train_mean = torch.tensor(self.classifier_config["train_mean"]).to(torch.float32).to(self.device)
        train_std = torch.tensor(self.classifier_config["train_std"]).to(torch.float32).to(self.device)
        input_tensor = (input_tensor_raw - train_mean) / train_std
        with torch.no_grad():
            logit = self.logistic_classifier(input_tensor)
        return logit.flatten().cpu().tolist()

    def _run_dependent_batch(self, batch, metrics_to_run=None):
        results = {}
        available_metrics = self.reg.get_group_metrics("dependent")

        for name, func in available_metrics.items():
            if metrics_to_run is not None and name not in metrics_to_run:
                continue
            results[name] = func(self, batch)
        return results

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def compute_single_metric(self, metric_name, overwrite=False):
        # check if exists
        if metric_name in self.dataset.column_names and not overwrite:
            logger.info(f"Metric '{metric_name}' already exists. Skipping.")
            return self.dataset

        # identify Group
        group = self.reg.get_metric_group(metric_name)
        if not group:
            raise ValueError(f"Metric '{metric_name}' not registered.")

        logger.info(f"Computing single metric: {metric_name} (Group: {group})")

        # map based on group
        if group == "cpu":
            self.dataset = self.dataset.map(
                self._run_cpu_batch,
                batched=True,
                batch_size=1000,
                num_proc=8,
                fn_kwargs={"metrics_to_run": [metric_name]} 
            )
        elif group == "gpu":
            self.dataset = self.dataset.map(
                self._run_gpu_batch,
                batched=True,
                batch_size=32,
                fn_kwargs={"metrics_to_run": [metric_name]}
            )
        elif group == "dependent":
            required_cols = self.classifier_config["selected_metrics"] # required columns
            missing = [c for c in required_cols if c not in self.dataset.column_names]
            if missing:
                raise ValueError(f"Cannot compute {metric_name}. Missing columns: {missing}")

            self.dataset = self.dataset.map(
                self._run_dependent_batch,
                batched=True,
                batch_size=1000,
                fn_kwargs={"metrics_to_run": [metric_name]}
            )
        
        return self.dataset

    def compute_all_metrics(self, overwrite=False):
 
        # Helper to filter metrics
        def get_missing(group_name):
            all_in_group = self.reg.get_group_metrics(group_name).keys()
            if overwrite:
                return list(all_in_group)
            return [m for m in all_in_group if m not in self.dataset.column_names]

        # CPU metrics
        cpu_metrics = get_missing("cpu")
        if cpu_metrics:
            logger.info(f"Computing CPU metrics: {cpu_metrics}")
            self.dataset = self.dataset.map(
                self._run_cpu_batch,
                batched=True,
                batch_size=1000,
                num_proc=8,
                fn_kwargs={"metrics_to_run": cpu_metrics}
            )

        # GPU metrics
        gpu_metrics = get_missing("gpu")
        if gpu_metrics:
            logger.info(f"Computing GPU metrics: {gpu_metrics}")
            self.dataset = self.dataset.map(
                self._run_gpu_batch,
                batched=True,
                batch_size=32,
                fn_kwargs={"metrics_to_run": gpu_metrics}
            )

        # DEPENDENT metrics
        dep_metrics = get_missing("dependent")
        if dep_metrics:
            logger.info(f"Computing Dependent metrics: {dep_metrics}")
            self.dataset = self.dataset.map(
                self._run_dependent_batch,
                batched=True,
                batch_size=1000,
                fn_kwargs={"metrics_to_run": dep_metrics}
            )
            
        return self.dataset

    