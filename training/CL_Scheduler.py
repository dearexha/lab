from datasets import Dataset, DatasetDict, concatenate_datasets
from transformers import DataCollatorForLanguageModeling
from training.CompetenceFunction import CompetenceFunction
import logging

logger = logging.getLogger(__name__)

class CL_Scheduler:
    def __init__(self, dataset_dict: DatasetDict, *, label_based: bool, difficulty_metric_name: str =None, label_schedule: list[list[int]] =None, competence_func: CompetenceFunction =None, tokenizer, num_bins: int =2):
        logger.info("Initializing CL_Scheduler ...")
        self.label_based = label_based
        self.difficulty_metric_name = difficulty_metric_name
        self.label_schedule = label_schedule
        self.num_bins = num_bins
        self.train_dataset_dict = DatasetDict({key.split("_")[-1]: val for key, val in dataset_dict.items() if "train" in key})
        self.val_dataset = dataset_dict["validation"]
        self.relevant_columns_for_training = ("input_ids", "attention_mask")
        self.competence_func = competence_func
        self.tokenizer = tokenizer

        # label_schedule validitiy check
        if self.label_based:
            label_schedule_is_valid = all(set(sub).issubset({0, 1}) for sub in label_schedule)
            if not label_schedule_is_valid:
                raise ValueError("CL Schedule for the labels contains non valid labels.")
        
        # difficulty_metric_name validity check
        if difficulty_metric_name not in self.val_dataset.column_names:
            raise ValueError("The CL difficulty_metric_name does not corresponds to a valid column of the dataset")
        
        if self.label_based: # label based CL
            if difficulty_metric_name and difficulty_metric_name!="label":
                # Bin difficulty metric into equal-sized bins for label-based CL
                logger.info(f"Binning training data by '{difficulty_metric_name}' into {num_bins} equal-sized bins...")

                # Step 1: Concatenate all training data
                all_train_data = concatenate_datasets(self.train_dataset_dict.values())
                n_total = len(all_train_data)

                # Step 2: Sort by difficulty metric (same logic as competence-based)
                reverse = (difficulty_metric_name in ["fre_score_words"])  # account for metrics where higher value indicates lower difficulty
                all_train_data = all_train_data.sort(difficulty_metric_name, reverse=reverse)

                # Step 3: Split into num_bins equal-frequency bins
                bin_size = n_total // num_bins
                new_train_dataset_dict = {}

                for i in range(num_bins):
                    start_idx = i * bin_size
                    end_idx = (i + 1) * bin_size if i < num_bins - 1 else n_total  # last bin gets remainder
                    bin_dataset = all_train_data.select(range(start_idx, end_idx))
                    new_train_dataset_dict[str(i)] = bin_dataset

                    # Log binning statistics
                    bin_metric_values = bin_dataset[difficulty_metric_name]
                    min_val = min(bin_metric_values)
                    max_val = max(bin_metric_values)
                    logger.info(f"  Bin {i}: {len(bin_dataset)} examples, {difficulty_metric_name} range [{min_val:.2f}, {max_val:.2f}]")

                # Step 4: Overwrite train_dataset_dict
                self.train_dataset_dict = new_train_dataset_dict
                logger.info(f"Binning complete. Train dataset reorganized into {num_bins} bins based on '{difficulty_metric_name}'")

                # Step 5: Set default label_schedule if not provided
                if self.label_schedule is None:
                    if num_bins == 2:
                        self.label_schedule = [[0], [0, 1]]  # Easy first, then all
                        logger.info(f"Using default label_schedule for num_bins=2: {self.label_schedule}")
                    else:
                        # TODO: Define sensible defaults for num_bins > 2 (cumulative vs separate stages)
                        raise ValueError(f"For num_bins={num_bins}, you must explicitly provide a label_schedule. Default schedules for num_bins > 2 are not yet implemented.")

            self.current_schedule_step = 0 # represents the index of the current subset of labels of the schedule

            self.current_train_subset = concatenate_datasets([self.train_dataset_dict[str(i)] for i in self.label_schedule[self.current_schedule_step]])
        
        else: # competence based CL
            reverse = (difficulty_metric_name in ["fre_score_words"])  # account for metrics, where higher value indicates lower difficulty
            self.train_dataset = concatenate_datasets(self.train_dataset_dict.values())
            self.train_dataset.sort(difficulty_metric_name, reverse=reverse)
            
            current_subset_size = int(self.competence_func.compute_competence(0) * len(self.train_dataset))
            self.current_train_subset = self.train_dataset.select(list(range(current_subset_size)))
        
        
        # set up masks and labels for the validation set -> using mask perplexity instead of pseudo perplexity
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, 
            mlm=True, 
            mlm_probability=0.15
        )
        def apply_masking(examples):
            features = []
            batch_size = len(examples['input_ids'])
            
            for i in range(batch_size):
                feature = {
                    'input_ids': examples['input_ids'][i],
                    'attention_mask': examples['attention_mask'][i],
                }
                if 'special_tokens_mask' in examples:
                    feature['special_tokens_mask'] = examples['special_tokens_mask'][i]
                    
                features.append(feature)

            batch = data_collator(features)
            
            return {
                'input_ids': batch['input_ids'],
                'attention_mask': batch['attention_mask'],
                'labels': batch['labels']
            }
        
        self.val_dataset = self.val_dataset.map(
            apply_masking,
            batched=True,
            batch_size=32,
            desc="computing validation masks",
        )
        self.val_dataset.set_format("torch", columns=self.relevant_columns_for_training + ("labels",), output_all_columns=False)
        self.current_train_subset.set_format("torch", columns=self.relevant_columns_for_training, output_all_columns=False)

    def get_current_train_subset(self):
        return self.current_train_subset

    def get_validation_set(self):
        return self.val_dataset
    
    def update_current_train_subset(self, t: int = None):
        """
        Updates the subset of the training data the model currently trains on and returns a flag indicating whether the
        training subset has been updated (i.e. could be updated any further).
        
        :param t: Current number of gradient update steps. Only relevant for competence based CL.
        :type t: int
        """
        if self.label_based:
            if self.current_schedule_step == len(self.label_schedule) - 1:  # we are done all subsets have been trained on until convergence
                return self.current_train_subset, False
            self.current_schedule_step += 1
            self.current_train_subset = concatenate_datasets([self.train_dataset_dict[str(i)] for i in self.label_schedule[self.current_schedule_step]])
        else:
            if t is None:
                raise ValueError("To update the current training subset, the current number of gradient update steps is required.")
            if t > self.competence_func.T:
                current_subset_size = len(self.train_dataset)
                self.current_train_subset = self.train_dataset.select(list(range(current_subset_size)))
                self.current_train_subset.set_format("torch", columns=self.relevant_columns_for_training, output_all_columns=False)
                return self.current_train_subset, False
            
            current_subset_size = int(self.competence_func.compute_competence(t) * len(self.train_dataset))
            self.current_train_subset = self.train_dataset.select(list(range(current_subset_size)))

        self.current_train_subset.set_format("torch", columns=self.relevant_columns_for_training, output_all_columns=False)

        return self.current_train_subset, True





