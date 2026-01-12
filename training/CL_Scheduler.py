from datasets import Dataset, DatasetDict, concatenate_datasets
from transformers import DataCollatorForLanguageModeling
from training.CompetenceFunction import CompetenceFunction
import logging

logger = logging.getLogger(__name__)

class CL_Scheduler:
    def __init__(self, dataset_dict: DatasetDict, *, label_based: bool, difficulty_metric_name: str =None, label_schedule: list[list[int]] =None, competence_func: CompetenceFunction =None, tokenizer):
        logger.info("Initializing CL_Scheduler ...")
        self.label_based = label_based
        self.difficulty_metric_name = difficulty_metric_name
        self.label_schedule = label_schedule
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
                # TODO: set up a difficulty metric for label based CL as a binary label by simply binning with equal bin sizes and split, Overwrite self.train_dataset_dict
                pass
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





