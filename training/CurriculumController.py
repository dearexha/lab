from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
from datasets import Dataset
import copy
import logging

from training.CL_Scheduler import CL_Scheduler

logger = logging.getLogger(__name__)


class PatienceState:
    """Encapsulates patience tracking state."""
    def __init__(self):
        self.counter: int = 0
        self.best_val_loss: float = float('inf')
        self.best_model_state: Optional[Dict[str, Any]] = None


class CurriculumController(ABC):
    """
    Abstract interface for curriculum learning controllers.
    Encapsulates all curriculum-specific logic, making the training loop curriculum-agnostic.
    """
    
    def __init__(self, cl_scheduler: CL_Scheduler, config: Dict):
        self.cl_scheduler = cl_scheduler
        self.config = config
    
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
    
    @abstractmethod
    def get_current_phase_description(self) -> str:
        """
        Get a description of the current curriculum phase.
        Used for logging when phases change.
        
        Returns:
            String describing current phase (e.g., "labels=[0, 1]" or "competence=0.5")
        """
        pass


class CompetenceBasedController(CurriculumController):
    """Controller for competence-based curriculum learning."""
    
    def __init__(self, cl_scheduler: CL_Scheduler, config: Dict):
        super().__init__(cl_scheduler, config)
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
            logger.info(f"No improvement in validation loss. Patience counter: {patience_state.counter}/{self.config['patience']}")
            
        if patience_state.counter >= self.config["patience"]:
            logger.info(
                f"Validation loss did not improve for {self.config['patience']} consecutive checks. Reverting and stopping training.")
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
    
    def get_current_phase_description(self) -> str:
        return f"competence={self.current_competence:.4f}"


class LabelBasedController(CurriculumController):
    """Controller for label-based curriculum learning."""
    
    def __init__(self, cl_scheduler: CL_Scheduler, config: Dict):
        super().__init__(cl_scheduler, config)
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
        Note: Patience state persists across label subsets (preserving original behavior).
        """
        if val_loss < patience_state.best_val_loss:
            patience_state.best_val_loss = val_loss
            patience_state.counter = 0
            return False, False  # No update, continue training
        else:
            patience_state.counter += 1
            logger.info(f"No improvement in validation loss. Patience counter: {patience_state.counter}/{self.config['patience']}")
            
        if patience_state.counter >= self.config["patience"]:
            # Patience exhausted: advance to next label subset
            logger.info(
                f"Validation loss did not improve for {self.config['patience']} consecutive checks. Reverting and stopping training for current label subset.")
            logger.info(f"Completed training on current label subset: {self.current_label_subset}")
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
    
    def get_current_phase_description(self) -> str:
        return f"labels={self.current_label_subset}"


def create_controller(cl_scheduler: CL_Scheduler, config: Dict) -> CurriculumController:
    """
    Factory function to create the appropriate controller based on config.
    
    Args:
        cl_scheduler: The curriculum learning scheduler
        config: Configuration dictionary
        
    Returns:
        Appropriate CurriculumController instance
    """
    if config["label_based"]:
        return LabelBasedController(cl_scheduler, config)
    else:
        return CompetenceBasedController(cl_scheduler, config)

