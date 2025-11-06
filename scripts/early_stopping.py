#!/usr/bin/env python3
"""
Early Stopping Implementation for Sheikh-2.5-Coder Training
Implements various early stopping strategies and patience mechanisms
"""

import os
import logging
import numpy as np
from typing import Dict, List, Optional, Union, Callable, Any
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timedelta
import json
import math

import torch
from transformers import TrainerControl, TrainerState
from transformers.trainer_utils import EvalPrediction


@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping"""
    # Basic early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.0
    
    # Minimum training time before early stopping can trigger
    min_train_time_seconds: int = 3600  # 1 hour
    min_steps: int = 1000
    
    # Metric requirements
    metric_to_monitor: str = "eval_loss"
    metric_mode: str = "min"  # "min" or "max"
    
    # Multiple metrics support
    multiple_metrics: Dict[str, str] = field(default_factory=dict)  # metric_name -> mode
    
    # Learning rate based early stopping
    lr_scheduler_type: str = "plateau"
    lr_scheduler_patience: int = 2
    lr_scheduler_factor: float = 0.5
    lr_scheduler_threshold: float = 0.0001
    lr_scheduler_verbose: bool = True
    
    # Gradient-based early stopping
    use_gradient_stopping: bool = False
    gradient_threshold: float = 1e-6
    gradient_patience: int = 5
    
    # Performance degradation stopping
    use_performance_stopping: bool = False
    performance_patience: int = 3
    performance_degradation_threshold: float = 0.05
    
    # Validation frequency
    validation_interval_steps: int = 500
    validation_budget_steps: Optional[int] = None  # Maximum validation steps
    
    # Warmup considerations
    ignore_warmup_steps: int = 1000
    ignore_warmup_time_seconds: int = 1800  # 30 minutes
    
    # Recovery strategies
    best_model_recovery: bool = True
    recovery_patience: int = 2
    recovery_threshold: float = 0.01
    
    # Save and resume
    save_best_on_early_stop: bool = True
    save_checkpoint_before_stop: bool = True
    
    # Logging and callbacks
    log_every_n_steps: int = 100
    save_history: bool = True
    history_file: str = "logs/early_stopping_history.json"


@dataclass
class EarlyStoppingState:
    """State tracking for early stopping"""
    best_metric: float = float('inf')
    best_step: int = 0
    best_epoch: float = 0.0
    wait_count: int = 0
    current_patience: int = 0
    lr_wait_count: int = 0
    gradient_wait_count: int = 0
    performance_wait_count: int = 0
    recovery_wait_count: int = 0
    
    # Learning rate monitoring
    last_lr: float = 0.0
    lr_history: List[float] = field(default_factory=list)
    
    # Gradient monitoring
    grad_norm_history: List[float] = field(default_factory=list)
    
    # Performance monitoring
    performance_history: List[float] = field(default_factory=list)
    
    # Validation history
    validation_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Early stopping triggers
    triggered: bool = False
    trigger_reason: Optional[str] = None
    trigger_step: int = 0
    trigger_time: Optional[str] = None
    
    # Training time tracking
    training_start_time: Optional[str] = None
    last_validation_time: Optional[str] = None
    
    # Recovery tracking
    is_recovering: bool = False
    recovery_attempts: int = 0
    
    # History
    stopping_history: List[Dict[str, Any]] = field(default_factory=list)


class EarlyStoppingCallback:
    """Custom callback for early stopping in HuggingFace Trainer"""
    
    def __init__(
        self,
        config: EarlyStoppingConfig,
        metric_function: Optional[Callable] = None,
        save_function: Optional[Callable] = None,
        log_function: Optional[Callable] = None
    ):
        self.config = config
        self.metric_function = metric_function
        self.save_function = save_function
        self.log_function = log_function
        self.early_stopping = EarlyStoppingLogic(config)
        
        # Training state
        self.state = EarlyStoppingState()
        self.step_counter = 0
        self.epoch_counter = 0
        
        logging.info("Early stopping callback initialized")
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training"""
        self.state.training_start_time = datetime.now().isoformat()
        logging.info("Training started - early stopping monitoring active")
    
    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step"""
        self.step_counter = state.global_step
        self.epoch_counter = state.epoch
        
        # Update learning rate history
        if 'learning_rate' in kwargs:
            self.state.last_lr = kwargs['learning_rate']
            self.state.lr_history.append(kwargs['learning_rate'])
        
        # Update gradient norm history
        if 'grad_norm' in kwargs:
            self.state.grad_norm_history.append(kwargs['grad_norm'])
        
        # Check for gradient-based early stopping
        if self.config.use_gradient_stopping:
            if self._check_gradient_stopping():
                control.should_training_stop = True
                self.state.triggered = True
                self.state.trigger_reason = "gradient_based"
                self.state.trigger_step = self.step_counter
                self.state.trigger_time = datetime.now().isoformat()
                
                logging.warning(f"Early stopping triggered: {self.state.trigger_reason}")
        
        # Update validation timing
        if state.do_eval:
            self.state.last_validation_time = datetime.now().isoformat()
    
    def on_evaluate(self, args, state, control, **kwargs):
        """Called after each evaluation"""
        self._update_state_from_evaluation(state, kwargs.get('logs', {}))
        
        # Check early stopping conditions
        should_stop = self.early_stopping.should_stop(self.state, self.step_counter, self.epoch_counter)
        
        if should_stop:
            control.should_training_stop = True
            logging.warning(f"Early stopping triggered at step {self.step_counter}")
            
            # Save best model if configured
            if self.config.save_best_on_early_stop and self.save_function:
                try:
                    self.save_function(self.step_counter)
                    logging.info("Best model saved on early stopping")
                except Exception as e:
                    logging.error(f"Failed to save best model: {e}")
        
        # Save history if configured
        if self.config.save_history:
            self._save_history()
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Called at the end of each epoch"""
        # Check learning rate scheduler based stopping
        if self.config.lr_scheduler_type == "plateau" and self.state.lr_history:
            if self._check_lr_stopping():
                logging.warning("Early stopping triggered: learning rate plateau")
                control.should_training_stop = True
    
    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training"""
        logging.info(f"Training completed. Early stopping was {'triggered' if self.state.triggered else 'not triggered'}")
        
        # Save final history
        if self.config.save_history:
            self._save_history()
        
        # Print final status
        self._print_final_status()
    
    def _update_state_from_evaluation(self, state: TrainerState, logs: Dict[str, Any]) -> None:
        """Update early stopping state from evaluation results"""
        # Get metric value
        metric_value = None
        if self.metric_function:
            metric_value = self.metric_function(logs)
        elif self.config.metric_to_monitor in logs:
            metric_value = logs[self.config.metric_to_monitor]
        
        if metric_value is not None:
            # Update validation history
            validation_entry = {
                'step': state.global_step,
                'epoch': state.epoch,
                'timestamp': datetime.now().isoformat(),
                'metrics': logs.copy()
            }
            self.state.validation_history.append(validation_entry)
            
            # Update best metric
            if self._is_better_metric(metric_value, self.state.best_metric):
                self.state.best_metric = metric_value
                self.state.best_step = state.global_step
                self.state.best_epoch = state.epoch
                self.state.wait_count = 0
                self.state.current_patience = 0
            else:
                self.state.wait_count += 1
                self.state.current_patience += 1
            
            logging.debug(f"Step {state.global_step}: {self.config.metric_to_monitor} = {metric_value:.6f}, patience = {self.state.current_patience}")
    
    def _is_better_metric(self, current: float, best: float) -> bool:
        """Check if current metric is better than best metric"""
        if self.config.metric_mode == "min":
            return current < best - self.config.early_stopping_threshold
        else:  # max
            return current > best + self.config.early_stopping_threshold
    
    def _check_gradient_stopping(self) -> bool:
        """Check for gradient-based early stopping"""
        if len(self.state.grad_norm_history) < self.config.gradient_patience:
            return False
        
        # Check if gradients have been consistently small
        recent_grads = self.state.grad_norm_history[-self.config.gradient_patience:]
        avg_grad = np.mean(recent_grads)
        
        if avg_grad < self.config.gradient_threshold:
            self.state.gradient_wait_count += 1
            
            if self.state.gradient_wait_count >= self.config.gradient_patience:
                return True
        else:
            self.state.gradient_wait_count = 0
        
        return False
    
    def _check_lr_stopping(self) -> bool:
        """Check for learning rate based early stopping"""
        if len(self.state.lr_history) < self.config.lr_scheduler_patience:
            return False
        
        # Check for plateau in learning rate (indicating no improvement)
        recent_lrs = self.state.lr_history[-self.config.lr_scheduler_patience:]
        lr_variance = np.var(recent_lrs)
        
        if lr_variance < self.config.lr_scheduler_threshold:
            self.state.lr_wait_count += 1
            
            if self.state.lr_wait_count >= self.config.lr_scheduler_patience:
                return True
        else:
            self.state.lr_wait_count = 0
        
        return False
    
    def _save_history(self) -> None:
        """Save early stopping history to file"""
        try:
            history_file = Path(self.config.history_file)
            history_file.parent.mkdir(parents=True, exist_ok=True)
            
            history_data = {
                'config': self.config.__dict__,
                'state': self.state.__dict__,
                'step_counter': self.step_counter,
                'epoch_counter': self.epoch_counter,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(history_file, 'w') as f:
                json.dump(history_data, f, indent=2, default=str)
                
        except Exception as e:
            logging.error(f"Failed to save early stopping history: {e}")
    
    def _print_final_status(self) -> None:
        """Print final training status"""
        print("\n" + "="*60)
        print("TRAINING STATUS SUMMARY")
        print("="*60)
        print(f"Best {self.config.metric_to_monitor}: {self.state.best_metric:.6f}")
        print(f"Best step: {self.state.best_step}")
        print(f"Best epoch: {self.state.best_epoch:.2f}")
        print(f"Total steps: {self.step_counter}")
        print(f"Total epochs: {self.epoch_counter:.2f}")
        print(f"Validation runs: {len(self.state.validation_history)}")
        print(f"Early stopping triggered: {self.state.triggered}")
        
        if self.state.triggered:
            print(f"Trigger reason: {self.state.trigger_reason}")
            print(f"Trigger step: {self.state.trigger_step}")
        
        if self.state.training_start_time:
            start_time = datetime.fromisoformat(self.state.training_start_time)
            end_time = datetime.now()
            duration = end_time - start_time
            print(f"Training duration: {duration}")
        
        print("="*60)


class EarlyStoppingLogic:
    """Core early stopping logic"""
    
    def __init__(self, config: EarlyStoppingConfig):
        self.config = config
    
    def should_stop(
        self,
        state: EarlyStoppingState,
        current_step: int,
        current_epoch: float
    ) -> bool:
        """Determine if training should stop"""
        
        # Check minimum training time
        if not self._meets_minimum_requirements(current_step, current_epoch):
            return False
        
        # Check patience-based stopping
        if self._check_patience_stopping(state):
            return True
        
        # Check performance degradation stopping
        if self.config.use_performance_stopping:
            if self._check_performance_stopping(state):
                return True
        
        # Check validation budget
        if self._exceeds_validation_budget(current_step):
            return True
        
        return False
    
    def _meets_minimum_requirements(self, current_step: int, current_epoch: float) -> bool:
        """Check if minimum training requirements are met"""
        # Check minimum steps
        if current_step < self.config.min_steps:
            return False
        
        # Check minimum time (would need training start time to check properly)
        # For now, we'll use a conservative estimate based on steps
        if current_step < self.config.ignore_warmup_steps:
            return False
        
        return True
    
    def _check_patience_stopping(self, state: EarlyStoppingState) -> bool:
        """Check patience-based stopping condition"""
        # Check primary metric patience
        if state.current_patience >= self.config.early_stopping_patience:
            return True
        
        # Check multiple metrics if configured
        for metric_name, mode in self.config.multiple_metrics.items():
            # This would require additional tracking for each metric
            # Simplified implementation for now
            pass
        
        return False
    
    def _check_performance_stopping(self, state: EarlyStoppingState) -> bool:
        """Check performance degradation stopping"""
        if len(state.validation_history) < self.config.performance_patience + 1:
            return False
        
        # Get recent validation results for the primary metric
        recent_validations = state.validation_history[-self.config.performance_patience-1:]
        
        if len(recent_validations) < 2:
            return False
        
        # Check if performance has been consistently degrading
        current_metric = self._extract_metric_from_validation(recent_validations[-1])
        previous_metric = self._extract_metric_from_validation(recent_validations[-2])
        
        if current_metric is None or previous_metric is None:
            return False
        
        # Calculate degradation percentage
        if self.config.metric_mode == "min":
            degradation = (current_metric - previous_metric) / previous_metric
        else:
            degradation = (previous_metric - current_metric) / previous_metric
        
        if degradation > self.config.performance_degradation_threshold:
            state.performance_wait_count += 1
            
            if state.performance_wait_count >= self.config.performance_patience:
                return True
        else:
            state.performance_wait_count = 0
        
        return False
    
    def _exceeds_validation_budget(self, current_step: int) -> bool:
        """Check if validation budget is exceeded"""
        if self.config.validation_budget_steps is None:
            return False
        
        return current_step >= self.config.validation_budget_steps
    
    def _extract_metric_from_validation(self, validation_entry: Dict[str, Any]) -> Optional[float]:
        """Extract metric value from validation entry"""
        metrics = validation_entry.get('metrics', {})
        return metrics.get(self.config.metric_to_monitor)


class LearningRateScheduler:
    """Custom learning rate scheduler for early stopping"""
    
    def __init__(self, optimizer: torch.optim.Optimizer, config: EarlyStoppingConfig):
        self.optimizer = optimizer
        self.config = config
        self.last_lr = optimizer.param_groups[0]['lr']
        self.num_bad_epochs = 0
        self.best = float('inf')
        self.mode = config.metric_mode
        
        # For plateau scheduler
        if config.lr_scheduler_type == "plateau":
            self.factor = config.lr_scheduler_factor
            self.patience = config.lr_scheduler_patience
            self.threshold = config.lr_scheduler_threshold
            self.verbose = config.lr_scheduler_verbose
    
    def step(self, metrics: float, epoch: Optional[int] = None) -> None:
        """Update learning rate based on metrics"""
        if self.config.lr_scheduler_type == "plateau":
            self._step_plateau(metrics)
        elif self.config.lr_scheduler_type == "exponential":
            self._step_exponential()
    
    def _step_plateau(self, metrics: float) -> None:
        """Step for plateau scheduler"""
        if self.mode == "min":
            if metrics < self.best - self.threshold:
                self.best = metrics
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1
        
        else:  # max mode
            if metrics > self.best + self.threshold:
                self.best = metrics
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1
        
        if self.num_bad_epochs >= self.patience:
            self._reduce_lr()
            self.num_bad_epochs = 0
    
    def _step_exponential(self) -> None:
        """Step for exponential scheduler"""
        self._reduce_lr()
    
    def _reduce_lr(self) -> None:
        """Reduce learning rate"""
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            if self.config.lr_scheduler_type == "plateau":
                new_lr = old_lr * self.factor
            else:  # exponential
                new_lr = old_lr * 0.95  # Default exponential decay
            
            param_group['lr'] = new_lr
            
            if self.config.lr_scheduler_verbose:
                print(f'Reducing learning rate of group {id(param_group)} to {new_lr:.4e}.')


class EarlyStoppingManager:
    """Manager for early stopping across multiple training runs"""
    
    def __init__(self, history_file: str = "logs/early_stopping_summary.json"):
        self.history_file = Path(history_file)
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.runs_history = []
        self._load_history()
    
    def add_run(
        self,
        run_name: str,
        config: EarlyStoppingConfig,
        final_state: EarlyStoppingState,
        success: bool = True
    ) -> None:
        """Add a training run to history"""
        run_summary = {
            'run_name': run_name,
            'config': config.__dict__,
            'final_state': final_state.__dict__,
            'success': success,
            'timestamp': datetime.now().isoformat()
        }
        
        self.runs_history.append(run_summary)
        self._save_history()
    
    def get_best_runs(
        self,
        metric: str = "best_metric",
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """Get best performing runs"""
        # Filter successful runs
        successful_runs = [run for run in self.runs_history if run['success']]
        
        # Sort by metric
        if metric == "best_metric":
            successful_runs.sort(
                key=lambda x: x['final_state']['best_metric'],
                reverse=True  # Assuming lower is better for most metrics
            )
        elif metric == "shortest_time":
            # Sort by training duration
            successful_runs.sort(key=lambda x: self._get_duration(x))
        
        return successful_runs[:top_n]
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics across all runs"""
        if not self.runs_history:
            return {}
        
        successful_runs = [run for run in self.runs_history if run['success']]
        
        if not successful_runs:
            return {'total_runs': len(self.runs_history), 'successful_runs': 0}
        
        # Extract metrics
        best_metrics = [run['final_state']['best_metric'] for run in successful_runs]
        durations = [self._get_duration(run) for run in successful_runs]
        total_steps = [run['final_state']['best_step'] for run in successful_runs]
        
        return {
            'total_runs': len(self.runs_history),
            'successful_runs': len(successful_runs),
            'success_rate': len(successful_runs) / len(self.runs_history),
            'best_metric_stats': {
                'mean': np.mean(best_metrics),
                'std': np.std(best_metrics),
                'min': np.min(best_metrics),
                'max': np.max(best_metrics)
            },
            'duration_stats': {
                'mean': np.mean(durations),
                'std': np.std(durations),
                'min': np.min(durations),
                'max': np.max(durations)
            },
            'steps_stats': {
                'mean': np.mean(total_steps),
                'std': np.std(total_steps),
                'min': np.min(total_steps),
                'max': np.max(total_steps)
            }
        }
    
    def _get_duration(self, run: Dict[str, Any]) -> float:
        """Get duration of a training run in seconds"""
        # This is a simplified implementation
        # In practice, you'd need to parse the timestamps properly
        try:
            start_time = datetime.fromisoformat(run['final_state']['training_start_time'])
            end_time = datetime.fromisoformat(run['timestamp'])
            return (end_time - start_time).total_seconds()
        except:
            return 0.0
    
    def _load_history(self) -> None:
        """Load history from file"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    self.runs_history = data.get('runs', [])
            except Exception as e:
                logging.error(f"Failed to load early stopping history: {e}")
                self.runs_history = []
    
    def _save_history(self) -> None:
        """Save history to file"""
        try:
            history_data = {
                'runs': self.runs_history,
                'summary': self.get_summary_statistics(),
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.history_file, 'w') as f:
                json.dump(history_data, f, indent=2, default=str)
                
        except Exception as e:
            logging.error(f"Failed to save early stopping history: {e}")


def create_early_stopping_callback(
    metric_to_monitor: str = "eval_loss",
    metric_mode: str = "min",
    patience: int = 3,
    threshold: float = 0.0,
    **kwargs
) -> EarlyStoppingCallback:
    """Create early stopping callback with default settings"""
    
    config = EarlyStoppingConfig(
        metric_to_monitor=metric_to_monitor,
        metric_mode=metric_mode,
        early_stopping_patience=patience,
        early_stopping_threshold=threshold,
        **kwargs
    )
    
    return EarlyStoppingCallback(config)


if __name__ == "__main__":
    # Example usage
    import torch.optim as optim
    
    # Create dummy optimizer
    optimizer = optim.Adam([torch.randn(1, requires_grad=True)], lr=1e-3)
    
    # Create early stopping configuration
    config = EarlyStoppingConfig(
        metric_to_monitor="eval_loss",
        metric_mode="min",
        early_stopping_patience=3,
        early_stopping_threshold=0.001,
        min_steps=500,
        save_best_on_early_stop=True
    )
    
    # Create callback
    callback = EarlyStoppingCallback(config)
    
    # Simulate training loop with early stopping
    print("Simulating training with early stopping...")
    
    # Create dummy metric function
    def dummy_metric(logs):
        return logs.get("eval_loss", 1.0)
    
    callback.metric_function = dummy_metric
    
    # Simulate some training steps
    state = TrainerState()
    control = TrainerControl()
    
    callback.on_train_begin(None, state, control)
    
    for step in range(10):
        # Simulate training step
        state.global_step = step
        state.epoch = step / 10.0
        
        # Simulate increasing loss to trigger early stopping
        fake_logs = {"eval_loss": 2.0 + step * 0.1}
        
        callback.on_evaluate(None, state, control, logs=fake_logs)
        
        if control.should_training_stop:
            print(f"Early stopping triggered at step {step}")
            break
        
        time.sleep(0.1)  # Simulate training time
    
    callback.on_train_end(None, state, control)
    
    # Print final summary
    print(f"Best metric: {callback.state.best_metric:.4f}")
    print(f"Best step: {callback.state.best_step}")
    print(f"Validation runs: {len(callback.state.validation_history)}")