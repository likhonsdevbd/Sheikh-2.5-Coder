#!/usr/bin/env python3
"""
Main Training Script for Sheikh-2.5-Coder
Comprehensive training orchestration with distributed training, monitoring, and validation
"""

import os
import sys
import json
import argparse
import logging
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import signal
import threading

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import torch
import torch.distributed as dist
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from datasets import Dataset, load_dataset, load_from_disk
from torch.utils.data import DataLoader

# Import our custom modules
from training_config import (
    TrainingConfiguration, 
    create_default_config,
    load_config,
    save_config
)
from distributed_trainer import (
    DistributedTrainer,
    DistributedConfig,
    create_distributed_trainer
)
from checkpoint_manager import (
    CheckpointManager,
    create_checkpoint_manager
)
from monitoring_utils import (
    TrainingMonitor,
    create_monitoring_system
)
from early_stopping import (
    EarlyStoppingCallback as CustomEarlyStoppingCallback,
    EarlyStoppingConfig,
    create_early_stopping_callback
)
from validate_model import (
    ModelValidator,
    ValidationConfig,
    run_validation
)

# Try to import optional dependencies
try:
    from peft import LoraConfig, get_peft_model, TaskType
    LORA_AVAILABLE = True
except ImportError:
    LORA_AVAILABLE = False

try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False


class TrainingOrchestrator:
    """Main training orchestrator that coordinates all components"""
    
    def __init__(
        self,
        config: TrainingConfiguration,
        training_config: TrainingConfiguration
    ):
        self.config = config
        self.training_config = training_config
        
        # Initialize components
        self.distributed_trainer = None
        self.checkpoint_manager = None
        self.monitor = None
        self.validator = None
        
        # Training state
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.is_main_process = True
        
        # Setup logging
        self._setup_logging()
        
        # Setup signal handlers
        self._setup_signal_handlers()
    
    def _setup_logging(self) -> None:
        """Setup comprehensive logging"""
        log_dir = Path(self.config.monitoring.log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, self.config.monitoring.log_level),
            format=self.config.monitoring.log_format,
            handlers=[
                logging.FileHandler(self.config.monitoring.log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Training orchestrator initialized")
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            self.logger.warning(f"Received signal {signum}, preparing for graceful shutdown...")
            
            if self.trainer and hasattr(self.trainer, 'save_model'):
                try:
                    self.trainer.save_model()
                    self.logger.info("Model saved during shutdown")
                except Exception as e:
                    self.logger.error(f"Failed to save model during shutdown: {e}")
            
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def initialize_components(self) -> bool:
        """Initialize all training components"""
        try:
            self.logger.info("Initializing training components...")
            
            # Initialize distributed trainer
            self.distributed_trainer = create_distributed_trainer(
                use_distributed=self.config.system.multi_gpu,
                use_deepspeed=self.config.training.use_deepspeed,
                use_accelerate=True
            )
            
            # Setup distributed training
            if not self.distributed_trainer.setup_distributed(self.config.system.seed):
                self.logger.error("Failed to setup distributed training")
                return False
            
            self.is_main_process = self.distributed_trainer.is_main_process
            
            # Initialize checkpoint manager
            self.checkpoint_manager = create_checkpoint_manager(
                checkpoint_dir=self.config.checkpoint.checkpoint_dir,
                output_dir=self.config.checkpoint.output_dir,
                compression_type="gzip",
                max_checkpoints=self.config.checkpoint.save_total_limit,
                save_interval=self.config.checkpoint.save_steps,
                cloud_storage={
                    'provider': self.config.checkpoint.cloud_storage_provider,
                    'bucket': self.config.checkpoint.cloud_bucket,
                    'prefix': self.config.checkpoint.cloud_prefix
                } if self.config.checkpoint.use_cloud_storage else None
            )
            
            # Initialize monitoring
            self.monitor = create_monitoring_system(
                log_dir="logs",
                use_wandb=self.config.monitoring.use_wandb,
                wandb_project=self.config.monitoring.wandb_project,
                wandb_entity=self.config.monitoring.wandb_entity
            )
            
            # Initialize validator
            self.validator = None  # Will be created when needed
            
            self.logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def load_model_and_tokenizer(self) -> bool:
        """Load model and tokenizer"""
        try:
            self.logger.info(f"Loading model: {self.config.model.base_model}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model.base_model,
                trust_remote_code=True,
                additional_special_tokens=self.config.model.additional_special_tokens
            )
            
            # Configure tokenizer
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.tokenizer.padding_side = self.config.training.padding_side
            self.tokenizer.truncation_side = self.config.training.truncation_side
            
            # Load model
            model_kwargs = {
                "trust_remote_code": True,
                "use_cache": self.config.model.use_cache,
                "torch_dtype": torch.float16 if self.config.training.mixed_precision in ["fp16", "bf16"] else torch.float32,
            }
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model.base_model,
                **model_kwargs
            )
            
            # Configure model for training
            self.model.config.use_cache = self.config.model.use_cache
            self.model.config.max_position_embeddings = self.config.model.max_position_embeddings
            
            # Apply LoRA if configured
            if hasattr(self.config.model, 'use_lora') and self.config.model.use_lora:
                if LORA_AVAILABLE:
                    lora_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        r=self.config.model.lora_r,
                        lora_alpha=self.config.model.lora_alpha,
                        lora_dropout=self.config.model.lora_dropout,
                        target_modules=self.config.model.lora_target_modules
                    )
                    self.model = get_peft_model(self.model, lora_config)
                    self.logger.info("LoRA configuration applied")
                else:
                    self.logger.warning("LoRA not available, skipping LoRA configuration")
            
            # Move model to device
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            
            self.logger.info("Model and tokenizer loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model and tokenizer: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def load_data(self) -> tuple[Optional[Dataset], Optional[Dataset]]:
        """Load training and evaluation datasets"""
        try:
            self.logger.info("Loading training data...")
            
            # Load training data
            train_datasets = []
            for data_file in self.config.data.train_data_files:
                data_path = Path(data_file)
                if data_path.exists():
                    if data_path.suffix == ".json":
                        with open(data_path, 'r') as f:
                            data = json.load(f)
                        train_datasets.append(Dataset.from_list(data))
                    else:
                        train_datasets.append(load_from_disk(str(data_path)))
            
            if not train_datasets:
                self.logger.error("No training data files found")
                return None, None
            
            # Combine training datasets
            train_dataset = train_datasets[0]
            for ds in train_datasets[1:]:
                train_dataset = train_dataset.concatenate(ds)
            
            # Apply data filtering
            train_dataset = self._filter_dataset(train_dataset)
            
            # Limit dataset size if configured
            if self.config.data.max_train_samples:
                train_dataset = train_dataset.select(
                    range(min(self.config.data.max_train_samples, len(train_dataset)))
                )
            
            # Load evaluation data
            eval_dataset = None
            if self.config.data.eval_data_files:
                eval_datasets = []
                for data_file in self.config.data.eval_data_files:
                    data_path = Path(data_file)
                    if data_path.exists():
                        if data_path.suffix == ".json":
                            with open(data_path, 'r') as f:
                                data = json.load(f)
                            eval_datasets.append(Dataset.from_list(data))
                        else:
                            eval_datasets.append(load_from_disk(str(data_path)))
                
                if eval_datasets:
                    eval_dataset = eval_datasets[0]
                    for ds in eval_datasets[1:]:
                        eval_dataset = eval_dataset.concatenate(ds)
                    
                    # Apply data filtering
                    eval_dataset = self._filter_dataset(eval_dataset)
                    
                    # Limit dataset size if configured
                    if self.config.data.max_eval_samples:
                        eval_dataset = eval_dataset.select(
                            range(min(self.config.data.max_eval_samples, len(eval_dataset)))
                        )
            
            self.logger.info(f"Training data: {len(train_dataset)} samples")
            if eval_dataset:
                self.logger.info(f"Evaluation data: {len(eval_dataset)} samples")
            
            return train_dataset, eval_dataset
            
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            self.logger.error(traceback.format_exc())
            return None, None
    
    def _filter_dataset(self, dataset: Dataset) -> Dataset:
        """Filter dataset based on configuration"""
        def filter_function(example):
            text = example.get('text', example.get('content', ''))
            
            # Length checks
            if len(text) < self.config.data.min_text_length:
                return False
            
            if len(text) > self.config.data.max_text_length:
                return False
            
            # Remove empty sequences
            if self.config.data.remove_empty_sequences and not text.strip():
                return False
            
            return True
        
        # Apply filters
        filtered_dataset = dataset.filter(
            filter_function,
            num_proc=self.config.data.preprocessing_num_workers
        )
        
        return filtered_dataset
    
    def setup_trainer(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset]
    ) -> bool:
        """Setup the trainer with all configurations"""
        try:
            self.logger.info("Setting up trainer...")
            
            # Get training arguments
            training_args_dict = self.config.get_training_arguments()
            
            # Create training arguments
            training_args = TrainingArguments(**training_args_dict)
            
            # Create data collator
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                model=self.model,
                padding=True,
                return_tensors="pt"
            )
            
            # Create callbacks
            callbacks = []
            
            # Early stopping callback
            if self.config.training.evaluation_strategy != "no":
                early_stopping_cb = create_early_stopping_callback(
                    metric_to_monitor=self.config.checkpoint.metric_for_best_model,
                    metric_mode="min" if not self.config.checkpoint.greater_is_better else "max",
                    patience=3
                )
                callbacks.append(early_stopping_cb)
            
            # Create trainer
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                callbacks=callbacks
            )
            
            self.logger.info("Trainer setup completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup trainer: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def train(self) -> bool:
        """Run the complete training process"""
        try:
            self.logger.info("Starting training process...")
            
            # Start monitoring
            self.monitor.start()
            
            # Log training configuration
            self.monitor.log_hyperparameters(self.config.get_training_arguments())
            self.monitor.log_model_info(self.config.model.__dict__)
            
            # Prepare model and data for distributed training
            model, train_dataloader, eval_dataloader = self.distributed_trainer.prepare_model_and_data(
                self.model,
                self.tokenizer,
                self.trainer.train_dataset,
                self.trainer.eval_dataset
            )
            
            # Start training
            self.logger.info("Starting distributed training...")
            
            # Use distributed trainer for actual training
            final_model = self.distributed_trainer.train(
                model=model,
                tokenizer=self.tokenizer,
                train_dataset=self.trainer.train_dataset,
                eval_dataset=self.trainer.eval_dataset,
                training_args=self.trainer.args,
                callbacks=self.trainer.callbacks
            )
            
            # Update local model reference
            self.model = final_model
            
            # Save final model
            if self.is_main_process:
                self.trainer.save_model()
                
                # Save training configuration
                config_file = Path(self.config.checkpoint.output_dir) / "training_config.yaml"
                save_config(self.config, config_file)
            
            # Stop monitoring
            self.monitor.stop()
            
            self.logger.info("Training completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def validate_model(self) -> bool:
        """Run model validation after training"""
        try:
            if not self.is_main_process:
                return True  # Only main process runs validation
            
            self.logger.info("Running post-training validation...")
            
            # Create validation configuration
            val_config = ValidationConfig(
                model_path=self.config.checkpoint.output_dir,
                tokenizer_path=self.config.checkpoint.output_dir,
                device="auto",
                xml_validation=True,
                js_validation=True,
                validation_datasets=["HumanEval", "mbpp"],
                output_dir=str(Path(self.config.checkpoint.output_dir) / "validation_results"),
                save_detailed_results=True,
                generate_report=True
            )
            
            # Create validator and run validation
            validator = ModelValidator(val_config)
            
            if not validator.load_model():
                self.logger.error("Failed to load model for validation")
                return False
            
            results = validator.validate_model()
            
            # Log validation results
            self.logger.info("Validation completed")
            self.logger.info(f"Results saved to: {val_config.output_dir}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        try:
            if self.distributed_trainer:
                self.distributed_trainer.cleanup()
            
            if self.monitor:
                self.monitor.finish()
            
            # Clean up distributed processes
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
            
            self.logger.info("Cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train Sheikh-2.5-Coder model with comprehensive infrastructure"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (YAML or JSON)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/output",
        help="Output directory for models and results"
    )
    
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        help="Resume training from checkpoint"
    )
    
    parser.add_argument(
        "--validate_after_training",
        action="store_true",
        help="Run validation after training"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Perform dry run without actual training"
    )
    
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        help="Local rank for distributed training"
    )
    
    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_arguments()
    
    # Setup basic logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Load or create configuration
        if args.config and Path(args.config).exists():
            logger.info(f"Loading configuration from {args.config}")
            config = load_config(args.config)
        else:
            logger.info("Creating default configuration")
            config = create_default_config()
        
        # Override output directory
        config.checkpoint.output_dir = args.output_dir
        config.checkpoint.resume_from_checkpoint = args.resume_from_checkpoint
        
        # Set debug mode
        config.system.debug_mode = args.debug
        
        # Validate configuration
        if not config.validate():
            logger.error("Configuration validation failed")
            return 1
        
        # Create orchestrator
        orchestrator = TrainingOrchestrator(config, config)
        
        # Initialize components
        if not orchestrator.initialize_components():
            logger.error("Failed to initialize components")
            return 1
        
        # Load model and tokenizer
        if not orchestrator.load_model_and_tokenizer():
            logger.error("Failed to load model and tokenizer")
            return 1
        
        # Load data
        train_dataset, eval_dataset = orchestrator.load_data()
        if train_dataset is None:
            logger.error("Failed to load training data")
            return 1
        
        # Setup trainer
        if not orchestrator.setup_trainer(train_dataset, eval_dataset):
            logger.error("Failed to setup trainer")
            return 1
        
        # Dry run
        if args.dry_run:
            logger.info("Dry run completed successfully")
            return 0
        
        # Run training
        if not orchestrator.train():
            logger.error("Training failed")
            return 1
        
        # Run validation if requested
        if args.validate_after_training:
            if not orchestrator.validate_model():
                logger.error("Validation failed")
                return 1
        
        logger.info("Training pipeline completed successfully")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        logger.error(traceback.format_exc())
        return 1
    finally:
        try:
            orchestrator.cleanup()
        except:
            pass


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)