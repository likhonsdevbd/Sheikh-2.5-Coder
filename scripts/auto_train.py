#!/usr/bin/env python3
"""
Automated Training Script for Sheikh-2.5-Coder
Handles distributed training, monitoring, and checkpoint management
"""

import os
import sys
import json
import argparse
import logging
import torch
import torch.distributed as dist
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import traceback
import subprocess
import time
import wandb
import shutil

# Add src to path
sys.path.append('src')
sys.path.append('../src')

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, TaskType
from accelerate import Accelerator, DistributedDataParallelKwargs
import bitsandbytes as bnb

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ModelConfig:
    """Configuration class for model training"""
    
    def __init__(self, args):
        self.model_name = args.model_name
        self.data_path = args.data_path
        self.output_path = args.output_path
        self.gpu_type = args.gpu_type
        self.training_steps = int(args.training_steps)
        self.batch_size = int(args.batch_size)
        self.learning_rate = float(args.learning_rate)
        self.run_id = args.run_id
        self.timestamp = args.timestamp
        
        # Auto-detect GPU configuration
        self.setup_gpu_config()
        
        # Create directories
        self.setup_directories()
        
        # Training parameters
        self.setup_training_params()
    
    def setup_gpu_config(self):
        """Configure GPU settings based on type"""
        if torch.cuda.is_available():
            self.device_count = torch.cuda.device_count()
            logger.info(f"Available GPUs: {self.device_count}")
            logger.info(f"GPU Type: {self.gpu_type}")
            
            # GPU-specific configurations
            gpu_configs = {
                'a100': {'max_memory_gb': 80, 'batch_size_multiplier': 4},
                'v100': {'max_memory_gb': 32, 'batch_size_multiplier': 2},
                't4': {'max_memory_gb': 16, 'batch_size_multiplier': 1}
            }
            
            self.gpu_config = gpu_configs.get(self.gpu_type, gpu_configs['t4'])
            
            # Set memory management
            if hasattr(torch.backends.cuda, 'cufft_plan_cache'):
                torch.backends.cuda.cufft_plan_cache.max_size = 512
        else:
            logger.warning("CUDA not available - using CPU")
            self.device_count = 1
            self.gpu_config = {'max_memory_gb': 8, 'batch_size_multiplier': 0.5}
    
    def setup_directories(self):
        """Create necessary directories"""
        directories = [
            'logs',
            'models/checkpoints',
            'models/metrics',
            'evaluation/results'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def setup_training_params(self):
        """Setup training parameters with optimizations"""
        # Adjust batch size based on GPU
        effective_batch_size = int(self.batch_size * self.gpu_config['batch_size_multiplier'])
        
        # Memory-efficient settings
        self.training_args = {
            'output_dir': self.output_path,
            'num_train_epochs': 1,
            'max_steps': self.training_steps,
            'per_device_train_batch_size': max(1, effective_batch_size // self.device_count),
            'per_device_eval_batch_size': max(1, effective_batch_size // self.device_count // 2),
            'gradient_accumulation_steps': 4,
            'learning_rate': self.learning_rate,
            'weight_decay': 0.01,
            'warmup_steps': 100,
            'lr_scheduler_type': 'cosine',
            'logging_steps': 10,
            'eval_steps': 500,
            'save_steps': 1000,
            'save_total_limit': 3,
            'load_best_model_at_end': True,
            'metric_for_best_model': 'eval_loss',
            'greater_is_better': False,
            'remove_unused_columns': False,
            'report_to': 'wandb',
            'run_name': f'sheikh-2.5-coder-{self.run_id}',
            
            # Memory optimization
            'fp16': True,
            'gradient_checkpointing': True,
            'dataloader_pin_memory': False,
            'dataloader_num_workers': 2,
            
            # Mixed precision training
            'bf16': False,
            
            # Distributed training
            'ddp_backend': 'nccl',
            'ddp_find_unused_parameters': False,
            
            # Logging
            'logging_dir': 'logs',
            'logging_strategy': 'steps',
            
            # Evaluation
            'evaluation_strategy': 'steps',
            'eval_delay': 1000,
            
            # Early stopping
            'early_stopping_patience': 3,
            'early_stopping_threshold': 0.001,
        }
        
        # LoRA configuration for efficient fine-tuning
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,  # Rank
            lora_alpha=32,  # Scaling parameter
            lora_dropout=0.1,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            bias='none',
            task_type='CAUSAL_LM',
        )


class DataProcessor:
    """Handles data loading and preprocessing"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.tokenizer = None
        
    def load_tokenizer(self):
        """Load and configure tokenizer"""
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # Add pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        # Set tokenizer properties for code
        self.tokenizer.model_max_length = 2048
        self.tokenizer.truncation_side = "left"
        
        logger.info(f"Tokenizer loaded. Vocab size: {len(self.tokenizer)}")
        
    def load_dataset(self):
        """Load and preprocess training dataset"""
        logger.info("Loading dataset...")
        
        # Check for local dataset
        data_path = Path(self.config.data_path)
        if data_path.exists():
            # Load local processed data
            train_files = list(data_path.glob("train_*.jsonl"))
            eval_files = list(data_path.glob("eval_*.jsonl"))
            
            if train_files:
                logger.info(f"Loading local dataset from {train_files}")
                train_dataset = load_dataset('json', data_files=train_files)['train']
            else:
                # Fallback to default dataset
                logger.info("No local dataset found, loading Stack dataset...")
                train_dataset = load_dataset('huggingface/codeparrot', split='train')
                
        else:
            logger.info("Loading default dataset...")
            # Fallback dataset
            train_dataset = load_dataset('huggingface/codeparrot', split='train')
        
        return train_dataset
    
    def preprocess_data(self, examples):
        """Preprocess training examples"""
        texts = []
        
        for instruction, input_text, output_text in zip(
            examples['instruction'],
            examples.get('input', [''] * len(examples['instruction'])),
            examples['output']
        ):
            # Format for instruction following
            if input_text:
                text = f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output_text}"
            else:
                text = f"Instruction: {instruction}\nOutput: {output_text}"
            
            texts.append(text)
        
        # Tokenize
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=2048,
            return_tensors='pt'
        )
        
        # Set labels (same as input for causal LM)
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    def prepare_datasets(self):
        """Prepare training and evaluation datasets"""
        self.load_tokenizer()
        
        # Load dataset
        raw_dataset = self.load_dataset()
        
        # Split dataset
        if len(raw_dataset) > 1000:
            # Use standard split for large datasets
            if 'train' in raw_dataset:
                train_dataset = raw_dataset['train']
                eval_dataset = raw_dataset.get('validation', raw_dataset['train'].select(range(100)))
            else:
                split = raw_dataset.train_test_split(test_size=0.1)
                train_dataset = split['train']
                eval_dataset = split['test']
        else:
            # For small datasets, use all for training
            train_dataset = raw_dataset
            eval_dataset = raw_dataset.select(range(min(100, len(raw_dataset))))
        
        # Preprocess datasets
        logger.info("Preprocessing datasets...")
        train_dataset = train_dataset.map(
            self.preprocess_data,
            batched=True,
            remove_columns=raw_dataset.column_names,
            desc="Preprocessing training data"
        )
        
        eval_dataset = eval_dataset.map(
            self.preprocess_data,
            batched=True,
            remove_columns=raw_dataset.column_names,
            desc="Preprocessing evaluation data"
        )
        
        # Filter out very long sequences
        max_length = 2048
        train_dataset = train_dataset.filter(
            lambda x: len(x['input_ids']) <= max_length,
            desc="Filtering long sequences"
        )
        
        logger.info(f"Training dataset size: {len(train_dataset)}")
        logger.info(f"Evaluation dataset size: {len(eval_dataset)}")
        
        return train_dataset, eval_dataset


class TrainingManager:
    """Main training management class"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.accelerator = None
        
    def setup_accelerate(self):
        """Setup distributed training with Accelerate"""
        logger.info("Setting up distributed training...")
        
        # Initialize accelerator
        self.accelerator = Accelerator(
            mixed_precision='fp16',
            gradient_accumulation_steps=4,
            log_with=['wandb']
        )
        
        logger.info(f"Accelerator initialized: {self.accelerator.device}")
        
    def load_model(self):
        """Load and configure model"""
        logger.info("Loading model...")
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16,
            device_map='auto' if not self.accelerator else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Configure model
        self.model.config.use_cache = False
        self.model.gradient_checkpointing_enable()
        
        # Add LoRA adapters
        peft_model = get_peft_model(self.model, self.config.lora_config)
        peft_model.print_trainable_parameters()
        
        logger.info("Model loaded and configured with LoRA adapters")
        
    def setup_trainer(self, train_dataset, eval_dataset):
        """Setup Trainer with optimization"""
        logger.info("Setting up trainer...")
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            return_tensors="pt"
        )
        
        # Training arguments
        training_args = TrainingArguments(**self.config.training_args)
        
        # Early stopping callback
        callbacks = [
            EarlyStoppingCallback(early_stopping_patience=3)
        ]
        
        # Create trainer
        self.trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            callbacks=callbacks
        )
        
        logger.info("Trainer setup completed")
        
    def train(self):
        """Execute training loop"""
        logger.info("Starting training...")
        start_time = time.time()
        
        try:
            # Start training
            train_result = self.trainer.train()
            
            # Save model
            self.trainer.save_model()
            self.trainer.save_state()
            
            # Calculate training time
            training_time = time.time() - start_time
            logger.info(f"Training completed in {training_time:.2f} seconds")
            
            # Save training metrics
            self.save_training_metrics(train_result, training_time)
            
            # Evaluation
            eval_result = self.trainer.evaluate()
            logger.info(f"Final evaluation result: {eval_result}")
            
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def save_training_metrics(self, train_result, training_time):
        """Save training metrics and metadata"""
        metrics = {
            'run_id': self.config.run_id,
            'timestamp': self.config.timestamp,
            'model_name': self.config.model_name,
            'gpu_type': self.config.gpu_type,
            'training_steps': self.config.training_steps,
            'final_loss': train_result.training_loss,
            'total_training_time': training_time,
            'model_size': self.get_model_size(),
            'training_args': self.config.training_args
        }
        
        # Save metrics
        metrics_path = Path('models/metrics') / f"training_metrics_{self.config.run_id}.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        logger.info(f"Training metrics saved to {metrics_path}")
    
    def get_model_size(self):
        """Calculate model size"""
        try:
            model_size = sum(p.numel() for p in self.model.parameters())
            trainable_size = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            return {
                'total_parameters': model_size,
                'trainable_parameters': trainable_size,
                'trainable_percentage': (trainable_size / model_size) * 100
            }
        except:
            return {'error': 'Could not calculate model size'}


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Automated training for Sheikh-2.5-Coder')
    
    # Required arguments
    parser.add_argument('--model_name', required=True, help='Base model name or path')
    parser.add_argument('--data_path', required=True, help='Path to training data')
    parser.add_argument('--output_path', required=True, help='Output directory for model')
    
    # Training configuration
    parser.add_argument('--gpu_type', default='t4', choices=['a100', 'v100', 't4'], help='GPU type')
    parser.add_argument('--training_steps', default='10000', help='Number of training steps')
    parser.add_argument('--batch_size', default='4', help='Batch size')
    parser.add_argument('--learning_rate', default='2e-5', help='Learning rate')
    
    # Run metadata
    parser.add_argument('--run_id', required=True, help='Unique run identifier')
    parser.add_argument('--timestamp', required=True, help='Run timestamp')
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = ModelConfig(args)
    
    # Initialize training manager
    training_manager = TrainingManager(config)
    
    try:
        # Setup training environment
        training_manager.setup_accelerate()
        
        # Load and prepare data
        data_processor = DataProcessor(config)
        train_dataset, eval_dataset = data_processor.prepare_datasets()
        
        # Load model
        training_manager.tokenizer = data_processor.tokenizer
        training_manager.load_model()
        
        # Setup trainer
        training_manager.setup_trainer(train_dataset, eval_dataset)
        
        # Start training
        success = training_manager.train()
        
        if success:
            logger.info("Training completed successfully!")
            return 0
        else:
            logger.error("Training failed!")
            return 1
            
    except Exception as e:
        logger.error(f"Training script failed: {str(e)}")
        logger.error(traceback.format_exc())
        return 1
    
    finally:
        # Cleanup
        if training_manager.accelerator:
            training_manager.accelerator.end_training()


if __name__ == '__main__':
    sys.exit(main())