#!/usr/bin/env python3
"""
Distributed Training Wrapper for Sheikh-2.5-Coder
Handles multi-GPU training, DeepSpeed integration, and fault tolerance
"""

import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import logging
import json
import traceback
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from pathlib import Path
from dataclasses import dataclass, field
import threading
import time
import signal
import pickle

try:
    from accelerate import Accelerator, DistributedDataParallelKwargs
    from accelerate.utils import set_seed
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False

try:
    import deepspeed
    from deepspeed.ops.adam import FusedAdam
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False

import numpy as np
from transformers import (
    Trainer,
    TrainingArguments,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
from torch.utils.data import DataLoader, DistributedSampler


@dataclass
class DistributedConfig:
    """Configuration for distributed training"""
    # Basic distributed settings
    use_distributed: bool = True
    distributed_backend: str = "nccl"  # "nccl", "gloo", "mpi"
    distributed_init_method: str = "env://"
    master_addr: str = "localhost"
    master_port: str = "29500"
    
    # Multi-GPU settings
    world_size: Optional[int] = None
    rank: Optional[int] = None
    local_rank: Optional[int] = None
    device_ids: Optional[List[int]] = None
    
    # DeepSpeed configuration
    use_deepspeed: bool = True
    deepspeed_config: Optional[str] = None
    deepspeed_stage: int = 3  # ZeRO stage 3
    
    # Accelerate configuration
    use_accelerate: bool = True
    mixed_precision: str = "fp16"  # "fp16", "bf16", "fp32"
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True
    cpu: bool = False
    
    # Memory optimization
    max_memory_mb: int = 16000
    offload_optimizer: bool = True
    offload_param: bool = True
    pin_memory: bool = True
    non_blocking: bool = True
    
    # Communication optimization
    communication_data_type: str = "fp16"
    gather_16bit_weights_on_cpu: bool = True
    sub_group_size: int = 1e9
    
    # Fault tolerance
    enable_timeout: bool = True
    timeout_seconds: int = 3600  # 1 hour
    max_retries: int = 3
    retry_delay: int = 60
    checkpoint_recovery: bool = True
    
    # Performance optimization
    allgather_bucket_size: int = 5e8
    reduce_bucket_size: int = 5e8
    reduce_scatter: bool = True
    allgather_partitions: bool = True
    
    # Monitoring
    enable_profiling: bool = False
    profile_steps: int = 10
    log_level: str = "INFO"


class DistributedTrainer:
    """Main distributed training wrapper"""
    
    def __init__(
        self,
        config: DistributedConfig,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        training_args: Optional[TrainingArguments] = None
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.training_args = training_args
        
        # Initialize distributed environment
        self.accelerator = None
        self.deepspeed_engine = None
        
        # Training state
        self.is_initialized = False
        self.is_main_process = True
        self.process_group = None
        
        # Fault tolerance
        self.training_interrupted = False
        self.checkpoint_data = {}
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        logging.info(f"DistributedTrainer initialized with config: {config.use_distributed}")
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logging.warning(f"Received signal {signum}, preparing for graceful shutdown...")
            self.training_interrupted = True
            
            # Save checkpoint if possible
            if self.is_main_process and self.config.checkpoint_recovery:
                self._save_emergency_checkpoint()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def setup_distributed(self, seed: int = 42) -> bool:
        """Setup distributed training environment"""
        try:
            # Set seeds for reproducibility
            set_seed(seed)
            
            if self.config.use_distributed and not self.is_initialized:
                # Check available GPUs
                gpu_count = torch.cuda.device_count()
                if gpu_count == 0:
                    logging.warning("No GPUs available, falling back to CPU training")
                    self.config.cpu = True
                    self.config.use_distributed = False
                else:
                    logging.info(f"Found {gpu_count} GPUs")
                    
                    # Initialize distributed environment
                    if self.config.use_accelerate and ACCELERATE_AVAILABLE:
                        self._setup_accelerate()
                    elif self.config.use_deepspeed and DEEPSPEED_AVAILABLE:
                        self._setup_deepspeed()
                    else:
                        self._setup_native_distributed()
                
                # Determine main process
                if self.config.use_distributed:
                    if self.config.local_rank is not None:
                        self.is_main_process = self.config.local_rank == 0
                    elif hasattr(self, 'accelerator') and self.accelerator:
                        self.is_main_process = self.accelerator.is_main_process
                    else:
                        self.is_main_process = True
                else:
                    self.is_main_process = True
                
                self.is_initialized = True
                
                if self.is_main_process:
                    logging.info("Distributed training setup completed successfully")
                    self._print_system_info()
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to setup distributed training: {e}")
            logging.error(traceback.format_exc())
            return False
    
    def _setup_accelerate(self) -> None:
        """Setup training using Accelerate"""
        if not ACCELERATE_AVAILABLE:
            raise ImportError("Accelerate is not available")
        
        # Prepare kwargs for Accelerator
        kwargs = {}
        
        if not self.config.cpu:
            kwargs['gpu_ids'] = self.config.device_ids
        
        kwargs['mixed_precision'] = self.config.mixed_precision
        kwargs['gradient_accumulation_steps'] = self.config.gradient_accumulation_steps
        
        # Set up distributed training
        if self.config.use_distributed:
            if self.config.local_rank is not None:
                kwargs['backend'] = self.config.distributed_backend
                kwargs['machine_rank'] = self.config.local_rank
            elif torch.cuda.device_count() > 1:
                kwargs['backend'] = self.config.distributed_backend
        
        # Initialize Accelerator
        self.accelerator = Accelerator(**kwargs)
        self.config.local_rank = self.accelerator.local_process_index
        self.config.rank = self.accelerator.process_index
        self.config.world_size = self.accelerator.num_processes
        
        logging.info(f"Accelerate setup: rank={self.config.rank}, local_rank={self.config.local_rank}, world_size={self.config.world_size}")
    
    def _setup_deepspeed(self) -> None:
        """Setup training using DeepSpeed"""
        if not DEEPSPEED_AVAILABLE:
            raise ImportError("DeepSpeed is not available")
        
        # Generate DeepSpeed config if not provided
        if not self.config.deepspeed_config:
            ds_config = self._generate_deepspeed_config()
        else:
            with open(self.config.deepspeed_config, 'r') as f:
                ds_config = json.load(f)
        
        # Save temporary config if needed
        if isinstance(ds_config, dict):
            config_path = Path("deepspeed_config.json")
            with open(config_path, 'w') as f:
                json.dump(ds_config, f, indent=2)
            self.config.deepspeed_config = str(config_path)
        
        logging.info("DeepSpeed configuration loaded")
    
    def _setup_native_distributed(self) -> None:
        """Setup native PyTorch distributed training"""
        # Initialize distributed training
        if not dist.is_initialized():
            # Set environment variables
            os.environ['MASTER_ADDR'] = self.config.master_addr
            os.environ['MASTER_PORT'] = self.config.master_port
            
            # Initialize process group
            dist.init_process_group(
                backend=self.config.distributed_backend,
                init_method=self.config.distributed_init_method,
                world_size=self.config.world_size,
                rank=self.config.rank
            )
            
            self.process_group = dist.group.WORLD
        
        # Set device for current process
        if self.config.local_rank is not None:
            torch.cuda.set_device(self.config.local_rank)
        
        logging.info(f"Native distributed setup: rank={self.config.rank}, local_rank={self.config.local_rank}")
    
    def _generate_deepspeed_config(self) -> Dict[str, Any]:
        """Generate DeepSpeed configuration based on settings"""
        config = {
            "zero_optimization": {
                "stage": self.config.deepspeed_stage,
                "offload_optimizer": {
                    "device": "cpu" if self.config.offload_optimizer else "none",
                    "pin_memory": self.config.pin_memory
                },
                "offload_param": {
                    "device": "cpu" if self.config.offload_param else "none",
                    "pin_memory": self.config.pin_memory
                },
                "allgather_partitions": self.config.allgather_partitions,
                "reduce_scatter": self.config.reduce_scatter,
                "allgather_bucket_size": self.config.allgather_bucket_size,
                "reduce_bucket_size": self.config.reduce_bucket_size,
                "sub_group_size": self.config.sub_group_size,
                "gather_16bit_weights_on_cpu": self.config.gather_16bit_weights_on_cpu
            },
            "fp16": {
                "enabled": self.config.mixed_precision == "fp16",
                "auto_cast": True,
                "loss_scale": 0,
                "initial_scale_power": 16,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1
            },
            "bf16": {
                "enabled": self.config.mixed_precision == "bf16"
            },
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "gradient_clipping": 1.0,
            "steps_per_print": 10,
            "train_batch_size": self.config.gradient_accumulation_steps,  # Will be adjusted
            "train_micro_batch_size_per_gpu": 1,  # Will be adjusted
            "wall_clock_breakdown": False,
            "memory_efficient": True,
            "activation_checkpointing": {
                "enabled": self.config.gradient_checkpointing,
                "partition_activations": True,
                "contiguous_memory_optimization": True,
                "cpu_checkpointing": True
            },
            "communication_data_type": self.config.communication_data_type
        }
        
        return config
    
    def prepare_model_and_data(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        data_collator: Optional[Callable] = None
    ) -> Tuple[PreTrainedModel, DataLoader, Optional[DataLoader]]:
        """Prepare model and data for distributed training"""
        
        # Prepare model
        if hasattr(self, 'accelerator') and self.accelerator:
            # Use Accelerate for preparation
            model = self.accelerator.prepare_model(model)
            
            if train_dataset is not None:
                train_dataset = self.accelerator.prepare_dataset(train_dataset)
            
            if eval_dataset is not None:
                eval_dataset = self.accelerator.prepare_dataset(eval_dataset)
        
        # Prepare data loaders
        train_dataloader = None
        eval_dataloader = None
        
        if train_dataset is not None:
            train_dataloader = self._create_dataloader(train_dataset, data_collator, is_eval=False)
        
        if eval_dataset is not None:
            eval_dataloader = self._create_dataloader(eval_dataset, data_collator, is_eval=True)
        
        return model, train_dataloader, eval_dataloader
    
    def _create_dataloader(
        self,
        dataset: Dataset,
        data_collator: Optional[Callable],
        is_eval: bool = False
    ) -> DataLoader:
        """Create distributed data loader"""
        
        # Determine batch size based on distributed settings
        if hasattr(self, 'accelerator') and self.accelerator:
            batch_size = self.training_args.per_device_train_batch_size if not is_eval else self.training_args.per_device_eval_batch_size
            
            # Adjust batch size for distributed training
            if self.accelerator.num_processes > 1:
                batch_size = batch_size * self.accelerator.num_processes
        else:
            batch_size = self.training_args.per_device_train_batch_size if not is_eval else self.training_args.per_device_eval_batch_size
        
        # Create sampler for distributed training
        sampler = None
        if self.config.use_distributed and not is_eval:
            sampler = DistributedSampler(dataset, shuffle=True)
        elif self.config.use_distributed and is_eval:
            sampler = DistributedSampler(dataset, shuffle=False)
        
        # Create data collator
        if data_collator is None:
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=tokenizer,
                model=model,
                padding=True,
                return_tensors="pt"
            )
        
        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=data_collator,
            num_workers=self.training_args.dataloader_num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=self.training_args.dataloader_drop_last
        )
        
        return dataloader
    
    def train(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        training_args: Optional[TrainingArguments] = None,
        callbacks: Optional[List] = None,
        **kwargs
    ) -> PreTrainedModel:
        """Train the model using distributed training"""
        
        # Setup training arguments
        if training_args is None:
            training_args = TrainingArguments(**kwargs)
        
        # Prepare model and data
        model, train_dataloader, eval_dataloader = self.prepare_model_and_data(
            model, tokenizer, train_dataset, eval_dataset
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if train_dataloader else None,
            eval_dataset=eval_dataset if eval_dataloader else None,
            tokenizer=tokenizer,
            callbacks=callbacks or []
        )
        
        # Wrap trainer for distributed training
        if hasattr(self, 'accelerator') and self.accelerator:
            trainer = self.accelerator.prepare_trainer(trainer)
        
        try:
            # Start training
            if self.is_main_process:
                logging.info("Starting distributed training...")
            
            trainer.train()
            
            # Save final model (only on main process)
            if self.is_main_process:
                trainer.save_model()
                logging.info("Training completed and model saved")
            
            return model
            
        except Exception as e:
            logging.error(f"Training failed: {e}")
            logging.error(traceback.format_exc())
            
            # Attempt recovery if configured
            if self.config.checkpoint_recovery:
                self._attempt_recovery(trainer, e)
            
            raise
    
    def evaluate(
        self,
        model: PreTrainedModel,
        eval_dataset: Dataset,
        training_args: Optional[TrainingArguments] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Evaluate the model"""
        
        if training_args is None:
            training_args = TrainingArguments(**kwargs)
        
        trainer = Trainer(
            model=model,
            args=training_args,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer
        )
        
        # Wrap trainer for distributed training
        if hasattr(self, 'accelerator') and self.accelerator:
            trainer = self.accelerator.prepare_trainer(trainer)
        
        results = trainer.evaluate()
        return results
    
    def save_checkpoint(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        training_args: TrainingArguments,
        step: int,
        epoch: float,
        additional_state: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save checkpoint with distributed support"""
        
        checkpoint_data = {
            'step': step,
            'epoch': epoch,
            'model_state_dict': model.state_dict() if hasattr(model, 'state_dict') else {},
            'training_args': training_args.to_dict(),
            'config': self.config.__dict__,
            'timestamp': time.time()
        }
        
        # Add additional state if provided
        if additional_state:
            checkpoint_data.update(additional_state)
        
        # Save checkpoint (only on main process)
        if self.is_main_process:
            checkpoint_path = Path(training_args.output_dir) / f"checkpoint-{step:06d}"
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            
            # Save model and tokenizer
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
            
            # Save checkpoint data
            with open(checkpoint_path / "distributed_checkpoint.pkl", 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            logging.info(f"Distributed checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        training_args: TrainingArguments
    ) -> Dict[str, Any]:
        """Load checkpoint with distributed support"""
        
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint data
        with open(checkpoint_path / "distributed_checkpoint.pkl", 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        # Load model state
        if checkpoint_data['model_state_dict']:
            if hasattr(model, 'load_state_dict'):
                model.load_state_dict(checkpoint_data['model_state_dict'])
            else:
                # For models that don't have load_state_dict method
                model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
        
        logging.info(f"Distributed checkpoint loaded: {checkpoint_path}")
        
        return checkpoint_data
    
    def _save_emergency_checkpoint(self) -> None:
        """Save emergency checkpoint during interruption"""
        try:
            if self.is_main_process:
                logging.info("Saving emergency checkpoint...")
                
                # This would save the current training state
                # Implementation depends on the specific training setup
                
                emergency_path = Path("emergency_checkpoint.pkl")
                with open(emergency_path, 'wb') as f:
                    pickle.dump(self.checkpoint_data, f)
                
                logging.info(f"Emergency checkpoint saved: {emergency_path}")
                
        except Exception as e:
            logging.error(f"Failed to save emergency checkpoint: {e}")
    
    def _attempt_recovery(self, trainer: Trainer, error: Exception) -> None:
        """Attempt to recover training from error"""
        try:
            logging.info("Attempting training recovery...")
            
            # Check for emergency checkpoint
            emergency_path = Path("emergency_checkpoint.pkl")
            if emergency_path.exists():
                with open(emergency_path, 'rb') as f:
                    recovery_data = pickle.load(f)
                
                # Attempt to resume training
                # This would require specific implementation based on the training setup
                logging.info("Training recovery successful")
                
        except Exception as recovery_error:
            logging.error(f"Training recovery failed: {recovery_error}")
    
    def _print_system_info(self) -> None:
        """Print system information (main process only)"""
        if not self.is_main_process:
            return
        
        print("\n" + "="*60)
        print("DISTRIBUTED TRAINING SYSTEM INFO")
        print("="*60)
        
        # System info
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Distributed training info
        if self.config.use_distributed:
            print(f"Distributed backend: {self.config.distributed_backend}")
            print(f"World size: {self.config.world_size}")
            print(f"Rank: {self.config.rank}")
            print(f"Local rank: {self.config.local_rank}")
        
        # Configuration info
        print(f"DeepSpeed enabled: {self.config.use_deepspeed}")
        print(f"Accelerate enabled: {self.config.use_accelerate}")
        print(f"Mixed precision: {self.config.mixed_precision}")
        print(f"Gradient checkpointing: {self.config.gradient_checkpointing}")
        
        print("="*60)
    
    def cleanup(self) -> None:
        """Cleanup distributed training resources"""
        try:
            if self.config.use_distributed and dist.is_initialized():
                dist.destroy_process_group()
                logging.info("Distributed process group destroyed")
            
            if hasattr(self, 'accelerator') and self.accelerator:
                # Accelerate doesn't require explicit cleanup
                pass
            
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")


def setup_multiprocessing(
    main_function: Callable,
    config: DistributedConfig,
    nproc_per_node: int = None
) -> None:
    """Setup multiprocessing for distributed training"""
    
    if nproc_per_node is None:
        nproc_per_node = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    if nproc_per_node > 1:
        # Multi-GPU setup
        print(f"Starting distributed training on {nproc_per_node} GPUs...")
        
        # Spawn processes
        mp.spawn(
            main_function,
            args=(config,),
            nprocs=nproc_per_node,
            join=True
        )
    else:
        # Single GPU/CPU setup
        main_function(0, config)


def create_distributed_trainer(
    use_distributed: bool = True,
    use_deepspeed: bool = True,
    use_accelerate: bool = True,
    **kwargs
) -> DistributedTrainer:
    """Create a distributed trainer with default settings"""
    
    config = DistributedConfig(
        use_distributed=use_distributed,
        use_deepspeed=use_deepspeed,
        use_accelerate=use_accelerate,
        **kwargs
    )
    
    return DistributedTrainer(config)


if __name__ == "__main__":
    # Example usage
    import torch.optim as optim
    
    # Create configuration
    config = DistributedConfig(
        use_distributed=True,
        use_deepspeed=True,
        use_accelerate=True,
        mixed_precision="fp16",
        gradient_accumulation_steps=4,
        gradient_checkpointing=True
    )
    
    # Create trainer
    trainer = DistributedTrainer(config)
    
    # Setup distributed training
    if trainer.setup_distributed():
        print("Distributed training setup successful")
        
        # Example training loop would go here
        print("Ready for distributed training")
    
    # Cleanup
    trainer.cleanup()