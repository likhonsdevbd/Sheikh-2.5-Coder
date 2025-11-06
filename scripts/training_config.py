#!/usr/bin/env python3
"""
Training Configuration Management for Sheikh-2.5-Coder
Handles all training parameters, hyperparameters, and environment settings
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model architecture and parameters"""
    base_model: str = "microsoft/phi-2"
    target_parameters: int = 400_000_000  # +400M parameters for XML/MDX/JavaScript
    context_length: int = 32768
    vocab_size: int = 51200
    hidden_size: int = 2048
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    intermediate_size: int = 8192
    max_position_embeddings: int = 32768
    layer_norm_epsilon: float = 1e-5
    use_cache: bool = True
    use_flash_attention: bool = False
    gradient_checkpointing: bool = True
    use_projection_bias: bool = True
    use_projection_head: bool = True
    # Special tokens for XML/MDX/JavaScript
    additional_special_tokens: list = field(default_factory=lambda: [
        "<xml>", "</xml>", "<mdx>", "</mdx>", 
        "<component>", "</component>", "<script>", "</script>",
        "<style>", "</style>", "<template>", "</template>"
    ])


@dataclass
class TrainingConfig:
    """Configuration for training parameters"""
    # Basic training parameters
    num_epochs: int = 3
    batch_size: int = 8
    eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Learning rate scheduling
    warmup_steps: int = 1000
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    lr_scheduler_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "num_cycles": 0.5,
        "lr_end": 1e-7
    })
    
    # Training optimization
    mixed_precision: str = "fp16"  # fp16, bf16, fp32
    fp16_opt_level: str = "O1"
    fp16_loss_scale: float = 0.0
    fp16_full_eval: bool = False
    dataloader_pin_memory: bool = True
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False
    group_by_length: bool = True
    
    # Memory optimization
    gradient_checkpointing: bool = True
    max_memory_MB: int = 16000
    low_cpu_mem_usage: bool = True
    use_cache: bool = False  # Disable for training to save memory
    
    # Training data
    max_seq_length: int = 32768
    truncation_side: str = "right"
    padding_side: str = "right"
    
    # Logging and saving
    logging_steps: int = 100
    save_steps: int = 1000
    eval_steps: int = 500
    evaluation_strategy: str = "steps"
    save_strategy: str = "steps"
    logging_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # DeepSpeed configuration
    use_deepspeed: bool = True
    deepspeed_config_path: Optional[str] = None


@dataclass
class DataConfig:
    """Configuration for data processing and loading"""
    # Data sources
    data_dir: str = "data/processed"
    train_data_files: list = field(default_factory=lambda: [
        "data/processed/code_samples.jsonl",
        "data/processed/xml_samples.jsonl", 
        "data/processed/mdx_samples.jsonl",
        "data/processed/javascript_samples.jsonl"
    ])
    eval_data_files: list = field(default_factory=lambda: [
        "data/processed/eval_code_samples.jsonl",
        "data/processed/eval_xml_samples.jsonl"
    ])
    
    # Data preprocessing
    tokenizer_name: str = "microsoft/phi-2"
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    preprocessing_num_workers: int = 4
    preprocessing_batch_size: int = 1000
    
    # Data filtering
    min_text_length: int = 10
    max_text_length: int = 32768
    filter_duplicate_sequences: bool = True
    filter_out_all_special_tokens: bool = False
    remove_empty_sequences: bool = True
    
    # Data augmentation
    data_augmentation: bool = False
    augmentation_probability: float = 0.1
    augmentation_types: list = field(default_factory=lambda: [
        "mask_code_tokens", "shuffle_operations", "comment_variations"
    ])


@dataclass
class MonitoringConfig:
    """Configuration for monitoring and logging"""
    # Weights & Biases
    use_wandb: bool = True
    wandb_project: str = "sheikh-2.5-coder"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: list = field(default_factory=lambda: ["training", "phi-2", "code-generation"])
    wandb_notes: Optional[str] = None
    wandb_config: Dict[str, Any] = field(default_factory=dict)
    
    # TensorBoard
    use_tensorboard: bool = True
    tensorboard_log_dir: str = "logs/tensorboard"
    tensorboard_flush_secs: int = 30
    
    # Local logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: str = "logs/training.log"
    log_max_bytes: int = 10 * 1024 * 1024  # 10MB
    log_backup_count: int = 5
    
    # Metrics tracking
    track_metrics: list = field(default_factory=lambda: [
        "loss", "perplexity", "accuracy", "throughput", "learning_rate",
        "grad_norm", "memory_usage", "gpu_utilization"
    ])
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckpointConfig:
    """Configuration for checkpoint management"""
    checkpoint_dir: str = "models/checkpoints"
    output_dir: str = "models/output"
    
    # Checkpoint frequency
    save_steps: int = 1000
    save_total_limit: int = 10
    save_safetensors: bool = True
    
    # Checkpoint strategy
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Resume training
    resume_from_checkpoint: Optional[str] = None
    ignore_data_skip: bool = False
    
    # Checkpoint compression and versioning
    compress_checkpoints: bool = True
    checkpoint_compression_type: str = "gzip"  # gzip, bz2, xz
    checkpoint_versioning: bool = True
    checkpoint_timestamp_format: str = "%Y%m%d_%H%M%S"
    
    # Cloud storage integration
    use_cloud_storage: bool = False
    cloud_storage_provider: str = "s3"  # s3, gcs, azure
    cloud_bucket: Optional[str] = None
    cloud_prefix: str = "checkpoints/sheikh-2.5-coder/"
    
    # Checkpoint validation
    validate_checkpoints: bool = True
    checkpoint_validation_interval: int = 5000


@dataclass
class SystemConfig:
    """Configuration for system resources and environment"""
    # GPU configuration
    gpu_ids: str = "auto"  # auto or comma-separated list
    gpu_memory_fraction: float = 0.8
    gpu_memory_reserved: int = 1000  # MB
    multi_gpu: bool = True
    device_map: Optional[Dict[str, str]] = None
    
    # CPU configuration
    cpu_threads: int = -1  # -1 for all available
    cpu_pin_memory: bool = True
    
    # Memory configuration
    max_memory_gb: int = 32
    memory_fraction: float = 0.8
    garbage_collection: bool = True
    
    # Distributed training
    distributed_backend: str = "nccl"  # nccl, gloo, mpi
    distributed_init_method: str = "env://"
    distributed_rank: int = 0
    distributed_world_size: int = 1
    
    # Fault tolerance
    max_retries: int = 3
    retry_delay: int = 60  # seconds
    checkpoint_recovery: bool = True
    
    # Environment
    environment: str = "development"  # development, staging, production
    debug_mode: bool = False
    seed: int = 42


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation and testing"""
    eval_datasets: list = field(default_factory=lambda: [
        "HumanEval",
        "MMLU", 
        "codex-eval",
        "mbpp",
        "truthfulqa",
        "hellaswag"
    ])
    
    eval_metrics: list = field(default_factory=lambda: [
        "perplexity",
        "bleu",
        "rouge",
        "code_bleu",
        "exact_match",
        "pass@k"
    ])
    
    # Code generation evaluation
    code_eval_timeout: int = 30  # seconds
    code_eval_num_samples: int = 1000
    code_eval_temperature: float = 0.2
    code_eval_top_p: float = 0.95
    code_eval_max_new_tokens: int = 512
    
    # XML/MDX evaluation
    xml_eval_samples: int = 500
    xml_eval_metrics: list = field(default_factory=lambda: [
        "xml_validity",
        "html_validity", 
        "structure_score",
        "content_relevance"
    ])
    
    # JavaScript evaluation
    js_eval_samples: int = 500
    js_eval_metrics: list = field(default_factory=lambda: [
        "syntax_validity",
        "test_pass_rate",
        "complexity_score",
        "performance_score"
    ])


class TrainingConfiguration:
    """Main configuration class that combines all training configurations"""
    
    def __init__(self, config_file: Optional[str] = None, **kwargs):
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.data = DataConfig()
        self.monitoring = MonitoringConfig()
        self.checkpoint = CheckpointConfig()
        self.system = SystemConfig()
        self.evaluation = EvaluationConfig()
        
        # Load configuration from file if provided
        if config_file:
            self.load_from_file(config_file)
        
        # Override with keyword arguments
        self._update_from_kwargs(kwargs)
    
    def _update_from_kwargs(self, kwargs: Dict[str, Any]):
        """Update configuration from keyword arguments"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                config_section = getattr(self, key)
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if hasattr(config_section, subkey):
                            setattr(config_section, subkey, subvalue)
                        else:
                            logger.warning(f"Unknown configuration key: {key}.{subkey}")
                else:
                    setattr(self, key, value)
            else:
                logger.warning(f"Unknown configuration section: {key}")
    
    def load_from_file(self, config_file: str) -> None:
        """Load configuration from YAML or JSON file"""
        config_path = Path(config_file)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config_data = yaml.safe_load(f)
            else:
                config_data = json.load(f)
        
        # Apply configuration to all sections
        for section_name, section_data in config_data.items():
            if hasattr(self, section_name):
                section_config = getattr(self, section_name)
                if isinstance(section_data, dict):
                    for key, value in section_data.items():
                        if hasattr(section_config, key):
                            setattr(section_config, key, value)
                        else:
                            logger.warning(f"Unknown configuration key: {section_name}.{key}")
    
    def save_to_file(self, config_file: str) -> None:
        """Save current configuration to YAML or JSON file"""
        config_data = {}
        
        for attr_name in dir(self):
            if not attr_name.startswith('_') and attr_name != 'load_from_file' and attr_name != 'save_to_file':
                config_obj = getattr(self, attr_name)
                if hasattr(config_obj, '__dict__'):
                    config_data[attr_name] = {}
                    for subattr_name, subattr_value in config_obj.__dict__.items():
                        if not subattr_name.startswith('_'):
                            config_data[attr_name][subattr_name] = subattr_value
        
        config_path = Path(config_file)
        
        with open(config_path, 'w') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            else:
                json.dump(config_data, f, indent=2)
    
    def get_deepspeed_config(self) -> Dict[str, Any]:
        """Generate DeepSpeed configuration based on training settings"""
        if not self.training.use_deepspeed:
            return {}
        
        config = {
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "allgather_partitions": True,
                "reduce_scatter": True,
                "allgather_bucket_size": 5e8,
                "reduce_bucket_size": 5e8,
                "sub_group_size": 1e9,
                "gather_16bit_weights_on_fp32_gpu": True
            },
            "fp16": {
                "enabled": self.training.mixed_precision == "fp16",
                "auto_cast": True,
                "loss_scale": self.training.fp16_loss_scale,
                "initial_scale_power": 16,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1
            },
            "gradient_accumulation_steps": self.training.gradient_accumulation_steps,
            "gradient_clipping": self.training.max_grad_norm,
            "steps_per_print": 10,
            "train_batch_size": self.training.batch_size,
            "train_micro_batch_size_per_gpu": self.training.batch_size,
            "wall_clock_breakdown": False,
            "memory_efficient": True,
            "activation_checkpointing": self.training.gradient_checkpointing,
            "bf16": {
                "enabled": self.training.mixed_precision == "bf16"
            }
        }
        
        # Add system-specific optimizations
        config.update({
            "communication_data_type": self.training.mixed_precision,
            "enable_gather_16bit_weights_on_cpu": True,
            "offload_param": {
                "device": "cpu",
                "pin_memory": True,
                "max_reuse_distance": 0,
                "gather_into_optim": True,
                "release_after_init": False
            },
            "gather_all_tokens": {
                "enabled": False
            }
        })
        
        return config
    
    def validate(self) -> bool:
        """Validate configuration settings"""
        errors = []
        
        # Validate training parameters
        if self.training.learning_rate <= 0:
            errors.append("Learning rate must be positive")
        
        if self.training.batch_size <= 0:
            errors.append("Batch size must be positive")
        
        if self.training.num_epochs <= 0:
            errors.append("Number of epochs must be positive")
        
        if self.training.warmup_steps < 0:
            errors.append("Warmup steps must be non-negative")
        
        # Validate memory settings
        if self.system.max_memory_gb <= 0:
            errors.append("Max memory must be positive")
        
        if self.system.gpu_memory_fraction <= 0 or self.system.gpu_memory_fraction > 1:
            errors.append("GPU memory fraction must be between 0 and 1")
        
        # Validate checkpoint settings
        if self.checkpoint.save_steps <= 0:
            errors.append("Save steps must be positive")
        
        # Validate data paths
        data_dir = Path(self.data.data_dir)
        if not data_dir.exists():
            errors.append(f"Data directory does not exist: {self.data.data_dir}")
        
        if errors:
            logger.error("Configuration validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
            return False
        
        logger.info("Configuration validation passed")
        return True
    
    def get_training_arguments(self) -> Dict[str, Any]:
        """Generate HuggingFace TrainingArguments from configuration"""
        return {
            "output_dir": self.checkpoint.output_dir,
            "num_train_epochs": self.training.num_epochs,
            "per_device_train_batch_size": self.training.batch_size,
            "per_device_eval_batch_size": self.training.eval_batch_size,
            "gradient_accumulation_steps": self.training.gradient_accumulation_steps,
            "learning_rate": self.training.learning_rate,
            "weight_decay": self.training.weight_decay,
            "adam_beta1": self.training.adam_beta1,
            "adam_beta2": self.training.adam_beta2,
            "adam_epsilon": self.training.adam_epsilon,
            "max_grad_norm": self.training.max_grad_norm,
            "warmup_steps": self.training.warmup_steps,
            "warmup_ratio": self.training.warmup_ratio,
            "lr_scheduler_type": self.training.lr_scheduler_type,
            "lr_scheduler_kwargs": self.training.lr_scheduler_kwargs,
            "fp16": self.training.mixed_precision == "fp16",
            "bf16": self.training.mixed_precision == "bf16",
            "fp16_opt_level": self.training.fp16_opt_level,
            "fp16_full_eval": self.training.fp16_full_eval,
            "dataloader_pin_memory": self.training.dataloader_pin_memory,
            "dataloader_num_workers": self.training.dataloader_num_workers,
            "remove_unused_columns": self.training.remove_unused_columns,
            "group_by_length": self.training.group_by_length,
            "report_to": ["wandb"] if self.monitoring.use_wandb else [],
            "logging_steps": self.training.logging_steps,
            "save_steps": self.checkpoint.save_steps,
            "eval_steps": self.training.eval_steps,
            "evaluation_strategy": self.training.evaluation_strategy,
            "save_strategy": self.training.save_strategy,
            "logging_strategy": self.training.logging_strategy,
            "load_best_model_at_end": self.checkpoint.load_best_model_at_end,
            "metric_for_best_model": self.checkpoint.metric_for_best_model,
            "greater_is_better": self.checkpoint.greater_is_better,
            "do_train": True,
            "do_eval": True,
            "do_predict": False,
            "disable_tqdm": False,
            "skip_memory_metrics": False,
            "no_cuda": False,
            "local_rank": -1,
            "deepspeed": self.get_deepspeed_config() if self.training.use_deepspeed else None,
            "dataloader_drop_last": True,
            "optim": "adamw_bnb_8bit",
            "optim_args": None,
            "save_total_limit": self.checkpoint.save_total_limit,
            "save_safetensors": self.checkpoint.save_safetensors,
            "resume_from_checkpoint": self.checkpoint.resume_from_checkpoint,
            "ignore_data_skip": self.checkpoint.ignore_data_skip,
            "max_steps": -1,
            "max_train_samples": self.data.max_train_samples,
            "max_eval_samples": self.data.max_eval_samples,
            "seed": self.system.seed,
            "data_seed": self.system.seed,
            "jit_mode_eval": False,
            "use_ipex": False,
            "torch_compile": False,
            "torch_compile_backend": "inductor",
            "torch_compile_mode": "default",
            "include_inputs_for_metrics": False,
            "label_names": ["labels"],
            "label_smoothing_factor": 0.0,
            "eval_delay": 0,
            "greater_is_better": False,
            "half_precision_backend": "auto",
            "hub_always_push": False,
            "ignore_skip_key": False,
            "include_for_metrics": [],
            "metric_for_best_model": "eval_loss",
            "neftune_noise_alpha": None,
            "prediction_loss_only": False,
            "run_name": self.monitoring.wandb_run_name or None,
            "trust_remote_code": False,
        }
    
    def __repr__(self) -> str:
        """String representation of the configuration"""
        return f"TrainingConfiguration(model={self.model.base_model}, epochs={self.training.num_epochs}, batch_size={self.training.batch_size}, lr={self.training.learning_rate})"


def create_default_config() -> TrainingConfiguration:
    """Create a default configuration for Sheikh-2.5-Coder training"""
    config = TrainingConfiguration()
    
    # Set Sheikh-2.5-Coder specific defaults
    config.model.base_model = "microsoft/phi-2"
    config.model.context_length = 32768
    config.model.max_position_embeddings = 32768
    
    # Training optimization for 400M parameter expansion
    config.training.learning_rate = 1e-4
    config.training.warmup_steps = 1000
    config.training.gradient_accumulation_steps = 4
    config.training.mixed_precision = "fp16"
    config.training.gradient_checkpointing = True
    
    # Data configuration
    config.data.train_data_files = [
        "data/processed/code_samples.jsonl",
        "data/processed/xml_samples.jsonl",
        "data/processed/mdx_samples.jsonl", 
        "data/processed/javascript_samples.jsonl"
    ]
    
    # Monitoring configuration
    config.monitoring.use_wandb = True
    config.monitoring.wandb_project = "sheikh-2.5-coder"
    
    # Checkpoint configuration
    config.checkpoint.save_steps = 1000
    config.checkpoint.save_total_limit = 10
    
    return config


def load_config(config_path: Union[str, Path]) -> TrainingConfiguration:
    """Load configuration from file"""
    return TrainingConfiguration(config_file=str(config_path))


def save_config(config: TrainingConfiguration, config_path: Union[str, Path]) -> None:
    """Save configuration to file"""
    config.save_to_file(str(config_path))


if __name__ == "__main__":
    # Example usage
    config = create_default_config()
    
    # Save default configuration
    config.save_to_file("training_config.yaml")
    
    # Validate configuration
    if config.validate():
        print("Configuration is valid")
        print(config)
    else:
        print("Configuration validation failed")