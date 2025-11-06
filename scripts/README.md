# Sheikh-2.5-Coder Training Infrastructure

Comprehensive training infrastructure for the Sheikh-2.5-Coder model, built on microsoft/phi-2 with specialized capabilities for XML/MDX/JavaScript code generation.

## 🎯 Overview

This training infrastructure provides a complete, production-ready system for training large language models with the following key features:

- **Base Model**: microsoft/phi-2 (2.7B parameters)
- **Target Specialization**: +400M parameters for XML/MDX/JavaScript
- **Training Tokens**: ~500B from multi-source pipeline
- **Context Length**: 32,768 tokens
- **Distributed Training**: Multi-GPU support with DeepSpeed integration
- **Advanced Monitoring**: Weights & Biases + TensorBoard integration
- **Automatic Checkpointing**: With compression and cloud storage support
- **Early Stopping**: Smart early stopping with multiple criteria
- **Comprehensive Validation**: XML/MDX/JavaScript code validation

## 📁 Training Scripts

The infrastructure consists of 7 main components:

### 1. `training_config.py` - Configuration Management
Handles all training parameters, hyperparameters, and environment settings.

```python
from scripts.training_config import create_default_config, load_config

# Create default configuration
config = create_default_config()

# Save/load configuration
config.save_to_file("my_config.yaml")
loaded_config = load_config("my_config.yaml")
```

**Key Features:**
- Hierarchical configuration system with dataclasses
- Support for YAML/JSON configuration files
- DeepSpeed configuration generation
- Comprehensive validation
- Default configurations for Sheikh-2.5-Coder

### 2. `checkpoint_manager.py` - Checkpoint Management
Handles automatic checkpoint saving, loading, compression, and cloud storage.

```python
from scripts.checkpoint_manager import create_checkpoint_manager

checkpoint_manager = create_checkpoint_manager(
    checkpoint_dir="./models/checkpoints",
    output_dir="./models/output",
    compression_type="gzip",
    max_checkpoints=10,
    save_interval=1000
)
```

**Key Features:**
- Automatic checkpoint saving every 1000 steps
- Multiple compression formats (gzip, bz2, xz, tar.gz)
- Cloud storage integration (S3, GCS, Azure)
- Best model selection based on validation metrics
- Resume training from checkpoints
- Checkpoint validation and recovery

### 3. `monitoring_utils.py` - Monitoring and Logging
Real-time monitoring with Weights & Biases and TensorBoard integration.

```python
from scripts.monitoring_utils import create_monitoring_system

monitor = create_monitoring_system(
    use_wandb=True,
    wandb_project="sheikh-2.5-coder",
    wandb_entity="your-entity"
)

monitor.start()
monitor.log_training_step(loss=1.5, learning_rate=1e-4, step=100)
monitor.stop()
```

**Key Features:**
- Real-time metric tracking (loss, perplexity, accuracy, throughput)
- System resource monitoring (CPU, GPU, memory, disk)
- Weights & Biases integration
- TensorBoard logging
- Automatic metric callbacks
- Training summary reports

### 4. `early_stopping.py` - Early Stopping
Advanced early stopping with multiple criteria and recovery strategies.

```python
from scripts.early_stopping import create_early_stopping_callback

early_stopping = create_early_stopping_callback(
    metric_to_monitor="eval_loss",
    metric_mode="min",
    patience=3,
    threshold=0.001
)
```

**Key Features:**
- Patience-based early stopping
- Learning rate plateau detection
- Gradient-based stopping
- Performance degradation detection
- Minimum training time requirements
- Recovery strategies
- Comprehensive state tracking

### 5. `distributed_trainer.py` - Distributed Training
Multi-GPU training wrapper with DeepSpeed and Accelerate integration.

```python
from scripts.distributed_trainer import create_distributed_trainer

trainer = create_distributed_trainer(
    use_distributed=True,
    use_deepspeed=True,
    mixed_precision="fp16",
    gradient_accumulation_steps=4
)

trainer.setup_distributed()
model, train_dataloader, eval_dataloader = trainer.prepare_model_and_data(
    model, tokenizer, train_dataset, eval_dataset
)
```

**Key Features:**
- Multi-GPU support with Data Parallel
- DeepSpeed ZeRO optimization
- Accelerate integration
- Automatic GPU detection and configuration
- Fault tolerance and checkpoint recovery
- Memory optimization (gradient checkpointing, mixed precision)

### 6. `validate_model.py` - Model Validation
Comprehensive model evaluation for code generation quality.

```python
from scripts.validate_model import run_validation

results = run_validation(
    model_path="./models/output",
    validation_datasets=["HumanEval", "mbpp"],
    xml_validation=True,
    js_validation=True
)
```

**Key Features:**
- Standard dataset evaluation (HumanEval, MMLU, MBPP)
- XML/MDX structure validation
- JavaScript syntax and execution testing
- Code quality metrics (CodeBLEU, Pass@k, Perplexity)
- Comprehensive validation reports
- Custom dataset support

### 7. `train_model.py` - Main Training Script
Orchestrates all components for complete training pipeline.

```python
from scripts.train_model import main

# Run training
python scripts/train_model.py --config training_config.yaml --validate_after_training
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Additional dependencies for full functionality
pip install deepspeed tensorboard wandb js2py beautifulsoup4 lxml
```

### 2. Setup Training Configuration

```bash
# Copy example configuration
cp scripts/training_config_example.yaml my_training_config.yaml

# Edit configuration with your settings
vim my_training_config.yaml
```

### 3. Prepare Training Data

```bash
# Ensure your data files are in the specified locations
ls data/processed/
# code_samples.jsonl
# xml_samples.jsonl  
# mdx_samples.jsonl
# javascript_samples.jsonl
# eval_code_samples.jsonl
# eval_xml_samples.jsonl
```

### 4. Run Training

```bash
# Basic training
python scripts/train_model.py --config my_training_config.yaml

# Multi-GPU training
python -m torch.distributed.launch --nproc_per_node=8 scripts/train_model.py --config my_training_config.yaml

# Resume from checkpoint
python scripts/train_model.py --config my_training_config.yaml --resume_from_checkpoint ./models/checkpoints/checkpoint-005000

# Dry run (validate setup)
python scripts/train_model.py --config my_training_config.yaml --dry_run
```

### 5. Monitor Training

```bash
# Start TensorBoard
tensorboard --logdir logs/tensorboard --port 6006

# Check W&B dashboard (if enabled)
open https://wandb.ai/your-entity/sheikh-2.5-coder
```

## ⚙️ Configuration Details

### Training Parameters

```yaml
training:
  num_epochs: 3
  batch_size: 8
  gradient_accumulation_steps: 4
  learning_rate: 1.0e-4
  warmup_steps: 1000
  mixed_precision: "fp16"
  gradient_checkpointing: true
```

### Distributed Training

```yaml
system:
  multi_gpu: true
  distributed_backend: "nccl"
  max_memory_gb: 32
  gpu_memory_fraction: 0.8
```

### Checkpoint Management

```yaml
checkpoint:
  save_steps: 1000
  save_total_limit: 10
  compress_checkpoints: true
  use_cloud_storage: false  # Set to true and configure for cloud backup
```

### Monitoring

```yaml
monitoring:
  use_wandb: true
  wandb_project: "sheikh-2.5-coder"
  use_tensorboard: true
  track_metrics: ["loss", "perplexity", "accuracy", "throughput"]
```

## 📊 Monitoring and Logging

### Weights & Biases Integration

The infrastructure automatically logs:
- Training metrics (loss, learning rate, grad norm)
- Evaluation metrics (eval loss, accuracy, perplexity)
- System metrics (GPU usage, memory, CPU)
- Model artifacts and checkpoints
- Training configuration and hyperparameters

### TensorBoard Logging

Local logging includes:
- Scalar metrics over time
- Histogram distributions
- Model graphs (when available)
- System resource usage
- Training progress visualization

### Custom Metrics

Add custom metrics by registering callbacks:

```python
from scripts.monitoring_utils import create_monitoring_system

def custom_metric_callback(metrics, step, metric_type):
    if metric_type == "training" and "custom_score" in metrics:
        print(f"Custom score at step {step}: {metrics['custom_score']}")

monitor = create_monitoring_system()
monitor.metric_tracker.register_callback('training', custom_metric_callback)
```

## 🎯 Validation Framework

### Code Generation Evaluation

The validation system includes:

1. **Standard Datasets**: HumanEval, MMLU, MBPP, TruthfulQA, HellaSwag
2. **Code Quality Metrics**: Perplexity, BLEU, ROUGE, CodeBLEU, Pass@k
3. **XML/MDX Validation**: Structure validation, HTML compliance
4. **JavaScript Validation**: Syntax checking, execution testing, complexity analysis

### Validation Results

Results are saved to:
- `evaluation_results/detailed_results.json` - Complete validation results
- `evaluation_results/predictions.json` - Generated code samples
- `evaluation_results/validation_report.md` - Human-readable report

## 🔧 Advanced Features

### Fault Tolerance

The infrastructure includes comprehensive fault tolerance:

- **Automatic retry mechanisms** for transient failures
- **Emergency checkpointing** during interruptions
- **Recovery from checkpoints** with full state restoration
- **Resource monitoring** and automatic cleanup
- **Graceful shutdown handling**

### Performance Optimization

- **Gradient accumulation** for effective large batch training
- **Mixed precision training** (FP16/BF16) for memory efficiency
- **Activation checkpointing** for memory savings
- **Optimizer state checkpointing**
- **Automatic batch size adaptation** based on GPU memory
- **DeepSpeed ZeRO optimization** for massive model parallelism

### Cloud Integration

Support for cloud storage services:

```yaml
checkpoint:
  use_cloud_storage: true
  cloud_storage_provider: "s3"  # or "gcs", "azure"
  cloud_bucket: "my-training-checkpoints"
  cloud_prefix: "sheikh-2.5-coder/"
```

## 🧪 Testing and Validation

### Running Validation

```python
# Run validation after training
python scripts/train_model.py --config config.yaml --validate_after_training

# Run standalone validation
python scripts/validate_model.py --model_path ./models/output --output_dir ./validation_results
```

### Custom Validation

```python
from scripts.validate_model import ModelValidator, ValidationConfig

config = ValidationConfig(
    model_path="./models/output",
    custom_datasets={
        "my_dataset": "path/to/custom/dataset.json"
    },
    xml_validation=True,
    js_validation=True
)

validator = ModelValidator(config)
results = validator.validate_model()
```

## 📈 Performance Metrics

The training infrastructure tracks comprehensive performance metrics:

### Training Metrics
- **Loss**: Training and evaluation loss
- **Perplexity**: Language model perplexity
- **Learning Rate**: Current learning rate
- **Gradient Norm**: Gradient magnitude
- **Throughput**: Tokens and samples per second

### System Metrics
- **GPU Utilization**: GPU usage percentage
- **Memory Usage**: RAM and GPU memory consumption
- **CPU Usage**: CPU utilization
- **Disk Usage**: Storage space utilization

### Code Quality Metrics
- **Pass@k**: Code generation success rate
- **CodeBLEU**: Code similarity score
- **XML Validity**: XML/MDX structure validation
- **JS Syntax**: JavaScript syntax validation
- **Execution Rate**: Successful code execution rate

## 🤝 Contributing

When adding new features to the training infrastructure:

1. **Follow the existing code structure** and naming conventions
2. **Add comprehensive type hints** and documentation
3. **Include error handling** and logging
4. **Add configuration support** for new parameters
5. **Update the README** with usage examples
6. **Test thoroughly** with different configurations

## 📝 License

This training infrastructure is part of the Sheikh-2.5-Coder project. See the main project LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or enable gradient checkpointing
2. **Distributed Training Fails**: Check GPU connectivity and initialize properly
3. **Validation Errors**: Ensure validation datasets are properly formatted
4. **Checkpoint Loading**: Verify checkpoint compatibility and file integrity

### Debug Mode

Enable debug logging for detailed information:

```python
from scripts.training_config import TrainingConfiguration

config = TrainingConfiguration()
config.monitoring.log_level = "DEBUG"
config.system.debug_mode = True
```

### Performance Tuning

For optimal performance:

1. **Increase batch size** if GPU memory allows
2. **Use mixed precision** (FP16/BF16) for memory efficiency
3. **Enable gradient checkpointing** for memory savings
4. **Use DeepSpeed ZeRO** for large models
5. **Optimize data loading** with multiple workers

---

For additional help or questions, please refer to the main project documentation or open an issue in the repository.