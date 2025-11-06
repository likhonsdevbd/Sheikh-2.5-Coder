# Sheikh-2.5-Coder Training Infrastructure - Quick Reference

## 🚀 Quick Start Commands

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Setup configuration
cp scripts/training_config_example.yaml my_config.yaml

# 3. Run demonstration
python scripts/demo.py

# 4. Basic training
python scripts/train_model.py --config my_config.yaml

# 5. Distributed training
python -m torch.distributed.launch --nproc_per_node=8 scripts/train_model.py --config my_config.yaml

# 6. Training with validation
python scripts/train_model.py --config my_config.yaml --validate_after_training

# 7. Resume from checkpoint
python scripts/train_model.py --config my_config.yaml --resume_from_checkpoint ./models/checkpoints/checkpoint-005000

# 8. Dry run (validate setup)
python scripts/train_model.py --config my_config.yaml --dry_run

# 9. Standalone validation
python scripts/validate_model.py --model_path ./models/output --output_dir ./validation_results
```

## 📁 Core Scripts

| Script | Purpose | Lines |
|--------|---------|-------|
| `train_model.py` | Main training orchestrator | 672 |
| `training_config.py` | Configuration management | 615 |
| `distributed_trainer.py` | Distributed training | 720 |
| `checkpoint_manager.py` | Checkpoint management | 882 |
| `monitoring_utils.py` | Monitoring & logging | 876 |
| `early_stopping.py` | Early stopping | 720 |
| `validate_model.py` | Model validation | 1288 |

## ⚙️ Key Configuration Parameters

```yaml
training:
  num_epochs: 3
  batch_size: 8
  learning_rate: 1.0e-4
  warmup_steps: 1000
  gradient_accumulation_steps: 4
  mixed_precision: "fp16"

monitoring:
  use_wandb: true
  wandb_project: "sheikh-2.5-coder"
  use_tensorboard: true

checkpoint:
  save_steps: 1000
  save_total_limit: 10
  compress_checkpoints: true

system:
  multi_gpu: true
  max_memory_gb: 32
```

## 🔧 Individual Component Usage

```python
from scripts.training_config import create_default_config
from scripts.distributed_trainer import create_distributed_trainer
from scripts.monitoring_utils import create_monitoring_system
from scripts.checkpoint_manager import create_checkpoint_manager

# Configuration
config = create_default_config()
config.training.num_epochs = 1
config.save_to_file("config.yaml")

# Distributed Training
trainer = create_distributed_trainer(use_distributed=True)
trainer.setup_distributed()

# Monitoring
monitor = create_monitoring_system(use_wandb=True)
monitor.start()

# Checkpointing
checkpoint_manager = create_checkpoint_manager("./checkpoints", "./output")
```

## 📊 Monitoring Endpoints

```bash
# TensorBoard
tensorboard --logdir logs/tensorboard --port 6006

# Weights & Biases Dashboard
open https://wandb.ai/your-entity/sheikh-2.5-coder

# Check logs
tail -f logs/training.log
```

## 🛠️ Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| CUDA OOM | Reduce batch_size or enable gradient_checkpointing |
| Distribute fails | Check GPU connectivity and nccl backend |
| Validation errors | Ensure datasets are properly formatted |
| Checkpoint loading | Verify checkpoint compatibility |

### Debug Mode

```python
config.monitoring.log_level = "DEBUG"
config.system.debug_mode = True
```

## 📈 Training Pipeline Flow

```
1. Configuration → training_config.py
2. Setup → distributed_trainer.py
3. Monitor → monitoring_utils.py
4. Train → train_model.py
   ├── Save checkpoints → checkpoint_manager.py
   ├── Early stop → early_stopping.py
   └── Validate → validate_model.py
5. Results → validation reports
```

## 🎯 Key Features

✅ **Distributed Training**: Multi-GPU with DeepSpeed  
✅ **Smart Checkpointing**: Auto-save with compression  
✅ **Real-time Monitoring**: W&B + TensorBoard  
✅ **Early Stopping**: Multiple criteria  
✅ **Model Validation**: XML/JS/MDX validation  
✅ **Fault Tolerance**: Graceful recovery  
✅ **Cloud Integration**: S3/GCS/Azure support  

## 📞 Support

- Full documentation: `scripts/README.md`
- Configuration guide: `scripts/training_config_example.yaml`
- Interactive demo: `python scripts/demo.py`
- Training summary: `scripts/TRAINING_INFRASTRUCTURE_SUMMARY.md`

---
**Ready to train Sheikh-2.5-Coder!** 🚀