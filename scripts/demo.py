#!/usr/bin/env python3
"""
Demonstration of Sheikh-2.5-Coder Training Infrastructure
Shows how to use all components together
"""

import os
import sys
import json
from pathlib import Path

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent))

from training_config import create_default_config, load_config, save_config
from distributed_trainer import create_distributed_trainer, DistributedConfig
from checkpoint_manager import create_checkpoint_manager
from monitoring_utils import create_monitoring_system
from early_stopping import create_early_stopping_callback
from validate_model import ModelValidator, ValidationConfig

def demo_configuration():
    """Demonstrate configuration management"""
    print("=== Configuration Management Demo ===")
    
    # Create default configuration
    config = create_default_config()
    
    # Customize configuration
    config.training.num_epochs = 1  # Demo with fewer epochs
    config.training.batch_size = 2  # Small batch for demo
    config.monitoring.use_wandb = False  # Disable W&B for demo
    config.checkpoint.save_steps = 10  # Frequent saves for demo
    
    # Save configuration
    config_file = "demo_config.yaml"
    save_config(config, config_file)
    print(f"Configuration saved to {config_file}")
    
    # Load configuration
    loaded_config = load_config(config_file)
    print(f"Configuration loaded: {loaded_config}")
    
    # Validate configuration
    if loaded_config.validate():
        print("✅ Configuration validation passed")
    else:
        print("❌ Configuration validation failed")
    
    return loaded_config

def demo_distributed_trainer():
    """Demonstrate distributed training setup"""
    print("\n=== Distributed Training Setup Demo ===")
    
    # Create distributed configuration
    ds_config = DistributedConfig(
        use_distributed=True,
        use_deepspeed=True,
        use_accelerate=True,
        mixed_precision="fp16",
        gradient_accumulation_steps=4,
        gradient_checkpointing=True
    )
    
    # Create distributed trainer
    trainer = create_distributed_trainer(
        use_distributed=True,
        use_deepspeed=True,
        mixed_precision="fp16",
        gradient_accumulation_steps=4
    )
    
    print(f"Distributed trainer created: {trainer}")
    
    # Setup distributed training
    if trainer.setup_distributed():
        print("✅ Distributed training setup successful")
        print(f"Main process: {trainer.is_main_process}")
    else:
        print("❌ Distributed training setup failed")
    
    return trainer

def demo_checkpoint_manager():
    """Demonstrate checkpoint management"""
    print("\n=== Checkpoint Management Demo ===")
    
    # Create checkpoint manager
    checkpoint_manager = create_checkpoint_manager(
        checkpoint_dir="./demo_checkpoints",
        output_dir="./demo_output",
        compression_type="gzip",
        max_checkpoints=5,
        save_interval=5
    )
    
    print(f"Checkpoint manager created: {checkpoint_manager}")
    
    # Get checkpoint statistics
    stats = checkpoint_manager.get_checkpoint_stats()
    print(f"Checkpoint statistics: {stats}")
    
    # List checkpoints
    checkpoints = checkpoint_manager.list_checkpoints()
    print(f"Current checkpoints: {checkpoints}")
    
    return checkpoint_manager

def demo_monitoring_system():
    """Demonstrate monitoring and logging"""
    print("\n=== Monitoring System Demo ===")
    
    # Create monitoring system
    monitor = create_monitoring_system(
        log_dir="./demo_logs",
        use_wandb=False,  # Disable for demo
        use_tensorboard=True
    )
    
    print(f"Monitoring system created: {monitor}")
    
    # Start monitoring
    monitor.start()
    print("✅ Monitoring started")
    
    # Log some training steps
    for step in range(5):
        loss = 2.0 - step * 0.3 + (step * 0.1)  # Simulated loss
        lr = 1e-4 * (0.95 ** step)
        
        monitor.log_training_step(
            loss=loss,
            learning_rate=lr,
            grad_norm=1.0 + step * 0.2,
            step=step,
            throughput_samples_per_sec=10.0 + step
        )
    
    # Log evaluation
    monitor.log_evaluation(
        eval_loss=1.0,
        eval_accuracy=0.85,
        step=4
    )
    
    # Create summary report
    summary = monitor.create_summary_report()
    print(f"Training summary: {summary}")
    
    # Stop monitoring
    monitor.stop()
    print("✅ Monitoring stopped")
    
    return monitor

def demo_early_stopping():
    """Demonstrate early stopping"""
    print("\n=== Early Stopping Demo ===")
    
    # Create early stopping callback
    early_stopping = create_early_stopping_callback(
        metric_to_monitor="eval_loss",
        metric_mode="min",
        patience=3,
        threshold=0.001,
        min_steps=5
    )
    
    print(f"Early stopping callback created: {early_stopping}")
    
    # Simulate training with early stopping
    from scripts.early_stopping import EarlyStoppingState, TrainerState
    
    state = EarlyStoppingState()
    trainer_state = TrainerState()
    
    # Simulate training steps with improving metrics
    for step in range(8):
        trainer_state.global_step = step
        trainer_state.epoch = step / 8.0
        
        # Simulate evaluation with improving loss
        eval_loss = 2.0 - step * 0.3  # Improving loss
        
        logs = {"eval_loss": eval_loss}
        
        # Update early stopping state
        early_stopping._update_state_from_evaluation(trainer_state, logs)
        
        print(f"Step {step}: eval_loss={eval_loss:.3f}, patience={early_stopping.state.current_patience}")
        
        # Check if early stopping should trigger
        if early_stopping.state.current_patience >= early_stopping.config.early_stopping_patience:
            print(f"✅ Early stopping triggered at step {step}")
            break
    
    return early_stopping

def demo_model_validation():
    """Demonstrate model validation (mock)"""
    print("\n=== Model Validation Demo ===")
    
    # This is a demonstration without an actual model
    # In practice, you would load a real model
    
    print("Creating validation configuration...")
    
    config = ValidationConfig(
        model_path="./demo_model",  # Mock path
        device="cpu",  # Use CPU for demo
        max_new_tokens=100,
        temperature=0.2,
        xml_validation=True,
        js_validation=True,
        validation_datasets=["mbpp"],  # Use smaller dataset
        output_dir="./demo_validation",
        save_detailed_results=True,
        generate_report=True
    )
    
    print(f"Validation config created: {config}")
    
    # Note: In a real scenario, you would:
    # 1. Have a trained model
    # 2. Load it with validator.load_model()
    # 3. Run validator.validate_model()
    
    print("ℹ️  Validation demo skipped (no model available)")
    print("To run actual validation:")
    print("  validator = ModelValidator(config)")
    print("  validator.load_model()")
    print("  results = validator.validate_model()")
    
    return config

def demo_training_orchestration():
    """Demonstrate the main training orchestration"""
    print("\n=== Training Orchestration Demo ===")
    
    # This demonstrates how all components work together
    print("Main training pipeline:")
    print("1. Configuration Management ✅")
    print("2. Distributed Training Setup ✅")
    print("3. Checkpoint Management ✅")
    print("4. Monitoring System ✅")
    print("5. Early Stopping ✅")
    print("6. Model Validation ✅")
    
    print("\nTo run actual training:")
    print("python scripts/train_model.py --config demo_config.yaml --dry_run")
    print("python scripts/train_model.py --config demo_config.yaml --validate_after_training")
    
    print("\nTraining would involve:")
    print("- Loading and configuring model")
    print("- Preparing training and validation datasets")
    print("- Setting up distributed training environment")
    print("- Initializing monitoring and checkpointing")
    print("- Running training loop with early stopping")
    print("- Saving checkpoints and final model")
    print("- Running comprehensive validation")
    print("- Generating training reports")

def main():
    """Run all demonstrations"""
    print("🚀 Sheikh-2.5-Coder Training Infrastructure Demo")
    print("=" * 60)
    
    # Run demos
    try:
        config = demo_configuration()
        
        trainer = demo_distributed_trainer()
        
        checkpoint_manager = demo_checkpoint_manager()
        
        monitor = demo_monitoring_system()
        
        early_stopping = demo_early_stopping()
        
        validation_config = demo_model_validation()
        
        demo_training_orchestration()
        
        print("\n" + "=" * 60)
        print("✅ All demos completed successfully!")
        print("\nNext steps:")
        print("1. Review the generated configuration files")
        print("2. Prepare your training data")
        print("3. Run: python scripts/train_model.py --config demo_config.yaml")
        print("4. Monitor training progress in logs/")
        print("5. Check validation results in demo_validation/")
        
        # Cleanup demo files
        demo_files = [
            "demo_config.yaml",
            "demo_checkpoints",
            "demo_output", 
            "demo_logs",
            "demo_validation"
        ]
        
        print("\nCleaning up demo files...")
        for file_path in demo_files:
            path = Path(file_path)
            if path.exists():
                if path.is_dir():
                    import shutil
                    shutil.rmtree(path)
                else:
                    path.unlink()
        
        print("Demo files cleaned up.")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)