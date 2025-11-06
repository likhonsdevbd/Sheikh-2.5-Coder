#!/usr/bin/env python3
"""
Training Environment Setup Script for Sheikh-2.5-Coder
Handles environment configuration, dependency management, and system setup
"""

import os
import sys
import json
import argparse
import logging
import subprocess
import shutil
import platform
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import requests

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/setup.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class TrainingEnvironmentSetup:
    """Comprehensive environment setup for model training"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.project_root = Path.cwd().parent.parent / "Sheikh-2.5-Coder"
        self.setup_scripts_dir = self.project_root / "scripts"
        self.config_dir = self.project_root / "configs"
        self.logs_dir = self.project_root / "logs"
        
        # Create directories if they don't exist
        self.create_directories()
        
        # Load configuration
        self.config = self.load_configuration(config_path)
        
        # System information
        self.system_info = self.get_system_info()
        
    def create_directories(self):
        """Create necessary directories"""
        directories = [
            self.logs_dir,
            self.config_dir,
            self.setup_scripts_dir,
            self.project_root / "data" / "processed",
            self.project_root / "data" / "tokenized",
            self.project_root / "models" / "checkpoints",
            self.project_root / "models" / "metrics",
            self.project_root / "evaluation" / "results"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    def load_configuration(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or create default"""
        
        if config_path and Path(config_path).exists():
            logger.info(f"Loading configuration from {config_path}")
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            logger.info("Using default configuration")
            config = self.create_default_config()
        
        return config
    
    def create_default_config(self) -> Dict:
        """Create default training configuration"""
        config = {
            "training": {
                "base_model": "microsoft/phi-2",
                "max_sequence_length": 2048,
                "gradient_accumulation_steps": 4,
                "warmup_steps": 100,
                "max_steps": 10000,
                "save_steps": 1000,
                "eval_steps": 500,
                "logging_steps": 10,
                "learning_rate": 2e-05,
                "weight_decay": 0.01,
                "lr_scheduler_type": "cosine",
                "num_train_epochs": 1,
                "per_device_train_batch_size": 4,
                "per_device_eval_batch_size": 4,
                "fp16": True,
                "gradient_checkpointing": True
            },
            "data": {
                "train_file": "train_dataset.jsonl",
                "eval_file": "eval_dataset.jsonl",
                "max_samples": 100000,
                "eval_samples": 1000,
                "shuffle_seed": 42
            },
            "lora": {
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "target_modules": [
                    "q_proj",
                    "k_proj", 
                    "v_proj",
                    "o_proj"
                ],
                "bias": "none",
                "task_type": "CAUSAL_LM"
            },
            "infrastructure": {
                "gpu_type": "t4",
                "mixed_precision": "fp16",
                "dataloader_num_workers": 2,
                "remove_unused_columns": False,
                "dataloader_pin_memory": False
            },
            "monitoring": {
                "wandb_project": "sheikh-2.5-coder-training",
                "report_to": ["wandb"],
                "logging_dir": "logs"
            }
        }
        
        # Save default config
        config_file = self.config_dir / "training_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Default configuration saved to {config_file}")
        return config
    
    def get_system_info(self) -> Dict:
        """Gather system information"""
        info = {
            "platform": platform.platform(),
            "python_version": sys.version,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpu_info": [],
            "memory_info": {},
            "disk_info": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # GPU information
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_info = {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_total": torch.cuda.get_device_properties(i).total_memory,
                    "memory_allocated": torch.cuda.memory_allocated(i),
                    "memory_reserved": torch.cuda.memory_reserved(i)
                }
                info["gpu_info"].append(gpu_info)
        
        # Memory information
        try:
            import psutil
            memory = psutil.virtual_memory()
            info["memory_info"] = {
                "total": memory.total,
                "available": memory.available,
                "used": memory.used,
                "percentage": memory.percent
            }
        except ImportError:
            logger.warning("psutil not available - skipping memory info")
        
        # Disk information
        try:
            import psutil
            disk = psutil.disk_usage('/')
            info["disk_info"] = {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percentage": (disk.used / disk.total) * 100
            }
        except ImportError:
            logger.warning("psutil not available - skipping disk info")
        
        return info
    
    def check_python_requirements(self) -> Dict:
        """Check Python version and requirements"""
        logger.info("Checking Python environment...")
        
        # Check Python version
        python_version = sys.version_info
        version_ok = python_version >= (3, 8) and python_version < (3, 12)
        
        requirements_check = {
            "python_version": f"{python_version.major}.{python_version.minor}.{python_version.micro}",
            "version_ok": version_ok,
            "required_version": ">=3.8,<3.12",
            "issues": []
        }
        
        if not version_ok:
            requirements_check["issues"].append(f"Python version {python_version.major}.{python_version.minor} not supported")
        
        # Check pip
        try:
            result = subprocess.run([sys.executable, "-m", "pip", "--version"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                requirements_check["pip_available"] = True
                requirements_check["pip_version"] = result.stdout.strip()
            else:
                requirements_check["pip_available"] = False
                requirements_check["issues"].append("pip not available")
        except Exception as e:
            requirements_check["pip_available"] = False
            requirements_check["issues"].append(f"pip check failed: {str(e)}")
        
        logger.info(f"Python requirements check: {requirements_check}")
        return requirements_check
    
    def install_dependencies(self) -> Dict:
        """Install required dependencies"""
        logger.info("Installing Python dependencies...")
        
        # Define required packages
        requirements = {
            "torch": "torch>=2.0.0",
            "torchvision": "torchvision>=0.15.0",
            "transformers": "transformers>=4.30.0",
            "datasets": "datasets>=2.12.0",
            "accelerate": "accelerate>=0.20.0",
            "peft": "peft>=0.4.0",
            "bitsandbytes": "bitsandbytes>=0.39.0",
            "trl": "trl>=0.4.0",
            "wandb": "wandb>=0.15.0",
            "huggingface_hub": "huggingface_hub>=0.15.0",
            "evaluate": "evaluate>=0.4.0",
            "scipy": "scipy>=1.10.0",
            "scikit-learn": "scikit-learn>=1.2.0",
            "matplotlib": "matplotlib>=3.6.0",
            "seaborn": "seaborn>=0.12.0",
            "pandas": "pandas>=1.5.0",
            "numpy": "numpy>=1.21.0",
            "psutil": "psutil>=5.9.0",
            "tqdm": "tqdm>=4.64.0",
            "requests": "requests>=2.28.0",
            "pyyaml": "pyyaml>=6.0",
            "jsonlines": "jsonlines>=3.1.0",
            "nltk": "nltk>=3.8",
            "jieba": "jieba>=0.42.1",
            "optimum": "optimum>=1.8.0",
            "onnxruntime": "onnxruntime>=1.15.0",
            "bert_score": "bert_score>=0.3.13"
        }
        
        installation_results = {
            "packages": {},
            "success": True,
            "errors": []
        }
        
        for package, version_spec in requirements.items():
            try:
                logger.info(f"Installing {package}...")
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", version_spec],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout per package
                )
                
                if result.returncode == 0:
                    installation_results["packages"][package] = "success"
                    logger.info(f"✓ {package} installed successfully")
                else:
                    installation_results["packages"][package] = "failed"
                    installation_results["errors"].append(f"{package}: {result.stderr}")
                    installation_results["success"] = False
                    logger.error(f"✗ {package} installation failed: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                installation_results["packages"][package] = "timeout"
                installation_results["errors"].append(f"{package}: Installation timeout")
                installation_results["success"] = False
                logger.error(f"✗ {package} installation timed out")
                
            except Exception as e:
                installation_results["packages"][package] = "error"
                installation_results["errors"].append(f"{package}: {str(e)}")
                installation_results["success"] = False
                logger.error(f"✗ {package} installation error: {str(e)}")
        
        # Create requirements.txt
        self.create_requirements_file(installation_results)
        
        logger.info(f"Dependency installation completed. Success: {installation_results['success']}")
        return installation_results
    
    def create_requirements_file(self, installation_results: Dict):
        """Create requirements.txt file based on installation results"""
        requirements_content = [
            "# Sheikh-2.5-Coder Requirements",
            "# Generated automatically on " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "",
            "# Core ML/DL libraries",
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "transformers>=4.30.0",
            "accelerate>=0.20.0",
            "",
            "# Model optimization",
            "peft>=0.4.0",
            "bitsandbytes>=0.39.0",
            "trl>=0.4.0",
            "optimum>=1.8.0",
            "",
            "# Data handling",
            "datasets>=2.12.0",
            "pandas>=1.5.0",
            "numpy>=1.21.0",
            "",
            "# Evaluation and metrics",
            "evaluate>=0.4.0",
            "scipy>=1.10.0",
            "scikit-learn>=1.2.0",
            "bert_score>=0.3.13",
            "",
            "# Monitoring and logging",
            "wandb>=0.15.0",
            "tqdm>=4.64.0",
            "",
            "# Utilities",
            "psutil>=5.9.0",
            "requests>=2.28.0",
            "pyyaml>=6.0",
            "jsonlines>=3.1.0",
            "nltk>=3.8",
            "",
            "# Visualization",
            "matplotlib>=3.6.0",
            "seaborn>=0.12.0",
            "",
            "# HuggingFace integration",
            "huggingface_hub>=0.15.0",
            "",
            "# ONNX support",
            "onnxruntime>=1.15.0",
            ""
        ]
        
        requirements_file = self.project_root / "requirements.txt"
        with open(requirements_file, 'w') as f:
            f.write('\n'.join(requirements_content))
        
        logger.info(f"Requirements file created: {requirements_file}")
    
    def setup_gpu_environment(self) -> Dict:
        """Setup GPU environment and CUDA configuration"""
        logger.info("Setting up GPU environment...")
        
        gpu_setup = {
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": 0,
            "gpu_info": [],
            "cuda_version": None,
            "setup_successful": True,
            "warnings": []
        }
        
        if not torch.cuda.is_available():
            gpu_setup["warnings"].append("CUDA not available - training will use CPU (much slower)")
            gpu_setup["setup_successful"] = False
            return gpu_setup
        
        gpu_setup["cuda_version"] = torch.version.cuda
        gpu_setup["gpu_count"] = torch.cuda.device_count()
        
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory
            
            gpu_setup["gpu_info"].append({
                "index": i,
                "name": gpu_name,
                "memory_gb": gpu_memory / (1024**3)
            })
            
            logger.info(f"GPU {i}: {gpu_name} ({gpu_memory / (1024**3):.1f}GB)")
        
        # Test GPU functionality
        try:
            test_tensor = torch.randn(10, 10).cuda()
            result = torch.matmul(test_tensor, test_tensor.t())
            gpu_setup["gpu_test"] = "success"
            logger.info("GPU functionality test passed")
        except Exception as e:
            gpu_setup["gpu_test"] = "failed"
            gpu_setup["warnings"].append(f"GPU functionality test failed: {str(e)}")
            gpu_setup["setup_successful"] = False
        
        # Set CUDA environment variables
        cuda_env_vars = {
            "CUDA_VISIBLE_DEVICES": "0" if gpu_setup["gpu_count"] > 0 else "",
            "TORCH_CUDA_ARCH_LIST": "Auto",
            "CUDA_CACHE_PATH": str(self.project_root / ".cuda_cache")
        }
        
        env_file = self.config_dir / "cuda_environment.env"
        with open(env_file, 'w') as f:
            f.write("# CUDA Environment Variables\n")
            for key, value in cuda_env_vars.items():
                f.write(f"export {key}={value}\n")
        
        logger.info(f"CUDA environment variables saved to {env_file}")
        gpu_setup["environment_variables"] = cuda_env_vars
        
        return gpu_setup
    
    def setup_wandb_configuration(self) -> Dict:
        """Setup Weights & Biases configuration"""
        logger.info("Setting up W&B configuration...")
        
        wandb_config = {
            "configured": False,
            "project": "sheikh-2.5-coder-training",
            "entity": "sheikh-team",
            "notes": "Automated training runs",
            "config_file": str(self.config_dir / "wandb_config.json")
        }
        
        # Create W&B config file
        wandb_settings = {
            "project": wandb_config["project"],
            "entity": wandb_config["entity"],
            "notes": wandb_config["notes"],
            "save_code": True,
            "sync_tensorboard": True,
            "monitor_gym": False,
            "reinit": True
        }
        
        with open(wandb_config["config_file"], 'w') as f:
            json.dump(wandb_settings, f, indent=2)
        
        logger.info(f"W&B configuration saved to {wandb_config['config_file']}")
        wandb_config["configured"] = True
        
        return wandb_config
    
    def setup_huggingface_configuration(self) -> Dict:
        """Setup HuggingFace Hub configuration"""
        logger.info("Setting up HuggingFace Hub configuration...")
        
        hf_config = {
            "configured": False,
            "cache_dir": str(self.project_root / ".hf_cache"),
            "config_file": str(self.config_dir / "hf_config.json"),
            "default_repo_type": "model"
        }
        
        # Create HF cache directory
        hf_cache_dir = Path(hf_config["cache_dir"])
        hf_cache_dir.mkdir(exist_ok=True)
        
        # Create HF config file
        hf_settings = {
            "cache_dir": hf_config["cache_dir"],
            "repo_type": hf_config["default_repo_type"],
            "use_auth_token": True,
            "local_files_only": False
        }
        
        with open(hf_config["config_file"], 'w') as f:
            json.dump(hf_settings, f, indent=2)
        
        logger.info(f"HuggingFace configuration saved to {hf_config['config_file']}")
        hf_config["configured"] = True
        
        return hf_config
    
    def create_training_scripts(self) -> Dict:
        """Create training automation scripts"""
        logger.info("Creating training automation scripts...")
        
        scripts_created = []
        
        # Quick training script
        quick_train_script = '''#!/bin/bash
# Quick Training Script for Sheikh-2.5-Coder

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="sheikh-2.5-coder-training"
export HF_HOME="~/.cache/huggingface"

# Source environment
source configs/cuda_environment.env

echo "Starting quick training run..."
python scripts/auto_train.py \\
    --model_name microsoft/phi-2 \\
    --data_path data/processed \\
    --output_path models/checkpoints \\
    --gpu_type t4 \\
    --training_steps 1000 \\
    --batch_size 2 \\
    --learning_rate 2e-5 \\
    --run_id "quick_$(date +%Y%m%d_%H%M%S)" \\
    --timestamp "$(date +%Y-%m-%d %H:%M:%S)"

echo "Quick training completed!"
'''
        
        script_path = self.project_root / "scripts" / "quick_train.sh"
        with open(script_path, 'w') as f:
            f.write(quick_train_script)
        os.chmod(script_path, 0o755)
        scripts_created.append(str(script_path))
        
        # Evaluation script
        eval_script = '''#!/bin/bash
# Model Evaluation Script

echo "Running comprehensive evaluation..."
python scripts/evaluate_model.py \\
    --model_path models/checkpoints/latest \\
    --output_path evaluation/results \\
    --run_id "eval_$(date +%Y%m%d_%H%M%S)"

echo "Evaluation completed!"
'''
        
        eval_path = self.project_root / "scripts" / "run_evaluation.sh"
        with open(eval_path, 'w') as f:
            f.write(eval_script)
        os.chmod(eval_path, 0o755)
        scripts_created.append(str(eval_path))
        
        # Deployment script
        deploy_script = '''#!/bin/bash
# Model Deployment Script

echo "Deploying model..."
python scripts/deploy_model.py \\
    --model_path models/checkpoints/latest \\
    --output_path models/deployment \\
    --quantization int8 \\
    --optimization memory-optimization

echo "Deployment completed!"
'''
        
        deploy_path = self.project_root / "scripts" / "deploy.sh"
        with open(deploy_path, 'w') as f:
            f.write(deploy_script)
        os.chmod(deploy_path, 0o755)
        scripts_created.append(str(deploy_path))
        
        logger.info(f"Created training scripts: {scripts_created}")
        return {"scripts_created": scripts_created}
    
    def run_system_diagnostics(self) -> Dict:
        """Run comprehensive system diagnostics"""
        logger.info("Running system diagnostics...")
        
        diagnostics = {
            "timestamp": datetime.now().isoformat(),
            "system_info": self.system_info,
            "python_requirements": self.check_python_requirements(),
            "gpu_environment": self.setup_gpu_environment(),
            "dependency_installation": self.install_dependencies(),
            "wandb_setup": self.setup_wandb_configuration(),
            "huggingface_setup": self.setup_huggingface_configuration(),
            "scripts_created": self.create_training_scripts(),
            "overall_status": "unknown",
            "recommendations": []
        }
        
        # Determine overall status
        issues = []
        
        if not diagnostics["python_requirements"]["version_ok"]:
            issues.append("Python version not supported")
        
        if not diagnostics["gpu_environment"]["setup_successful"]:
            issues.append("GPU setup failed")
        
        if not diagnostics["dependency_installation"]["success"]:
            issues.append("Some dependencies failed to install")
        
        if issues:
            diagnostics["overall_status"] = "failed"
            diagnostics["recommendations"].extend(issues)
        else:
            diagnostics["overall_status"] = "success"
            diagnostics["recommendations"].extend([
                "System ready for training",
                "Run 'python scripts/auto_train.py --help' for training options",
                "Use 'bash scripts/quick_train.sh' for a quick test run",
                "Monitor training with: wandb sync"
            ])
        
        # Save diagnostics report
        report_file = self.project_root / "logs" / f"environment_setup_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(diagnostics, f, indent=2, default=str)
        
        logger.info(f"Environment setup report saved to {report_file}")
        
        return diagnostics
    
    def print_setup_summary(self, diagnostics: Dict):
        """Print setup summary"""
        print("\n" + "="*60)
        print("🏗️  SHEIKH-2.5-CODER ENVIRONMENT SETUP COMPLETE")
        print("="*60)
        
        print(f"\n📊 System Status: {diagnostics['overall_status'].upper()}")
        print(f"🖥️  Platform: {diagnostics['system_info']['platform']}")
        print(f"🐍 Python: {diagnostics['system_info']['python_version']}")
        
        if diagnostics['system_info']['cuda_available']:
            print(f"🔥 GPUs: {diagnostics['system_info']['gpu_count']} available")
            for gpu in diagnostics['system_info']['gpu_info']:
                print(f"   - GPU {gpu['index']}: {gpu['name']} ({gpu['memory_gb']:.1f}GB)")
        else:
            print("❌ No CUDA GPUs detected - training will be slow")
        
        print(f"\n📦 Dependencies: {'✅ OK' if diagnostics['dependency_installation']['success'] else '❌ Issues'}")
        print(f"🎯 W&B: {'✅ Configured' if diagnostics['wandb_setup']['configured'] else '❌ Failed'}")
        print(f"🤗 HuggingFace: {'✅ Configured' if diagnostics['huggingface_setup']['configured'] else '❌ Failed'}")
        
        if diagnostics['recommendations']:
            print(f"\n💡 Recommendations:")
            for rec in diagnostics['recommendations']:
                print(f"   - {rec}")
        
        print(f"\n🚀 Ready to train! Run:")
        print(f"   python scripts/auto_train.py --help")
        print(f"   bash scripts/quick_train.sh")
        
        print("\n" + "="*60)


def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description='Setup training environment for Sheikh-2.5-Coder')
    
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--skip-deps', action='store_true', help='Skip dependency installation')
    parser.add_argument('--skip-gpu', action='store_true', help='Skip GPU setup')
    
    args = parser.parse_args()
    
    logger.info("Starting Sheikh-2.5-Coder environment setup...")
    
    # Initialize setup
    setup = TrainingEnvironmentSetup(config_path=args.config)
    
    try:
        # Run diagnostics
        diagnostics = setup.run_system_diagnostics()
        
        # Print summary
        setup.print_setup_summary(diagnostics)
        
        # Exit with appropriate code
        if diagnostics['overall_status'] == 'success':
            logger.info("Environment setup completed successfully!")
            return 0
        else:
            logger.error("Environment setup completed with issues!")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Setup interrupted by user")
        return 130
        
    except Exception as e:
        logger.error(f"Setup failed with error: {str(e)}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main())