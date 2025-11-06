#!/usr/bin/env python3
"""
Monitoring and Logging Utilities for Sheikh-2.5-Coder Training
Handles real-time monitoring, metrics tracking, and logging
"""

import os
import json
import logging
import time
import psutil
import threading
import queue
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import io
import sys

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    try:
        from tensorboardX import SummaryWriter
        TENSORBOARD_AVAILABLE = True
    except ImportError:
        TENSORBOARD_AVAILABLE = False

try:
    import GPUtil
    GPU_MONITORING_AVAILABLE = True
except ImportError:
    GPU_MONITORING_AVAILABLE = False

import torch
import numpy as np


@dataclass
class MetricData:
    """Container for metric data"""
    name: str
    value: float
    step: int
    timestamp: str
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class SystemMetrics:
    """System resource metrics"""
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None
    gpu_utilization: Optional[float] = None
    gpu_temperature: Optional[float] = None
    disk_usage_percent: float
    disk_free_gb: float
    timestamp: str


@dataclass
class TrainingMetrics:
    """Training-specific metrics"""
    loss: Optional[float] = None
    perplexity: Optional[float] = None
    accuracy: Optional[float] = None
    learning_rate: Optional[float] = None
    grad_norm: Optional[float] = None
    throughput_tokens_per_sec: Optional[float] = None
    throughput_samples_per_sec: Optional[float] = None
    epoch: Optional[float] = None
    step: Optional[int] = None
    eval_loss: Optional[float] = None
    eval_accuracy: Optional[float] = None
    eval_perplexity: Optional[float] = None
    timestamp: str


class LogFormatter(logging.Formatter):
    """Custom log formatter with colors and timestamps"""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        # Add color to level name
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        
        # Format the message
        formatted = super().format(record)
        return formatted


class MetricTracker:
    """Tracks and manages training metrics"""
    
    def __init__(
        self,
        max_history: int = 10000,
        track_percentiles: List[int] = [50, 90, 95, 99]
    ):
        self.max_history = max_history
        self.track_percentiles = track_percentiles
        
        # Storage for different types of metrics
        self.training_metrics = defaultdict(lambda: deque(maxlen=max_history))
        self.evaluation_metrics = defaultdict(lambda: deque(maxlen=max_history))
        self.system_metrics = deque(maxlen=max_history)
        
        # Current metric values
        self.current_training = {}
        self.current_evaluation = {}
        self.current_system = None
        
        # Threading
        self._lock = threading.RLock()
        
        # Callback functions
        self.metric_callbacks = defaultdict(list)
    
    def update_training_metrics(self, metrics: Dict[str, Union[float, int]], step: int) -> None:
        """Update training metrics"""
        with self._lock:
            timestamp = datetime.now().isoformat()
            
            # Add to history
            for name, value in metrics.items():
                if value is not None:
                    self.training_metrics[name].append((step, value, timestamp))
                    self.current_training[name] = value
            
            # Calculate derived metrics
            self._calculate_derived_metrics('training')
            
            # Trigger callbacks
            self._trigger_callbacks('training', metrics, step)
    
    def update_evaluation_metrics(self, metrics: Dict[str, Union[float, int]], step: int) -> None:
        """Update evaluation metrics"""
        with self._lock:
            timestamp = datetime.now().isoformat()
            
            # Add to history
            for name, value in metrics.items():
                if value is not None:
                    self.evaluation_metrics[name].append((step, value, timestamp))
                    self.current_evaluation[name] = value
            
            # Calculate derived metrics
            self._calculate_derived_metrics('evaluation')
            
            # Trigger callbacks
            self._trigger_callbacks('evaluation', metrics, step)
    
    def update_system_metrics(self, metrics: SystemMetrics) -> None:
        """Update system resource metrics"""
        with self._lock:
            self.system_metrics.append(metrics)
            self.current_system = metrics
            self._trigger_callbacks('system', metrics, 0)
    
    def _calculate_derived_metrics(self, metric_type: str) -> None:
        """Calculate derived metrics like perplexity from loss"""
        if metric_type == 'training':
            metrics_dict = self.current_training
            history = self.training_metrics
        else:
            metrics_dict = self.current_evaluation
            history = self.evaluation_metrics
        
        # Calculate perplexity from loss
        if 'loss' in metrics_dict and metrics_dict['loss'] is not None:
            perplexity = np.exp(metrics_dict['loss'])
            metrics_dict['perplexity'] = perplexity
            
            if metric_type == 'training':
                self.current_training['perplexity'] = perplexity
                self.training_metrics['perplexity'].append(
                    (list(history['loss'])[-1][0], perplexity, datetime.now().isoformat())
                )
            else:
                self.current_evaluation['perplexity'] = perplexity
                self.evaluation_metrics['perplexity'].append(
                    (list(history['loss'])[-1][0], perplexity, datetime.now().isoformat())
                )
    
    def _trigger_callbacks(self, metric_type: str, metrics: Dict[str, Any], step: int) -> None:
        """Trigger registered callbacks"""
        for callback in self.metric_callbacks[metric_type]:
            try:
                callback(metrics, step, metric_type)
            except Exception as e:
                logging.error(f"Error in metric callback: {e}")
    
    def register_callback(self, metric_type: str, callback: Callable) -> None:
        """Register a callback for metric updates"""
        self.metric_callbacks[metric_type].append(callback)
    
    def get_metric_history(
        self, 
        metric_name: str, 
        metric_type: str = 'training',
        last_n: Optional[int] = None
    ) -> List[tuple]:
        """Get metric history"""
        with self._lock:
            if metric_type == 'training':
                history = self.training_metrics[metric_name]
            elif metric_type == 'evaluation':
                history = self.evaluation_metrics[metric_name]
            else:
                raise ValueError(f"Unknown metric type: {metric_type}")
            
            if last_n:
                history = list(history)[-last_n:]
            
            return list(history)
    
    def get_metric_statistics(self, metric_name: str, metric_type: str = 'training') -> Dict[str, float]:
        """Get statistical summary of a metric"""
        history = self.get_metric_history(metric_name, metric_type)
        
        if not history:
            return {}
        
        values = [item[1] for item in history]  # Extract values
        
        stats = {
            'count': len(values),
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values)
        }
        
        # Add percentiles
        for percentile in self.track_percentiles:
            stats[f'p{percentile}'] = np.percentile(values, percentile)
        
        return stats
    
    def get_current_metrics(self, metric_type: str = 'training') -> Dict[str, Any]:
        """Get current metric values"""
        if metric_type == 'training':
            return self.current_training.copy()
        elif metric_type == 'evaluation':
            return self.current_evaluation.copy()
        elif metric_type == 'system':
            return self.current_system.__dict__ if self.current_system else {}
        else:
            raise ValueError(f"Unknown metric type: {metric_type}")


class SystemMonitor:
    """Monitors system resource usage"""
    
    def __init__(
        self,
        monitor_gpu: bool = True,
        monitor_disk: bool = True,
        monitor_network: bool = False,
        update_interval: int = 10
    ):
        self.monitor_gpu = monitor_gpu and GPU_MONITORING_AVAILABLE
        self.monitor_disk = monitor_disk
        self.monitor_network = monitor_network
        self.update_interval = update_interval
        
        self.running = False
        self.monitor_thread = None
        self.latest_metrics: Optional[SystemMetrics] = None
        
        # Network monitoring
        self.last_network_stats = None
        self.last_update_time = None
    
    def start(self) -> None:
        """Start system monitoring"""
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logging.info("System monitoring started")
    
    def stop(self) -> None:
        """Stop system monitoring"""
        self.running = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=30)
        
        logging.info("System monitoring stopped")
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop"""
        while self.running:
            try:
                metrics = self._collect_system_metrics()
                self.latest_metrics = metrics
                time.sleep(self.update_interval)
            except Exception as e:
                logging.error(f"Error in system monitoring: {e}")
                time.sleep(self.update_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_gb = memory.used / (1024**3)
        memory_total_gb = memory.total / (1024**3)
        
        # GPU metrics
        gpu_memory_used_mb = None
        gpu_memory_total_mb = None
        gpu_utilization = None
        gpu_temperature = None
        
        if self.monitor_gpu:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Assume single GPU for now
                    gpu_memory_used_mb = gpu.memoryUsed
                    gpu_memory_total_mb = gpu.memoryTotal
                    gpu_utilization = gpu.load * 100
                    gpu_temperature = gpu.temperature
            except Exception as e:
                logging.warning(f"GPU monitoring error: {e}")
        
        # Disk metrics
        disk_usage_percent = 0
        disk_free_gb = 0
        
        if self.monitor_disk:
            try:
                disk = psutil.disk_usage('/')
                disk_usage_percent = disk.percent
                disk_free_gb = disk.free / (1024**3)
            except Exception as e:
                logging.warning(f"Disk monitoring error: {e}")
        
        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_gb=memory_used_gb,
            memory_total_gb=memory_total_gb,
            gpu_memory_used_mb=gpu_memory_used_mb,
            gpu_memory_total_mb=gpu_memory_total_mb,
            gpu_utilization=gpu_utilization,
            gpu_temperature=gpu_temperature,
            disk_usage_percent=disk_usage_percent,
            disk_free_gb=disk_free_gb,
            timestamp=datetime.now().isoformat()
        )
    
    def get_latest_metrics(self) -> Optional[SystemMetrics]:
        """Get the latest system metrics"""
        return self.latest_metrics


class TensorBoardLogger:
    """TensorBoard logging wrapper"""
    
    def __init__(self, log_dir: Union[str, Path], flush_secs: int = 30):
        if not TENSORBOARD_AVAILABLE:
            raise ImportError("TensorBoard is not available. Install tensorboard or tensorboardX.")
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=str(self.log_dir), flush_secs=flush_secs)
        
        logging.info(f"TensorBoard logging initialized: {self.log_dir}")
    
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar value"""
        self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, tag: str, values: Dict[str, float], step: int) -> None:
        """Log multiple scalar values"""
        self.writer.add_scalars(tag, values, step)
    
    def log_histogram(self, tag: str, values: np.ndarray, step: int) -> None:
        """Log histogram values"""
        self.writer.add_histogram(tag, values, step)
    
    def log_image(self, tag: str, image: np.ndarray, step: int, dataformats: str = 'CHW') -> None:
        """Log an image"""
        self.writer.add_image(tag, image, step, dataformats=dataformats)
    
    def log_text(self, tag: str, text: str, step: int) -> None:
        """Log text"""
        self.writer.add_text(tag, text, step)
    
    def flush(self) -> None:
        """Flush TensorBoard writer"""
        self.writer.flush()
    
    def close(self) -> None:
        """Close TensorBoard writer"""
        self.writer.close()


class WandBLogger:
    """Weights & Biases logging wrapper"""
    
    def __init__(
        self,
        project: str,
        entity: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        run_name: Optional[str] = None
    ):
        if not WANDB_AVAILABLE:
            raise ImportError("Weights & Biases is not available. Install wandb.")
        
        self.project = project
        self.entity = entity
        self.config = config or {}
        self.tags = tags or []
        self.notes = notes
        self.run_name = run_name
        
        # Initialize W&B
        self.run = wandb.init(
            project=project,
            entity=entity,
            config=config,
            tags=tags,
            notes=notes,
            name=run_name
        )
        
        logging.info(f"WandB logging initialized: {self.run.name}")
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to W&B"""
        wandb.log(metrics, step=step)
    
    def log_config(self, config: Dict[str, Any]) -> None:
        """Log configuration"""
        wandb.config.update(config)
    
    def log_table(self, table_name: str, dataframe, step: Optional[int] = None) -> None:
        """Log a table to W&B"""
        wandb.log({table_name: wandb.Table(dataframe=dataframe)}, step=step)
    
    def watch_model(self, model: torch.nn.Module, log: str = "gradients", log_freq: int = 100) -> None:
        """Watch model gradients"""
        wandb.watch(model, log=log, log_freq=log_freq)
    
    def finish(self) -> None:
        """Finish W&B run"""
        wandb.finish()


class TrainingMonitor:
    """Main monitoring coordinator"""
    
    def __init__(
        self,
        log_dir: Union[str, Path] = "logs",
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: str = "sheikh-2.5-coder",
        wandb_entity: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
        log_level: str = "INFO",
        console_logging: bool = True,
        file_logging: bool = True
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging(log_level, console_logging, file_logging)
        
        # Initialize components
        self.metric_tracker = MetricTracker()
        self.system_monitor = SystemMonitor()
        
        # TensorBoard logging
        self.tensorboard = None
        if use_tensorboard and TENSORBOARD_AVAILABLE:
            self.tensorboard = TensorBoardLogger(
                log_dir=self.log_dir / "tensorboard"
            )
        
        # WandB logging
        self.wandb = None
        if use_wandb and WANDB_AVAILABLE:
            self.wandb = WandBLogger(
                project=wandb_project,
                entity=wandb_entity,
                config=wandb_config
            )
        
        # Monitoring state
        self.running = False
        
        # Register callbacks
        self._setup_callbacks()
        
        logging.info("TrainingMonitor initialized")
    
    def _setup_logging(self, log_level: str, console_logging: bool, file_logging: bool) -> None:
        """Setup logging configuration"""
        # Convert log level string to logging constant
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f'Invalid log level: {log_level}')
        
        # Create formatter
        formatter = LogFormatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(numeric_level)
        root_logger.handlers.clear()
        
        # Console logging
        if console_logging:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(numeric_level)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # File logging
        if file_logging:
            log_file = self.log_dir / "training.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
    
    def _setup_callbacks(self) -> None:
        """Setup metric callbacks"""
        
        def tensorboard_callback(metrics: Dict[str, Any], step: int, metric_type: str) -> None:
            """Callback to log to TensorBoard"""
            if self.tensorboard:
                for name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        tag = f"{metric_type}/{name}"
                        self.tensorboard.log_scalar(tag, value, step)
        
        def wandb_callback(metrics: Dict[str, Any], step: int, metric_type: str) -> None:
            """Callback to log to WandB"""
            if self.wandb:
                # Add metric type prefix
                prefixed_metrics = {
                    f"{metric_type}/{name}": value 
                    for name, value in metrics.items() 
                    if isinstance(value, (int, float))
                }
                self.wandb.log_metrics(prefixed_metrics, step=step)
        
        def system_callback(metrics: Any, step: int, metric_type: str) -> None:
            """Callback for system metrics"""
            if metric_type == 'system' and isinstance(metrics, SystemMetrics):
                if self.tensorboard:
                    # Log system metrics
                    system_metrics = {
                        'system/cpu_percent': metrics.cpu_percent,
                        'system/memory_percent': metrics.memory_percent,
                        'system/gpu_utilization': metrics.gpu_utilization or 0,
                        'system/gpu_memory_percent': (
                            (metrics.gpu_memory_used_mb / metrics.gpu_memory_total_mb * 100) 
                            if metrics.gpu_memory_used_mb and metrics.gpu_memory_total_mb else 0
                        ),
                        'system/disk_usage_percent': metrics.disk_usage_percent
                    }
                    
                    for name, value in system_metrics.items():
                        if value is not None:
                            self.tensorboard.log_scalar(name, value, step)
                
                if self.wandb:
                    wandb_metrics = {
                        'system/cpu_percent': metrics.cpu_percent,
                        'system/memory_percent': metrics.memory_percent,
                        'system/memory_used_gb': metrics.memory_used_gb,
                        'system/gpu_utilization': metrics.gpu_utilization,
                        'system/gpu_memory_mb': metrics.gpu_memory_used_mb,
                        'system/disk_usage_percent': metrics.disk_usage_percent,
                        'system/disk_free_gb': metrics.disk_free_gb
                    }
                    
                    # Remove None values
                    wandb_metrics = {k: v for k, v in wandb_metrics.items() if v is not None}
                    self.wandb.log_metrics(wandb_metrics, step=step)
        
        # Register callbacks
        self.metric_tracker.register_callback('training', tensorboard_callback)
        self.metric_tracker.register_callback('evaluation', tensorboard_callback)
        self.metric_tracker.register_callback('training', wandb_callback)
        self.metric_tracker.register_callback('evaluation', wandb_callback)
        self.metric_tracker.register_callback('system', system_callback)
    
    def start(self) -> None:
        """Start monitoring"""
        if self.running:
            return
        
        self.running = True
        self.system_monitor.start()
        
        logging.info("Training monitoring started")
    
    def stop(self) -> None:
        """Stop monitoring"""
        self.running = False
        self.system_monitor.stop()
        
        if self.tensorboard:
            self.tensorboard.flush()
        
        logging.info("Training monitoring stopped")
    
    def log_training_step(
        self,
        loss: float,
        learning_rate: float,
        grad_norm: Optional[float] = None,
        epoch: Optional[float] = None,
        step: Optional[int] = None,
        throughput_samples_per_sec: Optional[float] = None,
        throughput_tokens_per_sec: Optional[float] = None
    ) -> None:
        """Log training step metrics"""
        metrics = {
            'loss': loss,
            'learning_rate': learning_rate,
            'epoch': epoch or 0.0,
            'step': step or 0
        }
        
        if grad_norm is not None:
            metrics['grad_norm'] = grad_norm
        
        if throughput_samples_per_sec is not None:
            metrics['throughput_samples_per_sec'] = throughput_samples_per_sec
        
        if throughput_tokens_per_sec is not None:
            metrics['throughput_tokens_per_sec'] = throughput_tokens_per_sec
        
        self.metric_tracker.update_training_metrics(metrics, step or 0)
        
        # Update system metrics
        self._update_system_metrics()
    
    def log_evaluation(
        self,
        eval_loss: float,
        eval_accuracy: Optional[float] = None,
        step: Optional[int] = None
    ) -> None:
        """Log evaluation results"""
        metrics = {
            'eval_loss': eval_loss,
            'eval_step': step or 0
        }
        
        if eval_accuracy is not None:
            metrics['eval_accuracy'] = eval_accuracy
        
        self.metric_tracker.update_evaluation_metrics(metrics, step or 0)
        
        logging.info(f"Evaluation - Loss: {eval_loss:.4f}, Step: {step}")
    
    def _update_system_metrics(self) -> None:
        """Update system metrics"""
        system_metrics = self.system_monitor.get_latest_metrics()
        if system_metrics:
            self.metric_tracker.update_system_metrics(system_metrics)
    
    def log_hyperparameters(self, hyperparameters: Dict[str, Any]) -> None:
        """Log training hyperparameters"""
        logging.info("Training Hyperparameters:")
        for key, value in hyperparameters.items():
            logging.info(f"  {key}: {value}")
        
        # Log to WandB
        if self.wandb:
            self.wandb.log_config(hyperparameters)
    
    def log_model_info(self, model_info: Dict[str, Any]) -> None:
        """Log model information"""
        logging.info("Model Information:")
        for key, value in model_info.items():
            logging.info(f"  {key}: {value}")
        
        # Log to WandB
        if self.wandb:
            self.wandb.log_config({'model': model_info})
    
    def log_data_info(self, data_info: Dict[str, Any]) -> None:
        """Log data information"""
        logging.info("Data Information:")
        for key, value in data_info.items():
            logging.info(f"  {key}: {value}")
        
        # Log to WandB
        if self.wandb:
            self.wandb.log_config({'data': data_info})
    
    def create_summary_report(self) -> Dict[str, Any]:
        """Create a summary report of training metrics"""
        training_stats = {}
        evaluation_stats = {}
        
        # Get training metric statistics
        for metric_name in ['loss', 'learning_rate', 'grad_norm', 'perplexity']:
            stats = self.metric_tracker.get_metric_statistics(metric_name, 'training')
            if stats:
                training_stats[metric_name] = stats
        
        # Get evaluation metric statistics
        for metric_name in ['eval_loss', 'eval_accuracy', 'eval_perplexity']:
            stats = self.metric_tracker.get_metric_statistics(metric_name, 'evaluation')
            if stats:
                evaluation_stats[metric_name] = stats
        
        # Get current system metrics
        current_system = self.metric_tracker.get_current_metrics('system')
        
        return {
            'training_metrics': training_stats,
            'evaluation_metrics': evaluation_stats,
            'system_metrics': current_system,
            'timestamp': datetime.now().isoformat()
        }
    
    def save_metrics_to_file(self, filepath: Optional[Union[str, Path]] = None) -> None:
        """Save all metrics to a JSON file"""
        if filepath is None:
            filepath = self.log_dir / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = Path(filepath)
        
        # Get all metric data
        data = {
            'training_metrics': dict(self.metric_tracker.training_metrics),
            'evaluation_metrics': dict(self.metric_tracker.evaluation_metrics),
            'system_metrics': [
                metrics.__dict__ for metrics in self.metric_tracker.system_metrics
            ],
            'summary': self.create_summary_report(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logging.info(f"Metrics saved to {filepath}")
    
    def watch_model(self, model: torch.nn.Module) -> None:
        """Watch model for gradient logging"""
        if self.wandb:
            self.wandb.watch_model(model)
    
    def finish(self) -> None:
        """Finish monitoring and cleanup"""
        self.stop()
        
        # Save final metrics
        self.save_metrics_to_file()
        
        # Close loggers
        if self.tensorboard:
            self.tensorboard.close()
        
        if self.wandb:
            self.wandb.finish()
        
        logging.info("Training monitoring finished")


def create_monitoring_system(
    log_dir: Union[str, Path] = "logs",
    use_wandb: bool = True,
    wandb_project: str = "sheikh-2.5-coder",
    wandb_entity: Optional[str] = None,
    **kwargs
) -> TrainingMonitor:
    """Create a monitoring system with default settings"""
    
    default_config = {
        'use_tensorboard': True,
        'use_wandb': use_wandb,
        'wandb_project': wandb_project,
        'wandb_entity': wandb_entity,
        'log_level': 'INFO',
        'console_logging': True,
        'file_logging': True
    }
    
    default_config.update(kwargs)
    
    return TrainingMonitor(
        log_dir=log_dir,
        **default_config
    )


if __name__ == "__main__":
    # Example usage
    monitor = create_monitoring_system(
        use_wandb=False,  # Set to True if W&B is available
        wandb_project="sheikh-2.5-coder-test"
    )
    
    # Start monitoring
    monitor.start()
    
    # Simulate some training
    for step in range(10):
        loss = 2.0 - step * 0.1 + np.random.normal(0, 0.05)
        lr = 1e-4 * (0.95 ** step)
        
        monitor.log_training_step(
            loss=loss,
            learning_rate=lr,
            grad_norm=np.random.uniform(0.5, 2.0),
            step=step,
            throughput_samples_per_sec=10 + np.random.normal(0, 1)
        )
        
        time.sleep(1)  # Simulate training time
    
    # Simulate evaluation
    monitor.log_evaluation(
        eval_loss=1.5,
        eval_accuracy=0.85,
        step=10
    )
    
    # Create summary report
    summary = monitor.create_summary_report()
    print("\nTraining Summary:")
    print(json.dumps(summary, indent=2, default=str))
    
    # Finish monitoring
    monitor.finish()