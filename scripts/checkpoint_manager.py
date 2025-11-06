#!/usr/bin/env python3
"""
Checkpoint Management System for Sheikh-2.5-Coder
Handles automatic checkpoint saving, loading, compression, and cloud storage
"""

import os
import shutil
import json
import gzip
import bz2
import lzma
import tarfile
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable
import logging
import threading
import queue
import subprocess
from dataclasses import dataclass, field

try:
    import boto3
    from botocore.exceptions import ClientError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

try:
    from google.cloud import storage as gcs
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False

try:
    from azure.storage.blob import BlobServiceClient
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

from transformers import (
    PreTrainedModel, 
    PreTrainedTokenizer,
    TrainingArguments,
    TrainerState,
    TrainerControl
)
import torch


logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint"""
    checkpoint_path: str
    step: int
    epoch: float
    timestamp: str
    trainer_state: Dict[str, Any]
    training_args: Dict[str, Any]
    model_config: Dict[str, Any]
    training_metrics: Dict[str, Any] = field(default_factory=dict)
    checkpoint_size_mb: float = 0.0
    compression_ratio: float = 1.0
    validation_loss: Optional[float] = None
    best_metric_value: Optional[float] = None
    is_best: bool = False
    cloud_stored: bool = False


class CheckpointCompressor:
    """Handles checkpoint compression and decompression"""
    
    COMPRESSION_METHODS = {
        'gzip': ('.gz', gzip.open, gzip.open),
        'bz2': ('.bz2', bz2.open, bz2.open),
        'xz': ('.xz', lzma.open, lzma.open),
        'tar.gz': ('.tar.gz', None, None),  # Special handling
    }
    
    def __init__(self, compression_type: str = 'gzip'):
        self.compression_type = compression_type
        self.compression_method = self.COMPRESSION_METHODS.get(compression_type)
        
        if not self.compression_method:
            raise ValueError(f"Unsupported compression type: {compression_type}")
    
    def compress_file(self, source_path: Union[str, Path], dest_path: Union[str, Path]) -> float:
        """Compress a single file and return compression ratio"""
        source_path = Path(source_path)
        dest_path = Path(dest_path)
        
        original_size = source_path.stat().st_size
        
        if self.compression_type == 'tar.gz':
            return self._compress_tar_gz(source_path, dest_path)
        
        _, open_write, open_read = self.compression_method
        
        with open_write(dest_path, 'wb') as f_out:
            with open(source_path, 'rb') as f_in:
                shutil.copyfileobj(f_in, f_out)
        
        compressed_size = dest_path.stat().st_size
        return original_size / compressed_size if compressed_size > 0 else 1.0
    
    def decompress_file(self, source_path: Union[str, Path], dest_path: Union[str, Path]) -> None:
        """Decompress a file"""
        source_path = Path(source_path)
        dest_path = Path(dest_path)
        
        if self.compression_type == 'tar.gz':
            self._decompress_tar_gz(source_path, dest_path)
            return
        
        _, _, open_read = self.compression_method
        
        with open_read(source_path, 'rb') as f_in:
            with open(dest_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    
    def _compress_tar_gz(self, source_path: Path, dest_path: Path) -> float:
        """Compress using tar.gz"""
        original_size = source_path.stat().st_size
        
        with tarfile.open(dest_path, 'w:gz') as tar:
            tar.add(source_path, arcname=source_path.name)
        
        compressed_size = dest_path.stat().st_size
        return original_size / compressed_size if compressed_size > 0 else 1.0
    
    def _decompress_tar_gz(self, source_path: Path, dest_path: Path) -> None:
        """Decompress tar.gz archive"""
        with tarfile.open(source_path, 'r:gz') as tar:
            tar.extractall(dest_path.parent)
            
        # Find the extracted file
        extracted_files = list(dest_path.parent.glob(source_path.stem.replace('.tar', '')))
        if extracted_files:
            shutil.move(str(extracted_files[0]), str(dest_path))


class CloudStorageManager:
    """Handles cloud storage operations for checkpoints"""
    
    def __init__(self, provider: str, bucket_name: str, prefix: str = ""):
        self.provider = provider.lower()
        self.bucket_name = bucket_name
        self.prefix = prefix
        
        if self.provider == 's3':
            if not AWS_AVAILABLE:
                raise ImportError("boto3 is required for S3 storage")
            self.s3_client = boto3.client('s3')
        elif self.provider == 'gcs':
            if not GCP_AVAILABLE:
                raise ImportError("google-cloud-storage is required for GCS")
            self.gcs_client = gcs.Client()
            self.bucket = self.gcs_client.bucket(bucket_name)
        elif self.provider == 'azure':
            if not AZURE_AVAILABLE:
                raise ImportError("azure-storage-blob is required for Azure")
            self.blob_service = BlobServiceClient(account_url=f"https://{bucket_name}.blob.core.windows.net")
            self.container = self.blob_service.get_container_client("checkpoints")
        else:
            raise ValueError(f"Unsupported cloud provider: {provider}")
    
    def upload_file(self, local_path: Union[str, Path], cloud_path: str) -> bool:
        """Upload a file to cloud storage"""
        local_path = Path(local_path)
        cloud_path = f"{self.prefix}{cloud_path}".strip('/')
        
        try:
            if self.provider == 's3':
                self.s3_client.upload_file(str(local_path), self.bucket_name, cloud_path)
            elif self.provider == 'gcs':
                blob = self.bucket.blob(cloud_path)
                blob.upload_from_filename(str(local_path))
            elif self.provider == 'azure':
                blob = self.container.get_blob_client(cloud_path)
                with open(local_path, 'rb') as f:
                    blob.upload_blob(f, overwrite=True)
            
            logger.info(f"Uploaded {local_path} to {cloud_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload {local_path} to {cloud_path}: {e}")
            return False
    
    def download_file(self, cloud_path: str, local_path: Union[str, Path]) -> bool:
        """Download a file from cloud storage"""
        cloud_path = f"{self.prefix}{cloud_path}".strip('/')
        
        try:
            if self.provider == 's3':
                self.s3_client.download_file(self.bucket_name, cloud_path, str(local_path))
            elif self.provider == 'gcs':
                blob = self.bucket.blob(cloud_path)
                blob.download_to_filename(str(local_path))
            elif self.provider == 'azure':
                blob = self.container.get_blob_client(cloud_path)
                with open(local_path, 'wb') as f:
                    blob.download_blob().readinto(f)
            
            logger.info(f"Downloaded {cloud_path} to {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {cloud_path} to {local_path}: {e}")
            return False
    
    def list_checkpoints(self) -> List[str]:
        """List all checkpoint files in cloud storage"""
        try:
            if self.provider == 's3':
                response = self.s3_client.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix=self.prefix
                )
                return [obj['Key'].replace(self.prefix, '') for obj in response.get('Contents', [])]
            
            elif self.provider == 'gcs':
                blobs = self.bucket.list_blobs(prefix=self.prefix)
                return [blob.name.replace(self.prefix, '') for blob in blobs]
            
            elif self.provider == 'azure':
                blob_list = self.container.list_blobs(name_starts_with=self.prefix)
                return [blob.name.replace(self.prefix, '') for blob in blob_list]
                
        except Exception as e:
            logger.error(f"Failed to list checkpoints: {e}")
            return []
    
    def delete_checkpoint(self, cloud_path: str) -> bool:
        """Delete a checkpoint from cloud storage"""
        cloud_path = f"{self.prefix}{cloud_path}".strip('/')
        
        try:
            if self.provider == 's3':
                self.s3_client.delete_object(Bucket=self.bucket_name, Key=cloud_path)
            elif self.provider == 'gcs':
                self.bucket.blob(cloud_path).delete()
            elif self.provider == 'azure':
                blob = self.container.get_blob_client(cloud_path)
                blob.delete_blob()
            
            logger.info(f"Deleted {cloud_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete {cloud_path}: {e}")
            return False


class CheckpointManager:
    """Main checkpoint management system"""
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        output_dir: Union[str, Path],
        compression_type: str = 'gzip',
        max_checkpoints: int = 10,
        save_interval: int = 1000,
        cloud_storage: Optional[Dict[str, str]] = None,
        auto_save: bool = True,
        validation_callback: Optional[Callable] = None
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.output_dir = Path(output_dir)
        self.compressor = CheckpointCompressor(compression_type)
        self.max_checkpoints = max_checkpoints
        self.save_interval = save_interval
        self.auto_save = auto_save
        self.validation_callback = validation_callback
        
        # Cloud storage
        self.cloud_manager = None
        if cloud_storage:
            self.cloud_manager = CloudStorageManager(
                provider=cloud_storage['provider'],
                bucket_name=cloud_storage['bucket'],
                prefix=cloud_storage.get('prefix', '')
            )
        
        # Checkpoint state
        self.checkpoints: List[CheckpointMetadata] = []
        self.best_checkpoints: Dict[str, CheckpointMetadata] = {}
        self.last_save_step = 0
        
        # Threading
        self.save_queue = queue.Queue()
        self.save_thread = None
        self.running = False
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing checkpoints
        self._load_existing_checkpoints()
        
        logger.info(f"Checkpoint manager initialized: {self.checkpoint_dir}")
    
    def _load_existing_checkpoints(self) -> None:
        """Load information about existing checkpoints"""
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint-*"))
        
        for checkpoint_file in checkpoint_files:
            if checkpoint_file.is_dir():
                metadata_file = checkpoint_file / "checkpoint_metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = CheckpointMetadata(**json.load(f))
                        
                        # Verify checkpoint exists
                        if self._verify_checkpoint(checkpoint_file):
                            self.checkpoints.append(metadata)
                        else:
                            logger.warning(f"Checkpoint {checkpoint_file} is corrupted, removing...")
                            shutil.rmtree(checkpoint_file)
                    
                    except Exception as e:
                        logger.error(f"Failed to load checkpoint metadata from {metadata_file}: {e}")
        
        # Sort by timestamp
        self.checkpoints.sort(key=lambda x: x.timestamp)
        
        logger.info(f"Loaded {len(self.checkpoints)} existing checkpoints")
    
    def _verify_checkpoint(self, checkpoint_path: Path) -> bool:
        """Verify that a checkpoint is valid"""
        try:
            # Check for essential files
            required_files = ['pytorch_model.bin', 'config.json', 'tokenizer.json']
            
            for file_name in required_files:
                file_path = checkpoint_path / file_name
                if not file_path.exists():
                    return False
            
            # Try to load checkpoint metadata
            metadata_file = checkpoint_path / "checkpoint_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Validate metadata structure
                required_fields = ['step', 'epoch', 'timestamp']
                if not all(field in metadata for field in required_fields):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Checkpoint verification failed for {checkpoint_path}: {e}")
            return False
    
    def start_auto_save(self) -> None:
        """Start automatic checkpoint saving thread"""
        if self.running:
            return
        
        self.running = True
        self.save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.save_thread.start()
        logger.info("Auto-save thread started")
    
    def stop_auto_save(self) -> None:
        """Stop automatic checkpoint saving thread"""
        self.running = False
        
        if self.save_thread:
            self.save_queue.put(None)  # Signal thread to stop
            self.save_thread.join(timeout=30)
        
        logger.info("Auto-save thread stopped")
    
    def _save_worker(self) -> None:
        """Worker thread for saving checkpoints"""
        while self.running:
            try:
                # Check for save tasks
                save_task = self.save_queue.get(timeout=1)
                
                if save_task is None:  # Stop signal
                    break
                
                # Execute save task
                checkpoint_path, metadata = save_task
                self._save_checkpoint_sync(checkpoint_path, metadata)
                
                self.save_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in save worker: {e}")
    
    def save_checkpoint(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        trainer_state: TrainerState,
        trainer_control: TrainerControl,
        training_args: TrainingArguments,
        step: int,
        epoch: float,
        training_metrics: Dict[str, Any] = None,
        force_save: bool = False
    ) -> Optional[CheckpointMetadata]:
        """Save a checkpoint"""
        
        if not force_save and step - self.last_save_step < self.save_interval:
            return None
        
        # Validate checkpoint if callback is provided
        if self.validation_callback:
            try:
                is_valid = self.validation_callback(model, trainer_state, step)
                if not is_valid:
                    logger.warning(f"Checkpoint validation failed for step {step}")
                    return None
            except Exception as e:
                logger.error(f"Checkpoint validation error: {e}")
                return None
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint-{step:06d}"
        
        # Create metadata
        metadata = CheckpointMetadata(
            checkpoint_path=str(checkpoint_path),
            step=step,
            epoch=epoch,
            timestamp=datetime.now().isoformat(),
            trainer_state=trainer_state.__dict__,
            training_args=training_args.to_dict(),
            model_config=model.config.to_dict(),
            training_metrics=training_metrics or {},
            checkpoint_size_mb=0.0,
            compression_ratio=1.0,
            is_best=False
        )
        
        # Add to queue for background saving
        if self.auto_save:
            self.save_queue.put((checkpoint_path, metadata))
        else:
            return self._save_checkpoint_sync(checkpoint_path, metadata)
        
        self.last_save_step = step
        logger.info(f"Checkpoint save scheduled for step {step}")
        return None
    
    def _save_checkpoint_sync(
        self, 
        checkpoint_path: Path, 
        metadata: CheckpointMetadata
    ) -> CheckpointMetadata:
        """Synchronous checkpoint saving"""
        try:
            logger.info(f"Saving checkpoint to {checkpoint_path}")
            
            # Remove existing checkpoint if it exists
            if checkpoint_path.exists():
                shutil.rmtree(checkpoint_path)
            
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            
            # Save model and tokenizer
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
            
            # Calculate checkpoint size
            total_size = 0
            for file_path in checkpoint_path.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            
            metadata.checkpoint_size_mb = total_size / (1024 * 1024)
            
            # Compress checkpoint if configured
            if self.compressor:
                compressed_path = checkpoint_path.with_suffix(
                    checkpoint_path.suffix + self.compressor.compression_method[0]
                )
                
                try:
                    # Create temporary copy for compression
                    temp_path = checkpoint_path.with_suffix('.temp')
                    shutil.copytree(checkpoint_path, temp_path)
                    
                    # Compress each file
                    compression_ratio = 1.0
                    for file_path in temp_path.rglob('*'):
                        if file_path.is_file():
                            compressed_file = compressed_path / file_path.relative_to(temp_path)
                            compressed_file.parent.mkdir(parents=True, exist_ok=True)
                            
                            ratio = self.compressor.compress_file(file_path, compressed_file)
                            compression_ratio = min(compression_ratio, ratio)
                    
                    # Replace original with compressed
                    shutil.rmtree(checkpoint_path)
                    shutil.move(str(compressed_path), str(checkpoint_path))
                    
                    metadata.compression_ratio = compression_ratio
                    logger.info(f"Checkpoint compressed with ratio {compression_ratio:.2f}")
                    
                except Exception as e:
                    logger.warning(f"Checkpoint compression failed: {e}")
                    # Continue with uncompressed version
            
            # Save metadata
            metadata_file = checkpoint_path / "checkpoint_metadata.json"
            with open(metadata_file, 'w') as f:
                # Convert metadata to dict for JSON serialization
                metadata_dict = {
                    'checkpoint_path': metadata.checkpoint_path,
                    'step': metadata.step,
                    'epoch': metadata.epoch,
                    'timestamp': metadata.timestamp,
                    'trainer_state': metadata.trainer_state,
                    'training_args': metadata.training_args,
                    'model_config': metadata.model_config,
                    'training_metrics': metadata.training_metrics,
                    'checkpoint_size_mb': metadata.checkpoint_size_mb,
                    'compression_ratio': metadata.compression_ratio,
                    'validation_loss': metadata.validation_loss,
                    'best_metric_value': metadata.best_metric_value,
                    'is_best': metadata.is_best,
                    'cloud_stored': metadata.cloud_stored
                }
                json.dump(metadata_dict, f, indent=2)
            
            # Update checkpoint list
            self.checkpoints.append(metadata)
            
            # Clean up old checkpoints
            self._cleanup_old_checkpoints()
            
            # Upload to cloud if configured
            if self.cloud_manager:
                self._upload_to_cloud(checkpoint_path, metadata)
            
            logger.info(f"Checkpoint saved successfully: {checkpoint_path}")
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint to {checkpoint_path}: {e}")
            # Clean up failed checkpoint
            if checkpoint_path.exists():
                shutil.rmtree(checkpoint_path)
            raise
    
    def _upload_to_cloud(self, checkpoint_path: Path, metadata: CheckpointMetadata) -> None:
        """Upload checkpoint to cloud storage"""
        if not self.cloud_manager:
            return
        
        try:
            cloud_path = f"checkpoint-{metadata.step:06d}/"
            
            # Upload each file in the checkpoint directory
            for file_path in checkpoint_path.rglob('*'):
                if file_path.is_file():
                    relative_path = file_path.relative_to(checkpoint_path)
                    cloud_file_path = f"{cloud_path}{relative_path}"
                    
                    if self.cloud_manager.upload_file(file_path, cloud_file_path):
                        metadata.cloud_stored = True
            
            logger.info(f"Checkpoint uploaded to cloud storage")
            
        except Exception as e:
            logger.error(f"Failed to upload checkpoint to cloud: {e}")
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints to maintain the maximum limit"""
        if len(self.checkpoints) <= self.max_checkpoints:
            return
        
        # Sort checkpoints by timestamp (oldest first)
        self.checkpoints.sort(key=lambda x: x.timestamp)
        
        # Remove oldest checkpoints
        checkpoints_to_remove = self.checkpoints[:-self.max_checkpoints]
        
        for checkpoint in checkpoints_to_remove:
            try:
                checkpoint_path = Path(checkpoint.checkpoint_path)
                
                # Remove from cloud storage first
                if self.cloud_manager and checkpoint.cloud_stored:
                    cloud_path = f"checkpoint-{checkpoint.step:06d}/"
                    # Note: We'd need to iterate through all files in cloud storage
                    # This is simplified for the example
                
                # Remove local checkpoint
                if checkpoint_path.exists():
                    shutil.rmtree(checkpoint_path)
                
                logger.info(f"Removed old checkpoint: {checkpoint_path}")
                
            except Exception as e:
                logger.error(f"Failed to remove old checkpoint: {e}")
        
        # Update checkpoint list
        self.checkpoints = self.checkpoints[-self.max_checkpoints:]
    
    def mark_as_best(
        self, 
        step: int, 
        metric_name: str, 
        metric_value: float,
        greater_is_better: bool = False
    ) -> Optional[CheckpointMetadata]:
        """Mark a checkpoint as the best based on a metric"""
        
        # Find the checkpoint
        checkpoint = None
        for cp in self.checkpoints:
            if cp.step == step:
                checkpoint = cp
                break
        
        if not checkpoint:
            logger.warning(f"Checkpoint for step {step} not found")
            return None
        
        # Update metadata
        checkpoint.best_metric_value = metric_value
        checkpoint.is_best = True
        
        # Update best checkpoints tracking
        if metric_name not in self.best_checkpoints:
            self.best_checkpoints[metric_name] = checkpoint
        
        # Check if this checkpoint is better
        current_best = self.best_checkpoints[metric_name]
        
        if greater_is_better:
            is_better = metric_value > current_best.best_metric_value
        else:
            is_better = metric_value < current_best.best_metric_value
        
        if is_better:
            # Remove "best" mark from previous best
            if current_best.step != checkpoint.step:
                current_best.is_best = False
            
            # Mark current as best
            checkpoint.is_best = True
            self.best_checkpoints[metric_name] = checkpoint
            
            logger.info(f"New best checkpoint for {metric_name}: step {step}, value {metric_value}")
            return checkpoint
        
        return None
    
    def get_best_checkpoint(self, metric_name: str = 'eval_loss') -> Optional[CheckpointMetadata]:
        """Get the best checkpoint for a metric"""
        return self.best_checkpoints.get(metric_name)
    
    def get_latest_checkpoint(self) -> Optional[CheckpointMetadata]:
        """Get the latest checkpoint"""
        if not self.checkpoints:
            return None
        
        return max(self.checkpoints, key=lambda x: x.step)
    
    def get_checkpoint_at_step(self, step: int) -> Optional[CheckpointMetadata]:
        """Get checkpoint at specific step"""
        for checkpoint in self.checkpoints:
            if checkpoint.step == step:
                return checkpoint
        
        return None
    
    def load_checkpoint(
        self, 
        step: Optional[int] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        load_training_args: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Load a checkpoint"""
        
        if checkpoint_path:
            checkpoint_path = Path(checkpoint_path)
        elif step is not None:
            metadata = self.get_checkpoint_at_step(step)
            if metadata:
                checkpoint_path = Path(metadata.checkpoint_path)
            else:
                logger.error(f"Checkpoint for step {step} not found")
                return None
        else:
            # Load latest checkpoint
            metadata = self.get_latest_checkpoint()
            if metadata:
                checkpoint_path = Path(metadata.checkpoint_path)
            else:
                logger.error("No checkpoints found to load")
                return None
        
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint path does not exist: {checkpoint_path}")
            return None
        
        try:
            # Check if it's a compressed checkpoint
            compressed_files = list(checkpoint_path.rglob("*.gz"))
            compressed_files.extend(list(checkpoint_path.rglob("*.bz2")))
            compressed_files.extend(list(checkpoint_path.rglob("*.xz")))
            
            if compressed_files and checkpoint_path.is_file():
                # Handle compressed checkpoint
                # This would require decompression logic
                logger.info("Compressed checkpoint detected, decompression not implemented in this example")
                return None
            
            # Load metadata
            metadata_file = checkpoint_path / "checkpoint_metadata.json"
            if not metadata_file.exists():
                logger.error(f"Checkpoint metadata not found: {metadata_file}")
                return None
            
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            result = {
                'checkpoint_path': checkpoint_path,
                'metadata': metadata,
                'load_training_args': load_training_args
            }
            
            # Load training arguments if requested
            if load_training_args:
                training_args_file = checkpoint_path / "trainer_state.json"
                if training_args_file.exists():
                    with open(training_args_file, 'r') as f:
                        result['training_args'] = json.load(f)
            
            logger.info(f"Checkpoint loaded: {checkpoint_path}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {checkpoint_path}: {e}")
            return None
    
    def cleanup_failed_checkpoints(self) -> None:
        """Clean up any failed or corrupted checkpoints"""
        failed_checkpoints = []
        
        for checkpoint in self.checkpoints:
            checkpoint_path = Path(checkpoint.checkpoint_path)
            
            if not self._verify_checkpoint(checkpoint_path):
                failed_checkpoints.append(checkpoint)
        
        for checkpoint in failed_checkpoints:
            try:
                checkpoint_path = Path(checkpoint.checkpoint_path)
                if checkpoint_path.exists():
                    shutil.rmtree(checkpoint_path)
                
                self.checkpoints.remove(checkpoint)
                logger.info(f"Removed failed checkpoint: {checkpoint_path}")
                
            except Exception as e:
                logger.error(f"Failed to remove failed checkpoint: {e}")
    
    def get_checkpoint_stats(self) -> Dict[str, Any]:
        """Get statistics about checkpoints"""
        if not self.checkpoints:
            return {
                'total_checkpoints': 0,
                'total_size_mb': 0.0,
                'average_size_mb': 0.0,
                'compression_ratio': 1.0,
                'cloud_stored_count': 0
            }
        
        total_size = sum(cp.checkpoint_size_mb for cp in self.checkpoints)
        avg_size = total_size / len(self.checkpoints)
        avg_compression = sum(cp.compression_ratio for cp in self.checkpoints) / len(self.checkpoints)
        cloud_count = sum(1 for cp in self.checkpoints if cp.cloud_stored)
        
        return {
            'total_checkpoints': len(self.checkpoints),
            'total_size_mb': total_size,
            'average_size_mb': avg_size,
            'compression_ratio': avg_compression,
            'cloud_stored_count': cloud_count,
            'best_checkpoints': {
                metric: {'step': cp.step, 'value': cp.best_metric_value}
                for metric, cp in self.best_checkpoints.items()
            }
        }
    
    def list_checkpoints(self, detailed: bool = False) -> List[Dict[str, Any]]:
        """List all checkpoints"""
        
        checkpoint_list = []
        for checkpoint in sorted(self.checkpoints, key=lambda x: x.step):
            checkpoint_info = {
                'step': checkpoint.step,
                'epoch': checkpoint.epoch,
                'timestamp': checkpoint.timestamp,
                'path': checkpoint.checkpoint_path,
                'size_mb': checkpoint.checkpoint_size_mb,
                'is_best': checkpoint.is_best,
                'cloud_stored': checkpoint.cloud_stored
            }
            
            if detailed:
                checkpoint_info.update({
                    'validation_loss': checkpoint.validation_loss,
                    'best_metric_value': checkpoint.best_metric_value,
                    'compression_ratio': checkpoint.compression_ratio,
                    'training_metrics': checkpoint.training_metrics
                })
            
            checkpoint_list.append(checkpoint_info)
        
        return checkpoint_list


def create_checkpoint_manager(
    checkpoint_dir: Union[str, Path],
    output_dir: Union[str, Path],
    **kwargs
) -> CheckpointManager:
    """Create a checkpoint manager with default settings"""
    
    # Default configuration
    default_config = {
        'compression_type': 'gzip',
        'max_checkpoints': 10,
        'save_interval': 1000,
        'auto_save': True,
        'cloud_storage': None
    }
    
    # Update with provided kwargs
    default_config.update(kwargs)
    
    return CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        **default_config
    )


if __name__ == "__main__":
    # Example usage
    checkpoint_manager = create_checkpoint_manager(
        checkpoint_dir="./models/checkpoints",
        output_dir="./models/output",
        compression_type='gzip',
        max_checkpoints=5,
        save_interval=100
    )
    
    # Print checkpoint statistics
    stats = checkpoint_manager.get_checkpoint_stats()
    print("Checkpoint Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # List all checkpoints
    checkpoints = checkpoint_manager.list_checkpoints(detailed=True)
    print(f"\nFound {len(checkpoints)} checkpoints")
    for checkpoint in checkpoints:
        print(f"  Step {checkpoint['step']}: {checkpoint['size_mb']:.2f} MB")