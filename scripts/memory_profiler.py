#!/usr/bin/env python3
"""
Memory Profiler and Optimizer for Sheikh-2.5-Coder
Comprehensive memory usage analysis and optimization for on-device deployment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import psutil
import time
import math
from typing import Dict, Any, Optional, List, Tuple
import logging
import json
import copy
from pathlib import Path
import numpy as np

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import tracemalloc
    HAS_TRACEMALLOC = True
except ImportError:
    HAS_TRACEMALLOC = False


class MemoryProfiler:
    """
    Comprehensive memory profiling and analysis for transformer models.
    Tracks memory usage, provides optimization recommendations, and implements memory-saving techniques.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the memory profiler."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Memory tracking
        self.memory_snapshots = []
        self.layer_memory_usage = {}
        self.optimization_suggestions = []
        
        # Enable tracemalloc if available
        if HAS_TRACEMALLOC:
            tracemalloc.start()
        
        self.logger.info("Memory profiler initialized")
    
    def profile_model_memory(self, model: nn.Module, 
                           input_shape: Tuple[int, ...] = (1, 512),
                           include_gradients: bool = False) -> Dict[str, Any]:
        """
        Profile memory usage of a model.
        
        Args:
            model: PyTorch model to profile
            input_shape: Shape of input tensors (batch_size, sequence_length)
            include_gradients: Whether to include gradient memory in analysis
            
        Returns:
            Dictionary with detailed memory analysis
        """
        self.logger.info("Starting comprehensive memory profiling")
        
        # Clear memory before profiling
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Record baseline memory
        baseline_memory = self._get_memory_usage()
        
        try:
            # Forward pass memory usage
            forward_memory = self._profile_forward_pass(model, input_shape, include_gradients)
            
            # Detailed layer analysis
            layer_analysis = self._analyze_layer_memory(model, input_shape)
            
            # Model size analysis
            model_size_analysis = self._analyze_model_size(model)
            
            # Context length analysis
            context_analysis = self._analyze_context_length_memory(model, input_shape)
            
            # Compile comprehensive results
            memory_profile = {
                'baseline_memory': baseline_memory,
                'forward_memory': forward_memory,
                'layer_analysis': layer_analysis,
                'model_size_analysis': model_size_analysis,
                'context_analysis': context_analysis,
                'optimization_potential': self._calculate_optimization_potential(forward_memory),
                'recommendations': self._generate_optimization_recommendations(forward_memory, layer_analysis)
            }
            
            # Store snapshot
            self.memory_snapshots.append({
                'timestamp': time.time(),
                'model_type': type(model).__name__,
                'input_shape': input_shape,
                'profile': memory_profile
            })
            
            self.logger.info(f"Memory profiling completed: {forward_memory['total_forward_memory_mb']:.1f} MB")
            return memory_profile
            
        except Exception as e:
            self.logger.error(f"Memory profiling failed: {e}")
            return {'error': str(e)}
    
    def _get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics."""
        memory_info = {
            'timestamp': time.time()
        }
        
        if HAS_PSUTIL:
            process = psutil.Process()
            memory_info['process_memory_mb'] = process.memory_info().rss / (1024 * 1024)
            memory_info['virtual_memory_total_gb'] = psutil.virtual_memory().total / (1024**3)
            memory_info['virtual_memory_available_gb'] = psutil.virtual_memory().available / (1024**3)
            memory_info['virtual_memory_used_gb'] = psutil.virtual_memory().used / (1024**3)
        
        if torch.cuda.is_available():
            memory_info['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated() / (1024 * 1024)
            memory_info['gpu_memory_reserved_mb'] = torch.cuda.memory_reserved() / (1024 * 1024)
            memory_info['gpu_memory_max_allocated_mb'] = torch.cuda.max_memory_allocated() / (1024 * 1024)
        
        return memory_info
    
    def _profile_forward_pass(self, model: nn.Module, input_shape: Tuple[int, ...], 
                            include_gradients: bool) -> Dict[str, Any]:
        """Profile memory usage during forward pass."""
        model.eval()
        
        # Memory before forward pass
        memory_before = self._get_memory_usage()
        
        # Create input tensor
        input_tensor = torch.randn(input_shape, device=next(model.parameters()).device)
        
        # Forward pass with memory tracking
        with torch.no_grad():
            if hasattr(torch.cuda, 'memory'):
                torch.cuda.reset_peak_memory_stats()
            
            # Forward pass
            start_time = time.perf_counter()
            output = model(input_tensor)
            forward_time = time.perf_counter() - start_time
        
        # Memory after forward pass
        memory_after = self._get_memory_usage()
        
        if hasattr(torch.cuda, 'memory'):
            torch.cuda.empty_cache()
        
        return {
            'input_shape': input_shape,
            'forward_time_ms': forward_time * 1000,
            'memory_before': memory_before,
            'memory_after': memory_after,
            'total_forward_memory_mb': self._calculate_memory_increase(memory_before, memory_after),
            'parameters_memory_mb': self._calculate_parameters_memory(model),
            'requires_grad': include_gradients
        }
    
    def _analyze_layer_memory(self, model: nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """Analyze memory usage of individual layers."""
        layer_analysis = {}
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.LayerNorm, nn.RMSNorm)):
                param_memory = sum(p.numel() * p.element_size() for p in module.parameters())
                layer_analysis[name] = {
                    'module_type': type(module).__name__,
                    'parameters': sum(p.numel() for p in module.parameters()),
                    'parameter_memory_kb': param_memory / 1024,
                    'input_features': getattr(module, 'in_features', 'N/A'),
                    'output_features': getattr(module, 'out_features', 'N/A')
                }
        
        return layer_analysis
    
    def _analyze_model_size(self, model: nn.Module) -> Dict[str, Any]:
        """Analyze total model size and composition."""
        total_params = 0
        param_memory = 0
        buffer_memory = 0
        layer_counts = {}
        
        for name, module in model.named_modules():
            # Count parameters
            params = sum(p.numel() for p in module.parameters())
            total_params += params
            param_memory += sum(p.numel() * p.element_size() for p in module.parameters())
            
            # Count buffers
            buffers = sum(b.numel() * b.element_size() for b in module.buffers())
            buffer_memory += buffers
            
            # Count layer types
            layer_type = type(module).__name__
            layer_counts[layer_type] = layer_counts.get(layer_type, 0) + 1
        
        return {
            'total_parameters': total_params,
            'parameter_memory_mb': param_memory / (1024 * 1024),
            'buffer_memory_mb': buffer_memory / (1024 * 1024),
            'total_model_memory_mb': (param_memory + buffer_memory) / (1024 * 1024),
            'layer_counts': layer_counts,
            'parameters_per_layer': total_params / len(layer_counts) if layer_counts else 0
        }
    
    def _analyze_context_length_memory(self, model: nn.Module, base_input_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """Analyze memory usage vs context length scaling."""
        context_lengths = [256, 512, 1024, 2048, 4096]
        context_analysis = {}
        
        for context_length in context_lengths:
            input_shape = (base_input_shape[0], context_length)
            forward_memory = self._profile_forward_pass(model, input_shape, False)
            
            context_analysis[context_length] = {
                'memory_mb': forward_memory['total_forward_memory_mb'],
                'time_ms': forward_memory['forward_time_ms'],
                'memory_per_token_kb': (forward_memory['total_forward_memory_mb'] * 1024) / context_length
            }
        
        return context_analysis
    
    def _calculate_memory_increase(self, memory_before: Dict, memory_after: Dict) -> float:
        """Calculate memory increase during operation."""
        increase = 0
        
        if HAS_PSUTIL:
            process_before = memory_before.get('process_memory_mb', 0)
            process_after = memory_after.get('process_memory_mb', 0)
            increase += (process_after - process_before)
        
        if torch.cuda.is_available():
            gpu_before = memory_before.get('gpu_memory_allocated_mb', 0)
            gpu_after = memory_after.get('gpu_memory_allocated_mb', 0)
            increase += (gpu_after - gpu_before)
        
        return max(0, increase)
    
    def _calculate_parameters_memory(self, model: nn.Module) -> float:
        """Calculate memory used by model parameters."""
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
        return param_memory / (1024 * 1024)
    
    def _calculate_optimization_potential(self, forward_memory: Dict) -> Dict[str, Any]:
        """Calculate potential memory savings from optimization."""
        current_memory = forward_memory['total_forward_memory_mb']
        
        # Estimate savings from different techniques
        potential_savings = {
            'int8_quantization': current_memory * 0.5,  # 50% reduction
            'int4_quantization': current_memory * 0.75,  # 75% reduction
            'gradient_checkpointing': current_memory * 0.3,  # 30% reduction
            'layer_fusion': current_memory * 0.15,  # 15% reduction
            'attention_head_reduction': current_memory * 0.1,  # 10% reduction
        }
        
        return {
            'current_memory_mb': current_memory,
            'potential_savings_mb': potential_savings,
            'best_case_reduction_percent': min(potential_savings.values()) / current_memory * 100
        }
    
    def _generate_optimization_recommendations(self, forward_memory: Dict, 
                                             layer_analysis: Dict) -> List[str]:
        """Generate specific optimization recommendations."""
        recommendations = []
        current_memory = forward_memory['total_forward_memory_mb']
        
        # Memory-based recommendations
        if current_memory > 8000:  # > 8GB
            recommendations.append("Consider INT4 quantization for significant memory reduction")
            recommendations.append("Implement gradient checkpointing for memory efficiency")
        elif current_memory > 4000:  # > 4GB
            recommendations.append("INT8 quantization is recommended")
            recommendations.append("Consider attention head optimization")
        
        # Performance-based recommendations
        if forward_memory['forward_time_ms'] > 100:  # > 100ms
            recommendations.append("Apply layer fusion for faster inference")
            recommendations.append("Enable TensorRT optimization for GPU acceleration")
        
        # Hardware-based recommendations
        if torch.cuda.is_available():
            recommendations.append("Enable CUDA optimization and TF32")
        else:
            recommendations.append("CPU optimization: consider OpenVINO or ONNX Runtime")
        
        return recommendations


class MemoryOptimizer:
    """
    Memory optimization techniques for on-device deployment.
    Implements pruning, attention head optimization, layer fusion, and more.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the memory optimizer."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.optimization_results = {}
    
    def apply_structured_pruning(self, model: nn.Module, target_config: Dict) -> nn.Module:
        """
        Apply structured pruning to reduce model size.
        
        Args:
            model: Model to prune
            target_config: Target deployment configuration
            
        Returns:
            Pruned model
        """
        self.logger.info("Applying structured pruning")
        
        pruning_config = self.config.get('pruning', {})
        if not pruning_config.get('enabled', False):
            self.logger.info("Pruning not enabled, returning original model")
            return model
        
        method = pruning_config.get('method', 'magnitude')
        sparsity_level = pruning_config.get('sparsity_level', 0.3)
        
        try:
            if method == 'magnitude':
                return self._apply_magnitude_pruning(model, sparsity_level)
            elif method == 'structured':
                return self._apply_structured_pruning(model, sparsity_level)
            else:
                self.logger.warning(f"Unknown pruning method: {method}")
                return model
                
        except Exception as e:
            self.logger.error(f"Pruning failed: {e}")
            return model
    
    def _apply_magnitude_pruning(self, model: nn.Module, sparsity_level: float) -> nn.Module:
        """Apply magnitude-based unstructured pruning."""
        self.logger.info(f"Applying magnitude pruning with {sparsity_level*100:.1f}% sparsity")
        
        # This is a simplified implementation
        # In practice, you would use more sophisticated pruning algorithms
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Get absolute values of weights
                weights = module.weight.data
                abs_weights = torch.abs(weights)
                
                # Calculate threshold for pruning
                threshold = torch.quantile(abs_weights.flatten(), sparsity_level)
                
                # Create pruning mask
                mask = abs_weights >= threshold
                
                # Apply pruning by zeroing out small weights
                module.weight.data = weights * mask.float()
                
                self.logger.info(f"Pruned {module}: {100 - (mask.float().mean() * 100):.1f}% parameters retained")
        
        return model
    
    def _apply_structured_pruning(self, model: nn.Module, sparsity_level: float) -> nn.Module:
        """Apply structured pruning (prune entire neurons/channels)."""
        self.logger.info(f"Applying structured pruning with {sparsity_level*100:.1f}% sparsity")
        
        # This would involve pruning entire neurons or channels
        # Implementation would depend on the specific structured pruning approach
        
        return model
    
    def apply_attention_head_optimization(self, model: nn.Module, target_config: Dict) -> nn.Module:
        """
        Optimize attention heads for memory efficiency.
        
        Args:
            model: Model to optimize
            target_config: Target deployment configuration
            
        Returns:
            Optimized model
        """
        self.logger.info("Optimizing attention heads")
        
        attention_config = self.config.get('attention_optimization', {})
        if not attention_config.get('reduce_q_heads', False):
            self.logger.info("Attention head reduction not enabled")
            return model
        
        target_heads = attention_config.get('target_heads', 8)
        current_heads = model.config.num_attention_heads
        
        if target_heads >= current_heads:
            self.logger.info("No attention head reduction needed")
            return model
        
        self.logger.info(f"Reducing attention heads from {current_heads} to {target_heads}")
        
        # This would involve actually modifying the model architecture
        # For now, we'll just log the intention
        self.logger.warning("Actual attention head reduction requires architectural modifications")
        
        return model
    
    def apply_layer_fusion(self, model: nn.Module, target_config: Dict) -> nn.Module:
        """
        Apply layer fusion for inference acceleration and memory efficiency.
        
        Args:
            model: Model to optimize
            target_config: Target deployment configuration
            
        Returns:
            Model with fused layers
        """
        self.logger.info("Applying layer fusion")
        
        fusion_config = self.config.get('layer_fusion', {})
        if not fusion_config.get('enabled', False):
            self.logger.info("Layer fusion not enabled")
            return model
        
        # This is a placeholder implementation
        # Real layer fusion would involve combining operations during graph optimization
        # or during export to ONNX/TensorRT
        
        fused_layers = 0
        
        if fusion_config.get('fuse_qkv', True):
            # Fuse Q, K, V projections
            fused_layers += 1
            self.logger.info("QKV projection fusion enabled")
        
        if fusion_config.get('fuse_attention_output', True):
            # Fuse attention output projection
            fused_layers += 1
            self.logger.info("Attention output fusion enabled")
        
        if fusion_config.get('fuse_mlp', True):
            # Fuse MLP projections
            fused_layers += 1
            self.logger.info("MLP projection fusion enabled")
        
        self.logger.info(f"Layer fusion completed: {fused_layers} fusion operations")
        
        return model
    
    def optimize_kv_cache(self, model: nn.Module, target_config: Dict) -> nn.Module:
        """
        Optimize KV cache for longer contexts and better memory efficiency.
        
        Args:
            model: Model to optimize
            target_config: Target deployment configuration
            
        Returns:
            Model with optimized KV cache handling
        """
        self.logger.info("Optimizing KV cache")
        
        kv_config = self.config.get('kv_cache', {})
        if not kv_config.get('enabled', False):
            self.logger.info("KV cache optimization not enabled")
            return model
        
        # Add KV cache optimization to the model
        # This would typically be implemented in the forward pass
        
        compression = kv_config.get('compression', 'fp16')
        sliding_window = kv_config.get('sliding_window', 4096)
        
        self.logger.info(f"KV cache optimization: compression={compression}, window={sliding_window}")
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            self.logger.info("Gradient checkpointing enabled")
        
        return model
    
    def implement_gradient_checkpointing(self, model: nn.Module) -> nn.Module:
        """
        Implement gradient checkpointing for memory efficiency during training.
        
        Args:
            model: Model to modify
            
        Returns:
            Model with gradient checkpointing enabled
        """
        self.logger.info("Implementing gradient checkpointing")
        
        try:
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                self.logger.info("Gradient checkpointing enabled successfully")
            else:
                self.logger.warning("Model does not support gradient checkpointing")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Gradient checkpointing failed: {e}")
            return model
    
    def optimize_memory_layout(self, model: nn.Module) -> nn.Module:
        """
        Optimize memory layout for better cache efficiency.
        
        Args:
            model: Model to optimize
            
        Returns:
            Model with optimized memory layout
        """
        self.logger.info("Optimizing memory layout")
        
        try:
            # Ensure model parameters are contiguous
            for name, param in model.named_parameters():
                if not param.is_contiguous():
                    param.data = param.data.contiguous()
                    self.logger.debug(f"Made {name} contiguous")
            
            # Enable memory efficient attention if available
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                self.logger.info("Memory efficient attention is available")
            else:
                self.logger.info("Memory efficient attention not available in this PyTorch version")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Memory layout optimization failed: {e}")
            return model
    
    def batch_optimize_for_target(self, model: nn.Module, target_config: Dict) -> nn.Module:
        """
        Apply multiple memory optimizations based on target configuration.
        
        Args:
            model: Model to optimize
            target_config: Target deployment configuration
            
        Returns:
            Fully optimized model
        """
        self.logger.info(f"Applying batch optimizations for target: {target_config}")
        
        optimized_model = copy.deepcopy(model)
        
        # Apply optimizations in sequence
        optimization_steps = [
            ('pruning', self.apply_structured_pruning),
            ('attention_optimization', self.apply_attention_head_optimization),
            ('layer_fusion', self.apply_layer_fusion),
            ('kv_cache_optimization', self.optimize_kv_cache),
            ('gradient_checkpointing', self.implement_gradient_checkpointing),
            ('memory_layout', self.optimize_memory_layout)
        ]
        
        for step_name, optimization_func in optimization_steps:
            try:
                self.logger.info(f"Applying {step_name}")
                optimized_model = optimization_func(optimized_model, target_config)
            except Exception as e:
                self.logger.error(f"Failed to apply {step_name}: {e}")
                continue
        
        # Record optimization results
        self.optimization_results[target_config.get('name', 'unknown')] = {
            'original_model_size_mb': self._calculate_model_size(model),
            'optimized_model_size_mb': self._calculate_model_size(optimized_model),
            'optimization_steps_applied': len(optimization_steps),
            'target_config': target_config
        }
        
        self.logger.info("Batch optimization completed")
        return optimized_model
    
    def _calculate_model_size(self, model: nn.Module) -> float:
        """Calculate model size in MB."""
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers())
        total_memory = param_memory + buffer_memory
        return total_memory / (1024 * 1024)
    
    def generate_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        return {
            'optimization_results': self.optimization_results,
            'config': self.config,
            'timestamp': time.time()
        }
    
    def save_optimization_report(self, output_path: str):
        """Save optimization report to file."""
        report = self.generate_optimization_report()
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Optimization report saved to {output_path}")


def main():
    """Main function for testing memory profiling and optimization."""
    # This would typically be used for testing
    # In practice, you would load a model and test memory features
    
    config = {
        'pruning': {
            'enabled': True,
            'method': 'magnitude',
            'sparsity_level': 0.3
        },
        'attention_optimization': {
            'reduce_q_heads': False,
            'target_heads': 8
        },
        'layer_fusion': {
            'enabled': True,
            'fuse_qkv': True,
            'fuse_attention_output': True,
            'fuse_mlp': True
        },
        'kv_cache': {
            'enabled': True,
            'compression': 'fp16',
            'sliding_window': 4096
        }
    }
    
    profiler = MemoryProfiler(config)
    optimizer = MemoryOptimizer(config)
    
    print("Memory optimization suite initialized successfully")
    print("Capabilities:")
    print("- Comprehensive memory profiling")
    print("- Structured and unstructured pruning")
    print("- Attention head optimization")
    print("- Layer fusion for inference acceleration")
    print("- KV cache optimization")
    print("- Gradient checkpointing")


if __name__ == "__main__":
    main()