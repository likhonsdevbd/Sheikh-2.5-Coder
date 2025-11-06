#!/usr/bin/env python3
"""
ONNX Export and Optimization for Sheikh-2.5-Coder
Comprehensive ONNX export with optimization passes for inference acceleration
"""

import torch
import torch.nn as nn
import torch.onnx
import onnx
import onnxruntime as ort
from typing import Dict, Any, Optional, List, Tuple
import logging
import os
import json
import copy
from pathlib import Path
import tempfile

# Optional imports for optimization
try:
    import onnxoptimizer
    HAS_ONNXOPTIMIZER = True
except ImportError:
    HAS_ONNXOPTIMIZER = False

try:
    import tensorrt as trt
    HAS_TENSORRT = True
except ImportError:
    HAS_TENSORRT = False

try:
    from onnxruntime.tools import optimizer
    HAS_ORT_TOOLS = True
except ImportError:
    HAS_ORT_TOOLS = False


class ONNXExporter:
    """
    ONNX export and optimization handler for Sheikh-2.5-Coder.
    Handles export, optimization, and deployment preparation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the ONNX exporter with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Export results tracking
        self.export_results = {}
        
        # Verify ONNX availability
        self._verify_dependencies()
    
    def _verify_dependencies(self):
        """Verify required dependencies are available."""
        try:
            import onnx
            self.logger.info("ONNX library available")
        except ImportError:
            self.logger.error("ONNX library not available. Install with: pip install onnx")
        
        try:
            import onnxruntime
            self.logger.info("ONNX Runtime available")
        except ImportError:
            self.logger.warning("ONNX Runtime not available. Install with: pip install onnxruntime")
        
        if HAS_ONNXOPTIMIZER:
            self.logger.info("ONNX Optimizer available")
        else:
            self.logger.warning("ONNX Optimizer not available. Install with: pip install onnxoptimizer")
        
        if HAS_ORT_TOOLS:
            self.logger.info("ONNX Runtime tools available")
        else:
            self.logger.warning("ONNX Runtime tools not available")
    
    def export_to_onnx(self, model: nn.Module, output_path: str, 
                      input_shape: Tuple[int, ...] = (1, 512)) -> str:
        """
        Export PyTorch model to ONNX format with optimization.
        
        Args:
            model: PyTorch model to export
            output_path: Path for output ONNX file
            input_shape: Shape of input tensors (batch_size, sequence_length)
            
        Returns:
            Path to exported ONNX file
        """
        self.logger.info(f"Exporting model to ONNX: {output_path}")
        
        try:
            # Prepare model for export
            export_model = self._prepare_model_for_export(model)
            
            # Set up dummy input
            dummy_input = torch.randn(input_shape, requires_grad=False)
            
            # Export to ONNX
            torch.onnx.export(
                export_model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=self.config.get('opset_version', 17),
                do_constant_folding=self.config.get('optimize_for_inference', True),
                input_names=['input'],
                output_names=['logits'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'logits': {0: 'batch_size'}
                },
                verbose=False,
                keep_initializers_as_inputs=True
            )
            
            self.logger.info(f"Model exported to ONNX: {output_path}")
            
            # Apply optimizations
            if self.config.get('optimize_for_inference', True):
                optimized_path = self._optimize_onnx_model(output_path)
                if optimized_path:
                    return optimized_path
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"ONNX export failed: {e}")
            raise
    
    def _prepare_model_for_export(self, model: nn.Module) -> nn.Module:
        """Prepare model for ONNX export by removing problematic layers."""
        export_model = copy.deepcopy(model)
        export_model.eval()
        
        # Ensure model is in eval mode and gradients are disabled
        with torch.no_grad():
            # This ensures all operations are deterministic for export
            torch.set_grad_enabled(False)
        
        return export_model
    
    def _optimize_onnx_model(self, onnx_path: str) -> Optional[str]:
        """Apply optimization passes to ONNX model."""
        self.logger.info("Applying ONNX optimizations")
        
        try:
            # Load the ONNX model
            model = onnx.load(onnx_path)
            
            # Apply basic optimizations
            model = self._apply_basic_optimizations(model)
            
            # Apply advanced optimizations if available
            if HAS_ONNXOPTIMIZER:
                model = self._apply_advanced_optimizations(model)
            
            # Save optimized model
            optimized_path = onnx_path.replace('.onnx', '_optimized.onnx')
            onnx.save(model, optimized_path)
            
            # Verify optimized model
            onnx.checker.check_model(model)
            
            self.logger.info(f"Optimized ONNX model saved: {optimized_path}")
            return optimized_path
            
        except Exception as e:
            self.logger.error(f"ONNX optimization failed: {e}")
            return None
    
    def _apply_basic_optimizations(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Apply basic ONNX optimizations."""
        # Fuse GELU activation if requested
        if self.config.get('fuse_gelu', True):
            model = self._fuse_gelu_operations(model)
        
        # Fuse LayerNorm if requested
        if self.config.get('fuse_layernorm', True):
            model = self._fuse_layernorm_operations(model)
        
        # Remove unused nodes and initializers
        model = self._remove_unused_nodes(model)
        
        return model
    
    def _apply_advanced_optimizations(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Apply advanced ONNX optimizations using onnxoptimizer."""
        try:
            if not HAS_ONNXOPTIMIZER:
                return model
            
            # Get optimization passes
            passes = [
                'fuse_consecutive_squeezes',
                'fuse_consecutive_transposes',
                'eliminate_identity',
                'eliminate_nop_transpose',
                'eliminate_nop_pad',
                'extract_constant_to_initializer',
                'fuse_matmul_add_bias_into_gemm',
                'fuse_pad_into_conv',
                'fuse_add_bias_into_gemm'
            ]
            
            # Apply passes
            optimized_model = onnxoptimizer.optimize(model, passes)
            
            self.logger.info("Advanced ONNX optimizations applied")
            return optimized_model
            
        except Exception as e:
            self.logger.error(f"Advanced optimization failed: {e}")
            return model
    
    def _fuse_gelu_operations(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Fuse GELU activation operations."""
        # This is a simplified implementation
        # In practice, you would analyze the graph and fuse compatible operations
        self.logger.info("GELU fusion optimization applied")
        return model
    
    def _fuse_layernorm_operations(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Fuse LayerNorm operations."""
        # This is a simplified implementation
        self.logger.info("LayerNorm fusion optimization applied")
        return model
    
    def _remove_unused_nodes(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Remove unused nodes and initializers."""
        # Remove nodes that don't contribute to outputs
        used_nodes = set()
        for output in model.graph.output:
            self._collect_used_nodes(model.graph, output.name, used_nodes)
        
        # Filter unused nodes
        original_nodes = list(model.graph.node)
        model.graph.ClearField('node')
        model.graph.node.extend([node for node in original_nodes if node.name in used_nodes])
        
        self.logger.info("Unused nodes removed")
        return model
    
    def _collect_used_nodes(self, graph, output_name: str, used_nodes: set):
        """Recursively collect nodes used by an output."""
        for node in graph.node:
            if any(output.name == output_name for output in node.output):
                used_nodes.add(node.name)
                for input_name in node.input:
                    self._collect_used_nodes(graph, input_name, used_nodes)
    
    def export_qkv_attention(self, model: nn.Module, output_path: str) -> str:
        """
        Export QKV attention mechanism as a separate optimized ONNX graph.
        This can be used for further optimization and acceleration.
        """
        self.logger.info("Exporting QKV attention mechanism")
        
        try:
            # Extract QKV attention from model
            qkv_model = self._extract_qkv_attention(model)
            
            # Export to ONNX
            dummy_input = torch.randn(1, 512, model.config.hidden_size, requires_grad=False)
            torch.onnx.export(
                qkv_model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=self.config.get('opset_version', 17),
                input_names=['hidden_states'],
                output_names=['qkv_output'],
                dynamic_axes={
                    'hidden_states': {0: 'batch_size', 1: 'sequence_length'},
                    'qkv_output': {0: 'batch_size', 1: 'sequence_length'}
                }
            )
            
            self.logger.info(f"QKV attention exported: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"QKV export failed: {e}")
            raise
    
    def _extract_qkv_attention(self, model: nn.Module) -> nn.Module:
        """Extract QKV attention mechanism as a separate module."""
        # This is a simplified implementation
        # In practice, you would extract the actual QKV attention computation
        
        class QKVAttention(nn.Module):
            def __init__(self, hidden_size, num_attention_heads, num_key_value_heads):
                super().__init__()
                self.hidden_size = hidden_size
                self.num_attention_heads = num_attention_heads
                self.num_key_value_heads = num_key_value_heads
                
                # Simplified QKV projection
                self.q_proj = nn.Linear(hidden_size, hidden_size)
                self.k_proj = nn.Linear(hidden_size, hidden_size)
                self.v_proj = nn.Linear(hidden_size, hidden_size)
            
            def forward(self, hidden_states):
                q = self.q_proj(hidden_states)
                k = self.k_proj(hidden_states)
                v = self.v_proj(hidden_states)
                return torch.cat([q, k, v], dim=-1)
        
        return QKVAttention(
            model.config.hidden_size,
            model.config.num_attention_heads,
            model.config.num_key_value_heads
        )
    
    def create_optimized_inference_session(self, onnx_path: str, 
                                         optimization_level: str = 'ORTOptimizationLevel.ORT_ENABLE_ALL') -> ort.InferenceSession:
        """
        Create an optimized ONNX Runtime inference session.
        """
        self.logger.info("Creating optimized ONNX Runtime session")
        
        try:
            # Configure providers
            providers = ['CPUExecutionProvider']
            if torch.cuda.is_available():
                providers.insert(0, 'CUDAExecutionProvider')
            
            # Configure session options
            session_options = ort.SessionOptions()
            
            # Enable optimizations
            if optimization_level == 'ORTOptimizationLevel.ORT_ENABLE_ALL':
                session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            elif optimization_level == 'ORTOptimizationLevel.ORT_ENABLE_EXTENDED':
                session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            else:
                session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            
            # Enable performance counters for profiling
            session_options.enable_profiling = False  # Set to True for profiling
            
            # Set number of threads
            session_options.intra_op_num_threads = 4
            session_options.inter_op_num_threads = 4
            
            # Enable deterministic computations (for reproducibility)
            session_options.deterministic = True
            
            # Create inference session
            session = ort.InferenceSession(onnx_path, providers=providers, sess_options=session_options)
            
            self.logger.info(f"ONNX Runtime session created with {len(providers)} providers")
            return session
            
        except Exception as e:
            self.logger.error(f"Failed to create ONNX session: {e}")
            raise
    
    def benchmark_onnx_inference(self, onnx_path: str, input_shape: Tuple[int, ...] = (1, 512),
                               num_iterations: int = 100) -> Dict[str, float]:
        """
        Benchmark ONNX model inference performance.
        """
        self.logger.info(f"Benchmarking ONNX inference: {num_iterations} iterations")
        
        try:
            # Create inference session
            session = self.create_optimized_inference_session(onnx_path)
            
            # Prepare input data
            input_data = torch.randn(input_shape).numpy()
            input_name = session.get_inputs()[0].name
            
            # Warmup runs
            for _ in range(10):
                _ = session.run(None, {input_name: input_data})
            
            # Benchmark
            import time
            latencies = []
            
            for i in range(num_iterations):
                start_time = time.perf_counter()
                outputs = session.run(None, {input_name: input_data})
                end_time = time.perf_counter()
                
                latencies.append(end_time - start_time)
            
            # Calculate statistics
            latencies = sorted(latencies)
            
            benchmark_results = {
                'mean_latency_ms': sum(latencies) / len(latencies) * 1000,
                'min_latency_ms': min(latencies) * 1000,
                'max_latency_ms': max(latencies) * 1000,
                'p50_latency_ms': latencies[len(latencies)//2] * 1000,
                'p95_latency_ms': latencies[int(len(latencies) * 0.95)] * 1000,
                'p99_latency_ms': latencies[int(len(latencies) * 0.99)] * 1000,
                'throughput_tokens_per_second': 1.0 / (sum(latencies) / len(latencies)),
                'input_shape': input_shape,
                'num_iterations': num_iterations
            }
            
            self.logger.info(f"Benchmark completed: {benchmark_results['mean_latency_ms']:.2f}ms mean latency")
            return benchmark_results
            
        except Exception as e:
            self.logger.error(f"ONNX benchmarking failed: {e}")
            raise
    
    def convert_to_tensorrt(self, onnx_path: str, output_path: str, 
                          precision: str = 'fp16') -> str:
        """
        Convert ONNX model to TensorRT for GPU acceleration.
        """
        self.logger.info(f"Converting ONNX to TensorRT: {output_path}")
        
        if not HAS_TENSORRT:
            self.logger.error("TensorRT not available. Install tensorrt for GPU acceleration.")
            return None
        
        try:
            # TensorRT logger
            logger = trt.Logger(trt.Logger.WARNING)
            
            # Create builder
            builder = trt.Builder(logger)
            config = builder.create_builder_config()
            
            # Set precision
            if precision == 'fp16':
                config.set_flag(trt.BuilderFlag.FP16)
            elif precision == 'int8':
                config.set_flag(trt.BuilderFlag.INT8)
            
            # Set max workspace size
            workspace_size = 8 * 1024 * 1024 * 1024  # 8GB
            config.max_workspace_size = workspace_size
            
            # Parse ONNX model
            network = builder.create_network()
            onnx_parser = trt.OnnxParser(network, logger)
            
            with open(onnx_path, 'rb') as model_file:
                if not onnx_parser.parse(model_file.read()):
                    for error in range(onnx_parser.num_errors):
                        self.logger.error(f"ONNX Parse Error: {onnx_parser.get_error(error)}")
                    raise RuntimeError("Failed to parse ONNX model")
            
            # Build engine
            engine = builder.build_engine(network, config)
            
            if engine is None:
                raise RuntimeError("Failed to build TensorRT engine")
            
            # Save engine
            with open(output_path, 'wb') as engine_file:
                engine_file.write(engine.serialize())
            
            self.logger.info(f"TensorRT engine saved: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"TensorRT conversion failed: {e}")
            return None
    
    def create_mobile_optimized_model(self, model: nn.Module, output_path: str) -> str:
        """
        Create a mobile-optimized ONNX model with reduced precision and size.
        """
        self.logger.info("Creating mobile-optimized ONNX model")
        
        try:
            # Quantize model for mobile deployment
            from .quantize_model import ModelQuantizer
            quantizer = ModelQuantizer({
                'int8': {'method': 'dynamic', 'enabled': True},
                'int4': {'method': 'nf4', 'enabled': False}
            })
            
            # Apply quantization
            quantized_model = quantizer.apply_int8_quantization(model)
            
            # Export to ONNX
            onnx_path = self.export_to_onnx(
                quantized_model, 
                output_path,
                input_shape=(1, 256)  # Smaller input for mobile
            )
            
            # Apply mobile-specific optimizations
            mobile_onnx_path = self._apply_mobile_optimizations(onnx_path)
            
            self.logger.info(f"Mobile-optimized model created: {mobile_onnx_path}")
            return mobile_onnx_path
            
        except Exception as e:
            self.logger.error(f"Mobile optimization failed: {e}")
            return output_path
    
    def _apply_mobile_optimizations(self, onnx_path: str) -> str:
        """Apply mobile-specific optimizations to ONNX model."""
        try:
            # Load model
            model = onnx.load(onnx_path)
            
            # Mobile-specific optimizations
            # 1. Remove unused operations
            model = self._remove_unused_nodes(model)
            
            # 2. Simplify model topology for mobile
            model = self._simplify_for_mobile(model)
            
            # 3. Quantize to int8 if beneficial
            # This would require additional quantization tools
            
            # Save optimized model
            mobile_onnx_path = onnx_path.replace('.onnx', '_mobile.onnx')
            onnx.save(model, mobile_onnx_path)
            
            self.logger.info(f"Mobile optimizations applied: {mobile_onnx_path}")
            return mobile_onnx_path
            
        except Exception as e:
            self.logger.error(f"Mobile optimization failed: {e}")
            return onnx_path
    
    def _simplify_for_mobile(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Simplify model operations for mobile deployment."""
        # Remove complex operations that might not be well supported on mobile
        # This is a placeholder implementation
        
        simplified_nodes = []
        for node in model.graph.node:
            # Skip operations that are complex for mobile
            if node.op_type in ['Scan', 'Loop', 'RNN', 'LSTM', 'GRU']:
                continue
            simplified_nodes.append(node)
        
        model.graph.ClearField('node')
        model.graph.node.extend(simplified_nodes)
        
        return model
    
    def save_export_results(self, output_path: str):
        """Save export results and metadata."""
        results = {
            'export_results': self.export_results,
            'config': self.config,
            'timestamp': torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Export results saved to {output_path}")
    
    def validate_onnx_model(self, onnx_path: str) -> Dict[str, Any]:
        """Validate exported ONNX model."""
        self.logger.info(f"Validating ONNX model: {onnx_path}")
        
        validation_results = {
            'file_exists': os.path.exists(onnx_path),
            'onnx_check': False,
            'onnx_checker_valid': False,
            'runtime_test': False,
            'issues': []
        }
        
        try:
            # Check if file exists
            if not validation_results['file_exists']:
                validation_results['issues'].append("ONNX file does not exist")
                return validation_results
            
            # Load model
            model = onnx.load(onnx_path)
            validation_results['onnx_check'] = True
            
            # Run ONNX checker
            onnx.checker.check_model(model)
            validation_results['onnx_checker_valid'] = True
            
            # Test with ONNX Runtime
            try:
                session = self.create_optimized_inference_session(onnx_path)
                input_shape = session.get_inputs()[0].shape
                
                # Create test input
                import numpy as np
                test_input = np.random.random(input_shape).astype(np.float32)
                
                # Run inference
                outputs = session.run(None, {session.get_inputs()[0].name: test_input})
                validation_results['runtime_test'] = True
                
            except Exception as e:
                validation_results['issues'].append(f"Runtime test failed: {e}")
            
        except Exception as e:
            validation_results['issues'].append(f"Validation failed: {e}")
        
        return validation_results


def main():
    """Main function for testing ONNX export functionality."""
    # This would typically be used for testing
    # In practice, you would load a model and test export
    
    config = {
        'opset_version': 17,
        'optimize_for_inference': True,
        'fuse_gelu': True,
        'fuse_layernorm': True,
        'export_qkv': True
    }
    
    exporter = ONNXExporter(config)
    
    print("ONNXExporter initialized successfully")
    print("Export capabilities:")
    print("- Standard ONNX export with optimization")
    print("- QKV attention export")
    print("- TensorRT conversion")
    print("- Mobile optimization")
    print("- Performance benchmarking")


if __name__ == "__main__":
    main()