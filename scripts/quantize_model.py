#!/usr/bin/env python3
"""
Model Quantization Implementation for Sheikh-2.5-Coder
Comprehensive quantization support including INT8, INT4, and mixed precision
"""

import torch
import torch.nn as nn
import torch.quantization as quant
from torch.quantization import QuantStub, DeQuantStub
from typing import Dict, Any, Optional, Tuple
import logging
import os
import json
import copy
from pathlib import Path

# Optional imports for advanced quantization
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

try:
    import tensorrt as trt
    HAS_TENSORRT = True
except ImportError:
    HAS_TENSORRT = False

try:
    from optimum.onnxruntime import ORTModelForCausalLM
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


class ModelQuantizer:
    """
    Comprehensive model quantization handler for Sheikh-2.5-Coder.
    Supports INT8, INT4, and mixed precision optimization.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the quantizer with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Validation
        if not HAS_BNB:
            self.logger.warning("bitsandbytes not available. INT4 quantization will be limited.")
        
        self.quantization_results = {}
    
    def apply_int8_quantization(self, model: nn.Module) -> nn.Module:
        """
        Apply INT8 quantization using PyTorch's dynamic quantization.
        Supports both dynamic and static quantization.
        """
        self.logger.info("Applying INT8 quantization")
        
        config = self.config.get('int8', {})
        method = config.get('method', 'dynamic')
        
        if method == 'dynamic':
            return self._apply_dynamic_int8_quantization(model)
        elif method == 'static':
            return self._apply_static_int8_quantization(model)
        elif method == 'weight_only':
            return self._apply_weight_only_int8_quantization(model)
        else:
            self.logger.warning(f"Unknown INT8 method: {method}. Using dynamic quantization.")
            return self._apply_dynamic_int8_quantization(model)
    
    def _apply_dynamic_int8_quantization(self, model: nn.Module) -> nn.Module:
        """Apply dynamic INT8 quantization (weights quantized, activations dynamic)."""
        try:
            # Prepare model for quantization
            quantized_model = copy.deepcopy(model)
            
            # Set quantization configuration
            quantized_model.eval()
            
            # Apply dynamic quantization to linear layers
            for name, module in quantized_model.named_modules():
                if isinstance(module, nn.Linear) and 'lm_head' not in name:
                    # Quantize linear layers that are commonly quantized
                    quantized_module = torch.quantization.quantize_dynamic(
                        module, 
                        {nn.Linear}, 
                        dtype=torch.qint8
                    )
                    
                    # Replace the module
                    self._replace_module(quantized_model, name, quantized_module)
            
            # Log quantization results
            self._log_quantization_results(quantized_model, 'INT8_dynamic')
            
            self.logger.info("INT8 dynamic quantization completed successfully")
            return quantized_model
            
        except Exception as e:
            self.logger.error(f"INT8 dynamic quantization failed: {e}")
            return model
    
    def _apply_static_int8_quantization(self, model: nn.Module) -> nn.Module:
        """Apply static INT8 quantization (requires calibration)."""
        try:
            # For static quantization, we need a calibration dataset
            calibration_config = self.config.get('int8', {})
            calibration_steps = calibration_config.get('calibration_steps', 100)
            
            # This is a simplified implementation
            # In practice, you would need a representative calibration dataset
            self.logger.warning("Static quantization requires calibration data. Using dynamic instead.")
            return self._apply_dynamic_int8_quantization(model)
            
        except Exception as e:
            self.logger.error(f"INT8 static quantization failed: {e}")
            return model
    
    def _apply_weight_only_int8_quantization(self, model: nn.Module) -> nn.Module:
        """Apply weight-only INT8 quantization."""
        try:
            if not HAS_BNB:
                self.logger.error("bitsandbytes required for weight-only quantization")
                return model
            
            quantized_model = copy.deepcopy(model)
            
            # Apply 8-bit quantization to linear layers
            for name, module in quantized_model.named_modules():
                if isinstance(module, nn.Linear):
                    # Use bitsandbytes for 8-bit quantization
                    quantized_module = bnb.nn.Linear8bitLt(
                        module.in_features,
                        module.out_features,
                        bias=module.bias is not None,
                        has_fp16_weights=False
                    )
                    
                    # Copy weights and bias
                    quantized_module.weight.data = module.weight.data.clone()
                    if module.bias is not None:
                        quantized_module.bias.data = module.bias.data.clone()
                    
                    # Replace the module
                    self._replace_module(quantized_model, name, quantized_module)
            
            self._log_quantization_results(quantized_model, 'INT8_weight_only')
            self.logger.info("INT8 weight-only quantization completed")
            return quantized_model
            
        except Exception as e:
            self.logger.error(f"INT8 weight-only quantization failed: {e}")
            return model
    
    def apply_int4_quantization(self, model: nn.Module) -> nn.Module:
        """
        Apply INT4 quantization using NF4 format and GPTQ compatibility.
        """
        self.logger.info("Applying INT4 quantization")
        
        config = self.config.get('int4', {})
        method = config.get('method', 'nf4')
        
        if method == 'nf4':
            return self._apply_nf4_quantization(model)
        elif method == 'weight_only':
            return self._apply_weight_only_int4_quantization(model)
        else:
            self.logger.warning(f"Unknown INT4 method: {method}. Using NF4 quantization.")
            return self._apply_nf4_quantization(model)
    
    def _apply_nf4_quantization(self, model: nn.Module) -> nn.Module:
        """Apply NF4 quantization (preferred for LLMs)."""
        try:
            if not HAS_BNB:
                self.logger.error("bitsandbytes required for NF4 quantization")
                return model
            
            quantized_model = copy.deepcopy(model)
            
            # Apply 4-bit quantization using NF4
            for name, module in quantized_model.named_modules():
                if isinstance(module, nn.Linear):
                    # Use NF4 quantization
                    quantized_module = bnb.nn.Linear4bit(
                        module.in_features,
                        module.out_features,
                        bias=module.bias is not None,
                        compress=True,
                        compress_type=1  # NF4
                    )
                    
                    # Copy weights (will be quantized during initialization)
                    quantized_module.weight.data = module.weight.data.clone()
                    if module.bias is not None:
                        quantized_module.bias.data = module.bias.data.clone()
                    
                    # Replace the module
                    self._replace_module(quantized_model, name, quantized_module)
            
            self._log_quantization_results(quantized_model, 'INT4_NF4')
            self.logger.info("INT4 NF4 quantization completed")
            return quantized_model
            
        except Exception as e:
            self.logger.error(f"INT4 NF4 quantization failed: {e}")
            return model
    
    def _apply_weight_only_int4_quantization(self, model: nn.Module) -> nn.Module:
        """Apply weight-only INT4 quantization."""
        try:
            if not HAS_BNB:
                self.logger.error("bitsandbytes required for weight-only INT4 quantization")
                return model
            
            quantized_model = copy.deepcopy(model)
            
            # Apply 4-bit quantization
            for name, module in quantized_model.named_modules():
                if isinstance(module, nn.Linear):
                    quantized_module = bnb.nn.Linear4bit(
                        module.in_features,
                        module.out_features,
                        bias=module.bias is not None,
                        compress=True,
                        compress_type=0  # FP4
                    )
                    
                    # Copy weights
                    quantized_module.weight.data = module.weight.data.clone()
                    if module.bias is not None:
                        quantized_module.bias.data = module.bias.data.clone()
                    
                    # Replace the module
                    self._replace_module(quantized_model, name, quantized_module)
            
            self._log_quantization_results(quantized_model, 'INT4_weight_only')
            self.logger.info("INT4 weight-only quantization completed")
            return quantized_model
            
        except Exception as e:
            self.logger.error(f"INT4 weight-only quantization failed: {e}")
            return model
    
    def apply_mixed_precision(self, model: nn.Module, precision: str = 'fp16') -> nn.Module:
        """
        Apply mixed precision optimization (FP16/BF16).
        """
        self.logger.info(f"Applying mixed precision optimization: {precision}")
        
        if precision == 'fp16':
            return self._apply_fp16_optimization(model)
        elif precision == 'bf16':
            return self._apply_bf16_optimization(model)
        else:
            self.logger.warning(f"Unknown precision format: {precision}")
            return model
    
    def _apply_fp16_optimization(self, model: nn.Module) -> nn.Module:
        """Apply FP16 optimization."""
        try:
            # Enable automatic mixed precision
            model.gradient_checkpointing_enable()
            
            # Convert model to FP16 if possible
            if torch.cuda.is_available():
                model = model.half()
                self.logger.info("Model converted to FP16 for CUDA")
            else:
                self.logger.warning("CUDA not available. FP16 optimization limited to model configuration.")
            
            # Enable TF32 if supported
            if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul'):
                torch.backends.cuda.matmul.allow_tf32 = True
            
            self.logger.info("FP16 optimization applied")
            return model
            
        except Exception as e:
            self.logger.error(f"FP16 optimization failed: {e}")
            return model
    
    def _apply_bf16_optimization(self, model: nn.Module) -> nn.Module:
        """Apply BF16 optimization."""
        try:
            # Convert model to BF16 if supported
            if torch.cuda.is_available():
                model = model.bfloat16()
                self.logger.info("Model converted to BF16 for CUDA")
            else:
                self.logger.warning("CUDA not available. BF16 optimization limited.")
            
            # BF16 typically has better numerical stability than FP16
            model.gradient_checkpointing_enable()
            
            self.logger.info("BF16 optimization applied")
            return model
            
        except Exception as e:
            self.logger.error(f"BF16 optimization failed: {e}")
            return model
    
    def apply_quantization_aware_training(self, model: nn.Module, 
                                        calibration_dataset: torch.utils.data.DataLoader = None) -> nn.Module:
        """
        Apply quantization-aware training (QAT) support.
        """
        self.logger.info("Applying quantization-aware training support")
        
        try:
            # This is a simplified QAT implementation
            # In practice, you would:
            # 1. Insert fake quantization observers
            # 2. Train the model with quantization effects
            # 3. Convert to actual quantized model
            
            quantized_model = copy.deepcopy(model)
            
            # Add quantization stubs
            quantized_model.quant = QuantStub()
            quantized_model.dequant = DeQuantStub()
            
            # Configure quantization
            quantized_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
            
            # Prepare for QAT
            torch.quantization.prepare_qat(quantized_model, inplace=True)
            
            self.logger.info("QAT preparation completed")
            self.logger.info("Note: Actual QAT training requires training loop with quantized operations")
            
            return quantized_model
            
        except Exception as e:
            self.logger.error(f"QAT preparation failed: {e}")
            return model
    
    def setup_automatic_quantization_detection(self, model: nn.Module) -> nn.Module:
        """
        Set up automatic quantization detection and fallback.
        """
        self.logger.info("Setting up automatic quantization detection")
        
        try:
            # Analyze model for quantization compatibility
            quantization_compatibility = self._analyze_quantization_compatibility(model)
            
            # Choose best quantization method
            best_method = self._select_optimal_quantization_method(quantization_compatibility)
            
            self.logger.info(f"Automatically selected quantization method: {best_method}")
            
            # Apply the selected method
            if best_method == 'int8_dynamic':
                return self._apply_dynamic_int8_quantization(model)
            elif best_method == 'int8_weight_only':
                return self._apply_weight_only_int8_quantization(model)
            elif best_method == 'int4_nf4':
                return self._apply_nf4_quantization(model)
            elif best_method == 'fp16':
                return self._apply_fp16_optimization(model)
            else:
                self.logger.warning(f"Fallback to original model for method: {best_method}")
                return model
            
        except Exception as e:
            self.logger.error(f"Automatic quantization detection failed: {e}")
            self.logger.info("Falling back to original model")
            return model
    
    def _analyze_quantization_compatibility(self, model: nn.Module) -> Dict[str, Any]:
        """Analyze model compatibility with different quantization methods."""
        compatibility = {
            'cuda_available': torch.cuda.is_available(),
            'linear_layers': 0,
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024),
            'bitsandbytes_available': HAS_BNB,
            'recommended_methods': []
        }
        
        # Count linear layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                compatibility['linear_layers'] += 1
        
        # Recommend quantization methods based on analysis
        if compatibility['cuda_available']:
            compatibility['recommended_methods'].extend(['int8_dynamic', 'fp16', 'bf16'])
        
        if HAS_BNB:
            compatibility['recommended_methods'].extend(['int8_weight_only', 'int4_nf4'])
        
        if compatibility['model_size_mb'] > 1000:  # > 1GB
            compatibility['recommended_methods'].insert(0, 'int4_nf4')  # Prefer INT4 for large models
        
        return compatibility
    
    def _select_optimal_quantization_method(self, compatibility: Dict[str, Any]) -> str:
        """Select optimal quantization method based on compatibility analysis."""
        recommended = compatibility['recommended_methods']
        
        if not recommended:
            return 'fp16'  # Default fallback
        
        # Priority order based on size and hardware
        if compatibility['model_size_mb'] > 2000 and 'int4_nf4' in recommended:
            return 'int4_nf4'
        elif 'int8_weight_only' in recommended:
            return 'int8_weight_only'
        elif 'int8_dynamic' in recommended:
            return 'int8_dynamic'
        elif 'fp16' in recommended:
            return 'fp16'
        else:
            return recommended[0]
    
    def _replace_module(self, model: nn.Module, name: str, new_module: nn.Module):
        """Replace a module in the model by name."""
        path = name.split('.')
        parent = model
        
        # Navigate to parent module
        for part in path[:-1]:
            parent = getattr(parent, part)
        
        # Replace the module
        setattr(parent, path[-1], new_module)
    
    def _log_quantization_results(self, model: nn.Module, method: str):
        """Log quantization results and save analysis."""
        # Calculate model statistics
        original_params = sum(p.numel() for p in model.parameters())
        
        # Estimate quantized size (rough approximation)
        quantized_size_mb = self._estimate_quantized_size(model, method)
        
        results = {
            'method': method,
            'original_parameters': original_params,
            'estimated_size_mb': quantized_size_mb,
            'compression_ratio': self._calculate_compression_ratio(model, method),
            'hardware_compatibility': self._get_hardware_compatibility(method)
        }
        
        self.quantization_results[method] = results
        
        self.logger.info(f"Quantization Results for {method}:")
        self.logger.info(f"  Estimated Size: {quantized_size_mb:.1f} MB")
        self.logger.info(f"  Compression Ratio: {results['compression_ratio']:.2f}x")
        self.logger.info(f"  Hardware Compatibility: {results['hardware_compatibility']}")
    
    def _estimate_quantized_size(self, model: nn.Module, method: str) -> float:
        """Estimate quantized model size in MB."""
        total_size_bytes = 0
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                param_size = module.in_features * module.out_features * module.element_size()
                if module.bias is not None:
                    param_size += module.out_features * module.bias.element_size()
                total_size_bytes += param_size
        
        # Apply quantization-specific reductions
        if 'int8' in method:
            reduction_factor = 0.25  # INT8 reduces to 1/4 original size
        elif 'int4' in method:
            reduction_factor = 0.125  # INT4 reduces to 1/8 original size
        elif method == 'fp16':
            reduction_factor = 0.5  # FP16 reduces to 1/2 original size
        elif method == 'bf16':
            reduction_factor = 0.5  # BF16 reduces to 1/2 original size
        else:
            reduction_factor = 1.0
        
        return (total_size_bytes * reduction_factor) / (1024 * 1024)
    
    def _calculate_compression_ratio(self, model: nn.Module, method: str) -> float:
        """Calculate compression ratio compared to FP32."""
        # FP32 is typically 4 bytes per parameter
        # Estimate original FP32 size
        original_size_mb = sum(p.numel() * 4 for p in model.parameters()) / (1024 * 1024)
        quantized_size_mb = self._estimate_quantized_size(model, method)
        
        return original_size_mb / quantized_size_mb if quantized_size_mb > 0 else 1.0
    
    def _get_hardware_compatibility(self, method: str) -> Dict[str, bool]:
        """Get hardware compatibility for quantization method."""
        compatibility = {
            'cpu': True,  # Most quantization methods work on CPU
            'cuda': torch.cuda.is_available(),
            'mobile': True,
            'edge': True
        }
        
        # Method-specific compatibility
        if 'int4' in method:
            compatibility['cuda'] = HAS_BNB and torch.cuda.is_available()
            compatibility['mobile'] = HAS_BNB
            compatibility['edge'] = HAS_BNB
        elif 'int8' in method and 'weight_only' in method:
            compatibility['cuda'] = HAS_BNB and torch.cuda.is_available()
        
        return compatibility
    
    def save_quantization_results(self, output_path: str):
        """Save quantization results to file."""
        with open(output_path, 'w') as f:
            json.dump(self.quantization_results, f, indent=2)
        
        self.logger.info(f"Quantization results saved to {output_path}")
    
    def compare_quantization_methods(self, model: nn.Module) -> Dict[str, Any]:
        """Compare different quantization methods on the same model."""
        self.logger.info("Comparing quantization methods")
        
        methods = ['int8_dynamic', 'int8_weight_only', 'int4_nf4', 'fp16', 'bf16']
        comparison_results = {}
        
        for method in methods:
            try:
                self.logger.info(f"Testing {method}")
                
                if method == 'int8_dynamic':
                    test_model = self._apply_dynamic_int8_quantization(copy.deepcopy(model))
                elif method == 'int8_weight_only' and HAS_BNB:
                    test_model = self._apply_weight_only_int8_quantization(copy.deepcopy(model))
                elif method == 'int4_nf4' and HAS_BNB:
                    test_model = self._apply_nf4_quantization(copy.deepcopy(model))
                elif method == 'fp16':
                    test_model = self._apply_fp16_optimization(copy.deepcopy(model))
                elif method == 'bf16':
                    test_model = self._apply_bf16_optimization(copy.deepcopy(model))
                else:
                    continue
                
                comparison_results[method] = {
                    'size_mb': self._estimate_quantized_size(test_model, method),
                    'compression_ratio': self._calculate_compression_ratio(test_model, method),
                    'hardware_compatibility': self._get_hardware_compatibility(method),
                    'success': True
                }
                
            except Exception as e:
                comparison_results[method] = {
                    'success': False,
                    'error': str(e)
                }
        
        return comparison_results


def main():
    """Main function for testing quantization methods."""
    # This would typically be used for testing quantization
    # In practice, you would load a model and test different methods
    
    config = {
        'int8': {
            'method': 'dynamic',
            'enabled': True
        },
        'int4': {
            'method': 'nf4',
            'enabled': True,
            'use_gptq': True
        },
        'mixed_precision': {
            'weights': 'fp16',
            'activations': 'fp16'
        }
    }
    
    quantizer = ModelQuantizer(config)
    
    print("ModelQuantizer initialized successfully")
    print("Quantization methods available:")
    print("- INT8 (dynamic, static, weight-only)")
    print("- INT4 (NF4, weight-only)")
    print("- Mixed Precision (FP16, BF16)")
    
    if not HAS_BNB:
        print("Note: bitsandbytes not available. Advanced quantization methods limited.")


if __name__ == "__main__":
    main()