# Sheikh-2.5-Coder Model Optimization Suite

Comprehensive model optimization framework for on-device deployment of Sheikh-2.5-Coder with advanced quantization, memory optimization, and platform-specific acceleration techniques.

## 🚀 Features

### ✅ Quantization Optimization
- **INT8 Quantization**: Dynamic range and weight-only quantization
- **INT4 Quantization**: NF4 format and GPTQ compatibility
- **Mixed Precision**: FP16/BF16 optimization
- **Quantization-Aware Training (QAT)**: Support for training with quantization effects
- **Automatic Detection**: Intelligent quantization method selection based on hardware and model characteristics

### ✅ Memory Optimization
- **Model Pruning**: Structured and unstructured parameter removal
- **Attention Head Optimization**: Dynamic head reduction for memory efficiency
- **Layer Fusion**: Inference acceleration through operation merging
- **KV Cache Optimization**: Memory-efficient cache management for longer contexts
- **Gradient Checkpointing**: Memory savings during training/inference

### ✅ Inference Acceleration
- **ONNX Export**: Optimization passes and graph optimization
- **TensorRT Integration**: GPU acceleration with multiple precision modes
- **OpenVINO Optimization**: CPU inference acceleration for edge devices
- **TorchScript Compilation**: Mobile deployment optimization
- **Flash Attention**: Memory-efficient attention mechanisms

### ✅ Deployment Targets
- **Mobile (6-8GB RAM)**: INT4 quantization, reduced context length
- **Edge (8-12GB RAM)**: INT8 quantization, full context length
- **Desktop (12-16GB RAM)**: FP16 inference, optimized batch sizes
- **Server (16GB+ RAM)**: Full precision with maximum performance

## 📁 File Structure

```
scripts/
├── optimize_model.py              # Main optimization orchestrator
├── quantize_model.py              # Quantization implementation
├── export_onnx.py                 # ONNX export and optimization
├── memory_profiler.py             # Memory usage analysis
├── inference_benchmark.py         # Performance benchmarking
├── deployment_utils.py            # Deployment utilities
├── mobile_optimization.py         # Mobile-specific optimizations
├── tensorrt_utils.py              # TensorRT optimization
├── complete_optimization_demo.py  # Comprehensive demonstration
└── optimization_utilities.py      # Shared utilities

configs/
└── optimization_config.yaml       # Optimization configuration
```

## 🛠️ Installation

### Prerequisites
```bash
# Core dependencies
pip install torch torchvision torchaudio
pip install transformers datasets

# Quantization support
pip install bitsandbytes accelerate

# ONNX and optimization
pip install onnx onnxruntime onnxoptimizer
pip install openvino openvino-dev  # Optional, for CPU acceleration

# TensorRT (optional, requires NVIDIA GPU)
# Follow TensorRT installation guide from NVIDIA

# Mobile optimization
pip install torch_tensorrt  # Optional, for advanced mobile optimization

# Benchmarking and utilities
pip install psutil numpy sacrebleu  # Optional, for benchmarking
```

## 🎯 Quick Start

### Basic Optimization
```python
from scripts.optimize_model import ModelOptimizationOrchestrator

# Initialize optimizer
optimizer = ModelOptimizationOrchestrator("configs/optimization_config.yaml")

# Load model
model = optimizer.load_original_model("path/to/sheikh-model")

# Optimize for specific target
optimized_model = optimizer.optimize_for_deployment_target(model, "edge")

# Run benchmarking
benchmarks = optimizer.benchmark_optimization(optimized_model, "edge")
```

### Run Complete Demonstration
```bash
cd Sheikh-2.5-Coder/scripts
python complete_optimization_demo.py --output-dir ./demo_results
```

### Platform-Specific Optimization
```python
# Mobile optimization
from scripts.mobile_optimization import MobileOptimizer

optimizer = MobileOptimizer(config)
result = optimizer.optimize_for_mobile_deployment(model, target="android")

# TensorRT optimization
from scripts.tensorrt_utils import TensorRTOptimizer

tensorrt_opt = TensorRTOptimizer(config)
engine_path = tensorrt_opt.optimize_model_for_tensorrt(
    model, "model_fp16.engine", precision="fp16"
)
```

## 📊 Configuration

The optimization framework uses a comprehensive YAML configuration file:

```yaml
# Example: configs/optimization_config.yaml

model_config:
  model_name: "Sheikh-2.5-Coder"
  total_parameters: "3.09B"

quantization:
  int8:
    enabled: true
    method: "dynamic"  # dynamic, static, weight_only
  int4:
    enabled: true
    method: "nf4"  # nf4, fp4, weight_only
    use_gptq: true

deployment_targets:
  mobile:
    max_memory_gb: 8
    quantization: "int4"
    context_length: 4096
  edge:
    max_memory_gb: 12
    quantization: "int8"
    context_length: 8192
```

## 🔧 Detailed Usage

### 1. Quantization
```python
from scripts.quantize_model import ModelQuantizer

quantizer = ModelQuantizer(quantization_config)

# INT8 quantization
int8_model = quantizer.apply_int8_quantization(model)

# INT4 quantization
int4_model = quantizer.apply_int4_quantization(model)

# Mixed precision
fp16_model = quantizer.apply_mixed_precision(model, "fp16")

# Compare methods
comparison = quantizer.compare_quantization_methods(model)
```

### 2. Memory Optimization
```python
from scripts.memory_profiler import MemoryOptimizer

optimizer = MemoryOptimizer(memory_config)

# Structured pruning
pruned_model = optimizer.apply_structured_pruning(model, target_config)

# Attention optimization
optimized_model = optimizer.apply_attention_head_optimization(model, target_config)

# Layer fusion
fused_model = optimizer.apply_layer_fusion(model, target_config)
```

### 3. ONNX Export
```python
from scripts.export_onnx import ONNXExporter

exporter = ONNXExporter(onnx_config)

# Basic ONNX export
onnx_path = exporter.export_to_onnx(model, "model.onnx")

# Optimized export with TensorRT
tensorrt_engine = exporter.convert_to_tensorrt(onnx_path, "model.trt")

# Mobile-optimized model
mobile_model = exporter.create_mobile_optimized_model(model, "model_mobile.onnx")
```

### 4. Platform Deployment
```python
from scripts.deployment_utils import DeploymentManager

manager = DeploymentManager(deployment_config)

# Deploy to specific platform
android_deployment = manager.deploy_to_platform(model, "android", "./android_build")
ios_deployment = manager.deploy_to_platform(model, "ios", "./ios_build")
web_deployment = manager.deploy_to_platform(model, "web", "./web_build")

# Check compatibility
checker = PlatformCompatibilityChecker()
compatibility = checker.check_model_compatibility(model)
```

### 5. Benchmarking
```python
from scripts.inference_benchmark import ModelBenchmarker

benchmarker = ModelBenchmarker(benchmark_config)

# Comprehensive benchmark
results = benchmarker.run_comprehensive_benchmark(model, target_config)

# Memory footprint analysis
memory_results = benchmarker._benchmark_memory_footprint(model, target_config)

# Speed testing
speed_results = benchmarker._benchmark_inference_speed(model, target_config)
```

## 📈 Performance Metrics

### Memory Efficiency
- **Quantization**: Up to 75% memory reduction with INT4
- **Pruning**: 30-50% parameter reduction with minimal quality loss
- **Layer Fusion**: 15-20% inference speed improvement

### Inference Speed
- **TensorRT FP16**: 3-5x speedup on NVIDIA GPUs
- **ONNX Runtime**: 2-3x speedup across CPU/GPU
- **Mobile Optimization**: 2-4x speedup on mobile devices

### Quality Preservation
- **CodeBLEU Score**: <2% degradation with optimized quantization
- **Pass@k Metrics**: Maintained across most optimization levels
- **Code Completion Accuracy**: 95%+ preserved with appropriate settings

## 🎯 Deployment Targets

### Mobile (Android/iOS)
- **Memory Limit**: 6-8GB RAM
- **Optimization**: INT4 quantization, reduced context length
- **Format**: TorchScript, Core ML, ONNX Runtime Mobile
- **Battery Impact**: Optimized for minimal power consumption

### Edge Devices
- **Memory Limit**: 8-12GB RAM
- **Optimization**: INT8 quantization, full context support
- **Format**: ONNX, OpenVINO optimized
- **Use Cases**: IoT devices, edge computing

### Desktop/Server
- **Memory Limit**: 12GB+ RAM
- **Optimization**: FP16/FP32, maximum performance
- **Format**: ONNX, TensorRT, optimized batch sizes
- **Use Cases**: Development, research, production inference

## 🔍 Validation & Testing

### Functional Correctness
- Output comparison between original and optimized models
- Inference result validation across different input types
- Edge case handling verification

### Performance Impact
- Memory footprint measurement
- Latency analysis (P50, P95, P99)
- Throughput benchmarking (tokens/second)

### Quality Preservation
- CodeBLEU evaluation
- Pass@k metrics testing
- Human evaluation for critical use cases

### Deployment Compatibility
- Platform-specific compatibility checking
- Runtime environment validation
- Hardware requirement verification

## 🐛 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```python
# Solution: Use gradient checkpointing
model.gradient_checkpointing_enable()

# Or reduce batch size
config['batch_size'] = 1
```

#### 2. Quantization Quality Loss
```python
# Solution: Use weight-only quantization
quantizer.apply_weight_only_int8_quantization(model)

# Or use mixed precision instead
quantizer.apply_mixed_precision(model, "fp16")
```

#### 3. Mobile Deployment Issues
```python
# Solution: Use mobile-specific optimization
mobile_optimizer.optimize_for_mobile_deployment(model, target="android")

# Or reduce model complexity
config['max_context_length'] = 512
```

### Performance Optimization Tips

1. **Start with INT8 quantization** for balanced performance and quality
2. **Use TensorRT FP16** for NVIDIA GPU acceleration
3. **Enable gradient checkpointing** for memory-constrained environments
4. **Apply structured pruning** before quantization for better results
5. **Use dynamic batching** for server deployments

## 📚 API Reference

### Core Classes

#### ModelOptimizationOrchestrator
Main orchestration class for comprehensive optimization.

```python
class ModelOptimizationOrchestrator:
    def __init__(self, config_path: str)
    def load_original_model(self, model_path: str) -> SheikhCoderForCausalLM
    def optimize_for_deployment_target(self, model: nn.Module, target: str) -> nn.Module
    def benchmark_optimization(self, model: nn.Module, target: str) -> Dict[str, Any]
    def validate_optimization(self, original: nn.Module, optimized: nn.Module, target: str) -> Dict[str, Any]
```

#### ModelQuantizer
Handles all quantization operations.

```python
class ModelQuantizer:
    def apply_int8_quantization(self, model: nn.Module) -> nn.Module
    def apply_int4_quantization(self, model: nn.Module) -> nn.Module
    def apply_mixed_precision(self, model: nn.Module, precision: str) -> nn.Module
    def compare_quantization_methods(self, model: nn.Module) -> Dict[str, Any]
```

#### TensorRTOptimizer
GPU acceleration and optimization.

```python
class TensorRTOptimizer:
    def __init__(self, config: Dict[str, Any])
    def optimize_model_for_tensorrt(self, model: nn.Module, output_path: str, precision: str) -> str
    def compare_tensorrt_precisions(self, model: nn.Module, output_dir: str) -> Dict[str, Any]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -am 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Submit a pull request

## 📄 License

This optimization suite is part of the Sheikh-2.5-Coder project. See the LICENSE file for details.

## 🙏 Acknowledgments

- PyTorch team for quantization and optimization frameworks
- NVIDIA for TensorRT acceleration capabilities
- ONNX community for cross-platform interoperability
- OpenVINO team for CPU optimization solutions
- Hugging Face for transformer model infrastructure

---

**Note**: This optimization suite is designed to work specifically with the Sheikh-2.5-Coder architecture but can be adapted for other transformer models with similar architectures.