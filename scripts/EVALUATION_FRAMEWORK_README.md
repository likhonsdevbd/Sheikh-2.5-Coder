# Sheikh-2.5-Coder Evaluation Framework

## Overview

This comprehensive evaluation framework provides systematic testing and benchmarking for the Sheikh-2.5-Coder model across multiple dimensions including code generation quality, performance, web development capabilities, and regression detection.

## Components

### 1. Main Evaluation Orchestrator (`evaluate_model.py`)
- **Purpose**: Coordinates all evaluation benchmarks and generates comprehensive reports
- **Features**: 
  - Integrates all evaluation components
  - Creates HTML dashboards and visualizations
  - Generates detailed markdown reports
  - Manages target achievement tracking

### 2. Benchmark Evaluations

#### MMLU Code Evaluation (`mmlu_evaluation.py`)
- **Target**: >60% accuracy on MMLU Code subset
- **Dataset**: `lukaemon/mmlu` with code subset
- **Metrics**: Accuracy, response time, confusion analysis
- **Features**:
  - Multiple choice question answering
  - Programming concept understanding
  - Categorized performance analysis

#### HumanEval Coding Tasks (`humaneval_evaluation.py`)
- **Target**: >40% Pass@1
- **Dataset**: OpenAI HumanEval
- **Metrics**: Pass@1, Pass@k, function correctness, syntax validity
- **Features**:
  - Multi-completion generation for Pass@k calculation
  - Automated function testing
  - Code syntax validation

#### Web Development Tests (`web_dev_tests.py`)
- **Target**: 75% quality score across web technologies
- **Coverage**: JavaScript/TypeScript, React, XML, MDX, CSS
- **Features**:
  - Language-specific quality assessment
  - Best practices compliance checking
  - Component pattern recognition

### 3. Performance Benchmarking (`performance_benchmark.py`)
- **Metrics**: Inference speed, memory usage, context scaling, multi-threading
- **Features**:
  - Hardware utilization monitoring
  - Batch size optimization testing
  - Memory profiling across quantization levels
  - Context length scalability analysis

### 4. Code Quality Assessment (`code_quality_tests.py`)
- **Targets**: >95% syntax validity, >0.65 CodeBLEU score
- **Features**:
  - Multi-language syntax validation
  - Code complexity analysis
  - Best practices compliance
  - CodeBLEU score calculation

### 5. Regression Testing (`regression_testing.py`)
- **Purpose**: Detect performance regressions against baselines
- **Features**:
  - Statistical significance testing
  - Multi-baseline comparison
  - Automated regression reporting
  - Performance degradation detection

## Configuration

### Evaluation Configuration (`evaluation_config.yaml`)
```yaml
evaluation:
  model_settings:
    device: "auto"
    dtype: "float16"
    max_new_tokens: 512
    temperature: 0.7
    
  targets:
    mmlu_code_accuracy: 0.60
    humaneval_pass1: 0.40
    codebleu_score: 0.65
    syntax_validity: 0.95
    web_dev_quality: 0.75
```

## Usage

### Quick Start
```bash
# Run comprehensive evaluation
python scripts/evaluate_model.py \
    --model_path /path/to/model \
    --config scripts/evaluation_config.yaml \
    --output_path ./evaluation_results \
    --run_id eval_$(date +%Y%m%d_%H%M%S)
```

### Individual Benchmark Runs
```bash
# MMLU Code evaluation
python scripts/mmlu_evaluation.py \
    --model_path /path/to/model \
    --config scripts/evaluation_config.yaml \
    --output_path ./results/mmlu \
    --run_id mmlu_eval

# HumanEval evaluation
python scripts/humaneval_evaluation.py \
    --model_path /path/to/model \
    --config scripts/evaluation_config.yaml \
    --output_path ./results/humaneval \
    --run_id humaneval_eval

# Web development tests
python scripts/web_dev_tests.py \
    --model_path /path/to/model \
    --config scripts/evaluation_config.yaml \
    --output_path ./results/webdev \
    --run_id webdev_eval

# Performance benchmarking
python scripts/performance_benchmark.py \
    --model_path /path/to/model \
    --config scripts/evaluation_config.yaml \
    --output_path ./results/performance \
    --run_id perf_eval

# Code quality tests
python scripts/code_quality_tests.py \
    --model_path /path/to/model \
    --config scripts/evaluation_config.yaml \
    --output_path ./results/quality \
    --run_id quality_eval

# Regression testing
python scripts/regression_testing.py \
    --model_path /path/to/model \
    --config scripts/evaluation_config.yaml \
    --output_path ./results/regression \
    --run_id regression_eval
```

### Advanced Configuration
```bash
# Custom targets and settings
python scripts/evaluate_model.py \
    --model_path /path/to/model \
    --config scripts/evaluation_config.yaml \
    --output_path ./evaluation_results \
    --run_id custom_eval \
    --skip_load  # Dry run without model loading
```

## Output Files

### Generated Reports
- `comprehensive_report_{run_id}.md` - Main evaluation report
- `evaluation_results_{run_id}.json` - Detailed JSON results
- `evaluation_summary_{run_id}.csv` - CSV summary
- `performance_metrics_{run_id}.json` - Performance metrics

### Individual Benchmark Outputs
Each benchmark generates:
- `{benchmark}_results_{run_id}.json` - Detailed results
- `{benchmark}_detailed_{run_id}.csv` - Sample-level data
- `{benchmark}_{run_id}.log` - Execution logs

## Target Achievement

The framework tracks the following performance targets:

| Benchmark | Target | Metric |
|-----------|--------|--------|
| MMLU Code | >60% | Accuracy |
| HumanEval | >40% | Pass@1 |
| Web Development | >75% | Quality Score |
| Code Quality | >95% | Syntax Validity |
| Code Quality | >0.65 | CodeBLEU Score |

## Performance Expectations

### Inference Speed
- **Excellent**: >50 tokens/second
- **Good**: 30-50 tokens/second  
- **Acceptable**: 20-30 tokens/second
- **Poor**: <20 tokens/second

### Memory Usage
- **Efficient**: <8GB model size
- **Standard**: 8-12GB model size
- **Large**: 12-20GB model size

## Integration

### Continuous Integration
```yaml
# .github/workflows/evaluation.yml
name: Model Evaluation
on: [push, pull_request]

jobs:
  evaluate:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v2
      - name: Run Evaluation
        run: |
          python scripts/evaluate_model.py \
            --model_path ${{ github.workspace }} \
            --config scripts/evaluation_config.yaml \
            --output_path ./results \
            --run_id ci_${{ github.sha }}
```

### Automated Reporting
The framework integrates with:
- **HuggingFace Evaluate Library**: Standard metrics
- **MLflow**: Experiment tracking
- **Weights & Biases**: Visualization dashboards
- **GitHub Actions**: CI/CD integration

## Troubleshooting

### Common Issues

1. **Model Loading Failures**
   ```bash
   # Check model path and permissions
   ls -la /path/to/model
   # Verify CUDA availability
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Memory Issues**
   ```yaml
   # Reduce batch sizes in config
   evaluation:
     model_settings:
       device_map: "cpu"  # Use CPU instead of GPU
   ```

3. **Dataset Access**
   ```bash
   # Login to HuggingFace
   huggingface-cli login
   # Or disable remote code loading
   ```

### Performance Optimization

1. **GPU Memory Optimization**
   - Use `device_map="auto"` for automatic placement
   - Enable gradient checkpointing for memory efficiency
   - Use quantization (int8, int4) for larger models

2. **Speed Optimization**
   - Increase batch sizes for throughput
   - Use faster attention implementations
   - Enable TensorRT optimization

## Customization

### Adding New Benchmarks
1. Create new evaluation script following existing patterns
2. Add to `evaluate_model.py` orchestrator
3. Update `evaluation_config.yaml` with new settings
4. Implement result saving and target tracking

### Modifying Targets
Edit `evaluation_config.yaml`:
```yaml
targets:
  mmlu_code_accuracy: 0.65  # Increased target
  humaneval_pass1: 0.45     # Increased target
  custom_metric: 0.80       # New metric
```

### Custom Quality Metrics
Extend existing evaluation classes:
```python
def evaluate_custom_metric(self, code_samples):
    # Implement custom quality assessment
    return custom_score
```

## Support

### Logging and Debugging
- All scripts generate detailed logs in output directories
- Enable debug mode in configuration:
  ```yaml
  logging:
    level: "DEBUG"
    debug_mode: true
  ```

### Resource Requirements
- **Minimum**: 8GB RAM, 1 GPU (4GB VRAM)
- **Recommended**: 16GB RAM, 1 GPU (8GB VRAM)
- **Optimal**: 32GB RAM, 2+ GPUs (16GB+ VRAM each)

### Best Practices
1. **Baseline Comparisons**: Always maintain baseline results for regression detection
2. **Incremental Testing**: Run individual benchmarks during development
3. **Regular Evaluation**: Schedule periodic comprehensive evaluations
4. **Result Archiving**: Save evaluation results for historical analysis

## License

This evaluation framework is part of the Sheikh-2.5-Coder project. See the main repository for license information.

---

**Note**: This framework is designed for systematic model evaluation and should be integrated into continuous development workflows for best results.