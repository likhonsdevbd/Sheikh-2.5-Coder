#!/usr/bin/env python3
"""
Documentation Automation System for Model Deployment
Generates comprehensive documentation, API guides, and tutorials
"""

import os
import sys
import json
import argparse
import logging
import subprocess
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import jinja2
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DocumentationGenerator:
    """Automated documentation generation system"""
    
    def __init__(self, output_dir: str, model_name: str = "Sheikh-2.5-Coder"):
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.docs_dir = self.output_dir / "docs"
        self.examples_dir = self.output_dir / "examples"
        self.api_dir = self.output_dir / "api"
        
        # Create directories
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.examples_dir.mkdir(parents=True, exist_ok=True)
        self.api_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Jinja2 environment
        self.jinja_env = jinja2.Environment(
            loader=jinja2.DictLoader({
                'base.html': self._get_base_template(),
                'api_reference.html': self._get_api_template(),
                'tutorial.html': self._get_tutorial_template(),
                'example.html': self._get_example_template()
            }),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        
        logger.info(f"Initialized documentation generator: {model_name}")
    
    def generate_complete_documentation(self, config: Dict) -> Dict[str, str]:
        """Generate complete documentation suite"""
        logger.info("Generating complete documentation suite")
        
        generated_files = {}
        
        try:
            # Generate main documentation files
            generated_files.update(self.generate_main_docs(config))
            
            # Generate API reference
            generated_files.update(self.generate_api_reference(config))
            
            # Generate tutorials
            generated_files.update(self.generate_tutorials(config))
            
            # Generate examples
            generated_files.update(self.generate_examples(config))
            
            # Generate deployment guides
            generated_files.update(self.generate_deployment_guides(config))
            
            # Generate performance guides
            generated_files.update(self.generate_performance_guides(config))
            
            # Generate FAQ
            generated_files.update(self.generate_faq(config))
            
            logger.info(f"Generated {len(generated_files)} documentation files")
            return {
                'status': 'success',
                'generated_files': generated_files,
                'total_files': len(generated_files)
            }
            
        except Exception as e:
            logger.error(f"Documentation generation failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'generated_files': generated_files
            }
    
    def generate_main_docs(self, config: Dict) -> Dict[str, str]:
        """Generate main documentation files"""
        logger.info("Generating main documentation")
        
        generated_files = {}
        
        # README.md
        readme_content = self._generate_main_readme(config)
        readme_path = self.docs_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        generated_files['README'] = str(readme_path)
        
        # Getting Started Guide
        getting_started = self._generate_getting_started_guide(config)
        getting_started_path = self.docs_dir / "getting_started.md"
        with open(getting_started_path, 'w') as f:
            f.write(getting_started)
        generated_files['getting_started'] = str(getting_started_path)
        
        # Installation Guide
        installation = self._generate_installation_guide(config)
        installation_path = self.docs_dir / "installation.md"
        with open(installation_path, 'w') as f:
            f.write(installation)
        generated_files['installation'] = str(installation_path)
        
        # Model Overview
        overview = self._generate_model_overview(config)
        overview_path = self.docs_dir / "model_overview.md"
        with open(overview_path, 'w') as f:
            f.write(overview)
        generated_files['model_overview'] = str(overview_path)
        
        return generated_files
    
    def generate_api_reference(self, config: Dict) -> Dict[str, str]:
        """Generate API reference documentation"""
        logger.info("Generating API reference")
        
        generated_files = {}
        
        # Main API reference
        api_ref = self._generate_api_reference_doc(config)
        api_ref_path = self.api_dir / "reference.md"
        with open(api_ref_path, 'w') as f:
            f.write(api_ref)
        generated_files['api_reference'] = str(api_ref_path)
        
        # Model loading API
        model_api = self._generate_model_loading_api(config)
        model_api_path = self.api_dir / "model_loading.md"
        with open(model_api_path, 'w') as f:
            f.write(model_api)
        generated_files['model_loading'] = str(model_api_path)
        
        # Generation API
        generation_api = self._generate_generation_api(config)
        generation_api_path = self.api_dir / "generation.md"
        with open(generation_api_path, 'w') as f:
            f.write(generation_api)
        generated_files['generation'] = str(generation_api_path)
        
        # Configuration API
        config_api = self._generate_configuration_api(config)
        config_api_path = self.api_dir / "configuration.md"
        with open(config_api_path, 'w') as f:
            f.write(config_api)
        generated_files['configuration'] = str(config_api_path)
        
        return generated_files
    
    def generate_tutorials(self, config: Dict) -> Dict[str, str]:
        """Generate tutorial documentation"""
        logger.info("Generating tutorials")
        
        generated_files = {}
        
        tutorials = [
            ('basic_usage', self._generate_basic_usage_tutorial),
            ('code_generation', self._generate_code_generation_tutorial),
            ('web_integration', self._generate_web_integration_tutorial),
            ('fine_tuning', self._generate_fine_tuning_tutorial),
            ('production_deployment', self._generate_production_deployment_tutorial),
            ('optimization', self._generate_optimization_tutorial)
        ]
        
        for tutorial_name, generator_func in tutorials:
            tutorial_content = generator_func(config)
            tutorial_path = self.docs_dir / f"tutorial_{tutorial_name}.md"
            with open(tutorial_path, 'w') as f:
                f.write(tutorial_content)
            generated_files[tutorial_name] = str(tutorial_path)
        
        return generated_files
    
    def generate_examples(self, config: Dict) -> Dict[str, str]:
        """Generate code examples"""
        logger.info("Generating code examples")
        
        generated_files = {}
        
        examples = [
            ('simple_inference', self._generate_simple_inference_example),
            ('quantized_inference', self._generate_quantized_inference_example),
            ('web_app', self._generate_web_app_example),
            ('cli_tool', self._generate_cli_tool_example),
            ('batch_processing', self._generate_batch_processing_example),
            ('custom_pipeline', self._generate_custom_pipeline_example)
        ]
        
        for example_name, generator_func in examples:
            example_content = generator_func(config)
            example_path = self.examples_dir / f"{example_name}.py"
            with open(example_path, 'w') as f:
                f.write(example_content)
            generated_files[example_name] = str(example_path)
        
        return generated_files
    
    def generate_deployment_guides(self, config: Dict) -> Dict[str, str]:
        """Generate deployment guides"""
        logger.info("Generating deployment guides")
        
        generated_files = {}
        
        deployment_guides = [
            ('huggingface', self._generate_huggingface_deployment),
            ('docker', self._generate_docker_deployment),
            ('cloud_deployment', self._generate_cloud_deployment),
            ('edge_deployment', self._generate_edge_deployment),
            ('mobile_deployment', self._generate_mobile_deployment)
        ]
        
        for guide_name, generator_func in deployment_guides:
            guide_content = generator_func(config)
            guide_path = self.docs_dir / f"deployment_{guide_name}.md"
            with open(guide_path, 'w') as f:
                f.write(guide_content)
            generated_files[f'deployment_{guide_name}'] = str(guide_path)
        
        return generated_files
    
    def generate_performance_guides(self, config: Dict) -> Dict[str, str]:
        """Generate performance optimization guides"""
        logger.info("Generating performance guides")
        
        generated_files = {}
        
        performance_guides = [
            ('optimization_basics', self._generate_optimization_basics),
            ('memory_optimization', self._generate_memory_optimization),
            ('speed_optimization', self._generate_speed_optimization),
            ('gpu_optimization', self._generate_gpu_optimization),
            ('quantization_guide', self._generate_quantization_guide)
        ]
        
        for guide_name, generator_func in performance_guides:
            guide_content = generator_func(config)
            guide_path = self.docs_dir / f"performance_{guide_name}.md"
            with open(guide_path, 'w') as f:
                f.write(guide_content)
            generated_files[f'performance_{guide_name}'] = str(guide_path)
        
        return generated_files
    
    def generate_faq(self, config: Dict) -> Dict[str, str]:
        """Generate FAQ documentation"""
        logger.info("Generating FAQ")
        
        faq_content = self._generate_faq_doc(config)
        faq_path = self.docs_dir / "faq.md"
        with open(faq_path, 'w') as f:
            f.write(faq_content)
        
        return {'faq': str(faq_path)}
    
    def _generate_main_readme(self, config: Dict) -> str:
        """Generate main README"""
        return f"""# {self.model_name}

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

{self.model_name} is a state-of-the-art language model specifically fine-tuned for code generation tasks. Built on top of Microsoft's Phi-2, it excels at generating high-quality code across multiple programming languages.

## 🚀 Quick Start

### Installation

```bash
pip install transformers torch
```

### Basic Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("sheikh-team/{self.model_name.replace('-', '').lower()}")
tokenizer = AutoTokenizer.from_pretrained("sheikh-team/{self.model_name.replace('-', '').lower()}")

# Generate code
prompt = "def fibonacci(n):"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

## 📚 Documentation

- [Getting Started](getting_started.md) - Quick start guide
- [Installation](installation.md) - Detailed installation instructions
- [API Reference](api/reference.md) - Complete API documentation
- [Tutorials](tutorial_basic_usage.md) - Step-by-step tutorials
- [Examples](../examples/) - Code examples and use cases

## 🎯 Features

- **Multi-language Support**: Python, JavaScript, HTML/CSS, TypeScript, and more
- **High-Quality Code**: Optimized for readability and correctness
- **Multiple Optimizations**: INT8/INT4 quantization, ONNX export, TensorRT optimization
- **Easy Integration**: Compatible with HuggingFace Transformers
- **Production Ready**: Optimized for deployment in various environments

## 📈 Performance

| Metric | Value |
|--------|--------|
| Parameters | {config.get('parameters', '2.7B')} |
| Context Length | {config.get('context_length', '2048')} tokens |
| Languages Supported | {config.get('supported_languages', '10+')} |
| Inference Speed | Optimized for production |

## 🔧 Model Variants

{self._generate_model_variants_section(config)}

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Microsoft for the base Phi-2 model
- HuggingFace for the Transformers library
- The open-source community for their invaluable contributions

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/sheikh-team/{self.model_name.replace('-', '').lower()}/issues)
- **Discussions**: [GitHub Discussions](https://github.com/sheikh-team/{self.model_name.replace('-', '').lower()}/discussions)
- **Documentation**: [Full Documentation](https://github.com/sheikh-team/{self.model_name.replace('-', '').lower()}/wiki)
"""
    
    def _generate_getting_started_guide(self, config: Dict) -> str:
        """Generate getting started guide"""
        return f"""# Getting Started with {self.model_name}

This guide will help you get up and running with {self.model_name} in just a few minutes.

## Prerequisites

- Python 3.8 or higher
- PyTorch 1.12 or higher
- At least 4GB of RAM (8GB+ recommended for optimal performance)
- Optional: GPU with CUDA support for faster inference

## Installation

### Basic Installation

```bash
pip install transformers torch
```

### With GPU Support

```bash
pip install transformers torch --index-url https://download.pytorch.org/whl/cu118
```

### Development Installation

```bash
git clone https://github.com/sheikh-team/{self.model_name.replace('-', '').lower()}
cd {self.model_name.replace('-', '').lower()}
pip install -e .
```

## Quick Test

Let's verify everything is working correctly:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {{device}}")

# Load model
model_name = "sheikh-team/{self.model_name.replace('-', '').lower()}"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Generate code
prompt = "def hello_world():"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Generated code:")
print(result)
```

## First Project: Code Assistant

Here's a simple code assistant example:

```python
class CodeAssistant:
    def __init__(self, model_name="sheikh-team/{self.model_name.replace('-', '').lower()}"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def complete_code(self, prompt, max_length=100):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def explain_code(self, code):
        prompt = f"Explain this code:\n{{code}}\nExplanation:"
        return self.complete_code(prompt, max_length=200)

# Usage
assistant = CodeAssistant()
code = assistant.complete_code("Create a function to calculate factorial:")
print(code)
```

## Next Steps

- Explore [Basic Usage Tutorial](tutorial_basic_usage.md)
- Learn about [Advanced Configuration](api/configuration.md)
- Check out [Code Examples](../examples/)
- Read about [Performance Optimization](performance_optimization_basics.md)

## Troubleshooting

### Common Issues

1. **Out of Memory Error**
   - Try using quantization: `load_in_8bit=True`
   - Reduce batch size or sequence length
   - Use gradient checkpointing

2. **Slow Inference**
   - Ensure CUDA is available: `torch.cuda.is_available()`
   - Consider using smaller model variants
   - Enable attention optimizations

3. **Import Errors**
   - Update to latest versions: `pip install --upgrade transformers torch`
   - Check Python version: `python --version`

For more help, see our [FAQ](faq.md) or open an issue on GitHub.
"""
    
    def _generate_installation_guide(self, config: Dict) -> str:
        """Generate installation guide"""
        return f"""# Installation Guide

This guide provides detailed instructions for installing and setting up {self.model_name} in various environments.

## System Requirements

### Minimum Requirements
- **OS**: Linux, Windows, or macOS
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB+ recommended
- **Storage**: 2GB free space for model files

### Recommended for Production
- **GPU**: NVIDIA GPU with 6GB+ VRAM (RTX 3060, A100, etc.)
- **RAM**: 16GB or higher
- **Storage**: SSD with 10GB+ free space
- **CUDA**: 11.8 or higher (for GPU acceleration)

## Installation Methods

### Method 1: pip (Recommended)

```bash
# Basic installation
pip install transformers torch

# With GPU support (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# With development dependencies
pip install transformers[torch] datasets accelerate
```

### Method 2: conda

```bash
# Create conda environment
conda create -n {self.model_name.replace('-', '').lower()} python=3.10
conda activate {self.model_name.replace('-', '').lower()}

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install Transformers
pip install transformers
```

### Method 3: Docker

```bash
# Pull pre-built image
docker pull sheikh-team/{self.model_name.replace('-', '').lower()}:latest

# Run container
docker run -it --gpus all sheikh-team/{self.model_name.replace('-', '').lower()}:latest
```

#### Dockerfile Example

```dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN pip install -e .

CMD ["python", "main.py"]
```

### Method 4: From Source

```bash
# Clone repository
git clone https://github.com/sheikh-team/{self.model_name.replace('-', '').lower()}.git
cd {self.model_name.replace('-', '').lower()}

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

## Configuration

### Environment Variables

```bash
# Set cache directory
export HF_HOME=/path/to/cache

# Enable CUDA optimizations
export CUDA_VISIBLE_DEVICES=0

# Set memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

### Model Download Configuration

```python
from transformers import AutoModelForCausalLM
import os

# Set custom cache directory
cache_dir = "/path/to/custom/cache"
os.environ['TRANSFORMERS_CACHE'] = cache_dir

# Download model
model = AutoModelForCausalLM.from_pretrained(
    "sheikh-team/{self.model_name.replace('-', '').lower()}",
    cache_dir=cache_dir
)
```

## Platform-Specific Instructions

### Ubuntu/Debian

```bash
# Install system dependencies
sudo apt update
sudo apt install python3-dev build-essential

# Install Python packages
pip3 install --user transformers torch
```

### Windows

```powershell
# Install PyTorch with CUDA (if available)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Transformers
pip install transformers
```

### macOS

```bash
# For Apple Silicon Macs
pip install transformers torch

# For Intel Macs
pip install transformers torch
```

## Verification

Run this script to verify your installation:

```python
#!/usr/bin/env python3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print(f"PyTorch version: {{torch.__version__}}")
print(f"CUDA available: {{torch.cuda.is_available()}}")

if torch.cuda.is_available():
    print(f"CUDA version: {{torch.version.cuda}}")
    print(f"GPU: {{torch.cuda.get_device_name(0)}}")

# Test model loading
try:
    model_name = "sheikh-team/{self.model_name.replace('-', '').lower()}"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Model loading failed: {{e}}")
```

## Troubleshooting

### Common Installation Issues

1. **CUDA Version Mismatch**
   ```bash
   # Check CUDA version
   nvcc --version
   
   # Install matching PyTorch version
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Permission Errors**
   ```bash
   # Install for user only
   pip install --user transformers torch
   
   # Or use virtual environment
   python -m venv myenv
   source myenv/bin/activate
   pip install transformers torch
   ```

3. **Memory Issues During Installation**
   ```bash
   # Clear pip cache
   pip cache purge
   
   # Install with minimal dependencies
   pip install transformers torch --no-deps
   ```

### Performance Optimization

```python
# Enable memory efficient attention
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Enable flash attention (if available)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="flash_attention_2"
)
```

## Advanced Installation

### Custom CUDA Installation

```bash
# Download and install CUDA 11.8
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# Install PyTorch with specific CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Multi-GPU Setup

```python
# Distribute model across multiple GPUs
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_8bit=True  # Reduces memory usage
)
```

For more help, check our [FAQ](faq.md) or visit our [community forum](https://github.com/sheikh-team/{self.model_name.replace('-', '').lower()}/discussions).
"""
    
    def _generate_model_overview(self, config: Dict) -> str:
        """Generate model overview"""
        return f"""# {self.model_name} - Model Overview

{self.model_name} is a specialized code generation language model fine-tuned for producing high-quality, functional code across multiple programming languages.

## Architecture

### Base Model
- **Architecture**: Transformer-based causal language model
- **Base Model**: Microsoft Phi-2
- **Parameters**: {config.get('parameters', '2.7B')} trainable parameters
- **Context Length**: {config.get('context_length', '2048')} tokens
- **Vocabulary**: Optimized for code and natural language

### Fine-tuning Details
- **Method**: LoRA (Low-Rank Adaptation)
- **Dataset**: Stack v2 + synthetic code examples
- **Training Framework**: PyTorch + Transformers
- **Optimization**: AdamW with learning rate scheduling

## Model Capabilities

### Programming Languages
{self._generate_language_support_section(config)}

### Code Generation Tasks
- **Function Generation**: Complete functions from signatures or descriptions
- **Class Creation**: Generate complete classes with methods and properties
- **Code Completion**: Fill in missing code segments
- **Documentation**: Generate code comments and docstrings
- **Testing**: Create unit tests for existing functions
- **Refactoring**: Propose code improvements and optimizations

### Special Features
- **Context Awareness**: Understands code context and maintains consistency
- **Best Practices**: Follows coding standards and best practices
- **Error Handling**: Incorporates proper error handling in generated code
- **Type Hints**: Includes type annotations when appropriate

## Model Variants

{self._generate_detailed_variants_section(config)}

## Performance Characteristics

### Inference Speed
- **Base Model**: Optimized for GPU inference
- **Quantized Models**: Reduced memory with minimal speed impact
- **CPU Inference**: Supported with reduced speed

### Memory Usage
- **FP16 Model**: ~5.4GB GPU memory
- **INT8 Quantized**: ~2.7GB GPU memory (50% reduction)
- **INT4 Quantized**: ~1.35GB GPU memory (75% reduction)

### Quality Metrics
- **Code Correctness**: High success rate on functional tests
- **Code Quality**: Follows best practices and style guidelines
- **Readability**: Generated code is well-structured and readable

## Limitations

### Current Limitations
- **Context Window**: Limited to {config.get('context_length', '2048')} tokens
- **Real-time Data**: No access to current information or live APIs
- **Domain Specificity**: Best performance on general programming tasks
- **Code Execution**: Cannot execute or test generated code

### Known Issues
- May generate incomplete functions for very complex algorithms
- Occasional syntax errors in edge cases
- Limited knowledge of very recent libraries or frameworks

## Use Cases

### Development Workflow
1. **Code Generation**: Start with function signatures or descriptions
2. **Code Completion**: Use as an intelligent code completion tool
3. **Code Review**: Generate alternative implementations for comparison
4. **Documentation**: Auto-generate docstrings and comments
5. **Testing**: Create unit tests and test cases

### Educational Applications
- **Learning Assistant**: Help students understand programming concepts
- **Code Examples**: Generate examples for teaching purposes
- **Practice Problems**: Create coding exercises and solutions

### Production Applications
- **API Documentation**: Generate comprehensive API documentation
- **Code Migration**: Help with code translation between languages
- **Legacy Code**: Modernize and refactor existing codebases

## Best Practices

### Getting the Best Results
1. **Clear Prompts**: Provide specific, detailed descriptions
2. **Context**: Include relevant code context when possible
3. **Iterative Refinement**: Review and refine generated code
4. **Testing**: Always test generated code before production use

### Prompt Engineering Tips
```python
# Good prompt
"Create a Python function to calculate the factorial of a number using recursion. Include error handling for negative inputs."

# Better prompt  
def fibonacci(n: int) -> int:
    \"\"\"Calculate the nth Fibonacci number using dynamic programming.
    
    Args:
        n: The position in the Fibonacci sequence (0-indexed)
        
    Returns:
        The nth Fibonacci number
        
    Raises:
        ValueError: If n is negative
    \"\"\"
    # Complete this function with optimized implementation
```

## Comparison with Other Models

{self._generate_comparison_section(config)}

## Future Developments

### Planned Improvements
- Expanded language support
- Longer context windows
- Enhanced code understanding
- Better integration with development tools

### Roadmap
- **Version 1.1**: Performance optimizations and bug fixes
- **Version 1.2**: Additional programming languages
- **Version 2.0**: Larger model with enhanced capabilities

For detailed usage instructions, see our [Getting Started Guide](getting_started.md) and [Tutorials](tutorial_basic_usage.md).
"""
    
    def _generate_api_reference_doc(self, config: Dict) -> str:
        """Generate API reference documentation"""
        return f"""# API Reference

Complete API reference for {self.model_name}.

## Model Loading

### AutoModelForCausalLM

Primary interface for loading the {self.model_name} model.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "sheikh-team/{self.model_name.replace('-', '').lower()}",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
```

#### Parameters

- `pretrained_model_name_or_path` (str): Model identifier or path
- `torch_dtype` (torch.dtype, optional): Data type for model weights
- `device_map` (str or dict, optional): Device mapping for model placement
- `trust_remote_code` (bool): Whether to trust remote code execution
- `load_in_8bit` (bool): Enable 8-bit quantization
- `load_in_4bit` (bool): Enable 4-bit quantization
- `cache_dir` (str, optional): Directory for caching downloaded models

### AutoTokenizer

Tokenizer for preprocessing text input.

```python
tokenizer = AutoTokenizer.from_pretrained(
    "sheikh-team/{self.model_name.replace('-', '').lower()}",
    trust_remote_code=True
)
```

#### Special Tokens

- `pad_token`: Set to `eos_token` if not configured
- `bos_token`: Beginning of sequence token
- `eos_token`: End of sequence token

## Generation

### model.generate()

Generate text from input prompts.

```python
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.7,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

#### Key Parameters

- `input_ids` (torch.Tensor): Tokenized input sequence
- `max_new_tokens` (int): Maximum number of tokens to generate
- `temperature` (float): Controls randomness (0.0 = deterministic, 1.0 = default)
- `top_p` (float): Nucleus sampling parameter
- `top_k` (int): Top-k sampling parameter
- `do_sample` (bool): Whether to use sampling vs. greedy decoding
- `pad_token_id` (int): Token ID for padding (usually eos_token_id)
- `eos_token_id` (int): Token ID for end of sequence

#### Generation Modes

**Deterministic (Greedy)**
```python
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=False
)
```

**Sampling**
```python
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)
```

**Beam Search**
```python
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    num_beams=5,
    do_sample=False
)
```

## Configuration

### GenerationConfig

Configuration class for generation parameters.

```python
from transformers import GenerationConfig

config = GenerationConfig(
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

outputs = model.generate(**inputs, generation_config=config)
```

## Quantization

### 8-bit Quantization

```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_skip_modules=None
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config
)
```

### 4-bit Quantization

```python
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config
)
```

## Utility Functions

### Text Processing

```python
def preprocess_code(code: str) -> str:
    \"\"\"Preprocess code for generation.\"\"\"
    # Add function signature if missing
    if not code.strip().startswith(('def ', 'class ')):
        code = f"def generated_function():\n    {code}"
    return code

def postprocess_code(generated_text: str, prompt: str) -> str:
    \"\"\"Postprocess generated code.\"\"\"
    # Remove prompt from generated text
    if prompt in generated_text:
        generated_text = generated_text.replace(prompt, "", 1)
    return generated_text.strip()
```

### Batch Processing

```python
def batch_generate(prompts: List[str], batch_size: int = 4):
    \"\"\"Generate code for multiple prompts.\"\"\"
    results = []
    
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        
        # Tokenize batch
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50)
        
        # Decode results
        batch_results = [
            tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        results.extend(batch_results)
    
    return results
```

## Error Handling

### Common Exceptions

```python
try:
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
except Exception as e:
    print(f"Model loading failed: {{e}}")

try:
    outputs = model.generate(**inputs)
except torch.cuda.OutOfMemoryError:
    print("CUDA out of memory. Try reducing batch size or using quantization.")
except Exception as e:
    print(f"Generation failed: {{e}}")
```

## Performance Monitoring

```python
import time
import torch

def measure_generation_speed(model, tokenizer, prompt, num_runs=10):
    \"\"\"Measure average generation speed.\"\"\"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50)
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = sum(times) / len(times)
    tokens_per_second = 50 / avg_time
    
    return {{
        'avg_time': avg_time,
        'tokens_per_second': tokens_per_second
    }}
```

## Memory Management

```python
import gc

def clear_memory():
    \"\"\"Clear GPU memory.\"\"\"
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def optimize_memory(model):
    \"\"\"Optimize model for memory efficiency.\"\"\"
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()
    
    # Enable gradient checkpointing
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    return model
```

For more detailed examples, see our [Examples Directory](../examples/) and [Tutorials](../tutorials/).
"""
    
    def _generate_basic_usage_tutorial(self, config: Dict) -> str:
        """Generate basic usage tutorial"""
        return f"""# Tutorial: Basic Usage

This tutorial will teach you the fundamentals of using {self.model_name} for code generation.

## What You'll Learn

- How to load the model and tokenizer
- Basic code generation
- Understanding generation parameters
- Best practices for prompting

## Prerequisites

Make sure you have installed {self.model_name}:

```bash
pip install transformers torch
```

## Step 1: Loading the Model

The first step is to load both the model and tokenizer:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_name = "sheikh-team/{self.model_name.replace('-', '').lower()}"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

# Set pad token (important!)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("✅ Model and tokenizer loaded successfully!")
```

## Step 2: Simple Code Generation

Let's start with a simple example:

```python
# Define a prompt
prompt = "def fibonacci(n):"

# Tokenize the input
inputs = tokenizer(prompt, return_tensors="pt")

# Generate code
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False  # Deterministic generation
    )

# Decode the output
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated code:")
print(generated_text)
```

## Step 3: Understanding Generation Parameters

Different parameters affect the output quality and style:

### Temperature

Temperature controls randomness:
- `temperature=0.0`: Deterministic (same output every time)
- `temperature=0.7`: Balanced (default)
- `temperature=1.0`: More creative/random

```python
# Conservative generation
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    temperature=0.2,  # Low temperature = more predictable
    do_sample=True
)

# Creative generation
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    temperature=0.9,  # Higher temperature = more creative
    do_sample=True
)
```

### Top-p and Top-k Sampling

Control vocabulary selection:

```python
# Top-p (nucleus) sampling
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    top_p=0.9,  # Consider tokens with cumulative probability > 0.9
    do_sample=True
)

# Top-k sampling
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    top_k=50,  # Consider only top 50 tokens
    do_sample=True
)
```

## Step 4: Prompt Engineering

The quality of generated code heavily depends on your prompts:

### Good Prompts

```python
good_prompts = [
    "def calculate_gcd(a, b):  # Greatest Common Divisor",
    "class BankAccount:  # Bank account with deposit/withdraw",
    "def binary_search(arr, target):  # Binary search algorithm",
    "# Create a function to validate email addresses",
    "def merge_sort(arr):  # Merge sort implementation"
]

for prompt in good_prompts:
    print(f"\\nPrompt: {{prompt}}")
    # Generate code...
```

### Adding Context

Provide more context for better results:

```python
# Simple prompt
prompt = "def process_data(data):"

# Better prompt with context
prompt = """
def process_user_data(user_data: dict) -> dict:
    '''Process user data with validation and sanitization.
    
    Args:
        user_data: Dictionary containing user information
        
    Returns:
        Processed user data dictionary
    '''
    # Complete this function
"""

# Even better with examples
prompt = """
def calculate_discount(price: float, discount_percent: float) -> float:
    '''Calculate the final price after discount.
    
    Examples:
        calculate_discount(100, 20) == 80
        calculate_discount(50, 10) == 45
    '''
    # Complete this function
"""
```

## Step 5: Creating a Simple Assistant

Let's build a simple code assistant class:

```python
class CodeAssistant:
    def __init__(self, model_name="sheikh-team/{self.model_name.replace('-', '').lower()}"):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_code(self, prompt, max_tokens=100, temperature=0.7):
        '''Generate code from a prompt.'''
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from the generated text
        if prompt in generated_text:
            generated_text = generated_text.replace(prompt, "", 1)
        
        return generated_text.strip()
    
    def complete_function(self, function_signature):
        '''Complete a function implementation.'''
        prompt = f"def {function_signature}:\n    '''Complete this function.'''\n    # Your code here"
        return self.generate_code(prompt)

# Usage
assistant = CodeAssistant()

# Generate a function
result = assistant.generate_code("def bubble_sort(lst):")
print("Generated function:")
print(result)

# Complete a function
result = assistant.complete_function("quick_sort(arr, low, high)")
print("Completed function:")
print(result)
```

## Step 6: Error Handling and Best Practices

```python
def safe_generate(assistant, prompt, max_retries=3):
    '''Safely generate code with error handling.'''
    for attempt in range(max_retries):
        try:
            return assistant.generate_code(prompt)
        except torch.cuda.OutOfMemoryError:
            if attempt == max_retries - 1:
                raise
            # Clear memory and try again
            torch.cuda.empty_cache()
            assistant.model = AutoModelForCausalLM.from_pretrained(
                "sheikh-team/{self.model_name.replace('-', '').lower()}",
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Generation failed (attempt {{attempt + 1}}): {{e}}")
            if attempt == max_retries - 1:
                raise
```

## Step 7: Performance Tips

```python
# Use batch processing for multiple prompts
def batch_generate_code(assistant, prompts, batch_size=4):
    '''Generate code for multiple prompts efficiently.'''
    results = []
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        
        # Tokenize batch
        inputs = assistant.tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True
        )
        
        # Generate for batch
        with torch.no_grad():
            outputs = assistant.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                pad_token_id=assistant.tokenizer.eos_token_id
            )
        
        # Decode results
        batch_results = [
            assistant.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        
        results.extend(batch_results)
    
    return results
```

## Next Steps

Now that you understand the basics:

1. Try the [Code Generation Tutorial](tutorial_code_generation.md)
2. Learn about [Web Integration](tutorial_web_integration.md)
3. Explore [Advanced Configuration](api/configuration.md)
4. Check out our [Examples](../examples/)

## Summary

In this tutorial, you learned:
- ✅ How to load {self.model_name}
- ✅ Basic code generation
- ✅ Generation parameters and their effects
- ✅ Prompt engineering best practices
- ✅ Building a simple code assistant
- ✅ Error handling and performance tips

Happy coding! 🚀
"""
    
    def _generate_simple_inference_example(self, config: Dict) -> str:
        """Generate simple inference example"""
        return f'''#!/usr/bin/env python3
"""
Simple Inference Example for {self.model_name}

This example demonstrates basic usage of {self.model_name} for code generation.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    """Main function demonstrating basic inference."""
    
    # Model configuration
    model_name = "sheikh-team/{self.model_name.replace('-', '').lower()}"
    
    print(f"🚀 Loading {{model_name}}...")
    
    # Load model and tokenizer
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Set pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("✅ Model loaded successfully!")
        
    except Exception as e:
        print(f"❌ Failed to load model: {{e}}")
        return
    
    # Define test prompts
    prompts = [
        "def fibonacci(n):",
        "class Calculator:",
        "def binary_search(arr, target):",
        "# Create a REST API endpoint",
        "def factorial(n):"
    ]
    
    print("\\n🔍 Generating code for sample prompts...")
    
    # Generate code for each prompt
    for i, prompt in enumerate(prompts, 1):
        print(f"\\n--- Prompt {{i}}: {{prompt}} ---")
        
        try:
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt")
            
            # Generate code
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=80,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode and display result
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove prompt from output
            if prompt in generated_text:
                generated_text = generated_text.replace(prompt, "", 1)
            
            print("Generated code:")
            print("-" * 40)
            print(generated_text.strip())
            print("-" * 40)
            
        except Exception as e:
            print(f"Generation failed: {{e}}")


def interactive_mode():
    """Interactive mode for testing prompts."""
    
    model_name = "sheikh-team/{self.model_name.replace('-', '').lower()}"
    
    print("🔧 Loading model for interactive mode...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ Model loaded! Enter your code prompts (type 'quit' to exit):")
    print("-" * 60)
    
    while True:
        try:
            # Get user input
            prompt = input("\\n💻 Enter prompt: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not prompt:
                continue
            
            # Generate code
            inputs = tokenizer(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove prompt
            if prompt in generated_text:
                generated_text = generated_text.replace(prompt, "", 1)
            
            print("\\n🤖 Generated code:")
            print("=" * 40)
            print(generated_text.strip())
            print("=" * 40)
            
        except KeyboardInterrupt:
            print("\\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {{e}}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        main()
        
        # Offer interactive mode
        try:
            response = input("\\n🌟 Would you like to try interactive mode? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                interactive_mode()
        except KeyboardInterrupt:
            print("\\n👋 Goodbye!")
'''
    
    def _get_base_template(self) -> str:
        """Get base HTML template"""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; margin: 0; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: #f8f9fa; padding: 2rem; border-radius: 8px; margin-bottom: 2rem; }
        .content { background: white; padding: 2rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .nav { background: #e9ecef; padding: 1rem; border-radius: 8px; margin-bottom: 2rem; }
        .nav a { margin-right: 1rem; text-decoration: none; color: #007bff; }
        .nav a:hover { text-decoration: underline; }
        pre { background: #f8f9fa; padding: 1rem; border-radius: 4px; overflow-x: auto; }
        code { background: #f8f9fa; padding: 0.2rem 0.4rem; border-radius: 3px; font-size: 0.9em; }
        .toc { background: #f8f9fa; padding: 1rem; border-radius: 8px; margin-bottom: 2rem; }
    </style>
</head>
<body>
    <div class="container">
        {% if header %}
        <div class="header">
            <h1>{{ title }}</h1>
            <p>{{ description }}</p>
        </div>
        {% endif %}
        
        {% if nav %}
        <div class="nav">
            {% for link in nav %}
            <a href="{{ link.url }}">{{ link.text }}</a>
            {% endfor %}
        </div>
        {% endif %}
        
        <div class="content">
            {{ content }}
        </div>
    </div>
</body>
</html>'''
    
    def _get_api_template(self) -> str:
        """Get API reference template"""
        return '''{% extends "base.html" %}

{% block content %}
<h2>API Reference</h2>

<h3>Model Loading</h3>
<pre><code>from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{{ model_name }}")
tokenizer = AutoTokenizer.from_pretrained("{{ model_name }}")</code></pre>

<h3>Code Generation</h3>
<pre><code>prompt = "def hello_world():"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)</code></pre>

<h3>Parameters</h3>
<ul>
    <li><strong>max_new_tokens</strong>: Maximum tokens to generate</li>
    <li><strong>temperature</strong>: Controls randomness (0.0-1.0)</li>
    <li><strong>do_sample</strong>: Enable sampling vs greedy decoding</li>
    <li><strong>top_p</strong>: Nucleus sampling parameter</li>
    <li><strong>top_k</strong>: Top-k sampling parameter</li>
</ul>
{% endblock %}'''
    
    def _get_tutorial_template(self) -> str:
        """Get tutorial template"""
        return '''{% extends "base.html" %}

{% block content %}
{{ content }}
{% endblock %}'''
    
    def _get_example_template(self) -> str:
        """Get example template"""
        return '''{% extends "base.html" %}

{% block content %}
<h2>{{ title }}</h2>
<p>{{ description }}</p>

<h3>Code Example</h3>
<pre><code>{{ code }}</code></pre>

<h3>Explanation</h3>
{{ explanation }}
{% endblock %}'''
    
    def _generate_model_variants_section(self, config: Dict) -> str:
        """Generate model variants section"""
        return """
| Variant | Size | Memory | Speed | Use Case |
|---------|------|--------|-------|----------|
| Base Model | 2.7B params | ~5.4GB | Full speed | Development & research |
| INT8 | 2.7B params | ~2.7GB | Slightly slower | Production deployment |
| INT4 | 2.7B params | ~1.35GB | Moderate | Edge & mobile devices |
| ONNX | Variable | Platform dependent | Accelerated | Cross-platform inference |
"""
    
    def _generate_language_support_section(self, config: Dict) -> str:
        """Generate language support section"""
        languages = [
            "Python", "JavaScript", "TypeScript", "HTML/CSS", "Java", "C++", 
            "C#", "Go", "Rust", "PHP", "Ruby", "Swift", "Kotlin", "SQL"
        ]
        
        return "### Supported Languages\n\n"
        for lang in languages:
            return += f"- **{lang}**: ✅ Full support\n"
    
    def _generate_detailed_variants_section(self, config: Dict) -> str:
        """Generate detailed variants section"""
        return """
## Model Variants

### 1. Base Model (Full Precision)
- **Format**: PyTorch (.safetensors)
- **Memory**: ~5.4GB GPU memory
- **Speed**: Maximum inference speed
- **Quality**: Highest quality output
- **Best for**: Development, research, high-quality production use

### 2. INT8 Quantized Model
- **Format**: 8-bit quantized PyTorch
- **Memory**: ~2.7GB GPU memory (50% reduction)
- **Speed**: Slightly slower than base model
- **Quality**: Minimal quality loss
- **Best for**: Production deployment with memory constraints

### 3. INT4 Quantized Model
- **Format**: 4-bit quantized PyTorch
- **Memory**: ~1.35GB GPU memory (75% reduction)
- **Speed**: Moderate slowdown
- **Quality**: Minor quality trade-off
- **Best for**: Edge devices, mobile deployment

### 4. ONNX Optimized Model
- **Format**: ONNX with optimizations
- **Memory**: Variable by hardware
- **Speed**: Accelerated on supported hardware
- **Quality**: Equivalent to base model
- **Best for**: Cross-platform deployment, hardware acceleration

### 5. TensorRT Optimized Model
- **Format**: TensorRT engine
- **Memory**: Optimized for NVIDIA GPUs
- **Speed**: Maximum GPU inference speed
- **Quality**: Equivalent to base model
- **Best for**: High-performance GPU inference
"""
    
    def _generate_comparison_section(self, config: Dict) -> str:
        """Generate comparison section"""
        return """
## Comparison with Other Models

### vs. CodeT5/StarCoder
- **Better Python Support**: Enhanced for Python-specific patterns
- **Improved Code Quality**: Higher success rate on functional tests
- **Smaller Size**: More efficient for deployment

### vs. GPT-based Models
- **Code-Focused**: Specialized for programming tasks
- **Faster Inference**: Optimized for code generation
- **Better Context**: Understanding of programming concepts

### vs. Other Phi Fine-tunes
- **Diverse Training**: Multi-language code dataset
- **Production Ready**: Optimized for real-world usage
- **Comprehensive Testing**: Extensive quality validation
"""
    
    def create_single_file_doc(self, title: str, content: str, output_path: str,
                             template_type: str = "base") -> str:
        """Create a single documentation file"""
        template = self.jinja_env.get_template(f"{template_type}.html")
        
        html_content = template.render(
            title=title,
            content=content,
            model_name=self.model_name
        )
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Created documentation: {output_path}")
        return output_path


def main():
    """Main function for documentation generation"""
    parser = argparse.ArgumentParser(description='Generate documentation')
    
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--model_name', default='Sheikh-2.5-Coder', help='Model name')
    parser.add_argument('--config', help='Configuration file')
    parser.add_argument('--action', required=True,
                       choices=['generate_docs', 'single_file'],
                       help='Action to perform')
    
    # Action-specific arguments
    parser.add_argument('--title', help='Document title')
    parser.add_argument('--content', help='Document content')
    parser.add_argument('--output_path', help='Output file path')
    parser.add_argument('--template', default='base', help='Template type')
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            if args.config.endswith('.yaml') or args.config.endswith('.yml'):
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
    
    # Initialize generator
    generator = DocumentationGenerator(
        output_dir=args.output_dir,
        model_name=args.model_name
    )
    
    # Execute action
    if args.action == 'generate_docs':
        result = generator.generate_complete_documentation(config)
        print(json.dumps(result, indent=2, default=str))
        
    elif args.action == 'single_file':
        if not args.title or not args.content or not args.output_path:
            print("Error: --title, --content, and --output_path required")
            return 1
        
        generator.create_single_file_doc(
            title=args.title,
            content=args.content,
            output_path=args.output_path,
            template_type=args.template
        )
        print(f"Documentation created: {args.output_path}")
    
    else:
        print(f"Unknown action: {args.action}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())