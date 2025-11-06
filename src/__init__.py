"""
Sheikh-2.5-Coder Model Package
MiniMax-M2 Architecture Implementation

This package provides the complete implementation of the Sheikh-2.5-Coder model
based on the MiniMax-M2 architecture specifications.

Features:
- 3.09B total parameters (2.77B non-embedding, 320M embedding)
- 36 transformer layers with GQA attention (16 Q heads, 2 KV heads)
- 32,768 token context length
- RoPE positional embeddings with theta=10000.0
- RMSNorm with epsilon=1e-6
- XML/MDX/JavaScript tokenization support
- Web development special tokens and patterns
- Memory-efficient attention computation
"""

from .configuration_sheikh_coder import SheikhCoderConfig, DEFAULT_SHEIKH_CODER_CONFIG
from .modeling_sheikh_coder import (
    SheikhCoderModel, 
    SheikhCoderForCausalLM,
    RotaryEmbedding,
    ModelOutput,
    CausalLMOutputWithPast
)
from .tokenization_sheikh_coder import (
    SheikhCoderTokenizer,
    create_default_tokenizer,
    batch_process_texts
)
from .modeling_utils import (
    RMSNorm,
    apply_rope_pos_emb,
    repeat_kv,
    get_memory_usage,
    get_model_flops,
    validate_model_architecture,
    optimize_for_inference,
    get_layer_statistics
)

__version__ = "2.5.0"
__author__ = "MiniMax Team"

# Model metadata
MODEL_INFO = {
    "name": "Sheikh-2.5-Coder",
    "architecture": "MiniMax-M2",
    "total_parameters": 3.09e9,
    "non_embedding_parameters": 2.77e9,
    "embedding_parameters": 0.32e9,
    "context_length": 32768,
    "vocab_size": 32000,
    "hidden_size": 2048,
    "num_layers": 36,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "intermediate_size": 8192,
    "rope_theta": 10000.0,
    "rms_norm_epsilon": 1e-6,
}

# Model configurations
MODEL_CONFIGS = {
    "default": DEFAULT_SHEIKH_CODER_CONFIG,
    "sheikh_coder_v2_5": SheikhCoderConfig(),
}

# Available models
AVAILABLE_MODELS = {
    "sheikh_coder_2_5": {
        "config_class": SheikhCoderConfig,
        "model_class": SheikhCoderForCausalLM,
        "tokenizer_class": SheikhCoderTokenizer,
        "description": "Sheikh-2.5-Coder model with MiniMax-M2 architecture"
    }
}

# Export all public interfaces
__all__ = [
    # Configuration
    "SheikhCoderConfig",
    "DEFAULT_SHEIKH_CODER_CONFIG",
    
    # Models
    "SheikhCoderModel",
    "SheikhCoderForCausalLM",
    
    # Tokenizer
    "SheikhCoderTokenizer",
    "create_default_tokenizer",
    "batch_process_texts",
    
    # Utilities
    "RMSNorm",
    "RotaryEmbedding",
    "apply_rope_pos_emb",
    "repeat_kv",
    "get_memory_usage",
    "get_model_flops",
    "validate_model_architecture",
    "optimize_for_inference",
    "get_layer_statistics",
    
    # Output classes
    "ModelOutput",
    "CausalLMOutputWithPast",
    
    # Metadata
    "MODEL_INFO",
    "MODEL_CONFIGS",
    "AVAILABLE_MODELS",
    
    # Package info
    "__version__",
    "__author__",
]

def get_model_info(model_name: str = "sheikh_coder_2_5") -> dict:
    """
    Get information about a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Model information dictionary
    """
    if model_name in AVAILABLE_MODELS:
        info = MODEL_INFO.copy()
        info.update(AVAILABLE_MODELS[model_name])
        return info
    else:
        available = list(AVAILABLE_MODELS.keys())
        raise ValueError(f"Model '{model_name}' not found. Available models: {available}")

def create_model(model_name: str = "sheikh_coder_2_5", **kwargs) -> SheikhCoderForCausalLM:
    """
    Create a model instance.
    
    Args:
        model_name: Name of the model to create
        **kwargs: Additional model configuration parameters
        
    Returns:
        Model instance
    """
    if model_name not in AVAILABLE_MODELS:
        available = list(AVAILABLE_MODELS.keys())
        raise ValueError(f"Model '{model_name}' not found. Available models: {available}")
    
    model_info = AVAILABLE_MODELS[model_name]
    config_class = model_info["config_class"]
    model_class = model_info["model_class"]
    
    # Create configuration
    if "config" in kwargs:
        config = kwargs["config"]
    else:
        config = config_class(**kwargs)
    
    # Create model
    model = model_class(config)
    
    return model

def create_tokenizer(**kwargs) -> SheikhCoderTokenizer:
    """
    Create a tokenizer instance.
    
    Args:
        **kwargs: Tokenizer configuration parameters
        
    Returns:
        Tokenizer instance
    """
    return SheikhCoderTokenizer(**kwargs)

# Helper functions for common operations
def analyze_model_architecture(model: SheikhCoderForCausalLM) -> dict:
    """
    Analyze a model's architecture and provide comprehensive statistics.
    
    Args:
        model: Model to analyze
        
    Returns:
        Dictionary with analysis results
    """
    return model.get_model_analysis()

def estimate_memory_requirements(
    precision: str = "float16", 
    sequence_length: int = 32768,
    batch_size: int = 1,
    **config_kwargs
) -> dict:
    """
    Estimate memory requirements for model training/inference.
    
    Args:
        precision: Model precision
        sequence_length: Input sequence length
        batch_size: Batch size
        **config_kwargs: Configuration parameters
        
    Returns:
        Dictionary with memory estimates
    """
    config = SheikhCoderConfig(**config_kwargs)
    memory_info = config.get_memory_requirements(precision)
    
    # Add sequence-specific calculations
    activation_memory = batch_size * sequence_length * config.hidden_size * 4  # 4 bytes for float32
    
    memory_info["sequence_specific"] = {
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "estimated_activation_memory_mb": round(activation_memory / (1024 * 1024), 2),
        "total_context_memory_gb": round((activation_memory * batch_size) / (1024**3), 2)
    }
    
    return memory_info

def validate_minimax_m2_specifications(config: SheikhCoderConfig) -> dict:
    """
    Validate that a configuration matches MiniMax-M2 specifications.
    
    Args:
        config: Configuration to validate
        
    Returns:
        Dictionary with validation results
    """
    expected_specs = {
        "total_parameters": 3.09e9,
        "non_embedding_parameters": 2.77e9,
        "embedding_parameters": 0.32e9,
        "hidden_size": 2048,
        "num_hidden_layers": 36,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "intermediate_size": 8192,
        "max_position_embeddings": 32768,
        "rope_theta": 10000.0,
        "rms_norm_epsilon": 1e-6,
    }
    
    validation_results = config.verify_parameter_count()
    config_dict = config.to_dict()
    
    # Check specification matches
    spec_matches = {}
    for spec_name, expected_value in expected_specs.items():
        if spec_name in config_dict:
            actual_value = config_dict[spec_name]
            spec_matches[spec_name] = {
                "expected": expected_value,
                "actual": actual_value,
                "match": actual_value == expected_value,
                "difference": abs(actual_value - expected_value) if isinstance(expected_value, (int, float)) else None
            }
        else:
            spec_matches[spec_name] = {
                "expected": expected_value,
                "actual": "Not found in config",
                "match": False,
                "difference": None
            }
    
    return {
        "specifications_match": spec_matches,
        "parameter_verification": validation_results,
        "overall_valid": all(
            match["match"] for match in spec_matches.values() if match["match"] is not None
        ) and all(match["match"] for match in validation_results.values() if match["match"] is not None),
    }

# Convenience functions for web development
def create_web_dev_tokenizer() -> SheikhCoderTokenizer:
    """
    Create a tokenizer optimized for web development tasks.
    
    Returns:
        Tokenizer instance with web development optimizations
    """
    return SheikhCoderTokenizer(
        has_web_tokens=True,
        has_xml_tokens=True,
        has_mdx_tokens=True,
        has_js_tokens=True,
    )

def create_code_generation_model() -> SheikhCoderForCausalLM:
    """
    Create a model optimized for code generation tasks.
    
    Returns:
        Model instance configured for code generation
    """
    config = SheikhCoderConfig()
    return SheikhCoderForCausalLM(config)

# Initialization function for package imports
def _init_package():
    """Initialize package-level settings."""
    # Set default precision for inference
    import torch
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

# Auto-initialize on import
_init_package()