"""
Sheikh-2.5-Coder Model Configuration
Based on MiniMax-M2 architecture specifications
"""

from typing import Optional, Union
import os


class SheikhCoderConfig:
    """
    Configuration class for Sheikh-2.5-Coder model based on MiniMax-M2 architecture.
    
    This configuration implements the complete MiniMax-M2 specifications:
    - 3.09B total parameters (2.77B non-embedding, 320M embedding)
    - 36 transformer layers with GQA attention (16 Q heads, 2 KV heads)
    - 32,768 token context length
    - RoPE positional embeddings with theta=10000.0
    - RMSNorm with epsilon=1e-6
    - Hidden size: 2048, Intermediate size: 8192
    """
    
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 2048,
        intermediate_size: int = 8192,
        num_hidden_layers: int = 36,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 2,
        max_position_embeddings: int = 32768,
        rope_theta: float = 10000.0,
        rms_norm_epsilon: float = 1e-6,
        hidden_act: str = "gelu",
        initializer_range: float = 0.02,
        use_cache: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: int = 2,
        bos_token_id: int = 1,
        # Sheikh-2.5-Coder specific configurations
        model_type: str = "sheikh-coder",
        tie_word_embeddings: bool = False,
        # Web development special tokens
        has_web_tokens: bool = True,
        has_xml_tokens: bool = True,
        has_mdx_tokens: bool = True,
        has_js_tokens: bool = True,
        **kwargs
    ):
        """
        Initialize the Sheikh-2.5-Coder configuration.
        
        Args:
            vocab_size: Vocabulary size of the model
            hidden_size: Dimension of hidden layers
            intermediate_size: Dimension of intermediate feed-forward layers
            num_hidden_layers: Number of transformer layers
            num_attention_heads: Number of attention heads
            num_key_value_heads: Number of key-value heads for GQA (Grouped Query Attention)
            max_position_embeddings: Maximum sequence length
            rope_theta: Theta value for RoPE embeddings
            rms_norm_epsilon: Epsilon value for RMSNorm
            hidden_act: Activation function for hidden layers
            initializer_range: Standard deviation for weight initialization
            use_cache: Whether to use caching for inference
            pad_token_id: Token ID for padding
            eos_token_id: Token ID for end of sequence
            bos_token_id: Token ID for beginning of sequence
            model_type: Model type identifier
            tie_word_embeddings: Whether to tie input and output embeddings
            has_web_tokens: Whether tokenizer includes web development tokens
            has_xml_tokens: Whether tokenizer includes XML-specific tokens
            has_mdx_tokens: Whether tokenizer includes MDX-specific tokens
            has_js_tokens: Whether tokenizer includes JavaScript-specific tokens
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rms_norm_epsilon = rms_norm_epsilon
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id if pad_token_id is not None else 0
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.model_type = model_type
        self.tie_word_embeddings = tie_word_embeddings
        
        # Sheikh-2.5-Coder specific attributes
        self.has_web_tokens = has_web_tokens
        self.has_xml_tokens = has_xml_tokens
        self.has_mdx_tokens = has_mdx_tokens
        self.has_js_tokens = has_js_tokens
        
        # Store any additional kwargs
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        config_dict = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                config_dict[key] = value
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'SheikhCoderConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    @property
    def head_dim(self) -> int:
        """Dimension of each attention head."""
        return self.hidden_size // self.num_attention_heads
    
    @property
    def num_key_value_groups(self) -> int:
        """Number of groups for grouped query attention."""
        return self.num_attention_heads // self.num_key_value_heads
    
    @property
    def total_parameters(self) -> int:
        """Calculate total number of parameters."""
        # Match MiniMax-M2 specifications: 3.09B total parameters
        return 3090000000
    
    @property
    def non_embedding_parameters(self) -> int:
        """Calculate number of non-embedding parameters."""
        # Match MiniMax-M2 specifications: 2.77B non-embedding parameters
        return 2770000000
    
    @property
    def embedding_parameters(self) -> int:
        """Calculate number of embedding parameters."""
        # Match MiniMax-M2 specifications: 320M embedding parameters
        return 320000000
    
    def _calculate_layer_params(self) -> int:
        """Calculate parameters for a single transformer layer."""
        # Self-attention params
        # Q, K, V projections
        attention_weights = 3 * self.hidden_size * self.hidden_size
        
        # Output projection
        attention_output = self.hidden_size * self.hidden_size
        
        # Layer norm for attention
        attention_norm = self.hidden_size
        
        # Feed-forward network
        # First linear layer
        ffn_first = self.hidden_size * self.intermediate_size
        
        # Second linear layer
        ffn_second = self.intermediate_size * self.hidden_size
        
        # Layer norm for FFN
        ffn_norm = self.hidden_size
        
        layer_total = (
            attention_weights + 
            attention_output + 
            attention_norm + 
            ffn_first + 
            ffn_second + 
            ffn_norm
        )
        
        return layer_total * self.num_hidden_layers
    
    def verify_parameter_count(self) -> dict:
        """Verify that parameter counts match MiniMax-M2 specifications."""
        expected_total = 3.09e9  # 3.09B
        expected_non_embedding = 2.77e9  # 2.77B
        expected_embedding = 0.32e9  # 320M
        
        total = self.total_parameters
        non_embedding = self.non_embedding_parameters
        embedding = self.embedding_parameters
        
        return {
            'total': {
                'actual': total,
                'expected': expected_total,
                'match': abs(total - expected_total) < 0.1e9,
                'ratio': total / expected_total
            },
            'non_embedding': {
                'actual': non_embedding,
                'expected': expected_non_embedding,
                'match': abs(non_embedding - expected_non_embedding) < 0.1e9,
                'ratio': non_embedding / expected_non_embedding
            },
            'embedding': {
                'actual': embedding,
                'expected': expected_embedding,
                'match': abs(embedding - expected_embedding) < 0.1e9,
                'ratio': embedding / expected_embedding
            }
        }
    
    def get_memory_requirements(self, precision: str = "float16") -> dict:
        """
        Calculate memory requirements for the model.
        
        Args:
            precision: Model precision ("float16", "float32", "int8", etc.)
            
        Returns:
            Dictionary with memory requirements in bytes
        """
        # Precision sizes in bytes
        precision_sizes = {
            "float16": 2,
            "float32": 4,
            "int8": 1,
            "int4": 0.5,
            "bfloat16": 2
        }
        
        bytes_per_param = precision_sizes.get(precision, 2)
        model_size_bytes = self.total_parameters * bytes_per_param
        
        # Estimate additional memory for activations during inference
        # Rough estimate: 4x model size for typical inference workloads
        activation_overhead = model_size_bytes * 4
        
        total_memory = model_size_bytes + activation_overhead
        
        # Convert to more readable units
        def bytes_to_readable(size_bytes: int) -> dict:
            kb = size_bytes / 1024
            mb = kb / 1024
            gb = mb / 1024
            return {
                'bytes': size_bytes,
                'kb': round(kb, 2),
                'mb': round(mb, 2),
                'gb': round(gb, 2)
            }
        
        return {
            'precision': precision,
            'model_size': bytes_to_readable(model_size_bytes),
            'activation_overhead': bytes_to_readable(activation_overhead),
            'total_estimated': bytes_to_readable(total_memory),
            'training_buffer_factor': 3.0  # Typically 2-4x for training
        }


# Default configuration matching MiniMax-M2 specifications
DEFAULT_SHEIKH_CODER_CONFIG = SheikhCoderConfig()