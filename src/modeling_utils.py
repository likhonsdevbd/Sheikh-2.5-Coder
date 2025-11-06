"""
Utility functions for Sheikh-2.5-Coder model operations
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Union, List
import warnings


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates half the hidden dims of the input for RoPE application.
    
    Args:
        x: Input tensor to rotate
        
    Returns:
        Rotated tensor with half dimensions
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    x: torch.Tensor,
    rope_type: str = "default",
) -> torch.Tensor:
    """
    Apply rotary positional embeddings to the input.
    
    Args:
        x: Input tensor to apply RoPE to
        rope_type: Type of RoPE to apply ("default" supported)
        
    Returns:
        Tensor with RoPE applied
    """
    return rotate_half(x)


def apply_rope_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary positional embeddings to query and key tensors.
    
    Args:
        q: Query tensor
        k: Key tensor  
        cos: Cosine positional encoding
        sin: Sine positional encoding
        
    Returns:
        Tuple of (q_with_pe, k_with_pe) with positional embeddings applied
    """
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rope_v2_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary positional embeddings using v2 implementation.
    
    Args:
        q: Query tensor
        k: Key tensor
        cos: Cosine positional encoding
        sin: Sine positional encoding
        
    Returns:
        Tuple of (q_with_pe, k_with_pe) with positional embeddings applied
    """
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


def rmsnorm_forward_pre(self, x: torch.Tensor) -> torch.Tensor:
    """
    RMSNorm forward pass (pre-normalization version).
    
    Args:
        x: Input tensor
        
    Returns:
        Normalized tensor
    """
    norm_rms = x.norm(p=2, dim=-1, keepdim=True) * (1.0 / math.sqrt(x.size(-1)))
    x_norm = x / norm_rms
    return x_norm


def rmsnorm_forward_post(self, x_norm: torch.Tensor) -> torch.Tensor:
    """
    RMSNorm forward pass (post-normalization version).
    
    Args:
        x_norm: Normalized input tensor
        
    Returns:
        Scaled tensor with epsilon
    """
    return x_norm * self.weight


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    Efficient implementation of layer normalization that is often
    used in modern transformer architectures.
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        """
        Initialize RMSNorm.
        
        Args:
            hidden_size: Size of hidden dimensions
            eps: Epsilon for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for RMSNorm.
        
        Args:
            x: Input tensor
            
        Returns:
            Normalized and scaled tensor
        """
        norm_rms = x.norm(p=2, dim=-1, keepdim=True) * (1.0 / math.sqrt(x.size(-1)))
        x_norm = x / norm_rms
        return x_norm * self.weight


def make_causal_mask(
    shape: Tuple[int, int],
    dtype: torch.dtype = torch.bool,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Create a causal attention mask.
    
    Args:
        shape: Shape of the mask (target_length, source_length)
        dtype: Data type of the mask
        device: Device to place the mask on
        
    Returns:
        Causal attention mask
    """
    mask = torch.triu(
        torch.ones(shape, dtype=dtype, device=device),
        diagonal=1
    )
    return mask


def extend_causal_mask(
    mask: torch.Tensor,
    x: torch.Tensor,
    tgt_len: Optional[int] = None,
) -> torch.Tensor:
    """
    Extend causal mask for input of size x.
    
    Args:
        mask: Existing mask
        x: Input tensor
        tgt_len: Target length for the mask
        
    Returns:
        Extended causal mask
    """
    if tgt_len is None:
        tgt_len = x.shape[1]
    return make_causal_mask(
        (mask.shape[0], tgt_len),
        dtype=mask.dtype,
        device=mask.device
    )


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key/value hidden states for grouped query attention.
    
    Args:
        hidden_states: Key/value hidden states to repeat
        n_rep: Number of times to repeat
        
    Returns:
        Repeated hidden states
    """
    if n_rep == 1:
        return hidden_states
    
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    hidden_states = hidden_states.reshape(
        batch, num_key_value_heads * n_rep, slen, head_dim
    )
    return hidden_states


def apply_rotary_pos_emb_single(
    x: torch.Tensor,
    position_ids: torch.Tensor,
    base: float = 10000.0,
    seq_len: Optional[int] = None,
) -> torch.Tensor:
    """
    Apply rotary positional embeddings to a single tensor.
    
    Args:
        x: Input tensor
        position_ids: Position IDs
        base: Base value for rotary embeddings
        seq_len: Sequence length
        
    Returns:
        Tensor with rotary positional embeddings applied
    """
    seq_len = x.shape[-2]
    
    if seq_len is None:
        seq_len = x.shape[-2]
    
    inv_freq = 1.0 / (
        base ** (torch.arange(0, x.shape[-1]//2, dtype=torch.float, device=x.device) / (x.shape[-1]//2))
    )
    t = torch.arange(seq_len, device=x.device, dtype=torch.float)
    
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype=x.dtype)
    sin = emb.sin().to(dtype=x.dtype)
    
    return x * cos + rotate_half(x) * sin


def get_memory_usage(model: nn.Module) -> dict:
    """
    Get memory usage information for a model.
    
    Args:
        model: PyTorch model to analyze
        
    Returns:
        Dictionary with memory usage information
    """
    param_count = sum(p.numel() for p in model.parameters())
    trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate model size in MB (assuming float32)
    model_size_mb = param_count * 4 / (1024 * 1024)
    
    return {
        'total_parameters': param_count,
        'trainable_parameters': trainable_param_count,
        'model_size_mb': round(model_size_mb, 2),
        'model_size_gb': round(model_size_mb / 1024, 2),
        'parameter_breakdown': {
            'non_trainable': param_count - trainable_param_count,
            'percentage_trainable': round((trainable_param_count / param_count) * 100, 2)
        }
    }


def get_model_flops(model: nn.Module, sequence_length: int, batch_size: int) -> dict:
    """
    Estimate FLOPs for a model forward pass.
    
    Args:
        model: PyTorch model
        sequence_length: Input sequence length
        batch_size: Batch size
        
    Returns:
        Dictionary with FLOP estimates
    """
    # This is a rough estimate based on transformer architecture
    config = getattr(model, 'config', None)
    if config is None:
        warnings.warn("Model config not found, using rough estimates")
        return {'flops_estimate': 'N/A - config not found'}
    
    num_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    head_dim = hidden_size // num_heads
    intermediate_size = config.intermediate_size
    
    # Attention FLOPs
    # QKV projection: 3 * hidden_size * hidden_size
    # Attention matrix: (B, H, L, L) = batch_size * num_heads * sequence_length^2
    # Output projection: hidden_size * hidden_size
    
    attention_flops = (
        (3 * hidden_size * hidden_size) +  # QKV projection
        (batch_size * num_heads * sequence_length * sequence_length * head_dim * 2) +  # Attention computation
        (hidden_size * hidden_size)  # Output projection
    ) * sequence_length * num_layers
    
    # Feed-forward FLOPs
    # First linear: hidden_size * intermediate_size
    # Second linear: intermediate_size * hidden_size
    
    ffn_flops = (
        (hidden_size * intermediate_size) + (intermediate_size * hidden_size)
    ) * sequence_length * num_layers
    
    total_flops = (attention_flops + ffn_flops) * batch_size
    
    return {
        'total_flops': total_flops,
        'attention_flops': attention_flops * batch_size,
        'ffn_flops': ffn_flops * batch_size,
        'flops_per_token': total_flops / (batch_size * sequence_length),
        'human_readable': {
            'total_gflops': round(total_flops / 1e9, 2),
            'total_tflops': round(total_flops / 1e12, 2)
        }
    }


def get_layer_statistics(model: nn.Module) -> dict:
    """
    Get detailed statistics about model layers.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with layer statistics
    """
    layer_count = {}
    param_count = {}
    total_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            layer_type = 'Linear'
        elif isinstance(module, RMSNorm):
            layer_type = 'RMSNorm'
        elif isinstance(module, nn.Embedding):
            layer_type = 'Embedding'
        else:
            layer_type = type(module).__name__
        
        layer_count[layer_type] = layer_count.get(layer_type, 0) + 1
        
        if hasattr(module, 'weight'):
            param_count[layer_type] = param_count.get(layer_type, 0) + module.weight.numel()
        if hasattr(module, 'bias') and module.bias is not None:
            param_count[layer_type] = param_count.get(layer_type, 0) + module.bias.numel()
        
        total_params += sum(p.numel() for p in module.parameters() if p.requires_grad or not p.is_leaf)
    
    return {
        'layer_counts': layer_count,
        'parameter_counts': param_count,
        'total_layers': sum(layer_count.values()),
        'parameter_breakdown': {
            layer: {
                'count': count,
                'percentage': round((count / total_params) * 100, 2) if total_params > 0 else 0
            }
            for layer, count in param_count.items()
        }
    }


def validate_model_architecture(model: nn.Module, expected_config: dict) -> List[str]:
    """
    Validate that a model matches expected architecture specifications.
    
    Args:
        model: Model to validate
        expected_config: Expected configuration parameters
        
    Returns:
        List of validation issues (empty if all good)
    """
    issues = []
    
    config = getattr(model, 'config', None)
    if config is None:
        issues.append("Model does not have a config attribute")
        return issues
    
    # Check key parameters
    checks = {
        'hidden_size': config.hidden_size,
        'num_hidden_layers': config.num_hidden_layers,
        'num_attention_heads': config.num_attention_heads,
        'num_key_value_heads': config.num_key_value_heads,
        'max_position_embeddings': config.max_position_embeddings,
        'rope_theta': config.rope_theta,
        'rms_norm_epsilon': config.rms_norm_epsilon,
        'intermediate_size': config.intermediate_size,
    }
    
    for param, actual_value in checks.items():
        expected_value = expected_config.get(param)
        if expected_value is not None and actual_value != expected_value:
            issues.append(f"Parameter {param}: expected {expected_value}, got {actual_value}")
    
    # Check parameter counts
    actual_total = config.total_parameters
    expected_total = expected_config.get('total_parameters')
    if expected_total is not None:
        # Allow for some tolerance in parameter count due to implementation differences
        tolerance = 0.1e9  # 100M parameters
        if abs(actual_total - expected_total) > tolerance:
            issues.append(f"Total parameters: expected {expected_total}, got {actual_total}")
    
    return issues


def optimize_for_inference(model: nn.Module, mode: str = "speed") -> nn.Module:
    """
    Optimize model for inference based on specified mode.
    
    Args:
        model: Model to optimize
        mode: Optimization mode ("speed", "memory", "balanced")
        
    Returns:
        Optimized model
    """
    if mode == "memory":
        # Enable gradient checkpointing-like behavior for inference
        for module in model.modules():
            if hasattr(module, 'requires_grad'):
                module.requires_grad = False
                
    elif mode == "speed":
        # Remove training-specific components
        model.eval()
        
    model.gradient_checkpointing_enable = False
    
    return model


def get_attention_pattern_analysis(attention_weights: torch.Tensor) -> dict:
    """
    Analyze attention patterns from attention weights.
    
    Args:
        attention_weights: Attention weights tensor (batch, heads, seq_len, seq_len)
        
    Returns:
        Dictionary with attention pattern analysis
    """
    # Calculate attention entropy
    attention_probs = torch.softmax(attention_weights, dim=-1)
    entropy = -torch.sum(attention_probs * torch.log2(attention_probs + 1e-9), dim=-1)
    
    # Calculate attention concentration (attention head specialization)
    attention_concentration = torch.var(attention_probs, dim=-1)
    
    # Average across heads and batch
    avg_entropy = entropy.mean().item()
    avg_concentration = attention_concentration.mean().item()
    
    # Calculate head specialization scores
    head_specialization = torch.std(attention_probs.mean(dim=-1), dim=-1).mean().item()
    
    return {
        'average_entropy': round(avg_entropy, 4),
        'average_concentration': round(avg_concentration, 6),
        'head_specialization_score': round(head_specialization, 6),
        'attention_stats': {
            'min_entropy': round(entropy.min().item(), 4),
            'max_entropy': round(entropy.max().item(), 4),
            'min_concentration': round(attention_concentration.min().item(), 6),
            'max_concentration': round(attention_concentration.max().item(), 6)
        }
    }