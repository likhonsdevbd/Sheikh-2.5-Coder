"""
Sheikh-2.5-Coder Model Implementation
Complete MiniMax-M2 architecture with GQA attention and specialized features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List, Union, Any
import math
import warnings
from dataclasses import dataclass

from .configuration_sheikh_coder import SheikhCoderConfig
from .modeling_utils import (
    RMSNorm, apply_rope_pos_emb, repeat_kv, 
    get_memory_usage, get_model_flops, validate_model_architecture,
    optimize_for_inference, get_layer_statistics
)


@dataclass
class ModelOutput:
    """Output class for model forward pass."""
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    past_key_values: Optional[List[Tuple[torch.FloatTensor]]] = None


@dataclass
class CausalLMOutputWithPast(ModelOutput):
    """Output for causal language modeling."""
    loss: Optional[torch.FloatTensor] = None


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) implementation.
    
    Efficient implementation of rotary position embeddings for transformer models.
    """
    
    def __init__(self, dim: int, max_position_embeddings: int = 32768, base: float = 10000.0):
        """
        Initialize RoPE.
        
        Args:
            dim: Dimensionality of embeddings
            max_position_embeddings: Maximum sequence length
            base: Base value for rotary embeddings
        """
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Compute inverse frequency
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Build frequency tensors
        self._build_frequency_tensors(max_position_embeddings)
    
    def _build_frequency_tensors(self, max_position_embeddings: int):
        """Build frequency tensors for RoPE."""
        t = torch.arange(max_position_embeddings, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_cached = emb.cos()[:max_position_embeddings]
        sin_cached = emb.sin()[:max_position_embeddings]
        
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)
    
    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cos and sin embeddings for given sequence length."""
        if seq_len > self.max_position_embeddings:
            self._build_frequency_tensors(seq_len)
        
        return (
            self.cos_cached[:seq_len].to(dtype=dtype, device=device),
            self.sin_cached[:seq_len].to(dtype=dtype, device=device),
        )


class MLP(nn.Module):
    """
    Feed-forward network (MLP) for transformer layers.
    
    Implements the standard transformer MLP with GELU activation.
    """
    
    def __init__(self, config: SheikhCoderConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = F.gelu
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MLP.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        
        # Gated activation
        activated = self.act_fn(gate) * up
        output = self.down_proj(activated)
        
        return output


class Attention(nn.Module):
    """
    Grouped Query Attention (GQA) implementation.
    
    Efficient attention implementation that supports grouped query attention
    for memory efficiency.
    """
    
    def __init__(self, config: SheikhCoderConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_key_value_groups
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=False)
        
        # RoPE embedding
        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass through attention layer.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask tensor
            position_ids: Position IDs for RoPE
            past_key_value: Cached key-value pairs for inference
            use_cache: Whether to use caching
            output_attentions: Whether to output attention weights
            
        Returns:
            Tuple of (output, attention_weights, past_key_values)
        """
        bsz, q_len, _ = hidden_states.size()
        
        # Compute query, key, value projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for attention computation
        query_states = query_states.view(bsz, q_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        if position_ids is not None:
            cos, sin = self.rotary_emb(q_len, device=hidden_states.device, dtype=query_states.dtype)
            query_states, key_states = apply_rope_pos_emb(query_states, key_states, cos, sin)
        
        # Handle past key values
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat((past_key, key_states), dim=2)
            value_states = torch.cat((past_value, value_states), dim=2)
        
        # Repeat key and value for GQA
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        # Compute attention
        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states,
            attn_mask=attention_mask,
            is_causal=True if attention_mask is None else False,
            dropout_p=0.0 if not self.training else 0.1,
        )
        
        # Reshape attention output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.num_attention_heads * self.head_dim)
        
        # Output projection
        output = self.o_proj(attn_output)
        
        if not output_attentions:
            attn_weights = None
        
        # Prepare cache for next forward pass
        if use_cache:
            past_key_value = (key_states, value_states)
        else:
            past_key_value = None
        
        return output, attn_weights, past_key_value


class DecoderLayer(nn.Module):
    """
    Transformer decoder layer with GQA attention and MLP.
    
    Implements a complete transformer decoder layer with residual connections,
    layer normalization, and attention mechanisms.
    """
    
    def __init__(self, config: SheikhCoderConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Attention(config, layer_idx)
        self.mlp = MLP(config)
        
        # Layer norms
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_epsilon)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_epsilon)
        
        self.dropout = nn.Dropout(config.dropout if hasattr(config, 'dropout') else 0.0)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass through decoder layer.
        
        Args:
            hidden_states: Input tensor
            attention_mask: Attention mask
            position_ids: Position IDs
            past_key_value: Cached key-value pairs
            use_cache: Whether to use caching
            output_attentions: Whether to output attention weights
            
        Returns:
            Tuple of (hidden_states, presents)
        """
        # Self-attention
        attn_output, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        
        # Residual connection
        hidden_states = hidden_states + self.dropout(attn_output)
        
        # Feed-forward network
        ffn_output = self.mlp(self.post_attention_layernorm(hidden_states))
        
        # Residual connection
        hidden_states = hidden_states + self.dropout(ffn_output)
        
        return hidden_states, present_key_value


class SheikhCoderModel(nn.Module):
    """
    Sheikh-2.5-Coder Model with complete MiniMax-M2 architecture.
    
    This model implements the full transformer architecture with:
    - 36 transformer layers
    - GQA attention (16 Q heads, 2 KV heads)
    - 32,768 token context length
    - RoPE positional embeddings
    - RMSNorm layer normalization
    - Memory-efficient attention computation
    """
    
    def __init__(self, config: SheikhCoderConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        # Word embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        
        # Transformer layers
        self.layers = nn.ModuleList(
            [DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        
        # Final layer norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_epsilon)
        
        # Initialize weights
        self.init_weights()
        
        # Gradient checkpointing
        self.gradient_checkpointing = False
        
    def init_weights(self):
        """Initialize model weights."""
        # Initialize embeddings
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=self.config.initializer_range)
        
        # Initialize transformer layers
        for layer in self.layers:
            # Initialize linear layers
            for name, param in layer.named_parameters():
                if 'weight' in name and len(param.shape) >= 2:
                    nn.init.normal_(param, mean=0.0, std=self.config.initializer_range)
                elif 'weight' in name:
                    nn.init.ones_(param)
        
        # Initialize layer norms
        for layer in self.layers:
            nn.init.ones_(layer.input_layernorm.weight)
            nn.init.ones_(layer.post_attention_layernorm.weight)
        
        nn.init.ones_(self.norm.weight)
        
        # Initialize final projection (will be set by language model head)
        self.final_logits_softcapping = None
        
    def get_input_embeddings(self) -> nn.Embedding:
        """Get input embeddings."""
        return self.embed_tokens
    
    def set_input_embeddings(self, value: nn.Embedding):
        """Set input embeddings."""
        self.embed_tokens = value
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: bool = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
    ) -> Union[Tuple[torch.Tensor], ModelOutput]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            position_ids: Position IDs
            past_key_values: Cached key-value pairs
            use_cache: Whether to use caching
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return output as dictionary
            
        Returns:
            ModelOutput or tuple of tensors
        """
        if use_cache is None:
            use_cache = self.config.use_cache
        
        if output_attentions is None:
            output_attentions = False
        
        if output_hidden_states is None:
            output_hidden_states = False
        
        if return_dict is None:
            return_dict = True
        
        # Handle input validation
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided")
        
        batch_size, seq_length = input_ids.shape if input_ids is not None else (inputs_embeds.shape[0], inputs_embeds.shape[1])
        
        # Prepare attention mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids) if input_ids is not None else torch.ones_like(inputs_embeds)
        
        # Create position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_length, device=(input_ids or inputs_embeds).device).unsqueeze(0).expand(batch_size, -1)
        
        # Get input embeddings
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)
        
        # Create causal attention mask
        if attention_mask.dim() == 2:
            attn_mask_2d = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_mask_2d = (1.0 - attn_mask_2d) * torch.finfo(self.config.hidden_size).min
            attn_mask_2d = attn_mask_2d.expand(
                attention_mask.shape[0], 1, seq_length, seq_length
            )
        else:
            attn_mask_2d = attention_mask
        
        # Initialize outputs
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        
        # Forward pass through transformer layers
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            layer_past = past_key_values[layer_idx] if past_key_values is not None else None
            layer_idx = len(self.layers)
            
            # Gradient checkpointing
            if self.gradient_checkpointing and self.training:
                raise NotImplementedError("Gradient checkpointing not implemented")
            
            layer_output = layer(
                hidden_states,
                attention_mask=attn_mask_2d,
                position_ids=position_ids,
                past_key_value=layer_past,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            
            hidden_states = layer_output[0]
            
            if use_cache:
                next_decoder_cache += (layer_output[1],)
            
            if output_attentions:
                all_self_attns += (layer_output[2],)
        
        # Final layer norm
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        # Prepare output
        if not return_dict:
            return tuple(v for v in [hidden_states, next_decoder_cache, all_hidden_states, all_self_attns] if v is not None)
        
        return ModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class SheikhCoderForCausalLM(nn.Module):
    """
    Sheikh-2.5-Coder Language Model for causal language modeling.
    
    Complete language model implementation with:
    - Language model head
    - Loss computation
    - Generation capabilities
    - Memory optimization
    """
    
    def __init__(self, config: SheikhCoderConfig):
        super().__init__()
        self.config = config
        
        # Base model
        self.model = SheikhCoderModel(config)
        
        # Language model head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.init_weights()
        
        # Tie weights if specified
        if config.tie_word_embeddings:
            self.tie_weights()
    
    def init_weights(self):
        """Initialize model weights."""
        # Initialize language model head
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=self.config.initializer_range)
        
    def tie_weights(self):
        """Tie input and output embeddings."""
        if hasattr(self.model, 'embed_tokens'):
            self.lm_head.weight = self.model.embed_tokens.weight
    
    def get_input_embeddings(self) -> nn.Embedding:
        """Get input embeddings."""
        return self.model.get_input_embeddings()
    
    def set_input_embeddings(self, value: nn.Embedding):
        """Set input embeddings."""
        self.model.set_input_embeddings(value)
    
    def get_output_embeddings(self) -> nn.Linear:
        """Get output embeddings (language model head)."""
        return self.lm_head
    
    def set_output_embeddings(self, value: nn.Linear):
        """Set output embeddings."""
        self.lm_head = value
    
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[List[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **model_kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for generation.
        
        Args:
            input_ids: Input token IDs
            past_key_values: Cached key-value pairs
            attention_mask: Attention mask
            **model_kwargs: Additional model arguments
            
        Returns:
            Dictionary with prepared inputs
        """
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
            
            # Some generation methods already provide only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: mask out the matching length
                remove_prefix_length = input_ids.shape[1] - 1
            
            input_ids = input_ids[:, remove_prefix_length:]
        
        return {
            'input_ids': input_ids,
            'past_key_values': past_key_values,
            'attention_mask': attention_mask,
            'use_cache': model_kwargs.get('use_cache', True),
        }
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: bool = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
    ) -> Union[CausalLMOutputWithPast, torch.FloatTensor]:
        """
        Forward pass through the language model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            position_ids: Position IDs
            past_key_values: Cached key-value pairs
            inputs_embeds: Input embeddings
            labels: Target labels for loss computation
            use_cache: Whether to use caching
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return output as dictionary
            
        Returns:
            CausalLMOutputWithPast or loss tensor
        """
        if return_dict is None:
            return_dict = True
        
        # Forward pass through base model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = outputs[0]
        
        # Compute logits
        logits = self.lm_head(hidden_states)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift labels for causal language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Compute loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """Reorder cache for beam search."""
        reordered_past = []
        for layer_past in past_key_values:
            reordered_past.append(
                (
                    layer_past[0].index_select(0, beam_idx),
                    layer_past[1].index_select(0, beam_idx),
                )
            )
        return reordered_past
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate text using the model.
        
        Args:
            input_ids: Input token IDs
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            repetition_penalty: Repetition penalty
            do_sample: Whether to use sampling
            **kwargs: Additional generation parameters
            
        Returns:
            Generated token IDs
        """
        self.eval()
        device = next(self.parameters()).device
        
        with torch.no_grad():
            # Prepare input
            input_ids = input_ids.to(device)
            batch_size = input_ids.size(0)
            
            # Initialize generated sequence
            generated = input_ids.clone()
            past_key_values = None
            
            # Generation loop
            for _ in range(max_length):
                # Get model output
                outputs = self(
                    input_ids=generated if past_key_values is None else generated[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                
                # Get next token logits
                next_token_logits = outputs.logits[:, -1, :]
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for i in range(batch_size):
                        for token_id in set(generated[i].tolist()):
                            if next_token_logits[i][token_id] < 0:
                                next_token_logits[i][token_id] *= repetition_penalty
                            else:
                                next_token_logits[i][token_id] /= repetition_penalty
                
                # Apply temperature and sampling
                if do_sample:
                    next_token_logits = next_token_logits / temperature
                    
                    if top_k > 0:
                        # Top-k filtering
                        indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    if top_p > 0:
                        # Top-p (nucleus) filtering
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # Remove tokens with cumulative probability above threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        for i in range(batch_size):
                            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                            next_token_logits[i][indices_to_remove] = float('-inf')
                    
                    # Sample next token
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append next token
                generated = torch.cat([generated, next_token], dim=-1)
                
                # Update past key values
                past_key_values = outputs.past_key_values
                
                # Stop if all sequences have generated EOS token
                if (next_token.squeeze(-1) == self.config.eos_token_id).all():
                    break
            
            return generated
    
    def get_model_analysis(self) -> Dict[str, Any]:
        """
        Get comprehensive model analysis.
        
        Returns:
            Dictionary with model analysis
        """
        # Get memory usage
        memory_info = get_memory_usage(self)
        
        # Get FLOPs estimates
        flops_info = get_model_flops(self, sequence_length=1024, batch_size=1)
        
        # Get layer statistics
        layer_stats = get_layer_statistics(self)
        
        # Get parameter verification
        param_verification = self.config.verify_parameter_count()
        
        return {
            'memory_usage': memory_info,
            'flops_estimation': flops_info,
            'layer_statistics': layer_stats,
            'parameter_verification': param_verification,
            'model_config': self.config.to_dict(),
        }
    
    def optimize_for_inference(self, mode: str = "speed") -> 'SheikhCoderForCausalLM':
        """
        Optimize model for inference.
        
        Args:
            mode: Optimization mode ("speed", "memory", "balanced")
            
        Returns:
            Optimized model
        """
        self.model = optimize_for_inference(self.model, mode)
        self.eval()
        return self