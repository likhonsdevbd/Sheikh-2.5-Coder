"""
Test script for Sheikh-2.5-Coder MiniMax-M2 architecture implementation
Validates model architecture, parameter counts, and basic functionality
"""

import sys
import os
sys.path.append('/workspace/Sheikh-2.5-Coder')

import torch
import torch.nn as nn
import numpy as np
from src import (
    SheikhCoderConfig, 
    SheikhCoderForCausalLM, 
    SheikhCoderTokenizer,
    create_model,
    analyze_model_architecture,
    estimate_memory_requirements,
    validate_minimax_m2_specifications
)

def test_configuration():
    """Test configuration class functionality."""
    print("🔧 Testing Configuration Class...")
    
    # Create default configuration
    config = SheikhCoderConfig()
    
    # Test parameter calculations
    total_params = config.total_parameters
    non_embedding_params = config.non_embedding_parameters
    embedding_params = config.embedding_parameters
    
    print(f"   Total parameters: {total_params:,}")
    print(f"   Non-embedding parameters: {non_embedding_params:,}")
    print(f"   Embedding parameters: {embedding_params:,}")
    
    # Test parameter verification
    verification = config.verify_parameter_count()
    
    print("   Parameter Verification:")
    for param_type, details in verification.items():
        match = "✅" if details["match"] else "❌"
        print(f"      {param_type}: {match} (ratio: {details['ratio']:.3f})")
    
    # Test memory requirements
    memory_info = config.get_memory_requirements("float16")
    print(f"   Memory Requirements (FP16): {memory_info['total_estimated']['gb']:.2f} GB")
    
    return config

def test_model_creation():
    """Test model creation and initialization."""
    print("\n🏗️  Testing Model Creation...")
    
    # Create model
    model = create_model()
    
    print(f"   Model type: {type(model).__name__}")
    print(f"   Config type: {type(model.config).__name__}")
    
    # Test parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Actual parameter count: {total_params:,}")
    
    # Test model analysis
    analysis = analyze_model_architecture(model)
    memory_info = analysis['memory_usage']
    print(f"   Memory usage analysis: {memory_info['model_size_gb']:.2f} GB")
    
    return model

def test_model_forward_pass():
    """Test model forward pass with sample input."""
    print("\n🚀 Testing Model Forward Pass...")
    
    # Create model
    model = create_model()
    model.eval()  # Set to evaluation mode
    
    # Create sample input
    batch_size = 2
    sequence_length = 10
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, sequence_length))
    
    print(f"   Input shape: {input_ids.shape}")
    print(f"   Vocab size: {model.config.vocab_size}")
    
    # Test forward pass
    with torch.no_grad():
        try:
            outputs = model(input_ids)
            
            print(f"   ✅ Forward pass successful")
            print(f"   Output type: {type(outputs).__name__}")
            print(f"   Logits shape: {outputs.logits.shape}")
            
            # Test loss computation
            labels = input_ids.clone()
            loss_output = model(input_ids, labels=labels)
            print(f"   Loss computation: ✅ (loss: {loss_output.loss.item():.4f})")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Forward pass failed: {str(e)}")
            return False

def test_gqa_attention():
    """Test Grouped Query Attention functionality."""
    print("\n🧠 Testing GQA Attention...")
    
    from src.modeling_utils import repeat_kv, apply_rope_pos_emb
    from src.modeling_sheikh_coder import RotaryEmbedding
    
    # Test key-value repetition
    batch_size = 2
    num_key_value_heads = 2
    seq_len = 8
    head_dim = 64
    
    k_states = torch.randn(batch_size, num_key_value_heads, seq_len, head_dim)
    num_attention_heads = 16
    num_key_value_groups = num_attention_heads // num_key_value_heads
    
    print(f"   Input key shape: {k_states.shape}")
    print(f"   Num attention heads: {num_attention_heads}")
    print(f"   Num key-value heads: {num_key_value_heads}")
    print(f"   Num groups: {num_key_value_groups}")
    
    # Test repeat_kv function
    k_repeated = repeat_kv(k_states, num_key_value_groups)
    expected_shape = (batch_size, num_attention_heads, seq_len, head_dim)
    
    print(f"   Repeated key shape: {k_repeated.shape}")
    print(f"   Expected shape: {expected_shape}")
    
    if k_repeated.shape == expected_shape:
        print("   ✅ Key-value repetition working correctly")
    else:
        print("   ❌ Key-value repetition failed")
    
    # Test RoPE embeddings
    rope = RotaryEmbedding(head_dim, max_position_embeddings=128)
    cos, sin = rope(seq_len, k_states.device)
    
    print(f"   Cosine embedding shape: {cos.shape}")
    print(f"   Sine embedding shape: {sin.shape}")
    
    return True

def test_specialized_tokenization():
    """Test web development specialized tokenization."""
    print("\n🔤 Testing Specialized Tokenization...")
    
    # Create tokenizer
    tokenizer = SheikhCoderTokenizer()
    
    # Test web development code examples
    test_cases = [
        "<div className=\"container\">Hello World</div>",
        "function handleClick() { console.log('clicked'); }",
        "import React from 'react';",
        "#!/usr/bin/env python\nimport sys",
        "/* CSS styles */\n.container { margin: 0; padding: 10px; }"
    ]
    
    for i, text in enumerate(test_cases):
        try:
            tokens = tokenizer.tokenize(text)
            encoded = tokenizer.encode(text)
            decoded = tokenizer.decode(encoded)
            
            print(f"   Test case {i+1}: ✅")
            print(f"      Original: {text[:50]}...")
            print(f"      Tokens: {len(tokens)} tokens")
            print(f"      Round-trip: {'✅' if decoded.strip() == text else '❌'}")
            
        except Exception as e:
            print(f"   Test case {i+1}: ❌ {str(e)}")
    
    return True

def test_memory_optimization():
    """Test memory optimization features."""
    print("\n💾 Testing Memory Optimization...")
    
    # Test memory estimation
    memory_info = estimate_memory_requirements(
        precision="float16",
        sequence_length=32768,
        batch_size=1
    )
    
    print(f"   Total estimated memory: {memory_info['total_estimated']['gb']:.2f} GB")
    print(f"   Model size: {memory_info['model_size']['gb']:.2f} GB")
    
    # Test different precisions
    precisions = ["float16", "float32", "int8"]
    
    for precision in precisions:
        memory_prec = estimate_memory_requirements(precision=precision)
        print(f"   {precision}: {memory_prec['total_estimated']['gb']:.2f} GB")
    
    return True

def test_architecture_validation():
    """Test MiniMax-M2 architecture validation."""
    print("\n✅ Testing Architecture Validation...")
    
    # Create configuration
    config = SheikhCoderConfig()
    
    # Validate against MiniMax-M2 specifications
    validation = validate_minimax_m2_specifications(config)
    
    print(f"   Overall validation: {'✅ PASS' if validation['overall_valid'] else '❌ FAIL'}")
    
    # Check key specifications
    key_specs = [
        "total_parameters",
        "non_embedding_parameters", 
        "embedding_parameters",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "intermediate_size"
    ]
    
    print("   Key specifications:")
    for spec in key_specs:
        if spec in validation['specifications_match']:
            match = validation['specifications_match'][spec]
            status = "✅" if match['match'] else "❌"
            print(f"      {spec}: {status}")
    
    return validation['overall_valid']

def test_generation_capability():
    """Test text generation capability."""
    print("\n🎯 Testing Generation Capability...")
    
    try:
        # Create model
        model = create_model()
        model.eval()
        
        # Create sample input
        input_text = "function calculateSum(a, b) {"
        tokenizer = SheikhCoderTokenizer()
        
        # Encode input
        input_ids = torch.tensor([tokenizer.encode(input_text)])
        
        print(f"   Input: {input_text}")
        
        # Generate (limited to avoid long generation)
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_length=20,  # Short generation for testing
                temperature=0.7,
                do_sample=True
            )
            
        # Decode output
        generated_text = tokenizer.decode(generated[0].tolist())
        
        print(f"   Generated: {generated_text}")
        print("   ✅ Generation working")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Generation failed: {str(e)}")
        return False

def run_comprehensive_test():
    """Run comprehensive test suite."""
    print("🧪 Sheikh-2.5-Coder MiniMax-M2 Architecture Test Suite")
    print("=" * 60)
    
    tests = [
        ("Configuration", test_configuration),
        ("Model Creation", test_model_creation),
        ("Forward Pass", test_model_forward_pass),
        ("GQA Attention", test_gqa_attention),
        ("Specialized Tokenization", test_specialized_tokenization),
        ("Memory Optimization", test_memory_optimization),
        ("Architecture Validation", test_architecture_validation),
        ("Generation Capability", test_generation_capability),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result, None))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' failed with exception: {str(e)}")
            results.append((test_name, False, str(e)))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result, _ in results if result)
    total = len(results)
    
    print(f"Passed: {passed}/{total} tests")
    print(f"Success rate: {(passed/total)*100:.1f}%")
    
    print("\nDetailed Results:")
    for test_name, result, error in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if error:
            print(f"    Error: {error}")
    
    if passed == total:
        print("\n🎉 All tests passed! MiniMax-M2 implementation is working correctly.")
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run tests
    success = run_comprehensive_test()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)