#!/usr/bin/env python3
"""
Simple validation script for Sheikh-2.5-Coder MiniMax-M2 implementation
"""

import sys
import os
import torch

def test_basic_functionality():
    """Test basic functionality without complex operations."""
    print("🔧 Testing Basic Functionality...")
    
    try:
        # Add current directory to path
        sys.path.insert(0, os.getcwd())
        
        # Test imports
        print("   Importing...")
        from src import SheikhCoderConfig, SheikhCoderForCausalLM, SheikhCoderTokenizer
        print("   ✅ Imports successful")
        
        # Test configuration
        print("   Creating config...")
        config = SheikhCoderConfig()
        print(f"   ✅ Config created - Total params: {config.total_parameters:,}")
        
        # Test model creation
        print("   Creating model...")
        model = SheikhCoderForCausalLM(config)
        print(f"   ✅ Model created - Actual params: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test simple forward pass
        print("   Testing forward pass...")
        model.eval()
        with torch.no_grad():
            # Small batch
            input_ids = torch.randint(0, config.vocab_size, (1, 5))
            print(f"   Input shape: {input_ids.shape}")
            outputs = model(input_ids)
            print(f"   ✅ Forward pass successful - Output shape: {outputs.logits.shape}")
        
        # Test tokenizer (simplified)
        print("   Testing tokenizer...")
        tokenizer = SheikhCoderTokenizer()
        text = "Hello world"
        tokens = tokenizer.tokenize(text)
        print(f"   ✅ Tokenization working - '{text}' -> {len(tokens)} tokens")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Sheikh-2.5-Coder Basic Validation")
    print("=" * 40)
    
    success = test_basic_functionality()
    
    if success:
        print("\n🎉 Basic functionality test PASSED!")
        sys.exit(0)
    else:
        print("\n⚠️  Basic functionality test FAILED!")
        sys.exit(1)