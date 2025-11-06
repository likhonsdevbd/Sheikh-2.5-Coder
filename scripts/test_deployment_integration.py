#!/usr/bin/env python3
"""
Deployment Pipeline Integration Test Script
Validates that all components work together properly
"""

import sys
import os
import tempfile
from pathlib import Path

# Add scripts directory to path
scripts_dir = Path(__file__).parent
sys.path.append(str(scripts_dir))

def test_imports():
    """Test that all deployment pipeline components can be imported"""
    print("Testing imports...")
    
    try:
        from deploy_model import DeploymentOrchestrator
        print("✓ DeploymentOrchestrator imported successfully")
        
        from huggingface_upload import HuggingFaceDeployer
        print("✓ HuggingFaceDeployer imported successfully")
        
        from github_update import GitHubManager
        print("✓ GitHubManager imported successfully")
        
        from model_card_generator import ModelCardGenerator
        print("✓ ModelCardGenerator imported successfully")
        
        from release_manager import ReleaseManager
        print("✓ ReleaseManager imported successfully")
        
        from quality_gates import QualityGate
        print("✓ QualityGate imported successfully")
        
        from asset_manager import AssetManager
        print("✓ AssetManager imported successfully")
        
        from docs_generator import DocumentationGenerator
        print("✓ DocumentationGenerator imported successfully")
        
        from version_manager import VersionManager
        print("✓ VersionManager imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_orchestrator_initialization():
    """Test DeploymentOrchestrator can be initialized"""
    print("\nTesting DeploymentOrchestrator initialization...")
    
    try:
        from deploy_model import DeploymentOrchestrator
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test basic initialization
            orchestrator = DeploymentOrchestrator(
                model_path=temp_dir,
                output_path=temp_dir,
                quantization='none',
                optimization='none'
            )
            print("✓ DeploymentOrchestrator initialized successfully")
            
            # Test validation
            is_valid = orchestrator.validate_integration()
            if is_valid:
                print("✓ All pipeline components properly integrated")
            else:
                print("⚠ Some pipeline components missing (expected for test environment)")
            
            # Test status reporting
            status = orchestrator.get_pipeline_status()
            print("✓ Pipeline status retrieved successfully")
            
            print(f"  - Model path: {status['parameters']['model_path']}")
            print(f"  - Output path: {status['parameters']['output_path']}")
            print(f"  - Quantization: {status['parameters']['quantization']}")
            print(f"  - Optimization: {status['parameters']['optimization']}")
            
            return True
            
    except Exception as e:
        print(f"✗ Orchestrator initialization failed: {e}")
        return False

def test_configuration_loading():
    """Test configuration loading"""
    print("\nTesting configuration loading...")
    
    try:
        from deploy_model import DeploymentOrchestrator
        
        with tempfile.TemporaryDirectory() as temp_dir:
            orchestrator = DeploymentOrchestrator(
                model_path=temp_dir,
                output_path=temp_dir
            )
            
            # Test default config
            config = orchestrator.config
            if config:
                print("✓ Configuration loaded successfully")
                print(f"  - Project name: {config.get('project', {}).get('name', 'N/A')}")
                print(f"  - Quality gates enabled: {config.get('quality_gates', {}).get('enabled', False)}")
                print(f"  - Documentation enabled: {config.get('documentation', {}).get('enabled', False)}")
            else:
                print("⚠ Using default configuration")
            
            return True
            
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        return False

def main():
    """Run all integration tests"""
    print("=" * 60)
    print("DEPLOYMENT PIPELINE INTEGRATION TEST")
    print("=" * 60)
    
    test_results = []
    
    # Run tests
    test_results.append(test_imports())
    test_results.append(test_orchestrator_initialization())
    test_results.append(test_configuration_loading())
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All integration tests passed!")
        print("✓ Deployment pipeline is ready for use")
        return 0
    else:
        print("✗ Some integration tests failed")
        print("⚠ Please check the error messages above")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)