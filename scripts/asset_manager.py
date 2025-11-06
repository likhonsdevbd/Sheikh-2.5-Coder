#!/usr/bin/env python3
"""
Asset Management System for Model Deployment
Handles model artifacts, optimization variants, and deployment package management
"""

import os
import sys
import json
import argparse
import logging
import shutil
import zipfile
import tarfile
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass
import subprocess
import tempfile
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class AssetInfo:
    """Asset information container"""
    name: str
    path: str
    size_mb: float
    format: str
    checksum: str
    created_at: str
    purpose: str
    optimization: Optional[str] = None
    platform: Optional[str] = None
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        data = {
            'name': self.name,
            'path': self.path,
            'size_mb': self.size_mb,
            'format': self.format,
            'checksum': self.checksum,
            'created_at': self.created_at,
            'purpose': self.purpose,
            'optimization': self.optimization,
            'platform': self.platform,
            'metadata': self.metadata or {}
        }
        return data


@dataclass
class DeploymentPackage:
    """Deployment package information"""
    name: str
    version: str
    assets: List[AssetInfo]
    total_size_mb: float
    created_at: str
    package_type: str
    platforms: List[str]
    documentation: Optional[str] = None
    installation_guide: Optional[str] = None
    
    def to_dict(self) -> Dict:
        data = {
            'name': self.name,
            'version': self.version,
            'assets': [asset.to_dict() for asset in self.assets],
            'total_size_mb': self.total_size_mb,
            'created_at': self.created_at,
            'package_type': self.package_type,
            'platforms': self.platforms,
            'documentation': self.documentation,
            'installation_guide': self.installation_guide
        }
        return data


class AssetManager:
    """Asset management for deployment packages"""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.assets_dir = self.base_dir / "assets"
        self.packages_dir = self.base_dir / "packages"
        self.metadata_file = self.base_dir / "asset_metadata.json"
        
        # Create directories
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self.packages_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing metadata
        self.asset_metadata = self._load_asset_metadata()
        
        logger.info(f"Initialized asset manager at {self.base_dir}")
    
    def _load_asset_metadata(self) -> Dict:
        """Load asset metadata from file"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file) as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load asset metadata: {e}")
        
        return {'assets': {}, 'packages': {}}
    
    def _save_asset_metadata(self):
        """Save asset metadata to file"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.asset_metadata, f, indent=2)
    
    def calculate_checksum(self, file_path: Path, algorithm: str = 'sha256') -> str:
        """Calculate file checksum"""
        hash_func = hashlib.new(algorithm)
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()
    
    def register_model_asset(self, model_path: Union[str, Path], asset_name: str,
                           purpose: str = "base_model", optimization: Optional[str] = None,
                           platform: Optional[str] = None) -> AssetInfo:
        """Register a model asset"""
        logger.info(f"Registering model asset: {asset_name}")
        
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")
        
        # Calculate file size
        if model_path.is_file():
            size_mb = model_path.stat().st_size / (1024 * 1024)
        else:
            # Calculate directory size
            size_mb = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file()) / (1024 * 1024)
        
        # Calculate checksum
        if model_path.is_file():
            checksum = self.calculate_checksum(model_path)
        else:
            # For directories, create a combined checksum
            checksums = []
            for file_path in sorted(model_path.rglob('*')):
                if file_path.is_file():
                    file_checksum = self.calculate_checksum(file_path)
                    checksums.append(f"{file_path.name}:{file_checksum}")
            
            combined = "\n".join(checksums)
            checksum = hashlib.sha256(combined.encode()).hexdigest()
        
        # Determine format
        if model_path.suffix == '.safetensors':
            format_type = 'safetensors'
        elif model_path.suffix == '.bin':
            format_type = 'pytorch'
        elif model_path.suffix == '.onnx':
            format_type = 'onnx'
        else:
            format_type = 'directory'
        
        # Create asset info
        asset_info = AssetInfo(
            name=asset_name,
            path=str(model_path.absolute()),
            size_mb=size_mb,
            format=format_type,
            checksum=checksum,
            created_at=datetime.now().isoformat(),
            purpose=purpose,
            optimization=optimization,
            platform=platform,
            metadata={
                'source_path': str(model_path),
                'is_directory': model_path.is_dir()
            }
        )
        
        # Register in metadata
        self.asset_metadata['assets'][asset_name] = asset_info.to_dict()
        self._save_asset_metadata()
        
        logger.info(f"Registered asset: {asset_name} ({size_mb:.2f} MB)")
        return asset_info
    
    def create_optimized_variants(self, base_model_path: Union[str, Path], 
                                variants: List[Dict]) -> List[AssetInfo]:
        """Create optimized model variants"""
        logger.info(f"Creating {len(variants)} optimized variants")
        
        base_path = Path(base_model_path)
        created_assets = []
        
        for variant_config in variants:
            variant_name = variant_config['name']
            optimization = variant_config['optimization']
            variant_path = self.assets_dir / f"{variant_name}"
            
            logger.info(f"Creating variant: {variant_name} ({optimization})")
            
            try:
                # Create variant based on optimization type
                if optimization == 'int8':
                    asset_info = self._create_int8_variant(base_path, variant_path, variant_name)
                elif optimization == 'int4':
                    asset_info = self._create_int4_variant(base_path, variant_path, variant_name)
                elif optimization == 'onnx':
                    asset_info = self._create_onnx_variant(base_path, variant_path, variant_name)
                elif optimization == 'tensorrt':
                    asset_info = self._create_tensorrt_variant(base_path, variant_path, variant_name)
                elif optimization == 'gguf':
                    asset_info = self._create_gguf_variant(base_path, variant_path, variant_name)
                else:
                    logger.warning(f"Unknown optimization: {optimization}")
                    continue
                
                created_assets.append(asset_info)
                
            except Exception as e:
                logger.error(f"Failed to create variant {variant_name}: {str(e)}")
        
        return created_assets
    
    def _create_int8_variant(self, base_path: Path, variant_path: Path, name: str) -> AssetInfo:
        """Create INT8 quantized variant"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        from transformers import BitsAndBytesConfig
        
        variant_path.mkdir(exist_ok=True)
        
        # Load and quantize model
        model = AutoModelForCausalLM.from_pretrained(
            base_path,
            torch_dtype=torch.float16,
            device_map="cpu",  # Use CPU for quantization
            load_in_8bit=True,
            trust_remote_code=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
        
        # Save quantized model
        model.save_pretrained(variant_path)
        tokenizer.save_pretrained(variant_path)
        
        return self.register_model_asset(
            model_path=variant_path,
            asset_name=name,
            purpose="optimized_model",
            optimization="int8",
            platform="universal"
        )
    
    def _create_int4_variant(self, base_path: Path, variant_path: Path, name: str) -> AssetInfo:
        """Create INT4 quantized variant"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        from transformers import BitsAndBytesConfig
        
        variant_path.mkdir(exist_ok=True)
        
        # Load and quantize model
        model = AutoModelForCausalLM.from_pretrained(
            base_path,
            torch_dtype=torch.float16,
            device_map="cpu",
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
        
        # Save quantized model
        model.save_pretrained(variant_path)
        tokenizer.save_pretrained(variant_path)
        
        return self.register_model_asset(
            model_path=variant_path,
            asset_name=name,
            purpose="optimized_model",
            optimization="int4",
            platform="universal"
        )
    
    def _create_onnx_variant(self, base_path: Path, variant_path: Path, name: str) -> AssetInfo:
        """Create ONNX optimized variant"""
        from optimum.onnxruntime import ORTModelForCausalLM
        from transformers import AutoTokenizer
        
        variant_path.mkdir(exist_ok=True)
        
        # Export to ONNX
        ort_model = ORTModelForCausalLM.from_pretrained(
            base_path,
            export=True,
            provider="CPUExecutionProvider"
        )
        
        tokenizer = AutoTokenizer.from_pretrained(base_path)
        
        # Save ONNX model
        ort_model.save_pretrained(variant_path)
        tokenizer.save_pretrained(variant_path)
        
        return self.register_model_asset(
            model_path=variant_path,
            asset_name=name,
            purpose="optimized_model",
            optimization="onnx",
            platform="cross_platform"
        )
    
    def _create_tensorrt_variant(self, base_path: Path, variant_path: Path, name: str) -> AssetInfo:
        """Create TensorRT optimized variant"""
        # Note: This would require TensorRT installation and proper setup
        # For now, create placeholder structure
        variant_path.mkdir(exist_ok=True)
        
        # Create placeholder files
        (variant_path / "tensorrt_model.plan").touch()
        (variant_path / "config.json").touch()
        
        logger.info(f"TensorRT variant created (placeholder): {variant_path}")
        
        return self.register_model_asset(
            model_path=variant_path,
            asset_name=name,
            purpose="optimized_model",
            optimization="tensorrt",
            platform="nvidia_gpu"
        )
    
    def _create_gguf_variant(self, base_path: Path, variant_path: Path, name: str) -> AssetInfo:
        """Create GGUF format variant"""
        # Note: This would require llama.cpp or similar tools
        # For now, create placeholder structure
        variant_path.mkdir(exist_ok=True)
        
        # Create placeholder GGUF file
        gguf_file = variant_path / "model.gguf"
        gguf_file.touch()
        
        (variant_path / "tokenizer.json").touch()
        
        logger.info(f"GGUF variant created (placeholder): {variant_path}")
        
        return self.register_model_asset(
            model_path=variant_path,
            asset_name=name,
            purpose="optimized_model",
            optimization="gguf",
            platform="cpu_optimized"
        )
    
    def create_deployment_package(self, package_name: str, version: str,
                                asset_names: List[str], package_type: str = "full") -> DeploymentPackage:
        """Create deployment package from assets"""
        logger.info(f"Creating deployment package: {package_name} v{version}")
        
        # Get assets
        assets = []
        total_size = 0
        
        for asset_name in asset_names:
            if asset_name not in self.asset_metadata['assets']:
                raise ValueError(f"Asset not found: {asset_name}")
            
            asset_data = self.asset_metadata['assets'][asset_name]
            asset_info = AssetInfo(**asset_data)
            assets.append(asset_info)
            total_size += asset_info.size_mb
        
        # Create package directory
        package_dir = self.packages_dir / f"{package_name}_{version}"
        package_dir.mkdir(exist_ok=True)
        
        # Copy assets to package
        copied_assets = []
        for asset_info in assets:
            source_path = Path(asset_info.path)
            target_path = package_dir / asset_info.name
            
            if source_path.is_file():
                shutil.copy2(source_path, target_path)
            else:
                shutil.copytree(source_path, target_path, dirs_exist_ok=True)
            
            copied_assets.append(asset_info)
        
        # Create package manifest
        manifest = {
            'name': package_name,
            'version': version,
            'created_at': datetime.now().isoformat(),
            'package_type': package_type,
            'total_assets': len(assets),
            'total_size_mb': total_size,
            'assets': [asset.to_dict() for asset in assets]
        }
        
        with open(package_dir / "manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Create installation guide
        installation_guide = self._generate_installation_guide(package_name, assets)
        with open(package_dir / "INSTALLATION.md", 'w') as f:
            f.write(installation_guide)
        
        # Create package documentation
        documentation = self._generate_package_documentation(package_name, version, assets)
        with open(package_dir / "README.md", 'w') as f:
            f.write(documentation)
        
        # Create deployment package
        package = DeploymentPackage(
            name=package_name,
            version=version,
            assets=copied_assets,
            total_size_mb=total_size,
            created_at=datetime.now().isoformat(),
            package_type=package_type,
            platforms=list(set(asset.platform for asset in assets if asset.platform)),
            documentation=documentation,
            installation_guide=installation_guide
        )
        
        # Register package
        self.asset_metadata['packages'][f"{package_name}_{version}"] = package.to_dict()
        self._save_asset_metadata()
        
        logger.info(f"Deployment package created: {package_dir}")
        return package
    
    def create_archive_package(self, package: DeploymentPackage, 
                             format: str = "zip") -> str:
        """Create archive of deployment package"""
        package_name = f"{package.name}_{package.version}"
        package_dir = self.packages_dir / package_name
        
        if not package_dir.exists():
            raise ValueError(f"Package directory not found: {package_dir}")
        
        # Create archive
        if format == "zip":
            archive_path = self.packages_dir / f"{package_name}.zip"
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(package_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, package_dir.parent)
                        zipf.write(file_path, arcname)
        
        elif format == "tar.gz":
            archive_path = self.packages_dir / f"{package_name}.tar.gz"
            with tarfile.open(archive_path, 'w:gz') as tarf:
                tarf.add(package_dir, arcname=package_name)
        
        else:
            raise ValueError(f"Unsupported archive format: {format}")
        
        logger.info(f"Archive created: {archive_path}")
        return str(archive_path)
    
    def _generate_installation_guide(self, package_name: str, assets: List[AssetInfo]) -> str:
        """Generate installation guide for package"""
        guide = f"""# {package_name} Installation Guide

## Overview
This package contains optimized model assets for {package_name}.

## Contents
"""
        
        for asset in assets:
            guide += f"- **{asset.name}**: {asset.purpose} ({asset.format})\n"
            guide += f"  - Size: {asset.size_mb:.2f} MB\n"
            guide += f"  - Platform: {asset.platform or 'Universal'}\n"
            if asset.optimization:
                guide += f"  - Optimization: {asset.optimization}\n"
            guide += "\n"
        
        guide += """## Installation Methods

### Method 1: HuggingFace Hub (Recommended)
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("sheikh-team/Sheikh-2.5-Coder")
tokenizer = AutoTokenizer.from_pretrained("sheikh-team/Sheikh-2.5-Coder")
```

### Method 2: Local Installation
1. Download the package
2. Extract to your project directory
3. Load model using the local path

### Method 3: With Quantization
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# For INT8 quantization
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "sheikh-team/Sheikh-2.5-Coder",
    quantization_config=quantization_config,
    device_map="auto"
)
```
"""
        
        return guide
    
    def _generate_package_documentation(self, package_name: str, version: str, 
                                      assets: List[AssetInfo]) -> str:
        """Generate package documentation"""
        doc = f"""# {package_name} v{version} - Deployment Package

## Package Information
- **Package Name**: {package_name}
- **Version**: {version}
- **Created**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Total Size**: {sum(asset.size_mb for asset in assets):.2f} MB

## Available Assets

"""
        
        for asset in assets:
            doc += f"### {asset.name}\n"
            doc += f"- **Purpose**: {asset.purpose}\n"
            doc += f"- **Format**: {asset.format}\n"
            doc += f"- **Size**: {asset.size_mb:.2f} MB\n"
            doc += f"- **Platform**: {asset.platform or 'Universal'}\n"
            
            if asset.optimization:
                doc += f"- **Optimization**: {asset.optimization}\n"
            
            doc += f"- **Checksum**: `{asset.checksum[:16]}...`\n\n"
        
        doc += """## Usage Instructions

### Quick Start
1. Choose the appropriate asset for your platform
2. Download or use locally
3. Load using the provided examples
4. Start generating code!

### Platform-Specific Recommendations
"""
        
        # Platform recommendations
        platforms = set(asset.platform for asset in assets if asset.platform)
        for platform in platforms:
            platform_assets = [asset for asset in assets if asset.platform == platform]
            doc += f"- **{platform.title()}**: Use {platform_assets[0].name}\n"
        
        doc += """
## Support
- Documentation: [GitHub Wiki](https://github.com/sheikh-team/Sheikh-2.5-Coder/wiki)
- Issues: [GitHub Issues](https://github.com/sheikh-team/Sheikh-2.5-Coder/issues)
- Discussions: [GitHub Discussions](https://github.com/sheikh-team/Sheikh-2.5-Coder/discussions)

## License
This package is released under the MIT License.
"""
        
        return doc
    
    def list_assets(self, filter_by: Optional[Dict] = None) -> List[AssetInfo]:
        """List all registered assets"""
        assets = []
        
        for asset_name, asset_data in self.asset_metadata['assets'].items():
            asset_info = AssetInfo(**asset_data)
            
            # Apply filters
            if filter_by:
                if filter_by.get('purpose') and asset_info.purpose != filter_by['purpose']:
                    continue
                if filter_by.get('optimization') and asset_info.optimization != filter_by['optimization']:
                    continue
                if filter_by.get('platform') and asset_info.platform != filter_by['platform']:
                    continue
            
            assets.append(asset_info)
        
        return assets
    
    def list_packages(self) -> List[DeploymentPackage]:
        """List all deployment packages"""
        packages = []
        
        for package_name, package_data in self.asset_metadata['packages'].items():
            assets = [AssetInfo(**asset) for asset in package_data['assets']]
            package = DeploymentPackage(
                name=package_data['name'],
                version=package_data['version'],
                assets=assets,
                total_size_mb=package_data['total_size_mb'],
                created_at=package_data['created_at'],
                package_type=package_data['package_type'],
                platforms=package_data['platforms'],
                documentation=package_data.get('documentation'),
                installation_guide=package_data.get('installation_guide')
            )
            packages.append(package)
        
        return packages
    
    def cleanup_assets(self, older_than_days: int = 30) -> Dict[str, int]:
        """Clean up old assets"""
        logger.info(f"Cleaning up assets older than {older_than_days} days")
        
        cutoff_date = datetime.now().timestamp() - (older_than_days * 24 * 60 * 60)
        cleaned_assets = []
        cleaned_packages = []
        
        # Clean old assets
        for asset_name, asset_data in list(self.asset_metadata['assets'].items()):
            asset_date = datetime.fromisoformat(asset_data['created_at']).timestamp()
            
            if asset_date < cutoff_date:
                asset_path = Path(asset_data['path'])
                
                # Remove asset files
                try:
                    if asset_path.exists():
                        if asset_path.is_file():
                            asset_path.unlink()
                        else:
                            shutil.rmtree(asset_path)
                    
                    # Remove from metadata
                    del self.asset_metadata['assets'][asset_name]
                    cleaned_assets.append(asset_name)
                    
                except Exception as e:
                    logger.error(f"Failed to clean asset {asset_name}: {e}")
        
        # Clean old packages
        for package_name, package_data in list(self.asset_metadata['packages'].items()):
            package_date = datetime.fromisoformat(package_data['created_at']).timestamp()
            
            if package_date < cutoff_date:
                package_dir = self.packages_dir / package_name
                
                # Remove package directory
                try:
                    if package_dir.exists():
                        shutil.rmtree(package_dir)
                    
                    # Remove from metadata
                    del self.asset_metadata['packages'][package_name]
                    cleaned_packages.append(package_name)
                    
                except Exception as e:
                    logger.error(f"Failed to clean package {package_name}: {e}")
        
        # Save updated metadata
        self._save_asset_metadata()
        
        result = {
            'cleaned_assets': len(cleaned_assets),
            'cleaned_packages': len(cleaned_packages),
            'asset_names': cleaned_assets,
            'package_names': cleaned_packages
        }
        
        logger.info(f"Cleanup completed: {result}")
        return result
    
    def validate_asset_integrity(self, asset_name: str) -> Dict[str, Any]:
        """Validate asset integrity"""
        if asset_name not in self.asset_metadata['assets']:
            raise ValueError(f"Asset not found: {asset_name}")
        
        asset_data = self.asset_metadata['assets'][asset_name]
        asset_path = Path(asset_data['path'])
        
        validation_result = {
            'asset_name': asset_name,
            'exists': asset_path.exists(),
            'size_match': True,
            'checksum_match': True,
            'issues': []
        }
        
        # Check file existence
        if not asset_path.exists():
            validation_result['issues'].append("Asset file/directory not found")
            return validation_result
        
        # Check size
        if asset_path.is_file():
            current_size = asset_path.stat().st_size / (1024 * 1024)
        else:
            current_size = sum(f.stat().st_size for f in asset_path.rglob('*') if f.is_file()) / (1024 * 1024)
        
        expected_size = asset_data['size_mb']
        if abs(current_size - expected_size) > 0.1:  # Allow 0.1 MB tolerance
            validation_result['size_match'] = False
            validation_result['issues'].append(f"Size mismatch: expected {expected_size:.2f} MB, got {current_size:.2f} MB")
        
        # Check checksum
        if asset_path.is_file():
            current_checksum = self.calculate_checksum(asset_path)
        else:
            # For directories, create a combined checksum
            checksums = []
            for file_path in sorted(asset_path.rglob('*')):
                if file_path.is_file():
                    file_checksum = self.calculate_checksum(file_path)
                    checksums.append(f"{file_path.name}:{file_checksum}")
            
            combined = "\n".join(checksums)
            current_checksum = hashlib.sha256(combined.encode()).hexdigest()
        
        expected_checksum = asset_data['checksum']
        if current_checksum != expected_checksum:
            validation_result['checksum_match'] = False
            validation_result['issues'].append(f"Checksum mismatch: {current_checksum[:16]}... vs {expected_checksum[:16]}...")
        
        validation_result['valid'] = len(validation_result['issues']) == 0
        
        return validation_result
    
    def export_asset_catalog(self, output_path: str, format: str = "json"):
        """Export asset catalog"""
        catalog = {
            'generated_at': datetime.now().isoformat(),
            'total_assets': len(self.asset_metadata['assets']),
            'total_packages': len(self.asset_metadata['packages']),
            'assets': self.asset_metadata['assets'],
            'packages': self.asset_metadata['packages']
        }
        
        if format == "json":
            with open(output_path, 'w') as f:
                json.dump(catalog, f, indent=2)
        elif format == "yaml":
            with open(output_path, 'w') as f:
                yaml.dump(catalog, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Asset catalog exported to {output_path}")


def main():
    """Main function for asset management"""
    parser = argparse.ArgumentParser(description='Asset management system')
    
    parser.add_argument('--base_dir', required=True, help='Base directory for assets')
    parser.add_argument('--action', required=True,
                       choices=['register_asset', 'create_variants', 'create_package',
                               'list_assets', 'list_packages', 'cleanup', 'validate',
                               'export_catalog'],
                       help='Action to perform')
    
    # Action-specific arguments
    parser.add_argument('--asset_path', help='Path to asset')
    parser.add_argument('--asset_name', help='Asset name')
    parser.add_argument('--purpose', help='Asset purpose')
    parser.add_argument('--optimization', help='Optimization type')
    parser.add_argument('--platform', help='Target platform')
    parser.add_argument('--variants_config', help='Variants configuration file')
    parser.add_argument('--package_name', help='Package name')
    parser.add_argument('--version', help='Package version')
    parser.add_argument('--asset_names', nargs='+', help='Asset names for package')
    parser.add_argument('--package_type', default='full', help='Package type')
    parser.add_argument('--filter_by', help='Filter criteria (JSON)')
    parser.add_argument('--output_path', help='Output path')
    parser.add_argument('--format', default='json', help='Export format')
    
    args = parser.parse_args()
    
    # Initialize asset manager
    manager = AssetManager(args.base_dir)
    
    # Execute action
    if args.action == 'register_asset':
        if not args.asset_path or not args.asset_name:
            print("Error: --asset_path and --asset_name required")
            return 1
        
        asset_info = manager.register_model_asset(
            model_path=args.asset_path,
            asset_name=args.asset_name,
            purpose=args.purpose or "base_model",
            optimization=args.optimization,
            platform=args.platform
        )
        
        print(json.dumps(asset_info.to_dict(), indent=2, default=str))
        
    elif args.action == 'create_variants':
        if not args.asset_path or not args.variants_config:
            print("Error: --asset_path and --variants_config required")
            return 1
        
        with open(args.variants_config) as f:
            variants_config = json.load(f)
        
        variants = variants_config.get('variants', [])
        created_assets = manager.create_optimized_variants(args.asset_path, variants)
        
        print(json.dumps([asset.to_dict() for asset in created_assets], indent=2, default=str))
        
    elif args.action == 'create_package':
        if not args.package_name or not args.version or not args.asset_names:
            print("Error: --package_name, --version, and --asset_names required")
            return 1
        
        package = manager.create_deployment_package(
            package_name=args.package_name,
            version=args.version,
            asset_names=args.asset_names,
            package_type=args.package_type
        )
        
        print(json.dumps(package.to_dict(), indent=2, default=str))
        
    elif args.action == 'list_assets':
        filter_by = None
        if args.filter_by:
            filter_by = json.loads(args.filter_by)
        
        assets = manager.list_assets(filter_by=filter_by)
        print(json.dumps([asset.to_dict() for asset in assets], indent=2, default=str))
        
    elif args.action == 'list_packages':
        packages = manager.list_packages()
        print(json.dumps([pkg.to_dict() for pkg in packages], indent=2, default=str))
        
    elif args.action == 'cleanup':
        older_than_days = 30
        if args.filter_by:
            filter_data = json.loads(args.filter_by)
            older_than_days = filter_data.get('older_than_days', 30)
        
        result = manager.cleanup_assets(older_than_days=older_than_days)
        print(json.dumps(result, indent=2, default=str))
        
    elif args.action == 'validate':
        if not args.asset_name:
            print("Error: --asset_name required")
            return 1
        
        result = manager.validate_asset_integrity(args.asset_name)
        print(json.dumps(result, indent=2, default=str))
        
    elif args.action == 'export_catalog':
        if not args.output_path:
            print("Error: --output_path required")
            return 1
        
        manager.export_asset_catalog(args.output_path, format=args.format)
        print(f"Asset catalog exported to {args.output_path}")
    
    else:
        print(f"Unknown action: {args.action}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())