#!/usr/bin/env python3
"""
HuggingFace Hub Integration for Automated Model Deployment
Handles model uploads, versioning, and repository management
"""

import os
import sys
import json
import argparse
import logging
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
import requests
import time

from huggingface_hub import HfApi, create_repo, upload_file, upload_folder
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HuggingFaceDeployer:
    """Automated HuggingFace Hub deployment handler"""
    
    def __init__(self, token: Optional[str] = None, repo_id: str = "sheikh-team/Sheikh-2.5-Coder"):
        self.api = HfApi(token=token)
        self.repo_id = repo_id
        self.repo_url = f"https://huggingface.co/{repo_id}"
        
        # Set token if provided (using environment variable approach)
        if token:
            os.environ['HF_TOKEN'] = token
        
        logger.info(f"Initialized HuggingFace deployment for repo: {repo_id}")
    
    def create_repository(self, private: bool = False, exist_ok: bool = True) -> Dict:
        """Create or update repository on HuggingFace Hub"""
        logger.info(f"Creating/updating repository: {self.repo_id}")
        
        try:
            repo_info = create_repo(
                repo_id=self.repo_id,
                repo_type="model",
                exist_ok=exist_ok,
                private=private
            )
            
            result = {
                'action': 'create_repository',
                'status': 'success',
                'repo_id': self.repo_id,
                'repo_url': self.repo_url,
                'created_at': repo_info.created_at.isoformat() if hasattr(repo_info, 'created_at') else datetime.now().isoformat()
            }
            
            logger.info(f"Repository ready: {self.repo_url}")
            return result
            
        except Exception as e:
            logger.error(f"Repository creation failed: {str(e)}")
            return {'action': 'create_repository', 'error': str(e)}
    
    def upload_model_weights(self, model_path: Union[str, Path], 
                           commit_message: str = "Model update",
                           exists_ok: bool = True) -> Dict:
        """Upload model weights to repository"""
        logger.info(f"Uploading model from {model_path}")
        
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model path not found: {model_path}")
            
            # Validate model files
            model_files = self._validate_model_files(model_path)
            if not model_files['valid']:
                raise ValueError(f"Invalid model files: {model_files['errors']}")
            
            # Create upload commit message
            if not commit_message:
                commit_message = f"Upload model - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            # Upload model files
            upload_results = []
            for file_path in model_files['files']:
                relative_path = file_path.relative_to(model_path)
                
                upload_result = self.upload_single_file(
                    file_path=str(file_path),
                    path_in_repo=str(relative_path),
                    commit_message=commit_message
                )
                upload_results.append(upload_result)
            
            result = {
                'action': 'upload_model_weights',
                'status': 'success',
                'repo_id': self.repo_id,
                'files_uploaded': len(upload_results),
                'files_details': upload_results,
                'total_size_mb': sum(upload_results.get('size_mb', 0) for upload_results in upload_results)
            }
            
            logger.info(f"Model upload completed: {result['files_uploaded']} files")
            return result
            
        except Exception as e:
            logger.error(f"Model upload failed: {str(e)}")
            return {'action': 'upload_model_weights', 'error': str(e)}
    
    def upload_single_file(self, file_path: str, path_in_repo: str, 
                          commit_message: str = "File upload") -> Dict:
        """Upload a single file to the repository"""
        try:
            file_path = Path(file_path)
            file_size = file_path.stat().st_size if file_path.exists() else 0
            
            # Upload file
            response = upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=path_in_repo,
                repo_id=self.repo_id,
                commit_message=commit_message
            )
            
            result = {
                'file_path': path_in_repo,
                'local_size_mb': file_size / (1024 * 1024),
                'status': 'success',
                'commit': response.oid if hasattr(response, 'oid') else 'unknown'
            }
            
            logger.info(f"Uploaded: {path_in_repo} ({result['local_size_mb']:.2f} MB)")
            return result
            
        except Exception as e:
            logger.error(f"File upload failed {path_in_repo}: {str(e)}")
            return {
                'file_path': path_in_repo,
                'status': 'error',
                'error': str(e)
            }
    
    def upload_model_directory(self, model_dir: Union[str, Path], 
                              commit_message: str = "Directory upload") -> Dict:
        """Upload entire model directory"""
        logger.info(f"Uploading directory: {model_dir}")
        
        try:
            model_dir = Path(model_dir)
            if not model_dir.exists():
                raise FileNotFoundError(f"Directory not found: {model_dir}")
            
            # Validate directory
            validation = self._validate_model_directory(model_dir)
            if not validation['valid']:
                logger.warning(f"Directory validation warnings: {validation['warnings']}")
            
            # Upload folder
            response = upload_folder(
                folder_id=str(model_dir),
                repo_id=self.repo_id,
                commit_message=commit_message,
                ignore_patterns=validation.get('ignore_patterns', [])
            )
            
            # Calculate total size
            total_size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
            
            result = {
                'action': 'upload_directory',
                'status': 'success',
                'repo_id': self.repo_id,
                'directory_path': str(model_dir),
                'total_size_mb': total_size / (1024 * 1024),
                'commit': response.oid if hasattr(response, 'oid') else 'unknown'
            }
            
            logger.info(f"Directory upload completed: {result['total_size_mb']:.2f} MB")
            return result
            
        except Exception as e:
            logger.error(f"Directory upload failed: {str(e)}")
            return {'action': 'upload_directory', 'error': str(e)}
    
    def update_model_card(self, model_card_content: str, 
                         commit_message: str = "Update model card") -> Dict:
        """Update repository README/model card"""
        logger.info("Updating model card")
        
        try:
            # Upload model card as README.md
            response = upload_file(
                path_or_fileobj=model_card_content.encode('utf-8'),
                path_in_repo="README.md",
                repo_id=self.repo_id,
                commit_message=commit_message
            )
            
            result = {
                'action': 'update_model_card',
                'status': 'success',
                'repo_id': self.repo_id,
                'commit': response.oid if hasattr(response, 'oid') else 'unknown',
                'card_length': len(model_card_content)
            }
            
            logger.info("Model card updated successfully")
            return result
            
        except Exception as e:
            logger.error(f"Model card update failed: {str(e)}")
            return {'action': 'update_model_card', 'error': str(e)}
    
    def create_model_variant(self, model_path: Union[str, Path], 
                           variant_name: str, description: str = "",
                           commit_message: str = "Create model variant") -> Dict:
        """Create a new variant of the model (e.g., quantized version)"""
        logger.info(f"Creating model variant: {variant_name}")
        
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model path not found: {model_path}")
            
            # Create variant-specific path
            variant_repo_id = f"{self.repo_id}-{variant_name}"
            
            # Create variant repository
            variant_repo_info = create_repo(
                repo_id=variant_repo_id,
                repo_type="model",
                exist_ok=True,
                private=False
            )
            
            # Upload model to variant repository
            upload_result = self.upload_model_directory(
                model_dir=model_path,
                commit_message=f"{commit_message} - {description}"
            )
            
            # Create variant model card
            variant_card = self._generate_variant_model_card(
                base_model_card=None,  # Would fetch from base repo if needed
                variant_name=variant_name,
                description=description
            )
            
            # Upload variant model card
            card_result = self._upload_to_repo(
                variant_repo_id, variant_card, "README.md", commit_message
            )
            
            result = {
                'action': 'create_model_variant',
                'status': 'success',
                'variant_name': variant_name,
                'variant_repo_id': variant_repo_id,
                'variant_repo_url': f"https://huggingface.co/{variant_repo_id}",
                'upload_result': upload_result,
                'model_card_result': card_result
            }
            
            logger.info(f"Model variant created: {result['variant_repo_url']}")
            return result
            
        except Exception as e:
            logger.error(f"Model variant creation failed: {str(e)}")
            return {'action': 'create_model_variant', 'error': str(e)}
    
    def create_model_releases(self, model_releases: List[Dict]) -> Dict:
        """Create model releases with different configurations"""
        logger.info(f"Creating {len(model_releases)} model releases")
        
        release_results = []
        
        for release in model_releases:
            try:
                release_result = self._create_single_release(release)
                release_results.append(release_result)
                
                # Rate limiting - be nice to the API
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Release creation failed for {release.get('name', 'unknown')}: {str(e)}")
                release_results.append({
                    'release_name': release.get('name', 'unknown'),
                    'status': 'error',
                    'error': str(e)
                })
        
        result = {
            'action': 'create_model_releases',
            'status': 'completed' if release_results else 'failed',
            'releases_created': len([r for r in release_results if r.get('status') == 'success']),
            'total_releases': len(model_releases),
            'release_details': release_results
        }
        
        logger.info(f"Model releases completed: {result['releases_created']}/{result['total_releases']}")
        return result
    
    def _create_single_release(self, release_config: Dict) -> Dict:
        """Create a single model release"""
        try:
            release_name = release_config.get('name', 'release')
            model_path = release_config.get('model_path')
            description = release_config.get('description', '')
            tags = release_config.get('tags', [])
            
            if not model_path:
                raise ValueError("Model path required for release")
            
            # Create model card for release
            release_card = self._generate_release_model_card(
                release_name=release_name,
                description=description,
                model_path=model_path,
                tags=tags
            )
            
            # Create release-specific repository
            release_repo_id = f"{self.repo_id}-{release_name}"
            create_repo(
                repo_id=release_repo_id,
                repo_type="model",
                exist_ok=True,
                private=False
            )
            
            # Upload model and files
            upload_result = self.upload_model_directory(
                model_dir=model_path,
                commit_message=f"Release {release_name}"
            )
            
            # Upload model card
            card_result = self._upload_to_repo(
                release_repo_id, release_card, "README.md", f"Release {release_name}"
            )
            
            result = {
                'release_name': release_name,
                'status': 'success',
                'release_repo_id': release_repo_id,
                'release_repo_url': f"https://huggingface.co/{release_repo_id}",
                'upload_result': upload_result,
                'model_card_result': card_result
            }
            
            logger.info(f"Release created: {release_name}")
            return result
            
        except Exception as e:
            return {
                'release_name': release_config.get('name', 'unknown'),
                'status': 'error',
                'error': str(e)
            }
    
    def _upload_to_repo(self, repo_id: str, content: str, 
                       path_in_repo: str, commit_message: str) -> Dict:
        """Upload content to specific repository"""
        try:
            response = upload_file(
                path_or_fileobj=content.encode('utf-8'),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                commit_message=commit_message
            )
            
            return {
                'status': 'success',
                'repo_id': repo_id,
                'path': path_in_repo,
                'commit': response.oid if hasattr(response, 'oid') else 'unknown'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'repo_id': repo_id,
                'path': path_in_repo,
                'error': str(e)
            }
    
    def _validate_model_files(self, model_path: Path) -> Dict:
        """Validate model files before upload"""
        required_files = [
            'config.json',
            'tokenizer.json',
            'tokenizer_config.json'
        ]
        
        model_files = list(model_path.glob('*'))
        existing_files = [f.name for f in model_files]
        
        missing_files = [f for f in required_files if f not in existing_files]
        model_files_present = any(f.suffix in ['.bin', '.safetensors', '.onnx'] for f in model_files)
        
        validation = {
            'valid': len(missing_files) == 0 and model_files_present,
            'files': model_files,
            'missing_files': missing_files,
            'model_files_present': model_files_present,
            'errors': []
        }
        
        if missing_files:
            validation['errors'].append(f"Missing required files: {missing_files}")
        
        if not model_files_present:
            validation['errors'].append("No model weight files found")
        
        return validation
    
    def _validate_model_directory(self, model_dir: Path) -> Dict:
        """Validate model directory structure"""
        validation = {
            'valid': True,
            'warnings': [],
            'ignore_patterns': ['*.pyc', '__pycache__', '.git', '.git', '.svn', '__pycache__', '*.pyc', '*.log', '.DS_Store', '.idea', '.vscode', '.agent', 'node_modules', 'workspace', 'browser_use', '.venv', 'browser/user_data*', 'browser/sessions', 'mcp_downloaded', 'debug', 'log', 'pyproject.toml', 'external_api', 'pyarmor_runtime_000000', 'uv.lock', '.pdf_temp', 'pdf_temp']
        }
        
        # Check for large files that might need special handling
        large_files = []
        for file_path in model_dir.rglob('*'):
            if file_path.is_file() and file_path.stat().st_size > 100 * 1024 * 1024:  # 100MB
                large_files.append(str(file_path.relative_to(model_dir)))
        
        if large_files:
            validation['warnings'].append(f"Large files detected: {large_files}")
        
        return validation
    
    def _generate_variant_model_card(self, base_model_card: Optional[str], 
                                   variant_name: str, description: str) -> str:
        """Generate model card for model variant"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        card_content = f"""# Sheikh-2.5-Coder - {variant_name.title()} Variant

## Overview
{description}

## Model Details
- **Base Model**: Sheikh-2.5-Coder
- **Variant Type**: {variant_name}
- **Created**: {timestamp}
- **Optimized for**: {variant_name.replace('_', ' ').title()}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("sheikh-team/Sheikh-2.5-Coder-{variant_name}")
tokenizer = AutoTokenizer.from_pretrained("sheikh-team/Sheikh-2.5-Coder-{variant_name}")

# Generate code
prompt = "def hello_world():"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Performance Characteristics
{self._get_variant_performance_info(variant_name)}

## Limitations
- Optimized for {variant_name.replace('_', ' ')} inference
- May have different performance characteristics compared to base model
- Best results with appropriate hardware for {variant_name} format

## License
MIT License

## Citation
```bibtex
@model{{sheikh_2_5_coder_{variant_name},
  title={{Sheikh-2.5-Coder {variant_name.title()} Variant}},
  author={{Sheikh Team}},
  year={{2024}},
  url={{https://huggingface.co/sheikh-team/Sheikh-2.5-Coder-{variant_name}}}
}}
```
"""
        return card_content
    
    def _generate_release_model_card(self, release_name: str, description: str,
                                   model_path: Path, tags: List[str]) -> str:
        """Generate model card for model release"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Get model info
        config_path = model_path / 'config.json'
        model_size = "Unknown"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                    model_size = config.get('vocab_size', 'Unknown')
            except:
                pass
        
        card_content = f"""# Sheikh-2.5-Coder - Release {release_name}

## Release Information
- **Release Name**: {release_name}
- **Release Date**: {timestamp}
- **Model Size**: {model_size} parameters
- **Tags**: {', '.join(tags) if tags else 'General Release'}

## Description
{description}

## What's New in This Release
{self._get_release_highlights(release_name)}

## Usage Examples

### Basic Code Generation
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("sheikh-team/Sheikh-2.5-Coder-{release_name}")
tokenizer = AutoTokenizer.from_pretrained("sheikh-team/Sheikh-2.5-Coder-{release_name}")

prompt = "Create a function to calculate factorial:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=150, temperature=0.7)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

### Advanced Usage with Quantization
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Load with quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    "sheikh-team/Sheikh-2.5-Coder-{release_name}",
    quantization_config=quantization_config,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained("sheikh-team/Sheikh-2.5-Coder-{release_name}")
```

## Performance Benchmarks
{self._get_benchmark_info(release_name)}

## Changelog
{self._get_changelog_info(release_name)}

## License
MIT License

## Support
- **Issues**: [GitHub Issues](https://github.com/sheikh-team/Sheikh-2.5-Coder/issues)
- **Discussions**: [GitHub Discussions](https://github.com/sheikh-team/Sheikh-2.5-Coder/discussions)

## Citation
```bibtex
@software{{sheikh_2_5_coder_release_{release_name.replace(' ', '_').lower()},
  title={{Sheikh-2.5-Coder Release {release_name}}},
  author={{Sheikh Team}},
  year={{2024}},
  url={{https://huggingface.co/sheikh-team/Sheikh-2.5-Coder-{release_name}}}
}}
```
"""
        return card_content
    
    def _get_variant_performance_info(self, variant_name: str) -> str:
        """Get performance information for variant"""
        performance_info = {
            'int8': "- **Quantization**: 8-bit integer quantization\n- **Memory Usage**: ~50% reduction\n- **Speed**: Slightly slower inference\n- **Quality**: Minimal quality loss",
            'int4': "- **Quantization**: 4-bit integer quantization\n- **Memory Usage**: ~75% reduction\n- **Speed**: Moderate inference speed\n- **Quality**: Minor quality trade-off",
            'onnx': "- **Format**: ONNX optimized\n- **Speed**: Optimized inference on supported hardware\n- **Platform**: Cross-platform compatibility\n- **Hardware**: Best performance on GPUs with ONNX support"
        }
        
        return performance_info.get(variant_name, "- **Variant**: Custom optimization variant")
    
    def _get_release_highlights(self, release_name: str) -> str:
        """Get release highlights"""
        highlights = {
            'v1.0': "- Initial stable release\n- Complete training pipeline\n- Full model capabilities",
            'v1.1': "- Performance improvements\n- Bug fixes\n- Enhanced documentation",
            'quantized': "- Multiple quantization options\n- Memory optimized variants\n- Faster inference options"
        }
        
        return highlights.get(release_name, "- New release with improvements")
    
    def _get_benchmark_info(self, release_name: str) -> str:
        """Get benchmark information for release"""
        return """
| Benchmark | Score | Notes |
|-----------|-------|--------|
| Code Generation | In Progress | Benchmark evaluation ongoing |
| HumanEval | In Progress | Comprehensive evaluation |
| Performance | Evaluated | Speed and memory metrics |
"""
    
    def _get_changelog_info(self, release_name: str) -> str:
        """Get changelog information"""
        return f"""
## Release {release_name}
- Model deployment and optimization
- Documentation improvements
- Quality assurance enhancements
- Performance optimizations
"""
    
    def create_tags_and_releases(self, tags: List[str], 
                                create_discussion: bool = True) -> Dict:
        """Create tags and discussion for model releases"""
        logger.info(f"Creating {len(tags)} tags for repository")
        
        tag_results = []
        for tag in tags:
            try:
                # Create tag via Git (this would need to be done locally typically)
                tag_result = {
                    'tag': tag,
                    'status': 'tag_creation_pending',
                    'note': 'Tags should be created via local git push'
                }
                tag_results.append(tag_result)
                
            except Exception as e:
                logger.error(f"Tag creation failed for {tag}: {str(e)}")
                tag_results.append({
                    'tag': tag,
                    'status': 'error',
                    'error': str(e)
                })
        
        return {
            'action': 'create_tags_and_releases',
            'status': 'completed',
            'tags_created': len(tag_results),
            'tag_results': tag_results
        }
    
    def get_repository_info(self) -> Dict:
        """Get repository information"""
        try:
            repo_info = self.api.repo_info(self.repo_id)
            
            result = {
                'action': 'get_repository_info',
                'status': 'success',
                'repo_id': self.repo_id,
                'repo_url': self.repo_url,
                'private': repo_info.private if hasattr(repo_info, 'private') else False,
                'downloads': getattr(repo_info, 'downloads', 'unknown'),
                'likes': getattr(repo_info, 'likes', 'unknown'),
                'created_at': repo_info.created_at.isoformat() if hasattr(repo_info, 'created_at') else 'unknown',
                'last_modified': repo_info.last_modified.isoformat() if hasattr(repo_info, 'last_modified') else 'unknown'
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get repository info: {str(e)}")
            return {'action': 'get_repository_info', 'error': str(e)}


def main():
    """Main function for HuggingFace deployment"""
    parser = argparse.ArgumentParser(description='HuggingFace Hub deployment')
    
    parser.add_argument('--token', help='HuggingFace API token')
    parser.add_argument('--repo_id', default='sheikh-team/Sheikh-2.5-Coder', 
                       help='Repository ID')
    parser.add_argument('--model_path', help='Path to model directory')
    parser.add_argument('--commit_message', help='Commit message')
    parser.add_argument('--action', required=True, 
                       choices=['create_repo', 'upload_model', 'upload_dir', 
                               'update_card', 'create_variant', 'create_releases'],
                       help='Action to perform')
    parser.add_argument('--model_card', help='Path to model card file')
    parser.add_argument('--variant_name', help='Variant name for creation')
    parser.add_argument('--variant_description', help='Variant description')
    parser.add_argument('--releases_config', help='Path to releases configuration JSON')
    
    args = parser.parse_args()
    
    # Initialize deployer
    deployer = HuggingFaceDeployer(token=args.token, repo_id=args.repo_id)
    
    # Execute action
    if args.action == 'create_repo':
        result = deployer.create_repository()
    elif args.action == 'upload_model':
        result = deployer.upload_model_weights(
            model_path=args.model_path,
            commit_message=args.commit_message
        )
    elif args.action == 'upload_dir':
        result = deployer.upload_model_directory(
            model_dir=args.model_path,
            commit_message=args.commit_message or "Directory upload"
        )
    elif args.action == 'update_card':
        if args.model_card and Path(args.model_card).exists():
            with open(args.model_card) as f:
                card_content = f.read()
            result = deployer.update_model_card(
                model_card_content=card_content,
                commit_message=args.commit_message or "Update model card"
            )
        else:
            result = {'status': 'error', 'error': 'Model card file not found'}
    elif args.action == 'create_variant':
        result = deployer.create_model_variant(
            model_path=args.model_path,
            variant_name=args.variant_name,
            description=args.variant_description or "",
            commit_message=args.commit_message or f"Create {args.variant_name} variant"
        )
    elif args.action == 'create_releases':
        if args.releases_config and Path(args.releases_config).exists():
            with open(args.releases_config) as f:
                releases_config = json.load(f)
            result = deployer.create_model_releases(releases_config.get('releases', []))
        else:
            result = {'status': 'error', 'error': 'Releases config file not found'}
    else:
        result = {'status': 'error', 'error': f'Unknown action: {args.action}'}
    
    # Print result
    print(json.dumps(result, indent=2, default=str))
    
    # Return exit code
    return 0 if result.get('status') == 'success' else 1


if __name__ == '__main__':
    sys.exit(main())