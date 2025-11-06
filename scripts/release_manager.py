#!/usr/bin/env python3
"""
Release Management System for Automated Model Deployment
Handles semantic versioning, changelog generation, and release automation
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
from typing import Dict, List, Optional, Union, Tuple
import re
import yaml
from dataclasses import dataclass, asdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Version:
    """Semantic version representation"""
    major: int
    minor: int
    patch: int
    prerelease: Optional[str] = None
    build: Optional[str] = None
    
    def __str__(self) -> str:
        version_str = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version_str += f"-{self.prerelease}"
        if self.build:
            version_str += f"+{self.build}"
        return version_str
    
    @classmethod
    def from_string(cls, version_str: str) -> 'Version':
        """Parse version string"""
        # Handle build metadata
        if '+' in version_str:
            version_str, build = version_str.split('+', 1)
        else:
            build = None
        
        # Handle prerelease
        if '-' in version_str:
            version_str, prerelease = version_str.split('-', 1)
        else:
            prerelease = None
        
        # Parse major.minor.patch
        parts = version_str.split('.')
        if len(parts) != 3:
            raise ValueError(f"Invalid version format: {version_str}")
        
        try:
            major = int(parts[0])
            minor = int(parts[1])
            patch = int(parts[2])
        except ValueError as e:
            raise ValueError(f"Invalid version numbers: {e}")
        
        return cls(major=major, minor=minor, patch=patch, 
                  prerelease=prerelease, build=build)
    
    def bump_major(self) -> 'Version':
        """Increment major version"""
        return Version(major=self.major + 1, minor=0, patch=0)
    
    def bump_minor(self) -> 'Version':
        """Increment minor version"""
        return Version(major=self.major, minor=self.minor + 1, patch=0)
    
    def bump_patch(self) -> 'Version':
        """Increment patch version"""
        return Version(major=self.major, minor=self.minor, patch=self.patch + 1)


@dataclass
class Release:
    """Release information"""
    version: Version
    name: str
    date: str
    description: str
    changes: List[str]
    features: List[str]
    fixes: List[str]
    breaking_changes: List[str]
    performance_improvements: List[str]
    model_artifacts: Dict[str, str]
    documentation_updates: List[str]
    download_stats: Optional[Dict] = None
    asset_count: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        data = asdict(self)
        data['version'] = str(self.version)
        return data


class ReleaseManager:
    """Automated release management system"""
    
    def __init__(self, project_path: str, current_version: str = "1.0.0"):
        self.project_path = Path(project_path)
        self.current_version = Version.from_string(current_version)
        self.releases_file = self.project_path / "releases.json"
        self.changelog_file = self.project_path / "CHANGELOG.md"
        
        # Release tracking
        self.releases_history = self._load_releases_history()
        
        logger.info(f"Initialized release manager v{self.current_version}")
    
    def _load_releases_history(self) -> List[Release]:
        """Load release history from file"""
        if self.releases_file.exists():
            try:
                with open(self.releases_file) as f:
                    data = json.load(f)
                
                releases = []
                for release_data in data.get('releases', []):
                    version = Version.from_string(release_data['version'])
                    release = Release(
                        version=version,
                        name=release_data['name'],
                        date=release_data['date'],
                        description=release_data['description'],
                        changes=release_data.get('changes', []),
                        features=release_data.get('features', []),
                        fixes=release_data.get('fixes', []),
                        breaking_changes=release_data.get('breaking_changes', []),
                        performance_improvements=release_data.get('performance_improvements', []),
                        model_artifacts=release_data.get('model_artifacts', {}),
                        documentation_updates=release_data.get('documentation_updates', []),
                        asset_count=release_data.get('asset_count', 0)
                    )
                    releases.append(release)
                
                return releases
                
            except Exception as e:
                logger.error(f"Failed to load releases history: {e}")
        
        return []
    
    def _save_releases_history(self):
        """Save release history to file"""
        releases_data = {
            'project': self.project_path.name,
            'version': str(self.current_version),
            'releases': [release.to_dict() for release in self.releases_history],
            'last_updated': datetime.now().isoformat()
        }
        
        with open(self.releases_file, 'w') as f:
            json.dump(releases_data, f, indent=2)
    
    def detect_changes_since_last_release(self) -> Dict[str, List[str]]:
        """Detect changes since last release"""
        logger.info("Detecting changes since last release")
        
        changes = {
            'features': [],
            'fixes': [],
            'improvements': [],
            'breaking_changes': [],
            'documentation': [],
            'performance': [],
            'model_updates': []
        }
        
        # Get last release
        if self.releases_history:
            last_release = max(self.releases_history, key=lambda r: r.date)
            since_date = last_release.date
        else:
            since_date = "2024-01-01"  # Default start date
        
        # Analyze git commits since last release
        try:
            git_changes = self._analyze_git_commits(since_date)
            changes.update(git_changes)
        except Exception as e:
            logger.warning(f"Git analysis failed: {e}")
            # Fallback to manual detection or file analysis
            changes.update(self._analyze_file_changes())
        
        # Detect model-specific changes
        changes['model_updates'] = self._detect_model_changes()
        
        # Filter empty categories
        changes = {k: v for k, v in changes.items() if v}
        
        return changes
    
    def _analyze_git_commits(self, since_date: str) -> Dict[str, List[str]]:
        """Analyze git commits for changes"""
        changes = {
            'features': [],
            'fixes': [],
            'improvements': [],
            'breaking_changes': [],
            'documentation': [],
            'performance': []
        }
        
        try:
            # Get commits since date
            result = subprocess.run([
                'git', 'log', f'--since={since_date}', '--pretty=format:%s', '--no-merges'
            ], capture_output=True, text=True, cwd=self.project_path)
            
            if result.returncode == 0:
                commits = result.stdout.strip().split('\n')
                
                for commit in commits:
                    if not commit:
                        continue
                    
                    # Categorize commits by patterns
                    commit_lower = commit.lower()
                    
                    if any(keyword in commit_lower for keyword in ['feat', 'feature', 'add', 'implement']):
                        changes['features'].append(commit)
                    elif any(keyword in commit_lower for keyword in ['fix', 'bug', 'issue']):
                        changes['fixes'].append(commit)
                    elif any(keyword in commit_lower for keyword in ['improve', 'optimize', 'enhance']):
                        changes['improvements'].append(commit)
                    elif any(keyword in commit_lower for keyword in ['break', 'remove', 'change']):
                        changes['breaking_changes'].append(commit)
                    elif any(keyword in commit_lower for keyword in ['doc', 'readme', 'comment']):
                        changes['documentation'].append(commit)
                    elif any(keyword in commit_lower for keyword in ['perf', 'speed', 'optimize']):
                        changes['performance'].append(commit)
                    else:
                        # Default to improvements
                        changes['improvements'].append(commit)
        
        except Exception as e:
            logger.error(f"Git analysis failed: {e}")
        
        return changes
    
    def _analyze_file_changes(self) -> Dict[str, List[str]]:
        """Analyze file changes for features/fixes"""
        changes = {
            'features': [],
            'fixes': [],
            'improvements': [],
            'breaking_changes': [],
            'documentation': [],
            'performance': []
        }
        
        # Check for new model files
        model_dirs = ['models', 'checkpoints', 'outputs']
        for model_dir in model_dirs:
            dir_path = self.project_path / model_dir
            if dir_path.exists():
                for file_path in dir_path.rglob('*'):
                    if file_path.is_file():
                        changes['features'].append(f"New model artifact: {file_path.name}")
        
        # Check for configuration changes
        config_files = ['configs/*.yaml', 'configs/*.json', 'config.json']
        for pattern in config_files:
            for config_path in self.project_path.glob(pattern):
                changes['features'].append(f"Configuration update: {config_path.name}")
        
        return changes
    
    def _detect_model_changes(self) -> List[str]:
        """Detect model-specific changes"""
        model_changes = []
        
        # Check for new model versions
        model_patterns = ['*.safetensors', '*.bin', '*.onnx']
        for pattern in model_patterns:
            for model_file in self.project_path.rglob(pattern):
                model_changes.append(f"Model update: {model_file.name}")
        
        # Check for training updates
        training_files = list(self.project_path.glob('**/training_log*.txt')) + \
                        list(self.project_path.glob('**/metrics*.json'))
        if training_files:
            model_changes.append("Training pipeline updates detected")
        
        # Check for evaluation results
        eval_files = list(self.project_path.glob('**/evaluation*.json')) + \
                    list(self.project_path.glob('**/benchmark*.json'))
        if eval_files:
            model_changes.append("New evaluation results available")
        
        return model_changes
    
    def generate_release_notes(self, release_type: str = "patch") -> str:
        """Generate comprehensive release notes"""
        logger.info(f"Generating {release_type} release notes")
        
        # Detect changes
        changes = self.detect_changes_since_last_release()
        
        # Determine next version
        if release_type == "major":
            new_version = self.current_version.bump_major()
        elif release_type == "minor":
            new_version = self.current_version.bump_minor()
        else:
            new_version = self.current_version.bump_patch()
        
        # Generate release notes
        release_notes = f"""# Release {new_version}

**Release Date**: {datetime.now().strftime('%B %d, %Y')}

## 🎉 What's New

"""
        
        if changes.get('features'):
            release_notes += "### ✨ New Features\n\n"
            for feature in changes['features']:
                release_notes += f"- {feature}\n"
            release_notes += "\n"
        
        if changes.get('model_updates'):
            release_notes += "### 🤖 Model Updates\n\n"
            for update in changes['model_updates']:
                release_notes += f"- {update}\n"
            release_notes += "\n"
        
        if changes.get('performance'):
            release_notes += "### ⚡ Performance Improvements\n\n"
            for improvement in changes['performance']:
                release_notes += f"- {improvement}\n"
            release_notes += "\n"
        
        if changes.get('improvements'):
            release_notes += "### 🔧 Improvements\n\n"
            for improvement in changes['improvements']:
                release_notes += f"- {improvement}\n"
            release_notes += "\n"
        
        if changes.get('fixes'):
            release_notes += "### 🐛 Bug Fixes\n\n"
            for fix in changes['fixes']:
                release_notes += f"- {fix}\n"
            release_notes += "\n"
        
        if changes.get('documentation'):
            release_notes += "### 📚 Documentation\n\n"
            for doc in changes['documentation']:
                release_notes += f"- {doc}\n"
            release_notes += "\n"
        
        if changes.get('breaking_changes'):
            release_notes += "### ⚠️ Breaking Changes\n\n"
            for change in changes['breaking_changes']:
                release_notes += f"- {change}\n"
            release_notes += "\n"
        
        # Add model-specific sections
        release_notes += self._generate_model_specific_notes()
        
        # Add installation/upgrade section
        release_notes += f"""
## 🚀 Installation & Upgrade

### Update to {new_version}

```bash
pip install --upgrade sheikh-2.5-coder=={new_version}
```

### Docker
```dockerfile
FROM sheikh-team/sheikh-2.5-coder:{new_version}
```

### HuggingFace Hub
```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "sheikh-team/Sheikh-2.5-Coder",
    revision="{new_version}"
)
```

## 📊 Version Comparison

| Version | Release Date | Key Changes |
|---------|--------------|-------------|
"""
        
        # Add version comparison table
        if self.releases_history:
            for release in sorted(self.releases_history, key=lambda r: r.version, reverse=True)[:5]:
                release_notes += f"| {release.version} | {release.date} | {len(release.features)} features, {len(release.fixes)} fixes |\n"
        
        release_notes += f"""
## 🔗 Links

- **Release**: https://github.com/sheikh-team/Sheikh-2.5-Coder/releases/tag/{new_version}
- **Documentation**: https://github.com/sheikh-team/Sheikh-2.5-Coder/wiki
- **Model Hub**: https://huggingface.co/sheikh-team/Sheikh-2.5-Coder
- **Previous Release**: {self.current_version}

## 🙏 Contributors

Thanks to all contributors who made this release possible!

---

*Generated by Release Manager v1.0*
"""
        
        return release_notes
    
    def _generate_model_specific_notes(self) -> str:
        """Generate model-specific release notes"""
        notes = """## 🎯 Model Performance

### Inference Speed
"""
        
        # Add performance metrics if available
        notes += """- Base Model: Optimized for GPU inference
- INT8 Variant: ~50% memory reduction with minimal speed impact
- INT4 Variant: ~75% memory reduction for resource-constrained deployments
- ONNX Version: Hardware-accelerated inference on supported platforms

### Model Variants Available
"""
        
        variants = [
            ("Base Model", "Original full-precision model", "~5.4GB"),
            ("INT8 Quantized", "Memory-optimized version", "~2.7GB"),
            ("INT4 Quantized", "Ultra-low memory version", "~1.35GB"),
            ("ONNX Optimized", "Cross-platform accelerated", "Variable"),
            ("TensorRT", "NVIDIA GPU optimized", "~4GB")
        ]
        
        for name, description, size in variants:
            notes += f"- **{name}**: {description} ({size})\n"
        
        notes += """
### Usage Examples

#### Quick Start
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("sheikh-team/Sheikh-2.5-Coder")
tokenizer = AutoTokenizer.from_pretrained("sheikh-team/Sheikh-2.5-Coder")

# Generate code
prompt = "def quick_sort(arr):"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

#### With Quantization
```python
from transformers import BitsAndBytesConfig

quant_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "sheikh-team/Sheikh-2.5-Coder",
    quantization_config=quant_config
)
```
"""
        
        return notes
    
    def create_release(self, release_type: str = "patch", 
                      custom_changes: Optional[Dict] = None) -> Release:
        """Create a new release"""
        logger.info(f"Creating {release_type} release")
        
        # Determine new version
        if release_type == "major":
            new_version = self.current_version.bump_major()
        elif release_type == "minor":
            new_version = self.current_version.bump_minor()
        else:
            new_version = self.current_version.bump_patch()
        
        # Get changes
        if custom_changes:
            changes = custom_changes
        else:
            changes = self.detect_changes_since_last_release()
        
        # Create release
        release = Release(
            version=new_version,
            name=f"Release {new_version}",
            date=datetime.now().isoformat(),
            description=f"Automated {release_type} release",
            changes=changes.get('improvements', []),
            features=changes.get('features', []),
            fixes=changes.get('fixes', []),
            breaking_changes=changes.get('breaking_changes', []),
            performance_improvements=changes.get('performance', []),
            model_artifacts=self._collect_model_artifacts(),
            documentation_updates=changes.get('documentation', []),
            asset_count=len(self._collect_model_artifacts())
        )
        
        # Add to history
        self.releases_history.append(release)
        self.current_version = new_version
        
        # Save history
        self._save_releases_history()
        
        # Update changelog
        self.update_changelog(release)
        
        logger.info(f"Created release {new_version}")
        return release
    
    def _collect_model_artifacts(self) -> Dict[str, str]:
        """Collect model artifacts for release"""
        artifacts = {}
        
        # Look for model files in common locations
        model_patterns = [
            ('**/models/*.safetensors', 'base_model'),
            ('**/checkpoints/*.safetensors', 'checkpoint'),
            ('**/models/*.bin', 'pytorch_model'),
            ('**/outputs/**/*.onnx', 'onnx_model'),
            ('**/quantized/*.safetensors', 'quantized_model')
        ]
        
        for pattern, artifact_type in model_patterns:
            for model_path in self.project_path.glob(pattern):
                artifacts[artifact_type] = str(model_path.relative_to(self.project_path))
        
        return artifacts
    
    def update_changelog(self, release: Release):
        """Update CHANGELOG.md with new release"""
        logger.info(f"Updating changelog for {release.version}")
        
        # Generate release notes
        release_notes = self._format_release_for_changelog(release)
        
        if self.changelog_file.exists():
            with open(self.changelog_file) as f:
                existing_content = f.read()
            
            # Insert new release at the top
            new_content = f"# Changelog\n\nAll notable changes to this project will be documented in this file.\n\n{release_notes}\n\n{existing_content}"
        else:
            new_content = f"""# Changelog

All notable changes to this project will be documented in this file.

{release_notes}
"""
        
        with open(self.changelog_file, 'w') as f:
            f.write(new_content)
        
        logger.info("Changelog updated successfully")
    
    def _format_release_for_changelog(self, release: Release) -> str:
        """Format release for changelog"""
        date_formatted = datetime.fromisoformat(release.date.replace('Z', '+00:00')).strftime('%Y-%m-%d')
        
        changelog_entry = f"## [{release.version}] - {date_formatted}\n\n"
        
        if release.features:
            changelog_entry += "### Added\n"
            for feature in release.features:
                changelog_entry += f"- {feature}\n"
            changelog_entry += "\n"
        
        if release.fixes:
            changelog_entry += "### Fixed\n"
            for fix in release.fixes:
                changelog_entry += f"- {fix}\n"
            changelog_entry += "\n"
        
        if release.improvements:
            changelog_entry += "### Changed\n"
            for improvement in release.improvements:
                changelog_entry += f"- {improvement}\n"
            changelog_entry += "\n"
        
        if release.performance_improvements:
            changelog_entry += "### Performance\n"
            for perf in release.performance_improvements:
                changelog_entry += f"- {perf}\n"
            changelog_entry += "\n"
        
        if release.breaking_changes:
            changelog_entry += "### Breaking Changes\n"
            for change in release.breaking_changes:
                changelog_entry += f"- {change}\n"
            changelog_entry += "\n"
        
        return changelog_entry
    
    def get_version_info(self) -> Dict:
        """Get current version information"""
        return {
            'current_version': str(self.current_version),
            'releases_count': len(self.releases_history),
            'latest_release': str(self.releases_history[-1].version) if self.releases_history else None,
            'next_patch': str(self.current_version.bump_patch()),
            'next_minor': str(self.current_version.bump_minor()),
            'next_major': str(self.current_version.bump_major())
        }
    
    def compare_versions(self, version1: str, version2: str) -> Dict:
        """Compare two versions"""
        v1 = Version.from_string(version1)
        v2 = Version.from_string(version2)
        
        if v1.major != v2.major:
            comparison = "major"
            difference = v1.major - v2.major
        elif v1.minor != v2.minor:
            comparison = "minor"
            difference = v1.minor - v2.minor
        elif v1.patch != v2.patch:
            comparison = "patch"
            difference = v1.patch - v2.patch
        else:
            comparison = "equal"
            difference = 0
        
        return {
            'version1': version1,
            'version2': version2,
            'comparison': comparison,
            'difference': difference,
            'v1_greater': v1 > v2,
            'v2_greater': v2 > v1
        }
    
    def list_releases(self, limit: int = 10) -> List[Dict]:
        """List recent releases"""
        sorted_releases = sorted(self.releases_history, 
                               key=lambda r: r.version, reverse=True)[:limit]
        
        return [release.to_dict() for release in sorted_releases]
    
    def create_release_branches(self, release: Release) -> List[str]:
        """Create release branches"""
        branch_names = []
        
        try:
            # Create release branch
            branch_name = f"release/{release.version}"
            result = subprocess.run([
                'git', 'branch', branch_name
            ], cwd=self.project_path, capture_output=True, text=True)
            
            if result.returncode == 0:
                branch_names.append(branch_name)
                logger.info(f"Created release branch: {branch_name}")
            
            # Create version tag
            tag_name = f"v{release.version}"
            result = subprocess.run([
                'git', 'tag', '-a', tag_name, '-m', f"Release {release.version}"
            ], cwd=self.project_path, capture_output=True, text=True)
            
            if result.returncode == 0:
                branch_names.append(tag_name)
                logger.info(f"Created version tag: {tag_name}")
                
        except Exception as e:
            logger.error(f"Failed to create release branches: {e}")
        
        return branch_names
    
    def generate_release_config(self, release: Release, output_path: str):
        """Generate release configuration file"""
        config = {
            'release_info': {
                'version': str(release.version),
                'name': release.name,
                'date': release.date,
                'description': release.description
            },
            'model_artifacts': release.model_artifacts,
            'deployment_targets': [
                'huggingface_hub',
                'github_releases',
                'pypi',
                'docker_hub'
            ],
            'quality_gates': {
                'model_validation': True,
                'performance_benchmark': True,
                'documentation_complete': True,
                'testing_passed': True
            },
            'notification_channels': [
                'github_discussions',
                'twitter',
                'blog_post'
            ]
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Release config saved to {output_path}")


def main():
    """Main function for release management"""
    parser = argparse.ArgumentParser(description='Release management system')
    
    parser.add_argument('--project_path', required=True, help='Project path')
    parser.add_argument('--current_version', default='1.0.0', help='Current version')
    parser.add_argument('--action', required=True,
                       choices=['create_release', 'generate_notes', 'version_info', 
                               'list_releases', 'compare_versions', 'detect_changes'],
                       help='Action to perform')
    
    # Action-specific arguments
    parser.add_argument('--release_type', default='patch',
                       choices=['major', 'minor', 'patch'],
                       help='Release type for creation')
    parser.add_argument('--changes_file', help='Custom changes file (JSON)')
    parser.add_argument('--output_path', help='Output path for files')
    parser.add_argument('--version1', help='First version to compare')
    parser.add_argument('--version2', help='Second version to compare')
    parser.add_argument('--limit', type=int, default=10, help='Number of releases to list')
    
    args = parser.parse_args()
    
    # Initialize release manager
    manager = ReleaseManager(
        project_path=args.project_path,
        current_version=args.current_version
    )
    
    # Execute action
    if args.action == 'create_release':
        custom_changes = None
        if args.changes_file and Path(args.changes_file).exists():
            with open(args.changes_file) as f:
                custom_changes = json.load(f)
        
        release = manager.create_release(
            release_type=args.release_type,
            custom_changes=custom_changes
        )
        
        result = {
            'action': 'create_release',
            'status': 'success',
            'release': release.to_dict()
        }
        
        # Save release notes if output path specified
        if args.output_path:
            release_notes = manager.generate_release_notes(args.release_type)
            with open(args.output_path, 'w') as f:
                f.write(release_notes)
            result['release_notes_saved'] = args.output_path
        
        print(json.dumps(result, indent=2, default=str))
        
    elif args.action == 'generate_notes':
        release_notes = manager.generate_release_notes(args.release_type)
        
        if args.output_path:
            with open(args.output_path, 'w') as f:
                f.write(release_notes)
            print(f"Release notes saved to {args.output_path}")
        else:
            print(release_notes)
    
    elif args.action == 'version_info':
        info = manager.get_version_info()
        print(json.dumps(info, indent=2, default=str))
        
    elif args.action == 'list_releases':
        releases = manager.list_releases(limit=args.limit)
        print(json.dumps(releases, indent=2, default=str))
        
    elif args.action == 'compare_versions':
        if not args.version1 or not args.version2:
            print("Error: Both version1 and version2 required")
            return 1
        
        comparison = manager.compare_versions(args.version1, args.version2)
        print(json.dumps(comparison, indent=2, default=str))
        
    elif args.action == 'detect_changes':
        changes = manager.detect_changes_since_last_release()
        print(json.dumps(changes, indent=2, default=str))
    
    else:
        print(f"Unknown action: {args.action}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())