#!/usr/bin/env python3
"""
Version Manager for Semantic Versioning and Model Release Management
Handles version bumping, compatibility checking, and release planning
"""

import os
import sys
import json
import argparse
import logging
import subprocess
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, asdict
import yaml

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
        """Parse version string according to semantic versioning"""
        # Remove build metadata first
        if '+' in version_str:
            version_str, build = version_str.split('+', 1)
        else:
            build = None
        
        # Handle prerelease
        if '-' in version_str:
            version_str, prerelease = version_str.split('-', 1)
        else:
            prerelease = None
        
        # Parse core version (major.minor.patch)
        core_pattern = r'^(\d+)\.(\d+)\.(\d+)$'
        match = re.match(core_pattern, version_str)
        
        if not match:
            # Try without patch version
            pattern = r'^(\d+)\.(\d+)$'
            match = re.match(pattern, version_str)
            if match:
                major, minor = int(match.group(1)), int(match.group(2))
                patch = 0
            else:
                raise ValueError(f"Invalid version format: {version_str}")
        else:
            major, minor, patch = int(match.group(1)), int(match.group(2)), int(match.group(3))
        
        return cls(
            major=major,
            minor=minor,
            patch=patch,
            prerelease=prerelease,
            build=build
        )
    
    def bump_major(self) -> 'Version':
        """Increment major version"""
        return Version(major=self.major + 1, minor=0, patch=0)
    
    def bump_minor(self) -> 'Version':
        """Increment minor version"""
        return Version(major=self.major, minor=self.minor + 1, patch=0)
    
    def bump_patch(self) -> 'Version':
        """Increment patch version"""
        return Version(major=self.major, minor=self.minor, patch=self.patch + 1)
    
    def with_prerelease(self, prerelease: str) -> 'Version':
        """Add or update prerelease identifier"""
        return Version(
            major=self.major,
            minor=self.minor,
            patch=self.patch,
            prerelease=prerelease
        )
    
    def with_build(self, build: str) -> 'Version':
        """Add or update build metadata"""
        return Version(
            major=self.major,
            minor=self.minor,
            patch=self.patch,
            prerelease=self.prerelease,
            build=build
        )
    
    def remove_prerelease(self) -> 'Version':
        """Remove prerelease identifier"""
        return Version(
            major=self.major,
            minor=self.minor,
            patch=self.patch,
            build=self.build
        )
    
    def __lt__(self, other: 'Version') -> bool:
        """Compare versions (semantic versioning rules)"""
        if not isinstance(other, Version):
            return NotImplemented
        
        # Compare core version
        if self.major != other.major:
            return self.major < other.major
        if self.minor != other.minor:
            return self.minor < other.minor
        if self.patch != other.patch:
            return self.patch < other.patch
        
        # Prerelease comparison
        if self.prerelease is None and other.prerelease is None:
            return False
        elif self.prerelease is None:
            return False  # None is greater than any prerelease
        elif other.prerelease is None:
            return True   # Any prerelease is less than None
        
        # Both have prerelease - compare as strings
        return self.prerelease < other.prerelease
    
    def __le__(self, other: 'Version') -> bool:
        return self < other or self == other
    
    def __gt__(self, other: 'Version') -> bool:
        return not (self <= other)
    
    def __ge__(self, other: 'Version') -> bool:
        return not (self < other)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Version):
            return False
        return (self.major == other.major and 
                self.minor == other.minor and 
                self.patch == other.patch and
                self.prerelease == other.prerelease)
    
    def __hash__(self) -> int:
        return hash((self.major, self.minor, self.patch, self.prerelease))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'major': self.major,
            'minor': self.minor,
            'patch': self.patch,
            'prerelease': self.prerelease,
            'build': self.build,
            'string': str(self)
        }


@dataclass
class CompatibilityMatrix:
    """Compatibility information for versions"""
    version: Version
    compatible_with: List[Version]
    breaking_changes: List[str]
    migration_notes: List[str]
    deprecated: bool = False
    lts: bool = False  # Long Term Support


class VersionManager:
    """Advanced version management system"""
    
    def __init__(self, project_path: str, initial_version: str = "1.0.0"):
        self.project_path = Path(project_path)
        self.version_file = self.project_path / "VERSION"
        self.changelog_file = self.project_path / "CHANGELOG.md"
        self.compatibility_file = self.project_path / "compatibility.json"
        
        # Load or create version
        self.current_version = self._load_current_version(initial_version)
        self.version_history = self._load_version_history()
        self.compatibility_matrix = self._load_compatibility_matrix()
        
        logger.info(f"Initialized version manager: {self.current_version}")
    
    def _load_current_version(self, initial_version: str) -> Version:
        """Load current version from file or initialize"""
        if self.version_file.exists():
            try:
                with open(self.version_file) as f:
                    version_str = f.read().strip()
                return Version.from_string(version_str)
            except Exception as e:
                logger.error(f"Failed to load version: {e}")
        
        # Create initial version file
        version = Version.from_string(initial_version)
        self._save_current_version(version)
        return version
    
    def _save_current_version(self, version: Version):
        """Save current version to file"""
        with open(self.version_file, 'w') as f:
            f.write(str(version))
        logger.info(f"Current version updated to {version}")
    
    def _load_version_history(self) -> List[Version]:
        """Load version history"""
        if not self.changelog_file.exists():
            return []
        
        try:
            # Parse changelog for version history
            with open(self.changelog_file) as f:
                content = f.read()
            
            # Extract versions from changelog
            version_pattern = r'## \[(\d+\.\d+\.\d+(?:-[^\]]+)?)\]'
            versions = re.findall(version_pattern, content)
            
            version_objects = []
            for version_str in versions:
                try:
                    version_objects.append(Version.from_string(version_str))
                except ValueError as e:
                    logger.warning(f"Could not parse version {version_str}: {e}")
            
            return sorted(set(version_objects), reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to load version history: {e}")
            return []
    
    def _load_compatibility_matrix(self) -> Dict[str, CompatibilityMatrix]:
        """Load compatibility matrix"""
        if not self.compatibility_file.exists():
            return {}
        
        try:
            with open(self.compatibility_file) as f:
                data = json.load(f)
            
            matrix = {}
            for version_str, compat_data in data.items():
                version = Version.from_string(version_str)
                
                compatible_versions = [Version.from_string(v) for v in compat_data.get('compatible_with', [])]
                matrix[version_str] = CompatibilityMatrix(
                    version=version,
                    compatible_with=compatible_versions,
                    breaking_changes=compat_data.get('breaking_changes', []),
                    migration_notes=compat_data.get('migration_notes', []),
                    deprecated=compat_data.get('deprecated', False),
                    lts=compat_data.get('lts', False)
                )
            
            return matrix
            
        except Exception as e:
            logger.error(f"Failed to load compatibility matrix: {e}")
            return {}
    
    def bump_version(self, bump_type: str, prerelease: Optional[str] = None,
                    build: Optional[str] = None, message: Optional[str] = None) -> Version:
        """Bump version according to semantic versioning"""
        logger.info(f"Bumping version: {bump_type}")
        
        # Create new version
        if bump_type == "major":
            new_version = self.current_version.bump_major()
        elif bump_type == "minor":
            new_version = self.current_version.bump_minor()
        elif bump_type == "patch":
            new_version = self.current_version.bump_patch()
        elif bump_type == "prerelease":
            if prerelease:
                new_version = self.current_version.with_prerelease(prerelease)
            else:
                # Auto-increment prerelease
                current_prerelease = self.current_version.prerelease
                if current_prerelease and current_prerelease.startswith('alpha'):
                    new_version = self.current_version.with_prerelease('alpha.2')
                elif current_prerelease and current_prerelease.startswith('beta'):
                    new_version = self.current_version.with_prerelease('beta.2')
                elif current_prerelease and current_prerelease.startswith('rc'):
                    # RC -> release
                    new_version = self.current_version.remove_prerelease()
                else:
                    new_version = self.current_version.with_prerelease('alpha.1')
        else:
            raise ValueError(f"Invalid bump type: {bump_type}")
        
        # Add build metadata if specified
        if build:
            new_version = new_version.with_build(build)
        
        # Update current version
        old_version = self.current_version
        self.current_version = new_version
        self._save_current_version(new_version)
        
        # Update history
        if new_version not in self.version_history:
            self.version_history.insert(0, new_version)
        
        logger.info(f"Version bumped from {old_version} to {new_version}")
        
        return new_version
    
    def set_version(self, version_string: str) -> Version:
        """Set specific version"""
        new_version = Version.from_string(version_string)
        
        old_version = self.current_version
        self.current_version = new_version
        self._save_current_version(new_version)
        
        if new_version not in self.version_history:
            self.version_history.insert(0, new_version)
        
        logger.info(f"Version changed from {old_version} to {new_version}")
        return new_version
    
    def get_version_info(self) -> Dict[str, Any]:
        """Get comprehensive version information"""
        return {
            'current': self.current_version.to_dict(),
            'next_major': self.current_version.bump_major().to_dict(),
            'next_minor': self.current_version.bump_minor().to_dict(),
            'next_patch': self.current_version.bump_patch().to_dict(),
            'history': [v.to_dict() for v in self.version_history[:10]],  # Last 10 versions
            'total_versions': len(self.version_history),
            'is_prerelease': self.current_version.prerelease is not None,
            'has_build_metadata': self.current_version.build is not None
        }
    
    def compare_versions(self, version1: str, version2: str) -> Dict[str, Any]:
        """Compare two versions"""
        v1 = Version.from_string(version1)
        v2 = Version.from_string(version2)
        
        # Determine relationship
        if v1 == v2:
            relationship = "equal"
            difference = 0
        elif v1 < v2:
            relationship = "less_than"
            difference = self._calculate_version_distance(v1, v2)
        else:
            relationship = "greater_than"
            difference = self._calculate_version_distance(v2, v1)
        
        # Check for breaking changes
        breaking_changes = []
        if relationship in ["greater_than", "less_than"]:
            breaking_changes = self._get_breaking_changes(v1, v2)
        
        # Compatibility assessment
        compatibility = self._assess_compatibility(v1, v2)
        
        return {
            'version1': version1,
            'version2': version2,
            'relationship': relationship,
            'difference': difference,
            'breaking_changes': breaking_changes,
            'compatibility': compatibility,
            'migration_required': len(breaking_changes) > 0,
            'details': {
                'v1_greater': v1 > v2,
                'v2_greater': v2 > v1,
                'same_major': v1.major == v2.major,
                'same_minor': v1.major == v2.major and v1.minor == v2.minor
            }
        }
    
    def _calculate_version_distance(self, older: Version, newer: Version) -> Dict[str, int]:
        """Calculate distance between versions"""
        return {
            'major_diff': newer.major - older.major,
            'minor_diff': newer.minor - older.minor,
            'patch_diff': newer.patch - older.patch
        }
    
    def _get_breaking_changes(self, v1: Version, v2: Version) -> List[str]:
        """Get breaking changes between versions"""
        breaking_changes = []
        
        # Check compatibility matrix
        for version_str, compat_info in self.compatibility_matrix.items():
            if compat_info.version in [v1, v2]:
                breaking_changes.extend(compat_info.breaking_changes)
        
        # General breaking change rules
        if v1.major != v2.major:
            breaking_changes.append("Major version change - breaking changes expected")
        
        if v1.major == v2.major and v1.minor < v2.minor:
            breaking_changes.append("Minor version change - review migration notes")
        
        return list(set(breaking_changes))  # Remove duplicates
    
    def _assess_compatibility(self, v1: Version, v2: Version) -> Dict[str, Any]:
        """Assess compatibility between versions"""
        # Check if versions are compatible
        same_major = v1.major == v2.major
        same_minor = v1.major == v2.major and v1.minor == v2.minor
        
        compatibility_level = "incompatible"
        if same_minor:
            compatibility_level = "fully_compatible"
        elif same_major:
            compatibility_level = "mostly_compatible"
        
        # Check explicit compatibility matrix
        explicit_compatible = False
        if str(v1) in self.compatibility_matrix:
            explicit_compatible = v2 in self.compatibility_matrix[str(v1)].compatible_with
        elif str(v2) in self.compatibility_matrix:
            explicit_compatible = v1 in self.compatibility_matrix[str(v2)].compatible_with
        
        return {
            'level': compatibility_level,
            'same_major': same_major,
            'same_minor': same_minor,
            'explicit_compatible': explicit_compatible,
            'migration_effort': self._estimate_migration_effort(v1, v2)
        }
    
    def _estimate_migration_effort(self, from_version: Version, to_version: Version) -> str:
        """Estimate migration effort"""
        if from_version.major != to_version.major:
            return "high"  # Major version bump
        elif from_version.minor != to_version.minor:
            return "medium"  # Minor version bump
        else:
            return "low"  # Patch version bump
    
    def create_compatibility_entry(self, version_string: str, compatible_versions: List[str],
                                  breaking_changes: List[str], migration_notes: List[str],
                                  deprecated: bool = False, lts: bool = False) -> CompatibilityMatrix:
        """Create compatibility matrix entry"""
        version = Version.from_string(version_string)
        compatible_list = [Version.from_string(v) for v in compatible_versions]
        
        compatibility = CompatibilityMatrix(
            version=version,
            compatible_with=compatible_list,
            breaking_changes=breaking_changes,
            migration_notes=migration_notes,
            deprecated=deprecated,
            lts=lts
        )
        
        # Update matrix
        self.compatibility_matrix[version_string] = compatibility
        self._save_compatibility_matrix()
        
        logger.info(f"Created compatibility entry for {version_string}")
        return compatibility
    
    def _save_compatibility_matrix(self):
        """Save compatibility matrix to file"""
        matrix_data = {}
        for version_str, compat_info in self.compatibility_matrix.items():
            matrix_data[version_str] = {
                'compatible_with': [str(v) for v in compat_info.compatible_with],
                'breaking_changes': compat_info.breaking_changes,
                'migration_notes': compat_info.migration_notes,
                'deprecated': compat_info.deprecated,
                'lts': compat_info.lts
            }
        
        with open(self.compatibility_file, 'w') as f:
            json.dump(matrix_data, f, indent=2)
    
    def get_recommendations(self) -> Dict[str, Any]:
        """Get version upgrade recommendations"""
        recommendations = {
            'next_version': None,
            'reasoning': '',
            'breaking_changes': [],
            'migration_effort': 'low',
            'lts_candidates': []
        }
        
        # Determine next version recommendation
        if self.current_version.prerelease:
            if self.current_version.prerelease.startswith('rc'):
                # Release candidate -> stable
                recommendations['next_version'] = str(self.current_version.remove_prerelease())
                recommendations['reasoning'] = "RC ready for release"
            else:
                # Alpha/beta -> next prerelease
                next_version = self.current_version.bump_patch()
                recommendations['next_version'] = str(next_version)
                recommendations['reasoning'] = "Continue prerelease cycle"
        else:
            # Stable release
            recommendations['next_version'] = str(self.current_version.bump_patch())
            recommendations['reasoning'] = "Patch release for bug fixes"
        
        # Check for LTS candidates
        if self.current_version.patch == 0 and self.current_version.prerelease is None:
            recommendations['lts_candidates'].append(str(self.current_version))
        
        # Get breaking changes
        recommendations['breaking_changes'] = self._get_breaking_changes(
            self.current_version, Version.from_string(recommendations['next_version'])
        )
        
        return recommendations
    
    def validate_version_string(self, version_string: str) -> Dict[str, Any]:
        """Validate version string"""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'parsed_version': None
        }
        
        try:
            version = Version.from_string(version_string)
            validation_result['parsed_version'] = version.to_dict()
        except ValueError as e:
            validation_result['valid'] = False
            validation_result['errors'].append(str(e))
            return validation_result
        
        # Additional validation
        if version.major < 0:
            validation_result['errors'].append("Major version cannot be negative")
            validation_result['valid'] = False
        
        if version.minor < 0:
            validation_result['errors'].append("Minor version cannot be negative")
            validation_result['valid'] = False
        
        if version.patch < 0:
            validation_result['errors'].append("Patch version cannot be negative")
            validation_result['valid'] = False
        
        # Warnings
        if version.major == 0 and version.prerelease is None:
            validation_result['warnings'].append("Major version 0 is for initial development")
        
        if version.prerelease and version.prerelease.startswith('.'):
            validation_result['warnings'].append("Prerelease identifier should not start with '.'")
        
        return validation_result
    
    def generate_version_report(self, output_path: str) -> Dict[str, Any]:
        """Generate comprehensive version report"""
        report = {
            'generated_at': datetime.now().isoformat(),
            'current_version': self.current_version.to_dict(),
            'version_info': self.get_version_info(),
            'recommendations': self.get_recommendations(),
            'compatibility_matrix': {
                version_str: {
                    'version': compat_info.version.to_dict(),
                    'compatible_with': [v.to_dict() for v in compat_info.compatible_with],
                    'breaking_changes': compat_info.breaking_changes,
                    'migration_notes': compat_info.migration_notes,
                    'deprecated': compat_info.deprecated,
                    'lts': compat_info.lts
                }
                for version_str, compat_info in self.compatibility_matrix.items()
            },
            'recent_changes': self._get_recent_changes(),
            'version_statistics': self._get_version_statistics()
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Version report generated: {output_path}")
        return report
    
    def _get_recent_changes(self) -> List[Dict[str, Any]]:
        """Get recent version changes"""
        changes = []
        
        # Parse git log for version changes
        try:
            result = subprocess.run([
                'git', 'log', '--oneline', '--grep=version', '--grep=bump', '--all', '-10'
            ], capture_output=True, text=True, cwd=self.project_path)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if line.strip():
                        changes.append({
                            'commit': line.split(' ')[0],
                            'message': ' '.join(line.split(' ')[1:])
                        })
        except Exception as e:
            logger.warning(f"Could not fetch git changes: {e}")
        
        return changes
    
    def _get_version_statistics(self) -> Dict[str, Any]:
        """Get version statistics"""
        if not self.version_history:
            return {}
        
        major_versions = set(v.major for v in self.version_history)
        minor_versions = set((v.major, v.minor) for v in self.version_history)
        
        # Count prereleases
        prereleases = sum(1 for v in self.version_history if v.prerelease is not None)
        
        return {
            'total_versions': len(self.version_history),
            'unique_major_versions': len(major_versions),
            'unique_minor_versions': len(minor_versions),
            'prerelease_versions': prereleases,
            'stable_releases': len(self.version_history) - prereleases,
            'first_version': str(self.version_history[-1]),
            'latest_version': str(self.version_history[0])
        }
    
    def export_version_data(self, output_path: str, format: str = "json") -> str:
        """Export version data"""
        data = {
            'current_version': self.current_version.to_dict(),
            'version_history': [v.to_dict() for v in self.version_history],
            'compatibility_matrix': {
                version_str: {
                    'version': compat_info.version.to_dict(),
                    'compatible_with': [v.to_dict() for v in compat_info.compatible_with],
                    'breaking_changes': compat_info.breaking_changes,
                    'migration_notes': compat_info.migration_notes
                }
                for version_str, compat_info in self.compatibility_matrix.items()
            }
        }
        
        if format == "json":
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
        elif format == "yaml":
            with open(output_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Version data exported to {output_path}")
        return output_path


def main():
    """Main function for version management"""
    parser = argparse.ArgumentParser(description='Version management system')
    
    parser.add_argument('--project_path', required=True, help='Project path')
    parser.add_argument('--current_version', help='Current version')
    parser.add_argument('--action', required=True,
                       choices=['bump', 'set', 'info', 'compare', 'validate',
                               'recommend', 'report', 'export', 'compat_entry'],
                       help='Action to perform')
    
    # Action-specific arguments
    parser.add_argument('--bump_type', choices=['major', 'minor', 'patch', 'prerelease'],
                       help='Bump type for version bumping')
    parser.add_argument('--new_version', help='New version string')
    parser.add_argument('--prerelease', help='Prerelease identifier')
    parser.add_argument('--build', help='Build metadata')
    parser.add_argument('--version1', help='First version for comparison')
    parser.add_argument('--version2', help='Second version for comparison')
    parser.add_argument('--output_path', help='Output path for reports/exports')
    parser.add_argument('--format', default='json', help='Export format')
    parser.add_argument('--compatible_versions', nargs='+', help='Compatible versions')
    parser.add_argument('--breaking_changes', nargs='+', help='Breaking changes')
    parser.add_argument('--migration_notes', nargs='+', help='Migration notes')
    parser.add_argument('--deprecated', action='store_true', help='Mark as deprecated')
    parser.add_argument('--lts', action='store_true', help='Mark as LTS')
    
    args = parser.parse_args()
    
    # Initialize version manager
    initial_version = args.current_version or "1.0.0"
    manager = VersionManager(args.project_path, initial_version)
    
    # Execute action
    if args.action == 'bump':
        if not args.bump_type:
            print("Error: --bump_type required")
            return 1
        
        new_version = manager.bump_version(
            bump_type=args.bump_type,
            prerelease=args.prerelease,
            build=args.build
        )
        
        print(f"Version bumped to: {new_version}")
        
    elif args.action == 'set':
        if not args.new_version:
            print("Error: --new_version required")
            return 1
        
        new_version = manager.set_version(args.new_version)
        print(f"Version set to: {new_version}")
        
    elif args.action == 'info':
        info = manager.get_version_info()
        print(json.dumps(info, indent=2, default=str))
        
    elif args.action == 'compare':
        if not args.version1 or not args.version2:
            print("Error: --version1 and --version2 required")
            return 1
        
        comparison = manager.compare_versions(args.version1, args.version2)
        print(json.dumps(comparison, indent=2, default=str))
        
    elif args.action == 'validate':
        if not args.new_version:
            print("Error: --new_version required")
            return 1
        
        validation = manager.validate_version_string(args.new_version)
        print(json.dumps(validation, indent=2, default=str))
        
    elif args.action == 'recommend':
        recommendations = manager.get_recommendations()
        print(json.dumps(recommendations, indent=2, default=str))
        
    elif args.action == 'report':
        if not args.output_path:
            print("Error: --output_path required")
            return 1
        
        report = manager.generate_version_report(args.output_path)
        print(f"Version report generated: {args.output_path}")
        
    elif args.action == 'export':
        if not args.output_path:
            print("Error: --output_path required")
            return 1
        
        export_path = manager.export_version_data(args.output_path, format=args.format)
        print(f"Version data exported: {export_path}")
        
    elif args.action == 'compat_entry':
        if not args.new_version or not args.compatible_versions:
            print("Error: --new_version and --compatible_versions required")
            return 1
        
        compatibility = manager.create_compatibility_entry(
            version_string=args.new_version,
            compatible_versions=args.compatible_versions,
            breaking_changes=args.breaking_changes or [],
            migration_notes=args.migration_notes or [],
            deprecated=args.deprecated,
            lts=args.lts
        )
        
        print(f"Compatibility entry created for {args.new_version}")
        print(json.dumps({
            'version': compatibility.version.to_dict(),
            'compatible_with': [v.to_dict() for v in compatibility.compatible_with],
            'breaking_changes': compatibility.breaking_changes,
            'migration_notes': compatibility.migration_notes
        }, indent=2, default=str))
    
    else:
        print(f"Unknown action: {args.action}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())