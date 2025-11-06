#!/usr/bin/env python3
"""
GitHub Repository Management for Automated Deployment
Handles repository updates, versioning, and release management
"""

import os
import sys
import json
import argparse
import logging
import subprocess
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
import requests
import base64
from urllib.parse import urlparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GitHubManager:
    """GitHub repository management for deployment automation"""
    
    def __init__(self, token: str, repo_owner: str, repo_name: str):
        self.token = token
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.api_base = "https://api.github.com"
        self.headers = {
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json',
            'Content-Type': 'application/json'
        }
        
        logger.info(f"Initialized GitHub manager for {repo_owner}/{repo_name}")
    
    def get_repository_info(self) -> Dict:
        """Get repository information"""
        logger.info("Getting repository information")
        
        try:
            url = f"{self.api_base}/repos/{self.repo_owner}/{self.repo_name}"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            repo_data = response.json()
            
            result = {
                'action': 'get_repository_info',
                'status': 'success',
                'repo_info': {
                    'name': repo_data['name'],
                    'full_name': repo_data['full_name'],
                    'description': repo_data['description'],
                    'private': repo_data['private'],
                    'default_branch': repo_data['default_branch'],
                    'created_at': repo_data['created_at'],
                    'updated_at': repo_data['updated_at'],
                    'pushed_at': repo_data['pushed_at'],
                    'size': repo_data['size'],
                    'language': repo_data['language'],
                    'topics': repo_data.get('topics', []),
                    'html_url': repo_data['html_url'],
                    'clone_url': repo_data['clone_url'],
                    'ssh_url': repo_data['ssh_url']
                }
            }
            
            logger.info(f"Repository info retrieved: {result['repo_info']['html_url']}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to get repository info: {str(e)}")
            return {'action': 'get_repository_info', 'error': str(e)}
    
    def create_release(self, tag_name: str, name: str, body: str, 
                      target_commitish: str = "main", draft: bool = False,
                      prerelease: bool = False, generate_release_notes: bool = True) -> Dict:
        """Create a new release"""
        logger.info(f"Creating release: {tag_name}")
        
        try:
            url = f"{self.api_base}/repos/{self.repo_owner}/{self.repo_name}/releases"
            
            payload = {
                'tag_name': tag_name,
                'target_commitish': target_commitish,
                'name': name,
                'body': body,
                'draft': draft,
                'prerelease': prerelease,
                'generate_release_notes': generate_release_notes
            }
            
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            release_data = response.json()
            
            result = {
                'action': 'create_release',
                'status': 'success',
                'release_info': {
                    'id': release_data['id'],
                    'tag_name': release_data['tag_name'],
                    'name': release_data['name'],
                    'body': release_data['body'],
                    'draft': release_data['draft'],
                    'prerelease': release_data['prerelease'],
                    'html_url': release_data['html_url'],
                    'upload_url': release_data['upload_url'],
                    'created_at': release_data['created_at'],
                    'published_at': release_data.get('published_at'),
                    'tarball_url': release_data['tarball_url'],
                    'zipball_url': release_data['zipball_url']
                }
            }
            
            logger.info(f"Release created: {result['release_info']['html_url']}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to create release: {str(e)}")
            return {'action': 'create_release', 'error': str(e)}
    
    def upload_release_asset(self, release_id: int, file_path: Union[str, Path], 
                           name: Optional[str] = None) -> Dict:
        """Upload an asset to a release"""
        logger.info(f"Uploading asset: {file_path}")
        
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Use filename as default name
            asset_name = name or file_path.name
            
            # Get upload URL from release
            url = f"{self.api_base}/repos/{self.repo_owner}/{self.repo_name}/releases/{release_id}"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            release_data = response.json()
            upload_url = release_data['upload_url']
            
            # Remove {?name,label} from upload URL
            upload_url = upload_url.replace('{?name,label}', '')
            
            # Upload asset
            with open(file_path, 'rb') as f:
                files = {'file': (asset_name, f)}
                upload_params = {'name': asset_name}
                
                upload_response = requests.post(
                    upload_url,
                    headers={'Authorization': f'token {self.token}'},
                    files=files,
                    params=upload_params
                )
                upload_response.raise_for_status()
            
            asset_data = upload_response.json()
            
            result = {
                'action': 'upload_release_asset',
                'status': 'success',
                'asset_info': {
                    'id': asset_data['id'],
                    'name': asset_data['name'],
                    'size': asset_data['size'],
                    'download_count': asset_data['download_count'],
                    'browser_download_url': asset_data['browser_download_url'],
                    'created_at': asset_data['created_at'],
                    'updated_at': asset_data['updated_at']
                }
            }
            
            logger.info(f"Asset uploaded: {result['asset_info']['browser_download_url']}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to upload asset: {str(e)}")
            return {'action': 'upload_release_asset', 'error': str(e)}
    
    def update_file(self, path: str, content: str, message: str, 
                   branch: str = "main", sha: Optional[str] = None) -> Dict:
        """Update or create a file in the repository"""
        logger.info(f"Updating file: {path}")
        
        try:
            # Get current file SHA if updating existing file
            if not sha:
                get_url = f"{self.api_base}/repos/{self.repo_owner}/{self.repo_name}/contents/{path}"
                get_params = {'ref': branch}
                
                get_response = requests.get(get_url, headers=self.headers, params=get_params)
                
                if get_response.status_code == 200:
                    file_data = get_response.json()
                    sha = file_data['sha']
                elif get_response.status_code != 404:
                    # Other error codes should be handled
                    get_response.raise_for_status()
            
            # Prepare content (base64 encode)
            content_encoded = base64.b64encode(content.encode('utf-8')).decode('utf-8')
            
            # Update/create file
            url = f"{self.api_base}/repos/{self.repo_owner}/{self.repo_name}/contents/{path}"
            
            payload = {
                'message': message,
                'content': content_encoded,
                'branch': branch
            }
            
            if sha:
                payload['sha'] = sha
            
            response = requests.put(url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            file_data = response.json()
            
            result = {
                'action': 'update_file',
                'status': 'success',
                'file_info': {
                    'name': file_data['content']['name'],
                    'path': file_data['content']['path'],
                    'sha': file_data['content']['sha'],
                    'size': file_data['content']['size'],
                    'html_url': file_data['content']['html_url'],
                    'git_url': file_data['content']['git_url'],
                    'download_url': file_data['content']['download_url']
                },
                'commit': {
                    'sha': file_data['commit']['sha'],
                    'html_url': file_data['commit']['html_url'],
                    'message': file_data['commit']['message']
                }
            }
            
            logger.info(f"File updated: {result['file_info']['path']}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to update file: {str(e)}")
            return {'action': 'update_file', 'error': str(e)}
    
    def create_directory(self, path: str, message: str, branch: str = "main") -> Dict:
        """Create a directory in the repository"""
        logger.info(f"Creating directory: {path}")
        
        try:
            # Create placeholder file to create directory
            content = "# Directory placeholder\n"
            content_encoded = base64.b64encode(content.encode('utf-8')).decode('utf-8')
            
            # Add .gitkeep to path
            dir_path = f"{path}/.gitkeep"
            
            url = f"{self.api_base}/repos/{self.repo_owner}/{self.repo_name}/contents/{dir_path}"
            
            payload = {
                'message': message,
                'content': content_encoded,
                'branch': branch
            }
            
            response = requests.put(url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            dir_data = response.json()
            
            result = {
                'action': 'create_directory',
                'status': 'success',
                'directory_info': {
                    'name': dir_data['content']['name'],
                    'path': dir_data['content']['path'],
                    'sha': dir_data['content']['sha'],
                    'size': dir_data['content']['size']
                }
            }
            
            logger.info(f"Directory created: {result['directory_info']['path']}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to create directory: {str(e)}")
            return {'action': 'create_directory', 'error': str(e)}
    
    def upload_directory(self, local_dir: Union[str, Path], repo_dir: str, 
                        branch: str = "main", message: str = "Upload directory") -> Dict:
        """Upload entire directory to repository"""
        logger.info(f"Uploading directory: {local_dir} -> {repo_dir}")
        
        try:
            local_dir = Path(local_dir)
            if not local_dir.exists():
                raise FileNotFoundError(f"Local directory not found: {local_dir}")
            
            upload_results = []
            
            for file_path in local_dir.rglob('*'):
                if file_path.is_file():
                    # Calculate relative path
                    relative_path = file_path.relative_to(local_dir)
                    repo_path = f"{repo_dir}/{relative_path}".replace('\\', '/')
                    
                    # Read file content
                    with open(file_path, 'rb') as f:
                        file_content = f.read()
                    
                    # Determine if binary or text
                    try:
                        content_str = file_content.decode('utf-8')
                        is_binary = False
                    except UnicodeDecodeError:
                        content_str = base64.b64encode(file_content).decode('utf-8')
                        is_binary = True
                    
                    # Upload file
                    upload_result = self.update_file(
                        path=repo_path,
                        content=content_str,
                        message=f"{message}: {relative_path}",
                        branch=branch
                    )
                    
                    upload_result['is_binary'] = is_binary
                    upload_result['local_path'] = str(file_path)
                    upload_result['repo_path'] = repo_path
                    
                    upload_results.append(upload_result)
            
            successful_uploads = [r for r in upload_results if r.get('status') == 'success']
            failed_uploads = [r for r in upload_results if r.get('status') != 'success']
            
            result = {
                'action': 'upload_directory',
                'status': 'completed',
                'total_files': len(upload_results),
                'successful_uploads': len(successful_uploads),
                'failed_uploads': len(failed_uploads),
                'upload_results': upload_results
            }
            
            logger.info(f"Directory upload completed: {len(successful_uploads)}/{len(upload_results)} files")
            return result
            
        except Exception as e:
            logger.error(f"Directory upload failed: {str(e)}")
            return {'action': 'upload_directory', 'error': str(e)}
    
    def create_branch(self, branch_name: str, from_branch: str = "main") -> Dict:
        """Create a new branch"""
        logger.info(f"Creating branch: {branch_name}")
        
        try:
            # Get SHA of source branch
            url = f"{self.api_base}/repos/{self.repo_owner}/{self.repo_name}/git/ref/heads/{from_branch}"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            ref_data = response.json()
            source_sha = ref_data['object']['sha']
            
            # Create new branch
            url = f"{self.api_base}/repos/{self.repo_owner}/{self.repo_name}/git/refs"
            
            payload = {
                'ref': f'refs/heads/{branch_name}',
                'sha': source_sha
            }
            
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            branch_data = response.json()
            
            result = {
                'action': 'create_branch',
                'status': 'success',
                'branch_info': {
                    'ref': branch_data['ref'],
                    'sha': branch_data['object']['sha'],
                    'url': branch_data['url']
                }
            }
            
            logger.info(f"Branch created: {branch_name}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to create branch: {str(e)}")
            return {'action': 'create_branch', 'error': str(e)}
    
    def create_pull_request(self, title: str, head: str, base: str = "main", 
                           body: str = "", draft: bool = False) -> Dict:
        """Create a pull request"""
        logger.info(f"Creating pull request: {title}")
        
        try:
            url = f"{self.api_base}/repos/{self.repo_owner}/{self.repo_name}/pulls"
            
            payload = {
                'title': title,
                'head': head,
                'base': base,
                'body': body,
                'draft': draft
            }
            
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            pr_data = response.json()
            
            result = {
                'action': 'create_pull_request',
                'status': 'success',
                'pr_info': {
                    'number': pr_data['number'],
                    'title': pr_data['title'],
                    'body': pr_data['body'],
                    'html_url': pr_data['html_url'],
                    'diff_url': pr_data['diff_url'],
                    'patch_url': pr_data['patch_url'],
                    'state': pr_data['state'],
                    'draft': pr_data['draft'],
                    'head': {
                        'ref': pr_data['head']['ref'],
                        'sha': pr_data['head']['sha']
                    },
                    'base': {
                        'ref': pr_data['base']['ref'],
                        'sha': pr_data['base']['sha']
                    },
                    'created_at': pr_data['created_at'],
                    'updated_at': pr_data['updated_at']
                }
            }
            
            logger.info(f"Pull request created: {result['pr_info']['html_url']}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to create pull request: {str(e)}")
            return {'action': 'create_pull_request', 'error': str(e)}
    
    def create_tag(self, tag: str, message: str, object_sha: str, 
                  object_type: str = "commit") -> Dict:
        """Create an annotated tag"""
        logger.info(f"Creating tag: {tag}")
        
        try:
            url = f"{self.api_base}/repos/{self.repo_owner}/{self.repo_name}/git/tags"
            
            payload = {
                'tag': tag,
                'message': message,
                'object': object_sha,
                'type': object_type
            }
            
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            tag_data = response.json()
            
            result = {
                'action': 'create_tag',
                'status': 'success',
                'tag_info': {
                    'tag': tag_data['tag'],
                    'sha': tag_data['sha'],
                    'url': tag_data['url'],
                    'message': tag_data['message'],
                    'tagger': tag_data['tagger'],
                    'object': {
                        'sha': tag_data['object']['sha'],
                        'type': tag_data['object']['type'],
                        'url': tag_data['object']['url']
                    }
                }
            }
            
            logger.info(f"Tag created: {tag}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to create tag: {str(e)}")
            return {'action': 'create_tag', 'error': str(e)}
    
    def get_commits_since(self, since: str, until: Optional[str] = None, 
                         path: Optional[str] = None) -> Dict:
        """Get commits since a specific date"""
        logger.info(f"Getting commits since: {since}")
        
        try:
            url = f"{self.api_base}/repos/{self.repo_owner}/{self.repo_name}/commits"
            
            params = {'since': since}
            if until:
                params['until'] = until
            if path:
                params['path'] = path
            
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            
            commits_data = response.json()
            
            result = {
                'action': 'get_commits_since',
                'status': 'success',
                'commits_count': len(commits_data),
                'commits': []
            }
            
            for commit in commits_data:
                commit_info = {
                    'sha': commit['sha'],
                    'message': commit['commit']['message'],
                    'author': {
                        'name': commit['commit']['author']['name'],
                        'email': commit['commit']['author']['email'],
                        'date': commit['commit']['author']['date']
                    },
                    'committer': {
                        'name': commit['commit']['committer']['name'],
                        'email': commit['commit']['committer']['email'],
                        'date': commit['commit']['committer']['date']
                    },
                    'url': commit['html_url']
                }
                result['commits'].append(commit_info)
            
            logger.info(f"Retrieved {result['commits_count']} commits")
            return result
            
        except Exception as e:
            logger.error(f"Failed to get commits: {str(e)}")
            return {'action': 'get_commits_since', 'error': str(e)}
    
    def update_repository_topics(self, topics: List[str]) -> Dict:
        """Update repository topics"""
        logger.info(f"Updating repository topics: {topics}")
        
        try:
            url = f"{self.api_base}/repos/{self.repo_owner}/{self.repo_name}"
            
            payload = {'topics': topics}
            
            response = requests.put(url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            repo_data = response.json()
            
            result = {
                'action': 'update_repository_topics',
                'status': 'success',
                'topics': repo_data.get('topics', [])
            }
            
            logger.info(f"Topics updated: {result['topics']}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to update topics: {str(e)}")
            return {'action': 'update_repository_topics', 'error': str(e)}
    
    def setup_deployment_workflows(self, local_workflows_dir: Union[str, Path]) -> Dict:
        """Setup GitHub Actions workflows"""
        logger.info("Setting up GitHub Actions workflows")
        
        try:
            workflows_dir = Path(local_workflows_dir)
            if not workflows_dir.exists():
                raise FileNotFoundError(f"Workflows directory not found: {workflows_dir}")
            
            # Ensure .github/workflows directory exists in repo
            workflows_dir_repo = ".github/workflows"
            self.create_directory(workflows_dir_repo, "Create workflows directory")
            
            upload_results = []
            
            for workflow_file in workflows_dir.glob('*.yml'):
                with open(workflow_file) as f:
                    workflow_content = f.read()
                
                upload_result = self.update_file(
                    path=f"{workflows_dir_repo}/{workflow_file.name}",
                    content=workflow_content,
                    message=f"Add workflow: {workflow_file.name}"
                )
                
                upload_results.append(upload_result)
            
            successful = [r for r in upload_results if r.get('status') == 'success']
            
            result = {
                'action': 'setup_deployment_workflows',
                'status': 'completed',
                'workflows_uploaded': len(successful),
                'total_workflows': len(upload_results),
                'upload_results': upload_results
            }
            
            logger.info(f"Workflows setup completed: {len(successful)}/{len(upload_results)}")
            return result
            
        except Exception as e:
            logger.error(f"Workflows setup failed: {str(e)}")
            return {'action': 'setup_deployment_workflows', 'error': str(e)}


def main():
    """Main function for GitHub management"""
    parser = argparse.ArgumentParser(description='GitHub repository management')
    
    parser.add_argument('--token', required=True, help='GitHub API token')
    parser.add_argument('--owner', required=True, help='Repository owner')
    parser.add_argument('--repo', required=True, help='Repository name')
    parser.add_argument('--action', required=True,
                       choices=['repo_info', 'create_release', 'upload_asset', 
                               'update_file', 'upload_dir', 'create_branch',
                               'create_pr', 'create_tag', 'get_commits',
                               'update_topics', 'setup_workflows'],
                       help='Action to perform')
    
    # Action-specific arguments
    parser.add_argument('--tag_name', help='Tag name for release')
    parser.add_argument('--release_name', help='Release name')
    parser.add_argument('--release_body', help='Release body/notes')
    parser.add_argument('--release_file', help='File to upload as release asset')
    parser.add_argument('--file_path', help='File path in repository')
    parser.add_argument('--local_file', help='Local file to upload')
    parser.add_argument('--content', help='File content')
    parser.add_argument('--message', help='Commit message')
    parser.add_argument('--branch', default='main', help='Branch name')
    parser.add_argument('--head_branch', help='Head branch for PR')
    parser.add_argument('--title', help='PR title or file path')
    parser.add_argument('--topics', nargs='+', help='Repository topics')
    parser.add_argument('--workflows_dir', help='Local workflows directory')
    
    args = parser.parse_args()
    
    # Initialize manager
    manager = GitHubManager(
        token=args.token,
        repo_owner=args.owner,
        repo_name=args.repo
    )
    
    # Execute action
    if args.action == 'repo_info':
        result = manager.get_repository_info()
    elif args.action == 'create_release':
        if not args.tag_name or not args.release_name:
            result = {'status': 'error', 'error': 'tag_name and release_name required'}
        else:
            result = manager.create_release(
                tag_name=args.tag_name,
                name=args.release_name,
                body=args.release_body or ""
            )
    elif args.action == 'upload_asset':
        if not args.release_file:
            result = {'status': 'error', 'error': 'release_file required'}
        else:
            # Note: This would need release_id, which would be obtained from create_release
            result = {'status': 'error', 'error': 'Implementation requires release_id'}
    elif args.action == 'update_file':
        if not args.file_path or not args.content:
            result = {'status': 'error', 'error': 'file_path and content required'}
        else:
            result = manager.update_file(
                path=args.file_path,
                content=args.content,
                message=args.message or f"Update {args.file_path}",
                branch=args.branch
            )
    elif args.action == 'upload_dir':
        if not args.local_file:
            result = {'status': 'error', 'error': 'local_file required (directory path)'}
        else:
            result = manager.upload_directory(
                local_dir=args.local_file,
                repo_dir=args.file_path or "uploaded",
                branch=args.branch,
                message=args.message or "Upload directory"
            )
    elif args.action == 'create_branch':
        if not args.title:  # Using title for branch name
            result = {'status': 'error', 'error': 'title required for branch name'}
        else:
            result = manager.create_branch(
                branch_name=args.title,
                from_branch=args.branch
            )
    elif args.action == 'create_pr':
        if not args.title or not args.head_branch:
            result = {'status': 'error', 'error': 'title and head_branch required'}
        else:
            result = manager.create_pull_request(
                title=args.title,
                head=args.head_branch,
                base=args.branch,
                body=args.release_body or ""
            )
    elif args.action == 'create_tag':
        if not args.tag_name:
            result = {'status': 'error', 'error': 'tag_name required'}
        else:
            result = manager.create_tag(
                tag=args.tag_name,
                message=args.message or f"Tag {args.tag_name}",
                object_sha="main"  # Would need to get actual SHA
            )
    elif args.action == 'get_commits':
        result = manager.get_commits_since(
            since=args.message or "2024-01-01T00:00:00Z",
            path=args.file_path
        )
    elif args.action == 'update_topics':
        if not args.topics:
            result = {'status': 'error', 'error': 'topics required'}
        else:
            result = manager.update_repository_topics(args.topics)
    elif args.action == 'setup_workflows':
        if not args.workflows_dir:
            result = {'status': 'error', 'error': 'workflows_dir required'}
        else:
            result = manager.setup_deployment_workflows(args.workflows_dir)
    else:
        result = {'status': 'error', 'error': f'Unknown action: {args.action}'}
    
    # Print result
    print(json.dumps(result, indent=2, default=str))
    
    # Return exit code
    return 0 if result.get('status') == 'success' or result.get('status') == 'completed' else 1


if __name__ == '__main__':
    sys.exit(main())