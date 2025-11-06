#!/usr/bin/env python3
"""
Web Development Specific Tests for Sheikh-2.5-Coder
Evaluates model capabilities across JavaScript/TypeScript, React, XML, MDX, and CSS generation
"""

import os
import sys
import json
import yaml
import argparse
import logging
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import time
import re
import ast
import subprocess
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add parent directories to path
sys.path.append('../')
sys.path.append('../../')

class WebDevEvaluator:
    """Web development specific evaluation framework"""
    
    def __init__(self, config_path: str, model_path: str, output_path: str, run_id: str):
        """Initialize web development evaluator"""
        self.config_path = config_path
        self.model_path = Path(model_path)
        self.output_path = Path(output_path)
        self.run_id = run_id
        
        # Ensure output directory exists
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config()
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize model components
        self.model = None
        self.tokenizer = None
        
        # Web development test settings
        self.web_dev_config = self.config.get('web_dev_tests', {})
        self.target_quality = self.config.get('targets', {}).get('web_dev_quality', 0.75)
        
        self.logger.info(f"Web Dev Evaluator initialized for run: {run_id}")
    
    def _load_config(self) -> Dict:
        """Load evaluation configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Default configuration if loading fails"""
        return {
            'web_dev_tests': {
                'javascript_typescript': {
                    'tasks': [
                        'Create a responsive navbar component',
                        'Implement form validation with error handling',
                        'Build a React component with hooks',
                        'Create an async data fetching service',
                        'Implement event-driven DOM manipulation'
                    ]
                },
                'react_components': {
                    'test_components': ['Button', 'Modal', 'DataTable', 'Form', 'Navigation']
                },
                'xml_configuration': {
                    'test_files': ['web.config', 'package.json', 'webpack.config.js']
                },
                'mdx_documentation': {
                    'components': ['getStarted', 'apiReference', 'codeExamples']
                },
                'css_styling': {
                    'test_categories': ['responsive_design', 'animations', 'layout_systems']
                }
            },
            'targets': {
                'web_dev_quality': 0.75
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for this evaluator"""
        log_file = self.output_path / f"webdev_{self.run_id}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(f'WebDevEvaluator_{self.run_id}')
    
    def load_model(self) -> bool:
        """Load model and tokenizer"""
        self.logger.info(f"Loading model from {self.model_path}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                padding_side="right"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map='auto',
                trust_remote_code=True
            )
            
            self.logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            return False
    
    def generate_code(self, prompt: str, language: str = 'javascript') -> str:
        """Generate code based on prompt"""
        try:
            # Add language-specific context
            enhanced_prompt = f"Language: {language}\nTask: {prompt}\n\nGenerate code:"
            
            # Tokenize input
            inputs = self.tokenizer(enhanced_prompt, return_tensors='pt').to(self.model.device)
            
            # Generate response
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=400,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract generated code
            generated_code = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            return generated_code
            
        except Exception as e:
            self.logger.error(f"Code generation failed: {str(e)}")
            return ""
    
    def evaluate_javascript_typescript(self) -> Dict:
        """Evaluate JavaScript/TypeScript generation capabilities"""
        self.logger.info("Evaluating JavaScript/TypeScript capabilities...")
        
        js_config = self.web_dev_config.get('javascript_typescript', {})
        tasks = js_config.get('tasks', [])
        
        results = []
        
        for i, task in enumerate(tasks):
            self.logger.info(f"Evaluating JavaScript task {i+1}/{len(tasks)}: {task}")
            
            # Generate code
            generated_code = self.generate_code(task, 'javascript')
            
            # Evaluate quality
            quality_score = self._evaluate_js_quality(generated_code, task)
            
            results.append({
                'task': task,
                'generated_code': generated_code[:500] + '...' if len(generated_code) > 500 else generated_code,
                'quality_score': quality_score,
                'syntax_valid': self._validate_js_syntax(generated_code),
                'feature_completeness': self._evaluate_js_features(generated_code, task),
                'code_style_score': self._evaluate_js_style(generated_code)
            })
        
        # Calculate aggregate score
        if results:
            avg_quality = np.mean([r['quality_score'] for r in results])
            avg_syntax_validity = np.mean([r['syntax_valid'] for r in results])
            avg_completeness = np.mean([r['feature_completeness'] for r in results])
        else:
            avg_quality = avg_syntax_validity = avg_completeness = 0
        
        return {
            'category': 'JavaScript/TypeScript',
            'individual_scores': results,
            'aggregate_scores': {
                'average_quality': avg_quality,
                'syntax_validity_rate': avg_syntax_validity,
                'feature_completeness': avg_completeness
            },
            'overall_score': avg_quality
        }
    
    def _evaluate_js_quality(self, code: str, task: str) -> float:
        """Evaluate JavaScript code quality"""
        score = 0.0
        
        # Length check
        if len(code) > 50:
            score += 0.1
        
        # Task-specific keyword analysis
        task_lower = task.lower()
        
        if 'responsive navbar' in task_lower:
            keywords = ['responsive', 'navbar', 'nav', 'menu', 'hamburger', 'flex', 'grid']
            found_keywords = sum(1 for keyword in keywords if keyword in code.lower())
            score += (found_keywords / len(keywords)) * 0.4
        
        elif 'form validation' in task_lower:
            keywords = ['validate', 'required', 'pattern', 'addEventListener', 'form', 'input']
            found_keywords = sum(1 for keyword in keywords if keyword in code.lower())
            score += (found_keywords / len(keywords)) * 0.4
        
        elif 'react component' in task_lower:
            keywords = ['react', 'component', 'function', 'useState', 'useEffect', 'jsx']
            found_keywords = sum(1 for keyword in keywords if keyword in code.lower())
            score += (found_keywords / len(keywords)) * 0.4
        
        elif 'async data' in task_lower:
            keywords = ['async', 'await', 'fetch', 'api', 'promise', 'axios']
            found_keywords = sum(1 for keyword in keywords if keyword in code.lower())
            score += (found_keywords / len(keywords)) * 0.4
        
        elif 'dom manipulation' in task_lower:
            keywords = ['getElementById', 'querySelector', 'appendChild', 'innerHTML', 'addEventListener']
            found_keywords = sum(1 for keyword in keywords if keyword in code.lower())
            score += (found_keywords / len(keywords)) * 0.4
        
        # General JavaScript features
        js_features = ['const', 'let', 'arrow', 'function', 'class', 'async', 'await', '=>']
        found_features = sum(1 for feature in js_features if feature in code)
        score += (found_features / len(js_features)) * 0.2
        
        # Structure validation
        if code.count('{') == code.count('}'):
            score += 0.1
        
        if code.count('(') == code.count(')'):
            score += 0.1
        
        return min(score, 1.0)
    
    def _validate_js_syntax(self, code: str) -> bool:
        """Basic JavaScript syntax validation"""
        try:
            # Check bracket/brace matching
            stack = []
            for char in code:
                if char in '({[':
                    stack.append(char)
                elif char in ')}]':
                    if not stack:
                        return False
                    opening = stack.pop()
                    if not self._matching_brackets(opening, char):
                        return False
            
            return len(stack) == 0
        except Exception:
            return False
    
    def _matching_brackets(self, open_bracket: str, close_bracket: str) -> bool:
        """Check if brackets match"""
        pairs = {'(': ')', '[': ']', '{': '}'}
        return pairs.get(open_bracket) == close_bracket
    
    def _evaluate_js_features(self, code: str, task: str) -> float:
        """Evaluate presence of task-specific features"""
        features_found = 0
        total_expected_features = 0
        
        task_lower = task.lower()
        
        if 'responsive navbar' in task_lower:
            expected_features = ['responsive', 'media', 'hamburger', 'nav', 'flex']
            total_expected_features = len(expected_features)
            features_found = sum(1 for feature in expected_features if feature in code.lower())
        
        elif 'form validation' in task_lower:
            expected_features = ['validate', 'required', 'pattern', 'error']
            total_expected_features = len(expected_features)
            features_found = sum(1 for feature in expected_features if feature in code.lower())
        
        elif 'react component' in task_lower:
            expected_features = ['react', 'jsx', 'component', 'function']
            total_expected_features = len(expected_features)
            features_found = sum(1 for feature in expected_features if feature.lower() in code.lower())
        
        if total_expected_features == 0:
            return 0.5  # Default score if no specific features
        
        return features_found / total_expected_features
    
    def _evaluate_js_style(self, code: str) -> float:
        """Evaluate JavaScript coding style"""
        score = 0.0
        
        # Check for modern ES6+ features
        modern_features = ['const', 'let', 'arrow', '=>', 'template', 'spread']
        found_features = sum(1 for feature in modern_features if feature in code)
        score += (found_features / len(modern_features)) * 0.4
        
        # Check for proper naming conventions
        if re.search(r'\b[a-z][a-zA-Z0-9]*\b', code):  # Basic camelCase check
            score += 0.2
        
        # Check for comments (basic indicator of documentation)
        if '//' in code or '/*' in code:
            score += 0.2
        
        # Check for error handling patterns
        if 'try' in code or 'catch' in code or 'error' in code.lower():
            score += 0.2
        
        return min(score, 1.0)
    
    def evaluate_react_components(self) -> Dict:
        """Evaluate React component generation capabilities"""
        self.logger.info("Evaluating React component capabilities...")
        
        react_config = self.web_dev_config.get('react_components', {})
        components = react_config.get('test_components', [])
        
        results = []
        
        for component in components:
            self.logger.info(f"Evaluating React component: {component}")
            
            # Create component-specific prompt
            prompt = f"Create a {component.lower()} React component with proper props and styling"
            generated_code = self.generate_code(prompt, 'jsx')
            
            # Evaluate component quality
            quality_score = self._evaluate_react_component(generated_code, component)
            
            results.append({
                'component': component,
                'generated_code': generated_code[:500] + '...' if len(generated_code) > 500 else generated_code,
                'quality_score': quality_score,
                'jsx_valid': self._validate_jsx_syntax(generated_code),
                'component_structure': self._evaluate_component_structure(generated_code),
                'prop_usage': self._evaluate_prop_usage(generated_code)
            })
        
        # Calculate aggregate scores
        if results:
            avg_quality = np.mean([r['quality_score'] for r in results])
            avg_jsx_validity = np.mean([r['jsx_valid'] for r in results])
        else:
            avg_quality = avg_jsx_validity = 0
        
        return {
            'category': 'React Components',
            'individual_scores': results,
            'aggregate_scores': {
                'average_quality': avg_quality,
                'jsx_validity_rate': avg_jsx_validity
            },
            'overall_score': avg_quality
        }
    
    def _evaluate_react_component(self, code: str, component_name: str) -> float:
        """Evaluate React component quality"""
        score = 0.0
        
        # Check for React import
        if 'react' in code.lower() or 'from \'react\'' in code or 'from "react"' in code:
            score += 0.2
        
        # Check for component definition
        component_patterns = [
            rf'function\s+{component_name}\s*\(',
            rf'const\s+{component_name}\s*=',
            rf'class\s+{component_name}\s+extends'
        ]
        
        for pattern in component_patterns:
            if re.search(pattern, code):
                score += 0.3
                break
        
        # Check for JSX syntax
        jsx_indicators = ['<', '>', '/>', '</', 'jsx']
        jsx_score = sum(1 for indicator in jsx_indicators if indicator in code)
        score += (jsx_score / len(jsx_indicators)) * 0.2
        
        # Component-specific checks
        if component_name.lower() in ['button', 'modal', 'form']:
            if 'className' in code or 'style' in code:
                score += 0.2
        
        # Props destructuring
        if 'props' in code or '}' in code:
            score += 0.1
        
        return min(score, 1.0)
    
    def _validate_jsx_syntax(self, code: str) -> bool:
        """Validate JSX syntax"""
        # Basic JSX validation - check for proper tag matching
        try:
            # Remove comments and strings for analysis
            cleaned_code = re.sub(r'//.*?\n|/\*.*?\*/', '', code, flags=re.DOTALL)
            cleaned_code = re.sub(r'"[^"]*"|\'[^\']*\'', '', cleaned_code)
            
            # Check for self-closing tags
            self_closing_pattern = r'<[^>]+/>'
            if re.search(self_closing_pattern, cleaned_code):
                return True
            
            # Basic bracket matching for JSX
            stack = []
            i = 0
            while i < len(cleaned_code):
                char = cleaned_code[i]
                if char == '<':
                    if i + 1 < len(cleaned_code) and cleaned_code[i + 1] == '/':
                        # Closing tag
                        tag_end = cleaned_code.find('>', i)
                        if tag_end != -1:
                            tag_name = cleaned_code[i+2:tag_end].strip()
                            if stack and stack[-1] == tag_name:
                                stack.pop()
                    else:
                        # Opening tag
                        tag_end = cleaned_code.find(' ', i)
                        if tag_end == -1:
                            tag_end = cleaned_code.find('>', i)
                        if tag_end != -1:
                            tag_name = cleaned_code[i+1:tag_end].strip().replace('/', '')
                            if tag_name and not tag_name.endswith('/'):
                                stack.append(tag_name)
                i += 1
            
            return len(stack) == 0
        except Exception:
            return False
    
    def _evaluate_component_structure(self, code: str) -> float:
        """Evaluate component structure quality"""
        score = 0.0
        
        # Check for return statement
        if 'return' in code:
            score += 0.3
        
        # Check for JSX elements
        if '<' in code and '>' in code:
            score += 0.3
        
        # Check for export
        if 'export' in code:
            score += 0.2
        
        # Check for proper function structure
        if 'function' in code or 'const' in code:
            score += 0.2
        
        return score
    
    def _evaluate_prop_usage(self, code: str) -> float:
        """Evaluate prop usage in component"""
        score = 0.0
        
        # Check for props parameter
        if 'props' in code:
            score += 0.5
        
        # Check for destructuring
        if '{' in code and '}' in code:
            score += 0.3
        
        # Check for prop usage
        if 'props.' in code or re.search(r'\b\w+\.\w+\b', code):
            score += 0.2
        
        return score
    
    def evaluate_xml_configuration(self) -> Dict:
        """Evaluate XML configuration file generation"""
        self.logger.info("Evaluating XML configuration capabilities...")
        
        xml_config = self.web_dev_config.get('xml_configuration', {})
        test_files = xml_config.get('test_files', [])
        
        results = []
        
        for file_type in test_files:
            self.logger.info(f"Generating XML configuration: {file_type}")
            
            # Create file-specific prompt
            prompt = f"Create a {file_type} configuration file with proper structure and common settings"
            generated_content = self.generate_code(prompt, 'xml')
            
            # Evaluate XML quality
            quality_score = self._evaluate_xml_content(generated_content, file_type)
            
            results.append({
                'file_type': file_type,
                'generated_content': generated_content[:500] + '...' if len(generated_content) > 500 else generated_content,
                'quality_score': quality_score,
                'xml_valid': self._validate_xml_syntax(generated_content),
                'structure_completeness': self._evaluate_xml_structure(generated_content, file_type)
            })
        
        # Calculate aggregate scores
        if results:
            avg_quality = np.mean([r['quality_score'] for r in results])
            avg_xml_validity = np.mean([r['xml_valid'] for r in results])
        else:
            avg_quality = avg_xml_validity = 0
        
        return {
            'category': 'XML Configuration',
            'individual_scores': results,
            'aggregate_scores': {
                'average_quality': avg_quality,
                'xml_validity_rate': avg_xml_validity
            },
            'overall_score': avg_quality
        }
    
    def _evaluate_xml_content(self, content: str, file_type: str) -> float:
        """Evaluate XML content quality"""
        score = 0.0
        
        # Length check
        if len(content) > 50:
            score += 0.1
        
        # File-type specific evaluation
        if file_type.endswith('.json'):
            # JSON configuration
            try:
                import json
                json.loads(content)
                score += 0.5  # Valid JSON
            except:
                score += 0.2  # Attempted JSON structure
        else:
            # XML configuration
            if content.count('<') == content.count('>'):
                score += 0.3  # Basic tag matching
            if content.count('{') <= content.count('}'):
                score += 0.2  # Proper bracing
        
        # Structure keywords
        common_keywords = ['config', 'setting', 'property', 'value']
        found_keywords = sum(1 for keyword in common_keywords if keyword in content.lower())
        score += (found_keywords / len(common_keywords)) * 0.3
        
        return min(score, 1.0)
    
    def _validate_xml_syntax(self, content: str) -> bool:
        """Basic XML syntax validation"""
        try:
            # Simple tag matching check
            import xml.etree.ElementTree as ET
            ET.fromstring(content)
            return True
        except Exception:
            # Fallback to bracket matching for JSON-like content
            try:
                import json
                json.loads(content)
                return True
            except Exception:
                return False
    
    def _evaluate_xml_structure(self, content: str, file_type: str) -> float:
        """Evaluate XML structure completeness"""
        score = 0.0
        
        if file_type.endswith('.json'):
            # JSON structure indicators
            if '{' in content and '}' in content:
                score += 0.3
            if '"' in content:
                score += 0.3
            if ':' in content:
                score += 0.2
            if file_type == 'package.json':
                if 'dependencies' in content or 'devDependencies' in content:
                    score += 0.2
        else:
            # XML structure indicators
            if '<' in content and '>' in content:
                score += 0.5
            if '<?xml' in content:
                score += 0.2
            if '</' in content:
                score += 0.3
        
        return score
    
    def evaluate_mdx_documentation(self) -> Dict:
        """Evaluate MDX documentation generation"""
        self.logger.info("Evaluating MDX documentation capabilities...")
        
        mdx_config = self.web_dev_config.get('mdx_documentation', {})
        components = mdx_config.get('components', [])
        
        results = []
        
        for component in components:
            self.logger.info(f"Generating MDX documentation: {component}")
            
            prompt = f"Create MDX documentation for {component} component with examples and usage instructions"
            generated_content = self.generate_code(prompt, 'markdown')
            
            # Evaluate MDX quality
            quality_score = self._evaluate_mdx_content(generated_content, component)
            
            results.append({
                'component': component,
                'generated_content': generated_content[:500] + '...' if len(generated_content) > 500 else generated_content,
                'quality_score': quality_score,
                'markdown_syntax': self._validate_markdown_syntax(generated_content),
                'content_completeness': self._evaluate_mdx_completeness(generated_content, component)
            })
        
        # Calculate aggregate scores
        if results:
            avg_quality = np.mean([r['quality_score'] for r in results])
        else:
            avg_quality = 0
        
        return {
            'category': 'MDX Documentation',
            'individual_scores': results,
            'aggregate_scores': {
                'average_quality': avg_quality
            },
            'overall_score': avg_quality
        }
    
    def _evaluate_mdx_content(self, content: str, component: str) -> float:
        """Evaluate MDX content quality"""
        score = 0.0
        
        # Basic markdown syntax
        if content.count('#') > 0:
            score += 0.2
        if '```' in content:  # Code blocks
            score += 0.2
        if '[' in content and ']' in content:  # Links
            score += 0.2
        
        # Content structure
        lines = content.split('\n')
        if len(lines) > 3:
            score += 0.1
        
        # Component-specific keywords
        if component.lower() in content.lower():
            score += 0.3
        
        return min(score, 1.0)
    
    def _validate_markdown_syntax(self, content: str) -> bool:
        """Basic markdown syntax validation"""
        # Check for balanced brackets in links
        open_brackets = content.count('[')
        close_brackets = content.count(']')
        
        if open_brackets != close_brackets:
            return False
        
        # Check for balanced backticks in code blocks
        open_backticks = content.count('```')
        if open_backticks % 2 != 0:
            return False
        
        return True
    
    def _evaluate_mdx_completeness(self, content: str, component: str) -> float:
        """Evaluate MDX content completeness"""
        score = 0.0
        
        # Check for common documentation elements
        elements = ['example', 'usage', 'props', 'description', 'parameter']
        found_elements = sum(1 for element in elements if element in content.lower())
        score += (found_elements / len(elements)) * 0.6
        
        # Check for headings (documentation structure)
        if '#' in content:
            score += 0.2
        
        # Check for component mentions
        if component.lower() in content.lower():
            score += 0.2
        
        return score
    
    def evaluate_css_styling(self) -> Dict:
        """Evaluate CSS styling generation capabilities"""
        self.logger.info("Evaluating CSS styling capabilities...")
        
        css_config = self.web_dev_config.get('css_styling', {})
        categories = css_config.get('test_categories', [])
        
        results = []
        
        for category in categories:
            self.logger.info(f"Generating CSS for category: {category}")
            
            prompt = f"Create CSS styles for {category} with modern best practices"
            generated_css = self.generate_code(prompt, 'css')
            
            # Evaluate CSS quality
            quality_score = self._evaluate_css_quality(generated_css, category)
            
            results.append({
                'category': category,
                'generated_css': generated_css[:500] + '...' if len(generated_css) > 500 else generated_css,
                'quality_score': quality_score,
                'css_valid': self._validate_css_syntax(generated_css),
                'modern_features': self._evaluate_css_modern_features(generated_css, category)
            })
        
        # Calculate aggregate scores
        if results:
            avg_quality = np.mean([r['quality_score'] for r in results])
        else:
            avg_quality = 0
        
        return {
            'category': 'CSS Styling',
            'individual_scores': results,
            'aggregate_scores': {
                'average_quality': avg_quality
            },
            'overall_score': avg_quality
        }
    
    def _evaluate_css_quality(self, css: str, category: str) -> float:
        """Evaluate CSS code quality"""
        score = 0.0
        
        # Length check
        if len(css) > 30:
            score += 0.1
        
        # Category-specific evaluation
        if category == 'responsive_design':
            responsive_keywords = ['@media', 'responsive', 'flex', 'grid', 'viewport']
            found_keywords = sum(1 for keyword in responsive_keywords if keyword in css.lower())
            score += (found_keywords / len(responsive_keywords)) * 0.4
        
        elif category == 'animations':
            animation_keywords = ['@keyframes', 'animation', 'transform', 'transition']
            found_keywords = sum(1 for keyword in animation_keywords if keyword in css.lower())
            score += (found_keywords / len(animation_keywords)) * 0.4
        
        elif category == 'layout_systems':
            layout_keywords = ['flex', 'grid', 'display', 'position', 'float']
            found_keywords = sum(1 for keyword in layout_keywords if keyword in css.lower())
            score += (found_keywords / len(layout_keywords)) * 0.4
        
        # General CSS features
        css_features = ['{', '}', ';', ':', '.class', '#id']
        found_features = sum(1 for feature in css_features if feature in css)
        score += (found_features / len(css_features)) * 0.2
        
        # Modern CSS features
        modern_features = ['var(', 'calc(', 'clamp(', 'min(', 'max(']
        found_modern = sum(1 for feature in modern_features if feature in css)
        score += (found_modern / len(modern_features)) * 0.2
        
        return min(score, 1.0)
    
    def _validate_css_syntax(self, css: str) -> bool:
        """Basic CSS syntax validation"""
        # Check for balanced braces
        if css.count('{') != css.count('}'):
            return False
        
        # Check for proper property declaration structure
        if ':' in css and ';' in css:
            # Basic structure check
            properties = css.split(';')
            valid_properties = 0
            for prop in properties:
                if ':' in prop and prop.strip():
                    valid_properties += 1
            
            return valid_properties > 0
        
        return False
    
    def _evaluate_css_modern_features(self, css: str, category: str) -> float:
        """Evaluate use of modern CSS features"""
        score = 0.0
        
        # Modern CSS features
        modern_features = {
            'flex': 'display: flex' in css,
            'grid': 'display: grid' in css,
            'var': 'var(' in css,
            'calc': 'calc(' in css,
            'media': '@media' in css
        }
        
        found_features = sum(1 for found in modern_features.values() if found)
        score += (found_features / len(modern_features)) * 0.7
        
        # Category-specific modern features
        if category == 'responsive_design':
            if '@media' in css:
                score += 0.3
        elif category == 'animations':
            if '@keyframes' in css or 'animation:' in css:
                score += 0.3
        
        return min(score, 1.0)
    
    def run_evaluation(self) -> Dict:
        """Run complete web development evaluation"""
        start_time = time.time()
        
        if not self.load_model():
            return {'status': 'failed', 'error': 'Model loading failed'}
        
        try:
            self.logger.info("Starting web development evaluation suite...")
            
            # Run all web development tests
            evaluation_results = {}
            
            # JavaScript/TypeScript evaluation
            try:
                evaluation_results['javascript_typescript'] = self.evaluate_javascript_typescript()
            except Exception as e:
                self.logger.error(f"JavaScript/TypeScript evaluation failed: {e}")
                evaluation_results['javascript_typescript'] = {'error': str(e), 'overall_score': 0}
            
            # React components evaluation
            try:
                evaluation_results['react_components'] = self.evaluate_react_components()
            except Exception as e:
                self.logger.error(f"React components evaluation failed: {e}")
                evaluation_results['react_components'] = {'error': str(e), 'overall_score': 0}
            
            # XML configuration evaluation
            try:
                evaluation_results['xml_configuration'] = self.evaluate_xml_configuration()
            except Exception as e:
                self.logger.error(f"XML configuration evaluation failed: {e}")
                evaluation_results['xml_configuration'] = {'error': str(e), 'overall_score': 0}
            
            # MDX documentation evaluation
            try:
                evaluation_results['mdx_documentation'] = self.evaluate_mdx_documentation()
            except Exception as e:
                self.logger.error(f"MDX documentation evaluation failed: {e}")
                evaluation_results['mdx_documentation'] = {'error': str(e), 'overall_score': 0}
            
            # CSS styling evaluation
            try:
                evaluation_results['css_styling'] = self.evaluate_css_styling()
            except Exception as e:
                self.logger.error(f"CSS styling evaluation failed: {e}")
                evaluation_results['css_styling'] = {'error': str(e), 'overall_score': 0}
            
            # Calculate overall web development score
            scores = []
            for category, results in evaluation_results.items():
                if 'overall_score' in results:
                    scores.append(results['overall_score'])
            
            overall_score = np.mean(scores) if scores else 0
            
            evaluation_time = time.time() - start_time
            
            # Compile final results
            final_results = {
                'status': 'completed',
                'benchmark': 'Web Development',
                'overall_quality_score': overall_score,
                'target_quality': self.target_quality,
                'target_met': overall_score >= self.target_quality,
                'evaluation_time_seconds': evaluation_time,
                'category_results': evaluation_results,
                'javascript_quality': evaluation_results.get('javascript_typescript', {}).get('overall_score', 0),
                'react_score': evaluation_results.get('react_components', {}).get('overall_score', 0),
                'xml_score': evaluation_results.get('xml_configuration', {}).get('overall_score', 0),
                'css_quality': evaluation_results.get('css_styling', {}).get('overall_score', 0),
                'category_summary': self._summarize_categories(evaluation_results)
            }
            
            self.logger.info(f"Web Development Evaluation completed: {overall_score:.3f} overall score")
            
            # Save results
            self._save_results(final_results)
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'evaluation_time_seconds': time.time() - start_time
            }
    
    def _summarize_categories(self, results: Dict) -> Dict:
        """Summarize evaluation by category"""
        summary = {}
        
        for category, category_results in results.items():
            if 'overall_score' in category_results:
                summary[category] = {
                    'score': category_results['overall_score'],
                    'status': 'completed' if 'error' not in category_results else 'failed'
                }
            elif 'error' in category_results:
                summary[category] = {'score': 0, 'status': 'failed', 'error': category_results['error']}
        
        return summary
    
    def _save_results(self, results: Dict):
        """Save evaluation results"""
        # Save detailed results as JSON
        results_file = self.output_path / f"webdev_results_{self.run_id}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary as CSV for easy analysis
        import pandas as pd
        category_data = []
        for category, data in results.get('category_results', {}).items():
            category_data.append({
                'category': category,
                'overall_score': data.get('overall_score', 0),
                'status': 'completed' if 'error' not in data else 'failed'
            })
        
        if category_data:
            df = pd.DataFrame(category_data)
            csv_file = self.output_path / f"webdev_summary_{self.run_id}.csv"
            df.to_csv(csv_file, index=False)
        
        self.logger.info(f"Results saved to {self.output_path}")


def main():
    """Main web development evaluation function"""
    parser = argparse.ArgumentParser(description='Web Development Evaluation')
    
    parser.add_argument('--model_path', required=True, help='Path to model directory')
    parser.add_argument('--config', required=True, help='Path to evaluation configuration')
    parser.add_argument('--output_path', required=True, help='Output directory for results')
    parser.add_argument('--run_id', required=True, help='Unique run identifier')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = WebDevEvaluator(
        config_path=args.config,
        model_path=args.model_path,
        output_path=args.output_path,
        run_id=args.run_id
    )
    
    # Run evaluation
    try:
        results = evaluator.run_evaluation()
        
        if results.get('status') == 'completed':
            print(f"\nWeb Development Evaluation Results:")
            print(f"Overall Quality Score: {results.get('overall_quality_score', 0):.3f}")
            print(f"Target Quality: {results.get('target_quality', 0):.3f}")
            print(f"Target Met: {'✅' if results.get('target_met') else '❌'}")
            
            print(f"\nCategory Scores:")
            for category, data in results.get('category_summary', {}).items():
                status_icon = '✅' if data.get('status') == 'completed' else '❌'
                print(f"  {status_icon} {category.replace('_', ' ').title()}: {data.get('score', 0):.3f}")
            
            print(f"\nEvaluation Time: {results.get('evaluation_time_seconds', 0):.1f}s")
            return 0
        else:
            print(f"Evaluation failed: {results.get('error', 'Unknown error')}")
            return 1
            
    except KeyboardInterrupt:
        print("Evaluation interrupted by user")
        return 1
    except Exception as e:
        print(f"Evaluation failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())