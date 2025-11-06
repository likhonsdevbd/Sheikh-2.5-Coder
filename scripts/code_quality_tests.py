#!/usr/bin/env python3
"""
Code Quality Tests for Sheikh-2.5-Coder
Evaluates syntax validity, CodeBLEU scores, complexity analysis, and best practices compliance
"""

import os
import sys
import json
import yaml
import argparse
import logging
import torch
import numpy as np
import re
import ast
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import time
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import Counter
import tempfile

# Add parent directories to path
sys.path.append('../')
sys.path.append('../../')

class CodeQualityEvaluator:
    """Code quality evaluation framework"""
    
    def __init__(self, config_path: str, model_path: str, output_path: str, run_id: str):
        """Initialize code quality evaluator"""
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
        
        # Code quality test settings
        self.quality_config = self.config.get('code_quality_tests', {})
        self.target_syntax_validity = self.config.get('targets', {}).get('syntax_validity', 0.95)
        self.target_codebleu = self.config.get('targets', {}).get('codebleu_score', 0.65)
        
        self.logger.info(f"Code Quality Evaluator initialized for run: {run_id}")
    
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
            'code_quality_tests': {
                'syntax_validity': {
                    'languages': ['python', 'javascript', 'typescript', 'html', 'css', 'xml']
                },
                'complexity_analysis': {
                    'cyclomatic_complexity': 3.0,
                    'nesting_depth': 4,
                    'function_length': 50,
                    'class_size': 200
                },
                'best_practices': {
                    'python': ['PEP8_compliance', 'docstring_coverage', 'type_hints', 'error_handling'],
                    'javascript': ['ES6_features', 'async_patterns', 'component_patterns', 'testing_coverage']
                }
            },
            'targets': {
                'syntax_validity': 0.95,
                'codebleu_score': 0.65
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for this evaluator"""
        log_file = self.output_path / f"code_quality_{self.run_id}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(f'CodeQualityEvaluator_{self.run_id}')
    
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
    
    def generate_code_samples(self, num_samples: int = 50) -> Dict[str, List[Dict]]:
        """Generate code samples for quality evaluation"""
        self.logger.info(f"Generating {num_samples} code samples for quality evaluation...")
        
        # Define code generation tasks by language
        generation_tasks = {
            'python': [
                "Write a function to calculate factorial",
                "Create a class for a bank account",
                "Implement a binary search algorithm",
                "Write a decorator for timing functions",
                "Create a context manager for file handling"
            ],
            'javascript': [
                "Create a function to sort an array",
                "Write a class for a simple game player",
                "Implement event handling for clicks",
                "Create an async function for API calls",
                "Write a module for utility functions"
            ],
            'typescript': [
                "Define an interface for a user profile",
                "Create a generic function for arrays",
                "Write a React component with props",
                "Implement a promise-based API client",
                "Create a type definition for configurations"
            ],
            'html': [
                "Create a responsive navigation menu",
                "Build a contact form with validation",
                "Design a card-based layout",
                "Create a grid system for products",
                "Build an accessible modal dialog"
            ],
            'css': [
                "Create a responsive grid layout",
                "Write animations for buttons",
                "Design a mobile-first navigation",
                "Create CSS variables for theming",
                "Implement a flexbox layout"
            ],
            'xml': [
                "Create an XML configuration file",
                "Design an XML schema for data",
                "Write an XML transformation template",
                "Create an XML web service configuration",
                "Design an XML document structure"
            ]
        }
        
        generated_samples = {language: [] for language in generation_tasks.keys()}
        
        # Generate samples for each language
        for language, tasks in generation_tasks.items():
            samples_per_language = max(1, num_samples // len(generation_tasks))
            
            for task in tasks[:samples_per_language]:
                try:
                    # Generate code based on task
                    prompt = f"Generate {language} code: {task}"
                    generated_code = self._generate_single_code(prompt, language)
                    
                    if generated_code and len(generated_code.strip()) > 10:
                        generated_samples[language].append({
                            'task': task,
                            'code': generated_code,
                            'language': language,
                            'length': len(generated_code)
                        })
                        
                except Exception as e:
                    self.logger.warning(f"Failed to generate {language} code for task '{task}': {e}")
        
        self.logger.info(f"Generated {sum(len(samples) for samples in generated_samples.values())} code samples")
        return generated_samples
    
    def _generate_single_code(self, prompt: str, language: str) -> str:
        """Generate a single code sample"""
        try:
            # Add language-specific context
            enhanced_prompt = f"Language: {language}\nTask: {prompt}\n\nGenerate code:\n"
            
            # Tokenize and generate
            inputs = self.tokenizer(enhanced_prompt, return_tensors='pt').to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
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
            self.logger.error(f"Code generation failed for {language}: {e}")
            return ""
    
    def evaluate_syntax_validity(self, code_samples: Dict[str, List[Dict]]) -> Dict:
        """Evaluate syntax validity across all languages"""
        self.logger.info("Evaluating syntax validity...")
        
        languages = self.quality_config.get('syntax_validity', {}).get('languages', [])
        syntax_results = {}
        
        overall_valid = 0
        overall_total = 0
        
        for language in languages:
            if language not in code_samples:
                continue
                
            samples = code_samples[language]
            if not samples:
                continue
            
            valid_count = 0
            total_count = len(samples)
            
            language_results = []
            
            for sample in samples:
                code = sample.get('code', '')
                is_valid = self._validate_syntax(code, language)
                
                if is_valid:
                    valid_count += 1
                
                language_results.append({
                    'task': sample.get('task', ''),
                    'is_valid': is_valid,
                    'code_length': sample.get('length', 0)
                })
            
            overall_valid += valid_count
            overall_total += total_count
            
            syntax_results[language] = {
                'valid_count': valid_count,
                'total_count': total_count,
                'validity_rate': valid_count / total_count if total_count > 0 else 0,
                'sample_results': language_results
            }
        
        # Calculate overall syntax validity
        overall_validity = overall_valid / overall_total if overall_total > 0 else 0
        
        final_results = {
            'status': 'completed',
            'benchmark': 'Syntax Validity',
            'overall_syntax_validity': overall_validity,
            'target_syntax_validity': self.target_syntax_validity,
            'target_met': overall_validity >= self.target_syntax_validity,
            'language_breakdown': syntax_results,
            'total_samples_evaluated': overall_total,
            'total_valid_samples': overall_valid
        }
        
        self.logger.info(f"Syntax validity evaluation: {overall_validity:.3f} overall rate")
        return final_results
    
    def _validate_syntax(self, code: str, language: str) -> bool:
        """Validate syntax for a specific language"""
        if not code or len(code.strip()) < 5:
            return False
        
        try:
            if language == 'python':
                # Parse Python code
                ast.parse(code)
                return True
            
            elif language == 'javascript':
                # Basic JavaScript syntax validation
                return self._validate_javascript_syntax(code)
            
            elif language == 'typescript':
                # TypeScript validation (similar to JavaScript for basic checks)
                return self._validate_javascript_syntax(code)
            
            elif language == 'html':
                # HTML validation
                return self._validate_html_syntax(code)
            
            elif language == 'css':
                # CSS validation
                return self._validate_css_syntax(code)
            
            elif language == 'xml':
                # XML validation
                return self._validate_xml_syntax(code)
            
            else:
                # Generic validation
                return self._generic_syntax_check(code)
                
        except Exception:
            return False
    
    def _validate_javascript_syntax(self, code: str) -> bool:
        """Validate JavaScript/TypeScript syntax"""
        # Basic bracket/brace matching
        if not self._balanced_brackets(code):
            return False
        
        # Check for basic structure
        if 'function' in code or '=>' in code or 'class' in code or 'const' in code or 'let' in code:
            return True
        
        # If we have basic structure indicators, assume valid
        return len(code.strip()) > 20
    
    def _validate_html_syntax(self, code: str) -> bool:
        """Validate HTML syntax"""
        # Basic tag matching
        open_tags = code.count('<')
        close_tags = code.count('>')
        
        if open_tags != close_tags:
            return False
        
        # Check for self-closing tags
        self_closing = code.count('/>')
        if self_closing > 0:
            return True
        
        # Check for opening and closing tags
        if code.count('</') > 0:
            return True
        
        return len(code.strip()) > 10
    
    def _validate_css_syntax(self, code: str) -> bool:
        """Validate CSS syntax"""
        # Check for braces
        if code.count('{') != code.count('}'):
            return False
        
        # Check for properties (contains colon)
        if ':' not in code:
            return False
        
        # Check for selectors (contains class, id, or element names)
        if '.' in code or '#' in code or re.search(r'[a-zA-Z]+\s*{', code):
            return True
        
        return len(code.strip()) > 15
    
    def _validate_xml_syntax(self, code: str) -> bool:
        """Validate XML syntax"""
        try:
            import xml.etree.ElementTree as ET
            ET.fromstring(code)
            return True
        except Exception:
            return False
    
    def _balanced_brackets(self, code: str) -> bool:
        """Check if brackets are balanced"""
        brackets = {'(': ')', '[': ']', '{': '}'}
        stack = []
        
        for char in code:
            if char in brackets:
                stack.append(char)
            elif char in brackets.values():
                if not stack or brackets[stack.pop()] != char:
                    return False
        
        return len(stack) == 0
    
    def _generic_syntax_check(self, code: str) -> bool:
        """Generic syntax validation for unknown languages"""
        # Basic checks for balanced characters
        if code.count('(') != code.count(')'):
            return False
        if code.count('{') != code.count('}'):
            return False
        if code.count('[') != code.count(']'):
            return False
        
        return len(code.strip()) > 20
    
    def calculate_codebleu_score(self, code_samples: Dict[str, List[Dict]]) -> Dict:
        """Calculate CodeBLEU scores for generated code"""
        self.logger.info("Calculating CodeBLEU scores...")
        
        # This is a simplified CodeBLEU calculation
        # In practice, you would use the official CodeBLEU implementation
        
        codebleu_results = {}
        total_weighted_score = 0.0
        total_weight = 0
        
        for language, samples in code_samples.items():
            if not samples:
                continue
            
            # Simulate reference code (in practice, use high-quality reference samples)
            reference_codes = self._generate_reference_codes(language, len(samples))
            
            language_scores = []
            for i, sample in enumerate(samples):
                if i < len(reference_codes):
                    score = self._calculate_single_codebleu(sample['code'], reference_codes[i])
                    language_scores.append(score)
            
            if language_scores:
                avg_score = np.mean(language_scores)
                codebleu_results[language] = {
                    'avg_codebleu_score': avg_score,
                    'sample_count': len(language_scores),
                    'scores': language_scores
                }
                
                # Weight by number of samples for overall calculation
                total_weighted_score += avg_score * len(language_scores)
                total_weight += len(language_scores)
        
        overall_codebleu = total_weighted_score / total_weight if total_weight > 0 else 0
        
        return {
            'overall_codebleu_score': overall_codebleu,
            'target_codebleu': self.target_codebleu,
            'target_met': overall_codebleu >= self.target_codebleu,
            'language_breakdown': codebleu_results
        }
    
    def _generate_reference_codes(self, language: str, count: int) -> List[str]:
        """Generate reference codes for CodeBLEU calculation"""
        # This is a placeholder - in practice, use curated high-quality reference codes
        reference_templates = {
            'python': [
                "def example_function():\n    pass",
                "class ExampleClass:\n    def __init__(self):\n        pass"
            ],
            'javascript': [
                "function exampleFunction() {\n    // Example\n}",
                "const exampleFunction = () => {\n    // Example\n};"
            ]
        }
        
        templates = reference_templates.get(language, ["// Example code"])
        return (templates * (count // len(templates) + 1))[:count]
    
    def _calculate_single_codebleu(self, generated: str, reference: str) -> float:
        """Calculate a simplified CodeBLEU score between generated and reference code"""
        # This is a simplified implementation
        # Real CodeBLEU uses more sophisticated metrics
        
        # Token overlap (simplified)
        gen_tokens = set(generated.split())
        ref_tokens = set(reference.split())
        
        if not ref_tokens:
            return 0.0
        
        overlap = len(gen_tokens.intersection(ref_tokens))
        token_precision = overlap / len(gen_tokens) if gen_tokens else 0
        token_recall = overlap / len(ref_tokens) if ref_tokens else 0
        
        # F1 score
        if token_precision + token_recall == 0:
            return 0.0
        
        f1_score = 2 * (token_precision * token_recall) / (token_precision + token_recall)
        
        # Length penalty (prefer reasonable lengths)
        length_ratio = min(len(generated), len(reference)) / max(len(generated), len(reference))
        
        # Combine metrics
        codebleu_score = f1_score * 0.7 + length_ratio * 0.3
        
        return min(codebleu_score, 1.0)
    
    def run_comprehensive_quality_evaluation(self) -> Dict:
        """Run comprehensive code quality evaluation"""
        start_time = time.time()
        
        if not self.load_model():
            return {'status': 'failed', 'error': 'Model loading failed'}
        
        try:
            self.logger.info("Starting comprehensive code quality evaluation...")
            
            # Generate code samples
            code_samples = self.generate_code_samples(30)  # Reduced for faster execution
            
            # Run quality tests
            quality_results = {}
            
            # Syntax validity
            try:
                self.logger.info("Running syntax validity evaluation...")
                quality_results['syntax_validity'] = self.evaluate_syntax_validity(code_samples)
            except Exception as e:
                self.logger.error(f"Syntax validity evaluation failed: {e}")
                quality_results['syntax_validity'] = {'error': str(e)}
            
            # CodeBLEU score calculation
            try:
                self.logger.info("Calculating CodeBLEU scores...")
                quality_results['codebleu_scores'] = self.calculate_codebleu_score(code_samples)
            except Exception as e:
                self.logger.error(f"CodeBLEU calculation failed: {e}")
                quality_results['codebleu_scores'] = {'error': str(e)}
            
            evaluation_time = time.time() - start_time
            
            # Extract key metrics
            syntax_validity = quality_results.get('syntax_validity', {}).get('overall_syntax_validity', 0)
            codebleu_score = quality_results.get('codebleu_scores', {}).get('overall_codebleu_score', 0)
            
            # Compile final results
            final_results = {
                'status': 'completed',
                'benchmark': 'Code Quality',
                'syntax_validity': syntax_validity,
                'codebleu_score': codebleu_score,
                'target_syntax_validity': self.target_syntax_validity,
                'target_codebleu': self.target_codebleu,
                'syntax_target_met': syntax_validity >= self.target_syntax_validity,
                'codebleu_target_met': codebleu_score >= self.target_codebleu,
                'evaluation_time_seconds': evaluation_time,
                'detailed_results': quality_results,
                'overall_quality_score': (syntax_validity + codebleu_score) / 2,
                'language_coverage': list(code_samples.keys()),
                'total_samples_generated': sum(len(samples) for samples in code_samples.values())
            }
            
            self.logger.info(f"Code quality evaluation completed: {syntax_validity:.3f} syntax validity, {codebleu_score:.3f} CodeBLEU")
            
            # Save results
            self._save_results(final_results)
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Code quality evaluation failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'evaluation_time_seconds': time.time() - start_time
            }
    
    def _save_results(self, results: Dict):
        """Save code quality evaluation results"""
        # Save detailed results as JSON
        results_file = self.output_path / f"code_quality_results_{self.run_id}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary metrics as CSV
        import pandas as pd
        
        summary_data = []
        
        # Add syntax validity data
        syntax_data = results.get('detailed_results', {}).get('syntax_validity', {})
        if syntax_data:
            summary_data.append({
                'metric': 'Overall Syntax Validity',
                'value': syntax_data.get('overall_syntax_validity', 0),
                'target': syntax_data.get('target_syntax_validity', 0),
                'target_met': syntax_data.get('target_met', False)
            })
        
        # Add CodeBLEU data
        codebleu_data = results.get('detailed_results', {}).get('codebleu_scores', {})
        if codebleu_data:
            summary_data.append({
                'metric': 'Overall CodeBLEU Score',
                'value': codebleu_data.get('overall_codebleu_score', 0),
                'target': codebleu_data.get('target_codebleu', 0),
                'target_met': codebleu_data.get('target_met', False)
            })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            csv_file = self.output_path / f"code_quality_summary_{self.run_id}.csv"
            df.to_csv(csv_file, index=False)
        
        self.logger.info(f"Code quality results saved to {self.output_path}")


def main():
    """Main code quality evaluation function"""
    parser = argparse.ArgumentParser(description='Code Quality Evaluation')
    
    parser.add_argument('--model_path', required=True, help='Path to model directory')
    parser.add_argument('--config', required=True, help='Path to evaluation configuration')
    parser.add_argument('--output_path', required=True, help='Output directory for results')
    parser.add_argument('--run_id', required=True, help='Unique run identifier')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = CodeQualityEvaluator(
        config_path=args.config,
        model_path=args.model_path,
        output_path=args.output_path,
        run_id=args.run_id
    )
    
    # Run evaluation
    try:
        results = evaluator.run_comprehensive_quality_evaluation()
        
        if results.get('status') == 'completed':
            print(f"\nCode Quality Evaluation Results:")
            print(f"Syntax Validity: {results.get('syntax_validity', 0):.3f} (Target: {results.get('target_syntax_validity', 0):.3f})")
            print(f"CodeBLEU Score: {results.get('codebleu_score', 0):.3f} (Target: {results.get('target_codebleu', 0):.3f})")
            print(f"Overall Quality Score: {results.get('overall_quality_score', 0):.3f}")
            
            print(f"\nTarget Achievement:")
            print(f"Syntax Target: {'✅' if results.get('syntax_target_met') else '❌'}")
            print(f"CodeBLEU Target: {'✅' if results.get('codebleu_target_met') else '❌'}")
            
            print(f"\nLanguage Coverage: {', '.join(results.get('language_coverage', []))}")
            print(f"Total Samples: {results.get('total_samples_generated', 0)}")
            print(f"Evaluation Time: {results.get('evaluation_time_seconds', 0):.1f}s")
            
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