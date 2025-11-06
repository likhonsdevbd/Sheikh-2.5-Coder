#!/usr/bin/env python3
"""
HumanEval Benchmark Evaluation for Sheikh-2.5-Coder
Evaluates model performance on OpenAI HumanEval coding benchmark with Pass@1 and Pass@k metrics
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add parent directories to path
sys.path.append('../')
sys.path.append('../../')

class HumanEvalEvaluator:
    """HumanEval coding benchmark evaluator"""
    
    def __init__(self, config_path: str, model_path: str, output_path: str, run_id: str):
        """Initialize HumanEval evaluator"""
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
        
        # Evaluation settings
        self.humaneval_config = self.config.get('humaneval_evaluation', {})
        self.target_pass1 = self.config.get('targets', {}).get('humaneval_pass1', 0.40)
        
        self.logger.info(f"HumanEval Evaluator initialized for run: {run_id}")
    
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
            'humaneval_evaluation': {
                'dataset': 'openai_humaneval',
                'max_samples': 100,
                'timeout_seconds': 60,
                'pass_at_k': [1, 10]
            },
            'targets': {
                'humaneval_pass1': 0.40
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for this evaluator"""
        log_file = self.output_path / f"humaneval_{self.run_id}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(f'HumanEvalEvaluator_{self.run_id}')
    
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
    
    def load_humaneval_dataset(self) -> List[Dict]:
        """Load HumanEval dataset"""
        try:
            self.logger.info("Loading HumanEval dataset...")
            
            dataset_name = self.humaneval_config.get('dataset', 'openai_humaneval')
            
            # Load dataset
            dataset = load_dataset(dataset_name, trust_remote_code=True)
            test_data = dataset['test']
            
            # Limit samples for computational efficiency
            max_samples = self.humaneval_config.get('max_samples', 100)
            if len(test_data) > max_samples:
                test_data = test_data.shuffle(seed=42).select(range(max_samples))
            
            self.logger.info(f"Loaded {len(test_data)} HumanEval test samples")
            return test_data
            
        except Exception as e:
            self.logger.error(f"Failed to load HumanEval dataset: {str(e)}")
            raise
    
    def generate_completions(self, prompt: str, entry_point: str, k: int = 10) -> List[str]:
        """Generate multiple completions for a coding prompt"""
        completions = []
        
        try:
            # Temperature settings for diverse sampling
            temperatures = [0.2, 0.5, 0.8, 1.0][:min(k, 4)]
            
            for i in range(k):
                # Use different temperatures for diversity
                temp = temperatures[i % len(temperatures)]
                
                # Tokenize input
                inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
                
                # Generate completion
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=temp,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
                # Extract generated code
                generated_code = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                ).strip()
                
                completions.append(generated_code)
                
        except Exception as e:
            self.logger.error(f"Generation failed: {str(e)}")
            return []
        
        return completions
    
    def extract_function_from_completion(self, completion: str, entry_point: str) -> Optional[str]:
        """Extract function definition from generated completion"""
        try:
            # Look for function definition patterns
            patterns = [
                rf'def\s+{re.escape(entry_point)}\s*\([^)]*\)\s*:',
                rf'function\s+{re.escape(entry_point)}\s*\([^)]*\)\s*\{{',
                rf'{re.escape(entry_point)}\s*=\s*function\s*\([^)]*\)\s*\{{',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, completion, re.IGNORECASE | re.MULTILINE)
                if match:
                    # Extract function from match onwards
                    start_pos = match.start()
                    function_code = completion[start_pos:]
                    
                    # Find the end of the function (basic heuristic)
                    if 'def ' in pattern:
                        # Python function - find matching indentation
                        lines = function_code.split('\n')
                        if lines:
                            first_line = lines[0]
                            base_indent = len(first_line) - len(first_line.lstrip())
                            
                            end_lines = []
                            for line in lines[1:]:
                                if line.strip():  # Non-empty line
                                    current_indent = len(line) - len(line.lstrip())
                                    if current_indent <= base_indent and not line.strip().startswith('def '):
                                        break
                                end_lines.append(line)
                            
                            extracted_function = first_line + '\n' + '\n'.join(end_lines)
                            return extracted_function
                    else:
                        # JavaScript function - basic extraction
                        brace_count = 0
                        extracted_chars = []
                        
                        for char in function_code:
                            extracted_chars.append(char)
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    break
                        
                        return ''.join(extracted_chars)
            
            # Fallback: return entire completion if no clear function found
            return completion
            
        except Exception as e:
            self.logger.error(f"Function extraction failed: {str(e)}")
            return completion
    
    def validate_syntax(self, code: str) -> bool:
        """Validate syntax of generated code"""
        try:
            # Try to parse as Python
            ast.parse(code)
            return True
        except SyntaxError:
            pass
        
        # Basic JavaScript syntax check
        try:
            # Simple bracket/brace matching check
            stack = []
            for char in code:
                if char in '({[':
                    stack.append(char)
                elif char in ')}]':
                    if not stack or not self._matching_bracket(stack.pop(), char):
                        return False
            return len(stack) == 0
        except Exception:
            return False
    
    def _matching_bracket(self, open_bracket: str, close_bracket: str) -> bool:
        """Check if brackets match"""
        pairs = {'(': ')', '[': ']', '{': '}'}
        return pairs.get(open_bracket) == close_bracket
    
    def test_function(self, function_code: str, entry_point: str, test_cases: List[str]) -> bool:
        """Test generated function against test cases"""
        try:
            # Create a namespace for testing
            namespace = {}
            
            # Execute the function code
            exec(function_code, namespace)
            
            # Check if function exists in namespace
            if entry_point not in namespace:
                return False
            
            func = namespace[entry_point]
            
            # Run test cases
            passed_tests = 0
            total_tests = len(test_cases)
            
            for test_case in test_cases:
                try:
                    # Execute test case
                    result = eval(test_case, namespace)
                    if result:
                        passed_tests += 1
                except Exception as e:
                    self.logger.debug(f"Test case failed: {test_case}, Error: {e}")
            
            # Consider function correct if all tests pass
            return passed_tests == total_tests and total_tests > 0
            
        except Exception as e:
            self.logger.debug(f"Function testing failed: {e}")
            return False
    
    def calculate_pass_at_k(self, results: List[Dict], k: int = 1) -> float:
        """Calculate Pass@k score"""
        if not results:
            return 0.0
        
        # Count problems where at least one completion passed tests
        problems_passed = 0
        
        for problem_results in results:
            # Get all completions for this problem
            completions = problem_results.get('completions', [])
            
            # Count how many completions passed tests
            passed_completions = sum(1 for comp in completions if comp.get('tests_passed', False))
            
            # Problem passes if at least k completions pass
            if passed_completions >= k:
                problems_passed += 1
        
        # Calculate Pass@k
        total_problems = len(results)
        pass_at_k = problems_passed / total_problems if total_problems > 0 else 0.0
        
        return pass_at_k
    
    def evaluate_problem(self, problem: Dict) -> Dict:
        """Evaluate a single HumanEval problem"""
        try:
            prompt = problem['prompt']
            entry_point = problem['entry_point']
            test_cases = problem.get('test', [])
            
            # Generate multiple completions for Pass@k calculation
            pass_at_k_values = self.humaneval_config.get('pass_at_k', [1, 10])
            max_completions = max(pass_at_k_values)
            
            # Generate completions
            completions = self.generate_completions(prompt, entry_point, max_completions)
            
            # Evaluate each completion
            evaluated_completions = []
            for i, completion in enumerate(completions):
                # Extract function
                function_code = self.extract_function_from_completion(completion, entry_point)
                
                # Validate syntax
                syntax_valid = self.validate_syntax(function_code)
                
                # Test function if syntax is valid and test cases exist
                tests_passed = False
                if syntax_valid and test_cases:
                    tests_passed = self.test_function(function_code, entry_point, test_cases)
                
                evaluated_completions.append({
                    'completion_index': i,
                    'generated_code': completion[:500] + '...' if len(completion) > 500 else completion,
                    'extracted_function': function_code[:500] + '...' if len(function_code) > 500 else function_code,
                    'syntax_valid': syntax_valid,
                    'tests_passed': tests_passed
                })
            
            # Determine if problem is solved (at least one completion passed)
            solved = any(comp.get('tests_passed', False) for comp in evaluated_completions)
            
            return {
                'problem_id': problem.get('task_id', ''),
                'entry_point': entry_point,
                'prompt': prompt[:200] + '...' if len(prompt) > 200 else prompt,
                'solved': solved,
                'total_completions': len(completions),
                'syntax_valid_count': sum(1 for comp in evaluated_completions if comp.get('syntax_valid', False)),
                'tests_passed_count': sum(1 for comp in evaluated_completions if comp.get('tests_passed', False)),
                'completions': evaluated_completions[:5]  # Store first 5 for analysis
            }
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate problem: {str(e)}")
            return {
                'problem_id': problem.get('task_id', ''),
                'entry_point': problem.get('entry_point', ''),
                'error': str(e),
                'solved': False
            }
    
    def run_evaluation(self) -> Dict:
        """Run complete HumanEval evaluation"""
        start_time = time.time()
        
        if not self.load_model():
            return {'status': 'failed', 'error': 'Model loading failed'}
        
        try:
            # Load dataset
            test_problems = self.load_humaneval_dataset()
            
            # Evaluate all problems
            problem_results = []
            total_problems = len(test_problems)
            
            self.logger.info(f"Evaluating {total_problems} HumanEval problems...")
            
            for i, problem in enumerate(test_problems):
                if i % 5 == 0:
                    self.logger.info(f"Progress: {i}/{total_problems} ({i/total_problems*100:.1f}%)")
                
                problem_result = self.evaluate_problem(problem)
                problem_results.append(problem_result)
            
            # Calculate Pass@k scores
            pass_at_k_values = self.humaneval_config.get('pass_at_k', [1, 10])
            pass_at_k_scores = {}
            
            for k in pass_at_k_values:
                pass_at_k_scores[f'pass_at_{k}'] = self.calculate_pass_at_k(problem_results, k)
            
            evaluation_time = time.time() - start_time
            
            # Compile final results
            final_results = {
                'status': 'completed',
                'benchmark': 'HumanEval',
                'total_problems': total_problems,
                'solved_problems': sum(1 for r in problem_results if r.get('solved', False)),
                'pass_at_k_scores': pass_at_k_scores,
                'target_pass1': self.target_pass1,
                'target_met': pass_at_k_scores.get('pass_at_1', 0) >= self.target_pass1,
                'evaluation_time_seconds': evaluation_time,
                'problems_per_second': total_problems / evaluation_time,
                'detailed_results': problem_results,
                'syntax_validity_rate': np.mean([r.get('syntax_valid_count', 0) / max(r.get('total_completions', 1), 1) for r in problem_results]),
                'problem_difficulty_analysis': self._analyze_problem_difficulty(problem_results),
                'completion_quality_analysis': self._analyze_completion_quality(problem_results)
            }
            
            # Add primary metric for target checking
            final_results['pass_at_1'] = pass_at_k_scores.get('pass_at_1', 0)
            
            self.logger.info(f"HumanEval Evaluation completed:")
            for k, score in pass_at_k_scores.items():
                self.logger.info(f"  {k}: {score:.3f}")
            
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
    
    def _analyze_problem_difficulty(self, results: List[Dict]) -> Dict:
        """Analyze difficulty distribution of problems"""
        difficulty_analysis = {
            'easy': 0,    # Solved by >50% of completions
            'medium': 0,  # Solved by 10-50% of completions  
            'hard': 0     # Solved by <10% of completions
        }
        
        for result in results:
            if 'tests_passed_count' in result and 'total_completions' in result:
                pass_rate = result['tests_passed_count'] / max(result['total_completions'], 1)
                
                if pass_rate > 0.5:
                    difficulty_analysis['easy'] += 1
                elif pass_rate >= 0.1:
                    difficulty_analysis['medium'] += 1
                else:
                    difficulty_analysis['hard'] += 1
        
        return difficulty_analysis
    
    def _analyze_completion_quality(self, results: List[Dict]) -> Dict:
        """Analyze quality of generated completions"""
        total_completions = 0
        syntax_valid_completions = 0
        tests_passed_completions = 0
        
        for result in results:
            completions = result.get('completions', [])
            for completion in completions:
                total_completions += 1
                if completion.get('syntax_valid', False):
                    syntax_valid_completions += 1
                if completion.get('tests_passed', False):
                    tests_passed_completions += 1
        
        return {
            'total_completions': total_completions,
            'syntax_validity_rate': syntax_valid_completions / max(total_completions, 1),
            'test_pass_rate': tests_passed_completions / max(total_completions, 1),
            'generation_success_rate': syntax_valid_completions / max(total_completions, 1)
        }
    
    def _save_results(self, results: Dict):
        """Save evaluation results"""
        # Save detailed results as JSON
        results_file = self.output_path / f"humaneval_results_{self.run_id}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary as CSV
        if 'detailed_results' in results:
            import pandas as pd
            df = pd.DataFrame(results['detailed_results'])
            csv_file = self.output_path / f"humaneval_detailed_{self.run_id}.csv"
            df.to_csv(csv_file, index=False)
        
        self.logger.info(f"Results saved to {self.output_path}")


def main():
    """Main HumanEval evaluation function"""
    parser = argparse.ArgumentParser(description='HumanEval Benchmark Evaluation')
    
    parser.add_argument('--model_path', required=True, help='Path to model directory')
    parser.add_argument('--config', required=True, help='Path to evaluation configuration')
    parser.add_argument('--output_path', required=True, help='Output directory for results')
    parser.add_argument('--run_id', required=True, help='Unique run identifier')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = HumanEvalEvaluator(
        config_path=args.config,
        model_path=args.model_path,
        output_path=args.output_path,
        run_id=args.run_id
    )
    
    # Run evaluation
    try:
        results = evaluator.run_evaluation()
        
        if results.get('status') == 'completed':
            print(f"\nHumanEval Evaluation Results:")
            print(f"Total Problems: {results.get('total_problems', 0)}")
            print(f"Solved Problems: {results.get('solved_problems', 0)}")
            for k, score in results.get('pass_at_k_scores', {}).items():
                print(f"{k}: {score:.3f}")
            print(f"Target Pass@1: {results.get('target_pass1', 0):.3f}")
            print(f"Target Met: {'✅' if results.get('target_met') else '❌'}")
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