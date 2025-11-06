#!/usr/bin/env python3
"""
Main Evaluation Orchestrator for Sheikh-2.5-Coder
Coordinates comprehensive evaluation across multiple benchmarks and testing procedures
"""

import os
import sys
import json
import argparse
import logging
import yaml
import torch
import numpy as np
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import time
import psutil
import traceback

# Add src to path
sys.path.append('src')
sys.path.append('../src')
sys.path.append('.')

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline
)
import evaluate
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Setup logging
def setup_logging(log_level: str, log_file: str):
    """Setup comprehensive logging configuration"""
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


class EvaluationOrchestrator:
    """Main orchestrator for comprehensive model evaluation"""
    
    def __init__(self, config_path: str, model_path: str, run_id: str, output_path: str):
        """Initialize the evaluation orchestrator"""
        self.config_path = config_path
        self.model_path = Path(model_path)
        self.run_id = run_id
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config()
        
        # Setup logging
        log_level = self.config.get('logging', {}).get('level', 'INFO')
        log_file = self.config.get('logging', {}).get('log_file', f'logs/evaluation_{run_id}.log')
        self.logger = setup_logging(log_level, log_file)
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
        # Results storage
        self.results = {}
        self.metrics = {}
        
        # Hardware monitoring
        self.monitor = HardwareMonitor()
        
        # Target metrics from config
        self.targets = self.config.get('targets', {})
        
        self.logger.info(f"Evaluation Orchestrator initialized for run: {run_id}")
        
    def _load_config(self) -> Dict:
        """Load evaluation configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {str(e)}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Return default configuration if loading fails"""
        return {
            'evaluation': {
                'model_settings': {
                    'device': 'auto',
                    'dtype': 'float16',
                    'max_new_tokens': 512
                }
            },
            'targets': {
                'mmlu_code_accuracy': 0.60,
                'humaneval_pass1': 0.40,
                'codebleu_score': 0.65,
                'syntax_validity': 0.95
            }
        }


    def load_model(self) -> bool:
        """Load model and tokenizer for evaluation"""
        self.logger.info(f"Loading model from {self.model_path}")
        
        try:
            model_config = self.config.get('evaluation', {}).get('model_settings', {})
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=model_config.get('trust_remote_code', True),
                padding_side="right"
            )
            
            # Set pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            model_dtype = getattr(torch, model_config.get('dtype', 'float16'))
            device_map = model_config.get('device', 'auto')
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=model_dtype,
                device_map=device_map,
                trust_remote_code=model_config.get('trust_remote_code', True)
            )
            
            # Create pipeline
            pipeline_config = model_config.copy()
            pipeline_config.update({
                'model': self.model,
                'tokenizer': self.tokenizer,
                'device_map': device_map
            })
            
            self.pipeline = pipeline(
                'text-generation',
                **pipeline_config
            )
            
            self.logger.info(f"Model loaded successfully on {device_map}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
    
    def run_comprehensive_evaluation(self) -> bool:
        """Run comprehensive evaluation across all benchmarks"""
        if not self.load_model():
            return False
            
        self.logger.info("Starting comprehensive evaluation suite...")
        
        # Get benchmark scripts
        benchmark_scripts = self._get_benchmark_scripts()
        
        # Run each benchmark
        evaluation_order = [
            'mmlu_evaluation',
            'humaneval_evaluation', 
            'web_dev_tests',
            'performance_benchmark',
            'code_quality_tests',
            'regression_testing'
        ]
        
        for benchmark in evaluation_order:
            if benchmark in benchmark_scripts:
                try:
                    self.logger.info(f"Running {benchmark}...")
                    result = self._run_benchmark_script(benchmark_scripts[benchmark], benchmark)
                    if result:
                        self.results[benchmark] = result
                        self.logger.info(f"{benchmark} completed successfully")
                    else:
                        self.logger.warning(f"{benchmark} completed with warnings")
                        
                except Exception as e:
                    self.logger.error(f"{benchmark} failed: {str(e)}")
                    self.results[benchmark] = {'status': 'failed', 'error': str(e)}
        
        # Generate comprehensive report
        self._generate_comprehensive_report()
        
        # Save all results
        self._save_all_results()
        
        # Performance summary
        self._print_performance_summary()
        
        return True
    
    def _get_benchmark_scripts(self) -> Dict[str, str]:
        """Get paths to benchmark evaluation scripts"""
        scripts_dir = Path(__file__).parent
        return {
            'mmlu_evaluation': str(scripts_dir / 'mmlu_evaluation.py'),
            'humaneval_evaluation': str(scripts_dir / 'humaneval_evaluation.py'),
            'web_dev_tests': str(scripts_dir / 'web_dev_tests.py'),
            'performance_benchmark': str(scripts_dir / 'performance_benchmark.py'),
            'code_quality_tests': str(scripts_dir / 'code_quality_tests.py'),
            'regression_testing': str(scripts_dir / 'regression_testing.py')
        }
    
    def _run_benchmark_script(self, script_path: str, benchmark_name: str) -> Optional[Dict]:
        """Execute a benchmark evaluation script"""
        try:
            cmd = [
                sys.executable,
                script_path,
                '--model_path', str(self.model_path),
                '--output_path', str(self.output_path / benchmark_name),
                '--run_id', f"{self.run_id}_{benchmark_name}",
                '--config', self.config_path
            ]
            
            self.logger.info(f"Executing: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                # Try to load results from expected output location
                results_file = self.output_path / benchmark_name / f"results_{self.run_id}_{benchmark_name}.json"
                if results_file.exists():
                    with open(results_file, 'r') as f:
                        return json.load(f)
                else:
                    return {'status': 'completed', 'output': result.stdout}
            else:
                self.logger.error(f"Script failed: {result.stderr}")
                return {'status': 'failed', 'error': result.stderr, 'stdout': result.stdout}
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"Benchmark {benchmark_name} timed out")
            return {'status': 'timeout'}
        except Exception as e:
            self.logger.error(f"Failed to run benchmark {benchmark_name}: {str(e)}")
            return {'status': 'failed', 'error': str(e)}
    
    def load_model(self):
        """Load model and tokenizer for evaluation"""
        logger.info(f"Loading model from {self.model_path}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                padding_side="right"
            )
            
            # Set pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map='auto',
                trust_remote_code=True
            )
            
            # Create pipeline for easier inference
            self.pipeline = pipeline(
                'text-generation',
                model=self.model,
                tokenizer=self.tokenizer,
                device_map='auto',
                do_sample=True,
                temperature=0.7,
                max_new_tokens=256
            )
            
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False
    
    def evaluate_mmlu_code(self) -> Dict:
        """Evaluate on MMLU Code benchmark"""
        logger.info("Evaluating on MMLU Code benchmark...")
        
        try:
            # Load MMLU dataset
            dataset = load_dataset('lukaemon/mmlu', 'code', trust_remote_code=True)
            test_data = dataset['test']
            
            # Evaluate subset due to computational constraints
            test_subset = test_data.shuffle(seed=42).select(range(100))
            
            correct = 0
            total = 0
            
            for item in test_subset:
                question = item['question']
                choices = item['choices']
                answer_idx = item['answer']
                
                # Format question with choices
                prompt = f"{question}\n"
                for i, choice in enumerate(choices):
                    prompt += f"({chr(65+i)}) {choice}\n"
                prompt += "\nAnswer:"
                
                # Generate answer
                inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                # Extract answer
                generated_text = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                ).strip()
                
                # Check if answer is correct
                if len(generated_text) > 0 and generated_text[0].upper() == chr(65 + answer_idx):
                    correct += 1
                
                total += 1
                
                if total % 10 == 0:
                    logger.info(f"MMLU Code: {correct}/{total} correct ({correct/total*100:.1f}%)")
            
            accuracy = correct / total
            
            result = {
                'benchmark': 'MMLU Code',
                'accuracy': accuracy,
                'correct': correct,
                'total': total,
                'evaluated_items': len(test_subset)
            }
            
            logger.info(f"MMLU Code Accuracy: {accuracy:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"MMLU Code evaluation failed: {str(e)}")
            return {'benchmark': 'MMLU Code', 'error': str(e)}
    
    def evaluate_humaneval(self) -> Dict:
        """Evaluate on HumanEval coding benchmark"""
        logger.info("Evaluating on HumanEval benchmark...")
        
        try:
            # Load HumanEval dataset
            dataset = load_dataset('openai_humaneval', trust_remote_code=True)
            test_data = dataset['test']
            
            correct = 0
            total = len(test_data)
            
            for item in test_data:
                prompt = item['prompt']
                entry_point = item['entry_point']
                
                # Generate completion
                inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                # Extract generated function
                generated_code = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                ).strip()
                
                # Simple check - look for function definition
                if f"def {entry_point}" in generated_code or "def " in generated_code:
                    correct += 1
                
                if total % 5 == 0:
                    logger.info(f"HumanEval: {correct}/{total} functions generated")
            
            success_rate = correct / total
            
            result = {
                'benchmark': 'HumanEval',
                'success_rate': success_rate,
                'correct': correct,
                'total': total,
                'completion_quality': 'basic_function_detected'
            }
            
            logger.info(f"HumanEval Success Rate: {success_rate:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"HumanEval evaluation failed: {str(e)}")
            return {'benchmark': 'HumanEval', 'error': str(e)}
    
    def evaluate_web_development(self) -> Dict:
        """Evaluate on web development specific tasks"""
        logger.info("Evaluating web development capabilities...")
        
        # Define web development tasks
        web_tasks = [
            {
                'task': 'Create a responsive navbar',
                'instruction': 'Create a responsive navigation bar with hamburger menu',
                'context': 'HTML/CSS/JavaScript responsive navbar'
            },
            {
                'task': 'Form validation',
                'instruction': 'Create a form with client-side validation',
                'context': 'JavaScript form validation with error handling'
            },
            {
                'task': 'CSS Grid layout',
                'instruction': 'Create a CSS Grid layout with responsive design',
                'context': 'Modern CSS Grid layout with media queries'
            },
            {
                'task': 'DOM manipulation',
                'instruction': 'Create dynamic content using JavaScript DOM API',
                'context': 'JavaScript DOM manipulation and event handling'
            },
            {
                'task': 'API integration',
                'instruction': 'Fetch data from REST API and display it',
                'context': 'JavaScript fetch API with async/await'
            }
        ]
        
        task_results = []
        
        for task in web_tasks:
            try:
                # Format prompt
                prompt = f"Instruction: {task['instruction']}\nContext: {task['context']}\n\nGenerate code:"
                
                # Generate response
                inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=300,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                # Extract response
                generated_code = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                ).strip()
                
                # Evaluate response quality
                quality_score = self.evaluate_code_quality(generated_code, task['task'])
                
                task_results.append({
                    'task': task['task'],
                    'quality_score': quality_score,
                    'generated_code': generated_code[:200] + '...' if len(generated_code) > 200 else generated_code
                })
                
                logger.info(f"Web Dev Task - {task['task']}: Quality score {quality_score}")
                
            except Exception as e:
                logger.error(f"Web dev task failed: {task['task']}, Error: {str(e)}")
                task_results.append({
                    'task': task['task'],
                    'quality_score': 0,
                    'error': str(e)
                })
        
        # Calculate average quality score
        valid_scores = [r['quality_score'] for r in task_results if 'quality_score' in r]
        avg_quality = np.mean(valid_scores) if valid_scores else 0
        
        result = {
            'benchmark': 'Web Development',
            'average_quality_score': avg_quality,
            'total_tasks': len(web_tasks),
            'completed_tasks': len([r for r in task_results if 'error' not in r]),
            'task_results': task_results
        }
        
        logger.info(f"Web Development Average Quality: {avg_quality:.3f}")
        return result
    
    def evaluate_code_quality(self, code: str, task_type: str) -> float:
        """Evaluate quality of generated code"""
        score = 0.0
        
        # Basic checks
        if len(code) > 50:
            score += 0.2
        
        # Language-specific checks
        if task_type in ['responsive navbar', 'CSS Grid layout']:
            # Check for CSS keywords
            css_keywords = ['css', 'style', 'responsive', 'grid', 'flex']
            if any(keyword in code.lower() for keyword in css_keywords):
                score += 0.3
                
            # Check for HTML structure
            if '<' in code and '>' in code:
                score += 0.3
        
        elif task_type in ['Form validation', 'DOM manipulation', 'API integration']:
            # Check for JavaScript keywords
            js_keywords = ['function', 'const', 'let', 'var', 'async', 'await', 'fetch']
            if any(keyword in code.lower() for keyword in js_keywords):
                score += 0.3
                
            # Check for specific features
            if task_type == 'Form validation':
                validation_keywords = ['validate', 'required', 'pattern', 'addEventListener']
                if any(keyword in code.lower() for keyword in validation_keywords):
                    score += 0.3
            elif task_type == 'DOM manipulation':
                dom_keywords = ['getElementById', 'querySelector', 'appendChild', 'innerHTML']
                if any(keyword in code.lower() for keyword in dom_keywords):
                    score += 0.3
            elif task_type == 'API integration':
                api_keywords = ['fetch', 'async', 'await', 'json']
                if any(keyword in code.lower() for keyword in api_keywords):
                    score += 0.3
        
        # Syntax structure
        if code.count('{') == code.count('}'):
            score += 0.1
            
        if code.count('(') == code.count(')'):
            score += 0.1
        
        return min(score, 1.0)
    
    def evaluate_performance_benchmarks(self) -> Dict:
        """Evaluate model performance benchmarks"""
        logger.info("Running performance benchmarks...")
        
        try:
            # Model size
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            # Memory usage
            memory_info = {}
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    memory_info[f'gpu_{i}'] = {
                        'allocated': torch.cuda.memory_allocated(i) / 1024**3,
                        'reserved': torch.cuda.memory_reserved(i) / 1024**3
                    }
            
            # Inference speed test
            prompt = "def fibonacci(n):"
            inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
            
            # Warm up
            for _ in range(3):
                _ = self.model.generate(**inputs, max_new_tokens=10, do_sample=False)
            
            # Speed benchmark
            start_time = time.time()
            num_runs = 10
            for _ in range(num_runs):
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=50, 
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            end_time = time.time()
            
            avg_generation_time = (end_time - start_time) / num_runs
            tokens_per_second = 50 / avg_generation_time
            
            result = {
                'benchmark': 'Performance',
                'model_size': {
                    'total_parameters': total_params,
                    'trainable_parameters': trainable_params,
                    'trainable_percentage': (trainable_params / total_params) * 100
                },
                'memory_usage': memory_info,
                'inference_speed': {
                    'avg_generation_time_ms': avg_generation_time * 1000,
                    'tokens_per_second': tokens_per_second,
                    'test_prompt': prompt
                }
            }
            
            logger.info(f"Average generation time: {avg_generation_time*1000:.1f}ms")
            logger.info(f"Tokens per second: {tokens_per_second:.1f}")
            return result
            
        except Exception as e:
            logger.error(f"Performance benchmark failed: {str(e)}")
            return {'benchmark': 'Performance', 'error': str(e)}
    
    def _generate_comprehensive_report(self):
        """Generate comprehensive evaluation report with visualizations"""
        report = self._create_markdown_report()
        self._create_html_dashboard()
        self._create_performance_charts()
        
        # Save markdown report
        report_file = self.output_path / f"comprehensive_report_{self.run_id}.md"
        with open(report_file, 'w') as f:
            f.write(report)
            
        self.logger.info(f"Comprehensive report generated: {report_file}")
    
    def _create_markdown_report(self) -> str:
        """Create detailed markdown evaluation report"""
        report = f"""# Sheikh-2.5-Coder Comprehensive Evaluation Report

## Executive Summary
**Run ID**: {self.run_id}  
**Evaluation Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Model Path**: {self.model_path}  
**Total Benchmarks**: {len(self.results)}

## Target Achievement Status

"""
        
        # Check target achievement
        targets_met = []
        targets_missed = []
        
        for benchmark, target_value in self.targets.items():
            benchmark_key = benchmark.replace('_', '_').lower()
            actual_value = self._extract_metric_value(benchmark_key)
            
            if actual_value is not None:
                if actual_value >= target_value:
                    targets_met.append((benchmark, target_value, actual_value))
                else:
                    targets_missed.append((benchmark, target_value, actual_value))
        
        # Report targets
        if targets_met:
            report += "### ✅ Targets Met\n"
            for benchmark, target, actual in targets_met:
                report += f"- **{benchmark.replace('_', ' ').title()}**: {actual:.3f} ≥ {target:.3f}\n"
        
        if targets_missed:
            report += "\n### ❌ Targets Missed\n"
            for benchmark, target, actual in targets_missed:
                report += f"- **{benchmark.replace('_', ' ').title()}**: {actual:.3f} < {target:.3f}\n"
        
        report += "\n## Detailed Benchmark Results\n\n"
        
        # Detailed results for each benchmark
        for benchmark_name, results in self.results.items():
            report += f"### {benchmark_name.replace('_', ' ').title()}\n\n"
            
            if 'status' in results and results['status'] == 'failed':
                report += f"**Status**: ❌ Failed\n"
                report += f"**Error**: {results.get('error', 'Unknown error')}\n\n"
                continue
            
            # Format results based on benchmark type
            report += self._format_benchmark_results(benchmark_name, results)
            report += "\n"
        
        # Performance summary
        report += "## Performance Summary\n\n"
        report += self._create_performance_summary()
        
        # Recommendations
        report += "\n## Recommendations\n\n"
        report += self._generate_recommendations()
        
        return report
    
    def _format_benchmark_results(self, benchmark_name: str, results: Dict) -> str:
        """Format results for a specific benchmark"""
        formatted = ""
        
        if benchmark_name == 'mmlu_evaluation':
            accuracy = results.get('accuracy', 0)
            formatted += f"- **Accuracy**: {accuracy:.3f} (Target: ≥{self.targets.get('mmlu_code_accuracy', 0.6)})\n"
            formatted += f"- **Correct Answers**: {results.get('correct', 0)}/{results.get('total', 0)}\n"
            formatted += f"- **Evaluation Time**: {results.get('evaluation_time', 'N/A')}\n"
            
        elif benchmark_name == 'humaneval_evaluation':
            pass1 = results.get('pass_at_1', 0)
            formatted += f"- **Pass@1**: {pass1:.3f} (Target: ≥{self.targets.get('humaneval_pass1', 0.4)})\n"
            formatted += f"- **Total Problems**: {results.get('total_problems', 0)}\n"
            formatted += f"- **Solved Problems**: {results.get('solved_problems', 0)}\n"
            
        elif benchmark_name == 'web_dev_tests':
            js_quality = results.get('javascript_quality', 0)
            formatted += f"- **JavaScript/TypeScript Quality**: {js_quality:.3f}\n"
            formatted += f"- **React Components Score**: {results.get('react_score', 0):.3f}\n"
            formatted += f"- **XML Configuration Score**: {results.get('xml_score', 0):.3f}\n"
            
        elif benchmark_name == 'performance_benchmark':
            tokens_per_sec = results.get('tokens_per_second', 0)
            formatted += f"- **Inference Speed**: {tokens_per_sec:.1f} tokens/second\n"
            formatted += f"- **Memory Usage**: {results.get('memory_usage_mb', 0):.1f} MB\n"
            formatted += f"- **Context Length**: {results.get('max_context_length', 'N/A')}\n"
            
        elif benchmark_name == 'code_quality_tests':
            syntax_validity = results.get('syntax_validity', 0)
            formatted += f"- **Syntax Validity**: {syntax_validity:.3f} (Target: ≥{self.targets.get('syntax_validity', 0.95)})\n"
            formatted += f"- **Language Coverage**: {results.get('language_coverage', 'N/A')}\n"
            
        # Add additional metrics if available
        if 'metrics' in results:
            formatted += "\n**Additional Metrics:**\n"
            for metric, value in results['metrics'].items():
                formatted += f"- {metric.replace('_', ' ').title()}: {value}\n"
        
        return formatted
    
    def _extract_metric_value(self, benchmark_key: str) -> Optional[float]:
        """Extract the main metric value for a benchmark"""
        if benchmark_key not in self.results:
            return None
            
        results = self.results[benchmark_key]
        
        metric_mapping = {
            'mmlu_evaluation': 'accuracy',
            'humaneval_evaluation': 'pass_at_1',
            'web_dev_tests': 'overall_quality',
            'performance_benchmark': 'tokens_per_second',
            'code_quality_tests': 'syntax_validity',
            'regression_testing': 'relative_score'
        }
        
        metric_name = metric_mapping.get(benchmark_key)
        if metric_name and metric_name in results:
            return float(results[metric_name])
        
        return None
    
    def _create_performance_summary(self) -> str:
        """Create performance summary section"""
        summary = "### Model Performance Metrics\n\n"
        
        # Hardware utilization
        if self.monitor:
            cpu_usage = self.monitor.get_cpu_usage()
            memory_usage = self.monitor.get_memory_usage()
            
            summary += f"- **CPU Utilization**: {cpu_usage:.1f}%\n"
            summary += f"- **Memory Usage**: {memory_usage:.1f} GB\n"
        
        # Benchmark completion status
        completed = sum(1 for r in self.results.values() if r.get('status') != 'failed')
        total = len(self.results)
        
        summary += f"\n### Benchmark Completion\n\n"
        summary += f"- **Completed**: {completed}/{total} benchmarks\n"
        summary += f"- **Success Rate**: {completed/total*100:.1f}%\n"
        
        return summary
    
    def _generate_recommendations(self) -> str:
        """Generate improvement recommendations based on results"""
        recommendations = []
        
        # Check each target
        for target_name, target_value in self.targets.items():
            metric_value = self._extract_metric_value(target_name)
            if metric_value and metric_value < target_value:
                gap = target_value - metric_value
                
                if target_name == 'mmlu_code_accuracy':
                    recommendations.append(
                        f"- **Code Understanding**: Focus on training with more programming-focused datasets "
                        f"(Gap: {gap:.3f})"
                    )
                elif target_name == 'humaneval_pass1':
                    recommendations.append(
                        f"- **Code Generation**: Improve prompt engineering and fine-tuning on coding tasks "
                        f"(Gap: {gap:.3f})"
                    )
                elif target_name == 'syntax_validity':
                    recommendations.append(
                        f"- **Syntax Validation**: Implement stricter syntax checking during training "
                        f"(Gap: {gap:.3f})"
                    )
        
        # Performance recommendations
        if 'performance_benchmark' in self.results:
            tokens_per_sec = self.results['performance_benchmark'].get('tokens_per_second', 0)
            if tokens_per_sec < 50:  # Threshold for acceptable speed
                recommendations.append(
                    "- **Inference Speed**: Consider model optimization techniques like quantization or pruning"
                )
        
        if not recommendations:
            recommendations.append("✅ Model performance meets all target criteria!")
            recommendations.append("Consider focusing on edge cases and specialized domain improvements.")
        
        return "\n".join(recommendations)
    
    def _create_html_dashboard(self):
        """Create HTML dashboard for interactive results viewing"""
        # This would create an interactive HTML dashboard
        # For now, just note that it should be implemented
        self.logger.info("HTML dashboard creation would be implemented here")
    
    def _create_performance_charts(self):
        """Create performance visualization charts"""
        # This would create matplotlib/seaborn charts
        # For now, just note that it should be implemented
        self.logger.info("Performance charts creation would be implemented here")
    
    def _save_all_results(self):
        """Save all evaluation results in multiple formats"""
        # Save raw JSON results
        json_file = self.output_path / f"evaluation_results_{self.run_id}.json"
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save CSV summary
        csv_file = self.output_path / f"evaluation_summary_{self.run_id}.csv"
        self._save_csv_summary(csv_file)
        
        # Save performance metrics
        metrics_file = self.output_path / f"performance_metrics_{self.run_id}.json"
        with open(metrics_file, 'w') as f:
            json.dump(self._extract_performance_metrics(), f, indent=2)
        
        self.logger.info(f"All results saved to {self.output_path}")
    
    def _save_csv_summary(self, csv_file: Path):
        """Save evaluation summary as CSV"""
        data = []
        for benchmark, results in self.results.items():
            if 'status' in results and results['status'] == 'failed':
                data.append({
                    'benchmark': benchmark,
                    'status': 'failed',
                    'error': results.get('error', 'Unknown')
                })
                continue
                
            # Extract key metrics based on benchmark type
            if benchmark == 'mmlu_evaluation':
                data.append({
                    'benchmark': benchmark,
                    'accuracy': results.get('accuracy', 0),
                    'correct': results.get('correct', 0),
                    'total': results.get('total', 0),
                    'status': 'completed'
                })
            elif benchmark == 'humaneval_evaluation':
                data.append({
                    'benchmark': benchmark,
                    'pass_at_1': results.get('pass_at_1', 0),
                    'solved': results.get('solved_problems', 0),
                    'total': results.get('total_problems', 0),
                    'status': 'completed'
                })
            elif benchmark == 'performance_benchmark':
                data.append({
                    'benchmark': benchmark,
                    'tokens_per_second': results.get('tokens_per_second', 0),
                    'memory_usage_mb': results.get('memory_usage_mb', 0),
                    'status': 'completed'
                })
            else:
                # Generic format for other benchmarks
                data.append({
                    'benchmark': benchmark,
                    'primary_metric': results.get('primary_metric', results.get('score', 0)),
                    'status': 'completed'
                })
        
        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False)
        self.logger.info(f"CSV summary saved: {csv_file}")
    
    def _extract_performance_metrics(self) -> Dict:
        """Extract performance metrics for tracking"""
        metrics = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'run_id': self.run_id,
            'model_path': str(self.model_path),
            'hardware_info': self._get_hardware_info(),
            'target_achievement': {},
            'benchmark_scores': {}
        }
        
        # Extract scores vs targets
        for target_name, target_value in self.targets.items():
            actual_value = self._extract_metric_value(target_name)
            if actual_value is not None:
                metrics['target_achievement'][target_name] = {
                    'target': target_value,
                    'actual': actual_value,
                    'achieved': actual_value >= target_value,
                    'gap': target_value - actual_value if actual_value < target_value else 0
                }
        
        # Extract benchmark scores
        for benchmark_name, results in self.results.items():
            if 'status' in results and results['status'] == 'failed':
                continue
                
            primary_metric = self._extract_metric_value(benchmark_name)
            if primary_metric is not None:
                metrics['benchmark_scores'][benchmark_name] = {
                    'score': primary_metric,
                    'metadata': {k: v for k, v in results.items() 
                               if k not in ['status', 'error'] and isinstance(v, (int, float, str))}
                }
        
        return metrics
    
    def _get_hardware_info(self) -> Dict:
        """Get hardware information"""
        try:
            import psutil
            return {
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'python_version': sys.version,
                'torch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
                'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
        except Exception as e:
            self.logger.warning(f"Failed to get hardware info: {str(e)}")
            return {}
    
    def _print_performance_summary(self):
        """Print final performance summary to console"""
        print("\n" + "="*80)
        print("SHEIKH-2.5-CODER EVALUATION SUMMARY")
        print("="*80)
        
        # Overall status
        total_benchmarks = len(self.results)
        successful_benchmarks = sum(1 for r in self.results.values() 
                                  if r.get('status') != 'failed')
        
        print(f"Total Benchmarks: {total_benchmarks}")
        print(f"Successful: {successful_benchmarks}")
        print(f"Failed: {total_benchmarks - successful_benchmarks}")
        
        # Target achievement
        print("\nTARGET ACHIEVEMENT:")
        targets_achieved = 0
        for target_name, target_value in self.targets.items():
            actual_value = self._extract_metric_value(target_name)
            if actual_value is not None:
                achieved = actual_value >= target_value
                status = "✅" if achieved else "❌"
                print(f"{status} {target_name.replace('_', ' ').title()}: "
                      f"{actual_value:.3f} / {target_value:.3f}")
                if achieved:
                    targets_achieved += 1
        
        print(f"\nTargets Achieved: {targets_achieved}/{len(self.targets)}")
        
        # Detailed results
        print("\nBENCHMARK RESULTS:")
        for benchmark_name, results in self.results.items():
            if 'status' in results and results['status'] == 'failed':
                print(f"❌ {benchmark_name}: Failed - {results.get('error', 'Unknown')}")
            else:
                score = self._extract_metric_value(benchmark_name)
                if score is not None:
                    print(f"✅ {benchmark_name}: {score:.3f}")
                else:
                    print(f"✅ {benchmark_name}: Completed")
        
        print("="*80)


class HardwareMonitor:
    """Hardware monitoring utility"""
    
    def __init__(self):
        self.start_time = time.time()
        self.initial_memory = self.get_memory_usage()
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        try:
            return psutil.cpu_percent(interval=1)
        except:
            return 0.0
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in GB"""
        try:
            return psutil.virtual_memory().used / (1024**3)
        except:
            return 0.0
    
    def get_gpu_memory(self) -> Dict[int, float]:
        """Get GPU memory usage if available"""
        memory_info = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_info[i] = torch.cuda.memory_allocated(i) / (1024**3)
        return memory_info


def main():
    """Main evaluation orchestrator function"""
    parser = argparse.ArgumentParser(description='Sheikh-2.5-Coder Comprehensive Evaluation')
    
    parser.add_argument('--model_path', required=True, help='Path to model directory')
    parser.add_argument('--config', required=True, help='Path to evaluation configuration YAML')
    parser.add_argument('--output_path', required=True, help='Output directory for results')
    parser.add_argument('--run_id', required=True, help='Unique run identifier')
    parser.add_argument('--skip_load', action='store_true', help='Skip model loading for dry run')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.model_path).exists():
        print(f"Error: Model path {args.model_path} does not exist")
        return 1
    
    if not Path(args.config).exists():
        print(f"Error: Config file {args.config} does not exist")
        return 1
    
    # Initialize orchestrator
    orchestrator = EvaluationOrchestrator(
        config_path=args.config,
        model_path=args.model_path,
        run_id=args.run_id,
        output_path=args.output_path
    )
    
    # Run evaluation
    try:
        if args.skip_load:
            orchestrator.logger.info("Dry run mode - skipping model loading")
            orchestrator.results = {'status': 'dry_run'}
        else:
            success = orchestrator.run_comprehensive_evaluation()
            if not success:
                orchestrator.logger.error("Evaluation failed")
                return 1
        
        orchestrator.logger.info("Evaluation pipeline completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        orchestrator.logger.info("Evaluation interrupted by user")
        return 1
    except Exception as e:
        orchestrator.logger.error(f"Evaluation failed with exception: {str(e)}")
        orchestrator.logger.error(traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main())