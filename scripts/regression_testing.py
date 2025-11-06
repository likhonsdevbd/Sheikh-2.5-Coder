#!/usr/bin/env python3
"""
Regression Testing for Sheikh-2.5-Coder
Compares model performance against baselines and detects regressions
"""

import os
import sys
import json
import yaml
import argparse
import logging
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import time
import hashlib
import pickle
from scipy import stats
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add parent directories to path
sys.path.append('../')
sys.path.append('../../')

class RegressionTester:
    """Regression testing framework for model evaluation"""
    
    def __init__(self, config_path: str, model_path: str, output_path: str, run_id: str):
        """Initialize regression tester"""
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
        
        # Regression test settings
        self.regression_config = self.config.get('regression_testing', {})
        self.degradation_thresholds = self.regression_config.get('degradation_thresholds', {})
        
        # Baseline storage
        self.baseline_results = {}
        
        self.logger.info(f"Regression Tester initialized for run: {run_id}")
    
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
            'regression_testing': {
                'baseline_models': ['previous_version'],
                'comparison_metrics': ['accuracy', 'rouge_score', 'bleu_score'],
                'statistical_tests': ['t_test', 'wilcoxon_test'],
                'degradation_thresholds': {
                    'accuracy': 0.05,
                    'performance': 0.1
                }
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for this tester"""
        log_file = self.output_path / f"regression_{self.run_id}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(f'RegressionTester_{self.run_id}')
    
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
    
    def load_baseline_results(self, baseline_paths: List[str]) -> Dict:
        """Load baseline results for comparison"""
        self.logger.info("Loading baseline results...")
        
        baseline_results = {}
        
        for baseline_path in baseline_paths:
            baseline_name = Path(baseline_path).name
            
            # Look for existing evaluation results
            potential_results = [
                Path(baseline_path) / "evaluation_results.json",
                Path(baseline_path) / "eval_summary.json",
                Path(baseline_path) / "results.json"
            ]
            
            results_file = None
            for result_path in potential_results:
                if result_path.exists():
                    results_file = result_path
                    break
            
            if results_file:
                try:
                    with open(results_file, 'r') as f:
                        baseline_data = json.load(f)
                    
                    baseline_results[baseline_name] = {
                        'results_file': str(results_file),
                        'data': baseline_data,
                        'loaded_at': datetime.now().isoformat()
                    }
                    
                    self.logger.info(f"Loaded baseline: {baseline_name}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to load baseline {baseline_name}: {e}")
            else:
                self.logger.warning(f"No results found for baseline: {baseline_path}")
        
        self.baseline_results = baseline_results
        return baseline_results
    
    def run_current_model_evaluation(self) -> Dict:
        """Run evaluation on current model"""
        self.logger.info("Running current model evaluation...")
        
        # Run simplified evaluation for regression testing
        try:
            current_results = {}
            
            # Basic syntax validation
            syntax_results = self._run_basic_syntax_test()
            current_results['syntax_validity'] = syntax_results
            
            # Code generation quality
            generation_results = self._run_code_generation_test()
            current_results['code_generation'] = generation_results
            
            # Performance metrics
            performance_results = self._run_performance_test()
            current_results['performance'] = performance_results
            
            # Compile final results
            final_results = {
                'status': 'completed',
                'model_path': str(self.model_path),
                'evaluation_timestamp': datetime.now().isoformat(),
                'run_id': self.run_id,
                'metrics': current_results,
                'summary_scores': self._calculate_summary_scores(current_results)
            }
            
            self.logger.info("Current model evaluation completed")
            return final_results
            
        except Exception as e:
            self.logger.error(f"Current model evaluation failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'evaluation_timestamp': datetime.now().isoformat()
            }
    
    def _run_basic_syntax_test(self) -> Dict:
        """Run basic syntax validation test"""
        test_prompts = [
            "def fibonacci(n):",
            "class Calculator:",
            "function calculateSum(numbers) {",
            "const fetchData = async () => {"
        ]
        
        valid_count = 0
        total_count = len(test_prompts)
        results = []
        
        for prompt in test_prompts:
            try:
                # Generate code
                generated_code = self._generate_code(prompt)
                
                # Basic syntax check
                is_valid = self._check_basic_syntax(generated_code)
                
                if is_valid:
                    valid_count += 1
                
                results.append({
                    'prompt': prompt,
                    'generated_code': generated_code[:100] + '...' if len(generated_code) > 100 else generated_code,
                    'is_valid': is_valid
                })
                
            except Exception as e:
                results.append({
                    'prompt': prompt,
                    'error': str(e),
                    'is_valid': False
                })
        
        return {
            'valid_count': valid_count,
            'total_count': total_count,
            'validity_rate': valid_count / total_count if total_count > 0 else 0,
            'sample_results': results
        }
    
    def _run_code_generation_test(self) -> Dict:
        """Run code generation quality test"""
        generation_tasks = [
            "Create a Python function to calculate factorial",
            "Write a JavaScript function to sort an array",
            "Create a CSS responsive grid layout",
            "Write an HTML form with validation"
        ]
        
        quality_scores = []
        results = []
        
        for task in generation_tasks:
            try:
                # Generate code
                generated_code = self._generate_code(task)
                
                # Basic quality assessment
                quality_score = self._assess_code_quality(generated_code, task)
                quality_scores.append(quality_score)
                
                results.append({
                    'task': task,
                    'quality_score': quality_score,
                    'generated_code': generated_code[:200] + '...' if len(generated_code) > 200 else generated_code
                })
                
            except Exception as e:
                results.append({
                    'task': task,
                    'error': str(e),
                    'quality_score': 0
                })
        
        return {
            'average_quality_score': np.mean(quality_scores) if quality_scores else 0,
            'quality_scores': quality_scores,
            'sample_results': results
        }
    
    def _run_performance_test(self) -> Dict:
        """Run basic performance test"""
        test_prompt = "def example_function():"
        
        latencies = []
        num_runs = 10
        
        try:
            for _ in range(num_runs):
                start_time = time.time()
                self._generate_code(test_prompt)
                latency = time.time() - start_time
                latencies.append(latency)
            
            return {
                'average_latency_ms': np.mean(latencies) * 1000,
                'p95_latency_ms': np.percentile(latencies, 95) * 1000,
                'tokens_per_second': 100 / (np.mean(latencies) * 1000) if latencies else 0,
                'sample_count': len(latencies)
            }
            
        except Exception as e:
            self.logger.error(f"Performance test failed: {e}")
            return {'error': str(e)}
    
    def _generate_code(self, prompt: str) -> str:
        """Generate code for a given prompt"""
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
            
            # Generate response
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
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
            self.logger.error(f"Code generation failed: {e}")
            return ""
    
    def _check_basic_syntax(self, code: str) -> bool:
        """Check basic syntax validity"""
        if not code or len(code.strip()) < 5:
            return False
        
        try:
            # Basic bracket matching
            if code.count('(') != code.count(')'):
                return False
            if code.count('{') != code.count('}'):
                return False
            
            # Basic structure check
            if any(keyword in code for keyword in ['def ', 'function', 'class ', 'const ', 'let ']):
                return True
            
            return len(code.strip()) > 20
            
        except Exception:
            return False
    
    def _assess_code_quality(self, code: str, task: str) -> float:
        """Assess code quality for a given task"""
        if not code or len(code.strip()) < 10:
            return 0.0
        
        score = 0.0
        
        # Length check
        if len(code) > 50:
            score += 0.2
        
        # Task-specific keywords
        task_lower = task.lower()
        
        if 'python' in task_lower:
            python_keywords = ['def ', 'class ', 'import ', 'return ']
            found_keywords = sum(1 for keyword in python_keywords if keyword in code)
            score += (found_keywords / len(python_keywords)) * 0.4
        
        elif 'javascript' in task_lower or 'js' in task_lower:
            js_keywords = ['function', 'const', 'let', '=>', 'return']
            found_keywords = sum(1 for keyword in js_keywords if keyword in code)
            score += (found_keywords / len(js_keywords)) * 0.4
        
        elif 'css' in task_lower:
            css_keywords = ['{', '}', ':', ';', '.', '#']
            found_keywords = sum(1 for keyword in css_keywords if keyword in code)
            score += (found_keywords / len(css_keywords)) * 0.4
        
        elif 'html' in task_lower:
            html_keywords = ['<', '>', 'form', 'input']
            found_keywords = sum(1 for keyword in html_keywords if keyword in code)
            score += (found_keywords / len(html_keywords)) * 0.4
        
        # Structure validation
        if code.count('{') == code.count('}'):
            score += 0.2
        
        if code.count('(') == code.count(')'):
            score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_summary_scores(self, results: Dict) -> Dict:
        """Calculate summary scores from detailed results"""
        summary = {}
        
        if 'syntax_validity' in results:
            syntax_data = results['syntax_validity']
            summary['syntax_validity_score'] = syntax_data.get('validity_rate', 0)
        
        if 'code_generation' in results:
            generation_data = results['code_generation']
            summary['code_generation_score'] = generation_data.get('average_quality_score', 0)
        
        if 'performance' in results and 'average_latency_ms' in results['performance']:
            latency = results['performance']['average_latency_ms']
            # Convert latency to performance score (lower is better)
            summary['performance_score'] = max(0, 100 - latency / 10) / 100
        
        return summary
    
    def compare_with_baselines(self, current_results: Dict, baseline_results: Dict) -> Dict:
        """Compare current results with baselines"""
        self.logger.info("Comparing with baselines...")
        
        comparison_results = {}
        
        for baseline_name, baseline_data in baseline_results.items():
            self.logger.info(f"Comparing with baseline: {baseline_name}")
            
            baseline_metrics = self._extract_metrics_for_comparison(baseline_data['data'])
            current_metrics = self._extract_metrics_for_comparison(current_results)
            
            # Perform comparisons
            comparison = self._perform_metric_comparisons(current_metrics, baseline_metrics)
            comparison['baseline_name'] = baseline_name
            comparison['baseline_file'] = baseline_data.get('results_file', '')
            comparison['comparison_timestamp'] = datetime.now().isoformat()
            
            # Statistical significance testing
            if self._can_perform_statistical_tests(current_metrics, baseline_metrics):
                statistical_tests = self._perform_statistical_tests(current_metrics, baseline_metrics)
                comparison['statistical_tests'] = statistical_tests
            
            # Regression detection
            regressions = self._detect_regressions(current_metrics, baseline_metrics)
            comparison['regressions'] = regressions
            comparison['regression_count'] = len(regressions)
            
            comparison_results[baseline_name] = comparison
        
        return comparison_results
    
    def _extract_metrics_for_comparison(self, data: Dict) -> Dict:
        """Extract metrics from evaluation data for comparison"""
        metrics = {}
        
        try:
            # Extract summary scores
            if 'summary_scores' in data:
                metrics.update(data['summary_scores'])
            elif 'metrics' in data:
                metrics.update(data['metrics'])
            
            # Extract individual benchmark scores
            if 'detailed_results' in data:
                detailed = data['detailed_results']
                
                # MMLU accuracy
                if 'mmlu_evaluation' in detailed:
                    mmlu_data = detailed['mmlu_evaluation']
                    if 'accuracy' in mmlu_data:
                        metrics['mmlu_accuracy'] = mmlu_data['accuracy']
                
                # HumanEval pass@1
                if 'humaneval_evaluation' in detailed:
                    humaneval_data = detailed['humaneval_evaluation']
                    if 'pass_at_1' in humaneval_data:
                        metrics['humaneval_pass1'] = humaneval_data['pass_at_1']
                
                # Web development score
                if 'web_dev_tests' in detailed:
                    webdev_data = detailed['web_dev_tests']
                    if 'overall_quality_score' in webdev_data:
                        metrics['webdev_quality'] = webdev_data['overall_quality_score']
                
                # Performance metrics
                if 'performance_benchmark' in detailed:
                    perf_data = detailed['performance_benchmark']
                    if 'tokens_per_second' in perf_data:
                        metrics['tokens_per_second'] = perf_data['tokens_per_second']
                
                # Code quality metrics
                if 'code_quality_tests' in detailed:
                    quality_data = detailed['code_quality_tests']
                    if 'syntax_validity' in quality_data:
                        metrics['syntax_validity'] = quality_data['syntax_validity']
                    if 'codebleu_score' in quality_data:
                        metrics['codebleu_score'] = quality_data['codebleu_score']
            
            # Extract from flattened structure
            if 'accuracy' in data:
                metrics['accuracy'] = data['accuracy']
            if 'pass_at_1' in data:
                metrics['pass_at_1'] = data['pass_at_1']
            if 'tokens_per_second' in data:
                metrics['tokens_per_second'] = data['tokens_per_second']
            if 'syntax_validity' in data:
                metrics['syntax_validity'] = data['syntax_validity']
            if 'codebleu_score' in data:
                metrics['codebleu_score'] = data['codebleu_score']
                
        except Exception as e:
            self.logger.warning(f"Failed to extract metrics: {e}")
        
        return metrics
    
    def _perform_metric_comparisons(self, current_metrics: Dict, baseline_metrics: Dict) -> Dict:
        """Perform pairwise comparisons of metrics"""
        comparisons = {}
        
        common_metrics = set(current_metrics.keys()) & set(baseline_metrics.keys())
        
        for metric in common_metrics:
            current_value = current_metrics[metric]
            baseline_value = baseline_metrics[metric]
            
            # Calculate difference and percentage change
            absolute_diff = current_value - baseline_value
            percentage_diff = (absolute_diff / baseline_value * 100) if baseline_value != 0 else 0
            
            # Determine if it's an improvement or regression
            is_improvement = absolute_diff > 0
            is_significant = abs(percentage_diff) > 5  # 5% threshold
            
            comparisons[metric] = {
                'current_value': current_value,
                'baseline_value': baseline_value,
                'absolute_difference': absolute_diff,
                'percentage_difference': percentage_diff,
                'is_improvement': is_improvement,
                'is_significant': is_significant
            }
        
        return comparisons
    
    def _can_perform_statistical_tests(self, current_metrics: Dict, baseline_metrics: Dict) -> bool:
        """Check if we can perform statistical tests"""
        # For statistical tests, we need distributions, not single values
        # This is a simplified check - in practice, you'd need sample distributions
        return False  # Placeholder
    
    def _perform_statistical_tests(self, current_metrics: Dict, baseline_metrics: Dict) -> Dict:
        """Perform statistical significance tests"""
        # Placeholder for statistical tests
        # In practice, you'd need multiple samples to perform t-tests, etc.
        return {'note': 'Statistical tests require sample distributions'}
    
    def _detect_regressions(self, current_metrics: Dict, baseline_metrics: Dict) -> List[Dict]:
        """Detect performance regressions"""
        regressions = []
        
        # Define metrics where lower is better vs higher is better
        higher_is_better = ['accuracy', 'pass_at_1', 'tokens_per_second', 'syntax_validity', 'codebleu_score']
        
        for metric, comparison in self._perform_metric_comparisons(current_metrics, baseline_metrics).items():
            current_value = comparison['current_value']
            baseline_value = comparison['baseline_value']
            percentage_diff = comparison['percentage_difference']
            
            # Check for degradation
            if metric in higher_is_better:
                # For metrics where higher is better, regression is when current < baseline
                if current_value < baseline_value:
                    # Check if degradation exceeds threshold
                    degradation_threshold = self.degradation_thresholds.get('accuracy', 0.05) * 100
                    if percentage_diff < -degradation_threshold:
                        regressions.append({
                            'metric': metric,
                            'regression_type': 'performance_degradation',
                            'current_value': current_value,
                            'baseline_value': baseline_value,
                            'degradation_percent': abs(percentage_diff),
                            'severity': 'high' if abs(percentage_diff) > 10 else 'medium'
                        })
            else:
                # For metrics where lower is better (like latency), regression is when current > baseline
                if current_value > baseline_value:
                    degradation_threshold = self.degradation_thresholds.get('performance', 0.1) * 100
                    if percentage_diff > degradation_threshold:
                        regressions.append({
                            'metric': metric,
                            'regression_type': 'performance_degradation',
                            'current_value': current_value,
                            'baseline_value': baseline_value,
                            'degradation_percent': percentage_diff,
                            'severity': 'high' if percentage_diff > 20 else 'medium'
                        })
        
        return regressions
    
    def generate_regression_report(self, comparison_results: Dict) -> str:
        """Generate comprehensive regression report"""
        report = f"""# Regression Analysis Report

## Summary
- **Run ID**: {self.run_id}
- **Model Path**: {self.model_path}
- **Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Baselines Compared**: {len(comparison_results)}

"""
        
        total_regressions = 0
        total_improvements = 0
        
        for baseline_name, comparison in comparison_results.items():
            report += f"\n## Comparison with {baseline_name}\n\n"
            
            # Regression summary
            regressions = comparison.get('regressions', [])
            improvements = self._count_improvements(comparison.get('comparisons', {}))
            
            total_regressions += len(regressions)
            total_improvements += improvements
            
            if len(regressions) == 0 and improvements == 0:
                report += "✅ **No significant changes detected**\n\n"
            else:
                if len(regressions) > 0:
                    report += f"❌ **{len(regressions)} regressions detected**\n\n"
                    for regression in regressions:
                        report += f"- {regression['metric']}: {regression['degradation_percent']:.1f}% degradation "
                        report += f"(Current: {regression['current_value']:.3f}, Baseline: {regression['baseline_value']:.3f})\n"
                    report += "\n"
                
                if improvements > 0:
                    report += f"✅ **{improvements} improvements detected**\n\n"
            
            # Detailed comparisons
            if 'comparisons' in comparison:
                report += "### Detailed Metric Comparison\n\n"
                report += "| Metric | Current | Baseline | Change | % Change |\n"
                report += "|--------|---------|----------|--------|----------|\n"
                
                for metric, comp_data in comparison['comparisons'].items():
                    current = comp_data['current_value']
                    baseline = comp_data['baseline_value']
                    change = comp_data['absolute_difference']
                    pct_change = comp_data['percentage_difference']
                    
                    status = "✅" if change >= 0 else "❌"
                    report += f"| {metric} | {current:.3f} | {baseline:.3f} | {change:+.3f} | {pct_change:+.1f}% {status} |\n"
        
        # Overall summary
        report += f"\n## Overall Summary\n\n"
        report += f"- **Total Regressions**: {total_regressions}\n"
        report += f"- **Total Improvements**: {total_improvements}\n"
        
        if total_regressions == 0:
            report += "- **Status**: ✅ **NO REGRESSIONS DETECTED**\n"
        else:
            report += "- **Status**: ❌ **REGRESSIONS DETECTED - REVIEW REQUIRED**\n"
        
        return report
    
    def _count_improvements(self, comparisons: Dict) -> int:
        """Count improvements in comparisons"""
        improvements = 0
        for metric, comp_data in comparisons.items():
            if comp_data.get('is_improvement', False) and comp_data.get('is_significant', False):
                improvements += 1
        return improvements
    
    def run_regression_testing(self) -> Dict:
        """Run complete regression testing pipeline"""
        start_time = time.time()
        
        if not self.load_model():
            return {'status': 'failed', 'error': 'Model loading failed'}
        
        try:
            self.logger.info("Starting regression testing pipeline...")
            
            # Get baseline paths
            baseline_paths = self.regression_config.get('baseline_models', [])
            
            # Load baseline results
            baseline_results = {}
            if baseline_paths:
                baseline_results = self.load_baseline_results(baseline_paths)
                if not baseline_results:
                    self.logger.warning("No baseline results found - will only evaluate current model")
            
            # Run current model evaluation
            current_results = self.run_current_model_evaluation()
            
            if current_results.get('status') != 'completed':
                return current_results
            
            # Compare with baselines if available
            comparison_results = {}
            if baseline_results:
                comparison_results = self.compare_with_baselines(current_results, baseline_results)
            
            evaluation_time = time.time() - start_time
            
            # Compile final results
            final_results = {
                'status': 'completed',
                'benchmark': 'Regression Testing',
                'evaluation_time_seconds': evaluation_time,
                'current_model_results': current_results,
                'baseline_results': baseline_results,
                'comparison_results': comparison_results,
                'regression_summary': {
                    'total_baselines': len(baseline_results),
                    'total_comparisons': len(comparison_results),
                    'total_regressions': sum(c.get('regression_count', 0) for c in comparison_results.values()),
                    'regression_free': all(c.get('regression_count', 0) == 0 for c in comparison_results.values())
                },
                'generated_at': datetime.now().isoformat()
            }
            
            # Generate and save regression report
            if comparison_results:
                report = self.generate_regression_report(comparison_results)
                report_file = self.output_path / f"regression_report_{self.run_id}.md"
                with open(report_file, 'w') as f:
                    f.write(report)
                
                final_results['report_file'] = str(report_file)
            
            self.logger.info("Regression testing completed")
            
            # Save results
            self._save_results(final_results)
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Regression testing failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'evaluation_time_seconds': time.time() - start_time
            }
    
    def _save_results(self, results: Dict):
        """Save regression testing results"""
        # Save detailed results as JSON
        results_file = self.output_path / f"regression_results_{self.run_id}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save comparison summary as CSV
        if 'comparison_results' in results:
            comparison_data = []
            for baseline_name, comparison in results['comparison_results'].items():
                comparison_data.append({
                    'baseline': baseline_name,
                    'regression_count': comparison.get('regression_count', 0),
                    'has_regressions': comparison.get('regression_count', 0) > 0
                })
            
            if comparison_data:
                df = pd.DataFrame(comparison_data)
                csv_file = self.output_path / f"regression_summary_{self.run_id}.csv"
                df.to_csv(csv_file, index=False)
        
        self.logger.info(f"Regression results saved to {self.output_path}")


def main():
    """Main regression testing function"""
    parser = argparse.ArgumentParser(description='Regression Testing')
    
    parser.add_argument('--model_path', required=True, help='Path to model directory')
    parser.add_argument('--config', required=True, help='Path to evaluation configuration')
    parser.add_argument('--output_path', required=True, help='Output directory for results')
    parser.add_argument('--run_id', required=True, help='Unique run identifier')
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = RegressionTester(
        config_path=args.config,
        model_path=args.model_path,
        output_path=args.output_path,
        run_id=args.run_id
    )
    
    # Run testing
    try:
        results = tester.run_regression_testing()
        
        if results.get('status') == 'completed':
            print(f"\nRegression Testing Results:")
            print(f"Baselines Compared: {results.get('regression_summary', {}).get('total_baselines', 0)}")
            print(f"Total Regressions: {results.get('regression_summary', {}).get('total_regressions', 0)}")
            print(f"Regression Free: {'✅' if results.get('regression_summary', {}).get('regression_free') else '❌'}")
            print(f"Evaluation Time: {results.get('evaluation_time_seconds', 0):.1f}s")
            
            if 'report_file' in results:
                print(f"Regression Report: {results['report_file']}")
            
            return 0
        else:
            print(f"Regression testing failed: {results.get('error', 'Unknown error')}")
            return 1
            
    except KeyboardInterrupt:
        print("Testing interrupted by user")
        return 1
    except Exception as e:
        print(f"Testing failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())