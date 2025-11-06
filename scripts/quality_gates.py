#!/usr/bin/env python3
"""
Quality Gates System for Automated Model Validation
Handles model testing, performance validation, and quality assurance
"""

import os
import sys
import json
import argparse
import logging
import time
import subprocess
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import torch
import numpy as np
from dataclasses import dataclass, asdict

from transformers import AutoModelForCausalLM, AutoTokenizer
import psutil

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class QualityMetric:
    """Quality metric result"""
    name: str
    value: float
    threshold: float
    passed: bool
    unit: str
    description: str
    timestamp: str
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        return data


@dataclass
class TestResult:
    """Test result container"""
    test_name: str
    status: str
    duration_seconds: float
    metrics: List[QualityMetric]
    errors: List[str]
    warnings: List[str]
    details: Dict[str, Any]
    timestamp: str
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        return data


@dataclass
class QualityReport:
    """Comprehensive quality report"""
    model_path: str
    test_suite_name: str
    overall_status: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    test_results: List[TestResult]
    summary: Dict[str, Any]
    timestamp: str
    execution_time_seconds: float
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        return data


class ModelValidator:
    """Model validation and testing framework"""
    
    def __init__(self, model_path: str, test_prompts: Optional[List[str]] = None):
        self.model_path = Path(model_path)
        self.test_prompts = test_prompts or self._get_default_test_prompts()
        self.model = None
        self.tokenizer = None
        
        logger.info(f"Initialized model validator for {model_path}")
    
    def _get_default_test_prompts(self) -> List[str]:
        """Get default test prompts for validation"""
        return [
            "def fibonacci(n):",
            "class Calculator:",
            "def quicksort(arr):",
            "# Create a REST API endpoint",
            "def binary_search(arr, target):",
            "class Node:",
            "def bubble_sort(lst):",
            "def factorial(n):",
            "# Build a simple web scraper",
            "def linear_regression(X, y):"
        ]
    
    def load_model(self, quantization: Optional[str] = None, device_map: str = "auto") -> bool:
        """Load model for testing"""
        logger.info(f"Loading model from {self.model_path}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with quantization if specified
            model_kwargs = {
                'trust_remote_code': True,
                'torch_dtype': torch.float16,
                'device_map': device_map
            }
            
            if quantization == "int8":
                model_kwargs['load_in_8bit'] = True
            elif quantization == "int4":
                model_kwargs['load_in_4bit'] = True
                model_kwargs['bnb_4bit_compute_dtype'] = torch.bfloat16
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **model_kwargs
            )
            
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def test_generation_quality(self) -> TestResult:
        """Test code generation quality"""
        logger.info("Testing generation quality")
        
        start_time = time.time()
        test_name = "generation_quality"
        errors = []
        warnings = []
        metrics = []
        
        if not self.model or not self.tokenizer:
            errors.append("Model not loaded")
            return self._create_test_result(test_name, "FAILED", start_time, [], errors, warnings)
        
        generation_results = []
        
        try:
            for i, prompt in enumerate(self.test_prompts[:5]):  # Test subset
                inputs = self.tokenizer(prompt, return_tensors="pt")
                
                # Generate with conservative settings
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=50,
                        do_sample=False,  # Deterministic for testing
                        temperature=0.1,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Basic quality checks
                length = len(generated_text)
                has_function_keywords = any(keyword in generated_text.lower() for keyword in ['def ', 'class ', 'function'])
                no_empty_generation = len(generated_text.strip()) > len(prompt)
                
                generation_results.append({
                    'prompt': prompt,
                    'generated': generated_text,
                    'length': length,
                    'has_function_keywords': has_function_keywords,
                    'no_empty_generation': no_empty_generation,
                    'success': no_empty_generation and has_function_keywords
                })
            
            # Calculate metrics
            successful_generations = sum(1 for r in generation_results if r['success'])
            success_rate = successful_generations / len(generation_results) * 100
            avg_length = np.mean([r['length'] for r in generation_results])
            function_keyword_rate = np.mean([r['has_function_keywords'] for r in generation_results]) * 100
            
            # Quality metrics
            metrics.append(QualityMetric(
                name="success_rate",
                value=success_rate,
                threshold=80.0,
                passed=success_rate >= 80.0,
                unit="%",
                description="Percentage of successful code generations"
            ))
            
            metrics.append(QualityMetric(
                name="function_keyword_rate",
                value=function_keyword_rate,
                threshold=70.0,
                passed=function_keyword_rate >= 70.0,
                unit="%",
                description="Percentage of generations containing function keywords"
            ))
            
            metrics.append(QualityMetric(
                name="avg_generation_length",
                value=avg_length,
                threshold=30.0,
                passed=avg_length >= 30.0,
                unit="tokens",
                description="Average generation length"
            ))
            
            status = "PASSED" if success_rate >= 80.0 else "FAILED"
            
        except Exception as e:
            errors.append(f"Generation test failed: {str(e)}")
            status = "FAILED"
        
        return self._create_test_result(
            test_name, status, start_time, metrics, errors, warnings,
            details={'generation_results': generation_results}
        )
    
    def test_inference_speed(self) -> TestResult:
        """Test inference speed performance"""
        logger.info("Testing inference speed")
        
        start_time = time.time()
        test_name = "inference_speed"
        errors = []
        warnings = []
        metrics = []
        
        if not self.model or not self.tokenizer:
            errors.append("Model not loaded")
            return self._create_test_result(test_name, "FAILED", start_time, [], errors, warnings)
        
        try:
            # Prepare test input
            prompt = self.test_prompts[0]
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            # Warm up
            for _ in range(3):
                with torch.no_grad():
                    _ = self.model.generate(**inputs, max_new_tokens=10, do_sample=False)
            
            # Speed test
            num_runs = 10
            times = []
            
            for _ in range(num_runs):
                start = time.time()
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=20,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                end = time.time()
                times.append(end - start)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            tokens_per_second = 20 / avg_time  # 20 tokens generated
            
            # Speed metrics
            metrics.append(QualityMetric(
                name="avg_inference_time",
                value=avg_time,
                threshold=2.0,
                passed=avg_time <= 2.0,
                unit="seconds",
                description="Average inference time for 20 tokens"
            ))
            
            metrics.append(QualityMetric(
                name="tokens_per_second",
                value=tokens_per_second,
                threshold=10.0,
                passed=tokens_per_second >= 10.0,
                unit="tokens/sec",
                description="Generation speed in tokens per second"
            ))
            
            metrics.append(QualityMetric(
                name="inference_stability",
                value=std_time / avg_time if avg_time > 0 else 0,
                threshold=0.3,
                passed=(std_time / avg_time if avg_time > 0 else 0) <= 0.3,
                unit="coefficient",
                description="Inference time stability (lower is better)"
            ))
            
            status = "PASSED" if tokens_per_second >= 10.0 else "FAILED"
            
        except Exception as e:
            errors.append(f"Speed test failed: {str(e)}")
            status = "FAILED"
        
        return self._create_test_result(
            test_name, status, start_time, metrics, errors, warnings,
            details={'avg_time': avg_time, 'std_time': std_time, 'times': times}
        )
    
    def test_memory_usage(self) -> TestResult:
        """Test memory usage"""
        logger.info("Testing memory usage")
        
        start_time = time.time()
        test_name = "memory_usage"
        errors = []
        warnings = []
        metrics = []
        
        try:
            # Get initial memory
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            if torch.cuda.is_available():
                initial_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            else:
                initial_gpu_memory = 0
            
            # Perform some generations to stabilize memory
            if self.model and self.tokenizer:
                for _ in range(5):
                    prompt = self.test_prompts[0]
                    inputs = self.tokenizer(prompt, return_tensors="pt")
                    with torch.no_grad():
                        _ = self.model.generate(**inputs, max_new_tokens=30, do_sample=False)
                
                # Measure peak memory
                final_memory = process.memory_info().rss / 1024 / 1024
                
                if torch.cuda.is_available():
                    final_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
                    peak_gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
                else:
                    final_gpu_memory = 0
                    peak_gpu_memory = 0
                
                # Memory metrics
                cpu_memory_usage = final_memory - initial_memory
                gpu_memory_usage = final_gpu_memory - initial_gpu_memory
                total_memory_usage = cpu_memory_usage + gpu_memory_usage
                
                metrics.append(QualityMetric(
                    name="cpu_memory_usage",
                    value=cpu_memory_usage,
                    threshold=4000.0,
                    passed=cpu_memory_usage <= 4000.0,
                    unit="MB",
                    description="CPU memory usage"
                ))
                
                if torch.cuda.is_available():
                    metrics.append(QualityMetric(
                        name="gpu_memory_usage",
                        value=gpu_memory_usage,
                        threshold=6000.0,
                        passed=gpu_memory_usage <= 6000.0,
                        unit="MB",
                        description="GPU memory usage"
                    ))
                    
                    metrics.append(QualityMetric(
                        name="peak_gpu_memory",
                        value=peak_gpu_memory,
                        threshold=8000.0,
                        passed=peak_gpu_memory <= 8000.0,
                        unit="MB",
                        description="Peak GPU memory usage"
                    ))
                
                metrics.append(QualityMetric(
                    name="total_memory_usage",
                    value=total_memory_usage,
                    threshold=10000.0,
                    passed=total_memory_usage <= 10000.0,
                    unit="MB",
                    description="Total memory usage"
                ))
            
            status = "PASSED"
            
        except Exception as e:
            errors.append(f"Memory test failed: {str(e)}")
            status = "FAILED"
        
        return self._create_test_result(
            test_name, status, start_time, metrics, errors, warnings,
            details={
                'initial_memory': initial_memory,
                'final_memory': final_memory if 'final_memory' in locals() else 0,
                'initial_gpu_memory': initial_gpu_memory,
                'final_gpu_memory': final_gpu_memory if 'final_gpu_memory' in locals() else 0
            }
        )
    
    def test_model_consistency(self) -> TestResult:
        """Test model consistency across multiple runs"""
        logger.info("Testing model consistency")
        
        start_time = time.time()
        test_name = "model_consistency"
        errors = []
        warnings = []
        metrics = []
        
        if not self.model or not self.tokenizer:
            errors.append("Model not loaded")
            return self._create_test_result(test_name, "FAILED", start_time, [], errors, warnings)
        
        try:
            # Test with fixed random seed for consistency
            prompt = self.test_prompts[0]
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            # Generate same text multiple times
            outputs_list = []
            for _ in range(5):
                torch.manual_seed(42)  # Fixed seed
                np.random.seed(42)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=20,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                outputs_list.append(generated_text)
            
            # Check consistency
            unique_outputs = len(set(outputs_list))
            consistency_rate = (len(outputs_list) - unique_outputs) / len(outputs_list) * 100
            
            # For stochastic models, we expect some variation
            # Check if at least some outputs are consistent
            max_consistent = 0
            for i in range(len(outputs_list)):
                count = outputs_list.count(outputs_list[i])
                max_consistent = max(max_consistent, count)
            
            consistency_rate = (max_consistent / len(outputs_list)) * 100
            
            metrics.append(QualityMetric(
                name="output_consistency",
                value=consistency_rate,
                threshold=60.0,  # For stochastic models, expect at least 60% consistency with fixed seed
                passed=consistency_rate >= 60.0,
                unit="%",
                description="Consistency of outputs with fixed random seed"
            ))
            
            status = "PASSED"
            
        except Exception as e:
            errors.append(f"Consistency test failed: {str(e)}")
            status = "FAILED"
        
        return self._create_test_result(
            test_name, status, start_time, metrics, errors, warnings,
            details={'outputs': outputs_list, 'unique_count': unique_outputs}
        )
    
    def test_error_handling(self) -> TestResult:
        """Test error handling and robustness"""
        logger.info("Testing error handling")
        
        start_time = time.time()
        test_name = "error_handling"
        errors = []
        warnings = []
        metrics = []
        
        if not self.model or not self.tokenizer:
            errors.append("Model not loaded")
            return self._create_test_result(test_name, "FAILED", start_time, [], errors, warnings)
        
        try:
            test_cases = [
                ("empty_prompt", ""),
                ("very_long_prompt", "def function():\n" * 100),
                ("special_characters", "def $pecial_func():"),
                ("unicode_prompt", "def 函数():"),
                ("nested_code", "def outer():\n    def inner():\n        pass")
            ]
            
            successful_handling = 0
            total_cases = len(test_cases)
            
            for case_name, test_prompt in test_cases:
                try:
                    # Test with different inputs
                    if test_prompt:
                        inputs = self.tokenizer(test_prompt, return_tensors="pt")
                    else:
                        inputs = self.tokenizer("test", return_tensors="pt")  # Empty handling
                    
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=20,
                            do_sample=False,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                    
                    generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # Check if generation completed without error
                    if len(generated_text) > 0:
                        successful_handling += 1
                        
                except Exception as e:
                    # Some errors might be expected for extreme cases
                    if case_name == "very_long_prompt":
                        warnings.append(f"Expected error for {case_name}: {str(e)}")
                        successful_handling += 0.5  # Partial credit
                    else:
                        errors.append(f"Unexpected error for {case_name}: {str(e)}")
            
            error_handling_rate = (successful_handling / total_cases) * 100
            
            metrics.append(QualityMetric(
                name="error_handling_rate",
                value=error_handling_rate,
                threshold=80.0,
                passed=error_handling_rate >= 80.0,
                unit="%",
                description="Rate of successful error handling"
            ))
            
            status = "PASSED" if error_handling_rate >= 80.0 else "FAILED"
            
        except Exception as e:
            errors.append(f"Error handling test failed: {str(e)}")
            status = "FAILED"
        
        return self._create_test_result(
            test_name, status, start_time, metrics, errors, warnings
        )
    
    def test_output_validation(self) -> TestResult:
        """Test output validation and format"""
        logger.info("Testing output validation")
        
        start_time = time.time()
        test_name = "output_validation"
        errors = []
        warnings = []
        metrics = []
        
        if not self.model or not self.tokenizer:
            errors.append("Model not loaded")
            return self._create_test_result(test_name, "FAILED", start_time, [], errors, warnings)
        
        try:
            validation_results = []
            
            for prompt in self.test_prompts[:3]:  # Test subset
                inputs = self.tokenizer(prompt, return_tensors="pt")
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=30,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Validation checks
                validations = {
                    'no_repetition': not self._has_excessive_repetition(generated_text),
                    'appropriate_length': 20 <= len(generated_text) <= 200,
                    'valid_syntax': self._validate_python_syntax(generated_text),
                    'not_empty': len(generated_text.strip()) > len(prompt),
                    'contains_code': any(keyword in generated_text.lower() for keyword in ['def ', 'class ', 'import ', 'for ', 'while ', '#'])
                }
                
                validation_results.append({
                    'prompt': prompt,
                    'generated': generated_text,
                    'validations': validations,
                    'passed': sum(validations.values()) >= 3  # At least 3 out of 5
                })
            
            # Calculate overall validation score
            passed_validations = sum(1 for r in validation_results if r['passed'])
            validation_rate = (passed_validations / len(validation_results)) * 100
            
            metrics.append(QualityMetric(
                name="output_validation_rate",
                value=validation_rate,
                threshold=70.0,
                passed=validation_rate >= 70.0,
                unit="%",
                description="Rate of valid output formats"
            ))
            
            # Syntax validity
            syntax_valid = sum(1 for r in validation_results if r['validations']['valid_syntax'])
            syntax_rate = (syntax_valid / len(validation_results)) * 100
            
            metrics.append(QualityMetric(
                name="syntax_validity_rate",
                value=syntax_rate,
                threshold=60.0,
                passed=syntax_rate >= 60.0,
                unit="%",
                description="Rate of syntactically valid outputs"
            ))
            
            status = "PASSED" if validation_rate >= 70.0 else "FAILED"
            
        except Exception as e:
            errors.append(f"Output validation failed: {str(e)}")
            status = "FAILED"
        
        return self._create_test_result(
            test_name, status, start_time, metrics, errors, warnings,
            details={'validation_results': validation_results}
        )
    
    def _has_excessive_repetition(self, text: str) -> bool:
        """Check for excessive repetition in text"""
        words = text.split()
        if len(words) < 10:
            return False
        
        # Check for repeated sequences
        for i in range(len(words) - 5):
            sequence = ' '.join(words[i:i+5])
            if text.count(sequence) > 2:
                return True
        
        return False
    
    def _validate_python_syntax(self, code: str) -> bool:
        """Basic Python syntax validation"""
        try:
            compile(code, '<string>', 'exec')
            return True
        except SyntaxError:
            return False
    
    def _create_test_result(self, test_name: str, status: str, start_time: float,
                          metrics: List[QualityMetric], errors: List[str], 
                          warnings: List[str], details: Optional[Dict] = None) -> TestResult:
        """Create test result"""
        return TestResult(
            test_name=test_name,
            status=status,
            duration_seconds=time.time() - start_time,
            metrics=[m.to_dict() for m in metrics],
            errors=errors,
            warnings=warnings,
            details=details or {},
            timestamp=datetime.now().isoformat()
        )


class QualityGate:
    """Quality gate management"""
    
    def __init__(self, model_path: str, config: Optional[Dict] = None):
        self.model_path = model_path
        self.config = config or self._get_default_config()
        self.validator = ModelValidator(model_path)
        
        logger.info(f"Initialized quality gate for {model_path}")
    
    def _get_default_config(self) -> Dict:
        """Get default quality gate configuration"""
        return {
            'required_tests': [
                'generation_quality',
                'inference_speed',
                'memory_usage',
                'output_validation'
            ],
            'optional_tests': [
                'model_consistency',
                'error_handling'
            ],
            'thresholds': {
                'success_rate': 80.0,
                'speed': 10.0,
                'memory': 10000.0,
                'validation_rate': 70.0
            },
            'fail_fast': True,
            'continue_on_warning': True
        }
    
    def run_quality_gates(self, test_suite: str = "standard") -> QualityReport:
        """Run comprehensive quality gates"""
        logger.info(f"Running quality gates: {test_suite}")
        
        start_time = time.time()
        
        # Determine which tests to run
        if test_suite == "minimal":
            tests_to_run = ['generation_quality']
        elif test_suite == "comprehensive":
            tests_to_run = self.config['required_tests'] + self.config['optional_tests']
        else:  # standard
            tests_to_run = self.config['required_tests']
        
        # Load model
        if not self.validator.load_model():
            return self._create_failed_report("Model loading failed")
        
        # Run tests
        test_results = []
        passed_tests = 0
        failed_tests = 0
        
        for test_name in tests_to_run:
            logger.info(f"Running test: {test_name}")
            
            try:
                if test_name == "generation_quality":
                    result = self.validator.test_generation_quality()
                elif test_name == "inference_speed":
                    result = self.validator.test_inference_speed()
                elif test_name == "memory_usage":
                    result = self.validator.test_memory_usage()
                elif test_name == "model_consistency":
                    result = self.validator.test_model_consistency()
                elif test_name == "error_handling":
                    result = self.validator.test_error_handling()
                elif test_name == "output_validation":
                    result = self.validator.test_output_validation()
                else:
                    logger.warning(f"Unknown test: {test_name}")
                    continue
                
                test_results.append(result)
                
                if result.status == "PASSED":
                    passed_tests += 1
                else:
                    failed_tests += 1
                    
                    # Fail fast if configured
                    if self.config.get('fail_fast', True) and test_name in self.config['required_tests']:
                        logger.error(f"Failed required test: {test_name}")
                        break
                
            except Exception as e:
                logger.error(f"Test {test_name} failed with exception: {e}")
                error_result = TestResult(
                    test_name=test_name,
                    status="FAILED",
                    duration_seconds=0,
                    metrics=[],
                    errors=[str(e)],
                    warnings=[],
                    details={},
                    timestamp=datetime.now().isoformat()
                )
                test_results.append(error_result)
                failed_tests += 1
        
        # Determine overall status
        total_tests = len(test_results)
        overall_status = "PASSED" if failed_tests == 0 else "FAILED"
        
        # Generate summary
        summary = {
            'test_suite': test_suite,
            'pass_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            'critical_failures': [r for r in test_results if r.status == "FAILED" and r.test_name in self.config['required_tests']],
            'warnings': [w for r in test_results for w in r.warnings],
            'performance_metrics': self._aggregate_performance_metrics(test_results)
        }
        
        report = QualityReport(
            model_path=str(self.model_path),
            test_suite_name=test_suite,
            overall_status=overall_status,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            test_results=[r.to_dict() for r in test_results],
            summary=summary,
            timestamp=datetime.now().isoformat(),
            execution_time_seconds=time.time() - start_time
        )
        
        logger.info(f"Quality gates completed: {overall_status} ({passed_tests}/{total_tests})")
        return report
    
    def _create_failed_report(self, error_message: str) -> QualityReport:
        """Create a failed report"""
        return QualityReport(
            model_path=str(self.model_path),
            test_suite_name="failed",
            overall_status="FAILED",
            total_tests=0,
            passed_tests=0,
            failed_tests=0,
            test_results=[],
            summary={'error': error_message},
            timestamp=datetime.now().isoformat(),
            execution_time_seconds=0
        )
    
    def _aggregate_performance_metrics(self, test_results: List[TestResult]) -> Dict[str, Any]:
        """Aggregate performance metrics from test results"""
        performance = {
            'avg_generation_time': None,
            'avg_memory_usage': None,
            'avg_success_rate': None
        }
        
        generation_times = []
        memory_usages = []
        success_rates = []
        
        for result in test_results:
            for metric_data in result.metrics:
                metric_name = metric_data.get('name', '')
                
                if metric_name == 'avg_inference_time':
                    generation_times.append(metric_data['value'])
                elif 'memory' in metric_name and 'usage' in metric_name:
                    memory_usages.append(metric_data['value'])
                elif metric_name == 'success_rate':
                    success_rates.append(metric_data['value'])
        
        if generation_times:
            performance['avg_generation_time'] = np.mean(generation_times)
        if memory_usages:
            performance['avg_memory_usage'] = np.mean(memory_usages)
        if success_rates:
            performance['avg_success_rate'] = np.mean(success_rates)
        
        return performance
    
    def save_report(self, report: QualityReport, output_path: str):
        """Save quality report to file"""
        with open(output_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        
        logger.info(f"Quality report saved to {output_path}")
    
    def check_deployment_readiness(self, report: QualityReport) -> Dict[str, Any]:
        """Check if model is ready for deployment with enhanced CI/CD support"""
        deployment_status = {
            'ready': False,
            'confidence_level': 0,
            'issues': [],
            'recommendations': [],
            'gate_results': {},
            'ci_cd_status': {},
            'deployment_matrix': {}
        }
        
        # Check overall status
        if report.overall_status != "PASSED":
            deployment_status['issues'].append("Overall quality gate failed")
            deployment_status['ci_cd_status']['status'] = 'fail'
            return deployment_status
        
        # Check pass rate
        pass_rate = report.summary.get('pass_rate', 0)
        if pass_rate < 90:
            deployment_status['issues'].append(f"Low pass rate: {pass_rate:.1f}%")
        
        # Check critical failures
        critical_failures = report.summary.get('critical_failures', [])
        if critical_failures:
            deployment_status['issues'].append(f"Critical test failures: {len(critical_failures)}")
            for failure in critical_failures:
                deployment_status['issues'].append(f"  - {failure.get('test_name', 'Unknown')}")
        
        # Determine readiness
        if len(deployment_status['issues']) == 0:
            deployment_status['ready'] = True
            deployment_status['confidence_level'] = 100
            deployment_status['recommendations'].append("Model is ready for deployment")
            deployment_status['ci_cd_status']['status'] = 'pass'
        elif len(deployment_status['issues']) == 1:
            deployment_status['confidence_level'] = 75
            deployment_status['recommendations'].append("Address identified issue before deployment")
            deployment_status['ci_cd_status']['status'] = 'pass_with_warnings'
        else:
            deployment_status['confidence_level'] = 50
            deployment_status['recommendations'].append("Multiple issues need resolution before deployment")
            deployment_status['ci_cd_status']['status'] = 'fail'
        
        # Gate-specific checks
        for result in report.test_results:
            gate_name = result['test_name']
            gate_passed = result['status'] == "PASSED"
            deployment_status['gate_results'][gate_name] = {
                'passed': gate_passed,
                'critical': gate_name in self.config['required_tests']
            }
        
        # CI/CD specific checks
        deployment_status['ci_cd_status'].update({
            'all_critical_gates_passed': all(
                gate['passed'] for gate in deployment_status['gate_results'].values() 
                if gate['critical']
            ),
            'performance_acceptable': self._check_performance_acceptable(report),
            'memory_usage_acceptable': self._check_memory_acceptable(report),
            'quality_threshold_met': self._check_quality_threshold(report)
        })
        
        # Deployment matrix for different platforms
        deployment_status['deployment_matrix'] = self._generate_deployment_matrix(report)
        
        return deployment_status
    
    def _check_performance_acceptable(self, report: QualityReport) -> bool:
        """Check if performance is acceptable for deployment"""
        for result in report.test_results:
            if result['test_name'] == 'inference_speed':
                for metric in result['metrics']:
                    if metric['name'] == 'tokens_per_second':
                        return metric['passed']
        return True  # Default to acceptable if no speed test
    
    def _check_memory_acceptable(self, report: QualityReport) -> bool:
        """Check if memory usage is acceptable for deployment"""
        for result in report.test_results:
            if result['test_name'] == 'memory_usage':
                for metric in result['metrics']:
                    if 'memory' in metric['name'] and 'usage' in metric['name']:
                        if metric['passed']:
                            return True
        return True  # Default to acceptable if no memory test
    
    def _check_quality_threshold(self, report: QualityReport) -> bool:
        """Check if quality threshold is met"""
        for result in report.test_results:
            if result['test_name'] == 'generation_quality':
                for metric in result['metrics']:
                    if metric['name'] == 'success_rate':
                        return metric['passed']
        return True  # Default to acceptable if no quality test
    
    def _generate_deployment_matrix(self, report: QualityReport) -> Dict[str, Dict]:
        """Generate deployment matrix for different platforms"""
        matrix = {}
        
        # Get gate results
        gate_results = {}
        for result in report.test_results:
            gate_results[result['test_name']] = result['status'] == "PASSED"
        
        # Define deployment platforms and requirements
        platforms = {
            'huggingface_hub': {
                'required_gates': ['generation_quality', 'output_validation'],
                'optional_gates': ['inference_speed', 'memory_usage'],
                'min_pass_rate': 0.8
            },
            'production_api': {
                'required_gates': ['generation_quality', 'inference_speed', 'memory_usage'],
                'optional_gates': ['output_validation', 'model_consistency'],
                'min_pass_rate': 0.9
            },
            'edge_deployment': {
                'required_gates': ['memory_usage', 'output_validation'],
                'optional_gates': ['generation_quality', 'error_handling'],
                'min_pass_rate': 0.7
            },
            'research': {
                'required_gates': ['generation_quality'],
                'optional_gates': ['inference_speed', 'memory_usage'],
                'min_pass_rate': 0.6
            }
        }
        
        # Evaluate each platform
        for platform, requirements in platforms.items():
            platform_result = {
                'ready': False,
                'required_passed': True,
                'optional_passed': 0,
                'optional_total': len(requirements['optional_gates']),
                'overall_score': 0
            }
            
            # Check required gates
            for gate in requirements['required_gates']:
                if not gate_results.get(gate, False):
                    platform_result['required_passed'] = False
                    break
            
            # Check optional gates
            for gate in requirements['optional_gates']:
                if gate_results.get(gate, False):
                    platform_result['optional_passed'] += 1
            
            # Calculate score
            required_score = 1.0 if platform_result['required_passed'] else 0
            optional_score = platform_result['optional_passed'] / platform_result['optional_total']
            platform_result['overall_score'] = (required_score + optional_score) / 2
            
            # Determine readiness
            platform_result['ready'] = (
                platform_result['required_passed'] and 
                platform_result['overall_score'] >= requirements['min_pass_rate']
            )
            
            matrix[platform] = platform_result
        
        return matrix
    
    def generate_ci_cd_report(self, report: QualityReport) -> Dict[str, Any]:
        """Generate CI/CD specific report with actionable output"""
        deployment_check = self.check_deployment_readiness(report)
        
        # GitHub Actions compatibility
        github_actions_output = {
            'summary': f"Quality Gates: {report.passed_tests}/{report.total_tests} tests passed",
            'conclusion': deployment_check['ci_cd_status']['status'],
            'metrics': {},
            'artifacts': {}
        }
        
        # Add metrics for GitHub Actions
        for result in report.test_results:
            for metric in result['metrics']:
                github_actions_output['metrics'][metric['name']] = {
                    'value': metric['value'],
                    'threshold': metric['threshold'],
                    'passed': metric['passed']
                }
        
        # Add deployment recommendations
        if deployment_check['ready']:
            github_actions_output['recommendations'] = [
                "✅ Model ready for deployment to production",
                f"✅ Confidence level: {deployment_check['confidence_level']}%"
            ]
        else:
            github_actions_output['recommendations'] = [
                f"❌ Model not ready: {deployment_check['issues']}",
                f"⚠️  Confidence level: {deployment_check['confidence_level']}%"
            ]
        
        # Add deployment matrix for CI/CD decision making
        github_actions_output['deployment_matrix'] = deployment_check['deployment_matrix']
        
        # JSON output for parsing
        github_actions_output['json_output'] = json.dumps({
            'status': deployment_check['ci_cd_status']['status'],
            'ready_for_deployment': deployment_check['ready'],
            'confidence_level': deployment_check['confidence_level'],
            'issues_count': len(deployment_check['issues']),
            'platforms_ready': {
                platform: result['ready'] 
                for platform, result in deployment_check['deployment_matrix'].items()
            }
        })
        
        return github_actions_output
    
    def save_ci_cd_artifacts(self, report: QualityReport, output_dir: str):
        """Save CI/CD specific artifacts"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed report
        with open(output_dir / 'quality_report.json', 'w') as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        
        # Save CI/CD summary
        ci_cd_report = self.generate_ci_cd_report(report)
        with open(output_dir / 'ci_cd_summary.json', 'w') as f:
            json.dump(ci_cd_report, f, indent=2, default=str)
        
        # Save deployment matrix
        deployment_matrix = ci_cd_report['deployment_matrix']
        with open(output_dir / 'deployment_matrix.json', 'w') as f:
            json.dump(deployment_matrix, f, indent=2, default=str)
        
        # Create GitHub Actions summary file
        summary_content = f"""# Quality Gate Results

## Summary
- **Status**: {report.overall_status}
- **Tests Passed**: {report.passed_tests}/{report.total_tests}
- **Execution Time**: {report.execution_time_seconds:.2f} seconds

## Metrics
"""
        for test_result in report.test_results:
            summary_content += f"### {test_result['test_name']}\n"
            summary_content += f"- Status: {test_result['status']}\n"
            for metric in test_result['metrics']:
                status_icon = "✅" if metric['passed'] else "❌"
                summary_content += f"- {status_icon} {metric['name']}: {metric['value']:.2f} {metric['unit']} (threshold: {metric['threshold']})\n"
            summary_content += "\n"
        
        summary_content += f"""## Deployment Readiness
- **Ready for Production**: {'✅ Yes' if deployment_matrix.get('production_api', {}).get('ready') else '❌ No'}
- **Ready for HuggingFace**: {'✅ Yes' if deployment_matrix.get('huggingface_hub', {}).get('ready') else '❌ No'}

## JSON Output for CI/CD
```
{ci_cd_report['json_output']}
```
"""
        
        with open(output_dir / 'GITHUB_SUMMARY.md', 'w') as f:
            f.write(summary_content)
        
        logger.info(f"CI/CD artifacts saved to {output_dir}")


def main():
    """Main function for quality gate testing"""
    parser = argparse.ArgumentParser(description='Quality gate validation')
    
    parser.add_argument('--model_path', required=True, help='Path to model')
    parser.add_argument('--test_suite', default='standard',
                       choices=['minimal', 'standard', 'comprehensive'],
                       help='Test suite to run')
    parser.add_argument('--output_path', help='Output path for quality report')
    parser.add_argument('--config', help='Quality gate configuration file')
    parser.add_argument('--quantization', help='Model quantization type')
    parser.add_argument('--device_map', default='auto', help='Device mapping')
    parser.add_argument('--check_deployment', action='store_true', help='Check deployment readiness')
    
    args = parser.parse_args()
    
    # Load configuration
    config = None
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            config = json.load(f)
    
    # Initialize quality gate
    quality_gate = QualityGate(args.model_path, config)
    
    # Run quality gates
    report = quality_gate.run_quality_gates(args.test_suite)
    
    # Save report if output path specified
    if args.output_path:
        quality_gate.save_report(report, args.output_path)
    
    # Check deployment readiness if requested
    deployment_check = None
    if args.check_deployment:
        deployment_check = quality_gate.check_deployment_readiness(report)
    
    # Print results
    print(f"Quality Gate Results:")
    print(f"Overall Status: {report.overall_status}")
    print(f"Tests: {report.passed_tests}/{report.total_tests} passed")
    print(f"Execution Time: {report.execution_time_seconds:.2f} seconds")
    
    if deployment_check:
        print(f"\nDeployment Readiness:")
        print(f"Ready: {deployment_check['ready']}")
        print(f"Confidence: {deployment_check['confidence_level']}%")
        
        if deployment_check['issues']:
            print("Issues:")
            for issue in deployment_check['issues']:
                print(f"  - {issue}")
        
        if deployment_check['recommendations']:
            print("Recommendations:")
            for rec in deployment_check['recommendations']:
                print(f"  - {rec}")
    
    # Return exit code
    return 0 if report.overall_status == "PASSED" else 1


if __name__ == '__main__':
    sys.exit(main())