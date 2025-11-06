#!/usr/bin/env python3
"""
Performance Benchmark Evaluation for Sheikh-2.5-Coder
Evaluates model performance across inference speed, memory usage, context scaling, and threading
"""

import os
import sys
import json
import yaml
import argparse
import logging
import torch
import numpy as np
import psutil
import time
import gc
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add parent directories to path
sys.path.append('../')
sys.path.append('../../')

class PerformanceBenchmark:
    """Performance benchmark evaluation framework"""
    
    def __init__(self, config_path: str, model_path: str, output_path: str, run_id: str):
        """Initialize performance benchmark"""
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
        
        # Performance test settings
        self.perf_config = self.config.get('performance_benchmark', {})
        
        self.logger.info(f"Performance Benchmark initialized for run: {run_id}")
    
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
            'performance_benchmark': {
                'inference_speed': {
                    'iterations': 50,
                    'warmup_iterations': 10,
                    'batch_sizes': [1, 2, 4],
                    'max_tokens': [32, 64, 128]
                },
                'memory_profiling': {
                    'quantization_levels': ['fp16', 'int8', 'int4']
                },
                'context_scaling': {
                    'context_lengths': [512, 1024, 2048]
                },
                'multi_threading': {
                    'thread_counts': [1, 2, 4]
                }
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for this benchmark"""
        log_file = self.output_path / f"performance_{self.run_id}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(f'PerformanceBenchmark_{self.run_id}')
    
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
            
            # Load model with performance-optimized settings
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map='auto',
                trust_remote_code=True,
                use_cache=True
            )
            
            self.logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            return False
    
    def get_hardware_info(self) -> Dict:
        """Get comprehensive hardware information"""
        info = {
            'cpu': {
                'cores_physical': psutil.cpu_count(logical=False),
                'cores_logical': psutil.cpu_count(logical=True),
                'frequency_current': psutil.cpu_freq().current if psutil.cpu_freq() else 0,
                'frequency_max': psutil.cpu_freq().max if psutil.cpu_freq() else 0,
                'usage_percent': psutil.cpu_percent(interval=1)
            },
            'memory': {
                'total_gb': psutil.virtual_memory().total / (1024**3),
                'available_gb': psutil.virtual_memory().available / (1024**3),
                'used_percent': psutil.virtual_memory().percent
            },
            'torch': {
                'version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
                'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
        }
        
        # Add GPU information if available
        if torch.cuda.is_available():
            info['gpu'] = {}
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                info['gpu'][f'gpu_{i}'] = {
                    'name': gpu_name,
                    'memory_total_gb': gpu_memory,
                    'memory_allocated_gb': torch.cuda.memory_allocated(i) / (1024**3),
                    'memory_reserved_gb': torch.cuda.memory_reserved(i) / (1024**3)
                }
        
        return info
    
    def benchmark_inference_speed(self) -> Dict:
        """Benchmark inference speed across different configurations"""
        self.logger.info("Running inference speed benchmarks...")
        
        speed_config = self.perf_config.get('inference_speed', {})
        iterations = speed_config.get('iterations', 50)
        warmup_iterations = speed_config.get('warmup_iterations', 10)
        batch_sizes = speed_config.get('batch_sizes', [1, 2, 4])
        max_tokens_list = speed_config.get('max_tokens', [32, 64, 128])
        
        # Test prompts for different scenarios
        test_prompts = [
            "def fibonacci(n):",
            "class Calculator:",
            "import numpy as np",
            "function calculateSum(numbers) {",
            "const fetchData = async () => {"
        ]
        
        results = {
            'iterations': iterations,
            'warmup_iterations': warmup_iterations,
            'batch_size_results': {},
            'token_length_results': {},
            'prompt_type_results': {},
            'overall_metrics': {}
        }
        
        # Benchmark different batch sizes
        for batch_size in batch_sizes:
            batch_results = []
            
            self.logger.info(f"Benchmarking batch size: {batch_size}")
            
            for max_tokens in max_tokens_list:
                # Prepare batch of prompts
                batch_prompts = test_prompts[:batch_size] if len(test_prompts) >= batch_size else test_prompts * (batch_size // len(test_prompts) + 1)
                batch_prompts = batch_prompts[:batch_size]
                
                # Warmup
                for _ in range(warmup_iterations):
                    self._generate_batch(batch_prompts[:1], max_tokens)
                
                # Actual benchmark
                latencies = []
                for _ in range(iterations):
                    start_time = time.time()
                    self._generate_batch(batch_prompts, max_tokens)
                    latency = time.time() - start_time
                    latencies.append(latency)
                
                # Calculate metrics
                avg_latency = np.mean(latencies)
                p95_latency = np.percentile(latencies, 95)
                tokens_per_second = (batch_size * max_tokens) / avg_latency
                
                batch_results.append({
                    'max_tokens': max_tokens,
                    'avg_latency_ms': avg_latency * 1000,
                    'p95_latency_ms': p95_latency * 1000,
                    'tokens_per_second': tokens_per_second,
                    'throughput_samples_per_second': batch_size / avg_latency
                })
            
            results['batch_size_results'][f'batch_{batch_size}'] = batch_results
        
        # Benchmark different prompt types
        for i, prompt in enumerate(test_prompts[:3]):  # Limit to 3 for time
            latencies = []
            
            self.logger.info(f"Benchmarking prompt type {i+1}: {prompt[:30]}...")
            
            for _ in range(iterations):
                start_time = time.time()
                self._generate_single(prompt, 64)
                latency = time.time() - start_time
                latencies.append(latency)
            
            results['prompt_type_results'][f'prompt_{i+1}'] = {
                'prompt_preview': prompt[:50],
                'avg_latency_ms': np.mean(latencies) * 1000,
                'p95_latency_ms': np.percentile(latencies, 95) * 1000,
                'tokens_per_second': 64 / np.mean(latencies)
            }
        
        # Calculate overall metrics
        all_latencies = []
        all_tokens_per_second = []
        
        for batch_data in results['batch_size_results'].values():
            for data in batch_data:
                all_latencies.append(data['avg_latency_ms'])
                all_tokens_per_second.append(data['tokens_per_second'])
        
        results['overall_metrics'] = {
            'avg_latency_ms': np.mean(all_latencies),
            'p95_latency_ms': np.percentile(all_latencies, 95),
            'avg_tokens_per_second': np.mean(all_tokens_per_second),
            'max_tokens_per_second': np.max(all_tokens_per_second),
            'min_tokens_per_second': np.min(all_tokens_per_second)
        }
        
        return results
    
    def _generate_single(self, prompt: str, max_tokens: int) -> str:
        """Generate completion for a single prompt"""
        try:
            inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            return generated_text
        except Exception as e:
            self.logger.error(f"Single generation failed: {e}")
            return ""
    
    def _generate_batch(self, prompts: List[str], max_tokens: int):
        """Generate completions for a batch of prompts"""
        try:
            # Tokenize batch
            inputs = self.tokenizer(
                prompts,
                return_tensors='pt',
                padding=True,
                truncation=True
            ).to(self.model.device)
            
            # Generate batch
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            return outputs
            
        except Exception as e:
            self.logger.error(f"Batch generation failed: {e}")
            return None
    
    def benchmark_memory_usage(self) -> Dict:
        """Benchmark memory usage across different scenarios"""
        self.logger.info("Running memory usage benchmarks...")
        
        memory_config = self.perf_config.get('memory_profiling', {})
        quantization_levels = memory_config.get('quantization_levels', ['fp16', 'int8', 'int4'])
        
        results = {
            'system_memory': {},
            'gpu_memory': {},
            'model_memory_footprint': {}
        }
        
        # System memory baseline
        results['system_memory']['baseline'] = {
            'total_gb': psutil.virtual_memory().total / (1024**3),
            'available_gb': psutil.virtual_memory().available / (1024**3),
            'used_gb': (psutil.virtual_memory().total - psutil.virtual_memory().available) / (1024**3),
            'used_percent': psutil.virtual_memory().percent
        }
        
        # GPU memory baseline if available
        if torch.cuda.is_available():
            gpu_baseline = {}
            for i in range(torch.cuda.device_count()):
                gpu_baseline[f'gpu_{i}'] = {
                    'allocated_gb': torch.cuda.memory_allocated(i) / (1024**3),
                    'reserved_gb': torch.cuda.memory_reserved(i) / (1024**3)
                }
            results['gpu_memory']['baseline'] = gpu_baseline
        
        # Memory usage during inference
        inference_memory = self._measure_inference_memory()
        results['system_memory']['during_inference'] = inference_memory['system']
        results['gpu_memory']['during_inference'] = inference_memory.get('gpu', {})
        
        # Model parameter count and memory estimate
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Calculate model size in different precisions
        precision_sizes = {}
        for dtype in ['float16', 'float32', 'int8', 'int4']:
            bytes_per_param = self._get_bytes_per_param(dtype)
            model_size_gb = (total_params * bytes_per_param) / (1024**3)
            precision_sizes[dtype] = model_size_gb
        
        results['model_memory_footprint'] = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'precision_sizes_gb': precision_sizes,
            'actual_model_size_gb': self._estimate_model_size()
        }
        
        return results
    
    def _get_bytes_per_param(self, dtype: str) -> int:
        """Get bytes per parameter for different data types"""
        dtype_map = {
            'float16': 2,
            'float32': 4,
            'int8': 1,
            'int4': 0.5
        }
        return dtype_map.get(dtype, 4)
    
    def _estimate_model_size(self) -> float:
        """Estimate actual model size"""
        try:
            # Get model state dict size
            model_size_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
            return model_size_bytes / (1024**3)
        except Exception:
            return 0.0
    
    def _measure_inference_memory(self) -> Dict:
        """Measure memory usage during inference"""
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Measure memory before
        before_system = psutil.virtual_memory().used / (1024**3)
        before_gpu = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                before_gpu[f'gpu_{i}'] = torch.cuda.memory_allocated(i) / (1024**3)
        
        # Run inference
        test_prompts = ["def fibonacci(n):", "class Calculator:"]
        for _ in range(10):
            self._generate_batch(test_prompts, 50)
        
        # Measure memory after
        after_system = psutil.virtual_memory().used / (1024**3)
        after_gpu = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                after_gpu[f'gpu_{i}'] = torch.cuda.memory_allocated(i) / (1024**3)
        
        return {
            'system': {
                'before_gb': before_system,
                'after_gb': after_system,
                'delta_gb': after_system - before_system
            },
            'gpu': {k: after_gpu.get(k, 0) - before_gpu.get(k, 0) 
                   for k in before_gpu.keys()}
        }
    
    def benchmark_context_scaling(self) -> Dict:
        """Benchmark performance across different context lengths"""
        self.logger.info("Running context scaling benchmarks...")
        
        context_config = self.perf_config.get('context_scaling', {})
        context_lengths = context_config.get('context_lengths', [512, 1024, 2048, 4096])
        
        results = {
            'context_length_results': {},
            'scaling_analysis': {}
        }
        
        # Create prompts of different lengths
        base_prompt = "def fibonacci(n):\n    if n <= 1:\n        return n\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)\n\n"
        test_prompt = "class Calculator:\n    def add(self, a, b):\n        return a + b\n\n    def subtract(self, a, b):\n        return a - b\n"
        
        for context_length in context_lengths:
            # Create context prompt
            context_prompt = (base_prompt * (context_length // len(base_prompt) + 1))[:context_length - len(test_prompt)]
            full_prompt = context_prompt + test_prompt
            
            self.logger.info(f"Benchmarking context length: {context_length}")
            
            # Benchmark generation speed
            latencies = []
            for _ in range(10):  # Fewer iterations for longer contexts
                start_time = time.time()
                try:
                    self._generate_single(full_prompt, 50)
                    latency = time.time() - start_time
                    latencies.append(latency)
                except Exception as e:
                    self.logger.warning(f"Context length {context_length} failed: {e}")
                    break
            
            if latencies:
                results['context_length_results'][str(context_length)] = {
                    'avg_latency_ms': np.mean(latencies) * 1000,
                    'p95_latency_ms': np.percentile(latencies, 95) * 1000,
                    'context_length': context_length,
                    'prompt_length': len(full_prompt),
                    'success_rate': len(latencies) / 10
                }
        
        # Analyze scaling behavior
        if len(results['context_length_results']) > 1:
            context_data = [(int(k), v['avg_latency_ms']) 
                           for k, v in results['context_length_results'].items()]
            context_data.sort()
            
            if len(context_data) >= 2:
                # Calculate scaling factor
                first_ctx, first_lat = context_data[0]
                last_ctx, last_lat = context_data[-1]
                
                scaling_factor = (last_lat / first_lat) / (last_ctx / first_ctx)
                
                results['scaling_analysis'] = {
                    'scaling_factor': scaling_factor,
                    'linear_scaling': scaling_factor < 1.2,  # Close to linear
                    'super_linear': scaling_factor > 2.0,
                    'context_growth_ratio': last_ctx / first_ctx,
                    'latency_growth_ratio': last_lat / first_lat
                }
        
        return results
    
    def benchmark_multi_threading(self) -> Dict:
        """Benchmark multi-threading performance"""
        self.logger.info("Running multi-threading benchmarks...")
        
        threading_config = self.perf_config.get('multi_threading', {})
        thread_counts = threading_config.get('thread_counts', [1, 2, 4, 8])
        
        results = {
            'threading_results': {},
            'scalability_analysis': {}
        }
        
        test_prompt = "def fibonacci(n):"
        num_tasks = 20  # Number of concurrent tasks
        target_tokens = 50
        
        # Single-threaded baseline
        latencies = []
        for _ in range(num_tasks):
            start_time = time.time()
            self._generate_single(test_prompt, target_tokens)
            latency = time.time() - start_time
            latencies.append(latency)
        
        single_thread_time = np.mean(latencies)
        baseline_throughput = num_tasks / sum(latencies)
        
        # Multi-threaded tests
        for thread_count in thread_counts:
            self.logger.info(f"Benchmarking with {thread_count} threads")
            
            def worker_task():
                start_time = time.time()
                self._generate_single(test_prompt, target_tokens)
                return time.time() - start_time
            
            # Run with different thread counts
            with ThreadPoolExecutor(max_workers=thread_count) as executor:
                start_time = time.time()
                futures = [executor.submit(worker_task) for _ in range(num_tasks)]
                thread_latencies = [future.result() for future in as_completed(futures)]
            
            total_time = time.time() - start_time
            avg_latency = np.mean(thread_latencies)
            throughput = num_tasks / total_time
            speedup = baseline_throughput / (throughput / thread_count)
            
            results['threading_results'][f'{thread_count}_threads'] = {
                'thread_count': thread_count,
                'avg_latency_ms': avg_latency * 1000,
                'total_time_ms': total_time * 1000,
                'throughput_tasks_per_second': throughput,
                'speedup_ratio': speedup,
                'efficiency': speedup / thread_count
            }
        
        # Analyze scalability
        threading_data = [(int(k.split('_')[0]), v['speedup_ratio']) 
                         for k, v in results['threading_results'].items()]
        threading_data.sort()
        
        if len(threading_data) > 1:
            results['scalability_analysis'] = {
                'max_speedup': max(v for _, v in threading_data),
                'scalable_threads': max(k for k, v in threading_data if v > k * 0.7),  # >70% efficiency
                'diminishing_returns': self._check_diminishing_returns(threading_data)
            }
        
        return results
    
    def _check_diminishing_returns(self, data: List[Tuple[int, float]]) -> bool:
        """Check if performance shows diminishing returns"""
        if len(data) < 3:
            return False
        
        # Check if speedup growth is decreasing
        speedup_growth = []
        for i in range(1, len(data)):
            prev_threads, prev_speed = data[i-1]
            curr_threads, curr_speed = data[i]
            
            thread_increase = curr_threads - prev_threads
            speedup_increase = curr_speed - prev_speed
            
            if thread_increase > 0:
                speedup_growth.append(speedup_increase / thread_increase)
        
        # Diminishing returns if growth is decreasing
        return len(speedup_growth) > 1 and speedup_growth[-1] < speedup_growth[-2] * 0.8
    
    def run_comprehensive_benchmark(self) -> Dict:
        """Run comprehensive performance benchmarking"""
        start_time = time.time()
        
        if not self.load_model():
            return {'status': 'failed', 'error': 'Model loading failed'}
        
        try:
            self.logger.info("Starting comprehensive performance benchmarking...")
            
            # Get hardware information
            hardware_info = self.get_hardware_info()
            
            # Run all benchmarks
            benchmark_results = {}
            
            # Inference speed benchmark
            try:
                self.logger.info("Running inference speed benchmark...")
                benchmark_results['inference_speed'] = self.benchmark_inference_speed()
                benchmark_results['inference_speed']['tokens_per_second'] = benchmark_results['inference_speed']['overall_metrics']['avg_tokens_per_second']
            except Exception as e:
                self.logger.error(f"Inference speed benchmark failed: {e}")
                benchmark_results['inference_speed'] = {'error': str(e), 'tokens_per_second': 0}
            
            # Memory usage benchmark
            try:
                self.logger.info("Running memory usage benchmark...")
                benchmark_results['memory_usage'] = self.benchmark_memory_usage()
                # Estimate memory usage in MB
                gpu_memory = benchmark_results['memory_usage'].get('gpu_memory', {})
                gpu_total = 0
                if 'during_inference' in gpu_memory:
                    for gpu_data in gpu_memory['during_inference'].values():
                        gpu_total += gpu_data
                benchmark_results['memory_usage']['memory_usage_mb'] = gpu_total * 1024
            except Exception as e:
                self.logger.error(f"Memory usage benchmark failed: {e}")
                benchmark_results['memory_usage'] = {'error': str(e), 'memory_usage_mb': 0}
            
            # Context scaling benchmark
            try:
                self.logger.info("Running context scaling benchmark...")
                benchmark_results['context_scaling'] = self.benchmark_context_scaling()
                benchmark_results['context_scaling']['max_context_length'] = max(
                    int(k) for k in benchmark_results['context_scaling'].get('context_length_results', {}).keys()
                ) if benchmark_results['context_scaling'].get('context_length_results') else 0
            except Exception as e:
                self.logger.error(f"Context scaling benchmark failed: {e}")
                benchmark_results['context_scaling'] = {'error': str(e), 'max_context_length': 0}
            
            # Multi-threading benchmark
            try:
                self.logger.info("Running multi-threading benchmark...")
                benchmark_results['multi_threading'] = self.benchmark_multi_threading()
            except Exception as e:
                self.logger.error(f"Multi-threading benchmark failed: {e}")
                benchmark_results['multi_threading'] = {'error': str(e)}
            
            evaluation_time = time.time() - start_time
            
            # Compile comprehensive results
            final_results = {
                'status': 'completed',
                'benchmark': 'Performance',
                'hardware_info': hardware_info,
                'evaluation_time_seconds': evaluation_time,
                'inference_speed': benchmark_results.get('inference_speed', {}),
                'memory_usage': benchmark_results.get('memory_usage', {}),
                'context_scaling': benchmark_results.get('context_scaling', {}),
                'multi_threading': benchmark_results.get('multi_threading', {}),
                'tokens_per_second': benchmark_results.get('inference_speed', {}).get('overall_metrics', {}).get('avg_tokens_per_second', 0),
                'memory_usage_mb': benchmark_results.get('memory_usage', {}).get('memory_usage_mb', 0),
                'max_context_length': benchmark_results.get('context_scaling', {}).get('max_context_length', 0),
                'performance_summary': self._create_performance_summary(benchmark_results)
            }
            
            self.logger.info("Performance benchmarking completed successfully")
            
            # Save results
            self._save_results(final_results)
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Performance benchmarking failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'evaluation_time_seconds': time.time() - start_time
            }
    
    def _create_performance_summary(self, results: Dict) -> Dict:
        """Create performance summary"""
        summary = {
            'overall_grade': 'Unknown',
            'strengths': [],
            'weaknesses': [],
            'recommendations': []
        }
        
        # Analyze inference speed
        speed_results = results.get('inference_speed', {})
        if 'overall_metrics' in speed_results:
            tokens_per_sec = speed_results['overall_metrics'].get('avg_tokens_per_second', 0)
            
            if tokens_per_sec > 50:
                summary['strengths'].append("Excellent inference speed")
                summary['overall_grade'] = 'A' if summary['overall_grade'] == 'Unknown' else summary['overall_grade']
            elif tokens_per_sec > 30:
                summary['strengths'].append("Good inference speed")
                summary['overall_grade'] = 'B' if summary['overall_grade'] in ['Unknown', 'C'] else summary['overall_grade']
            elif tokens_per_sec < 20:
                summary['weaknesses'].append("Slow inference speed")
                summary['recommendations'].append("Consider model optimization or quantization")
        
        # Analyze memory efficiency
        memory_results = results.get('memory_usage', {})
        if 'model_memory_footprint' in memory_results:
            model_size = memory_results['model_memory_footprint'].get('actual_model_size_gb', 0)
            if model_size > 10:
                summary['weaknesses'].append("Large memory footprint")
                summary['recommendations'].append("Consider model quantization or pruning")
            else:
                summary['strengths'].append("Efficient memory usage")
        
        # Analyze scaling
        context_results = results.get('context_scaling', {})
        if 'scaling_analysis' in context_results:
            scaling_factor = context_results['scaling_analysis'].get('scaling_factor', 1)
            if scaling_factor < 1.2:
                summary['strengths'].append("Excellent context scaling")
            else:
                summary['weaknesses'].append("Poor context scaling")
                summary['recommendations'].append("Optimize attention mechanisms for longer contexts")
        
        # Determine overall grade
        if summary['overall_grade'] == 'Unknown':
            summary['overall_grade'] = 'C'
        
        return summary
    
    def _save_results(self, results: Dict):
        """Save performance benchmark results"""
        # Save detailed results as JSON
        results_file = self.output_path / f"performance_results_{self.run_id}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary metrics as CSV
        import pandas as pd
        
        # Create summary table
        summary_data = []
        
        # Inference speed summary
        speed_data = results.get('inference_speed', {}).get('overall_metrics', {})
        if speed_data:
            summary_data.append({
                'metric': 'Tokens per Second',
                'value': speed_data.get('avg_tokens_per_second', 0),
                'unit': 'tokens/sec'
            })
            summary_data.append({
                'metric': 'Average Latency',
                'value': speed_data.get('avg_latency_ms', 0),
                'unit': 'ms'
            })
        
        # Memory summary
        memory_data = results.get('memory_usage', {})
        if memory_data.get('model_memory_footprint'):
            model_size = memory_data['model_memory_footprint'].get('actual_model_size_gb', 0)
            summary_data.append({
                'metric': 'Model Size',
                'value': model_size,
                'unit': 'GB'
            })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            csv_file = self.output_path / f"performance_summary_{self.run_id}.csv"
            df.to_csv(csv_file, index=False)
        
        self.logger.info(f"Performance results saved to {self.output_path}")


def main():
    """Main performance benchmark function"""
    parser = argparse.ArgumentParser(description='Performance Benchmark Evaluation')
    
    parser.add_argument('--model_path', required=True, help='Path to model directory')
    parser.add_argument('--config', required=True, help='Path to evaluation configuration')
    parser.add_argument('--output_path', required=True, help='Output directory for results')
    parser.add_argument('--run_id', required=True, help='Unique run identifier')
    
    args = parser.parse_args()
    
    # Initialize benchmark
    benchmark = PerformanceBenchmark(
        config_path=args.config,
        model_path=args.model_path,
        output_path=args.output_path,
        run_id=args.run_id
    )
    
    # Run benchmark
    try:
        results = benchmark.run_comprehensive_benchmark()
        
        if results.get('status') == 'completed':
            print(f"\nPerformance Benchmark Results:")
            
            # Print key metrics
            tokens_per_sec = results.get('tokens_per_second', 0)
            print(f"Inference Speed: {tokens_per_sec:.1f} tokens/second")
            
            memory_mb = results.get('memory_usage_mb', 0)
            print(f"Memory Usage: {memory_mb:.1f} MB")
            
            max_context = results.get('max_context_length', 0)
            print(f"Max Context Length: {max_context}")
            
            # Print hardware info
            hardware = results.get('hardware_info', {})
            if hardware.get('torch', {}).get('cuda_available'):
                print(f"GPU: {hardware['torch'].get('cuda_version', 'N/A')}")
            
            print(f"Evaluation Time: {results.get('evaluation_time_seconds', 0):.1f}s")
            
            # Print summary
            perf_summary = results.get('performance_summary', {})
            if perf_summary:
                print(f"\nOverall Grade: {perf_summary.get('overall_grade', 'N/A')}")
                if perf_summary.get('strengths'):
                    print("Strengths:", ", ".join(perf_summary['strengths']))
                if perf_summary.get('recommendations'):
                    print("Recommendations:", ", ".join(perf_summary['recommendations']))
            
            return 0
        else:
            print(f"Benchmark failed: {results.get('error', 'Unknown error')}")
            return 1
            
    except KeyboardInterrupt:
        print("Benchmark interrupted by user")
        return 1
    except Exception as e:
        print(f"Benchmark failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())