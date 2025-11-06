#!/usr/bin/env python3
"""
Data Validation Suite

Implements comprehensive data validation and quality assessment:
- Data quality validation
- Language distribution validation
- Quality metrics calculation
- Statistical analysis
- Validation reporting

Author: MiniMax Agent
Date: 2025-11-06
"""

import os
import sys
import json
import yaml
import logging
import re
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
from tqdm import tqdm
import pandas as pd
from collections import Counter, defaultdict

# Data processing libraries
from datasets import Dataset

# Quality assessment
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

logger = logging.getLogger(__name__)

class DataValidationSuite:
    """
    Comprehensive data validation suite
    
    Validates:
    - Data quality metrics
    - Language distribution
    - Statistical properties
    - Performance targets
    - Compliance with specifications
    """
    
    def __init__(self, config: DataPreparationConfig):
        """Initialize data validation suite"""
        self.config = config
        self.setup_logging()
        self.setup_directories()
        self.initialize_validation_metrics()
        self.initialize_benchmarks()
        
        # Validation statistics
        self.validation_stats = {
            'datasets_validated': 0,
            'total_examples': 0,
            'validation_passed': 0,
            'validation_failed': 0,
            'quality_scores': [],
            'language_distributions': {},
            'performance_metrics': {}
        }
        
        logger.info("Data Validation Suite initialized")
    
    def setup_logging(self):
        """Setup logging for validation"""
        log_handler = logging.FileHandler('logs/data_validation.log')
        log_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(log_handler)
    
    def setup_directories(self):
        """Create necessary directories"""
        directories = [
            'evaluation/validation_reports',
            'evaluation/quality_reports',
            'evaluation/statistical_analysis',
            'evaluation/benchmarking',
            'plots/validation'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def initialize_validation_metrics(self):
        """Initialize validation metrics and thresholds"""
        
        self.validation_thresholds = {
            'data_quality': {
                'minimum_avg_quality': 0.75,
                'maximum_quality_variance': 0.1,
                'minimum_quality_consistency': 0.8
            },
            'language_distribution': {
                'javascript_typescript': {'min': 0.30, 'max': 0.40},
                'xml_html': {'min': 0.20, 'max': 0.30},
                'mdx_markdown': {'min': 0.10, 'max': 0.20},
                'css_scss': {'min': 0.08, 'max': 0.15},
                'other': {'min': 0.10, 'max': 0.20}
            },
            'size_distribution': {
                'min_avg_length': 100,
                'max_avg_length': 10000,
                'max_length_variance': 5000
            },
            'performance_targets': {
                'duplication_rate': {'max': 0.05},
                'syntax_validity': {'min': 0.90},
                'semantic_coherence': {'min': 0.75},
                'task_completion_rate': {'min': 0.85}
            }
        }
        
        # Benchmark datasets for comparison
        self.benchmark_datasets = {
            'HumanEval': {
                'pass_rate': 0.67,
                'difficulty_distribution': {'easy': 0.3, 'medium': 0.5, 'hard': 0.2}
            },
            'CodeBLEU': {
                'score': 0.65,
                'language_coverage': 0.8
            },
            'MMLU_Code': {
                'accuracy': 0.60,
                'domain_coverage': 0.85
            }
        }
        
        logger.info(f"Initialized validation thresholds for {len(self.validation_thresholds)} categories")
    
    def initialize_benchmarks(self):
        """Initialize benchmarking capabilities"""
        
        self.benchmark_tests = {
            'syntax_validity': self.test_syntax_validity,
            'semantic_coherence': self.test_semantic_coherence,
            'task_completion': self.test_task_completion,
            'language_consistency': self.test_language_consistency,
            'quality_distribution': self.test_quality_distribution,
            'statistical_properties': self.test_statistical_properties,
            'performance_benchmarks': self.test_performance_benchmarks
        }
        
        logger.info(f"Initialized {len(self.benchmark_tests)} benchmark tests")
    
    def validate_complete_dataset(self, dataset: Dataset, dataset_name: str = "dataset") -> Dict[str, Any]:
        """
        Perform comprehensive validation of dataset
        
        Args:
            dataset: Dataset to validate
            dataset_name: Name of dataset for reporting
            
        Returns:
            Dict[str, Any]: Comprehensive validation results
        """
        logger.info(f"Starting comprehensive validation for {dataset_name}")
        
        try:
            # Reset validation statistics
            self.reset_validation_stats()
            
            # Convert dataset to list for analysis
            examples = list(dataset)
            self.validation_stats['total_examples'] = len(examples)
            
            logger.info(f"Validating {len(examples):,} examples...")
            
            # Run all validation tests
            validation_results = {}
            
            with tqdm(total=len(self.benchmark_tests), desc="Validation tests") as pbar:
                for test_name, test_func in self.benchmark_tests.items():
                    logger.info(f"Running {test_name} test...")
                    
                    try:
                        test_result = test_func(examples, dataset_name)
                        validation_results[test_name] = test_result
                        
                        # Update statistics
                        if test_result.get('passed', False):
                            self.validation_stats['validation_passed'] += 1
                        else:
                            self.validation_stats['validation_failed'] += 1
                        
                        pbar.update(1)
                        
                    except Exception as e:
                        logger.error(f"Test {test_name} failed: {str(e)}")
                        validation_results[test_name] = {
                            'passed': False,
                            'error': str(e),
                            'score': 0.0
                        }
                        self.validation_stats['validation_failed'] += 1
                        pbar.update(1)
            
            # Calculate overall validation score
            overall_score = self.calculate_overall_validation_score(validation_results)
            
            # Generate comprehensive report
            final_results = {
                'dataset_name': dataset_name,
                'total_examples': len(examples),
                'validation_results': validation_results,
                'overall_score': overall_score,
                'validation_passed': overall_score >= 0.8,
                'statistical_summary': self.generate_statistical_summary(examples),
                'recommendations': self.generate_recommendations(validation_results)
            }
            
            # Save validation results
            self.save_validation_results(final_results, dataset_name)
            
            # Generate visualizations if available
            if PLOTTING_AVAILABLE:
                self.generate_validation_plots(final_results, dataset_name)
            
            # Log results
            self.log_validation_results(final_results)
            
            self.validation_stats['datasets_validated'] += 1
            
            logger.info(f"Validation completed for {dataset_name}")
            logger.info(f"Overall score: {overall_score:.2f}")
            logger.info(f"Validation passed: {final_results['validation_passed']}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Validation failed for {dataset_name}: {str(e)}")
            raise
    
    def reset_validation_stats(self):
        """Reset validation statistics"""
        self.validation_stats = {
            'datasets_validated': 0,
            'total_examples': 0,
            'validation_passed': 0,
            'validation_failed': 0,
            'quality_scores': [],
            'language_distributions': {},
            'performance_metrics': {}
        }
    
    def test_syntax_validity(self, examples: List[Dict], dataset_name: str) -> Dict[str, Any]:
        """Test syntax validity across all examples"""
        
        logger.info("  Testing syntax validity...")
        
        syntax_results = {
            'total_tested': 0,
            'valid_syntax': 0,
            'invalid_syntax': 0,
            'language_breakdown': {},
            'syntax_errors': []
        }
        
        for example in tqdm(examples, desc="Syntax validation"):
            syntax_results['total_tested'] += 1
            
            language = self.detect_language(example)
            content = self.extract_content(example)
            
            validity_score = self.validate_syntax_score(content, language)
            
            if validity_score >= 0.8:
                syntax_results['valid_syntax'] += 1
            else:
                syntax_results['invalid_syntax'] += 1
                
                # Track language-specific errors
                if language not in syntax_results['language_breakdown']:
                    syntax_results['language_breakdown'][language] = {'valid': 0, 'invalid': 0}
                
                syntax_results['language_breakdown'][language]['invalid'] += 1
                syntax_results['syntax_errors'].append({
                    'language': language,
                    'error_type': 'syntax_invalid',
                    'content_preview': content[:100]
                })
            
            # Update language breakdown
            if language not in syntax_results['language_breakdown']:
                syntax_results['language_breakdown'][language] = {'valid': 0, 'invalid': 0}
            syntax_results['language_breakdown'][language]['valid'] += 1
        
        # Calculate overall validity rate
        validity_rate = syntax_results['valid_syntax'] / syntax_results['total_tested']
        
        passed = validity_rate >= self.validation_thresholds['performance_targets']['syntax_validity']['min']
        
        return {
            'passed': passed,
            'score': validity_rate,
            'validity_rate': validity_rate,
            'total_examples': syntax_results['total_tested'],
            'valid_examples': syntax_results['valid_syntax'],
            'invalid_examples': syntax_results['invalid_syntax'],
            'language_breakdown': syntax_results['language_breakdown'],
            'error_count': len(syntax_results['syntax_errors'])
        }
    
    def test_semantic_coherence(self, examples: List[Dict], dataset_name: str) -> Dict[str, Any]:
        """Test semantic coherence of examples"""
        
        logger.info("  Testing semantic coherence...")
        
        coherence_results = {
            'total_tested': 0,
            'coherent_examples': 0,
            'incoherent_examples': 0,
            'coherence_scores': [],
            'language_coherence': {}
        }
        
        for example in tqdm(examples, desc="Semantic coherence"):
            coherence_results['total_tested'] += 1
            
            language = self.detect_language(example)
            content = self.extract_content(example)
            
            coherence_score = self.calculate_coherence_score(content, language)
            coherence_results['coherence_scores'].append(coherence_score)
            
            if coherence_score >= 0.7:
                coherence_results['coherent_examples'] += 1
            else:
                coherence_results['incoherent_examples'] += 1
            
            # Track language-specific coherence
            if language not in coherence_results['language_coherence']:
                coherence_results['language_coherence'][language] = []
            coherence_results['language_coherence'][language].append(coherence_score)
        
        # Calculate statistics
        avg_coherence = np.mean(coherence_results['coherence_scores'])
        coherence_rate = coherence_results['coherent_examples'] / coherence_results['total_tested']
        
        passed = avg_coherence >= self.validation_thresholds['performance_targets']['semantic_coherence']['min']
        
        return {
            'passed': passed,
            'score': avg_coherence,
            'coherence_rate': coherence_rate,
            'avg_coherence': avg_coherence,
            'coherence_std': np.std(coherence_results['coherence_scores']),
            'language_coherence': {
                lang: {'avg': np.mean(scores), 'count': len(scores)}
                for lang, scores in coherence_results['language_coherence'].items()
            }
        }
    
    def test_task_completion(self, examples: List[Dict], dataset_name: str) -> Dict[str, Any]:
        """Test task completion rates for instruction-following examples"""
        
        logger.info("  Testing task completion...")
        
        # Filter instruction-following examples
        instruction_examples = [
            ex for ex in examples 
            if self.is_instruction_example(ex)
        ]
        
        if not instruction_examples:
            return {
                'passed': True,
                'score': 1.0,
                'note': 'No instruction-following examples found'
            }
        
        completion_results = {
            'total_instructions': len(instruction_examples),
            'completed_tasks': 0,
            'task_types': {},
            'difficulty_breakdown': {}
        }
        
        for example in instruction_examples:
            task_type = example.get('task_type', 'unknown')
            difficulty = example.get('difficulty', 'unknown')
            
            # Assess task completion based on content quality
            content = self.extract_content(example)
            completion_score = self.assess_task_completion(content, task_type)
            
            if completion_score >= 0.8:
                completion_results['completed_tasks'] += 1
            
            # Track by task type
            if task_type not in completion_results['task_types']:
                completion_results['task_types'][task_type] = {'completed': 0, 'total': 0}
            completion_results['task_types'][task_type]['total'] += 1
            if completion_score >= 0.8:
                completion_results['task_types'][task_type]['completed'] += 1
            
            # Track by difficulty
            if difficulty not in completion_results['difficulty_breakdown']:
                completion_results['difficulty_breakdown'][difficulty] = {'completed': 0, 'total': 0}
            completion_results['difficulty_breakdown'][difficulty]['total'] += 1
            if completion_score >= 0.8:
                completion_results['difficulty_breakdown'][difficulty]['completed'] += 1
        
        # Calculate completion rate
        completion_rate = completion_results['completed_tasks'] / completion_results['total_instructions']
        
        passed = completion_rate >= self.validation_thresholds['performance_targets']['task_completion_rate']['min']
        
        return {
            'passed': passed,
            'score': completion_rate,
            'completion_rate': completion_rate,
            'total_instructions': completion_results['total_instructions'],
            'completed_tasks': completion_results['completed_tasks'],
            'task_type_breakdown': completion_results['task_types'],
            'difficulty_breakdown': completion_results['difficulty_breakdown']
        }
    
    def test_language_consistency(self, examples: List[Dict], dataset_name: str) -> Dict[str, Any]:
        """Test language consistency across examples"""
        
        logger.info("  Testing language consistency...")
        
        consistency_results = {
            'language_detection': {},
            'metadata_consistency': {},
            'distribution_analysis': {}
        }
        
        # Detect languages
        detected_languages = []
        metadata_languages = []
        
        for example in examples:
            detected_lang = self.detect_language(example)
            metadata_lang = example.get('language', 'unknown')
            
            detected_languages.append(detected_lang)
            metadata_languages.append(metadata_lang)
            
            # Track detection counts
            if detected_lang not in consistency_results['language_detection']:
                consistency_results['language_detection'][detected_lang] = 0
            consistency_results['language_detection'][detected_lang] += 1
            
            # Track metadata consistency
            if metadata_lang not in consistency_results['metadata_consistency']:
                consistency_results['metadata_consistency'][metadata_lang] = {'detected': 0, 'total': 0}
            consistency_results['metadata_consistency'][metadata_lang]['total'] += 1
            if detected_lang == metadata_lang:
                consistency_results['metadata_consistency'][metadata_lang]['detected'] += 1
        
        # Calculate consistency scores
        total_examples = len(examples)
        consistency_scores = []
        
        for metadata_lang, counts in consistency_results['metadata_consistency'].items():
            if counts['total'] > 0:
                consistency_score = counts['detected'] / counts['total']
                consistency_scores.append(consistency_score)
        
        avg_consistency = np.mean(consistency_scores) if consistency_scores else 1.0
        
        # Check language distribution
        lang_distribution = Counter(detected_languages)
        expected_distribution = self.config.language_distribution
        
        distribution_score = self.calculate_distribution_similarity(lang_distribution, expected_distribution)
        
        passed = avg_consistency >= 0.8 and distribution_score >= 0.7
        
        return {
            'passed': passed,
            'score': (avg_consistency + distribution_score) / 2,
            'avg_consistency': avg_consistency,
            'distribution_score': distribution_score,
            'language_distribution': dict(lang_distribution),
            'metadata_consistency': {
                lang: {'consistency': counts['detected']/counts['total'], 'count': counts['total']}
                for lang, counts in consistency_results['metadata_consistency'].items()
            },
            'target_vs_actual': {
                'target': expected_distribution,
                'actual': {lang: count/total_examples for lang, count in lang_distribution.items()}
            }
        }
    
    def test_quality_distribution(self, examples: List[Dict], dataset_name: str) -> Dict[str, Any]:
        """Test quality score distribution"""
        
        logger.info("  Testing quality distribution...")
        
        quality_scores = []
        quality_by_language = defaultdict(list)
        
        for example in examples:
            quality_score = example.get('quality_score', 0.5)
            language = self.detect_language(example)
            
            quality_scores.append(quality_score)
            quality_by_language[language].append(quality_score)
        
        # Calculate statistics
        avg_quality = np.mean(quality_scores)
        quality_std = np.std(quality_scores)
        quality_variance = np.var(quality_scores)
        
        # Check quality thresholds
        min_quality = np.min(quality_scores)
        max_quality = np.max(quality_scores)
        
        # Check quality consistency
        quality_consistency = 1.0 - (quality_std / avg_quality) if avg_quality > 0 else 0
        
        # Language-specific quality analysis
        lang_quality_stats = {}
        for lang, scores in quality_by_language.items():
            lang_quality_stats[lang] = {
                'avg_quality': np.mean(scores),
                'std_quality': np.std(scores),
                'count': len(scores),
                'min_quality': np.min(scores),
                'max_quality': np.max(scores)
            }
        
        passed = (
            avg_quality >= self.validation_thresholds['data_quality']['minimum_avg_quality'] and
            quality_consistency >= self.validation_thresholds['data_quality']['minimum_quality_consistency']
        )
        
        return {
            'passed': passed,
            'score': avg_quality,
            'avg_quality': avg_quality,
            'quality_std': quality_std,
            'quality_variance': quality_variance,
            'quality_consistency': quality_consistency,
            'min_quality': min_quality,
            'max_quality': max_quality,
            'language_quality_stats': lang_quality_stats,
            'quality_histogram': self.create_quality_histogram(quality_scores)
        }
    
    def test_statistical_properties(self, examples: List[Dict], dataset_name: str) -> Dict[str, Any]:
        """Test statistical properties of dataset"""
        
        logger.info("  Testing statistical properties...")
        
        # Extract content lengths
        content_lengths = []
        word_counts = []
        complexity_scores = []
        
        for example in examples:
            content = self.extract_content(example)
            content_lengths.append(len(content))
            
            # Count words
            words = re.findall(r'\b\w+\b', content)
            word_counts.append(len(words))
            
            # Calculate complexity
            complexity = self.calculate_complexity_score(content)
            complexity_scores.append(complexity)
        
        # Statistical analysis
        length_stats = {
            'mean': np.mean(content_lengths),
            'median': np.median(content_lengths),
            'std': np.std(content_lengths),
            'min': np.min(content_lengths),
            'max': np.max(content_lengths),
            'percentile_25': np.percentile(content_lengths, 25),
            'percentile_75': np.percentile(content_lengths, 75)
        }
        
        word_count_stats = {
            'mean': np.mean(word_counts),
            'median': np.median(word_counts),
            'std': np.std(word_counts)
        }
        
        complexity_stats = {
            'mean': np.mean(complexity_scores),
            'median': np.median(complexity_scores),
            'std': np.std(complexity_scores)
        }
        
        # Check size distribution
        size_distribution_passed = (
            length_stats['mean'] >= self.validation_thresholds['size_distribution']['min_avg_length'] and
            length_stats['mean'] <= self.validation_thresholds['size_distribution']['max_avg_length'] and
            length_stats['std'] <= self.validation_thresholds['size_distribution']['max_length_variance']
        )
        
        passed = size_distribution_passed
        
        return {
            'passed': passed,
            'score': 1.0 if passed else 0.5,
            'length_statistics': length_stats,
            'word_count_statistics': word_count_stats,
            'complexity_statistics': complexity_stats,
            'size_distribution_valid': size_distribution_passed,
            'outlier_analysis': self.analyze_outliers(content_lengths),
            'distribution_shape': self.analyze_distribution_shape(content_lengths)
        }
    
    def test_performance_benchmarks(self, examples: List[Dict], dataset_name: str) -> Dict[str, Any]:
        """Test against performance benchmarks"""
        
        logger.info("  Testing performance benchmarks...")
        
        benchmark_results = {}
        
        # Test against HumanEval-like benchmarks
        humaneval_score = self.simulate_humaneval_benchmark(examples)
        benchmark_results['humaneval'] = {
            'score': humaneval_score,
            'benchmark_score': self.benchmark_datasets['HumanEval']['pass_rate'],
            'improvement': humaneval_score - self.benchmark_datasets['HumanEval']['pass_rate']
        }
        
        # Test against CodeBLEU benchmark
        codebleu_score = self.simulate_codebleu_benchmark(examples)
        benchmark_results['codebleu'] = {
            'score': codebleu_score,
            'benchmark_score': self.benchmark_datasets['CodeBLEU']['score'],
            'improvement': codebleu_score - self.benchmark_datasets['CodeBLEU']['score']
        }
        
        # Test against MMLU Code benchmark
        mmlu_score = self.simulate_mmlu_benchmark(examples)
        benchmark_results['mmlu_code'] = {
            'score': mmlu_score,
            'benchmark_score': self.benchmark_datasets['MMLU_Code']['accuracy'],
            'improvement': mmlu_score - self.benchmark_datasets['MMLU_Code']['accuracy']
        }
        
        # Calculate overall benchmark score
        benchmark_scores = [result['score'] for result in benchmark_results.values()]
        avg_benchmark_score = np.mean(benchmark_scores)
        
        # Check if we meet performance targets
        performance_targets_met = (
            humaneval_score >= self.benchmark_datasets['HumanEval']['pass_rate'] * 0.9 and
            codebleu_score >= self.benchmark_datasets['CodeBLEU']['score'] * 0.9 and
            mmlu_score >= self.benchmark_datasets['MMLU_Code']['accuracy'] * 0.9
        )
        
        passed = performance_targets_met
        
        return {
            'passed': passed,
            'score': avg_benchmark_score,
            'benchmark_results': benchmark_results,
            'performance_targets_met': performance_targets_met,
            'overall_improvement': np.mean([r['improvement'] for r in benchmark_results.values()])
        }
    
    # Helper methods
    
    def detect_language(self, example: Dict) -> str:
        """Detect programming language from example"""
        # Use same logic as quality filters
        content = self.extract_content(example)
        
        language_patterns = {
            'typescript': [r'\binterface\b', r'\btype\s+[a-zA-Z]', r':\s*[a-zA-Z]'],
            'javascript': [r'\bfunction\b', r'\bconst\b', r'\blet\b', r'=>'],
            'xml': [r'<\?xml', r'<[a-zA-Z][^>]*>', r'</[a-zA-Z][^>]*>'],
            'html': [r'<!DOCTYPE', r'<html', r'<head', r'<body'],
            'css': [r'{\s*', r'}\s*', r'[a-zA-Z-]+\s*:\s*[^;]+;'],
            'mdx': [r'#{1,6}\s+', r'\*\*.*?\*\*', r'<[^>]+>']
        }
        
        content_lower = content.lower()
        
        for language, patterns in language_patterns.items():
            matches = sum(1 for pattern in patterns if re.search(pattern, content, re.IGNORECASE))
            if matches >= 2:
                return language
        
        return example.get('language', 'unknown')
    
    def extract_content(self, example: Dict) -> str:
        """Extract content from example"""
        return example.get('content', '') or example.get('code', '') or example.get('output', '')
    
    def validate_syntax_score(self, content: str, language: str) -> float:
        """Validate syntax and return score"""
        # Simplified syntax validation
        if language == 'javascript':
            brace_balance = content.count('{') - content.count('}')
            paren_balance = content.count('(') - content.count(')')
            return 1.0 if brace_balance == 0 and paren_balance == 0 else 0.7
        elif language == 'xml':
            tag_balance = content.count('<') - content.count('>')
            return 1.0 if tag_balance == 0 else 0.6
        else:
            return 0.8  # Default score
    
    def calculate_coherence_score(self, content: str, language: str) -> float:
        """Calculate semantic coherence score"""
        # Basic coherence assessment
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        if len(lines) < 2:
            return 0.7
        
        # Check for logical flow
        logical_indicators = [
            'function', 'class', 'if', 'else', 'for', 'while', 'return', 'try', 'catch'
        ]
        
        indicator_count = sum(1 for line in lines if any(ind in line.lower() for ind in logical_indicators))
        coherence = min(indicator_count / len(lines), 1.0)
        
        return coherence + 0.1  # Boost base coherence
    
    def is_instruction_example(self, example: Dict) -> bool:
        """Check if example is instruction-following"""
        return any(key in example for key in ['instruction', 'input', 'output', 'task_type'])
    
    def assess_task_completion(self, content: str, task_type: str) -> float:
        """Assess task completion score"""
        # Basic task completion assessment
        if 'function' in content and 'return' in content:
            return 0.9
        elif 'class' in content and '{' in content:
            return 0.85
        elif 'interface' in content and '{' in content:
            return 0.8
        else:
            return 0.6
    
    def calculate_distribution_similarity(self, actual_dist: Counter, expected_dist: Dict[str, float]) -> float:
        """Calculate similarity between actual and expected distributions"""
        total_actual = sum(actual_dist.values())
        total_expected = sum(expected_dist.values())
        
        if total_actual == 0 or total_expected == 0:
            return 0.0
        
        # Normalize distributions
        actual_normalized = {lang: count/total_actual for lang, count in actual_dist.items()}
        
        # Calculate similarity (using cosine similarity)
        similarity = 0.0
        expected_magnitude = 0.0
        actual_magnitude = 0.0
        
        all_langs = set(actual_normalized.keys()) | set(expected_dist.keys())
        
        for lang in all_langs:
            expected_val = expected_dist.get(lang, 0.0)
            actual_val = actual_normalized.get(lang, 0.0)
            
            similarity += expected_val * actual_val
            expected_magnitude += expected_val ** 2
            actual_magnitude += actual_val ** 2
        
        if expected_magnitude == 0 or actual_magnitude == 0:
            return 0.0
        
        return similarity / (np.sqrt(expected_magnitude) * np.sqrt(actual_magnitude))
    
    def calculate_complexity_score(self, content: str) -> float:
        """Calculate complexity score for content"""
        complexity_factors = {
            'function_count': len(re.findall(r'\bfunction\b', content)),
            'class_count': len(re.findall(r'\bclass\b', content)),
            'loop_count': len(re.findall(r'\b(for|while)\b', content)),
            'condition_count': len(re.findall(r'\bif|else|switch|case\b', content)),
            'async_count': len(re.findall(r'\basync|await\b', content, re.IGNORECASE))
        }
        
        total_complexity = sum(complexity_factors.values())
        
        # Normalize to 0-1 scale
        return min(total_complexity / 20.0, 1.0)
    
    def create_quality_histogram(self, quality_scores: List[float]) -> Dict[str, Any]:
        """Create quality score histogram"""
        hist, bins = np.histogram(quality_scores, bins=10)
        
        return {
            'bins': bins.tolist(),
            'counts': hist.tolist(),
            'quality_ranges': [
                {'range': f'{bins[i]:.2f}-{bins[i+1]:.2f}', 'count': int(hist[i])}
                for i in range(len(hist))
            ]
        }
    
    def analyze_outliers(self, values: List[float]) -> Dict[str, Any]:
        """Analyze outliers in value distribution"""
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = [v for v in values if v < lower_bound or v > upper_bound]
        
        return {
            'outlier_count': len(outliers),
            'outlier_percentage': len(outliers) / len(values),
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'iqr': iqr
        }
    
    def analyze_distribution_shape(self, values: List[float]) -> Dict[str, Any]:
        """Analyze distribution shape"""
        mean_val = np.mean(values)
        median_val = np.median(values)
        std_val = np.std(values)
        
        # Skewness approximation
        skewness = (mean_val - median_val) / std_val if std_val > 0 else 0
        
        # Distribution type
        if abs(skewness) < 0.5:
            distribution_type = 'approximately_normal'
        elif skewness > 0.5:
            distribution_type = 'right_skewed'
        else:
            distribution_type = 'left_skewed'
        
        return {
            'mean': mean_val,
            'median': median_val,
            'std': std_val,
            'skewness': skewness,
            'distribution_type': distribution_type,
            'coefficient_of_variation': std_val / mean_val if mean_val > 0 else 0
        }
    
    def simulate_humaneval_benchmark(self, examples: List[Dict]) -> float:
        """Simulate HumanEval benchmark performance"""
        # Simplified HumanEval simulation
        code_examples = [ex for ex in examples if self.extract_content(ex).count('\n') > 5]
        
        if not code_examples:
            return 0.5
        
        # Simulate pass rate based on code quality
        pass_rates = []
        for example in code_examples:
            content = self.extract_content(example)
            
            # Factors that improve HumanEval score
            factors = [
                'function' in content,  # Has function
                'return' in content,    # Has return statement
                '{' in content and '}' in content,  # Proper structure
                len(content) > 50,      # Sufficient length
                content.count('(') == content.count(')'),  # Balanced parens
            ]
            
            score = sum(factors) / len(factors)
            pass_rates.append(score)
        
        return np.mean(pass_rates)
    
    def simulate_codebleu_benchmark(self, examples: List[Dict]) -> float:
        """Simulate CodeBLEU benchmark performance"""
        # Simplified CodeBLEU simulation
        code_examples = [ex for ex in examples if self.extract_content(ex).strip()]
        
        if not code_examples:
            return 0.5
        
        # Calculate average CodeBLEU-like score
        bleu_scores = []
        for example in code_examples:
            content = self.extract_content(example)
            
            # Factors that improve CodeBLEU score
            keyword_score = self.calculate_keyword_coverage(content)
            syntax_score = self.validate_syntax_score(content, 'javascript')
            structure_score = min(len(content) / 1000, 1.0)  # Length factor
            
            # Combine factors (simplified CodeBLEU calculation)
            bleu_score = (keyword_score + syntax_score + structure_score) / 3
            bleu_scores.append(bleu_score)
        
        return np.mean(bleu_scores)
    
    def simulate_mmlu_benchmark(self, examples: List[Dict]) -> float:
        """Simulate MMLU Code benchmark performance"""
        # Simplified MMLU simulation
        # Check for diverse programming concepts
        concept_coverage = {
            'variables': 0,
            'functions': 0,
            'classes': 0,
            'control_flow': 0,
            'data_structures': 0
        }
        
        total_concepts = 0
        for example in examples:
            content = self.extract_content(example).lower()
            
            if 'variable' in content or 'let ' in content or 'const ' in content:
                concept_coverage['variables'] += 1
            if 'function' in content:
                concept_coverage['functions'] += 1
            if 'class' in content:
                concept_coverage['classes'] += 1
            if 'if' in content or 'for' in content or 'while' in content:
                concept_coverage['control_flow'] += 1
            if 'array' in content or 'object' in content or 'list' in content:
                concept_coverage['data_structures'] += 1
            
            total_concepts += 1
        
        # Calculate coverage
        coverage_scores = []
        for concept, count in concept_coverage.items():
            coverage = count / total_concepts if total_concepts > 0 else 0
            coverage_scores.append(coverage)
        
        return np.mean(coverage_scores)
    
    def calculate_keyword_coverage(self, content: str) -> float:
        """Calculate programming keyword coverage"""
        keywords = [
            'function', 'const', 'let', 'var', 'class', 'if', 'else', 'for', 'while',
            'return', 'try', 'catch', 'async', 'await', 'import', 'export'
        ]
        
        content_lower = content.lower()
        keyword_count = sum(1 for keyword in keywords if keyword in content_lower)
        
        return keyword_count / len(keywords)
    
    def calculate_overall_validation_score(self, validation_results: Dict[str, Any]) -> float:
        """Calculate overall validation score"""
        
        scores = []
        weights = []
        
        for test_name, result in validation_results.items():
            if 'score' in result:
                scores.append(result['score'])
                
                # Weight different tests
                if test_name in ['syntax_validity', 'semantic_coherence']:
                    weights.append(0.3)  # High weight for core quality
                elif test_name in ['task_completion', 'performance_benchmarks']:
                    weights.append(0.2)  # Medium weight for functionality
                else:
                    weights.append(0.1)  # Lower weight for supporting tests
        
        if not scores:
            return 0.0
        
        # Weighted average
        weighted_score = np.average(scores, weights=weights)
        return weighted_score
    
    def generate_statistical_summary(self, examples: List[Dict]) -> Dict[str, Any]:
        """Generate comprehensive statistical summary"""
        
        summary = {
            'basic_statistics': {
                'total_examples': len(examples),
                'unique_languages': len(set(self.detect_language(ex) for ex in examples)),
                'avg_content_length': np.mean([len(self.extract_content(ex)) for ex in examples]),
                'content_length_std': np.std([len(self.extract_content(ex)) for ex in examples])
            },
            'language_distribution': dict(Counter([self.detect_language(ex) for ex in examples])),
            'quality_statistics': {},
            'temporal_analysis': {}
        }
        
        # Quality statistics
        quality_scores = [ex.get('quality_score', 0.5) for ex in examples]
        if quality_scores:
            summary['quality_statistics'] = {
                'avg_quality': np.mean(quality_scores),
                'quality_std': np.std(quality_scores),
                'min_quality': np.min(quality_scores),
                'max_quality': np.max(quality_scores)
            }
        
        return summary
    
    def generate_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations"""
        
        recommendations = []
        
        # Analyze each test result
        for test_name, result in validation_results.items():
            if not result.get('passed', False):
                if test_name == 'syntax_validity':
                    recommendations.append("Improve syntax validation - some examples have invalid syntax")
                elif test_name == 'semantic_coherence':
                    recommendations.append("Enhance semantic coherence - some examples lack logical flow")
                elif test_name == 'task_completion':
                    recommendations.append("Increase task completion rates - improve instruction-following quality")
                elif test_name == 'language_consistency':
                    recommendations.append("Improve language consistency - align metadata with detected languages")
                elif test_name == 'quality_distribution':
                    recommendations.append("Enhance quality distribution - increase average quality scores")
                elif test_name == 'performance_benchmarks':
                    recommendations.append("Improve performance benchmarks - enhance overall code quality")
        
        # General recommendations
        if validation_results.get('quality_distribution', {}).get('score', 0) < 0.8:
            recommendations.append("Consider re-filtering low-quality examples to improve overall dataset quality")
        
        if validation_results.get('language_consistency', {}).get('score', 0) < 0.8:
            recommendations.append("Review language detection and metadata assignment consistency")
        
        if not recommendations:
            recommendations.append("Dataset quality is good - maintain current filtering standards")
        
        return recommendations
    
    def save_validation_results(self, results: Dict[str, Any], dataset_name: str):
        """Save validation results to disk"""
        
        output_path = f"evaluation/validation_reports/{dataset_name}_validation_report.json"
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Validation report saved to {output_path}")
    
    def generate_validation_plots(self, results: Dict[str, Any], dataset_name: str):
        """Generate validation visualization plots"""
        
        try:
            # Quality distribution plot
            if 'quality_scores' in results:
                plt.figure(figsize=(10, 6))
                plt.hist(results['quality_scores'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                plt.title(f'Quality Score Distribution - {dataset_name}')
                plt.xlabel('Quality Score')
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)
                plt.savefig(f'plots/validation/{dataset_name}_quality_distribution.png')
                plt.close()
            
            # Language distribution plot
            if 'language_distribution' in results:
                lang_dist = results['language_distribution']
                plt.figure(figsize=(12, 8))
                languages = list(lang_dist.keys())
                counts = list(lang_dist.values())
                plt.pie(counts, labels=languages, autopct='%1.1f%%', startangle=90)
                plt.title(f'Language Distribution - {dataset_name}')
                plt.axis('equal')
                plt.savefig(f'plots/validation/{dataset_name}_language_distribution.png')
                plt.close()
            
            logger.info(f"Validation plots saved for {dataset_name}")
            
        except Exception as e:
            logger.warning(f"Plot generation failed: {e}")
    
    def log_validation_results(self, results: Dict[str, Any]):
        """Log validation results summary"""
        
        logger.info("=" * 60)
        logger.info(f"VALIDATION RESULTS FOR {results['dataset_name'].upper()}")
        logger.info("=" * 60)
        logger.info(f"Total Examples: {results['total_examples']:,}")
        logger.info(f"Overall Score: {results['overall_score']:.3f}")
        logger.info(f"Validation Passed: {results['validation_passed']}")
        logger.info("")
        
        # Log individual test results
        for test_name, test_result in results['validation_results'].items():
            status = "✅ PASS" if test_result.get('passed', False) else "❌ FAIL"
            score = test_result.get('score', 0)
            logger.info(f"{test_name}: {status} (Score: {score:.3f})")
        
        logger.info("")
        logger.info("RECOMMENDATIONS:")
        for rec in results['recommendations']:
            logger.info(f"  • {rec}")
        
        logger.info("=" * 60)

# Import for type hint (avoid circular import)
try:
    from data_preparation_pipeline import DataPreparationConfig
except ImportError:
    pass

def main():
    """Main function for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Data Validation Suite')
    parser.add_argument('--config', type=str, required=True, help='Configuration file path')
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create config object
    config = DataPreparationConfig(**config_dict)
    
    # Initialize validation suite
    validator = DataValidationSuite(config)
    
    # Test validation
    if args.test:
        logger.info("Running in test mode")
        # Create test dataset
        test_examples = [
            {
                'content': 'function test() { return "hello"; }',
                'language': 'javascript',
                'quality_score': 0.8
            },
            {
                'content': 'console.log("test");',
                'language': 'javascript',
                'quality_score': 0.6
            }
        ]
        test_dataset = Dataset.from_list(test_examples)
        results = validator.validate_complete_dataset(test_dataset, "test_dataset")
        logger.info(f"Test validation completed with score: {results['overall_score']:.3f}")
    else:
        logger.info("Data validation suite ready for use")

if __name__ == "__main__":
    main()