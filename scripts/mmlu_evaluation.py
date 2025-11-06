#!/usr/bin/env python3
"""
MMLU Code Benchmark Evaluation for Sheikh-2.5-Coder
Evaluates model performance on Massive Multitask Language Understanding Code subset
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
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add parent directories to path
sys.path.append('../')
sys.path.append('../../')

class MMLEvaluator:
    """MMLU Code benchmark evaluator"""
    
    def __init__(self, config_path: str, model_path: str, output_path: str, run_id: str):
        """Initialize MMLU evaluator"""
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
        self.mmlu_config = self.config.get('mmlu_evaluation', {})
        self.target_accuracy = self.config.get('targets', {}).get('mmlu_code_accuracy', 0.60)
        
        self.logger.info(f"MMLU Evaluator initialized for run: {run_id}")
    
    def _load_config(self) -> Dict:
        """Load evaluation configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Failed to load config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Default configuration if loading fails"""
        return {
            'mmlu_evaluation': {
                'dataset': 'lukaemon/mmlu',
                'subset': 'code',
                'max_test_samples': 100,
                'batch_size': 8
            },
            'targets': {
                'mmlu_code_accuracy': 0.60
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for this evaluator"""
        log_file = self.output_path / f"mmlu_{self.run_id}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(f'MMLEvaluator_{self.run_id}')
    
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
    
    def load_mmlu_dataset(self) -> Dataset:
        """Load MMLU code dataset"""
        try:
            self.logger.info("Loading MMLU dataset...")
            
            dataset_name = self.mmlu_config.get('dataset', 'lukaemon/mmlu')
            subset = self.mmlu_config.get('subset', 'code')
            
            # Load dataset with trust_remote_code
            dataset = load_dataset(dataset_name, subset, trust_remote_code=True)
            
            # Get test split
            test_data = dataset['test']
            
            # Limit samples for computational efficiency
            max_samples = self.mmlu_config.get('max_test_samples', 200)
            if len(test_data) > max_samples:
                test_data = test_data.shuffle(seed=42).select(range(max_samples))
            
            self.logger.info(f"Loaded {len(test_data)} MMLU test samples")
            return test_data
            
        except Exception as e:
            self.logger.error(f"Failed to load MMLU dataset: {str(e)}")
            raise
    
    def format_question(self, item: Dict) -> str:
        """Format MMLU question with choices"""
        question = item['question']
        choices = item['choices']
        
        # Format choices with letters A, B, C, D
        formatted_choices = []
        for i, choice in enumerate(choices):
            letter = chr(65 + i)  # A, B, C, D
            formatted_choices.append(f"({letter}) {choice}")
        
        # Construct full prompt
        prompt = f"{question}\n\n"
        prompt += "\n".join(formatted_choices)
        prompt += "\n\nAnswer:"
        
        return prompt
    
    def generate_answer(self, prompt: str) -> str:
        """Generate answer for a given prompt"""
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
            
            # Generate response
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,  # Deterministic for evaluation
                temperature=0.0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract generated answer
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            return generated_text
            
        except Exception as e:
            self.logger.error(f"Generation failed: {str(e)}")
            return ""
    
    def extract_answer_letter(self, generated_text: str) -> Optional[str]:
        """Extract answer letter (A, B, C, D) from generated text"""
        # Look for single letter answers
        for char in generated_text.strip():
            if char.upper() in ['A', 'B', 'C', 'D']:
                return char.upper()
        
        # If no single letter found, look for patterns like "(A)", "A)", etc.
        import re
        letter_pattern = r'[A-D](?=[)\s]|$|\.)'
        matches = re.findall(letter_pattern, generated_text.upper())
        if matches:
            return matches[0]
        
        return None
    
    def evaluate_sample(self, item: Dict) -> Dict:
        """Evaluate a single MMLU sample"""
        try:
            # Format question
            prompt = self.format_question(item)
            
            # Generate answer
            generated_text = self.generate_answer(prompt)
            
            # Extract answer letter
            predicted_letter = self.extract_answer_letter(generated_text)
            
            # Get correct answer
            correct_letter = chr(65 + item['answer'])
            
            # Determine if correct
            is_correct = predicted_letter == correct_letter if predicted_letter else False
            
            return {
                'question_id': item.get('id', ''),
                'question': item['question'][:100] + '...',  # Truncate for storage
                'correct_answer': correct_letter,
                'predicted_answer': predicted_letter,
                'generated_text': generated_text[:200] + '...' if len(generated_text) > 200 else generated_text,
                'is_correct': is_correct,
                'question_length': len(prompt),
                'generation_length': len(generated_text)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate sample: {str(e)}")
            return {
                'question_id': item.get('id', ''),
                'error': str(e),
                'is_correct': False
            }
    
    def run_evaluation(self) -> Dict:
        """Run complete MMLU evaluation"""
        start_time = time.time()
        
        if not self.load_model():
            return {'status': 'failed', 'error': 'Model loading failed'}
        
        try:
            # Load dataset
            test_data = self.load_mmlu_dataset()
            
            # Evaluate all samples
            results = []
            correct_count = 0
            
            self.logger.info(f"Evaluating {len(test_data)} samples...")
            
            for i, item in enumerate(test_data):
                if i % 10 == 0:
                    self.logger.info(f"Progress: {i}/{len(test_data)} ({i/len(test_data)*100:.1f}%)")
                
                sample_result = self.evaluate_sample(item)
                results.append(sample_result)
                
                if sample_result.get('is_correct', False):
                    correct_count += 1
            
            # Calculate metrics
            total_samples = len(results)
            accuracy = correct_count / total_samples if total_samples > 0 else 0
            
            evaluation_time = time.time() - start_time
            
            # Compile final results
            final_results = {
                'status': 'completed',
                'benchmark': 'MMLU Code',
                'accuracy': accuracy,
                'correct': correct_count,
                'total': total_samples,
                'target_accuracy': self.target_accuracy,
                'target_met': accuracy >= self.target_accuracy,
                'evaluation_time_seconds': evaluation_time,
                'samples_per_second': total_samples / evaluation_time,
                'detailed_results': results[:20],  # Store first 20 for inspection
                'error_count': len([r for r in results if 'error' in r]),
                'question_categories': self._analyze_categories(test_data),
                'performance_breakdown': self._analyze_performance_breakdown(results),
                'prompt_examples': self._extract_prompt_examples(results[:5])
            }
            
            self.logger.info(f"MMLU Evaluation completed: {accuracy:.3f} accuracy ({correct_count}/{total_samples})")
            
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
    
    def _analyze_categories(self, dataset: Dataset) -> Dict:
        """Analyze performance by question categories"""
        categories = {}
        
        # Extract category information if available
        for item in dataset:
            # This is dataset-specific - MMLU might not have explicit categories
            # We'll analyze based on question keywords
            question = item.get('question', '').lower()
            
            # Simple categorization based on keywords
            if any(keyword in question for keyword in ['function', 'class', 'method', 'object']):
                category = 'oop'
            elif any(keyword in question for keyword in ['algorithm', 'complexity', 'sorting', 'search']):
                category = 'algorithms'
            elif any(keyword in question for keyword in ['data structure', 'array', 'list', 'tree']):
                category = 'data_structures'
            elif any(keyword in question for keyword in ['python', 'javascript', 'java', 'code']):
                category = 'programming_languages'
            else:
                category = 'general'
            
            if category not in categories:
                categories[category] = 0
            categories[category] += 1
        
        return categories
    
    def _analyze_performance_breakdown(self, results: List[Dict]) -> Dict:
        """Analyze performance breakdown by various factors"""
        breakdown = {
            'by_question_length': {},
            'by_generation_length': {},
            'error_analysis': {}
        }
        
        correct_results = [r for r in results if r.get('is_correct', False)]
        incorrect_results = [r for r in results if not r.get('is_correct', False)]
        
        # Question length analysis
        if correct_results:
            correct_qlengths = [r.get('question_length', 0) for r in correct_results]
            breakdown['by_question_length']['correct_avg'] = np.mean(correct_qlengths)
        
        if incorrect_results:
            incorrect_qlengths = [r.get('question_length', 0) for r in incorrect_results]
            breakdown['by_question_length']['incorrect_avg'] = np.mean(incorrect_qlengths)
        
        # Error analysis
        errors = [r for r in results if 'error' in r]
        if errors:
            breakdown['error_analysis']['total_errors'] = len(errors)
            breakdown['error_analysis']['error_types'] = {}
            for error_result in errors:
                error_type = type(error_result.get('error', '')).__name__
                breakdown['error_analysis']['error_types'][error_type] = \
                    breakdown['error_analysis']['error_types'].get(error_type, 0) + 1
        
        return breakdown
    
    def _extract_prompt_examples(self, results: List[Dict]) -> List[Dict]:
        """Extract example prompts and responses"""
        examples = []
        
        for result in results:
            if 'question' in result and 'generated_text' in result:
                examples.append({
                    'question': result['question'],
                    'correct_answer': result.get('correct_answer', ''),
                    'predicted_answer': result.get('predicted_answer', ''),
                    'generated_text': result.get('generated_text', ''),
                    'is_correct': result.get('is_correct', False)
                })
        
        return examples
    
    def _save_results(self, results: Dict):
        """Save evaluation results"""
        # Save detailed results as JSON
        results_file = self.output_path / f"mmlu_results_{self.run_id}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary as CSV
        if 'detailed_results' in results:
            df = pd.DataFrame(results['detailed_results'])
            csv_file = self.output_path / f"mmlu_detailed_{self.run_id}.csv"
            df.to_csv(csv_file, index=False)
        
        self.logger.info(f"Results saved to {self.output_path}")


def main():
    """Main MMLU evaluation function"""
    parser = argparse.ArgumentParser(description='MMLU Code Benchmark Evaluation')
    
    parser.add_argument('--model_path', required=True, help='Path to model directory')
    parser.add_argument('--config', required=True, help='Path to evaluation configuration')
    parser.add_argument('--output_path', required=True, help='Output directory for results')
    parser.add_argument('--run_id', required=True, help='Unique run identifier')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = MMLEvaluator(
        config_path=args.config,
        model_path=args.model_path,
        output_path=args.output_path,
        run_id=args.run_id
    )
    
    # Run evaluation
    try:
        results = evaluator.run_evaluation()
        
        if results.get('status') == 'completed':
            print(f"\nMMLU Evaluation Results:")
            print(f"Accuracy: {results.get('accuracy', 0):.3f}")
            print(f"Target: {results.get('target_accuracy', 0):.3f}")
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