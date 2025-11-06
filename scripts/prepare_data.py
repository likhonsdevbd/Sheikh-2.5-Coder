#!/usr/bin/env python3
"""
Data Preparation Script for Sheikh-2.5-Coder

This script implements the data preparation pipeline as described in the
comprehensive data preparation strategy document.
"""

import os
import sys
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any
from datasets import Dataset
import argparse

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from preprocessing.stack_v2_processor import StackV2Processor
from preprocessing.instruction_processor import InstructionProcessor
from preprocessing.code_comment_processor import CodeCommentProcessor
from synthetic_generation.web_dev_synthetic import WebDevSyntheticGenerator
from quality_filtering.advanced_deduplicator import AdvancedDeduplicator
from optimization.on_device_optimizer import OnDeviceOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_preparation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataPreparationPipeline:
    """Main data preparation pipeline for Sheikh-2.5-Coder"""
    
    def __init__(self, config_path: str):
        """Initialize the pipeline with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.setup_directories()
        self.initialize_components()
    
    def setup_directories(self):
        """Create necessary directories"""
        directories = [
            self.config['output']['raw_data'],
            self.config['output']['processed_data'],
            self.config['output']['tokenized_data'],
            self.config['output']['logs'],
            self.config['output']['reports']
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        logger.info("Directory structure created")
    
    def initialize_components(self):
        """Initialize pipeline components"""
        self.stack_processor = StackV2Processor(self.config)
        self.instruction_processor = InstructionProcessor(self.config)
        self.code_comment_processor = CodeCommentProcessor(self.config)
        self.synthetic_generator = WebDevSyntheticGenerator(self.config)
        self.deduplicator = AdvancedDeduplicator(self.config)
        self.optimizer = OnDeviceOptimizer(self.config)
        
        logger.info("Pipeline components initialized")
    
    def phase_1_dataset_acquisition(self):
        """Phase 1: Dataset Acquisition and Initial Preprocessing (Weeks 1-4)"""
        logger.info("Starting Phase 1: Dataset Acquisition")
        
        try:
            # Process Stack v2 dataset
            logger.info("Processing Stack v2 dataset...")
            stack_data = self.stack_processor.download_and_process()
            
            # Process instruction datasets
            logger.info("Processing instruction datasets...")
            instruction_data = self.instruction_processor.process_datasets()
            
            # Process code-comment pairs
            logger.info("Processing code-comment pairs...")
            code_comment_data = self.code_comment_processor.process_datasets()
            
            # Save phase 1 results
            self.save_phase_results('phase_1', {
                'stack_v2': stack_data,
                'instructions': instruction_data,
                'code_comments': code_comment_data
            })
            
            logger.info("Phase 1 completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Phase 1 failed: {str(e)}")
            return False
    
    def phase_2_quality_filtering(self):
        """Phase 2: Quality Filtering and Deduplication (Weeks 5-8)"""
        logger.info("Starting Phase 2: Quality Filtering and Deduplication")
        
        try:
            # Load phase 1 results
            phase_results = self.load_phase_results('phase_1')
            
            # Apply quality filters
            logger.info("Applying quality filters...")
            filtered_data = {}
            
            for dataset_name, dataset in phase_results.items():
                logger.info(f"Filtering {dataset_name}...")
                filtered_dataset = self.apply_quality_filters(dataset)
                filtered_data[dataset_name] = filtered_dataset
            
            # Apply deduplication
            logger.info("Applying deduplication...")
            deduped_data = {}
            
            for dataset_name, dataset in filtered_data.items():
                logger.info(f"Deduplicating {dataset_name}...")
                deduped_dataset = self.deduplicator.deduplicate_dataset(dataset)
                deduped_data[dataset_name] = deduped_dataset
            
            # Save phase 2 results
            self.save_phase_results('phase_2', deduped_data)
            
            # Validate quality targets
            validation_results = self.validate_quality_targets(deduped_data)
            self.save_validation_report('phase_2', validation_results)
            
            logger.info("Phase 2 completed successfully")
            return validation_results['all_targets_met']
            
        except Exception as e:
            logger.error(f"Phase 2 failed: {str(e)}")
            return False
    
    def phase_3_synthetic_generation(self):
        """Phase 3: Synthetic Data Generation (Weeks 9-12)"""
        logger.info("Starting Phase 3: Synthetic Data Generation")
        
        try:
            # Generate synthetic data for XML/MDX/JavaScript
            logger.info("Generating synthetic data...")
            synthetic_data = self.synthetic_generator.generate_all_domains()
            
            # Validate synthetic data quality
            logger.info("Validating synthetic data quality...")
            quality_results = self.validate_synthetic_quality(synthetic_data)
            
            # Save phase 3 results
            self.save_phase_results('phase_3', {
                'synthetic_data': synthetic_data,
                'quality_report': quality_results
            })
            
            logger.info("Phase 3 completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Phase 3 failed: {str(e)}")
            return False
    
    def phase_4_integration_testing(self):
        """Phase 4: Integration Testing and Benchmarking (Weeks 13-16)"""
        logger.info("Starting Phase 4: Integration Testing and Benchmarking")
        
        try:
            # Load all previous phase results
            all_data = {}
            for phase in ['phase_1', 'phase_2', 'phase_3']:
                phase_data = self.load_phase_results(phase)
                all_data.update(phase_data)
            
            # Integrate datasets
            logger.info("Integrating all datasets...")
            integrated_dataset = self.integrate_datasets(all_data)
            
            # Run benchmarking tests
            logger.info("Running MMLU benchmarking...")
            benchmark_results = self.run_mmlu_benchmarking(integrated_dataset)
            
            # Save phase 4 results
            self.save_phase_results('phase_4', {
                'integrated_dataset': integrated_dataset,
                'benchmark_results': benchmark_results
            })
            
            logger.info("Phase 4 completed successfully")
            return benchmark_results['performance_targets_met']
            
        except Exception as e:
            logger.error(f"Phase 4 failed: {str(e)}")
            return False
    
    def phase_5_final_optimization(self):
        """Phase 5: Final Training and Optimization (Weeks 17-20)"""
        logger.info("Starting Phase 5: Final Training and Optimization")
        
        try:
            # Load integrated dataset
            integrated_data = self.load_phase_results('phase_4')['integrated_dataset']
            
            # Apply on-device optimizations
            logger.info("Applying on-device optimizations...")
            optimized_data = self.optimizer.optimize_for_deployment(integrated_data)
            
            # Prepare final training dataset
            logger.info("Preparing final training dataset...")
            final_dataset = self.prepare_training_dataset(optimated_data)
            
            # Save final results
            self.save_final_dataset(final_dataset)
            
            # Generate summary report
            self.generate_summary_report()
            
            logger.info("Phase 5 completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Phase 5 failed: {str(e)}")
            return False
    
    def run_complete_pipeline(self):
        """Run the complete data preparation pipeline"""
        logger.info("Starting Sheikh-2.5-Coder Data Preparation Pipeline")
        
        phases = [
            ("Dataset Acquisition", self.phase_1_dataset_acquisition),
            ("Quality Filtering", self.phase_2_quality_filtering),
            ("Synthetic Generation", self.phase_3_synthetic_generation),
            ("Integration Testing", self.phase_4_integration_testing),
            ("Final Optimization", self.phase_5_final_optimization)
        ]
        
        results = {}
        
        for phase_name, phase_func in phases:
            logger.info(f"Executing {phase_name}...")
            success = phase_func()
            results[phase_name] = success
            
            if not success:
                logger.error(f"Pipeline failed at {phase_name}")
                break
        
        # Generate final report
        self.generate_pipeline_report(results)
        
        return all(results.values())
    
    def save_phase_results(self, phase_name: str, results: Dict[str, Any]):
        """Save results from a pipeline phase"""
        phase_dir = Path(self.config['output']['processed_data']) / phase_name
        phase_dir.mkdir(parents=True, exist_ok=True)
        
        for dataset_name, dataset in results.items():
            if isinstance(dataset, Dataset):
                dataset.save_to_disk(str(phase_dir / dataset_name))
            else:
                with open(phase_dir / f"{dataset_name}.json", 'w') as f:
                    json.dump(dataset, f, indent=2)
        
        logger.info(f"Phase {phase_name} results saved to {phase_dir}")
    
    def load_phase_results(self, phase_name: str) -> Dict[str, Any]:
        """Load results from a pipeline phase"""
        phase_dir = Path(self.config['output']['processed_data']) / phase_name
        
        if not phase_dir.exists():
            raise FileNotFoundError(f"Phase {phase_name} results not found")
        
        results = {}
        for file_path in phase_dir.glob('*'):
            if file_path.is_dir():
                # Load dataset
                dataset = Dataset.load_from_disk(str(file_path))
                results[file_path.name] = dataset
            elif file_path.suffix == '.json':
                # Load JSON data
                with open(file_path, 'r') as f:
                    data = json.load(f)
                results[file_path.stem] = data
        
        logger.info(f"Loaded Phase {phase_name} results")
        return results

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Data Preparation Pipeline for Sheikh-2.5-Coder')
    parser.add_argument('--config', type=str, default='configs/data_prep_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--phase', type=str, choices=['1', '2', '3', '4', '5', 'all'],
                        default='all', help='Which phase(s) to run')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = DataPreparationPipeline(args.config)
    
    # Run specified phases
    if args.phase == 'all':
        success = pipeline.run_complete_pipeline()
    else:
        phase_methods = {
            '1': pipeline.phase_1_dataset_acquisition,
            '2': pipeline.phase_2_quality_filtering,
            '3': pipeline.phase_3_synthetic_generation,
            '4': pipeline.phase_4_integration_testing,
            '5': pipeline.phase_5_final_optimization
        }
        success = phase_methods[args.phase]()
    
    if success:
        logger.info("Data preparation pipeline completed successfully!")
        sys.exit(0)
    else:
        logger.error("Data preparation pipeline failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
