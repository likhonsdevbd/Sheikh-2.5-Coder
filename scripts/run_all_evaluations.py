#!/usr/bin/env python3
"""
Quick Reference Script for Running All Sheikh-2.5-Coder Evaluations
Runs comprehensive evaluation suite with minimal configuration
"""

import os
import sys
import argparse
import subprocess
import json
from datetime import datetime
from pathlib import Path

def run_command(cmd, description=""):
    """Run a command and handle output"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ SUCCESS")
        if result.stdout:
            print("Output:", result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ FAILED")
        print("Error:", e.stderr if e.stderr else str(e))
        return False

def main():
    parser = argparse.ArgumentParser(description='Run All Sheikh-2.5-Coder Evaluations')
    
    parser.add_argument('--model_path', required=True, help='Path to model directory')
    parser.add_argument('--config', default='scripts/evaluation_config.yaml', help='Path to evaluation config')
    parser.add_argument('--output_base', default='./evaluation_results', help='Base output directory')
    parser.add_argument('--run_id', help='Run ID (auto-generated if not provided)')
    parser.add_argument('--individual', action='store_true', help='Run individual benchmarks separately')
    parser.add_argument('--skip_regression', action='store_true', help='Skip regression testing')
    parser.add_argument('--quick', action='store_true', help='Quick evaluation (fewer samples)')
    
    args = parser.parse_args()
    
    # Generate run ID if not provided
    if not args.run_id:
        args.run_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create output directories
    base_dir = Path(args.output_base)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Sheikh-2.5-Coder Evaluation Suite")
    print(f"Model: {args.model_path}")
    print(f"Run ID: {args.run_id}")
    print(f"Output Base: {base_dir}")
    print(f"Individual Mode: {'Yes' if args.individual else 'No'}")
    print(f"Quick Mode: {'Yes' if args.quick else 'No'}")
    
    success_count = 0
    total_count = 6 if not args.skip_regression else 5
    
    if args.individual:
        # Run individual benchmarks
        evaluations = [
            {
                'name': 'MMLU Code Evaluation',
                'script': 'scripts/mmlu_evaluation.py',
                'output_dir': base_dir / 'mmlu',
            },
            {
                'name': 'HumanEval Coding Tasks',
                'script': 'scripts/humaneval_evaluation.py',
                'output_dir': base_dir / 'humaneval',
            },
            {
                'name': 'Web Development Tests',
                'script': 'scripts/web_dev_tests.py',
                'output_dir': base_dir / 'webdev',
            },
            {
                'name': 'Performance Benchmark',
                'script': 'scripts/performance_benchmark.py',
                'output_dir': base_dir / 'performance',
            },
            {
                'name': 'Code Quality Tests',
                'script': 'scripts/code_quality_tests.py',
                'output_dir': base_dir / 'quality',
            }
        ]
        
        if not args.skip_regression:
            evaluations.append({
                'name': 'Regression Testing',
                'script': 'scripts/regression_testing.py',
                'output_dir': base_dir / 'regression',
            })
        
        # Run each evaluation
        for i, eval_config in enumerate(evaluations, 1):
            print(f"\n[{i}/{total_count}] Running {eval_config['name']}...")
            
            cmd = [
                sys.executable,
                eval_config['script'],
                '--model_path', args.model_path,
                '--config', args.config,
                '--output_path', str(eval_config['output_dir']),
                '--run_id', f"{args.run_id}_{eval_config['name'].lower().replace(' ', '_')}"
            ]
            
            if run_command(cmd, eval_config['name']):
                success_count += 1
            else:
                print(f"⚠️  {eval_config['name']} failed, continuing...")
        
        # Try to run main orchestrator for comprehensive report
        if success_count >= 3:  # Need at least 3 successful evaluations
            print(f"\n[{total_count + 1}/{total_count + 1}] Running Main Orchestrator...")
            
            orchestrator_cmd = [
                sys.executable,
                'scripts/evaluate_model.py',
                '--model_path', args.model_path,
                '--config', args.config,
                '--output_path', str(base_dir / 'comprehensive'),
                '--run_id', args.run_id
            ]
            
            run_command(orchestrator_cmd, 'Comprehensive Report Generation')
        
    else:
        # Run main orchestrator (recommended approach)
        cmd = [
            sys.executable,
            'scripts/evaluate_model.py',
            '--model_path', args.model_path,
            '--config', args.config,
            '--output_path', str(base_dir),
            '--run_id', args.run_id
        ]
        
        if run_command(cmd, 'Comprehensive Evaluation Suite'):
            success_count = total_count
        else:
            print("❌ Main orchestrator failed, trying individual evaluations...")
            # Fall back to individual evaluations
            return main()  # Recursive call with individual mode
    
    # Generate summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Successful: {success_count}/{total_count}")
    print(f"Success Rate: {success_count/total_count*100:.1f}%")
    
    if success_count == total_count:
        print("🎉 All evaluations completed successfully!")
    elif success_count >= total_count // 2:
        print("⚠️  Most evaluations completed successfully")
    else:
        print("❌ Many evaluations failed")
    
    print(f"\nResults saved to: {base_dir}")
    
    # Look for report files
    report_files = list(base_dir.glob("**/*report*.md")) + list(base_dir.glob("**/*summary*.md"))
    if report_files:
        print(f"\nReport files generated:")
        for report in report_files[:5]:  # Show first 5 reports
            print(f"  - {report}")
    
    # Exit with appropriate code
    return 0 if success_count >= total_count // 2 else 1

if __name__ == '__main__':
    sys.exit(main())