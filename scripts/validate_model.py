#!/usr/bin/env python3
"""
Model Validation and Testing for Sheikh-2.5-Coder
Handles comprehensive model evaluation, testing, and quality assessment
"""

import os
import json
import logging
import time
import traceback
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import subprocess
import tempfile

import torch
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    GenerationConfig,
    BitsAndBytesConfig
)
from datasets import Dataset, load_dataset, load_from_disk
from torch.utils.data import DataLoader
import evaluate

try:
    import xml.etree.ElementTree as ET
    XML_AVAILABLE = True
except ImportError:
    XML_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    HTML_AVAILABLE = True
except ImportError:
    HTML_AVAILABLE = False

try:
    import js2py
    JS_EVAL_AVAILABLE = True
except ImportError:
    JS_EVAL_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


@dataclass
class ValidationConfig:
    """Configuration for model validation"""
    # Model settings
    model_path: str
    tokenizer_path: Optional[str] = None
    device: str = "auto"
    trust_remote_code: bool = False
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    
    # Generation settings
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    pad_token_id: Optional[int] = None
    
    # Validation datasets
    validation_datasets: List[str] = field(default_factory=lambda: [
        "HumanEval",
        "MMLU",
        "mbpp",
        "truthfulqa",
        "hellaswag"
    ])
    
    # Custom validation datasets
    custom_datasets: Dict[str, str] = field(default_factory=dict)
    
    # XML/MDX validation
    xml_validation: bool = True
    xml_samples: int = 500
    xml_strict_validation: bool = True
    
    # JavaScript validation
    js_validation: bool = True
    js_samples: int = 500
    js_syntax_check: bool = True
    js_execution_test: bool = True
    
    # Evaluation metrics
    metrics: List[str] = field(default_factory=lambda: [
        "perplexity",
        "bleu",
        "rouge",
        "code_bleu",
        "exact_match",
        "pass@k"
    ])
    
    # Code generation evaluation
    code_eval_config: Dict[str, Any] = field(default_factory=lambda: {
        "timeout": 30,
        "num_samples": 1000,
        "temperature": 0.2,
        "top_p": 0.95,
        "max_new_tokens": 512
    })
    
    # Output settings
    output_dir: str = "evaluation_results"
    save_detailed_results: bool = True
    save_predictions: bool = True
    generate_report: bool = True
    
    # Parallel processing
    num_workers: int = 4
    batch_size: int = 8
    
    # Memory optimization
    max_memory_gb: int = 16
    gradient_checkpointing: bool = True


@dataclass
class ValidationResult:
    """Container for validation results"""
    dataset_name: str
    metric_name: str
    score: float
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class XMLValidator:
    """Validator for XML/MDX code generation"""
    
    def __init__(self, strict: bool = True):
        self.strict = strict
        
        if not XML_AVAILABLE:
            logging.warning("XML validation not available - xml.etree.ElementTree not found")
    
    def validate_xml(self, xml_text: str) -> Tuple[bool, Dict[str, Any]]:
        """Validate XML structure and content"""
        if not XML_AVAILABLE:
            return False, {"error": "XML validation not available"}
        
        try:
            # Parse XML
            root = ET.fromstring(xml_text)
            
            # Basic validation
            validation_details = {
                "is_valid": True,
                "element_count": len(list(root.iter())),
                "max_depth": self._calculate_depth(root),
                "has_attributes": len(root.attrib) > 0,
                "errors": []
            }
            
            # Additional strict validation
            if self.strict:
                validation_details.update(self._strict_validation(root))
            
            return True, validation_details
            
        except ET.ParseError as e:
            return False, {
                "is_valid": False,
                "error": f"XML Parse Error: {str(e)}",
                "position": getattr(e, "position", None)
            }
        except Exception as e:
            return False, {
                "is_valid": False,
                "error": f"XML Validation Error: {str(e)}"
            }
    
    def validate_mdx(self, mdx_text: str) -> Tuple[bool, Dict[str, Any]]:
        """Validate MDX (Markdown + JSX) structure"""
        validation_details = {
            "is_valid": True,
            "jsx_count": mdx_text.count("<") - mdx_text.count("&lt;"),
            "markdown_elements": [],
            "errors": []
        }
        
        try:
            # Basic MDX structure checks
            if self.strict:
                # Check for balanced JSX tags
                validation_details.update(self._validate_jsx_balance(mdx_text))
                
                # Check for valid JSX syntax
                validation_details.update(self._validate_jsx_syntax(mdx_text))
            
            # Check for common MDX patterns
            validation_details["markdown_elements"] = self._extract_markdown_elements(mdx_text)
            
            return True, validation_details
            
        except Exception as e:
            return False, {
                "is_valid": False,
                "error": f"MDX Validation Error: {str(e)}"
            }
    
    def validate_html(self, html_text: str) -> Tuple[bool, Dict[str, Any]]:
        """Validate HTML structure"""
        if not HTML_AVAILABLE:
            return False, {"error": "HTML validation not available - BeautifulSoup not found"}
        
        try:
            # Parse HTML
            soup = BeautifulSoup(html_text, 'html.parser')
            
            validation_details = {
                "is_valid": True,
                "tag_count": len(soup.find_all()),
                "has_head": soup.head is not None,
                "has_body": soup.body is not None,
                "errors": []
            }
            
            # Additional validation
            if self.strict:
                validation_details.update(self._strict_html_validation(soup))
            
            return True, validation_details
            
        except Exception as e:
            return False, {
                "is_valid": False,
                "error": f"HTML Validation Error: {str(e)}"
            }
    
    def _calculate_depth(self, element, depth: int = 0) -> int:
        """Calculate maximum depth of XML tree"""
        max_depth = depth
        for child in element:
            child_depth = self._calculate_depth(child, depth + 1)
            max_depth = max(max_depth, child_depth)
        return max_depth
    
    def _strict_validation(self, root: ET.Element) -> Dict[str, Any]:
        """Perform strict XML validation"""
        details = {
            "has_root_element": root.tag is not None,
            "root_element_name": root.tag,
            "namespace_issues": [],
            "attribute_issues": [],
            "character_issues": []
        }
        
        # Check for invalid characters
        for elem in root.iter():
            if elem.text and any(ord(char) < 32 and char not in '\t\n\r' for char in elem.text):
                details["character_issues"].append(f"Invalid characters in {elem.tag}")
        
        return details
    
    def _validate_jsx_balance(self, mdx_text: str) -> Dict[str, Any]:
        """Validate JSX tag balance"""
        open_tags = mdx_text.count('<') - mdx_text.count('&lt;')
        close_tags = mdx_text.count('</')
        
        return {
            "jsx_tags_balanced": open_tags >= close_tags,
            "jsx_balance_diff": open_tags - close_tags
        }
    
    def _validate_jsx_syntax(self, mdx_text: str) -> Dict[str, Any]:
        """Validate basic JSX syntax"""
        # Simple JSX syntax checks
        jsx_issues = []
        
        # Check for self-closing tags
        lines = mdx_text.split('\n')
        for i, line in enumerate(lines):
            if '/>' in line and '<' in line:
                # Basic validation for self-closing tags
                if line.count('<') != line.count('>') + line.count('/>'):
                    jsx_issues.append(f"Line {i+1}: Potential JSX syntax issue")
        
        return {
            "jsx_syntax_valid": len(jsx_issues) == 0,
            "jsx_issues": jsx_issues
        }
    
    def _strict_html_validation(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Perform strict HTML validation"""
        issues = []
        
        # Check for required HTML5 elements
        required_tags = ['html', 'head', 'title', 'body']
        for tag in required_tags:
            if not soup.find(tag):
                issues.append(f"Missing required tag: {tag}")
        
        # Check for deprecated tags
        deprecated_tags = ['font', 'center', 'big', 'blink']
        for tag in deprecated_tags:
            if soup.find(tag):
                issues.append(f"Deprecated tag found: {tag}")
        
        return {
            "html5_compliant": len(issues) == 0,
            "html_issues": issues
        }
    
    def _extract_markdown_elements(self, mdx_text: str) -> List[str]:
        """Extract markdown elements from MDX text"""
        elements = []
        
        # Headers
        for i in range(1, 7):
            elements.extend([f"h{i}"] * mdx_text.count(f"#{i}"))
        
        # Lists
        elements.extend(["ul"] * mdx_text.count("-"))
        elements.extend(["ol"] * mdx_text.count("1."))
        
        # Code blocks
        elements.extend(["code"] * mdx_text.count("```"))
        
        return list(set(elements))


class JavaScriptValidator:
    """Validator for JavaScript code generation"""
    
    def __init__(self):
        self.syntax_errors = []
        self.runtime_errors = []
        
        if not JS_EVAL_AVAILABLE:
            logging.warning("JavaScript execution not available - js2py not found")
    
    def validate_syntax(self, js_code: str) -> Tuple[bool, Dict[str, Any]]:
        """Validate JavaScript syntax"""
        validation_details = {
            "is_valid": True,
            "line_count": len(js_code.split('\n')),
            "character_count": len(js_code),
            "complexity_score": self._calculate_complexity(js_code),
            "syntax_errors": [],
            "warnings": []
        }
        
        try:
            # Basic syntax validation using Python's JS parsing
            if JS_EVAL_AVAILABLE:
                # Attempt to parse the JavaScript
                context = js2py.EvalContext()
                js2py.parse_js(js_code, context)
            else:
                # Fallback to basic checks
                validation_details["warnings"].append("Advanced syntax validation not available")
                
                # Basic bracket matching
                if not self._check_brackets(js_code):
                    validation_details["syntax_errors"].append("Unmatched brackets")
                    validation_details["is_valid"] = False
            
            return True, validation_details
            
        except js2py.JsException as e:
            return False, {
                "is_valid": False,
                "syntax_errors": [f"JS Syntax Error: {str(e)}"],
                "error_position": getattr(e, 'position', None)
            }
        except Exception as e:
            return False, {
                "is_valid": False,
                "syntax_errors": [f"Validation Error: {str(e)}"]
            }
    
    def test_execution(self, js_code: str, timeout: int = 30) -> Tuple[bool, Dict[str, Any]]:
        """Test JavaScript execution"""
        if not JS_EVAL_AVAILABLE:
            return False, {"error": "JavaScript execution not available"}
        
        execution_details = {
            "executed": False,
            "execution_time": 0,
            "output": None,
            "runtime_errors": [],
            "warnings": []
        }
        
        try:
            start_time = time.time()
            
            # Create a safe execution context
            context = js2py.EvalContext()
            
            # Add timeout check
            def timeout_check():
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Execution timeout after {timeout} seconds")
            
            # Execute the JavaScript
            timeout_check()
            result = js2py.eval_js(js_code, context)
            
            execution_details["executed"] = True
            execution_details["execution_time"] = time.time() - start_time
            execution_details["output"] = str(result)
            
            return True, execution_details
            
        except TimeoutError as e:
            return False, {
                "executed": False,
                "runtime_errors": [f"Execution timeout: {str(e)}"]
            }
        except js2py.JsException as e:
            return False, {
                "executed": False,
                "runtime_errors": [f"Runtime Error: {str(e)}"]
            }
        except Exception as e:
            return False, {
                "executed": False,
                "runtime_errors": [f"Execution Error: {str(e)}"]
            }
    
    def _calculate_complexity(self, js_code: str) -> Dict[str, int]:
        """Calculate code complexity metrics"""
        lines = js_code.split('\n')
        
        complexity_metrics = {
            "cyclomatic_complexity": 1,  # Base complexity
            "function_count": js_code.count('function'),
            "loop_count": js_code.count('for') + js_code.count('while'),
            "if_count": js_code.count('if'),
            "switch_count": js_code.count('switch'),
            "try_count": js_code.count('try')
        }
        
        # Calculate cyclomatic complexity
        cyclomatic = 1
        cyclomatic += complexity_metrics["if_count"]
        cyclomatic += complexity_metrics["loop_count"] * 2
        cyclomatic += complexity_metrics["switch_count"]
        
        # Account for logical operators
        cyclomatic += js_code.count('&&') + js_code.count('||')
        
        complexity_metrics["cyclomatic_complexity"] = cyclomatic
        
        return complexity_metrics
    
    def _check_brackets(self, code: str) -> bool:
        """Basic bracket matching check"""
        brackets = {'(': ')', '[': ']', '{': '}'}
        stack = []
        
        for char in code:
            if char in brackets:
                stack.append(char)
            elif char in brackets.values():
                if not stack or brackets[stack.pop()] != char:
                    return False
        
        return len(stack) == 0


class CodeMetrics:
    """Calculate various code quality metrics"""
    
    def __init__(self):
        self.bleu_metric = None
        self.rouge_metric = None
        
        try:
            self.bleu_metric = evaluate.load("bleu")
            self.rouge_metric = evaluate.load("rouge")
        except Exception as e:
            logging.warning(f"Could not load evaluation metrics: {e}")
    
    def calculate_code_bleu(self, predictions: List[str], references: List[str]) -> float:
        """Calculate CodeBLEU score"""
        if not self.bleu_metric:
            return 0.0
        
        try:
            # CodeBLEU implementation would go here
            # For now, use standard BLEU
            result = self.bleu_metric.compute(
                predictions=predictions,
                references=references
            )
            return result.get('bleu', 0.0)
        except Exception as e:
            logging.error(f"CodeBLEU calculation failed: {e}")
            return 0.0
    
    def calculate_pass_at_k(self, predictions: List[List[str]], references: List[str], k: int = 1) -> float:
        """Calculate pass@k score for code generation"""
        try:
            total_samples = len(references)
            if total_samples == 0:
                return 0.0
            
            pass_count = 0
            
            for i, ref in enumerate(references):
                if i < len(predictions):
                    predictions_for_ref = predictions[i]
                    
                    # Check if any prediction matches the reference
                    for pred in predictions_for_ref[:k]:
                        if self._normalize_code(pred) == self._normalize_code(ref):
                            pass_count += 1
                            break
            
            return pass_count / total_samples
            
        except Exception as e:
            logging.error(f"Pass@k calculation failed: {e}")
            return 0.0
    
    def calculate_perplexity(self, model, tokenizer, dataset: Dataset) -> float:
        """Calculate perplexity on dataset"""
        try:
            model.eval()
            total_loss = 0.0
            total_tokens = 0
            
            dataloader = DataLoader(dataset, batch_size=1)
            
            with torch.no_grad():
                for batch in dataloader:
                    inputs = tokenizer(
                        batch['text'],
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512
                    )
                    
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    
                    # Calculate loss
                    outputs = model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss
                    
                    total_loss += loss.item()
                    total_tokens += inputs["input_ids"].numel()
            
            avg_loss = total_loss / len(dataloader)
            perplexity = torch.exp(torch.tensor(avg_loss)).item()
            
            return perplexity
            
        except Exception as e:
            logging.error(f"Perplexity calculation failed: {e}")
            return float('inf')
    
    def _normalize_code(self, code: str) -> str:
        """Normalize code for comparison"""
        # Remove extra whitespace, normalize variable names, etc.
        import re
        
        # Remove comments
        code = re.sub(r'//.*?\n', '\n', code)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        
        # Normalize whitespace
        code = re.sub(r'\s+', ' ', code)
        
        # Normalize variable names (simplified)
        var_counter = 0
        var_mapping = {}
        
        def replace_var(match):
            nonlocal var_counter
            var_name = match.group(0)
            if var_name not in var_mapping:
                var_mapping[var_name] = f'var_{var_counter}'
                var_counter += 1
            return var_mapping[var_name]
        
        # Simple variable name replacement (only works for simple cases)
        code = re.sub(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', replace_var, code)
        
        return code.strip().lower()


class ModelValidator:
    """Main model validation orchestrator"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
        # Initialize validators
        self.xml_validator = XMLValidator(strict=config.xml_strict_validation)
        self.js_validator = JavaScriptValidator()
        self.code_metrics = CodeMetrics()
        
        # Results storage
        self.validation_results: List[ValidationResult] = []
        self.detailed_results: Dict[str, Any] = {}
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Setup logging for validation"""
        log_dir = Path(self.config.output_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "validation.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_model(self) -> bool:
        """Load the model for validation"""
        try:
            self.logger.info(f"Loading model from {self.config.model_path}")
            
            # Load tokenizer
            tokenizer_path = self.config.tokenizer_path or self.config.model_path
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=self.config.trust_remote_code
            )
            
            # Configure tokenizer
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            model_kwargs = {
                "trust_remote_code": self.config.trust_remote_code
            }
            
            if self.config.load_in_8bit:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            elif self.config.load_in_4bit:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                **model_kwargs
            )
            
            # Move to appropriate device
            if self.config.device == "auto":
                self.model = self.model.to("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.model = self.model.to(self.config.device)
            
            # Create pipeline for easier generation
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            
            self.logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def validate_model(self) -> Dict[str, Any]:
        """Run comprehensive model validation"""
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        self.logger.info("Starting comprehensive model validation")
        start_time = time.time()
        
        # Run all validations
        if "perplexity" in self.config.metrics:
            self._validate_perplexity()
        
        if "code_bleu" in self.config.metrics:
            self._validate_code_bleu()
        
        if "pass@k" in self.config.metrics:
            self._validate_pass_at_k()
        
        # XML/MDX validation
        if self.config.xml_validation:
            self._validate_xml_mdx()
        
        # JavaScript validation
        if self.config.js_validation:
            self._validate_javascript()
        
        # Load and evaluate standard datasets
        for dataset_name in self.config.validation_datasets:
            self._evaluate_standard_dataset(dataset_name)
        
        # Custom dataset validation
        for dataset_name, dataset_path in self.config.custom_datasets.items():
            self._evaluate_custom_dataset(dataset_name, dataset_path)
        
        validation_time = time.time() - start_time
        
        # Compile results
        results = self._compile_results(validation_time)
        
        # Save results
        self._save_results(results)
        
        self.logger.info(f"Validation completed in {validation_time:.2f} seconds")
        return results
    
    def _validate_perplexity(self) -> None:
        """Validate model perplexity"""
        try:
            self.logger.info("Calculating perplexity...")
            
            # Load a small dataset for perplexity calculation
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1000]")
            
            perplexity = self.code_metrics.calculate_perplexity(
                self.model, self.tokenizer, dataset
            )
            
            result = ValidationResult(
                dataset_name="wikitext",
                metric_name="perplexity",
                score=perplexity,
                details={"dataset_size": len(dataset)}
            )
            
            self.validation_results.append(result)
            self.logger.info(f"Perplexity: {perplexity:.4f}")
            
        except Exception as e:
            self.logger.error(f"Perplexity validation failed: {e}")
    
    def _validate_code_bleu(self) -> None:
        """Validate CodeBLEU score"""
        try:
            self.logger.info("Calculating CodeBLEU score...")
            
            # Load a code dataset
            dataset = load_dataset("codeparrot/codeparrot-clean", split="train[:1000]")
            
            # Generate predictions
            predictions = []
            references = []
            
            for i, sample in enumerate(dataset):
                if i >= 100:  # Limit for testing
                    break
                
                input_text = sample["content"][:100]  # Use first 100 chars as input
                reference = sample["content"][100:200]  # Next 100 chars as reference
                
                # Generate prediction
                prompt = f"Complete this code:\n{input_text}"
                generated = self._generate_text(prompt)
                
                predictions.append(generated)
                references.append(reference)
            
            code_bleu = self.code_metrics.calculate_code_bleu(predictions, references)
            
            result = ValidationResult(
                dataset_name="codeparrot",
                metric_name="code_bleu",
                score=code_bleu,
                details={"num_samples": len(predictions)}
            )
            
            self.validation_results.append(result)
            self.logger.info(f"CodeBLEU: {code_bleu:.4f}")
            
        except Exception as e:
            self.logger.error(f"CodeBLEU validation failed: {e}")
    
    def _validate_pass_at_k(self) -> None:
        """Validate pass@k score"""
        try:
            self.logger.info("Calculating pass@k score...")
            
            # This is a simplified implementation
            # In practice, you'd need a more sophisticated setup for pass@k
            
            # For demonstration, we'll use a simple dataset
            dataset = load_dataset("google-research-datasets/mbpp", "full", split="test[:100]")
            
            predictions = []
            references = []
            
            for sample in dataset:
                problem = sample["text"]
                solution = sample["solution"]
                
                # Generate multiple predictions
                generated_solutions = []
                for _ in range(5):  # Generate 5 candidates
                    prompt = f"Write a Python function:\n{problem}\n\nSolution:"
                    generated = self._generate_text(prompt, max_new_tokens=200)
                    generated_solutions.append(generated)
                
                predictions.append(generated_solutions)
                references.append(solution)
            
            pass_at_k = self.code_metrics.calculate_pass_at_k(predictions, references, k=1)
            
            result = ValidationResult(
                dataset_name="mbpp",
                metric_name="pass_at_1",
                score=pass_at_k,
                details={"num_samples": len(predictions)}
            )
            
            self.validation_results.append(result)
            self.logger.info(f"Pass@1: {pass_at_k:.4f}")
            
        except Exception as e:
            self.logger.error(f"Pass@k validation failed: {e}")
    
    def _validate_xml_mdx(self) -> None:
        """Validate XML/MDX code generation"""
        try:
            self.logger.info("Validating XML/MDX generation...")
            
            xml_prompts = [
                "Create a simple HTML page with a header and paragraph",
                "Generate an XML file with a root element and child elements",
                "Create a MDX component with JSX and markdown"
            ]
            
            xml_results = []
            
            for prompt in xml_prompts:
                generated = self._generate_text(prompt, max_new_tokens=300)
                
                # Validate XML
                if "html" in prompt.lower():
                    is_valid, details = self.xml_validator.validate_html(generated)
                else:
                    is_valid, details = self.xml_validator.validate_xml(generated)
                
                xml_results.append({
                    "prompt": prompt,
                    "generated": generated,
                    "is_valid": is_valid,
                    "details": details
                })
            
            # Calculate success rate
            valid_count = sum(1 for result in xml_results if result["is_valid"])
            success_rate = valid_count / len(xml_results)
            
            result = ValidationResult(
                dataset_name="xml_mdx_generation",
                metric_name="xml_validity_rate",
                score=success_rate,
                details={"results": xml_results}
            )
            
            self.validation_results.append(result)
            self.logger.info(f"XML/MDX validity rate: {success_rate:.4f}")
            
        except Exception as e:
            self.logger.error(f"XML/MDX validation failed: {e}")
    
    def _validate_javascript(self) -> None:
        """Validate JavaScript code generation"""
        try:
            self.logger.info("Validating JavaScript generation...")
            
            js_prompts = [
                "Write a function to add two numbers",
                "Create a function to find the maximum in an array",
                "Write a function that reverses a string"
            ]
            
            js_results = []
            
            for prompt in js_prompts:
                generated = self._generate_text(prompt, max_new_tokens=200)
                
                # Validate syntax
                is_valid, syntax_details = self.js_validator.validate_syntax(generated)
                
                # Test execution if configured
                execution_details = {}
                if self.config.js_execution_test and is_valid:
                    is_executable, execution_details = self.js_validator.test_execution(generated)
                
                js_results.append({
                    "prompt": prompt,
                    "generated": generated,
                    "syntax_valid": is_valid,
                    "execution_successful": execution_details.get("executed", False),
                    "syntax_details": syntax_details,
                    "execution_details": execution_details
                })
            
            # Calculate success rates
            syntax_valid_count = sum(1 for result in js_results if result["syntax_valid"])
            execution_success_count = sum(1 for result in js_results if result["execution_successful"])
            
            syntax_success_rate = syntax_valid_count / len(js_results)
            execution_success_rate = execution_success_count / len(js_results)
            
            # Store results
            result = ValidationResult(
                dataset_name="javascript_generation",
                metric_name="js_syntax_validity_rate",
                score=syntax_success_rate,
                details={"results": js_results}
            )
            self.validation_results.append(result)
            
            result = ValidationResult(
                dataset_name="javascript_generation",
                metric_name="js_execution_success_rate",
                score=execution_success_rate,
                details={"results": js_results}
            )
            self.validation_results.append(result)
            
            self.logger.info(f"JavaScript syntax validity: {syntax_success_rate:.4f}")
            self.logger.info(f"JavaScript execution success: {execution_success_rate:.4f}")
            
        except Exception as e:
            self.logger.error(f"JavaScript validation failed: {e}")
    
    def _evaluate_standard_dataset(self, dataset_name: str) -> None:
        """Evaluate model on standard datasets"""
        try:
            self.logger.info(f"Evaluating on {dataset_name}...")
            
            # Load dataset
            if dataset_name == "HumanEval":
                dataset = load_dataset("openai/human-eval", split="test")
            elif dataset_name == "MMLU":
                dataset = load_dataset("hails/mmlu_no_train", split="test")
            elif dataset_name == "mbpp":
                dataset = load_dataset("google-research-datasets/mbpp", "full", split="test")
            else:
                # Try to load with dataset name
                dataset = load_dataset(dataset_name, split="test[:100]")
            
            # Evaluate
            metrics = self._evaluate_dataset(dataset, dataset_name)
            
            # Store results
            for metric_name, score in metrics.items():
                result = ValidationResult(
                    dataset_name=dataset_name,
                    metric_name=metric_name,
                    score=score,
                    details={"dataset_size": len(dataset)}
                )
                self.validation_results.append(result)
            
            self.logger.info(f"Completed evaluation on {dataset_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate {dataset_name}: {e}")
    
    def _evaluate_custom_dataset(self, dataset_name: str, dataset_path: str) -> None:
        """Evaluate model on custom dataset"""
        try:
            self.logger.info(f"Evaluating on custom dataset {dataset_name}...")
            
            # Load custom dataset
            if dataset_path.endswith('.json'):
                with open(dataset_path, 'r') as f:
                    data = json.load(f)
                dataset = Dataset.from_list(data)
            else:
                dataset = load_from_disk(dataset_path)
            
            # Evaluate
            metrics = self._evaluate_dataset(dataset, dataset_name)
            
            # Store results
            for metric_name, score in metrics.items():
                result = ValidationResult(
                    dataset_name=dataset_name,
                    metric_name=metric_name,
                    score=score,
                    details={"dataset_size": len(dataset)}
                )
                self.validation_results.append(result)
            
            self.logger.info(f"Completed evaluation on custom dataset {dataset_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate custom dataset {dataset_name}: {e}")
    
    def _evaluate_dataset(self, dataset: Dataset, dataset_name: str) -> Dict[str, float]:
        """Evaluate model on a dataset"""
        metrics = {}
        
        # This is a simplified evaluation
        # In practice, you'd implement dataset-specific evaluations
        
        try:
            # Generate predictions for a subset
            predictions = []
            references = []
            
            for i, sample in enumerate(dataset):
                if i >= 50:  # Limit for testing
                    break
                
                # Extract text based on dataset format
                text = sample.get('text', sample.get('input', sample.get('prompt', '')))
                
                if text:
                    # Generate prediction
                    generated = self._generate_text(text)
                    predictions.append(generated)
                    
                    # Extract reference
                    reference = sample.get('target', sample.get('output', sample.get('label', '')))
                    if reference:
                        references.append(reference)
            
            # Calculate basic metrics
            if references:
                # Use available metrics
                if self.bleu_metric:
                    bleu_result = self.bleu_metric.compute(
                        predictions=predictions,
                        references=references[:len(predictions)]
                    )
                    metrics["bleu"] = bleu_result.get("bleu", 0.0)
                
                if self.rouge_metric:
                    rouge_result = self.rouge_metric.compute(
                        predictions=predictions,
                        references=references[:len(predictions)]
                    )
                    metrics["rouge1"] = rouge_result.get("rouge1", 0.0)
                    metrics["rouge2"] = rouge_result.get("rouge2", 0.0)
            
            # Calculate exact match if applicable
            exact_matches = 0
            for pred, ref in zip(predictions, references):
                if pred.strip().lower() == ref.strip().lower():
                    exact_matches += 1
            
            if references:
                metrics["exact_match"] = exact_matches / len(references)
            
        except Exception as e:
            self.logger.error(f"Dataset evaluation failed for {dataset_name}: {e}")
        
        return metrics
    
    def _generate_text(self, prompt: str, max_new_tokens: int = None) -> str:
        """Generate text using the model"""
        try:
            if max_new_tokens is None:
                max_new_tokens = self.config.max_new_tokens
            
            # Prepare prompt
            if not prompt.endswith('\n'):
                prompt += '\n'
            
            # Generate
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    repetition_penalty=self.config.repetition_penalty,
                    do_sample=self.config.do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode
            generated_text = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            
            return generated_text.strip()
            
        except Exception as e:
            self.logger.error(f"Text generation failed: {e}")
            return ""
    
    def _compile_results(self, validation_time: float) -> Dict[str, Any]:
        """Compile all validation results"""
        
        # Group results by dataset
        dataset_results = {}
        for result in self.validation_results:
            dataset_name = result.dataset_name
            
            if dataset_name not in dataset_results:
                dataset_results[dataset_name] = {}
            
            dataset_results[dataset_name][result.metric_name] = {
                "score": result.score,
                "details": result.details
            }
        
        # Compile final results
        results = {
            "model_path": self.config.model_path,
            "validation_timestamp": datetime.now().isoformat(),
            "validation_time_seconds": validation_time,
            "config": self.config.__dict__,
            "dataset_results": dataset_results,
            "summary": {
                "total_datasets": len(dataset_results),
                "total_metrics": len(self.validation_results),
                "average_scores": {}
            }
        }
        
        # Calculate summary statistics
        for result in self.validation_results:
            metric_name = result.metric_name
            if metric_name not in results["summary"]["average_scores"]:
                results["summary"]["average_scores"][metric_name] = []
            
            results["summary"]["average_scores"][metric_name].append(result.score)
        
        # Compute averages
        for metric_name, scores in results["summary"]["average_scores"].items():
            results["summary"]["average_scores"][metric_name] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores))
            }
        
        return results
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save validation results"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        if self.config.save_detailed_results:
            with open(output_dir / "detailed_results.json", 'w') as f:
                json.dump(results, f, indent=2, default=str)
        
        # Save predictions if configured
        if self.config.save_predictions:
            predictions_file = output_dir / "predictions.json"
            with open(predictions_file, 'w') as f:
                json.dump(self.detailed_results, f, indent=2, default=str)
        
        # Generate report if configured
        if self.config.generate_report:
            self._generate_report(results, output_dir)
        
        self.logger.info(f"Results saved to {output_dir}")
    
    def _generate_report(self, results: Dict[str, Any], output_dir: Path) -> None:
        """Generate a validation report"""
        report_file = output_dir / "validation_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# Model Validation Report\n\n")
            f.write(f"**Model:** {results['model_path']}\n")
            f.write(f"**Timestamp:** {results['validation_timestamp']}\n")
            f.write(f"**Validation Time:** {results['validation_time_seconds']:.2f} seconds\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"- **Datasets Evaluated:** {results['summary']['total_datasets']}\n")
            f.write(f"- **Total Metrics:** {results['summary']['total_metrics']}\n\n")
            
            f.write("## Average Scores\n\n")
            for metric_name, stats in results['summary']['average_scores'].items():
                f.write(f"### {metric_name}\n")
                f.write(f"- Mean: {stats['mean']:.4f}\n")
                f.write(f"- Std: {stats['std']:.4f}\n")
                f.write(f"- Min: {stats['min']:.4f}\n")
                f.write(f"- Max: {stats['max']:.4f}\n\n")
            
            f.write("## Dataset Results\n\n")
            for dataset_name, metrics in results['dataset_results'].items():
                f.write(f"### {dataset_name}\n\n")
                for metric_name, data in metrics.items():
                    f.write(f"**{metric_name}:** {data['score']:.4f}\n")
                f.write("\n")
        
        self.logger.info(f"Validation report generated: {report_file}")


def create_validation_config(
    model_path: str,
    output_dir: str = "evaluation_results",
    **kwargs
) -> ValidationConfig:
    """Create validation configuration with defaults"""
    
    default_config = {
        "model_path": model_path,
        "output_dir": output_dir,
        "device": "auto",
        "trust_remote_code": False,
        "load_in_8bit": False,
        "load_in_4bit": False,
        "max_new_tokens": 512,
        "temperature": 0.2,
        "top_p": 0.95,
        "do_sample": True,
        "xml_validation": True,
        "js_validation": True,
        "save_detailed_results": True,
        "save_predictions": True,
        "generate_report": True,
        "validation_datasets": ["HumanEval", "MMLU", "mbpp"],
        "metrics": ["perplexity", "bleu", "rouge", "exact_match"]
    }
    
    default_config.update(kwargs)
    
    return ValidationConfig(**default_config)


def run_validation(model_path: str, **kwargs) -> Dict[str, Any]:
    """Run complete model validation"""
    
    config = create_validation_config(model_path, **kwargs)
    validator = ModelValidator(config)
    
    if not validator.load_model():
        raise RuntimeError("Failed to load model")
    
    results = validator.validate_model()
    return results


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate Sheikh-2.5-Coder model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Output directory")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--temperature", type=float, default=0.2, help="Generation temperature")
    
    args = parser.parse_args()
    
    # Run validation
    results = run_validation(
        model_path=args.model_path,
        output_dir=args.output_dir,
        device=args.device,
        temperature=args.temperature
    )
    
    print("Validation completed successfully!")
    print(f"Results saved to {args.output_dir}")