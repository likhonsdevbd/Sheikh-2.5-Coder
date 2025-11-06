#!/usr/bin/env python3
"""
Quality Filters Pipeline

Implements comprehensive quality filtering for all dataset components:
- Language-specific quality filters
- Content quality assessment
- Semantic coherence validation
- Syntax validity checks
- Duplication detection and removal

Author: MiniMax Agent
Date: 2025-11-06
"""

import os
import sys
import json
import yaml
import logging
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
from tqdm import tqdm

# Data processing libraries
from datasets import Dataset

# Quality filtering libraries
from datasketch import MinHash, LSH

# Natural language processing
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

logger = logging.getLogger(__name__)

class QualityFilterPipeline:
    """
    Comprehensive quality filtering pipeline
    
    Applies multiple quality filters:
    - Language-specific filtering
    - Content quality assessment
    - Semantic coherence validation
    - Syntax validity checks
    - Duplication detection
    """
    
    def __init__(self, config: DataPreparationConfig):
        """Initialize quality filter pipeline"""
        self.config = config
        self.setup_logging()
        self.setup_directories()
        self.initialize_quality_filters()
        self.initialize_language_filters()
        self.initialize_semantic_filters()
        
        # Filtering statistics
        self.stats = {
            'total_examples': 0,
            'language_filtered': 0,
            'quality_filtered': 0,
            'syntax_filtered': 0,
            'semantic_filtered': 0,
            'duplicate_filtered': 0,
            'final_passed': 0
        }
        
        logger.info("Quality Filter Pipeline initialized")
    
    def setup_logging(self):
        """Setup logging for quality filtering"""
        log_handler = logging.FileHandler('logs/quality_filters.log')
        log_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(log_handler)
    
    def setup_directories(self):
        """Create necessary directories"""
        directories = [
            'cache/quality_filters',
            'cache/semantic_analysis',
            'cache/duplicate_detection',
            'logs'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def initialize_quality_filters(self):
        """Initialize quality filtering parameters"""
        self.quality_thresholds = {
            'minimum_length': 30,
            'maximum_length': 100000,
            'quality_score_threshold': 0.7,
            'semantic_coherence_threshold': 0.6,
            'syntax_validity_threshold': 0.8,
            'duplicate_threshold': 0.8,
            'complexity_score_threshold': 0.3
        }
        
        # Language-specific thresholds
        self.language_thresholds = {
            'javascript': {
                'min_length': 50,
                'max_length': 50000,
                'quality_score': 0.7,
                'complexity_threshold': 0.3
            },
            'typescript': {
                'min_length': 50,
                'max_length': 50000,
                'quality_score': 0.75,
                'complexity_threshold': 0.35
            },
            'xml': {
                'min_length': 30,
                'max_length': 30000,
                'quality_score': 0.8,
                'complexity_threshold': 0.25
            },
            'html': {
                'min_length': 100,
                'max_length': 40000,
                'quality_score': 0.7,
                'complexity_threshold': 0.3
            },
            'css': {
                'min_length': 50,
                'max_length': 20000,
                'quality_score': 0.75,
                'complexity_threshold': 0.25
            },
            'mdx': {
                'min_length': 100,
                'max_length': 30000,
                'quality_score': 0.8,
                'complexity_threshold': 0.4
            }
        }
        
        logger.info(f"Initialized quality thresholds for {len(self.language_thresholds)} languages")
    
    def initialize_language_filters(self):
        """Initialize language-specific filtering patterns"""
        self.language_patterns = {
            'javascript': {
                'required_patterns': [
                    r'\b(function|const|let|var|class|import|export)\b',
                    r'[{}();]',  # Basic syntax
                    r'[a-zA-Z_$][a-zA-Z0-9_$]*',  # Identifiers
                ],
                'quality_patterns': [
                    r'\b(console|window|document)\b',
                    r'\b(=>|async|await)\b',
                    r'\b(try|catch|throw)\b'
                ],
                'exclusion_patterns': [
                    r'console\.log\([^)]*\);?\s*$',  # Solo console.log statements
                    r'//\s*$',  # Empty comments
                    r'/\*\s*\*/'  # Empty comment blocks
                ]
            },
            'typescript': {
                'required_patterns': [
                    r'\b(function|const|let|var|class|interface|type|import|export)\b',
                    r'[{}();:]',  # Basic syntax + type annotations
                    r'[a-zA-Z_$][a-zA-Z0-9_$<>,[\]\s]*',  # Identifiers + generics
                ],
                'quality_patterns': [
                    r':\s*[a-zA-Z_$][a-zA-Z0-9_$<>,\[\]\s\|&]*',
                    r'\b(interface|type)\s+[a-zA-Z_$][a-zA-Z0-9_$]*',
                    r'\b(public|private|protected|readonly)\b'
                ],
                'exclusion_patterns': [
                    r':\s*any\b',  # : any annotations
                    r'//\s*$',  # Empty comments
                ]
            },
            'xml': {
                'required_patterns': [
                    r'<\?xml[^>]*\?>',  # XML declaration
                    r'<[a-zA-Z][^>]*>',  # Valid tags
                    r'</[a-zA-Z][^>]*>',  # Closing tags
                    r'[a-zA-Z_][a-zA-Z0-9_\-]*=',  # Attributes
                ],
                'quality_patterns': [
                    r'xmlns[=:][\'"][^\'"]*[\'"]',
                    r'<!DOCTYPE[^>]*>',
                    r'<!--.*?-->'
                ],
                'exclusion_patterns': [
                    r'<[^>]*>[^<]*$',  # Unclosed tags
                    r'>[^<]*<'  # Overlapping tags
                ]
            },
            'html': {
                'required_patterns': [
                    r'<(!DOCTYPE|html|head|body|div|span|p|a|img|form|input)[^>]*>',  # Common tags
                    r'</(html|head|body|div|span|p|a|img|form|input)[^>]*>',  # Closing tags
                    r'class\s*=\s*["\'][^"\']*["\']',  # CSS classes
                    r'id\s*=\s*["\'][^"\']*["\']',  # IDs
                ],
                'quality_patterns': [
                    r'<meta[^>]*>',
                    r'<title[^>]*>.*?</title>',
                    r'charset\s*=\s*["\'][^"\']*["\']'
                ],
                'exclusion_patterns': [
                    r'<[^>]*>[^<]*$',  # Unclosed tags
                    r'<[^>]*<[^>]*>',  # Nested same-type tags
                ]
            },
            'css': {
                'required_patterns': [
                    r'[a-zA-Z_-]+\s*{',  # Selectors with opening brace
                    r'}\s*[a-zA-Z_-]*\s*{',  # Multiple rules
                    r'[a-zA-Z-]+\s*:\s*[^;]+;',  # Properties
                    r'/*.*?\*/',  # Comments
                ],
                'quality_patterns': [
                    r'@media[^{]+{',
                    r'@import[^{;]+;',
                    r'var\(--[^)]+\)',
                ],
                'exclusion_patterns': [
                    r'{\s*}',  # Empty rules
                    r':\s*;',  # Empty properties
                ]
            },
            'mdx': {
                'required_patterns': [
                    r'#{1,6}\s+',  # Headers
                    r'\*\*.*?\*\*',  # Bold text
                    r'\*.*?\*',  # Italic text
                    r'```',  # Code blocks
                    r'<[^>]+>',  # JSX components
                ],
                'quality_patterns': [
                    r'```[a-zA-Z]*\n',
                    r'<[A-Z][a-zA-Z]*[^>]*/?>',  # React components
                    r'\[.*?\]\(.*?\)',  # Links
                ],
                'exclusion_patterns': [
                    r'^#{1,6}\s*$',  # Empty headers
                    r'```\s*$',  # Empty code blocks
                ]
            }
        }
        
        logger.info(f"Initialized language patterns for {len(self.language_patterns)} languages")
    
    def initialize_semantic_filters(self):
        """Initialize semantic filtering capabilities"""
        self.stop_words = set([
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
        ])
        
        # Initialize NLTK if available
        if NLTK_AVAILABLE:
            try:
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                self.stop_words = set(nltk.corpus.stopwords.words('english'))
            except Exception as e:
                logger.warning(f"NLTK data download failed: {e}")
        
        logger.info("Semantic filters initialized")
    
    def apply_all_filters(self, dataset: Dataset, dataset_name: str = "unknown") -> Dataset:
        """
        Apply comprehensive quality filtering to dataset
        
        Args:
            dataset: Input dataset to filter
            dataset_name: Name of the dataset for logging
            
        Returns:
            Dataset: Filtered dataset
        """
        logger.info(f"Starting comprehensive quality filtering for {dataset_name}")
        
        try:
            # Reset statistics
            self.reset_statistics()
            
            # Convert dataset to list for processing
            examples = list(dataset)
            self.stats['total_examples'] = len(examples)
            
            logger.info(f"Processing {len(examples):,} examples...")
            
            # Stage 1: Language-specific filtering
            logger.info("Stage 1: Language-specific filtering")
            stage1_filtered = self.apply_language_filtering(examples, dataset_name)
            
            # Stage 2: Content quality filtering
            logger.info("Stage 2: Content quality filtering")
            stage2_filtered = self.apply_content_quality_filtering(stage1_filtered)
            
            # Stage 3: Syntax validity filtering
            logger.info("Stage 3: Syntax validity filtering")
            stage3_filtered = self.apply_syntax_validity_filtering(stage2_filtered)
            
            # Stage 4: Semantic coherence filtering
            logger.info("Stage 4: Semantic coherence filtering")
            stage4_filtered = self.apply_semantic_coherence_filtering(stage3_filtered)
            
            # Stage 5: Duplication filtering
            logger.info("Stage 5: Duplication filtering")
            stage5_filtered = self.apply_duplication_filtering(stage4_filtered, dataset_name)
            
            # Create final dataset
            final_dataset = Dataset.from_list(stage5_filtered)
            self.stats['final_passed'] = len(stage5_filtered)
            
            # Log filtering statistics
            self.log_filtering_statistics(dataset_name)
            
            # Save filtered dataset
            self.save_filtered_dataset(final_dataset, dataset_name)
            
            logger.info(f"Quality filtering completed for {dataset_name}: {len(final_dataset):,} examples passed")
            
            return final_dataset
            
        except Exception as e:
            logger.error(f"Quality filtering failed for {dataset_name}: {str(e)}")
            raise
    
    def reset_statistics(self):
        """Reset filtering statistics"""
        self.stats = {
            'total_examples': 0,
            'language_filtered': 0,
            'quality_filtered': 0,
            'syntax_filtered': 0,
            'semantic_filtered': 0,
            'duplicate_filtered': 0,
            'final_passed': 0
        }
    
    def apply_language_filtering(self, examples: List[Dict], dataset_name: str) -> List[Dict]:
        """Apply language-specific filtering"""
        filtered_examples = []
        
        with tqdm(total=len(examples), desc="Language filtering") as pbar:
            for example in examples:
                language = self.detect_language(example)
                
                if self.passes_language_filter(example, language):
                    filtered_examples.append(example)
                else:
                    self.stats['language_filtered'] += 1
                
                pbar.update(1)
        
        logger.info(f"Language filtering: {len(filtered_examples):,}/{len(examples):,} passed")
        return filtered_examples
    
    def detect_language(self, example: Dict) -> str:
        """Detect programming language from example"""
        content = example.get('content', '') or example.get('code', '') or example.get('output', '')
        
        # Language detection patterns
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
            if matches >= 2:  # Require at least 2 pattern matches
                return language
        
        # Fallback to metadata language if available
        return example.get('language', 'unknown')
    
    def passes_language_filter(self, example: Dict, language: str) -> bool:
        """Check if example passes language-specific filter"""
        
        if language not in self.language_patterns:
            return True  # Accept unknown languages
        
        content = example.get('content', '') or example.get('code', '') or example.get('output', '')
        
        # Check required patterns
        required_patterns = self.language_patterns[language]['required_patterns']
        required_matches = sum(1 for pattern in required_patterns if re.search(pattern, content))
        
        if required_matches < len(required_patterns) * 0.7:  # Require 70% of required patterns
            return False
        
        # Check exclusion patterns
        exclusion_patterns = self.language_patterns[language]['exclusion_patterns']
        exclusion_matches = sum(1 for pattern in exclusion_patterns if re.search(pattern, content, re.MULTILINE))
        
        if exclusion_matches > 0:
            return False
        
        # Check length constraints
        if language in self.language_thresholds:
            thresholds = self.language_thresholds[language]
            content_length = len(content)
            
            if content_length < thresholds['min_length']:
                return False
            
            if content_length > thresholds['max_length']:
                return False
        
        return True
    
    def apply_content_quality_filtering(self, examples: List[Dict]) -> List[Dict]:
        """Apply content quality filtering"""
        filtered_examples = []
        
        with tqdm(total=len(examples), desc="Content quality filtering") as pbar:
            for example in examples:
                quality_score = self.calculate_content_quality_score(example)
                
                if quality_score >= self.quality_thresholds['quality_score_threshold']:
                    # Add quality score to metadata
                    if isinstance(example, dict):
                        example['quality_score'] = quality_score
                    filtered_examples.append(example)
                else:
                    self.stats['quality_filtered'] += 1
                
                pbar.update(1)
        
        logger.info(f"Content quality filtering: {len(filtered_examples):,}/{len(examples):,} passed")
        return filtered_examples
    
    def calculate_content_quality_score(self, example: Dict) -> float:
        """Calculate comprehensive content quality score"""
        
        content = example.get('content', '') or example.get('code', '') or example.get('output', '')
        
        # Base quality factors
        factors = []
        
        # 1. Length appropriateness
        length_score = self.assess_length_quality(content)
        factors.append(length_score)
        
        # 2. Code structure quality
        structure_score = self.assess_structure_quality(content)
        factors.append(structure_score)
        
        # 3. Complexity appropriateness
        complexity_score = self.assess_complexity_quality(content)
        factors.append(complexity_score)
        
        # 4. Language consistency
        language_consistency = self.assess_language_consistency(example)
        factors.append(language_consistency)
        
        # 5. Semantic richness
        semantic_score = self.assess_semantic_richness(content)
        factors.append(semantic_score)
        
        return np.mean(factors)
    
    def assess_length_quality(self, content: str) -> float:
        """Assess quality based on content length"""
        length = len(content)
        
        # Optimal range depends on content type
        if content.strip().startswith('<'):  # XML/HTML
            optimal_range = (100, 10000)
        elif 'function' in content or 'class' in content:  # Code
            optimal_range = (100, 5000)
        else:  # Other content
            optimal_range = (50, 2000)
        
        min_length, max_length = optimal_range
        
        if min_length <= length <= max_length:
            return 1.0
        elif length < min_length:
            return length / min_length
        else:
            return max_length / length
    
    def assess_structure_quality(self, content: str) -> float:
        """Assess quality based on code structure"""
        
        # Basic structure checks
        structure_elements = {
            'braces': content.count('{') == content.count('}'),
            'parentheses': content.count('(') == content.count(')'),
            'brackets': content.count('[') == content.count(']'),
        }
        
        # Language-specific structure checks
        if 'function' in content:
            structure_elements['function_definition'] = 'function' in content or '=>' in content
        
        if 'class' in content:
            structure_elements['class_definition'] = 'class' in content
        
        if content.strip().startswith('<'):
            structure_elements['xml_structure'] = '<' in content and '>' in content
        
        # Calculate structure score
        passed_checks = sum(structure_elements.values())
        total_checks = len(structure_elements)
        
        return passed_checks / total_checks
    
    def assess_complexity_quality(self, content: str) -> float:
        """Assess quality based on complexity appropriateness"""
        
        # Complexity indicators
        complexity_indicators = {
            'variables': len(re.findall(r'\b[a-zA-Z_$][a-zA-Z0-9_$]*\b', content)),
            'functions': len(re.findall(r'\bfunction\b', content)),
            'classes': len(re.findall(r'\bclass\b', content)),
            'loops': len(re.findall(r'\b(for|while)\b', content)),
            'conditions': len(re.findall(r'\bif|else|switch|case\b', content)),
            'async_operations': len(re.findall(r'\basync|await|promise\b', content, re.IGNORECASE))
        }
        
        # Calculate complexity score
        total_indicators = sum(complexity_indicators.values())
        
        if total_indicators == 0:
            return 0.5  # Basic complexity
        
        # Optimal complexity range
        if 5 <= total_indicators <= 50:
            return 1.0
        elif total_indicators < 5:
            return total_indicators / 5
        else:
            return max(0.3, 50 / total_indicators)
    
    def assess_language_consistency(self, example: Dict) -> float:
        """Assess language consistency in example"""
        
        content = example.get('content', '') or example.get('code', '') or example.get('output', '')
        metadata_language = example.get('language', 'unknown')
        
        if metadata_language == 'unknown':
            return 0.8  # Unknown language gets neutral score
        
        # Detect language from content
        detected_language = self.detect_language(example)
        
        # Check consistency
        if metadata_language == detected_language:
            return 1.0
        elif self.are_languages_compatible(metadata_language, detected_language):
            return 0.8  # Compatible languages get good score
        else:
            return 0.3  # Inconsistent languages get low score
    
    def are_languages_compatible(self, lang1: str, lang2: str) -> bool:
        """Check if two languages are compatible"""
        
        compatible_pairs = {
            ('javascript', 'typescript'),
            ('html', 'xml'),
            ('js', 'ts'),  # Short forms
        }
        
        return (lang1, lang2) in compatible_pairs or (lang2, lang1) in compatible_pairs
    
    def assess_semantic_richness(self, content: str) -> float:
        """Assess semantic richness of content"""
        
        # Remove code-specific elements for semantic analysis
        semantic_content = re.sub(r'[{}();,:=<>"\']', ' ', content)
        semantic_content = re.sub(r'\b\w+\.(?:\w+|\*)', ' ', semantic_content)  # Remove method calls
        
        # Tokenize and analyze
        tokens = re.findall(r'\b\w+\b', semantic_content.lower())
        
        if len(tokens) < 3:
            return 0.3  # Very few tokens
        
        # Filter out common programming words
        programming_words = {
            'function', 'var', 'let', 'const', 'class', 'return', 'if', 'else', 'for', 'while',
            'new', 'this', 'true', 'false', 'null', 'undefined', 'try', 'catch', 'throw'
        }
        
        semantic_tokens = [token for token in tokens if token not in self.stop_words and token not in programming_words]
        
        if len(semantic_tokens) < 3:
            return 0.4
        
        # Calculate vocabulary richness
        unique_tokens = set(semantic_tokens)
        vocabulary_richness = len(unique_tokens) / len(semantic_tokens)
        
        # Calculate information density
        info_density = len(semantic_tokens) / len(tokens)
        
        # Combine factors
        richness_score = (vocabulary_richness + info_density) / 2
        
        return min(richness_score, 1.0)
    
    def apply_syntax_validity_filtering(self, examples: List[Dict]) -> List[Dict]:
        """Apply syntax validity filtering"""
        filtered_examples = []
        
        with tqdm(total=len(examples), desc="Syntax validity filtering") as pbar:
            for example in examples:
                validity_score = self.validate_syntax(example)
                
                if validity_score >= self.quality_thresholds['syntax_validity_threshold']:
                    filtered_examples.append(example)
                else:
                    self.stats['syntax_filtered'] += 1
                
                pbar.update(1)
        
        logger.info(f"Syntax validity filtering: {len(filtered_examples):,}/{len(examples):,} passed")
        return filtered_examples
    
    def validate_syntax(self, example: Dict) -> float:
        """Validate syntax of example content"""
        
        content = example.get('content', '') or example.get('code', '') or example.get('output', '')
        language = self.detect_language(example)
        
        if language == 'javascript':
            return self.validate_javascript_syntax(content)
        elif language == 'typescript':
            return self.validate_typescript_syntax(content)
        elif language == 'xml':
            return self.validate_xml_syntax(content)
        elif language == 'html':
            return self.validate_html_syntax(content)
        elif language == 'css':
            return self.validate_css_syntax(content)
        else:
            return 0.8  # Default score for unsupported languages
    
    def validate_javascript_syntax(self, content: str) -> float:
        """Validate JavaScript syntax"""
        
        # Basic syntax checks
        syntax_checks = []
        
        # Brace matching
        brace_balance = content.count('{') - content.count('}')
        syntax_checks.append(brace_balance == 0)
        
        # Parenthesis matching
        paren_balance = content.count('(') - content.count(')')
        syntax_checks.append(paren_balance == 0)
        
        # Bracket matching
        bracket_balance = content.count('[') - content.count(']')
        syntax_checks.append(bracket_balance == 0)
        
        # Basic pattern validation
        patterns = [
            (r'\bfunction\b.*\(.*\)\s*{', 0.9),  # Function declaration
            (r'const\s+\w+\s*=', 0.8),  # Variable declaration
            (r'=>\s*{', 0.8),  # Arrow function
            (r'class\s+\w+', 0.8),  # Class declaration
        ]
        
        pattern_score = 0
        for pattern, weight in patterns:
            if re.search(pattern, content):
                pattern_score += weight
        
        pattern_score = min(pattern_score, 1.0)
        
        # Combine checks
        basic_score = sum(syntax_checks) / len(syntax_checks)
        final_score = (basic_score + pattern_score) / 2
        
        return final_score
    
    def validate_typescript_syntax(self, content: str) -> float:
        """Validate TypeScript syntax"""
        
        # TypeScript-specific checks
        ts_checks = []
        
        # Interface checks
        if 'interface' in content:
            ts_checks.append(re.search(r'interface\s+\w+\s*{', content) is not None)
        
        # Type alias checks
        if 'type ' in content:
            ts_checks.append(re.search(r'type\s+\w+\s*=', content) is not None)
        
        # Type annotations
        if ':' in content:
            type_annotation_patterns = [
                r':\s*[a-zA-Z_$][a-zA-Z0-9_$<>,\[\]\s\|&]*',
                r':\s*{[^}]*}',
                r':\s*\[[^\]]*\]',
            ]
            ts_checks.append(any(re.search(pattern, content) for pattern in type_annotation_patterns))
        
        # Basic JavaScript syntax (inherits from JS validation)
        js_score = self.validate_javascript_syntax(content)
        
        if ts_checks:
            ts_score = sum(ts_checks) / len(ts_checks)
            return (js_score + ts_score) / 2
        else:
            return js_score * 0.9  # Penalize if no TS-specific features
    
    def validate_xml_syntax(self, content: str) -> float:
        """Validate XML syntax"""
        
        try:
            import xml.etree.ElementTree as ET
            ET.fromstring(content)
            return 1.0
        except:
            # Basic XML validation
            xml_checks = [
                content.strip().startswith('<?xml') or content.strip().startswith('<'),
                content.count('<') == content.count('>'),
                not re.search(r'<[^>]*<[^>]*>', content),  # No overlapping tags
            ]
            return sum(xml_checks) / len(xml_checks)
    
    def validate_html_syntax(self, content: str) -> float:
        """Validate HTML syntax"""
        
        # Basic HTML structure checks
        html_checks = [
            content.strip().startswith('<!DOCTYPE') or content.strip().startswith('<html'),
            content.count('<') == content.count('>'),
            re.search(r'<html[^>]*>', content, re.IGNORECASE) is not None,
            re.search(r'</html>', content, re.IGNORECASE) is not None,
        ]
        
        return sum(html_checks) / len(html_checks)
    
    def validate_css_syntax(self, content: str) -> float:
        """Validate CSS syntax"""
        
        # Basic CSS structure checks
        css_checks = [
            content.count('{') == content.count('}'),
            bool(re.search(r'[a-zA-Z_-]+\s*{', content)),
            bool(re.search(r'[a-zA-Z-]+\s*:\s*[^;]+;', content)),
        ]
        
        return sum(css_checks) / len(css_checks)
    
    def apply_semantic_coherence_filtering(self, examples: List[Dict]) -> List[Dict]:
        """Apply semantic coherence filtering"""
        filtered_examples = []
        
        with tqdm(total=len(examples), desc="Semantic coherence filtering") as pbar:
            for example in examples:
                coherence_score = self.calculate_semantic_coherence(example)
                
                if coherence_score >= self.quality_thresholds['semantic_coherence_threshold']:
                    filtered_examples.append(example)
                else:
                    self.stats['semantic_filtered'] += 1
                
                pbar.update(1)
        
        logger.info(f"Semantic coherence filtering: {len(filtered_examples):,}/{len(examples):,} passed")
        return filtered_examples
    
    def calculate_semantic_coherence(self, example: Dict) -> float:
        """Calculate semantic coherence score"""
        
        # For code examples, coherence is about logical structure
        content = example.get('content', '') or example.get('code', '') or example.get('output', '')
        
        # Code coherence indicators
        coherence_indicators = {
            'logical_flow': self.assess_logical_flow(content),
            'naming_consistency': self.assess_naming_consistency(content),
            'structure_rationality': self.assess_structure_rationality(content),
        }
        
        coherence_score = np.mean(list(coherence_indicators.values()))
        return coherence_score
    
    def assess_logical_flow(self, content: str) -> float:
        """Assess logical flow in code"""
        
        # Check for logical ordering of statements
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        if len(lines) < 3:
            return 0.5
        
        # Look for logical patterns
        patterns = [
            (r'^import|^from', 'imports'),  # Imports first
            (r'^const|^let|^var', 'declarations'),  # Declarations
            (r'^function', 'functions'),  # Functions
            (r'^\s*}', 'closing'),  # Proper closing
        ]
        
        pattern_scores = []
        for line in lines:
            line_score = 0
            for pattern, pattern_type in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    line_score += 1
            pattern_scores.append(line_score / len(patterns))
        
        return np.mean(pattern_scores)
    
    def assess_naming_consistency(self, content: str) -> float:
        """Assess naming consistency"""
        
        # Extract identifiers
        identifiers = re.findall(r'\b[a-zA-Z_$][a-zA-Z0-9_$]*\b', content)
        
        # Filter out common keywords
        keywords = {
            'function', 'class', 'const', 'let', 'var', 'if', 'else', 'for', 'while',
            'return', 'new', 'this', 'true', 'false', 'null', 'undefined', 'try', 'catch'
        }
        
        semantic_identifiers = [id for id in identifiers if id.lower() not in keywords and len(id) > 2]
        
        if len(semantic_identifiers) < 3:
            return 0.5
        
        # Check naming patterns
        camel_case = sum(1 for id in semantic_identifiers if re.match(r'^[a-z]+([A-Z][a-z]*)*$', id))
        snake_case = sum(1 for id in semantic_identifiers if '_' in id and id.islower())
        
        total_identifiers = len(semantic_identifiers)
        consistency_score = (camel_case + snake_case) / total_identifiers
        
        return consistency_score
    
    def assess_structure_rationality(self, content: str) -> float:
        """Assess structural rationality"""
        
        # Check for reasonable code structure
        structure_elements = {
            'proper_indentation': self.check_indentation(content),
            'balanced_braces': content.count('{') == content.count('}'),
            'logical_grouping': self.check_logical_grouping(content),
        }
        
        return np.mean(list(structure_elements.values()))
    
    def check_indentation(self, content: str) -> float:
        """Check code indentation consistency"""
        lines = content.split('\n')
        indent_levels = []
        
        for line in lines:
            if line.strip():
                leading_spaces = len(line) - len(line.lstrip())
                indent_levels.append(leading_spaces)
        
        if len(indent_levels) < 3:
            return 0.5
        
        # Check for consistent indentation patterns
        unique_indents = set(indent_levels)
        if len(unique_indents) <= 4:  # Reasonable number of indent levels
            return 1.0
        elif len(unique_indents) <= 8:
            return 0.7
        else:
            return 0.4
    
    def check_logical_grouping(self, content: str) -> float:
        """Check for logical code grouping"""
        
        # Look for logical blocks
        logical_markers = [
            r'{\s*',  # Block starts
            r'}\s*',  # Block ends
            r';\s*$',  # Statement ends
        ]
        
        # Count logical markers
        marker_count = sum(len(re.findall(pattern, content, re.MULTILINE)) for pattern in logical_markers)
        content_length = len(content)
        
        if content_length == 0:
            return 0.5
        
        marker_density = marker_count / (content_length / 100)  # Markers per 100 characters
        
        if 2 <= marker_density <= 20:  # Reasonable marker density
            return 1.0
        elif marker_density < 2:
            return 0.6
        else:
            return 0.7
    
    def apply_duplication_filtering(self, examples: List[Dict], dataset_name: str) -> List[Dict]:
        """Apply duplication filtering using MinHash LSH"""
        
        logger.info("Building MinHash LSH index for deduplication...")
        
        # Configure MinHash parameters
        threshold = self.config.deduplication_threshold
        num_perm = 128
        
        # Initialize LSH
        lsh = LSH(threshold=threshold, num_perm=num_perm)
        unique_examples = []
        duplicate_groups = {}
        
        with tqdm(total=len(examples), desc="Deduplication") as pbar:
            for idx, example in enumerate(examples):
                # Prepare content for hashing
                content = self.prepare_content_for_hashing(example)
                
                # Create MinHash
                minhash = MinHash(num_perm=num_perm)
                minhash.update(content.encode('utf-8'))
                
                # Query for duplicates
                query_result = lsh.query(minhash)
                
                if not query_result:
                    # New unique content
                    lsh.insert(str(idx), minhash)
                    unique_examples.append(example)
                    
                    # Store in duplicate tracking
                    duplicate_groups[str(idx)] = []
                else:
                    # Found duplicates
                    self.stats['duplicate_filtered'] += 1
                    
                    # Track duplicate relationships
                    for duplicate_idx in query_result:
                        if duplicate_idx not in duplicate_groups:
                            duplicate_groups[duplicate_idx] = []
                        duplicate_groups[duplicate_idx].append(idx)
                
                pbar.update(1)
                
                # Progress update
                if idx % 10000 == 0 and idx > 0:
                    duplicate_rate = self.stats['duplicate_filtered'] / idx
                    logger.info(f"  Progress: {idx:,}/{len(examples):,} processed, "
                              f"duplicates: {self.stats['duplicate_filtered']:,} ({duplicate_rate:.2%})")
        
        final_duplicate_rate = self.stats['duplicate_filtered'] / len(examples)
        
        logger.info(f"Deduplication completed:")
        logger.info(f"  Original examples: {len(examples):,}")
        logger.info(f"  Unique examples: {len(unique_examples):,}")
        logger.info(f"  Duplicates removed: {self.stats['duplicate_filtered']:,}")
        logger.info(f"  Final duplicate rate: {final_duplicate_rate:.2%}")
        
        # Save deduplication results
        self.save_deduplication_results(duplicate_groups, dataset_name)
        
        return unique_examples
    
    def prepare_content_for_hashing(self, example: Dict) -> str:
        """Prepare content for deduplication hashing"""
        
        content = example.get('content', '') or example.get('code', '') or example.get('output', '')
        
        # Normalize content for better deduplication
        # Remove comments
        content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove string literals (keep structure)
        content = re.sub(r'["\'`][^"\']*["\'`]', '"string"', content)
        
        # Remove numeric literals (keep structure)
        content = re.sub(r'\b\d+\b', 'number', content)
        
        # Remove identifier names (keep structure)
        content = re.sub(r'\b[a-zA-Z_$][a-zA-Z0-9_$]*\b', 'identifier', content)
        
        return content.lower().strip()
    
    def log_filtering_statistics(self, dataset_name: str):
        """Log comprehensive filtering statistics"""
        
        total = self.stats['total_examples']
        final = self.stats['final_passed']
        
        logger.info(f"Filtering statistics for {dataset_name}:")
        logger.info(f"  Total examples: {total:,}")
        logger.info(f"  Language filtered: {self.stats['language_filtered']:,} ({self.stats['language_filtered']/total:.1%})")
        logger.info(f"  Quality filtered: {self.stats['quality_filtered']:,} ({self.stats['quality_filtered']/total:.1%})")
        logger.info(f"  Syntax filtered: {self.stats['syntax_filtered']:,} ({self.stats['syntax_filtered']/total:.1%})")
        logger.info(f"  Semantic filtered: {self.stats['semantic_filtered']:,} ({self.stats['semantic_filtered']/total:.1%})")
        logger.info(f"  Duplicate filtered: {self.stats['duplicate_filtered']:,} ({self.stats['duplicate_filtered']/total:.1%})")
        logger.info(f"  Final passed: {final:,} ({final/total:.1%})")
        
        # Save statistics
        stats_path = f"cache/quality_filters/{dataset_name}_filtering_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
    
    def save_filtered_dataset(self, dataset: Dataset, dataset_name: str):
        """Save filtered dataset to disk"""
        
        output_path = f"data/processed/{dataset_name}_quality_filtered"
        dataset.save_to_disk(output_path)
        
        logger.info(f"Filtered dataset saved to {output_path}")
    
    def save_deduplication_results(self, duplicate_groups: Dict[str, List[int]], dataset_name: str):
        """Save deduplication results"""
        
        results_path = f"cache/duplicate_detection/{dataset_name}_duplicates.json"
        
        with open(results_path, 'w') as f:
            json.dump(duplicate_groups, f, indent=2)
        
        logger.info(f"Deduplication results saved to {results_path}")
    
    def get_filtering_statistics(self) -> Dict[str, Any]:
        """Get comprehensive filtering statistics"""
        
        return {
            'total_examples': self.stats['total_examples'],
            'language_filtered': self.stats['language_filtered'],
            'quality_filtered': self.stats['quality_filtered'],
            'syntax_filtered': self.stats['syntax_filtered'],
            'semantic_filtered': self.stats['semantic_filtered'],
            'duplicate_filtered': self.stats['duplicate_filtered'],
            'final_passed': self.stats['final_passed'],
            'filtering_efficiency': self.stats['final_passed'] / self.stats['total_examples'] if self.stats['total_examples'] > 0 else 0
        }

# Import for type hint (avoid circular import)
try:
    from data_preparation_pipeline import DataPreparationConfig
except ImportError:
    pass

def main():
    """Main function for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Quality Filter Pipeline')
    parser.add_argument('--config', type=str, required=True, help='Configuration file path')
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create config object
    config = DataPreparationConfig(**config_dict)
    
    # Initialize filter pipeline
    filter_pipeline = QualityFilterPipeline(config)
    
    # Test filtering
    if args.test:
        logger.info("Running in test mode")
        # Create test dataset
        test_examples = [
            {
                'content': 'function hello() { console.log("Hello World"); }',
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
        logger.info(f"Test filtering completed with {len(test_dataset)} examples")
    else:
        logger.info("Quality filter pipeline ready for use")

if __name__ == "__main__":
    main()