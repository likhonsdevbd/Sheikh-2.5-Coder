# Sheikh-2.5-Coder Data Preparation Strategy

**Author:** MiniMax Agent  
**Date:** 2025-11-06  
**Model:** Sheikh-2.5-Coder (3.09B parameters)  
**Target:** On-device deployment with XML/MDX/JavaScript specialization  

---

## 1. Executive Summary (Six Thinking Hats Synthesis)

### White Hat (Facts & Data)
Sheikh-2.5-Coder is a 3.09B parameter code language model (2.77B non-embedding parameters, 36 layers, GQA with 16Q/2KV heads, 32K context length) optimized for on-device deployment. Current research establishes five key data sources: The Stack v2 (67.5TB, 900B tokens), OpenCodeInstruct (instruction-following with unit tests), CodeSearchNet (code-comment pairs), synthetic generation methods, and comprehensive preprocessing pipelines using CodeBERT tokenization and MinHash deduplication.

### Red Hat (Intuition & Emotions)
The development team feels confident about the technical architecture but concerned about data quality at scale. There's excitement about XML/MDX/JavaScript specialization potential but anxiety about 6-12GB memory constraints affecting model capacity. The parallel thinking analysis reveals optimism about on-device capabilities but realistic concerns about training efficiency.

### Black Hat (Risks & Cautions)
**Critical Risks:**
- Data quality degradation from synthetic generation at scale
- On-device memory constraints limiting model expressiveness
- XML/MDX data sparsity compared to mainstream languages
- Preprocessing pipeline bottlenecks with 900B+ tokens
- Quality filtering false positives removing valuable code

**Mitigation Strategies:**
- Implement multi-stage quality gates with human validation sampling
- Prioritize compression techniques (quantization-aware training)
- Create XML/MDX augmentation pipelines from existing web datasets
- Deploy distributed preprocessing with checkpointing
- Use ensemble quality scoring to reduce filtering bias

### Yellow Hat (Benefits & Optimism)
**Key Opportunities:**
- Specialized XML/MDX/JavaScript capabilities create market differentiation
- On-device deployment enables privacy-preserving code assistance
- 32K context length supports complex project understanding
- GQA architecture provides efficient attention computation
- Open-source ecosystem encourages community contributions

**Strategic Advantages:**
- First-mover advantage in on-device code generation
- Reduced deployment costs compared to cloud-based alternatives
- Enhanced security through local data processing
- Faster inference times for developer workflows

### Green Hat (Creative Solutions)
**Innovation Opportunities:**
- **Hybrid Tokenization:** Combine CodeBERT subword tokens with XML-specific token streams
- **Adaptive Context Windows:** Dynamic context allocation based on project size
- **Multi-Task Joint Training:** Simultaneously optimize for completion, explanation, and generation
- **Progressive Quantization:** Train with mixed precision from the start
- **Community-Contributed Datasets:** Incentivize XML/MDX data collection through gamification

### Blue Hat (Process Control)
**Implementation Framework:**
1. **Phase 1 (Weeks 1-4):** Dataset acquisition and initial preprocessing
2. **Phase 2 (Weeks 5-8):** Quality filtering and deduplication implementation
3. **Phase 3 (Weeks 9-12):** Synthetic data generation and augmentation
4. **Phase 4 (Weeks 13-16):** Integration testing and benchmark validation
5. **Phase 5 (Weeks 17-20):** Model training and on-device optimization

---

## 2. Dataset Selection Strategy (Prioritizing XML/MDX/JavaScript Support)

### Primary Dataset Priorities

**Tier 1 - Core Code Sources (70% of training data)**
1. **The Stack v2 - train-smol-ids subset**
   - **Target Languages:** JavaScript, TypeScript, XML, HTML, CSS
   - **Estimated Size:** ~12TB (17 languages × 700GB average)
   - **Rationale:** Largest available high-quality codebase with permissive licensing
   - **XML/MDX Strategy:** Prioritize XML (35%), HTML (25%), Markdown (15%) subsets

2. **OpenCodeInstruct (Enhanced)**
   - **Target Size:** ~50M instruction pairs
   - **Language Distribution:** 
     - JavaScript/TypeScript: 40%
     - XML configuration files: 20%
     - MDX/React components: 15%
     - General programming: 25%
   - **Quality Filter:** Unit test pass rate >70%

**Tier 2 - Specialized Sources (20% of training data)**
3. **CodeSearchNet (XML/MDX Enhanced)**
   - **Repository Focus:** React projects with extensive MDX usage
   - **Code-Comment Quality:** Minimum 0.8 semantic similarity score
   - **Augmentation:** Add 200K XML documentation examples from Mozilla MDN

4. **Web Development Datasets**
   - **Next.js Documentation:** 50K XML/MDX examples
   - **React Component Library:** 100K JSX/TSX examples
   - **Vue.js Documentation:** 30K Vue template examples

**Tier 3 - Synthetic & Augmented (10% of training data)**
5. **Domain-Specific Generation**
   - **React MDX Components:** 100K examples via AST mutations
   - **XML Configuration Templates:** 75K examples from real projects
   - **JavaScript Algorithm Explanations:** 50K generated with teacher models

### Data Distribution Strategy
```yaml
Total Training Tokens: ~500B (suitable for 3B parameter model)
Language Distribution:
  JavaScript/TypeScript: 35% (175B tokens)
  XML/HTML: 25% (125B tokens)
  MDX/Markdown: 15% (75B tokens)
  CSS/SCSS: 10% (50B tokens)
  Other Languages: 15% (75B tokens)
```

---

## 3. The Stack v2 Integration (train-smol-ids Configuration)

### Dataset Acquisition Commands
```bash
# Download using BigQuery (recommended for scale)
pip install google-cloud-bigquery
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account.json"

# Query for target languages
bq query --use_legacy_sql=false \
  'SELECT content, language 
   FROM `bigquery-public-data.github_repos.contents` 
   WHERE language IN ("JavaScript", "TypeScript", "XML", "HTML", "CSS")
     AND content IS NOT NULL
     AND LENGTH(content) > 100
     AND LENGTH(content) < 100000
   LIMIT 500000000'

# Alternative: Direct HuggingFace download
pip install datasets
from datasets import load_dataset
dataset = load_dataset("bigcode/the-stack-smol-ids", 
                      data_dir="data/programming_languages_subset")
```

### Preprocessing Configuration
```python
# Stack v2 preprocessing pipeline
from datasets import Dataset
import re
from typing import List, Dict

class StackV2Preprocessor:
    def __init__(self):
        self.language_filters = {
            'javascript': {
                'extensions': ['.js', '.jsx', '.mjs'],
                'min_length': 50,
                'max_length': 50000,
                'quality_score': 0.7
            },
            'typescript': {
                'extensions': ['.ts', '.tsx'],
                'min_length': 50,
                'max_length': 50000,
                'quality_score': 0.75
            },
            'xml': {
                'extensions': ['.xml', '.xsd', '.svg', '.xhtml'],
                'min_length': 30,
                'max_length': 30000,
                'quality_score': 0.8
            },
            'html': {
                'extensions': ['.html', '.htm'],
                'min_length': 100,
                'max_length': 40000,
                'quality_score': 0.7
            }
        }
    
    def filter_quality(self, content: str, language: str) -> bool:
        """Apply quality filters specific to language"""
        config = self.language_filters.get(language.lower())
        if not config:
            return False
            
        # Length checks
        if not (config['min_length'] <= len(content) <= config['max_length']):
            return False
            
        # Language-specific patterns
        if language.lower() == 'xml':
            xml_patterns = [
                r'<\?xml[^>]*\?>',  # XML declaration
                r'<[a-zA-Z][^>]*>',  # Valid tags
                r'</[a-zA-Z][^>]*>',  # Closing tags
            ]
            quality_score = sum(1 for pattern in xml_patterns 
                              if re.search(pattern, content))
            return quality_score >= 3
            
        elif language.lower() in ['javascript', 'typescript']:
            js_patterns = [
                r'\b(function|const|let|var|class|import|export)\b',
                r'[{}();]',  # Basic syntax
                r'[a-zA-Z_$][a-zA-Z0-9_$]*',  # Identifiers
            ]
            quality_score = sum(1 for pattern in js_patterns 
                              if re.search(pattern, content))
            return quality_score >= 4
            
        return True
    
    def deduplicate_content(self, dataset: Dataset) -> Dataset:
        """Remove near-duplicates using MinHash LSH"""
        from datasketch import MinHash, LSH
        
        lsh = LSH(threshold=0.8, num_perm=128)
        unique_contents = []
        
        for idx, example in enumerate(dataset):
            content = example['content']
            minhash = MinHash(num_perm=128)
            minhash.update(content.encode('utf-8'))
            
            # Check for duplicates
            query_result = lsh.query(minhash)
            if not query_result:
                lsh.insert(idx, minhash)
                unique_contents.append(example)
                
        return Dataset.from_list(unique_contents)
```

### Target Statistics After Filtering
```yaml
Stack v2 Processed Dataset:
  Raw Size: ~12TB
  After Language Filtering: ~4.2TB (35% reduction)
  After Quality Filtering: ~2.8TB (33% further reduction)
  After Deduplication: ~2.1TB (25% further reduction)
  
Language Breakdown:
  JavaScript: 840GB
  TypeScript: 420GB
  XML: 350GB
  HTML: 280GB
  CSS: 210GB
```

---

## 4. Instruction-Following Data (OpenCodeInstruct + Quality Filtering)

### Enhanced OpenCodeInstruct Strategy
```bash
# Download and process OpenCodeInstruct
git clone https://github.com/OpenLLMAI/OpenCodeInstruct.git
cd OpenCodeInstruct
pip install -r requirements.txt

# Process with XML/MDX focus
python scripts/filter_for_web_dev.py \
  --input_dir data/raw \
  --output_dir data/processed \
  --languages javascript,typescript,xml,html,jsx,tsx,mdx \
  --min_quality_score 0.75 \
  --max_length 8192 \
  --unit_test_validation True
```

### Custom Data Generation Pipeline
```python
# Enhanced instruction generation for web development
class WebDevInstructionGenerator:
    def __init__(self):
        self.templates = {
            'xml_generation': [
                "Create a complete XML schema for {topic}",
                "Generate XML configuration for {framework} deployment",
                "Write XML transformation (XSLT) for {data_type}",
                "Create XML sitemap for {website_type}"
            ],
            'mdx_creation': [
                "Create interactive MDX component for {library}",
                "Generate MDX documentation with code examples for {framework}",
                "Write MDX blog post with {feature_type} examples",
                "Create MDX component with {styling_library} integration"
            ],
            'js_enhancement': [
                "Optimize this JavaScript {algorithm_type} for {performance_target}",
                "Refactor this React component to use {pattern_type} pattern",
                "Add TypeScript types for this {library_name} interface",
                "Implement error handling for this {api_type} API call"
            ]
        }
    
    def generate_instructions(self, count: int = 100000) -> List[Dict]:
        instructions = []
        
        for _ in range(count):
            # Select template type based on target distribution
            template_type = np.random.choice(
                ['xml_generation', 'mdx_creation', 'js_enhancement'],
                p=[0.25, 0.25, 0.5]
            )
            
            template = random.choice(self.templates[template_type])
            context = self.generate_context(template_type)
            
            instruction = template.format(**context)
            expected_output = self.generate_expected_output(instruction, context)
            
            instructions.append({
                'instruction': instruction,
                'input': context.get('code_snippet', ''),
                'output': expected_output,
                'task_type': template_type,
                'domain': 'web_development',
                'difficulty': self.assess_difficulty(instruction)
            })
            
        return instructions
```

### Quality Filtering Implementation
```python
# Multi-stage quality filtering for instruction data
class InstructionQualityFilter:
    def __init__(self):
        self.quality_thresholds = {
            'semantic_similarity': 0.7,
            'code_syntax_validity': 0.85,
            'instruction_clarity': 0.8,
            'output_completeness': 0.9
        }
    
    def filter_instructions(self, dataset: Dataset) -> Dataset:
        """Apply comprehensive quality filtering"""
        filtered_data = []
        
        for example in dataset:
            quality_scores = self.calculate_quality_scores(example)
            
            if all(score >= self.quality_thresholds[key] 
                   for key, score in quality_scores.items()):
                filtered_data.append(example)
                
        return Dataset.from_list(filtered_data)
    
    def calculate_quality_scores(self, example: Dict) -> Dict[str, float]:
        """Calculate multi-dimensional quality scores"""
        scores = {}
        
        # Semantic similarity (instruction-input alignment)
        scores['semantic_similarity'] = self.bert_similarity(
            example['instruction'], example.get('input', '')
        )
        
        # Code syntax validity
        scores['code_syntax_validity'] = self.validate_code_syntax(
            example.get('output', '')
        )
        
        # Instruction clarity (readability score)
        scores['instruction_clarity'] = self.calculate_readability(
            example['instruction']
        )
        
        # Output completeness (length and structure)
        scores['output_completeness'] = self.assess_output_completeness(
            example['output']
        )
        
        return scores
```

---

## 5. Code-Comment Pairs (CodeSearchNet + CAT Cleaning)

### Enhanced CodeSearchNet Processing
```python
# Enhanced CodeSearchNet pipeline with XML/MDX focus
from datasets import load_dataset
import subprocess
import json

class CodeSearchNetProcessor:
    def __init__(self):
        self.language_priorities = {
            'javascript': 0.4,
            'typescript': 0.3,
            'xml': 0.15,
            'html': 0.1,
            'css': 0.05
        }
    
    def download_and_filter(self) -> Dataset:
        """Download and filter CodeSearchNet for target languages"""
        # Download CodeSearchNet
        datasets = {}
        for lang in ['javascript', 'typescript']:
            datasets[lang] = load_dataset("code_search_net", lang)
        
        # Process and filter
        filtered_examples = []
        
        for lang, dataset in datasets.items():
            for split in ['train', 'valid', 'test']:
                examples = dataset[split]
                
                # Language-specific filtering
                if lang in ['javascript', 'typescript']:
                    filtered = self.filter_js_ts_examples(examples)
                else:
                    continue
                    
                filtered_examples.extend(filtered)
        
        return Dataset.from_list(filtered_examples)
    
    def filter_js_ts_examples(self, examples: Dataset) -> List[Dict]:
        """Filter JavaScript/TypeScript examples for quality"""
        filtered = []
        
        for example in examples:
            # Quality checks
            if (len(example['func_documentation_string']) < 50 or
                len(example['func_documentation_string']) > 2000 or
                len(example['code']) < 100 or
                len(example['code']) > 10000):
                continue
                
            # Semantic quality check
            similarity = self.calculate_doc_code_similarity(
                example['func_documentation_string'], example['code']
            )
            
            if similarity > 0.6:
                # Add XML/MDX context if applicable
                example['extended_context'] = self.add_web_context(example)
                filtered.append(example)
                
        return filtered
    
    def add_web_context(self, example: Dict) -> Dict:
        """Add XML/MDX context for web development examples"""
        # Detect if function is part of web framework
        framework_indicators = {
            'react': ['React', 'JSX', 'Component', 'useState', 'useEffect'],
            'vue': ['Vue', 'template', 'script', 'style'],
            'angular': ['Angular', '@Component', 'NgModule'],
            'xml': ['XML', 'schema', 'XSD', 'XSLT']
        }
        
        framework = self.detect_framework(example['code'])
        example['framework_type'] = framework
        
        return example
```

### CAT (Clean, Annotate, Transform) Pipeline Implementation
```python
# CAT (Clean, Annotate, Transform) pipeline
class CATProcessor:
    def __init__(self):
        self.cleaning_rules = {
            'code_removal': [
                r'//\s*TODO[^\n]*',
                r'/\*.*TODO.*\*/',
                r'console\.log[^\n]*',
                r'alert\([^\)]*\)',
                r'debugger;'
            ],
            'comment_fixes': [
                (r'/\*\s*\*\s*([^}]+)\s*\*/', r'/** \1 */'),  # Fix malformed docstrings
                (r'//\s*([^/]+)//', r'// \1'),  # Remove trailing slashes
            ]
        }
    
    def clean_code(self, code: str) -> str:
        """Apply cleaning rules to code"""
        cleaned = code
        
        for pattern in self.cleaning_rules['code_removal']:
            cleaned = re.sub(pattern, '', cleaned)
            
        for pattern, replacement in self.cleaning_rules['comment_fixes']:
            cleaned = re.sub(pattern, replacement, cleaned)
            
        return cleaned.strip()
    
    def annotate_code(self, code: str, language: str) -> str:
        """Add language-specific annotations"""
        if language == 'xml':
            return self.annotate_xml(code)
        elif language in ['javascript', 'typescript']:
            return self.annotate_js(code)
        else:
            return code
    
    def transform_for_learning(self, code: str, comments: str, language: str) -> Dict:
        """Transform code-comment pairs for model training"""
        # Create multiple learning objectives
        transformations = []
        
        # 1. Code completion from comments
        transformations.append({
            'task_type': 'comment_to_code',
            'input': comments,
            'target': code,
            'language': language
        })
        
        # 2. Comment generation from code
        transformations.append({
            'task_type': 'code_to_comment',
            'input': code,
            'target': comments,
            'language': language
        })
        
        # 3. Code explanation (detailed)
        if len(comments) > 100:  # Only for detailed comments
            transformations.append({
                'task_type': 'code_explanation',
                'input': code,
                'target': self.expand_explanation(comments),
                'language': language
            })
        
        return transformations
```

---

## 6. Synthetic Data Generation (LLM-based + AST Mutations)

### LLM-Based Generation Pipeline
```python
# Enhanced synthetic data generation for web technologies
import ast
import random
from typing import List, Dict, Optional

class WebDevSyntheticGenerator:
    def __init__(self):
        self.generator_models = {
            'gpt3.5': 'openai/gpt-3.5-turbo',
            'codellama': 'codellama/CodeLlama-7b-Instruct-hf',
            'deepseek': 'deepseek-ai/deepseek-coder-6.7b-instruct'
        }
        
        self.generation_strategies = {
            'self_instruct': self.self_instruct_generation,
            'evol_instruct': self.evol_instruct_generation,
            'chain_of_thought': self.chain_of_thought_generation,
            'domain_specific': self.domain_specific_generation
        }
    
    def self_instruct_generation(self, seed_code: str, count: int = 1000) -> List[Dict]:
        """Generate instructions using Self-Instruct methodology"""
        instructions = []
        
        for _ in range(count):
            # Generate diverse instruction templates
            template = self.select_instruction_template(seed_code)
            context = self.generate_context(template)
            
            instruction = template.format(**context)
            response = self.generate_with_teacher_model(instruction)
            
            instructions.append({
                'instruction': instruction,
                'input': seed_code,
                'output': response,
                'generation_method': 'self_instruct',
                'quality_score': self.assess_generation_quality(instruction, response)
            })
            
        return instructions
    
    def evol_instruct_generation(self, base_examples: List[Dict], count: int = 1000) -> List[Dict]:
        """Generate more complex examples using Evol-Instruct"""
        evolved_examples = []
        
        for _ in range(count):
            # Select base example
            base = random.choice(base_examples)
            
            # Apply evolution operations
            evolved_instruction = self.evolve_instruction(base['instruction'])
            evolved_output = self.evolve_output(base['output'])
            
            evolved_examples.append({
                'instruction': evolved_instruction,
                'input': base['input'],
                'output': evolved_output,
                'generation_method': 'evol_instruct',
                'evolution_operations': self.record_evolution_operations(),
                'difficulty_increase': self.calculate_difficulty_increase(base, evolved)
            })
            
        return evolved_examples
    
    def domain_specific_generation(self) -> Dict[str, List[Dict]]:
        """Generate domain-specific examples for XML/MDX/JavaScript"""
        synthetic_data = {}
        
        # XML generation
        synthetic_data['xml'] = self.generate_xml_examples(10000)
        
        # MDX generation
        synthetic_data['mdx'] = self.generate_mdx_examples(8000)
        
        # JavaScript/React generation
        synthetic_data['javascript'] = self.generate_js_examples(15000)
        
        return synthetic_data
```

### AST Mutation Strategies
```python
# Advanced AST mutation for code augmentation
class ASTMutator:
    def __init__(self):
        self.mutation_operators = {
            'javascript': [
                self.replace_variable_names,
                self.add_error_handling,
                self.insert_logging_statements,
                self.modify_function_signatures,
                self.add_type_annotations
            ],
            'xml': [
                self.modify_attribute_values,
                self.add_nested_elements,
                self.reorganize_element_structure,
                self.add_namespace_declarations,
                self.insert_processing_instructions
            ]
        }
    
    def mutate_code(self, code: str, language: str, mutation_rate: float = 0.3) -> str:
        """Apply AST-based mutations to code"""
        if language == 'javascript':
            return self.mutate_js_code(code, mutation_rate)
        elif language == 'xml':
            return self.mutate_xml_code(code, mutation_rate)
        else:
            return code
    
    def mutate_js_code(self, code: str, mutation_rate: float) -> str:
        """Mutate JavaScript/TypeScript code using AST"""
        try:
            # Parse to AST
            tree = ast.parse(code)
            
            # Apply random mutations
            mutations_applied = []
            for node in ast.walk(tree):
                if random.random() < mutation_rate:
                    mutation = random.choice(self.mutation_operators['javascript'])
                    new_node = mutation(node)
                    if new_node:
                        mutations_applied.append(mutation.__name__)
            
            # Generate mutated code
            mutated_code = ast.unparse(tree)
            
            # Add metadata
            return {
                'code': mutated_code,
                'mutations_applied': mutations_applied,
                'original_code': code,
                'mutation_count': len(mutations_applied)
            }
            
        except SyntaxError:
            return {'code': code, 'mutations_applied': [], 'error': 'syntax_error'}
```

---

## 7. Preprocessing Pipeline (CodeBERT Tokenization + MinHash Deduplication)

### CodeBERT Tokenization Strategy
```python
# CodeBERT-based preprocessing pipeline
from transformers import AutoTokenizer
from typing import List, Dict, Tuple
import hashlib
from datasketch import MinHash, LSH

class CodeBERTPreprocessor:
    def __init__(self, model_name: str = "microsoft/codebert-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.model_max_length = 8192  # Increased for long code sequences
        
        # Language-specific tokenization configurations
        self.language_configs = {
            'javascript': {
                'special_tokens': ['<js>', '</js>', '<function>', '</function>'],
                'context_tokens': ['<react>', '<node>', '<browser>']
            },
            'xml': {
                'special_tokens': ['<xml>', '</xml>', '<element>', '</element>'],
                'context_tokens': ['<web>', '<config>', '<schema>']
            },
            'mdx': {
                'special_tokens': ['<mdx>', '</mdx>', '<component>', '</component>'],
                'context_tokens': ['<react>', '<markdown>', '<interactive>']
            }
        }
    
    def tokenize_code(self, code: str, language: str, max_length: int = 1024) -> Dict:
        """Tokenize code with language-specific enhancements"""
        config = self.language_configs.get(language, {})
        
        # Add language-specific tokens
        enhanced_code = self.add_language_tokens(code, language)
        
        # Tokenize with CodeBERT
        tokens = self.tokenizer.encode_plus(
            enhanced_code,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_special_tokens_mask=True
        )
        
        # Calculate statistics
        stats = self.calculate_tokenization_stats(enhanced_code, tokens)
        
        return {
            'tokens': tokens,
            'input_ids': tokens['input_ids'].squeeze().tolist(),
            'attention_mask': tokens['attention_mask'].squeeze().tolist(),
            'special_tokens_mask': tokens['special_tokens_mask'].squeeze().tolist(),
            'statistics': stats,
            'language': language,
            'original_code': code
        }
```

### MinHash Deduplication System
```python
# Advanced deduplication using MinHash + LSH
class AdvancedDeduplicator:
    def __init__(self, threshold: float = 0.8, num_perm: int = 128):
        self.threshold = threshold
        self.num_perm = num_perm
        self.lsh = LSH(threshold=threshold, num_perm=num_perm)
        self.minhash_registry = {}
        
    def build_dedup_index(self, dataset: Dataset) -> Dict[str, List[int]]:
        """Build deduplication index using MinHash LSH"""
        print("Building MinHash deduplication index...")
        
        duplicates = {}
        total_examples = len(dataset)
        
        for idx, example in enumerate(dataset):
            # Create content representation
            content = self.preprocess_for_hashing(example)
            
            # Create MinHash
            minhash = MinHash(num_perm=self.num_perm)
            minhash.update(content.encode('utf-8'))
            
            # Query existing index
            query_result = self.lsh.query(minhash)
            
            if not query_result:
                # New unique content
                self.lsh.insert(str(idx), minhash)
                self.minhash_registry[str(idx)] = minhash
            else:
                # Found duplicates
                for duplicate_idx in query_result:
                    if duplicate_idx not in duplicates:
                        duplicates[duplicate_idx] = []
                    duplicates[duplicate_idx].append(idx)
            
            # Progress tracking
            if idx % 10000 == 0:
                print(f"Processed {idx}/{total_examples} examples")
        
        print(f"Deduplication complete. Found {len(duplicates)} duplicate groups")
        return duplicates
```

---

## 8. Quality Assurance & Metrics (MMLU Benchmarking Strategy)

### MMLU Benchmark Implementation
```python
# MMLU benchmark adaptation for code generation
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

class MMLUCodeBenchmark:
    def __init__(self, model_path: str, tokenizer_path: str):
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model.eval()
        
        # MMLU domains adapted for coding
        self.code_domains = [
            'programming_fundamentals',
            'web_development',
            'data_structures',
            'algorithms',
            'software_engineering',
            'cybersecurity',
            'databases',
            'computer_networks'
        ]
    
    def create_code_mmlu_dataset(self) -> Dict[str, List[Dict]]:
        """Create MMLU-style dataset for coding evaluation"""
        dataset = {}
        
        for domain in self.code_domains:
            domain_questions = self.generate_domain_questions(domain)
            dataset[domain] = domain_questions
        
        return dataset
    
    def generate_web_dev_questions(self) -> List[Dict]:
        """Generate web development questions"""
        questions = [
            {
                'question': 'Which of the following is the correct way to create a React component?',
                'options': [
                    'function MyComponent() { return <div>Hello</div>; }',
                    'class MyComponent extends React.Component { render() { return <div>Hello</div>; } }',
                    'const MyComponent = () => <div>Hello</div>;',
                    'All of the above'
                ],
                'correct_answer': 3,
                'domain': 'web_development',
                'difficulty': 'medium',
                'context': 'react_components'
            },
            {
                'question': 'What is the purpose of the useState hook in React?',
                'options': [
                    'To handle side effects',
                    'To manage component state',
                    'To make API calls',
                    'To style components'
                ],
                'correct_answer': 1,
                'domain': 'web_development',
                'difficulty': 'easy',
                'context': 'react_hooks'
            },
            {
                'question': 'Which XML namespace declaration is required for XSLT transformations?',
                'options': [
                    'xmlns:xsl="http://www.w3.org/1999/XSL/Transform"',
                    'xmlns="http://www.w3.org/TR/xslt"',
                    'xmlns:transform="http://www.w3.org/xslt"',
                    'xmlns:xalan="http://xml.apache.org/xslt"'
                ],
                'correct_answer': 0,
                'domain': 'web_development',
                'difficulty': 'hard',
                'context': 'xml_xslt'
            }
        ]
        
        # Generate additional questions programmatically
        for _ in range(100):  # Generate 100 questions per domain
            question = self.generate_random_web_question()
            if question:
                questions.append(question)
        
        return questions
```

### Code-Specific Evaluation Metrics
```python
# Advanced evaluation metrics for code generation
class CodeEvaluationMetrics:
    def __init__(self):
        self.bleu_weights = (0.25, 0.25, 0.25, 0.25)
        self.bertscore_model = 'microsoft/codebert-base'
    
    def evaluate_code_completion(self, references: List[str], predictions: List[str]) -> Dict[str, float]:
        """Evaluate code completion quality"""
        metrics = {}
        
        # BLEU score
        metrics['bleu'] = self.calculate_bleu(references, predictions)
        
        # CodeBLEU (simplified version)
        metrics['codebleu'] = self.calculate_codebleu(references, predictions)
        
        # BERTScore
        metrics['bertscore'] = self.calculate_bertscore(references, predictions)
        
        # Syntax validity
        metrics['syntax_validity'] = self.calculate_syntax_validity(predictions)
        
        # Semantic similarity
        metrics['semantic_similarity'] = self.calculate_semantic_similarity(
            references, predictions
        )
        
        return metrics
    
    def calculate_syntax_validity(self, code_predictions: List[str]) -> float:
        """Calculate percentage of predictions with valid syntax"""
        valid_count = 0
        
        for code in code_predictions:
            if self.validate_syntax(code):
                valid_count += 1
        
        return valid_count / len(code_predictions) if code_predictions else 0
    
    def validate_syntax(self, code: str) -> bool:
        """Validate code syntax for different languages"""
        try:
            # Try to parse as JavaScript
            if any(keyword in code for keyword in ['function', 'const', 'let', 'var']):
                import subprocess
                result = subprocess.run(['node', '-c'], 
                                      input=code, 
                                      text=True, 
                                      capture_output=True)
                return result.returncode == 0
            
            # Try to parse as XML
            if code.strip().startswith('<'):
                import xml.etree.ElementTree as ET
                ET.fromstring(code)
                return True
            
            return False
        except:
            return False
```

---

## 9. On-Device Optimization Considerations (3.09B Parameter Constraints)

### Memory Optimization Strategy
```python
# On-device optimization for 3.09B parameter model
import torch
import torch.nn as nn
from transformers import BitsAndBytesConfig
from typing import Dict, Tuple

class OnDeviceOptimizer:
    def __init__(self, target_memory_gb: float = 8.0):
        self.target_memory_gb = target_memory_gb
        self.quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_skip_modules=["embed_tokens", "lm_head"]
        )
    
    def calculate_memory_requirements(self, model_config: Dict) -> Dict[str, float]:
        """Calculate memory requirements for different configurations"""
        base_memory_gb = 3.09 * 4 / 1024  # 3.09B parameters * 4 bytes/float32
        
        memory_breakdown = {
            'base_model_fp32': base_memory_gb,
            'base_model_fp16': base_memory_gb / 2,
            'base_model_int8': base_memory_gb / 4,
            'base_model_int4': base_memory_gb / 8,
            'with_optimizer_states': base_memory_gb * 1.5,
            'with_gradient_checkpointing': base_memory_gb * 0.7,
            'estimated_runtime': 0
        }
        
        # Calculate runtime memory (model + activations)
        runtime_memory = self.estimate_runtime_memory(model_config)
        memory_breakdown['estimated_runtime'] = runtime_memory
        
        return memory_breakdown
    
    def estimate_runtime_memory(self, config: Dict) -> float:
        """Estimate runtime memory including activations"""
        # Estimate activation memory
        batch_size = config.get('batch_size', 1)
        seq_length = config.get('seq_length', 2048)
        hidden_size = config.get('hidden_size', 2048)
        
        # Attention activation memory
        attention_memory = (batch_size * seq_length * seq_length * 4) / (1024**3)  # GB
        
        # Feed-forward activation memory
        ff_memory = (batch_size * seq_length * hidden_size * 8) / (1024**3)  # GB
        
        # Total runtime memory
        runtime_memory = attention_memory + ff_memory
        
        return runtime_memory
```

### Inference Optimization
```python
# Inference optimization for on-device deployment
class InferenceOptimizer:
    def __init__(self):
        self.optimization_strategies = {
            'flash_attention': self.enable_flash_attention,
            'gradient_checkpointing': self.enable_gradient_checkpointing,
            'mixed_precision': self.enable_mixed_precision,
            'dynamic_batching': self.enable_dynamic_batching
        }
    
    def optimize_inference(self, model: nn.Module, 
                          optimization_level: str = 'medium') -> nn.Module:
        """Apply inference optimizations based on optimization level"""
        
        if optimization_level == 'light':
            model = self.enable_mixed_precision(model)
        elif optimization_level == 'medium':
            model = self.enable_flash_attention(model)
            model = self.enable_gradient_checkpointing(model)
        elif optimization_level == 'aggressive':
            model = self.enable_all_optimizations(model)
        
        return model
    
    def enable_flash_attention(self, model: nn.Module) -> nn.Module:
        """Enable Flash Attention for memory efficiency"""
        try:
            from flash_attn import flash_attn_func
            
            # Replace attention implementation with Flash Attention
            for name, module in model.named_modules():
                if 'attention' in name.lower():
                    # Create Flash Attention wrapper
                    flash_attn_wrapper = FlashAttentionWrapper(module)
                    # Replace module (implementation depends on specific model)
                    # self.replace_module(model, name, flash_attn_wrapper)
            
        except ImportError:
            print("Flash Attention not available, skipping optimization")
        
        return model
```

---

## 10. Implementation Roadmap (Specific Tools and Configurations)

### Phase 1: Dataset Acquisition & Initial Preprocessing (Weeks 1-4)

#### Week 1: Infrastructure Setup
```bash
# Environment setup
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install transformers==4.30.0 datasets==2.14.0 accelerate==0.20.0
pip install bitsandbytes==0.41.0 safetensors==0.3.0
pip install google-cloud-bigquery datasets[bigquery]
pip install datasketch==1.6.4 nltk==3.8.1 rouge==1.1.1

# Install language-specific tools
npm install -g @babel/parser @babel/traverse @babel/types
pip install tree-sitter==0.20.0

# Setup directory structure
mkdir -p {data/{raw,processed,tokenized},models,logs,scripts,evaluation}
cd data
```

#### Week 2: The Stack v2 Integration
```python
# scripts/stack_v2_download.py
import os
from datasets import load_dataset
from datasets.dataset_dict import DatasetDict

def download_stack_v2_subset():
    """Download and process Stack v2 subset"""
    
    # Configuration
    target_languages = ['javascript', 'typescript', 'xml', 'html', 'css']
    max_examples_per_lang = 1000000  # 1M examples per language
    
    # Download dataset
    print("Downloading Stack v2 dataset...")
    dataset = load_dataset("bigcode/the-stack-smol-ids", 
                          data_dir="programming_languages_subset")
    
    # Process each language
    processed_data = {}
    for lang in target_languages:
        print(f"Processing {lang} data...")
        
        if lang in dataset:
            lang_data = dataset[lang]
            
            # Filter and clean
            filtered_data = filter_language_data(lang_data, lang)
            
            # Deduplicate
            deduped_data = deduplicate_data(filtered_data)
            
            # Quality filter
            quality_filtered = apply_quality_filters(deduped_data, lang)
            
            processed_data[lang] = quality_filtered
            
            print(f"  {lang}: {len(quality_filtered)} examples after processing")
    
    # Save processed data
    for lang, data in processed_data.items():
        data.save_to_disk(f"data/processed/stack_v2_{lang}")
    
    return processed_data

if __name__ == "__main__":
    download_stack_v2_subset()
```

#### Week 3: Instruction Dataset Processing
```python
# scripts/process_instructions.py
import json
from datasets import Dataset

def process_instruction_datasets():
    """Process and enhance instruction datasets"""
    
    # Download OpenCodeInstruct
    print("Downloading OpenCodeInstruct...")
    instruct_dataset = load_dataset("bigcode/instructcodet5p-px")

    # Process with quality filtering
    enhanced_instructions = []
    
    for example in instruct_dataset['train']:
        # Language detection
        detected_lang = detect_programming_language(example['code'])
        
        if detected_lang in ['javascript', 'typescript', 'xml', 'html']:
            # Quality scoring
            quality_score = calculate_instruction_quality(example)
            
            if quality_score > 0.75:
                # Add web development context
                enhanced_example = add_web_dev_context(example, detected_lang)
                enhanced_instructions.append(enhanced_example)
    
    # Save enhanced instructions
    enhanced_dataset = Dataset.from_list(enhanced_instructions)
    enhanced_dataset.save_to_disk("data/processed/enhanced_instructions")
    
    print(f"Enhanced instructions: {len(enhanced_instructions)} examples")

if __name__ == "__main__":
    process_instruction_datasets()
```

### Phase 2: Quality Filtering & Deduplication (Weeks 5-8)

#### Week 5: Advanced Deduplication System
```python
# scripts/advanced_deduplication.py
from datasketch import MinHash, LSH
from datasets import Dataset
import numpy as np

class AdvancedDeduplicator:
    def __init__(self, threshold=0.8, num_perm=128):
        self.threshold = threshold
        self.num_perm = num_perm
        self.lsh = LSH(threshold=threshold, num_perm=num_perm)
    
    def deduplicate_dataset(self, dataset_path: str, language: str):
        """Advanced deduplication with semantic similarity"""
        
        dataset = Dataset.load_from_disk(dataset_path)
        duplicates = self.find_duplicates(dataset)
        
        # Remove duplicates, keeping highest quality
        unique_data = self.remove_duplicates(dataset, duplicates)
        
        # Save deduplicated dataset
        unique_dataset = Dataset.from_list(unique_data)
        unique_dataset.save_to_disk(f"{dataset_path}_deduped")
        
        return unique_dataset
```

### Phase 3: Synthetic Data Generation (Weeks 9-12)

#### Week 9: LLM-Based Generation Setup
```bash
# Setup synthetic data generation environment
pip install openai anthropic

# Configure API keys
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# Create synthetic data generation script
touch scripts/synthetic_generation.py
chmod +x scripts/synthetic_generation.py
```

### Phase 4: Integration & Benchmarking (Weeks 13-16)

#### Week 13: Model Integration Testing
```python
# scripts/integration_test.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def test_model_integration():
    """Test data integration with model architecture"""
    
    # Load model configuration
    model_config = {
        'model_name': 'microsoft/phi-2',
        'vocab_size': 51200,
        'max_position_embeddings': 2048,
        'num_attention_heads': 32,
        'num_hidden_layers': 36,
        'intermediate_size': 8192
    }
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config['model_name'])
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load sample data
    sample_data = load_sample_processed_data()
    
    # Test tokenization
    tokenized_data = []
    for example in sample_data[:1000]:  # Test with 1000 examples
        tokenized = tokenizer(
            example['content'],
            max_length=1024,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        tokenized_data.append(tokenized)
    
    print(f"Tokenization test completed with {len(tokenized_data)} examples")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Special tokens: {tokenizer.all_special_tokens}")
    
    return tokenized_data
```

### Phase 5: Final Training & Optimization (Weeks 17-20)

#### Week 17: Training Configuration
```bash
# Setup training environment
pip install deepspeed fairscale wandb

# Create training script
touch scripts/train_model.py
chmod +x scripts/train_model.py
```

#### Week 18: Training Execution
```python
# scripts/training_config.py
training_config = {
    'model_name_or_path': 'microsoft/phi-2',
    'output_dir': './outputs/sheikh-2.5-coder',
    'per_device_train_batch_size': 8,
    'per_device_eval_batch_size': 8,
    'gradient_accumulation_steps': 4,
    'learning_rate': 1e-4,
    'num_train_epochs': 3,
    'logging_steps': 100,
    'save_steps': 1000,
    'eval_steps': 1000,
    'warmup_steps': 1000,
    'max_grad_norm': 1.0,
    'weight_decay': 0.01,
    'save_total_limit': 3,
    'load_best_model_at_end': True,
    'report_to': 'wandb',
    'run_name': 'sheikh-2.5-coder-training'
}
```

### Success Metrics & Validation

#### Technical Metrics
```yaml
Model Performance Targets:
  MMLU Code Score: >60% accuracy
  HumanEval: >40% pass@1
  CodeBLEU: >0.65
  Syntax Validity: >95%
  Semantic Coherence: >0.80

On-Device Performance:
  Memory Footprint: <8GB (INT8 quantized)
  Inference Speed: <100ms for 512 token completion
  Context Length: 32K tokens
  Battery Impact: <5% per inference session
```

#### Quality Validation Pipeline
```python
# Quality validation at each phase
class QualityValidator:
    def __init__(self):
        self.thresholds = {
            'data_quality': 0.85,
            'duplication_rate': <0.05,
            'language_accuracy': 0.95,
            'syntax_validity': 0.90,
            'semantic_coherence': 0.75
        }
    
    def validate_phase_completion(self, phase: str, outputs: Dict):
        """Validate that each phase meets quality thresholds"""
        
        validation_results = {}
        
        if phase == "dataset_acquisition":
            validation_results = self.validate_dataset_acquisition(outputs)
        elif phase == "quality_filtering":
            validation_results = self.validate_quality_filtering(outputs)
        elif phase == "synthetic_generation":
            validation_results = self.validate_synthetic_generation(outputs)
        
        # Check all thresholds met
        all_passed = all(
            validation_results[metric] >= self.thresholds[metric] 
            for metric in validation_results
        )
        
        return {
            'phase': phase,
            'validation_results': validation_results,
            'all_thresholds_met': all_passed,
            'blocking_issues': self.identify_blocking_issues(validation_results)
        }
```

### Deployment Readiness Checklist
- [ ] Dataset quality validation completed (>95% samples pass)
- [ ] Deduplication implemented (duplication rate <5%)
- [ ] Synthetic data diversity validated (DCS score >0.7)
- [ ] On-device memory requirements confirmed (<8GB)
- [ ] Inference optimization applied (Flash Attention, quantization)
- [ ] MMLU benchmarking completed (>60% accuracy)
- [ ] Code generation quality validated (CodeBLEU >0.65)
- [ ] Performance testing on target hardware completed
- [ ] Documentation and examples prepared
- [ ] GitHub repository structured and documented

This comprehensive implementation plan provides a complete roadmap for developing Sheikh-2.5-Coder's data preparation strategy, ensuring high-quality training data that supports the model's specialization in XML/MDX/JavaScript while maintaining the on-device deployment requirements.
