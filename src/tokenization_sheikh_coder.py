"""
Sheikh-2.5-Coder Specialized Tokenizer
Optimized for XML, MDX, JavaScript, and web development code
"""

import re
import os
from typing import List, Dict, Optional, Union, Any, Tuple
from tokenizers import Tokenizer, AddedToken
from tokenizers.models import BPE, WordLevel
from tokenizers.trainers import BpeTrainer, WordLevelTrainer
from tokenizers.pre_tokenizers import ByteLevel, Whitespace, WhitespaceSplit
from tokenizers.processors import TemplateProcessing, ByteLevel as ByteLevelProcessor
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
import json
import pickle

# Conditional torch import
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None


class SheikhCoderTokenizer:
    """
    Specialized tokenizer for Sheikh-2.5-Coder model.
    
    Optimized for web development and code generation with support for:
    - XML/HTML tags and attributes
    - MDX (Markdown with JSX)
    - JavaScript/TypeScript syntax
    - CSS selectors and properties
    - Special tokens for code context
    """
    
    def __init__(
        self,
        vocab_file: Optional[str] = None,
        merges_file: Optional[str] = None,
        tokenizer_file: Optional[str] = None,
        vocab_size: int = 32000,
        eos_token: str = "</s>",
        bos_token: str = "<s>",
        pad_token: str = "<pad>",
        unk_token: str = "<unk>",
        mask_token: str = "<mask>",
        # Special web development tokens
        xml_open_tag: str = "<xml_tag>",
        xml_close_tag: str = "</xml_tag>",
        xml_self_closing: str = "<xml_self>",
        js_function: str = "<js_func>",
        js_variable: str = "<js_var>",
        css_selector: str = "<css_sel>",
        css_property: str = "<css_prop>",
        mdx_component: str = "<mdx_comp>",
        code_comment: str = "<code_comment>",
        code_string: str = "<code_string>",
        # Tokenization options
        do_lower_case: bool = False,
        strip_accents: bool = False,
        add_prefix_space: bool = False,
        **kwargs
    ):
        """
        Initialize Sheikh-2.5-Coder tokenizer.
        
        Args:
            vocab_file: Path to vocabulary file
            merges_file: Path to BPE merges file
            tokenizer_file: Path to saved tokenizer file
            vocab_size: Vocabulary size
            eos_token: End of sequence token
            bos_token: Beginning of sequence token
            pad_token: Padding token
            unk_token: Unknown token
            mask_token: Mask token
            xml_open_tag: XML opening tag token
            xml_close_tag: XML closing tag token
            xml_self_closing: XML self-closing tag token
            js_function: JavaScript function token
            js_variable: JavaScript variable token
            css_selector: CSS selector token
            css_property: CSS property token
            mdx_component: MDX component token
            code_comment: Code comment token
            code_string: Code string token
            do_lower_case: Whether to lowercase text
            strip_accents: Whether to strip accents
            add_prefix_space: Whether to add prefix space
        """
        self.vocab_size = vocab_size
        self.do_lower_case = do_lower_case
        self.strip_accents = strip_accents
        self.add_prefix_space = add_prefix_space
        
        # Initialize tokenizer
        if tokenizer_file and os.path.exists(tokenizer_file):
            self.tokenizer = Tokenizer.from_file(tokenizer_file)
        elif vocab_file and merges_file:
            self.tokenizer = BPE.from_files(vocab_file, merges_file)
            if hasattr(self.tokenizer, 'pre_tokenizer'):
                self.tokenizer.pre_tokenizer = self._get_pre_tokenizer()
            if hasattr(self.tokenizer, 'decoder'):
                self.tokenizer.decoder = ByteLevelDecoder()
            if hasattr(self.tokenizer, 'post_processor'):
                self.tokenizer.post_processor = self._get_post_processor()
        else:
            # Initialize with default BPE tokenizer
            self.tokenizer = BPE(unk_token=unk_token)
            # We'll set these up properly when needed
        
        # Store special tokens
        self.special_tokens = {
            'eos_token': eos_token,
            'bos_token': bos_token,
            'pad_token': pad_token,
            'unk_token': unk_token,
            'mask_token': mask_token,
            'xml_open_tag': xml_open_tag,
            'xml_close_tag': xml_close_tag,
            'xml_self_closing': xml_self_closing,
            'js_function': js_function,
            'js_variable': js_variable,
            'css_selector': css_selector,
            'css_property': css_property,
            'mdx_component': mdx_component,
            'code_comment': code_comment,
            'code_string': code_string,
        }
        
        # Web development patterns for tokenization
        self.web_patterns = self._init_web_patterns()
        
        # Add special tokens to tokenizer if they don't exist
        self._add_special_tokens()
    
    def _get_pre_tokenizer(self):
        """Get appropriate pre-tokenizer for web development code."""
        if self.do_lower_case:
            pre_tokenizer = ByteLevel(trim_offsets=False)
        else:
            pre_tokenizer = ByteLevel(trim_offsets=False)
        return pre_tokenizer
    
    def _get_post_processor(self):
        """Get post-processor for proper token handling."""
        return TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> $B:1 </s>",
            special_tokens=[
                ("<s>", self.tokenizer.token_to_id("<s>")),
                ("</s>", self.tokenizer.token_to_id("</s>")),
            ],
        )
    
    def _init_web_patterns(self) -> Dict[str, re.Pattern]:
        """Initialize regex patterns for web development tokenization."""
        patterns = {}
        
        # XML/HTML tags
        patterns['xml_open_tag'] = re.compile(r'<[^/\s][^>]*>')
        patterns['xml_close_tag'] = re.compile(r'</[^>]+>')
        patterns['xml_self_closing'] = re.compile(r'<[^/>]+/>')
        patterns['xml_attribute'] = re.compile(r'\s+[^=\s]+(?:\s*=\s*["\'][^"\']*["\'])?')
        
        # JavaScript/TypeScript patterns
        patterns['js_function'] = re.compile(r'function\s+\w+\s*\([^)]*\)\s*{')
        patterns['js_arrow_function'] = re.compile(r'\([^)]*\)\s*=>\s*{?')
        patterns['js_variable'] = re.compile(r'(?:var|let|const|const)\s+\w+')
        patterns['js_string'] = re.compile(r'["\'`][^"\']*["\'`]')
        patterns['js_comment'] = re.compile(r'(?://[^\n]*)|(/\*.*?\*/)')
        
        # CSS patterns
        patterns['css_selector'] = re.compile(r'[.#]?[\w-]+\s*{')
        patterns['css_property'] = re.compile(r'\w+\s*:\s*[^;]+;')
        patterns['css_value'] = re.compile(r':\s*[^;]+;')
        
        # MDX patterns
        patterns['mdx_component'] = re.compile(r'<[A-Z][a-zA-Z0-9_]*[^/>]*>')
        patterns['mdx_import'] = re.compile(r'import\s+.*?\s+from\s+["\'][^"\']*["\'];?')
        
        # Code structure patterns
        patterns['indent_block'] = re.compile(r'(?:^\s+.*\n?)+', re.MULTILINE)
        patterns['parentheses'] = re.compile(r'\([^)]*\)')
        patterns['brackets'] = re.compile(r'\[[^\]]*\]')
        patterns['braces'] = re.compile(r'{[^}]*}')
        
        return patterns
    
    def _add_special_tokens(self):
        """Add special tokens to the tokenizer."""
        special_tokens = []
        for token_name, token_value in self.special_tokens.items():
            special_tokens.append(AddedToken(token_value, lstrip=False, rstrip=False))
        
        # Only add if the tokenizer is not already trained
        if hasattr(self.tokenizer, 'vocab') and len(self.tokenizer.vocab) > 0:
            # Add special tokens to existing vocabulary
            current_vocab_size = len(self.tokenizer.vocab)
            for token in special_tokens:
                if token.content not in self.tokenizer.vocab:
                    self.tokenizer.add_special_tokens([token.content])
    
    def encode_special_tokens(self, text: str) -> str:
        """
        Encode special tokens in text for better tokenization.
        
        Args:
            text: Input text to encode
            
        Returns:
            Text with special tokens encoded
        """
        # Handle XML tags
        text = re.sub(self.web_patterns['xml_open_tag'], 
                     f' {self.special_tokens["xml_open_tag"]} ', text)
        text = re.sub(self.web_patterns['xml_close_tag'], 
                     f' {self.special_tokens["xml_close_tag"]} ', text)
        text = re.sub(self.web_patterns['xml_self_closing'], 
                     f' {self.special_tokens["xml_self_closing"]} ', text)
        
        # Handle JavaScript patterns
        text = re.sub(self.web_patterns['js_function'], 
                     f' {self.special_tokens["js_function"]} ', text)
        text = re.sub(self.web_patterns['js_variable'], 
                     f' {self.special_tokens["js_variable"]} ', text)
        text = re.sub(self.web_patterns['js_string'], 
                     f' {self.special_tokens["code_string"]} ', text)
        text = re.sub(self.web_patterns['js_comment'], 
                     f' {self.special_tokens["code_comment"]} ', text)
        
        # Handle CSS patterns
        text = re.sub(self.web_patterns['css_selector'], 
                     f' {self.special_tokens["css_selector"]} ', text)
        text = re.sub(self.web_patterns['css_property'], 
                     f' {self.special_tokens["css_property"]} ', text)
        
        # Handle MDX patterns
        text = re.sub(self.web_patterns['mdx_component'], 
                     f' {self.special_tokens["mdx_component"]} ', text)
        
        return text
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text for tokenization.
        
        Args:
            text: Input text to normalize
            
        Returns:
            Normalized text
        """
        # Handle special cases for web development code
        text = self.encode_special_tokens(text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        if self.do_lower_case and not self._is_code(text):
            text = text.lower()
        
        return text.strip()
    
    def _is_code(self, text: str) -> bool:
        """Determine if text appears to be code."""
        code_indicators = [
            r'function\s*\(',
            r'class\s+\w+',
            r'\{[^}]*\}',
            r'</?\w+[^>]*>',
            r'const\s+\w+',
            r'let\s+\w+',
            r'var\s+\w+',
            r'import\s+',
            r'export\s+',
        ]
        
        for pattern in code_indicators:
            if re.search(pattern, text):
                return True
        
        return False
    
    def tokenize(self, text: str, **kwargs) -> List[str]:
        """
        Tokenize input text with web development optimization.
        
        Args:
            text: Text to tokenize
            **kwargs: Additional tokenization parameters
            
        Returns:
            List of tokens
        """
        # Normalize text
        normalized_text = self.normalize_text(text)
        
        # Tokenize using the underlying tokenizer
        tokens = self.tokenizer.encode(normalized_text, **kwargs).tokens
        
        return tokens
    
    def encode(self, text: str, **kwargs) -> List[int]:
        """
        Encode input text to token IDs.
        
        Args:
            text: Text to encode
            **kwargs: Additional encoding parameters
            
        Returns:
            List of token IDs
        """
        normalized_text = self.normalize_text(text)
        return self.tokenizer.encode(normalized_text, **kwargs).ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True, **kwargs) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs to decode
            skip_special_tokens: Whether to skip special tokens
            **kwargs: Additional decoding parameters
            
        Returns:
            Decoded text
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens, **kwargs)
    
    def batch_encode_plus(self, texts: List[str], **kwargs) -> Dict[str, Union[List[int], List[str]]]:
        """
        Encode a batch of texts.
        
        Args:
            texts: List of texts to encode
            **kwargs: Additional encoding parameters
            
        Returns:
            Dictionary with encoded results
        """
        normalized_texts = [self.normalize_text(text) for text in texts]
        
        # Use tokenizer's batch methods for efficiency
        encodings = self.tokenizer.encode_batch(normalized_texts, **kwargs)
        
        return {
            'input_ids': [encoding.ids for encoding in encodings],
            'attention_mask': [encoding.attention_mask for encoding in encodings],
            'tokens': [encoding.tokens for encoding in encodings],
        }
    
    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary mapping."""
        if hasattr(self.tokenizer, 'get_vocab'):
            return self.tokenizer.get_vocab()
        else:
            return {}
    
    def get_special_tokens_map(self) -> Dict[str, str]:
        """Get mapping of special token types to their values."""
        return self.special_tokens.copy()
    
    def save_vocabulary(self, save_directory: str) -> Tuple[str, str]:
        """
        Save tokenizer vocabulary and merges to a directory.
        
        Args:
            save_directory: Directory to save tokenizer files
            
        Returns:
            Tuple of (vocab_file, merges_file) paths
        """
        os.makedirs(save_directory, exist_ok=True)
        
        if self.tokenizer.model.type == 'BPE':
            vocab_file = os.path.join(save_directory, 'vocab.json')
            merges_file = os.path.join(save_directory, 'merges.txt')
            
            self.tokenizer.save(vocab_file)
            self.tokenizer.save_pretrained(save_directory)
            
            return vocab_file, merges_file
        else:
            vocab_file = os.path.join(save_directory, 'tokenizer.json')
            self.tokenizer.save(vocab_file)
            return vocab_file, None
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs) -> 'SheikhCoderTokenizer':
        """
        Load tokenizer from pretrained model path.
        
        Args:
            pretrained_model_name_or_path: Path to pretrained tokenizer
            **kwargs: Additional tokenizer parameters
            
        Returns:
            Loaded tokenizer instance
        """
        tokenizer_file = os.path.join(pretrained_model_name_or_path, 'tokenizer.json')
        vocab_file = os.path.join(pretrained_model_name_or_path, 'vocab.json')
        merges_file = os.path.join(pretrained_model_name_or_path, 'merges.txt')
        
        if os.path.exists(tokenizer_file):
            return cls(tokenizer_file=tokenizer_file, **kwargs)
        elif os.path.exists(vocab_file) and os.path.exists(merges_file):
            return cls(vocab_file=vocab_file, merges_file=merges_file, **kwargs)
        else:
            raise ValueError(f"No tokenizer found at {pretrained_model_name_or_path}")
    
    def train_from_files(
        self, 
        file_paths: List[str], 
        vocab_size: int = 32000,
        min_frequency: int = 2,
        special_tokens: Optional[List[str]] = None
    ):
        """
        Train tokenizer from files.
        
        Args:
            file_paths: List of file paths to train on
            vocab_size: Target vocabulary size
            min_frequency: Minimum frequency for tokens
            special_tokens: List of special tokens to add
        """
        if special_tokens is None:
            special_tokens = list(self.special_tokens.values())
        
        # Create BPE trainer
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            show_progress=True,
        )
        
        # Train on files
        self.tokenizer.train(files=file_paths, trainer=trainer)
        
        # Set post-processor
        self.tokenizer.post_processor = self._get_post_processor()
    
    def get_code_tokens(self) -> List[str]:
        """Get list of special tokens related to code."""
        code_tokens = [
            self.special_tokens['code_comment'],
            self.special_tokens['code_string'],
            self.special_tokens['js_function'],
            self.special_tokens['js_variable'],
            self.special_tokens['xml_open_tag'],
            self.special_tokens['xml_close_tag'],
            self.special_tokens['mdx_component'],
        ]
        return code_tokens
    
    def get_web_tokens(self) -> List[str]:
        """Get list of web development specific tokens."""
        web_tokens = [
            self.special_tokens['xml_open_tag'],
            self.special_tokens['xml_close_tag'],
            self.special_tokens['xml_self_closing'],
            self.special_tokens['css_selector'],
            self.special_tokens['css_property'],
            self.special_tokens['mdx_component'],
        ]
        return web_tokens
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get tokenizer statistics."""
        vocab = self.get_vocab()
        special_tokens = self.get_special_tokens_map()
        
        return {
            'vocab_size': len(vocab),
            'special_tokens_count': len(special_tokens),
            'total_tokens': len(vocab),
            'vocabulary_distribution': self._get_vocab_distribution(vocab),
            'special_tokens': special_tokens,
            'configuration': {
                'do_lower_case': self.do_lower_case,
                'strip_accents': self.strip_accents,
                'add_prefix_space': self.add_prefix_space,
            }
        }
    
    def _get_vocab_distribution(self, vocab: Dict[str, int]) -> Dict[str, int]:
        """Get distribution of token types in vocabulary."""
        distribution = {
            'special_tokens': 0,
            'common_words': 0,
            'code_tokens': 0,
            'numbers': 0,
            'single_chars': 0,
        }
        
        for token in vocab.keys():
            if token in self.special_tokens.values():
                distribution['special_tokens'] += 1
            elif re.match(r'^\d+$', token):
                distribution['numbers'] += 1
            elif len(token) == 1:
                distribution['single_chars'] += 1
            elif any(word in token.lower() for word in ['function', 'class', 'import', 'export']):
                distribution['code_tokens'] += 1
            else:
                distribution['common_words'] += 1
        
        return distribution


# Factory function for creating default tokenizer
def create_default_tokenizer() -> SheikhCoderTokenizer:
    """Create default Sheikh-2.5-Coder tokenizer."""
    return SheikhCoderTokenizer()


# Helper function for batch processing
def batch_process_texts(
    tokenizer: SheikhCoderTokenizer,
    texts: List[str],
    max_length: Optional[int] = None,
    padding: str = 'max_length',
    truncation: bool = True,
    return_tensors: str = 'pt'
) -> Dict[str, Union[List[int], 'torch.Tensor']]:
    """
    Batch process texts for model input.
    
    Args:
        tokenizer: Tokenizer instance
        texts: List of texts to process
        max_length: Maximum sequence length
        padding: Padding strategy
        truncation: Whether to truncate
        return_tensors: Type of tensors to return
        
    Returns:
        Dictionary with processed batch
    """
    try:
        import torch
        has_torch = True
    except ImportError:
        has_torch = False
        if return_tensors == 'pt':
            return_tensors = 'np'
    
    # Encode batch
    encoded = tokenizer.batch_encode_plus(
        texts,
        padding=padding,
        truncation=truncation,
        max_length=max_length,
        return_tensors=return_tensors
    )
    
    # Convert to appropriate tensor format
    if has_torch and return_tensors == 'pt':
        encoded['input_ids'] = torch.tensor(encoded['input_ids'])
        encoded['attention_mask'] = torch.tensor(encoded['attention_mask'])
    
    return encoded