#!/usr/bin/env python3
"""
Enhanced Speculative Decoding Draft Model Finder v6.1.0

Finds optimal draft models for speculative decoding with comprehensive analysis.

Author:  Michael Stal, 2026
License: MIT
"""

import argparse
import json
import logging
import os
import sys
import signal
import re
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Set
from pathlib import Path

# ============================================================================
# DEPENDENCY IMPORTS WITH ERROR HANDLING
# ============================================================================

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import AutoConfig, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from huggingface_hub import list_models, model_info, HfApi
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

DEPENDENCIES_AVAILABLE = TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE and HF_HUB_AVAILABLE

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

VERSION = "6.1.0"
_INTERRUPTED = False

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# SIGNAL HANDLING
# ============================================================================

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    global _INTERRUPTED
    _INTERRUPTED = True
    logger.warning("\n⚠️  Interrupt received. Finishing current operation...")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ============================================================================
# DEPENDENCY CHECK
# ============================================================================

def check_dependencies():
    """Check if all required dependencies are available."""
    missing = []
    
    if not TORCH_AVAILABLE:
        missing.append("torch")
    if not TRANSFORMERS_AVAILABLE:
        missing.append("transformers")
    if not HF_HUB_AVAILABLE:
        missing.append("huggingface-hub")
    
    if missing:
        logger.error("❌ Missing required dependencies!")
        logger.error(f"   Missing: {', '.join(missing)}")
        logger.error("\n📦 Please install required packages:")
        logger.error(f"   pip install {' '.join(missing)}")
        sys.exit(1)


# ============================================================================
# CONFIGURATION CLASSES
# ============================================================================

@dataclass
class SearchConfig:
    """Search configuration."""
    include_quantized: bool = True
    min_size_ratio: float = 0.05
    max_size_ratio: float = 0.5
    prefer_same_family: bool = True
    prefer_official_models: bool = True


@dataclass
class CompatibilityConfig:
    """Compatibility checking configuration."""
    strict_vocab_match: bool = True
    strict_tokenization_match: bool = True
    min_tokenization_match_rate: float = 0.95
    min_compatibility_score: float = 70.0
    require_special_tokens: bool = False
    tokenizer_load_timeout: int = 30


@dataclass
class PerformanceConfig:
    """Performance estimation configuration."""
    base_acceptance_rate: float = 0.80
    same_family_bonus: float = 0.10
    tokenizer_match_bonus: float = 0.05
    optimal_size_ratio_min: float = 0.1
    optimal_size_ratio_max: float = 0.3


@dataclass
class OutputConfig:
    """Output configuration."""
    show_warnings: bool = True
    show_compatibility_details: bool = False
    show_memory_info: bool = True
    format: str = "table"  # table, json, yaml


@dataclass
class DraftFinderConfig:
    """Complete DraftFinder configuration."""
    target_model: str
    token: Optional[str] = None
    top_n: int = 5
    task: str = "general"
    max_candidates: int = 100
    verbose: bool = False
    export: Optional[str] = None
    
    search: SearchConfig = field(default_factory=SearchConfig)
    compatibility: CompatibilityConfig = field(default_factory=CompatibilityConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    @classmethod
    def from_file(cls, config_path: str) -> 'DraftFinderConfig':
        """Load configuration from YAML or JSON file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Load file
        with open(config_path, 'r') as f:
            if config_path.suffix in ['.yaml', '.yml']:
                if not YAML_AVAILABLE:
                    raise ImportError("PyYAML not installed. Install with: pip install pyyaml")
                data = yaml.safe_load(f)
            elif config_path.suffix == '.json':
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")
        
        # Validate required fields
        if 'target_model' not in data:
            raise ValueError("Config file must contain 'target_model' field")
        
        # Parse nested configs with defaults
        search_config = SearchConfig(**data.get('search', {}))
        compat_config = CompatibilityConfig(**data.get('compatibility', {}))
        perf_config = PerformanceConfig(**data.get('performance', {}))
        output_config = OutputConfig(**data.get('output', {}))
        
        # Create main config
        return cls(
            target_model=data['target_model'],
            token=data.get('token'),
            top_n=data.get('top_n', 5),
            task=data.get('task', 'general'),
            max_candidates=data.get('max_candidates', 100),
            verbose=data.get('verbose', False),
            export=data.get('export'),
            search=search_config,
            compatibility=compat_config,
            performance=perf_config,
            output=output_config
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'target_model': self.target_model,
            'token': self.token,
            'top_n': self.top_n,
            'task': self.task,
            'max_candidates': self.max_candidates,
            'verbose': self.verbose,
            'export': self.export,
            'search': asdict(self.search),
            'compatibility': asdict(self.compatibility),
            'performance': asdict(self.performance),
            'output': asdict(self.output)
        }
    
    def to_yaml(self, output_path: str):
        """Export to YAML file."""
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML not installed. Install with: pip install pyyaml")
        with open(output_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    def to_json(self, output_path: str):
        """Export to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def generate_template(cls, format_type: str = 'yaml') -> str:
        """Generate a template configuration."""
        template = cls(target_model="gpt2-xl")
        if format_type == 'yaml':
            if not YAML_AVAILABLE:
                raise ImportError("PyYAML not installed. Install with: pip install pyyaml")
            return yaml.dump(template.to_dict(), default_flow_style=False, sort_keys=False)
        elif format_type == 'json':
            return json.dumps(template.to_dict(), indent=2)
        else:
            raise ValueError(f"Unsupported format: {format_type}")


def merge_configs(config: DraftFinderConfig, cli_args: Dict[str, Any]) -> DraftFinderConfig:
    """Merge CLI arguments with config file (CLI takes precedence)."""
    # Override with CLI args if provided
    if cli_args.get('target_model'):
        config.target_model = cli_args['target_model']
    if cli_args.get('token'):
        config.token = cli_args['token']
    if cli_args.get('top_n') is not None:
        config.top_n = cli_args['top_n']
    if cli_args.get('task'):
        config.task = cli_args['task']
    if cli_args.get('max_candidates') is not None:
        config.max_candidates = cli_args['max_candidates']
    if cli_args.get('verbose') is not None:
        config.verbose = cli_args['verbose']
    if cli_args.get('export'):
        config.export = cli_args['export']
    
    return config


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class Task(Enum):
    """Task types for model evaluation."""
    GENERAL = "general"
    CODE = "code"
    CHAT = "chat"
    SUMMARIZATION = "summarization"


class QuantizationType(Enum):
    """Quantization types."""
    NONE = "none"
    INT8 = "int8"
    INT4 = "int4"
    GPTQ = "gptq"
    AWQ = "awq"
    GGUF = "gguf"
    BNB_4BIT = "bnb-4bit"
    BNB_8BIT = "bnb-8bit"


class QualityTier(Enum):
    """Quality tier for draft model recommendations."""
    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    ACCEPTABLE = "ACCEPTABLE"
    MARGINAL = "MARGINAL"
    POOR = "POOR"


@dataclass
class CompatibilityResult:
    """Results of tokenizer compatibility check."""
    is_compatible: bool
    vocab_size_match: bool
    tokenizer_match: bool
    tokenization_match: bool
    decode_match: bool
    special_tokens_match: bool
    quantization_compatible: bool
    architecture_match: bool
    model_accessible: bool
    tokenization_match_rate: float = 0.0
    num_test_strings: int = 0
    num_passed_tests: int = 0
    num_failed_tests: int = 0
    failed_test_examples: List[Dict[str, Any]] = field(default_factory=list)
    incompatibility_reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    compatibility_score: float = 100.0


@dataclass
class ModelInfo:
    """Comprehensive model information."""
    model_id: str
    architecture: str
    num_parameters: int
    num_layers: int
    hidden_size: int
    intermediate_size: Optional[int]
    num_attention_heads: int
    vocab_size: int
    quantization: QuantizationType
    bits_per_param: float
    memory_footprint_mb: float
    is_peft: bool = False
    model_family: Optional[str] = None
    is_official: bool = False
    is_quantized_variant: bool = False
    base_model_id: Optional[str] = None
    
    def size_ratio_to(self, other: 'ModelInfo') -> float:
        """Calculate size ratio compared to another model."""
        if other.num_parameters == 0:
            return float('inf')
        return self.num_parameters / other.num_parameters
    
    def effective_memory_ratio_to(self, other: 'ModelInfo') -> float:
        """Calculate effective memory ratio (accounting for quantization)."""
        if other.memory_footprint_mb == 0:
            return float('inf')
        return self.memory_footprint_mb / other.memory_footprint_mb
    
    def __str__(self) -> str:
        """String representation."""
        quant_str = f" ({self.quantization.value.upper()})" if self.quantization != QuantizationType.NONE else ""
        return (f"{self.model_id}\n"
                f"  Architecture: {self.architecture}\n"
                f"  Parameters: {self.num_parameters:,}{quant_str}\n"
                f"  Layers: {self.num_layers}, Hidden: {self.hidden_size}\n"
                f"  Memory: {self.memory_footprint_mb:,.0f} MB")


@dataclass
class DraftModelCandidate:
    """Draft model candidate with evaluation metrics."""
    model_id: str
    model_info: ModelInfo
    compatibility: CompatibilityResult
    size_ratio: float
    memory_ratio: float
    estimated_speedup: float
    recommended_k: int
    acceptance_rate_estimate: float
    same_family: bool
    quality_tier: QualityTier
    quality_score: float
    warnings: List[str] = field(default_factory=list)
    
    def __str__(self) -> str:
        """String representation."""
        # Quality-based status
        if self.quality_tier == QualityTier.EXCELLENT:
            status = "🌟 EXCELLENT"
        elif self.quality_tier == QualityTier.GOOD:
            status = "✅ GOOD"
        elif self.quality_tier == QualityTier.ACCEPTABLE:
            status = "👍 ACCEPTABLE"
        elif self.quality_tier == QualityTier.MARGINAL:
            status = "⚡ MARGINAL"
        else:
            status = "⚠️  POOR"
        
        # Size ratio display - ALWAYS show as "Xx smaller"
        inverse_ratio = 1 / self.size_ratio if self.size_ratio > 0 else 0
        if inverse_ratio >= 1:
            size_str = f"{inverse_ratio:.1f}x smaller"
        else:
            size_str = f"{self.size_ratio:.2%} of target"
        
        # Memory ratio display
        inverse_mem_ratio = 1 / self.memory_ratio if self.memory_ratio > 0 else 0
        if inverse_mem_ratio >= 1:
            mem_str = f"{inverse_mem_ratio:.1f}x less memory"
        else:
            mem_str = f"{self.memory_ratio:.2%} of target memory"
        
        # Quantization info
        quant_info = ""
        if self.model_info.quantization != QuantizationType.NONE:
            quant_info = f"\n    Quantization: {self.model_info.quantization.value.upper()}"
        
        result = (f"{self.model_id}\n"
                 f"    Family: {self.model_info.model_family or 'unknown'}\n"
                 f"    Parameters: {self.model_info.num_parameters:,}\n"
                 f"    Size: {size_str}\n"
                 f"    Memory: {self.model_info.memory_footprint_mb:,.0f} MB ({mem_str}){quant_info}\n"
                 f"    Tokenization match: {self.compatibility.tokenization_match_rate:.1%}\n"
                 f"    Estimated speedup: {self.estimated_speedup:.2f}x\n"
                 f"    Acceptance rate: {self.acceptance_rate_estimate:.1%}\n"
                 f"    Recommended K: {self.recommended_k}\n"
                 f"    Quality score: {self.quality_score:.1f}/100\n"
                 f"    {status}")
        
        # Add warnings if any
        if self.warnings:
            result += "\n    ⚠️  Warnings:"
            for warning in self.warnings:
                result += f"\n      - {warning}"
        
        return result


# ============================================================================
# ENHANCED TOKENIZER CHECKER
# ============================================================================

class EnhancedTokenizerChecker:
    """Enhanced tokenizer compatibility verification with strict tokenization testing."""
    
    # Comprehensive test strings covering various scenarios
    TEST_STRINGS = [
        # Basic sentences
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "This is a test sentence.",
        
        # Numbers and punctuation
        "Test 123 with numbers!",
        "Price: $99.99",
        "Email: test@example.com",
        
        # Special characters
        "Special chars: @#$%^&amp;*()",
        "Quotes: \"double\" and 'single'",
        "Newline test\nSecond line",
        
        # Code-like text
        "def hello_world():",
        "print('Hello, World!')",
        "x = 42; y = 3.14",
        
        # Mixed case and spacing
        "CamelCaseText",
        "UPPERCASE TEXT",
        "  leading and trailing spaces  ",
        
        # Unicode and emoji
        "Unicode: café, naïve, résumé",
        "Emoji: 😀 🎉 ❤️",
        
        # Long text
        "This is a longer sentence that contains multiple words and should test the tokenizer's ability to handle longer sequences of text properly.",
        
        # Empty and edge cases
        "",
        " ",
        "a",
    ]
    
    @staticmethod
    def verify_compatibility(tokenizer1, tokenizer2, model1_name: str, model2_name: str,
                           strict_tokenization: bool = True,
                           min_match_rate: float = 0.95) -> CompatibilityResult:
        """
        Verify comprehensive tokenizer compatibility.
        
        Args:
            tokenizer1: Target model tokenizer
            tokenizer2: Draft model tokenizer
            model1_name: Target model name
            model2_name: Draft model name
            strict_tokenization: If True, enforce tokenization match
            min_match_rate: Minimum percentage of test strings that must match
        """
        reasons = []
        warnings = []
        score = 100.0
        
        vocab1 = len(tokenizer1)
        vocab2 = len(tokenizer2)
        vocab_match = vocab1 == vocab2
        
        # ⭐ CRITICAL CHECK 1: Vocab size MUST match
        if not vocab_match:
            reasons.append(f"Incompatible vocab sizes: {vocab1:,} vs {vocab2:,}")
            return CompatibilityResult(
                is_compatible=False,
                vocab_size_match=False,
                tokenizer_match=False,
                tokenization_match=False,
                decode_match=False,
                special_tokens_match=False,
                quantization_compatible=True,
                architecture_match=False,
                model_accessible=True,
                tokenization_match_rate=0.0,
                num_test_strings=0,
                num_passed_tests=0,
                num_failed_tests=0,
                failed_test_examples=[],
                incompatibility_reasons=reasons,
                warnings=warnings,
                compatibility_score=0.0
            )
        
        # ⭐ CRITICAL CHECK 2: Special tokens must match
        special_tokens_match = True
        special_token_mismatches = []
        
        for attr, name in [
            ('eos_token_id', 'EOS'),
            ('bos_token_id', 'BOS'),
            ('pad_token_id', 'PAD'),
            ('unk_token_id', 'UNK'),
        ]:
            id1 = getattr(tokenizer1, attr, None)
            id2 = getattr(tokenizer2, attr, None)
            
            if (id1 is None) != (id2 is None):
                special_tokens_match = False
                mismatch = f"{name} token existence mismatch: {id1} vs {id2}"
                special_token_mismatches.append(mismatch)
                score -= 10
            elif id1 is not None and id1 != id2:
                special_tokens_match = False
                mismatch = f"{name} token ID mismatch: {id1} vs {id2}"
                special_token_mismatches.append(mismatch)
                score -= 10
        
        if not special_tokens_match:
            if strict_tokenization:
                reasons.append(f"Special token mismatches: {'; '.join(special_token_mismatches)}")
            else:
                warnings.extend(special_token_mismatches)
        
        # ⭐ CRITICAL CHECK 3: Tokenization must produce identical token IDs
        tokenization_matches = 0
        tokenization_total = 0
        failed_examples = []
        
        for test_str in EnhancedTokenizerChecker.TEST_STRINGS:
            try:
                # Encode without special tokens for fair comparison
                ids1 = tokenizer1.encode(test_str, add_special_tokens=False)
                ids2 = tokenizer2.encode(test_str, add_special_tokens=False)
                
                tokenization_total += 1
                
                if ids1 == ids2:
                    tokenization_matches += 1
                else:
                    # Record the mismatch with detailed information
                    failed_examples.append({
                        'text': test_str[:50] + ('...' if len(test_str) > 50 else ''),
                        'text_full': test_str,
                        'target_ids': ids1[:10],
                        'draft_ids': ids2[:10],
                        'target_tokens': [tokenizer1.decode([id]) for id in ids1[:5]] if len(ids1) > 0 else [],
                        'draft_tokens': [tokenizer2.decode([id]) for id in ids2[:5]] if len(ids2) > 0 else [],
                    })
                    
            except Exception as e:
                # Tokenization failed - this is a problem
                tokenization_total += 1
                failed_examples.append({
                    'text': test_str[:50] + ('...' if len(test_str) > 50 else ''),
                    'text_full': test_str,
                    'error': str(e),
                })
        
        # Calculate match rate
        tokenization_match_rate = tokenization_matches / tokenization_total if tokenization_total > 0 else 0.0
        tokenization_match = tokenization_match_rate >= min_match_rate
        
        # Report failures with detailed information
        if not tokenization_match:
            failure_msg = f"Tokenization mismatch: {tokenization_matches}/{tokenization_total} tests passed ({tokenization_match_rate:.1%})"
            
            if strict_tokenization:
                reasons.append(failure_msg)
                # Add specific examples with token-level details
                for i, example in enumerate(failed_examples[:3], 1):
                    if 'error' in example:
                        reasons.append(f"  Test {i}: '{example['text']}' - Error: {example['error']}")
                    else:
                        reasons.append(f"  Test {i}: '{example['text']}'")
                        reasons.append(f"    Target IDs: {example['target_ids']}")
                        reasons.append(f"    Draft IDs:  {example['draft_ids']}")
                        if 'target_tokens' in example and example['target_tokens']:
                            reasons.append(f"    Target tokens: {example['target_tokens']}")
                            reasons.append(f"    Draft tokens:  {example['draft_tokens']}")
            else:
                warnings.append(failure_msg)
            
            score -= 30 * (1 - tokenization_match_rate)
        
        # ⭐ CRITICAL CHECK 4: Decode must produce identical text
        decode_match = True
        decode_failures = []
        
        # Test decoding with sample token sequences
        test_token_sequences = [
            [100, 200, 300],
            [1, 2, 3, 4, 5],
            list(range(10, 20)),
        ]
        
        for tokens in test_token_sequences:
            try:
                # Only test tokens that exist in both vocabularies
                valid_tokens = [t for t in tokens if t < min(vocab1, vocab2)]
                if not valid_tokens:
                    continue
                
                text1 = tokenizer1.decode(valid_tokens, skip_special_tokens=True)
                text2 = tokenizer2.decode(valid_tokens, skip_special_tokens=True)
                
                if text1 != text2:
                    decode_match = False
                    decode_failures.append({
                        'tokens': valid_tokens[:5],
                        'target_text': text1[:50],
                        'draft_text': text2[:50],
                    })
            except Exception:
                pass
        
        if not decode_match:
            if strict_tokenization:
                reasons.append(f"Decode mismatch: {len(decode_failures)} test sequences failed")
            else:
                warnings.append(f"Decode mismatch: {len(decode_failures)} test sequences failed")
            score -= 10
        
        # Final compatibility decision
        if strict_tokenization:
            is_compatible = (
                vocab_match and
                special_tokens_match and
                tokenization_match and
                decode_match and
                score >= 70
            )
        else:
            # Lenient mode: only require vocab match
            is_compatible = vocab_match and score >= 50
        
        return CompatibilityResult(
            is_compatible=is_compatible,
            vocab_size_match=vocab_match,
            tokenizer_match=vocab_match and special_tokens_match,
            tokenization_match=tokenization_match,
            decode_match=decode_match,
            special_tokens_match=special_tokens_match,
            quantization_compatible=True,
            architecture_match=True,
            model_accessible=True,
            tokenization_match_rate=tokenization_match_rate,
            num_test_strings=tokenization_total,
            num_passed_tests=tokenization_matches,
            num_failed_tests=len(failed_examples),
            failed_test_examples=failed_examples[:10],  # Store first 10 failures
            incompatibility_reasons=reasons,
            warnings=warnings,
            compatibility_score=max(0, score)
        )


# ============================================================================
# MODEL ANALYZER
# ============================================================================

class ModelAnalyzer:
    """Analyze model configurations and extract detailed information."""
    
    # Architecture families for compatibility checking
    ARCHITECTURE_FAMILIES = {
        'gpt2': ['gpt2', 'gpt_neo', 'gpt_neox', 'gptj'],
        'llama': ['llama', 'llama2', 'llama3', 'mistral', 'mixtral'],
        'opt': ['opt'],
        'bloom': ['bloom'],
        'falcon': ['falcon'],
        'mpt': ['mpt'],
        'qwen': ['qwen', 'qwen2'],
        'phi': ['phi', 'phi-msft'],
        'gemma': ['gemma'],
    }
    
    # Model family patterns
    MODEL_FAMILY_PATTERNS = {
        'gpt2': ['gpt2', 'distilgpt2'],
        'llama': ['llama', 'llama-2', 'llama-3', 'vicuna', 'alpaca', 'tinyllama'],
        'mistral': ['mistral', 'mixtral', 'zephyr'],
        'qwen': ['qwen', 'qwen1.5', 'qwen2'],
        'phi': ['phi-1', 'phi-2', 'phi-3'],
        'gemma': ['gemma'],
        'falcon': ['falcon'],
        'mpt': ['mpt'],
        'opt': ['opt'],
        'bloom': ['bloom'],
        'pythia': ['pythia'],
        'stablelm': ['stablelm'],
    }
    
    # Model type blacklist (non-generative models + test models)
    MODEL_BLACKLIST_PATTERNS = [
        'classifier', 'sentiment', 'ner', 'token-classification',
        'question-answering', 'qa', 'image-captioning', 'vit-gpt',
        'vision', 'clip', 'blip', 'embedding', 'retrieval',
        'reward', 'critic', 'discriminator', 'vit-', 'dino',
        'siglip', 'owlvit',
        # Test/dummy models
        'tiny-random', 'dummy-', 'test-', 'testing/', '-testing/',
        'trl-internal', 'sshleifer/tiny',
    ]
    
    # Official model organizations
    OFFICIAL_ORGS = [
        'openai-community', 'facebook', 'meta-llama', 'mistralai',
        'google', 'bigscience', 'EleutherAI', 'stabilityai',
        'microsoft', 'Qwen', 'tiiuae',
    ]
    
    # Quantization patterns
    QUANTIZATION_PATTERNS = {
        QuantizationType.GPTQ: ['gptq', '-gptq'],
        QuantizationType.AWQ: ['awq', '-awq'],
        QuantizationType.GGUF: ['gguf', 'ggml'],
        QuantizationType.INT8: ['int8', '8bit', 'w8a8', 'int8wo'],
        QuantizationType.INT4: ['int4', '4bit', 'w4a16', 'int4wo'],
        QuantizationType.BNB_4BIT: ['bnb-4bit', 'bitsandbytes-4bit'],
        QuantizationType.BNB_8BIT: ['bnb-8bit', 'bitsandbytes-8bit'],
    }
    
    @classmethod
    def get_model_family(cls, model_id: str) -> Optional[str]:
        """Determine model family from model ID."""
        model_id_lower = model_id.lower()
        for family, patterns in cls.MODEL_FAMILY_PATTERNS.items():
            if any(pattern in model_id_lower for pattern in patterns):
                return family
        return None
    
    @classmethod
    def get_architecture_family(cls, architecture: str) -> Optional[str]:
        """Get architecture family for compatibility checking."""
        arch_lower = architecture.lower()
        for family, archs in cls.ARCHITECTURE_FAMILIES.items():
            if any(arch in arch_lower for arch in archs):
                return family
        return None
    
    @classmethod
    def is_official_model(cls, model_id: str) -> bool:
        """Check if model is from an official organization."""
        if '/' not in model_id:
            return True  # Models without org prefix (e.g., 'gpt2')
        org = model_id.split('/')[0]
        return org in cls.OFFICIAL_ORGS
    
    @classmethod
    def extract_base_model_id(cls, model_id: str) -> str:
        """Extract base model ID from quantized variant."""
        model_id_lower = model_id.lower()
        
        # Check for quantization patterns
        for quant_type, patterns in cls.QUANTIZATION_PATTERNS.items():
            for pattern in patterns:
                if pattern in model_id_lower:
                    # Try to extract base model name
                    if '/' in model_id:
                        org, name = model_id.split('/', 1)
                        # Remove quantization suffix
                        base_name = re.sub(r'[-_](gptq|awq|gguf|int[48]|[48]bit|bnb|bitsandbytes).*$', '', name, flags=re.IGNORECASE)
                        return f"{org}/{base_name}"
                    else:
                        return re.sub(r'[-_](gptq|awq|gguf|int[48]|[48]bit|bnb|bitsandbytes).*$', '', model_id, flags=re.IGNORECASE)
        
        return model_id
    
    @staticmethod
    def detect_quantization(config: Any, model_id: str) -> QuantizationType:
        """Detect quantization type from config and model ID."""
        model_id_lower = model_id.lower()
        
        # Check model ID for quantization indicators
        for quant_type, patterns in ModelAnalyzer.QUANTIZATION_PATTERNS.items():
            if any(pattern in model_id_lower for pattern in patterns):
                return quant_type
        
        # Check config for quantization
        if hasattr(config, 'quantization_config'):
            quant_config = config.quantization_config
            if isinstance(quant_config, dict):
                quant_method = quant_config.get('quant_method', '').lower()
                if 'gptq' in quant_method:
                    return QuantizationType.GPTQ
                if 'awq' in quant_method:
                    return QuantizationType.AWQ
                bits = quant_config.get('bits', 0)
                if bits == 8:
                    return QuantizationType.INT8
                if bits == 4:
                    return QuantizationType.INT4
        
        return QuantizationType.NONE
    
    @staticmethod
    def calculate_bits_per_param(quant_type: QuantizationType) -> float:
        """Calculate effective bits per parameter."""
        bits_map = {
            QuantizationType.NONE: 16.0,  # FP16/BF16
            QuantizationType.INT8: 8.0,
            QuantizationType.INT4: 4.0,
            QuantizationType.GPTQ: 4.0,
            QuantizationType.AWQ: 4.0,
            QuantizationType.GGUF: 4.0,
            QuantizationType.BNB_4BIT: 4.0,
            QuantizationType.BNB_8BIT: 8.0,
        }
        return bits_map.get(quant_type, 16.0)
    
    @classmethod
    def analyze_model(cls, model_id: str, token: Optional[str] = None) -> Optional[ModelInfo]:
        """Analyze model and extract comprehensive information."""
        try:
            # Load config
            config = AutoConfig.from_pretrained(model_id, token=token, trust_remote_code=True)
            
            # Extract architecture
            architecture = getattr(config, 'model_type', 'unknown')
            
            # Get model family FIRST (before filtering)
            model_family = cls.get_model_family(model_id)
            
            # Check if official
            is_official = cls.is_official_model(model_id)
            
            # Detect quantization
            quantization = cls.detect_quantization(config, model_id)
            is_quantized_variant = quantization != QuantizationType.NONE
            
            # Extract base model ID
            base_model_id = cls.extract_base_model_id(model_id) if is_quantized_variant else model_id
            
            # Apply blacklist filtering
            model_id_lower = model_id.lower()
            for pattern in cls.MODEL_BLACKLIST_PATTERNS:
                if pattern in model_id_lower:
                    logger.debug(f"  ❌ Rejected {model_id}: matches blacklist pattern '{pattern}'")
                    return None
            
            # Only apply strict filtering to UNKNOWN models
            if model_family is None:
                # Unknown model - apply additional checks
                if hasattr(config, 'num_labels') and hasattr(config, 'id2label'):
                    num_labels = getattr(config, 'num_labels', 0)
                    if 0 < num_labels < 1000:
                        logger.debug(f"  ❌ Rejected {model_id}: appears to be a classifier (num_labels={num_labels})")
                        return None
            
            # Extract basic dimensions
            num_layers = getattr(config, 'num_hidden_layers', 
                                getattr(config, 'n_layer', 
                                       getattr(config, 'num_layers', 0)))
            
            hidden_size = getattr(config, 'hidden_size', 
                                 getattr(config, 'n_embd', 
                                        getattr(config, 'd_model', 0)))
            
            intermediate_size = getattr(config, 'intermediate_size', 
                                       getattr(config, 'ffn_dim', 
                                              getattr(config, 'n_inner', None)))
            
            num_attention_heads = getattr(config, 'num_attention_heads', 
                                         getattr(config, 'n_head', 0))
            
            vocab_size = getattr(config, 'vocab_size', 0)
            
            # Sanity check
            if num_layers == 0 or hidden_size == 0 or vocab_size == 0:
                logger.debug(f"  ❌ Rejected {model_id}: invalid dimensions")
                return None
            
            bits_per_param = cls.calculate_bits_per_param(quantization)
            
            # Calculate parameters
            num_parameters = cls.estimate_parameters(
                num_layers, hidden_size, intermediate_size, 
                num_attention_heads, vocab_size
            )
            
            # Calculate memory footprint (accounting for quantization)
            memory_footprint_mb = (num_parameters * bits_per_param) / (8 * 1024 * 1024)
            
            # Detect PEFT
            is_peft = hasattr(config, 'base_model_name_or_path')
            
            return ModelInfo(
                model_id=model_id,
                architecture=architecture,
                num_parameters=num_parameters,
                num_layers=num_layers,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_attention_heads=num_attention_heads,
                vocab_size=vocab_size,
                quantization=quantization,
                bits_per_param=bits_per_param,
                memory_footprint_mb=memory_footprint_mb,
                is_peft=is_peft,
                model_family=model_family,
                is_official=is_official,
                is_quantized_variant=is_quantized_variant,
                base_model_id=base_model_id
            )
            
        except Exception as e:
            logger.debug(f"Failed to analyze {model_id}: {e}")
            return None
    
    @staticmethod
    def estimate_parameters(num_layers: int, hidden_size: int, 
                           intermediate_size: Optional[int],
                           num_attention_heads: int, vocab_size: int) -> int:
        """Estimate total model parameters."""
        if num_layers == 0 or hidden_size == 0:
            return 0
        
        # Embedding parameters
        embedding_params = vocab_size * hidden_size
        
        # Attention parameters per layer
        attention_params_per_layer = 4 * (hidden_size * hidden_size)
        
        # FFN parameters per layer
        if intermediate_size is None:
            intermediate_size = 4 * hidden_size
        ffn_params_per_layer = 2 * (hidden_size * intermediate_size)
        
        # Layer norm parameters per layer
        layernorm_params_per_layer = 2 * hidden_size
        
        # Total parameters per layer
        params_per_layer = (attention_params_per_layer + 
                           ffn_params_per_layer + 
                           layernorm_params_per_layer)
        
        # Total parameters
        total_params = embedding_params + (num_layers * params_per_layer)
        total_params += hidden_size  # Final layer norm
        total_params += vocab_size * hidden_size  # LM head
        
        return total_params


# ============================================================================
# SPEEDUP ESTIMATOR
# ============================================================================

class SpeedupEstimator:
    """Estimate speculative decoding speedup with realistic modeling."""
    
    @staticmethod
    def estimate_acceptance_rate(target_info: ModelInfo, draft_info: ModelInfo,
                                 same_family: bool, tokenizer_match: bool,
                                 tokenization_match_rate: float) -> float:
        """
        Estimate token acceptance rate with realistic penalties.
        
        Acceptance rate depends on:
        1. Size ratio (quality degradation for very small models)
        2. Model family match
        3. Tokenizer compatibility
        4. Quantization impact
        """
        base_rate = 0.75  # Conservative base rate
        
        # Bonus for same family
        if same_family:
            base_rate += 0.15
        
        # Bonus for tokenizer match
        if tokenizer_match:
            base_rate += 0.05
        
        # Penalty for tokenization mismatch
        tokenization_penalty = (1.0 - tokenization_match_rate) * 0.20
        base_rate -= tokenization_penalty
        
        # Size ratio impact (critical factor)
        size_ratio = draft_info.size_ratio_to(target_info)
        
        if size_ratio < 0.005:  # < 0.5% (e.g., 31M vs 6.7B)
            base_rate -= 0.35  # Massive penalty - quality cliff
        elif size_ratio < 0.01:  # < 1%
            base_rate -= 0.30
        elif size_ratio < 0.02:  # < 2%
            base_rate -= 0.25
        elif size_ratio < 0.05:  # < 5%
            base_rate -= 0.20
        elif size_ratio < 0.10:  # < 10%
            base_rate -= 0.10
        elif size_ratio < 0.15:  # < 15%
            base_rate -= 0.05
        elif 0.15 <= size_ratio <= 0.35:  # Optimal range
            base_rate += 0.05  # Bonus for optimal size
        elif size_ratio > 0.5:  # > 50% - draft too large
            base_rate -= 0.15
        
        # Quantization penalty (quantized models may have lower quality)
        if draft_info.quantization in [QuantizationType.INT4, QuantizationType.GPTQ, 
                                       QuantizationType.AWQ, QuantizationType.BNB_4BIT]:
            base_rate -= 0.05
        elif draft_info.quantization in [QuantizationType.INT8, QuantizationType.BNB_8BIT]:
            base_rate -= 0.02
        
        return max(0.40, min(0.95, base_rate))
    
    @staticmethod
    def estimate_speedup(target_info: ModelInfo, draft_info: ModelInfo,
                        acceptance_rate: float, k: int = 4) -> float:
        """
        Estimate speedup factor with realistic latency modeling.
        
        Speedup formula:
        speedup = (1 + α * k) / (1 + k * β)
        
        where:
        - α = acceptance rate
        - β = draft latency ratio
        - k = number of speculative tokens
        """
        # Memory-based latency ratio (more accurate than parameter count)
        memory_ratio = draft_info.effective_memory_ratio_to(target_info)
        
        # Latency scales sublinearly with memory
        # Smaller models benefit from better cache utilization
        latency_ratio = memory_ratio ** 0.65
        
        # Expected accepted tokens per iteration
        expected_accepted = acceptance_rate * k
        
        # Speedup calculation
        speedup = (1 + expected_accepted) / (1 + k * latency_ratio)
        
        # Cap unrealistic speedups
        max_theoretical_speedup = 1 + k * 0.8  # 80% efficiency ceiling
        speedup = min(speedup, max_theoretical_speedup)
        
        return max(1.0, speedup)
    
    @staticmethod
    def recommend_k(target_info: ModelInfo, draft_info: ModelInfo,
                   acceptance_rate: float) -> int:
        """Recommend optimal k value based on acceptance rate and size ratio."""
        size_ratio = draft_info.size_ratio_to(target_info)
        
        # Base k on size ratio
        if size_ratio < 0.05:
            base_k = 7  # Very small draft - try more tokens
        elif size_ratio < 0.1:
            base_k = 6
        elif size_ratio < 0.2:
            base_k = 5
        elif size_ratio < 0.3:
            base_k = 4
        else:
            base_k = 3
        
        # Adjust for acceptance rate
        if acceptance_rate > 0.85:
            base_k += 1
        elif acceptance_rate < 0.60:
            base_k -= 1
        
        return max(2, min(8, base_k))
    
    @staticmethod
    def calculate_quality_score(target_info: ModelInfo, draft_info: ModelInfo,
                               acceptance_rate: float, speedup: float,
                               same_family: bool, tokenization_match_rate: float) -> float:
        """
        Calculate overall quality score (0-100) for ranking.
        
        Factors:
        - Speedup (40%)
        - Acceptance rate (30%)
        - Size ratio optimality (15%)
        - Family match (10%)
        - Tokenization match (5%)
        """
        score = 0.0
        
        # Speedup component (40 points max)
        # Normalize speedup: 1.0x = 0 points, 3.0x = 40 points
        speedup_score = min(40, (speedup - 1.0) * 20)
        score += speedup_score
        
        # Acceptance rate component (30 points max)
        acceptance_score = acceptance_rate * 30
        score += acceptance_score
        
        # Size ratio optimality (15 points max)
        size_ratio = draft_info.size_ratio_to(target_info)
        if 0.15 <= size_ratio <= 0.35:  # Optimal range
            size_score = 15
        elif 0.10 <= size_ratio < 0.15 or 0.35 < size_ratio <= 0.45:
            size_score = 12
        elif 0.05 <= size_ratio < 0.10 or 0.45 < size_ratio <= 0.55:
            size_score = 8
        else:
            size_score = max(0, 15 - abs(size_ratio - 0.25) * 40)
        score += size_score
        
        # Family match (10 points max)
        if same_family:
            score += 10
        
        # Tokenization match (5 points max)
        score += tokenization_match_rate * 5
        
        return min(100, max(0, score))
    
    @staticmethod
    def determine_quality_tier(quality_score: float, acceptance_rate: float, 
                              speedup: float) -> QualityTier:
        """Determine quality tier based on metrics."""
        if quality_score >= 80 and acceptance_rate >= 0.80 and speedup >= 2.0:
            return QualityTier.EXCELLENT
        elif quality_score >= 65 and acceptance_rate >= 0.70 and speedup >= 1.5:
            return QualityTier.GOOD
        elif quality_score >= 50 and acceptance_rate >= 0.60 and speedup >= 1.3:
            return QualityTier.ACCEPTABLE
        elif quality_score >= 35 and speedup >= 1.2:
            return QualityTier.MARGINAL
        else:
            return QualityTier.POOR


# ============================================================================
# DRAFT MODEL FINDER
# ============================================================================

class DraftModelFinder:
    """Find and evaluate draft model candidates with comprehensive analysis."""
    
    # Minimum parameters for practical draft models (10M)
    MIN_DRAFT_PARAMETERS = 10_000_000
    
    # Hardcoded known good draft models by family
    KNOWN_DRAFT_MODELS = {
        'gpt2': [
            'gpt2',
            'gpt2-medium',
            'gpt2-large',
            'distilgpt2',
            'openai-community/gpt2',
            'openai-community/gpt2-medium',
            'openai-community/gpt2-large',
        ],
        'llama': [
            'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
            'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T',
        ],
        'mistral': [
            'mistralai/Mistral-7B-v0.1',
            'mistralai/Mistral-7B-Instruct-v0.1',
        ],
        'qwen': [
            'Qwen/Qwen2.5-0.5B',
            'Qwen/Qwen2.5-1.5B',
            'Qwen/Qwen2.5-3B',
        ],
        'opt': [
            'facebook/opt-125m',
            'facebook/opt-350m',
            'facebook/opt-1.3b',
            'facebook/opt-2.7b',
            'facebook/opt-6.7b',
            'facebook/opt-13b',
        ],
        'bloom': [
            'bigscience/bloom-560m',
            'bigscience/bloom-1b1',
            'bigscience/bloom-1b7',
            'bigscience/bloom-3b',
            'bigscience/bloom-7b1',
        ],
        'pythia': [
            'EleutherAI/pythia-70m',
            'EleutherAI/pythia-160m',
            'EleutherAI/pythia-410m',
            'EleutherAI/pythia-1b',
            'EleutherAI/pythia-1.4b',
            'EleutherAI/pythia-2.8b',
            'EleutherAI/pythia-6.9b',
            'EleutherAI/pythia-12b',
        ],
    }
    
    def __init__(self, target_model_id: str, token: Optional[str] = None,
                 task: Task = Task.GENERAL, verbose: bool = False,
                 max_candidates: int = 100,
                 strict_tokenization: bool = True,
                 prefer_official: bool = True,
                 show_test_details: bool = False):
        """Initialize finder."""
        self.target_model_id = target_model_id
        self.token = token
        self.task = task
        self.verbose = verbose
        self.max_candidates = max_candidates
        self.strict_tokenization = strict_tokenization
        self.prefer_official = prefer_official
        self.show_test_details = show_test_details
        
        self.target_info: Optional[ModelInfo] = None
        self.target_tokenizer = None
        self.candidates: List[DraftModelCandidate] = []
    
    def analyze_target_model(self) -> bool:
        """Analyze target model."""
        logger.info(f"📊 Analyzing target model: {self.target_model_id}")
        
        # Analyze model
        self.target_info = ModelAnalyzer.analyze_model(self.target_model_id, self.token)
        if not self.target_info:
            logger.error(f"❌ Failed to analyze target model: {self.target_model_id}")
            logger.error("   This could be due to:")
            logger.error("   1. Model not found on HuggingFace Hub")
            logger.error("   2. Network connectivity issues")
            logger.error("   3. Invalid model ID")
            logger.error("   4. Missing authentication token (for gated models)")
            return False
        
        # Load tokenizer
        try:
            self.target_tokenizer = AutoTokenizer.from_pretrained(
                self.target_model_id, 
                token=self.token,
                trust_remote_code=True
            )
        except Exception as e:
            logger.error(f"❌ Failed to load tokenizer: {e}")
            return False
        
        logger.info(f"✅ Target model analyzed successfully")
        logger.info(f"   Parameters: {self.target_info.num_parameters:,}")
        logger.info(f"   Memory: {self.target_info.memory_footprint_mb:,.0f} MB")
        logger.info(f"   Family: {self.target_info.model_family or 'unknown'}")
        
        if self.verbose:
            logger.info(f"\n{self.target_info}")
        
        return True
    
    def search_candidates(self) -> List[str]:
        """Search for potential draft model candidates."""
        logger.info(f"🔍 Searching for draft model candidates...")
        
        candidates = []
        seen = set()
        
        # First, add known good models for this family
        if self.target_info.model_family in self.KNOWN_DRAFT_MODELS:
            known_models = self.KNOWN_DRAFT_MODELS[self.target_info.model_family]
            logger.info(f"  Adding {len(known_models)} known good models for {self.target_info.model_family}")
            for model_id in known_models:
                if model_id != self.target_model_id and model_id not in seen:
                    candidates.append(model_id)
                    seen.add(model_id)
        
        # Add hardcoded model families based on target model ID
        target_lower = self.target_model_id.lower()
        
        # OPT family
        if 'opt' in target_lower:
            opt_models = [
                'facebook/opt-125m',
                'facebook/opt-350m',
                'facebook/opt-1.3b',
                'facebook/opt-2.7b',
                'facebook/opt-6.7b',
                'facebook/opt-13b',
            ]
            logger.info(f"  Adding {len(opt_models)} known OPT models")
            for model_id in opt_models:
                if model_id != self.target_model_id and model_id not in seen:
                    candidates.append(model_id)
                    seen.add(model_id)
        
        # BLOOM family
        if 'bloom' in target_lower:
            bloom_models = [
                'bigscience/bloom-560m',
                'bigscience/bloom-1b1',
                'bigscience/bloom-1b7',
                'bigscience/bloom-3b',
                'bigscience/bloom-7b1',
            ]
            logger.info(f"  Adding {len(bloom_models)} known BLOOM models")
            for model_id in bloom_models:
                if model_id != self.target_model_id and model_id not in seen:
                    candidates.append(model_id)
                    seen.add(model_id)
        
        # Pythia family
        if 'pythia' in target_lower:
            pythia_models = [
                'EleutherAI/pythia-70m',
                'EleutherAI/pythia-160m',
                'EleutherAI/pythia-410m',
                'EleutherAI/pythia-1b',
                'EleutherAI/pythia-1.4b',
                'EleutherAI/pythia-2.8b',
                'EleutherAI/pythia-6.9b',
                'EleutherAI/pythia-12b',
            ]
            logger.info(f"  Adding {len(pythia_models)} known Pythia models")
            for model_id in pythia_models:
                if model_id != self.target_model_id and model_id not in seen:
                    candidates.append(model_id)
                    seen.add(model_id)
        
        # GPT-Neo family
        if 'gpt-neo' in target_lower or 'gpt-neox' in target_lower:
            gpt_neo_models = [
                'EleutherAI/gpt-neo-125m',
                'EleutherAI/gpt-neo-1.3B',
                'EleutherAI/gpt-neo-2.7B',
            ]
            logger.info(f"  Adding {len(gpt_neo_models)} known GPT-Neo models")
            for model_id in gpt_neo_models:
                if model_id != self.target_model_id and model_id not in seen:
                    candidates.append(model_id)
                    seen.add(model_id)
        
        # Search strategies
        searches = []
        
        # 1. Same model family
        if self.target_info.model_family:
            searches.append(self.target_info.model_family)
        
        # 2. Same architecture
        arch_family = ModelAnalyzer.get_architecture_family(self.target_info.architecture)
        if arch_family and arch_family not in searches:
            searches.append(arch_family)
        
        # 3. General small models
        searches.extend(['small', 'tiny', 'mini'])
        
        # Execute searches
        for search_term in searches:
            if _INTERRUPTED:
                break
            
            try:
                if self.verbose:
                    logger.info(f"  Searching HuggingFace Hub: {search_term}")
                
                models = list_models(
                    search=search_term,
                    filter='text-generation',
                    sort='downloads',
                    limit=50,
                    token=self.token
                )
                
                count = 0
                for model in models:
                    if _INTERRUPTED:
                        break
                    
                    model_id = model.id if hasattr(model, 'id') else str(model)
                    if model_id not in seen and model_id != self.target_model_id:
                        candidates.append(model_id)
                        seen.add(model_id)
                        count += 1
                        
                        if len(candidates) >= self.max_candidates:
                            break
                
                if count > 0 and self.verbose:
                    logger.info(f"    Found {count} new candidates from Hub")
                
            except Exception as e:
                logger.debug(f"Search failed for '{search_term}': {e}")
                continue
        
        logger.info(f"✅ Found {len(candidates)} candidate models")
        return candidates[:self.max_candidates]
    
    def evaluate_candidate(self, draft_model_id: str) -> Optional[DraftModelCandidate]:
        """Evaluate a single draft model candidate."""
        try:
            # Analyze draft model
            draft_info = ModelAnalyzer.analyze_model(draft_model_id, self.token)
            if not draft_info:
                if self.verbose:
                    logger.debug(f"  ❌ {draft_model_id}: Failed to analyze")
                return None
            
            # Filter out test models (< 10M parameters)
            if draft_info.num_parameters < self.MIN_DRAFT_PARAMETERS:
                if self.verbose:
                    logger.debug(f"  ❌ {draft_model_id}: Too small ({draft_info.num_parameters:,} < {self.MIN_DRAFT_PARAMETERS:,})")
                return None
            
            # Check if draft is smaller
            if draft_info.num_parameters >= self.target_info.num_parameters:
                if self.verbose:
                    logger.debug(f"  ❌ {draft_model_id}: Too large ({draft_info.num_parameters:,} >= {self.target_info.num_parameters:,})")
                return None
            
            # Load draft tokenizer
            try:
                draft_tokenizer = AutoTokenizer.from_pretrained(
                    draft_model_id,
                    token=self.token,
                    trust_remote_code=True
                )
            except Exception as e:
                if self.verbose:
                    logger.debug(f"  ❌ {draft_model_id}: Failed to load tokenizer: {e}")
                return None
            
            # Check compatibility
            compatibility = EnhancedTokenizerChecker.verify_compatibility(
                self.target_tokenizer,
                draft_tokenizer,
                self.target_model_id,
                draft_model_id,
                strict_tokenization=self.strict_tokenization,
                min_match_rate=0.95
            )
            
            # Detailed verbose output
            if self.verbose:
                logger.info(f"\n  Checking: {draft_model_id}")
                logger.info(f"    Parameters: {draft_info.num_parameters:,}")
                logger.info(f"    Vocab sizes: Target={len(self.target_tokenizer):,}, Draft={len(draft_tokenizer):,}")
                logger.info(f"    Vocab match: {compatibility.vocab_size_match}")
                logger.info(f"    Special tokens match: {compatibility.special_tokens_match}")
                logger.info(f"    Tokenization tests: {compatibility.num_passed_tests}/{compatibility.num_test_strings} passed")
                logger.info(f"    Tokenization match rate: {compatibility.tokenization_match_rate:.1%}")
                logger.info(f"    Decode match: {compatibility.decode_match}")
                logger.info(f"    Compatibility score: {compatibility.compatibility_score:.1f}/100")
                
                # Show failed test examples if requested
                if self.show_test_details and compatibility.failed_test_examples:
                    logger.info(f"\n    📋 Failed Test Details:")
                    for i, example in enumerate(compatibility.failed_test_examples[:5], 1):
                        logger.info(f"\n       Test {i}: '{example.get('text', 'N/A')}'")
                        if 'error' in example:
                            logger.info(f"         Error: {example['error']}")
                        else:
                            logger.info(f"         Target IDs: {example.get('target_ids', [])}")
                            logger.info(f"         Draft IDs:  {example.get('draft_ids', [])}")
                            if 'target_tokens' in example and example['target_tokens']:
                                logger.info(f"         Target tokens: {example['target_tokens']}")
                                logger.info(f"         Draft tokens:  {example['draft_tokens']}")
                
                # Show incompatibility reasons
                if compatibility.tokenization_match_rate < 1.0 and compatibility.incompatibility_reasons:
                    logger.info(f"    ⚠️  Tokenization issues:")
                    for reason in compatibility.incompatibility_reasons[:3]:
                        logger.info(f"       {reason}")
                
                if compatibility.warnings:
                    logger.info(f"    ⚠️  Warnings:")
                    for warning in compatibility.warnings[:3]:
                        logger.info(f"       {warning}")
            
            if not compatibility.is_compatible:
                if self.verbose:
                    logger.debug(f"  ❌ Incompatible")
                    if not compatibility.incompatibility_reasons:
                        logger.debug(f"      Compatibility score too low: {compatibility.compatibility_score:.1f}/100")
                return None
            
            # Calculate metrics
            same_family = (draft_info.model_family == self.target_info.model_family and 
                          draft_info.model_family is not None)
            
            size_ratio = draft_info.size_ratio_to(self.target_info)
            memory_ratio = draft_info.effective_memory_ratio_to(self.target_info)
            
            acceptance_rate = SpeedupEstimator.estimate_acceptance_rate(
                self.target_info, draft_info, same_family, 
                compatibility.tokenizer_match,
                compatibility.tokenization_match_rate
            )
            
            recommended_k = SpeedupEstimator.recommend_k(
                self.target_info, draft_info, acceptance_rate
            )
            
            estimated_speedup = SpeedupEstimator.estimate_speedup(
                self.target_info, draft_info, acceptance_rate, recommended_k
            )
            
            quality_score = SpeedupEstimator.calculate_quality_score(
                self.target_info, draft_info, acceptance_rate, estimated_speedup,
                same_family, compatibility.tokenization_match_rate
            )
            
            quality_tier = SpeedupEstimator.determine_quality_tier(
                quality_score, acceptance_rate, estimated_speedup
            )
            
            # Generate warnings
            warnings = []
            if size_ratio < 0.01:
                warnings.append("Very small model - may have poor quality")
            if size_ratio > 0.5:
                warnings.append("Large draft model - limited speedup benefit")
            if draft_info.is_quantized_variant:
                warnings.append(f"Quantized variant ({draft_info.quantization.value})")
            if not draft_info.is_official:
                warnings.append("Community model (not official)")
            if compatibility.tokenization_match_rate < 1.0:
                warnings.append(f"Tokenization mismatch on {(1-compatibility.tokenization_match_rate)*100:.0f}% of tests")
            
            return DraftModelCandidate(
                model_id=draft_model_id,
                model_info=draft_info,
                compatibility=compatibility,
                size_ratio=size_ratio,
                memory_ratio=memory_ratio,
                estimated_speedup=estimated_speedup,
                recommended_k=recommended_k,
                acceptance_rate_estimate=acceptance_rate,
                same_family=same_family,
                quality_tier=quality_tier,
                quality_score=quality_score,
                warnings=warnings
            )
            
        except Exception as e:
            if self.verbose:
                logger.debug(f"  ❌ {draft_model_id}: Evaluation failed: {e}")
            return None
    
    def deduplicate_candidates(self, candidates: List[DraftModelCandidate]) -> List[DraftModelCandidate]:
        """Deduplicate quantized variants, preferring official and non-quantized models."""
        base_model_map = {}
        
        for candidate in candidates:
            base_id = candidate.model_info.base_model_id
            
            if base_id not in base_model_map:
                base_model_map[base_id] = candidate
            else:
                existing = base_model_map[base_id]
                
                # Prefer official models
                if candidate.model_info.is_official and not existing.model_info.is_official:
                    base_model_map[base_id] = candidate
                    if self.verbose:
                        logger.debug(f"  Replacing {existing.model_id} with official {candidate.model_id}")
                # Prefer non-quantized
                elif (not candidate.model_info.is_quantized_variant and 
                      existing.model_info.is_quantized_variant and
                      candidate.model_info.is_official == existing.model_info.is_official):
                    base_model_map[base_id] = candidate
                    if self.verbose:
                        logger.debug(f"  Replacing quantized {existing.model_id} with {candidate.model_id}")
                # Keep higher quality score
                elif candidate.quality_score > existing.quality_score:
                    base_model_map[base_id] = candidate
                    if self.verbose:
                        logger.debug(f"  Replacing {existing.model_id} with higher quality {candidate.model_id}")
        
        return list(base_model_map.values())
    
    def find_draft_models(self, top_n: int = 5) -> List[DraftModelCandidate]:
        """Find and rank draft models."""
        # Analyze target
        if not self.analyze_target_model():
            return []
        
        # Search candidates
        candidate_ids = self.search_candidates()
        
        if not candidate_ids:
            logger.warning("⚠️  No candidate models found to evaluate")
            return []
        
        # Evaluate candidates
        logger.info(f"🔬 Evaluating {len(candidate_ids)} candidates...")
        evaluated = 0
        compatible = 0
        
        for i, candidate_id in enumerate(candidate_ids, 1):
            if _INTERRUPTED:
                break
            
            if self.verbose and i % 10 == 0:
                logger.info(f"  Progress: {i}/{len(candidate_ids)}")
            
            candidate = self.evaluate_candidate(candidate_id)
            if candidate:
                self.candidates.append(candidate)
                compatible += 1
                if self.verbose:
                    logger.info(f"  ✅ {candidate_id}: Quality={candidate.quality_score:.0f}, Speedup={candidate.estimated_speedup:.2f}x")
            
            evaluated += 1
        
        logger.info(f"\n✅ Evaluation complete: {compatible}/{evaluated} compatible models found")
        
        # Deduplicate
        if len(self.candidates) > 0:
            logger.info(f"🔄 Deduplicating variants...")
            original_count = len(self.candidates)
            self.candidates = self.deduplicate_candidates(self.candidates)
            if len(self.candidates) < original_count:
                logger.info(f"   Removed {original_count - len(self.candidates)} duplicate variants")
        
        # Sort by quality score (primary) and speedup (secondary)
        self.candidates.sort(key=lambda x: (x.quality_score, x.estimated_speedup), reverse=True)
        
        return self.candidates[:top_n]
    
    def print_results(self, top_n: int = 5):
        """Print results in a formatted way."""
        if not self.candidates:
            logger.warning("⚠️  No compatible draft models found!")
            logger.info("\n💡 Suggestions:")
            logger.info("   1. Try a different target model")
            logger.info("   2. Check if the model family has smaller variants")
            logger.info("   3. Ensure network connectivity to HuggingFace")
            logger.info("   4. For gated models (Llama, Mistral), set HF_TOKEN")
            logger.info("   5. Try --lenient-tokenization for cross-family experiments")
            return
        
        print("\n" + "="*80)
        print("🎯 DRAFT MODEL RECOMMENDATIONS")
        print("="*80)
        
        print(f"\n📊 Target Model: {self.target_info.model_id}")
        print(f"   Parameters: {self.target_info.num_parameters:,}")
        print(f"   Memory: {self.target_info.memory_footprint_mb:,.0f} MB")
        print(f"   Vocab Size: {self.target_info.vocab_size:,}")
        print(f"   Family: {self.target_info.model_family or 'unknown'}")
        
        print(f"\n🏆 Top {min(top_n, len(self.candidates))} Draft Models:\n")
        
        for i, candidate in enumerate(self.candidates[:top_n], 1):
            print(f"{i}. {candidate}")
            print()
        
        # Summary statistics
        if len(self.candidates) > 0:
            avg_speedup = sum(c.estimated_speedup for c in self.candidates[:top_n]) / min(top_n, len(self.candidates))
            avg_acceptance = sum(c.acceptance_rate_estimate for c in self.candidates[:top_n]) / min(top_n, len(self.candidates))
            
            print("="*80)
            print("📈 Summary Statistics:")
            print(f"   Average speedup: {avg_speedup:.2f}x")
            print(f"   Average acceptance rate: {avg_acceptance:.1%}")
            print(f"   Total candidates evaluated: {len(self.candidates)}")
        
        print("="*80)
        print("📝 Usage Example:")
        if self.candidates:
            best = self.candidates[0]
            print(f"""
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load models
target_model = AutoModelForCausalLM.from_pretrained("{self.target_info.model_id}")
draft_model = AutoModelForCausalLM.from_pretrained("{best.model_id}")
tokenizer = AutoTokenizer.from_pretrained("{self.target_info.model_id}")

# Use with speculative decoding
outputs = target_model.generate(
    input_ids,
    assistant_model=draft_model,
    do_sample=False,
    max_new_tokens=100
)

# Expected speedup: ~{best.estimated_speedup:.1f}x
# Recommended K: {best.recommended_k}
""")
        print("="*80)
    
    def export_results(self, output_file: str):
        """Export results to JSON file."""
        results = {
            'version': VERSION,
            'target_model': {
                'model_id': self.target_info.model_id,
                'parameters': self.target_info.num_parameters,
                'memory_mb': self.target_info.memory_footprint_mb,
                'vocab_size': self.target_info.vocab_size,
                'family': self.target_info.model_family,
                'architecture': self.target_info.architecture
            },
            'draft_models': []
        }
        
        for candidate in self.candidates:
            results['draft_models'].append({
                'rank': len(results['draft_models']) + 1,
                'model_id': candidate.model_id,
                'parameters': candidate.model_info.num_parameters,
                'memory_mb': candidate.model_info.memory_footprint_mb,
                'vocab_size': candidate.model_info.vocab_size,
                'family': candidate.model_info.model_family,
                'quantization': candidate.model_info.quantization.value,
                'is_official': candidate.model_info.is_official,
                'size_ratio': candidate.size_ratio,
                'memory_ratio': candidate.memory_ratio,
                'tokenization_match_rate': candidate.compatibility.tokenization_match_rate,
                'tokenization_tests_passed': candidate.compatibility.num_passed_tests,
                'tokenization_tests_total': candidate.compatibility.num_test_strings,
                'estimated_speedup': candidate.estimated_speedup,
                'acceptance_rate_estimate': candidate.acceptance_rate_estimate,
                'recommended_k': candidate.recommended_k,
                'quality_score': candidate.quality_score,
                'quality_tier': candidate.quality_tier.value,
                'same_family': candidate.same_family,
                'warnings': candidate.warnings,
                'compatibility_score': candidate.compatibility.compatibility_score
            })
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"✅ Results exported to: {output_file}")


# ============================================================================
# CLI
# ============================================================================

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Find optimal draft models for speculative decoding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Find draft models for GPT-2 XL
  python draftfinder.py gpt2-xl --top-n 5

  # Find draft models for OPT-6.7B
  python draftfinder.py facebook/opt-6.7b --top-n 5

  # Verbose mode with detailed tokenization test output
  python draftfinder.py facebook/opt-6.7b --verbose --show-test-details

  # Use configuration file
  python draftfinder.py --config config.yaml

  # Override config with CLI args
  python draftfinder.py --config config.yaml --top-n 10 --verbose

  # Generate config template
  python draftfinder.py --generate-config yaml > config.yaml
  python draftfinder.py --generate-config json > config.json

  # Find draft models for Llama with HF token
  python draftfinder.py meta-llama/Llama-2-7b-hf --token hf_xxx --top-n 5

  # Export results to JSON
  python draftfinder.py mistralai/Mistral-7B-v0.1 --export results.json
  
  # Lenient tokenization matching
  python draftfinder.py gpt2-xl --lenient-tokenization
        """
    )
    
    parser.add_argument(
        'target_model',
        type=str,
        nargs='?',
        help='Target model ID (e.g., gpt2-xl, facebook/opt-6.7b)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Load settings from config file (YAML or JSON)'
    )
    
    parser.add_argument(
        '--generate-config',
        type=str,
        choices=['yaml', 'json'],
        default=None,
        help='Generate config template and exit'
    )
    
    parser.add_argument(
        '--top-n',
        type=int,
        default=None,
        help='Number of top draft models to return (default: 5)'
    )
    
    parser.add_argument(
        '--token',
        type=str,
        default=None,
        help='HuggingFace API token (or set HF_TOKEN env var)'
    )
    
    parser.add_argument(
        '--task',
        type=str,
        choices=['general', 'code', 'chat', 'summarization'],
        default=None,
        help='Task type for optimization (default: general)'
    )
    
    parser.add_argument(
        '--max-candidates',
        type=int,
        default=None,
        help='Maximum number of candidates to evaluate (default: 100)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--export',
        type=str,
        default=None,
        help='Export results to JSON file'
    )
    
    parser.add_argument(
        '--lenient-tokenization',
        action='store_true',
        help='Allow minor tokenization differences (less strict matching)'
    )
    
    parser.add_argument(
        '--show-test-details',
        action='store_true',
        help='Show detailed tokenization test results (requires --verbose)'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version=f'DraftFinder v{VERSION}'
    )
    
    args = parser.parse_args()
    
    # Handle config template generation
    if args.generate_config:
        try:
            template = DraftFinderConfig.generate_template(args.generate_config)
            print(template)
            sys.exit(0)
        except ImportError as e:
            logger.error(f"❌ {e}")
            sys.exit(1)
    
    # Check dependencies
    check_dependencies()
    
    # Load configuration
    if args.config:
        try:
            config = DraftFinderConfig.from_file(args.config)
            config = merge_configs(config, vars(args))
        except Exception as e:
            logger.error(f"❌ Failed to load config file: {e}")
            sys.exit(1)
    else:
        if not args.target_model:
            parser.error("target_model is required when not using --config")
        
        config = DraftFinderConfig(
            target_model=args.target_model,
            token=args.token or os.getenv('HF_TOKEN'),
            top_n=args.top_n if args.top_n is not None else 5,
            task=args.task or 'general',
            max_candidates=args.max_candidates if args.max_candidates is not None else 100,
            verbose=args.verbose,
            export=args.export
        )
    
    # Set logging level
    if config.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Create finder
    finder = DraftModelFinder(
        target_model_id=config.target_model,
        token=config.token,
        task=Task(config.task),
        verbose=config.verbose,
        max_candidates=config.max_candidates,
        strict_tokenization=not args.lenient_tokenization,
        prefer_official=True,
        show_test_details=args.show_test_details
    )
    
    # Find draft models
    try:
        candidates = finder.find_draft_models(top_n=config.top_n)
        
        # Print results
        finder.print_results(top_n=config.top_n)
        
        # Export if requested
        if config.export:
            finder.export_results(config.export)
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        if config.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
