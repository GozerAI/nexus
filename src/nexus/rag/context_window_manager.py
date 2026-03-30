
"""
Context Window Manager for handling 150M+ token contexts efficiently.
Uses hierarchical compression, intelligent chunking, and multi-tier storage strategies.

Key Techniques for Maximizing Context Window:
1. Multi-algorithm compression (LZ-style, dictionary, delta, semantic)
2. Hierarchical storage tiers (hot/warm/cold/archive)
3. Intelligent deduplication with content fingerprinting
4. Token optimization (whitespace, redundancy removal)
5. Streaming/pagination for massive contexts
6. Priority-based eviction with LRU/LFU hybrid
7. Semantic summarization for low-priority content
8. Reference-based storage for repeated patterns
"""

import logging
import json
import zlib
import hashlib
import re
from typing import Dict, Any, List, Optional, Tuple, Iterator, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import heapq
from collections import defaultdict, deque, Counter
import statistics

logger = logging.getLogger(__name__)


# ==================== Utility Functions for Compression ====================

def _compute_content_hash(content: str) -> str:
    """Compute SHA-256 hash for content fingerprinting."""
    return hashlib.sha256(content.encode('utf-8', errors='replace')).hexdigest()


def _compute_similarity_hash(content: str, shingle_size: int = 5) -> str:
    """
    Compute locality-sensitive hash for near-duplicate detection.
    Uses min-hash of character shingles.
    """
    content_lower = content.lower()
    if len(content_lower) < shingle_size:
        return hashlib.md5(content_lower.encode()).hexdigest()[:16]

    # Generate shingles
    shingles = set()
    for i in range(len(content_lower) - shingle_size + 1):
        shingles.add(content_lower[i:i + shingle_size])

    # Compute min-hash (simplified)
    if not shingles:
        return hashlib.md5(content_lower.encode()).hexdigest()[:16]

    min_shingle = min(shingles)
    return hashlib.md5(min_shingle.encode()).hexdigest()[:16]


def _lz77_compress(data: str, window_size: int = 4096, lookahead_size: int = 256) -> List[Tuple]:
    """
    LZ77-style compression returning (offset, length, next_char) tuples.
    Returns list of tuples for reconstruction.
    """
    if not data:
        return []

    compressed = []
    pos = 0

    while pos < len(data):
        best_offset = 0
        best_length = 0

        # Search window
        window_start = max(0, pos - window_size)

        # Find longest match in window
        for offset in range(1, min(pos - window_start + 1, window_size)):
            match_pos = pos - offset
            length = 0

            while (length < lookahead_size and
                   pos + length < len(data) and
                   data[match_pos + (length % offset)] == data[pos + length]):
                length += 1

            if length > best_length:
                best_length = length
                best_offset = offset

        if best_length >= 3:  # Only encode if worthwhile
            next_char = data[pos + best_length] if pos + best_length < len(data) else ''
            compressed.append((best_offset, best_length, next_char))
            pos += best_length + (1 if next_char else 0)
        else:
            compressed.append((0, 0, data[pos]))
            pos += 1

    return compressed


def _lz77_decompress(compressed: List[Tuple]) -> str:
    """Decompress LZ77 compressed data."""
    result = []

    for offset, length, next_char in compressed:
        if offset == 0 and length == 0:
            result.append(next_char)
        else:
            start = len(result) - offset
            for i in range(length):
                result.append(result[start + (i % offset)])
            if next_char:
                result.append(next_char)

    return ''.join(result)


def _build_dictionary(texts: List[str], max_entries: int = 1000, min_freq: int = 3) -> Dict[str, int]:
    """
    Build a compression dictionary from common phrases.
    Returns mapping of phrase -> token_id.
    """
    # Count n-grams (2-5 words)
    phrase_counts = Counter()

    for text in texts:
        words = text.split()
        for n in range(2, 6):  # 2 to 5 word phrases
            for i in range(len(words) - n + 1):
                phrase = ' '.join(words[i:i + n])
                if len(phrase) > 10:  # Only meaningful phrases
                    phrase_counts[phrase] += 1

    # Select most common phrases
    common_phrases = [
        phrase for phrase, count in phrase_counts.most_common(max_entries)
        if count >= min_freq
    ]

    return {phrase: idx for idx, phrase in enumerate(common_phrases)}


def _dictionary_compress(text: str, dictionary: Dict[str, int]) -> Tuple[str, List[Tuple[int, int, int]]]:
    """
    Dictionary-based compression replacing common phrases with tokens.
    Returns (remaining_text, list of (position, length, token_id)).
    """
    if not dictionary:
        return text, []

    replacements = []

    # Sort dictionary by phrase length (longest first)
    sorted_phrases = sorted(dictionary.keys(), key=len, reverse=True)

    # Find and record replacements
    remaining = text
    offset_adjustment = 0

    for phrase in sorted_phrases:
        token_id = dictionary[phrase]
        pos = 0

        while True:
            idx = remaining.find(phrase, pos)
            if idx == -1:
                break

            replacements.append((idx + offset_adjustment, len(phrase), token_id))
            # Replace with placeholder
            remaining = remaining[:idx] + f"<T{token_id}>" + remaining[idx + len(phrase):]
            pos = idx + len(f"<T{token_id}>")

    return remaining, replacements


def _remove_redundant_whitespace(text: str) -> Tuple[str, Dict[int, int]]:
    """
    Remove redundant whitespace while tracking positions for reconstruction.
    Returns (compressed_text, position_mapping).
    """
    result = []
    position_map = {}
    original_pos = 0
    new_pos = 0
    prev_was_space = False

    for char in text:
        if char in ' \t':
            if not prev_was_space:
                result.append(' ')
                position_map[new_pos] = original_pos
                new_pos += 1
            prev_was_space = True
        elif char == '\n':
            result.append('\n')
            position_map[new_pos] = original_pos
            new_pos += 1
            prev_was_space = True
        else:
            result.append(char)
            position_map[new_pos] = original_pos
            new_pos += 1
            prev_was_space = False

        original_pos += 1

    return ''.join(result), position_map


def _extract_key_sentences(text: str, max_sentences: int = 10,
                          importance_keywords: Optional[Set[str]] = None) -> str:
    """
    Extract key sentences for semantic summarization.
    Uses keyword density and position-based scoring.
    """
    if importance_keywords is None:
        importance_keywords = {
            'important', 'key', 'main', 'critical', 'essential', 'must', 'should',
            'note', 'remember', 'summary', 'conclusion', 'result', 'therefore'
        }

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)

    if len(sentences) <= max_sentences:
        return text

    # Score sentences
    scored = []
    for i, sentence in enumerate(sentences):
        words = set(sentence.lower().split())

        # Keyword score
        keyword_score = len(words & importance_keywords) * 2

        # Position score (first and last sentences are important)
        if i < 3:
            position_score = 3 - i
        elif i >= len(sentences) - 2:
            position_score = 2
        else:
            position_score = 0

        # Length score (medium-length sentences preferred)
        length = len(sentence.split())
        length_score = 1 if 10 <= length <= 30 else 0

        total_score = keyword_score + position_score + length_score
        scored.append((total_score, i, sentence))

    # Select top sentences, maintaining original order
    scored.sort(reverse=True)
    selected_indices = sorted([item[1] for item in scored[:max_sentences]])

    return ' '.join(sentences[i] for i in selected_indices)

class CompressionStrategy(Enum):
    LOSSLESS = "lossless"
    PATTERN_BASED = "pattern_based"
    SEMANTIC_PRESERVING = "semantic_preserving"
    HIERARCHICAL = "hierarchical"
    ADAPTIVE = "adaptive"
    LZ77 = "lz77"
    DICTIONARY = "dictionary"
    DELTA = "delta"
    HYBRID = "hybrid"  # Combines multiple strategies
    AGGRESSIVE = "aggressive"  # Maximum compression


class PriorityLevel(Enum):
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    ARCHIVE = 1


class StorageTier(Enum):
    """Storage tiers for hierarchical context management."""
    HOT = "hot"  # Frequently accessed, minimal compression
    WARM = "warm"  # Moderate access, balanced compression
    COLD = "cold"  # Rarely accessed, heavy compression
    ARCHIVE = "archive"  # Historical, maximum compression + summarization
    REFERENCE = "reference"  # Deduplicated reference storage


class ChunkState(Enum):
    """State of a context chunk."""
    ACTIVE = "active"
    COMPRESSED = "compressed"
    SUMMARIZED = "summarized"
    ARCHIVED = "archived"
    EVICTED = "evicted"


@dataclass
class CompressionResult:
    """Result of a compression operation."""
    original_size: int
    compressed_size: int
    compression_ratio: float
    strategy_used: CompressionStrategy
    can_decompress: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StorageTierConfig:
    """Configuration for a storage tier."""
    tier: StorageTier
    max_size_tokens: int
    compression_level: int  # 0-9, higher = more aggressive
    access_threshold: int  # Access count to promote to higher tier
    age_threshold_hours: int  # Hours before demotion to lower tier
    summarization_enabled: bool = False

@dataclass
class ContextChunk:
    """A chunk of context with metadata and compression info."""
    chunk_id: str
    content: Any
    original_length: int
    compressed_length: int
    compression_ratio: float
    priority: PriorityLevel
    access_frequency: int
    last_accessed: datetime
    semantic_signature: str
    retrieval_keys: List[str]
    parent_chunks: List[str] = field(default_factory=list)
    child_chunks: List[str] = field(default_factory=list)
    # Enhanced fields for maximum context utilization
    storage_tier: StorageTier = StorageTier.HOT
    state: ChunkState = ChunkState.ACTIVE
    content_hash: str = ""  # For deduplication
    similarity_hash: str = ""  # For near-duplicate detection
    compressed_data: Optional[bytes] = None  # Actual compressed bytes
    compression_strategy: CompressionStrategy = CompressionStrategy.ADAPTIVE
    summary: Optional[str] = None  # Semantic summary for archived content
    reference_id: Optional[str] = None  # Reference to deduplicated content
    token_count: int = 0  # Actual token count (not just char length)
    created_at: datetime = field(default_factory=datetime.now)
    last_compressed: Optional[datetime] = None
    compression_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextWindow:
    """A context window managing multiple chunks with tiered storage."""
    window_id: str
    max_token_length: int
    current_token_length: int
    chunks: Dict[str, ContextChunk]
    chunk_hierarchy: Dict[str, List[str]]
    compression_statistics: Dict[str, float]
    performance_metrics: Dict[str, float]
    created_at: datetime
    last_optimized: datetime
    # Enhanced fields for massive context handling
    tier_allocation: Dict[StorageTier, int] = field(default_factory=dict)
    deduplication_index: Dict[str, List[str]] = field(default_factory=dict)  # hash -> chunk_ids
    dictionary: Dict[str, int] = field(default_factory=dict)  # Compression dictionary
    reference_pool: Dict[str, str] = field(default_factory=dict)  # Shared content references
    total_original_size: int = 0
    total_compressed_size: int = 0
    effective_compression_ratio: float = 0.0
    streaming_enabled: bool = False
    pagination_state: Dict[str, Any] = field(default_factory=dict)

class ContextWindowManager:
    """
    Manages context windows for 150M+ token contexts using intelligent
    compression, chunking, hierarchical organization, and multi-tier storage.

    Key Features for Maximum Context Utilization:
    - Multi-algorithm compression (LZ77, dictionary, delta, semantic)
    - Hierarchical storage tiers (hot/warm/cold/archive)
    - Intelligent deduplication with content fingerprinting
    - Token optimization and redundancy removal
    - Streaming/pagination for massive contexts
    - Priority-based eviction with LRU/LFU hybrid
    - Semantic summarization for archived content
    """

    def __init__(self, rag_engine, pattern_engine, knowledge_base):
        self.rag_engine = rag_engine
        self.pattern_engine = pattern_engine
        self.knowledge_base = knowledge_base

        # Configuration - Maximized for large contexts
        self.max_context_length = 150_000_000  # 150M tokens base
        self.effective_max_context = 500_000_000  # 500M with compression (3-4x)
        self.chunk_size_range = (1000, 100000)  # Min/max chunk sizes
        self.compression_threshold = 0.5  # More aggressive compression
        self.max_windows = 100
        self.deduplication_enabled = True
        self.streaming_chunk_size = 1_000_000  # 1M tokens per stream chunk

        # Storage tier configuration
        self.tier_configs = {
            StorageTier.HOT: StorageTierConfig(
                tier=StorageTier.HOT,
                max_size_tokens=10_000_000,  # 10M for hot content
                compression_level=1,
                access_threshold=10,
                age_threshold_hours=1
            ),
            StorageTier.WARM: StorageTierConfig(
                tier=StorageTier.WARM,
                max_size_tokens=40_000_000,  # 40M for warm content
                compression_level=5,
                access_threshold=3,
                age_threshold_hours=6
            ),
            StorageTier.COLD: StorageTierConfig(
                tier=StorageTier.COLD,
                max_size_tokens=100_000_000,  # 100M for cold content
                compression_level=7,
                access_threshold=1,
                age_threshold_hours=24
            ),
            StorageTier.ARCHIVE: StorageTierConfig(
                tier=StorageTier.ARCHIVE,
                max_size_tokens=350_000_000,  # 350M for archived (summarized)
                compression_level=9,
                access_threshold=0,
                age_threshold_hours=168,  # 1 week
                summarization_enabled=True
            )
        }

        # Storage
        self.active_windows: Dict[str, ContextWindow] = {}
        self.chunk_registry: Dict[str, ContextChunk] = {}
        self.priority_queue: List[Tuple[float, str]] = []  # (priority_score, chunk_id)
        self.compression_cache: Dict[str, bytes] = {}

        # Deduplication storage
        self.content_hash_index: Dict[str, str] = {}  # hash -> canonical_chunk_id
        self.similarity_hash_index: Dict[str, Set[str]] = defaultdict(set)  # sim_hash -> chunk_ids
        self.reference_pool: Dict[str, str] = {}  # reference_id -> content

        # Global compression dictionary (built from all content)
        self.global_dictionary: Dict[str, int] = {}
        self.dictionary_inverse: Dict[int, str] = {}
        self.dictionary_update_threshold = 10000  # Rebuild after N chunks

        # Performance tracking
        self.metrics = {
            "total_chunks_created": 0,
            "total_compressions": 0,
            "total_decompressions": 0,
            "avg_compression_ratio": 0.0,
            "cache_hit_rate": 0.0,
            "window_utilization": 0.0,
            "deduplication_savings": 0,
            "tier_promotions": 0,
            "tier_demotions": 0,
            "effective_capacity_multiplier": 1.0
        }

        # Compression strategies - expanded
        self.compression_strategies = {
            CompressionStrategy.LOSSLESS: self._lossless_compression,
            CompressionStrategy.PATTERN_BASED: self._pattern_based_compression,
            CompressionStrategy.SEMANTIC_PRESERVING: self._semantic_preserving_compression,
            CompressionStrategy.HIERARCHICAL: self._hierarchical_compression,
            CompressionStrategy.ADAPTIVE: self._adaptive_compression,
            CompressionStrategy.LZ77: self._lz77_compression,
            CompressionStrategy.DICTIONARY: self._dictionary_compression,
            CompressionStrategy.DELTA: self._delta_compression,
            CompressionStrategy.HYBRID: self._hybrid_compression,
            CompressionStrategy.AGGRESSIVE: self._aggressive_compression
        }

        # LRU/LFU hybrid eviction tracking
        self.access_history: deque = deque(maxlen=100000)
        self.frequency_counter: Counter = Counter()

        self.initialized = False
        self.is_initialized = False  # Alias for compatibility
    
    def initialize(self):
        """Initialize the context window manager."""
        if self.initialized:
            return
        
        logger.info("Initializing Context Window Manager for 150M token contexts...")
        
        # Create initial context window
        self._create_initial_window()
        
        # Setup compression algorithms
        self._setup_compression_algorithms()
        
        # Initialize performance monitoring
        self._initialize_performance_monitoring()
        
        self.initialized = True
        logger.info("Context Window Manager initialized successfully")
    
    def create_context_window(self, content: Any, window_type: str = "general",
                            compression_strategy: CompressionStrategy = CompressionStrategy.ADAPTIVE) -> str:
        """Create a new context window with intelligent chunking."""
        window_id = f"window_{int(datetime.now().timestamp())}_{hash(str(content)[:100]) % 10000}"
        
        # Analyze content for optimal chunking
        chunk_strategy = self._determine_chunking_strategy(content, window_type)
        
        # Create chunks
        chunks = self._create_chunks(content, chunk_strategy, compression_strategy)
        
        # Build chunk hierarchy
        chunk_hierarchy = self._build_chunk_hierarchy(chunks)
        
        # Calculate compression statistics
        compression_stats = self._calculate_compression_statistics(chunks)
        
        # Create context window
        window = ContextWindow(
            window_id=window_id,
            max_token_length=self.max_context_length,
            current_token_length=sum(chunk.compressed_length for chunk in chunks.values()),
            chunks={chunk.chunk_id: chunk for chunk in chunks.values()},
            chunk_hierarchy=chunk_hierarchy,
            compression_statistics=compression_stats,
            performance_metrics={},
            created_at=datetime.now(),
            last_optimized=datetime.now()
        )
        
        # Store window and update registries
        self.active_windows[window_id] = window
        for chunk in chunks.values():
            self.chunk_registry[chunk.chunk_id] = chunk
            self._update_priority_queue(chunk)
        
        # Optimize window if needed
        if window.current_token_length > self.max_context_length * 0.8:
            self._optimize_window(window_id)
        
        logger.info(f"Created context window {window_id} with {len(chunks)} chunks")
        return window_id
    
    def retrieve_context(self, window_id: str, query: str, max_chunks: int = 10) -> Dict[str, Any]:
        """Retrieve relevant context chunks based on query."""
        if window_id not in self.active_windows:
            return {"error": "Window not found"}
        
        window = self.active_windows[window_id]
        
        # Score chunks based on relevance to query
        chunk_scores = []
        for chunk_id, chunk in window.chunks.items():
            relevance_score = self._calculate_chunk_relevance(chunk, query)
            chunk_scores.append((relevance_score, chunk_id, chunk))
        
        # Sort by relevance and select top chunks
        chunk_scores.sort(reverse=True, key=lambda x: x[0])
        selected_chunks = chunk_scores[:max_chunks]
        
        # Decompress and prepare context
        context_data = []
        total_tokens = 0
        
        for score, chunk_id, chunk in selected_chunks:
            # Decompress chunk if needed
            decompressed_content = self._decompress_chunk(chunk)
            
            # Update access statistics
            chunk.access_frequency += 1
            chunk.last_accessed = datetime.now()
            
            context_data.append({
                "chunk_id": chunk_id,
                "content": decompressed_content,
                "relevance_score": score,
                "priority": chunk.priority.value,
                "semantic_signature": chunk.semantic_signature
            })
            
            total_tokens += chunk.original_length
            
            # Stop if we exceed reasonable context limits for processing
            if total_tokens > 1_000_000:  # 1M token processing limit
                break
        
        return {
            "window_id": window_id,
            "query": query,
            "selected_chunks": context_data,
            "total_chunks": len(context_data),
            "total_tokens": total_tokens,
            "retrieval_strategy": "relevance_based"
        }
    
    def compress_context_window(self, window_id: str, 
                              strategy: CompressionStrategy = CompressionStrategy.ADAPTIVE) -> Dict[str, Any]:
        """Compress a context window to reduce memory usage."""
        if window_id not in self.active_windows:
            return {"error": "Window not found"}
        
        window = self.active_windows[window_id]
        compression_results = {
            "original_size": window.current_token_length,
            "compressed_size": 0,
            "chunks_compressed": 0,
            "compression_ratio": 0.0,
            "strategy_used": strategy.value
        }
        
        # Compress chunks based on priority and access patterns
        for chunk_id, chunk in window.chunks.items():
            if chunk.compression_ratio < self.compression_threshold:
                # Compress this chunk
                compression_func = self.compression_strategies[strategy]
                compressed_chunk = compression_func(chunk)
                
                # Update chunk with compressed data
                window.chunks[chunk_id] = compressed_chunk
                self.chunk_registry[chunk_id] = compressed_chunk
                
                compression_results["chunks_compressed"] += 1
        
        # Recalculate window statistics
        window.current_token_length = sum(chunk.compressed_length for chunk in window.chunks.values())
        compression_results["compressed_size"] = window.current_token_length
        compression_results["compression_ratio"] = (
            compression_results["original_size"] - compression_results["compressed_size"]
        ) / compression_results["original_size"] if compression_results["original_size"] > 0 else 0
        
        # Update metrics
        self.metrics["total_compressions"] += compression_results["chunks_compressed"]
        self._update_compression_metrics()
        
        logger.info(f"Compressed window {window_id}: {compression_results['compression_ratio']:.2%} reduction")
        return compression_results
    
    def optimize_all_windows(self) -> Dict[str, Any]:
        """Optimize all active context windows for better performance."""
        optimization_results = {
            "windows_optimized": 0,
            "total_space_saved": 0,
            "performance_improvements": [],
            "optimization_strategies": []
        }
        
        for window_id in list(self.active_windows.keys()):
            window_results = self._optimize_window(window_id)
            
            if window_results["optimized"]:
                optimization_results["windows_optimized"] += 1
                optimization_results["total_space_saved"] += window_results["space_saved"]
                optimization_results["performance_improvements"].append(window_results["performance_gain"])
                optimization_results["optimization_strategies"].extend(window_results["strategies_used"])
        
        # Global optimizations
        self._perform_global_optimizations()
        
        logger.info(f"Optimized {optimization_results['windows_optimized']} context windows")
        return optimization_results
    
    def create_learning_context_window(self, learning_pathway: Dict[str, Any], 
                                     user_profile: Dict[str, Any]) -> str:
        """Create a specialized context window for learning pathways."""
        # Extract learning content and goals
        learning_content = self._extract_learning_content(learning_pathway, user_profile)
        
        # Create adaptive context window
        window_id = self.create_context_window(
            content=learning_content,
            window_type="learning",
            compression_strategy=CompressionStrategy.SEMANTIC_PRESERVING
        )
        
        # Optimize for learning patterns
        if window_id in self.active_windows:
            window = self.active_windows[window_id]
            self._optimize_for_learning(window, user_profile)
        
        return window_id
    
    def _create_chunks(self, content: Any, chunk_strategy: Dict[str, Any], 
                      compression_strategy: CompressionStrategy) -> Dict[str, ContextChunk]:
        """Create chunks from content using the specified strategy."""
        chunks = {}
        content_str = str(content)
        
        # Determine chunk boundaries
        if chunk_strategy["type"] == "semantic":
            boundaries = self._find_semantic_boundaries(content_str)
        elif chunk_strategy["type"] == "syntactic":
            boundaries = self._find_syntactic_boundaries(content_str)
        else:
            boundaries = self._find_uniform_boundaries(content_str, chunk_strategy["size"])
        
        # Create chunks
        for i, (start, end) in enumerate(boundaries):
            chunk_content = content_str[start:end]
            chunk_id = f"chunk_{int(datetime.now().timestamp())}_{i}"
            
            # Calculate semantic signature
            semantic_signature = self._calculate_semantic_signature(chunk_content)
            
            # Extract retrieval keys
            retrieval_keys = self._extract_chunk_retrieval_keys(chunk_content)
            
            # Determine priority
            priority = self._calculate_chunk_priority(chunk_content, retrieval_keys)
            
            chunk = ContextChunk(
                chunk_id=chunk_id,
                content=chunk_content,
                original_length=len(chunk_content),
                compressed_length=len(chunk_content),  # Will be updated after compression
                compression_ratio=0.0,
                priority=priority,
                access_frequency=0,
                last_accessed=datetime.now(),
                semantic_signature=semantic_signature,
                retrieval_keys=retrieval_keys
            )
            
            chunks[chunk_id] = chunk
            self.metrics["total_chunks_created"] += 1
        
        return chunks
    
    def get_manager_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the context window manager."""
        total_chunks = len(self.chunk_registry)
        total_windows = len(self.active_windows)
        
        if total_chunks == 0:
            return {"total_windows": total_windows, "total_chunks": 0}
        
        # Calculate memory usage
        total_original_size = sum(chunk.original_length for chunk in self.chunk_registry.values())
        total_compressed_size = sum(chunk.compressed_length for chunk in self.chunk_registry.values())
        
        # Calculate access patterns
        access_frequencies = [chunk.access_frequency for chunk in self.chunk_registry.values()]
        avg_access_frequency = sum(access_frequencies) / len(access_frequencies)
        
        return {
            "total_windows": total_windows,
            "total_chunks": total_chunks,
            "memory_usage": {
                "original_size_mb": total_original_size / (1024 * 1024),
                "compressed_size_mb": total_compressed_size / (1024 * 1024),
                "compression_ratio": (total_original_size - total_compressed_size) / total_original_size if total_original_size > 0 else 0,
                "space_saved_mb": (total_original_size - total_compressed_size) / (1024 * 1024)
            },
            "access_patterns": {
                "avg_access_frequency": avg_access_frequency,
                "most_accessed_chunks": sorted(
                    [(chunk.chunk_id, chunk.access_frequency) for chunk in self.chunk_registry.values()],
                    key=lambda x: x[1], reverse=True
                )[:5]
            },
            "performance_metrics": self.metrics,
            "context_capacity": {
                "max_tokens": self.max_context_length,
                "current_usage": sum(window.current_token_length for window in self.active_windows.values()),
                "utilization_percentage": (sum(window.current_token_length for window in self.active_windows.values()) / self.max_context_length) * 100
            }
        }
    
    # Placeholder implementations for helper methods
    def _create_initial_window(self):
        """Create initial context window."""
        pass
    
    def _setup_compression_algorithms(self):
        """Setup compression algorithms."""
        pass
    
    def _initialize_performance_monitoring(self):
        """Initialize performance monitoring."""
        pass
    
    def _lossless_compression(self, chunk: ContextChunk) -> ContextChunk:
        """
        Apply lossless compression using zlib (DEFLATE algorithm).
        Guarantees perfect reconstruction of original content.
        """
        content_str = str(chunk.content)
        content_bytes = content_str.encode('utf-8', errors='replace')

        # Apply zlib compression
        compressed = zlib.compress(content_bytes, level=6)

        chunk.compressed_data = compressed
        chunk.compressed_length = len(compressed)
        chunk.compression_ratio = chunk.compressed_length / max(chunk.original_length, 1)
        chunk.compression_strategy = CompressionStrategy.LOSSLESS
        chunk.state = ChunkState.COMPRESSED
        chunk.last_compressed = datetime.now()
        chunk.compression_metadata = {
            "algorithm": "zlib",
            "level": 6,
            "original_encoding": "utf-8"
        }

        return chunk

    def _pattern_based_compression(self, chunk: ContextChunk) -> ContextChunk:
        """
        Apply pattern-based compression using recognized patterns.
        Replaces repeated patterns with references.
        """
        content_str = str(chunk.content)

        # Find repeated patterns in content
        pattern_index = {}
        compressed_parts = []
        pos = 0

        # Simple pattern detection: repeated substrings
        min_pattern_len = 20
        content_len = len(content_str)

        while pos < content_len:
            best_match = None
            best_length = 0

            # Search for patterns
            for pattern_start in range(max(0, pos - 10000), pos):
                for length in range(min_pattern_len, min(500, content_len - pos, pos - pattern_start)):
                    pattern = content_str[pattern_start:pattern_start + length]
                    if content_str[pos:pos + length] == pattern:
                        if length > best_length:
                            best_length = length
                            best_match = (pattern_start, length)

            if best_match and best_length >= min_pattern_len:
                # Store reference instead of content
                ref_start, ref_len = best_match
                compressed_parts.append(f"<REF:{ref_start}:{ref_len}>")
                pos += ref_len
            else:
                compressed_parts.append(content_str[pos])
                pos += 1

        compressed_content = ''.join(compressed_parts)
        compressed_bytes = compressed_content.encode('utf-8', errors='replace')

        # Apply additional zlib compression
        final_compressed = zlib.compress(compressed_bytes, level=6)

        chunk.compressed_data = final_compressed
        chunk.compressed_length = len(final_compressed)
        chunk.compression_ratio = chunk.compressed_length / max(chunk.original_length, 1)
        chunk.compression_strategy = CompressionStrategy.PATTERN_BASED
        chunk.state = ChunkState.COMPRESSED
        chunk.last_compressed = datetime.now()
        chunk.compression_metadata = {
            "algorithm": "pattern+zlib",
            "patterns_found": len([p for p in compressed_parts if p.startswith("<REF:")])
        }

        return chunk

    def _semantic_preserving_compression(self, chunk: ContextChunk) -> ContextChunk:
        """
        Apply semantic-preserving compression.
        Removes redundant whitespace and normalizes text while preserving meaning.
        """
        content_str = str(chunk.content)

        # Remove redundant whitespace
        compressed_str, _ = _remove_redundant_whitespace(content_str)

        # Normalize common patterns
        compressed_str = re.sub(r'\s+', ' ', compressed_str)  # Multiple spaces to single
        compressed_str = re.sub(r'\n\s*\n', '\n\n', compressed_str)  # Multiple newlines to double

        # Apply zlib compression
        compressed_bytes = compressed_str.encode('utf-8', errors='replace')
        final_compressed = zlib.compress(compressed_bytes, level=7)

        chunk.compressed_data = final_compressed
        chunk.compressed_length = len(final_compressed)
        chunk.compression_ratio = chunk.compressed_length / max(chunk.original_length, 1)
        chunk.compression_strategy = CompressionStrategy.SEMANTIC_PRESERVING
        chunk.state = ChunkState.COMPRESSED
        chunk.last_compressed = datetime.now()

        return chunk

    def _hierarchical_compression(self, chunk: ContextChunk) -> ContextChunk:
        """
        Apply hierarchical compression based on content structure.
        Preserves structure while compressing content at different levels.
        """
        content_str = str(chunk.content)

        # Detect content hierarchy (headers, sections, etc.)
        lines = content_str.split('\n')
        hierarchy_levels = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith('#'):
                level = len(stripped) - len(stripped.lstrip('#'))
                hierarchy_levels.append((level, stripped))
            elif stripped.isupper() and len(stripped) > 3:
                hierarchy_levels.append((1, stripped))  # Likely header
            else:
                hierarchy_levels.append((0, stripped))

        # Compress based on level (deeper levels get more compression)
        compressed_parts = []
        for level, content in hierarchy_levels:
            if level > 2:
                # Heavy compression for deep content
                compressed_parts.append(_extract_key_sentences(content, max_sentences=2))
            elif level > 0:
                # Moderate compression for headers
                compressed_parts.append(content)
            else:
                # Light compression for body text
                compressed_parts.append(_extract_key_sentences(content, max_sentences=5))

        compressed_str = '\n'.join(compressed_parts)
        compressed_bytes = compressed_str.encode('utf-8', errors='replace')
        final_compressed = zlib.compress(compressed_bytes, level=8)

        chunk.compressed_data = final_compressed
        chunk.compressed_length = len(final_compressed)
        chunk.compression_ratio = chunk.compressed_length / max(chunk.original_length, 1)
        chunk.compression_strategy = CompressionStrategy.HIERARCHICAL
        chunk.state = ChunkState.COMPRESSED
        chunk.last_compressed = datetime.now()

        return chunk

    def _adaptive_compression(self, chunk: ContextChunk) -> ContextChunk:
        """
        Apply adaptive compression based on chunk characteristics.
        Selects optimal strategy based on content analysis.
        """
        content_str = str(chunk.content)
        content_len = len(content_str)

        # Analyze content characteristics
        word_count = len(content_str.split())
        unique_words = len(set(content_str.lower().split()))
        repetition_ratio = 1 - (unique_words / max(word_count, 1))

        # Choose strategy based on analysis
        if chunk.priority == PriorityLevel.CRITICAL:
            return self._lossless_compression(chunk)
        elif repetition_ratio > 0.5:
            # High repetition -> pattern-based or dictionary
            return self._dictionary_compression(chunk)
        elif content_len > 50000:
            # Large content -> hierarchical
            return self._hierarchical_compression(chunk)
        elif chunk.storage_tier == StorageTier.ARCHIVE:
            # Archived -> aggressive
            return self._aggressive_compression(chunk)
        else:
            # Default -> semantic preserving
            return self._semantic_preserving_compression(chunk)

    def _lz77_compression(self, chunk: ContextChunk) -> ContextChunk:
        """
        Apply LZ77-style compression for repetitive content.
        Uses sliding window to find and encode repeated sequences.
        """
        content_str = str(chunk.content)

        # Apply custom LZ77 compression
        compressed_tuples = _lz77_compress(content_str)

        # Serialize compressed data
        serialized = json.dumps(compressed_tuples)
        serialized_bytes = serialized.encode('utf-8')

        # Apply additional zlib compression
        final_compressed = zlib.compress(serialized_bytes, level=6)

        chunk.compressed_data = final_compressed
        chunk.compressed_length = len(final_compressed)
        chunk.compression_ratio = chunk.compressed_length / max(chunk.original_length, 1)
        chunk.compression_strategy = CompressionStrategy.LZ77
        chunk.state = ChunkState.COMPRESSED
        chunk.last_compressed = datetime.now()
        chunk.compression_metadata = {
            "algorithm": "lz77+zlib",
            "tuple_count": len(compressed_tuples)
        }

        return chunk

    def _dictionary_compression(self, chunk: ContextChunk) -> ContextChunk:
        """
        Apply dictionary-based compression using global phrase dictionary.
        Replaces common phrases with short tokens.
        """
        content_str = str(chunk.content)

        # Use global dictionary if available
        if self.global_dictionary:
            compressed_str, replacements = _dictionary_compress(content_str, self.global_dictionary)
        else:
            # Build local dictionary
            local_dict = _build_dictionary([content_str], max_entries=100, min_freq=2)
            compressed_str, replacements = _dictionary_compress(content_str, local_dict)

        # Serialize with replacement info
        package = {
            "text": compressed_str,
            "replacements": replacements
        }
        serialized = json.dumps(package)
        serialized_bytes = serialized.encode('utf-8')

        # Apply zlib compression
        final_compressed = zlib.compress(serialized_bytes, level=7)

        chunk.compressed_data = final_compressed
        chunk.compressed_length = len(final_compressed)
        chunk.compression_ratio = chunk.compressed_length / max(chunk.original_length, 1)
        chunk.compression_strategy = CompressionStrategy.DICTIONARY
        chunk.state = ChunkState.COMPRESSED
        chunk.last_compressed = datetime.now()
        chunk.compression_metadata = {
            "algorithm": "dictionary+zlib",
            "replacements_count": len(replacements)
        }

        return chunk

    def _delta_compression(self, chunk: ContextChunk) -> ContextChunk:
        """
        Apply delta compression for content similar to existing chunks.
        Stores only differences from a reference chunk.
        """
        content_str = str(chunk.content)

        # Find similar chunk using similarity hash
        similar_chunk_id = None
        if chunk.similarity_hash and chunk.similarity_hash in self.similarity_hash_index:
            candidates = self.similarity_hash_index[chunk.similarity_hash]
            for candidate_id in candidates:
                if candidate_id != chunk.chunk_id and candidate_id in self.chunk_registry:
                    similar_chunk_id = candidate_id
                    break

        if similar_chunk_id:
            # Compute delta from reference
            reference_chunk = self.chunk_registry[similar_chunk_id]
            reference_content = str(reference_chunk.content)

            # Simple delta: store differences
            delta = self._compute_text_delta(reference_content, content_str)

            package = {
                "reference_id": similar_chunk_id,
                "delta": delta
            }
        else:
            # No reference, store full content
            package = {
                "reference_id": None,
                "content": content_str
            }

        serialized = json.dumps(package)
        serialized_bytes = serialized.encode('utf-8')
        final_compressed = zlib.compress(serialized_bytes, level=6)

        chunk.compressed_data = final_compressed
        chunk.compressed_length = len(final_compressed)
        chunk.compression_ratio = chunk.compressed_length / max(chunk.original_length, 1)
        chunk.compression_strategy = CompressionStrategy.DELTA
        chunk.state = ChunkState.COMPRESSED
        chunk.last_compressed = datetime.now()
        chunk.compression_metadata = {
            "algorithm": "delta+zlib",
            "has_reference": similar_chunk_id is not None,
            "reference_id": similar_chunk_id
        }

        return chunk

    def _hybrid_compression(self, chunk: ContextChunk) -> ContextChunk:
        """
        Apply hybrid compression combining multiple strategies.
        Uses dictionary + LZ77 + zlib for maximum compression.
        """
        content_str = str(chunk.content)

        # Step 1: Dictionary compression
        if self.global_dictionary:
            content_str, replacements = _dictionary_compress(content_str, self.global_dictionary)
        else:
            replacements = []

        # Step 2: Whitespace normalization
        content_str, _ = _remove_redundant_whitespace(content_str)

        # Step 3: Apply zlib with maximum compression
        content_bytes = content_str.encode('utf-8', errors='replace')
        final_compressed = zlib.compress(content_bytes, level=9)

        # Package with metadata for decompression
        package = {
            "data": final_compressed.hex(),
            "replacements": replacements
        }
        package_bytes = json.dumps(package).encode('utf-8')

        chunk.compressed_data = zlib.compress(package_bytes, level=9)
        chunk.compressed_length = len(chunk.compressed_data)
        chunk.compression_ratio = chunk.compressed_length / max(chunk.original_length, 1)
        chunk.compression_strategy = CompressionStrategy.HYBRID
        chunk.state = ChunkState.COMPRESSED
        chunk.last_compressed = datetime.now()
        chunk.compression_metadata = {
            "algorithm": "hybrid(dict+norm+zlib)",
            "compression_stages": 3
        }

        return chunk

    def _aggressive_compression(self, chunk: ContextChunk) -> ContextChunk:
        """
        Apply aggressive compression with semantic summarization.
        Used for archived content where some information loss is acceptable.
        Maximum space savings at cost of some detail.
        """
        content_str = str(chunk.content)

        # Step 1: Extract key sentences (semantic summarization)
        summarized = _extract_key_sentences(content_str, max_sentences=20)

        # Store full content hash for verification
        content_hash = _compute_content_hash(content_str)

        # Step 2: Remove all redundant whitespace
        summarized, _ = _remove_redundant_whitespace(summarized)

        # Step 3: Maximum zlib compression
        compressed = zlib.compress(summarized.encode('utf-8'), level=9)

        # Store summary separately for quick access
        chunk.summary = summarized[:500]  # First 500 chars of summary

        package = {
            "compressed": compressed.hex(),
            "original_hash": content_hash,
            "original_length": chunk.original_length,
            "is_summarized": True
        }
        package_bytes = json.dumps(package).encode('utf-8')

        chunk.compressed_data = zlib.compress(package_bytes, level=9)
        chunk.compressed_length = len(chunk.compressed_data)
        chunk.compression_ratio = chunk.compressed_length / max(chunk.original_length, 1)
        chunk.compression_strategy = CompressionStrategy.AGGRESSIVE
        chunk.state = ChunkState.SUMMARIZED
        chunk.last_compressed = datetime.now()
        chunk.compression_metadata = {
            "algorithm": "aggressive(summarize+zlib)",
            "summarization_ratio": len(summarized) / max(len(content_str), 1),
            "original_hash": content_hash
        }

        return chunk

    def _compute_text_delta(self, reference: str, target: str) -> List[Tuple[str, int, str]]:
        """
        Compute delta between reference and target text.
        Returns list of operations: ('keep', length), ('insert', text), ('delete', length)
        """
        delta = []

        # Simple longest common subsequence approach
        ref_words = reference.split()
        tgt_words = target.split()

        i, j = 0, 0
        while i < len(ref_words) and j < len(tgt_words):
            if ref_words[i] == tgt_words[j]:
                delta.append(('keep', 1, ''))
                i += 1
                j += 1
            elif j + 1 < len(tgt_words) and ref_words[i] == tgt_words[j + 1]:
                delta.append(('insert', 0, tgt_words[j]))
                j += 1
            else:
                delta.append(('replace', 1, tgt_words[j]))
                i += 1
                j += 1

        # Handle remaining
        while j < len(tgt_words):
            delta.append(('insert', 0, tgt_words[j]))
            j += 1

        return delta
    
    def _determine_chunking_strategy(self, content, window_type):
        """Determine optimal chunking strategy."""
        return {"type": "uniform", "size": 10000}
    
    def _find_uniform_boundaries(self, content_str, chunk_size):
        """Find uniform chunk boundaries."""
        boundaries = []
        for i in range(0, len(content_str), chunk_size):
            boundaries.append((i, min(i + chunk_size, len(content_str))))
        return boundaries
    
    def _find_semantic_boundaries(self, content_str):
        """Find semantic boundaries in text."""
        # Simple sentence-based boundaries
        sentences = content_str.split('.')
        boundaries = []
        start = 0
        for sentence in sentences:
            end = start + len(sentence) + 1  # +1 for the period
            if end <= len(content_str):
                boundaries.append((start, end))
            start = end
        return boundaries if boundaries else [(0, len(content_str))]
    
    def _find_syntactic_boundaries(self, content_str):
        """Find syntactic boundaries in text."""
        # Simple paragraph-based boundaries
        paragraphs = content_str.split('\n\n')
        boundaries = []
        start = 0
        for paragraph in paragraphs:
            end = start + len(paragraph) + 2  # +2 for double newline
            if end <= len(content_str):
                boundaries.append((start, end))
            start = end
        return boundaries if boundaries else [(0, len(content_str))]
    
    def _calculate_semantic_signature(self, content):
        """Calculate semantic signature for content."""
        # Simple hash-based signature
        import hashlib
        return hashlib.md5(str(content).encode()).hexdigest()[:16]
    
    def _extract_chunk_retrieval_keys(self, content):
        """Extract retrieval keys from chunk content."""
        # Simple word extraction
        words = str(content).lower().split()
        return list(set([word for word in words if len(word) > 3]))[:10]
    
    def _calculate_chunk_priority(self, content, retrieval_keys):
        """Calculate chunk priority."""
        # Simple heuristic based on content length and key count
        if len(retrieval_keys) > 5:
            return PriorityLevel.HIGH
        elif len(str(content)) > 1000:
            return PriorityLevel.MEDIUM
        else:
            return PriorityLevel.LOW

    # ==================== Deduplication Methods ====================

    def deduplicate_chunk(self, chunk: ContextChunk) -> Tuple[ContextChunk, bool]:
        """
        Deduplicate chunk against existing content.
        Returns (chunk, is_duplicate) tuple.
        """
        content_str = str(chunk.content)

        # Compute hashes
        content_hash = _compute_content_hash(content_str)
        similarity_hash = _compute_similarity_hash(content_str)

        chunk.content_hash = content_hash
        chunk.similarity_hash = similarity_hash

        # Check for exact duplicate
        if content_hash in self.content_hash_index:
            canonical_id = self.content_hash_index[content_hash]
            chunk.reference_id = canonical_id
            chunk.content = None  # Remove content, use reference
            chunk.compressed_length = 32  # Just the reference
            chunk.compression_ratio = chunk.compressed_length / max(chunk.original_length, 1)
            chunk.state = ChunkState.COMPRESSED
            self.metrics["deduplication_savings"] += chunk.original_length

            logger.debug(f"Exact duplicate found: {chunk.chunk_id} -> {canonical_id}")
            return chunk, True

        # Check for near-duplicate (similarity hash collision)
        if similarity_hash in self.similarity_hash_index:
            candidates = self.similarity_hash_index[similarity_hash]
            for candidate_id in candidates:
                if candidate_id in self.chunk_registry:
                    candidate = self.chunk_registry[candidate_id]
                    # Use delta compression instead of full storage
                    chunk = self._delta_compression(chunk)
                    logger.debug(f"Near-duplicate found, using delta: {chunk.chunk_id}")
                    break

        # Register this chunk
        self.content_hash_index[content_hash] = chunk.chunk_id
        self.similarity_hash_index[similarity_hash].add(chunk.chunk_id)

        return chunk, False

    def build_global_dictionary(self, sample_size: int = 1000) -> Dict[str, int]:
        """
        Build global compression dictionary from existing chunks.
        Improves compression for new content.
        """
        # Sample content from chunks
        sample_texts = []
        chunk_list = list(self.chunk_registry.values())[:sample_size]

        for chunk in chunk_list:
            if chunk.content:
                sample_texts.append(str(chunk.content))

        if not sample_texts:
            return {}

        # Build dictionary
        self.global_dictionary = _build_dictionary(
            sample_texts,
            max_entries=2000,
            min_freq=5
        )

        # Build inverse dictionary for decompression
        self.dictionary_inverse = {v: k for k, v in self.global_dictionary.items()}

        logger.info(f"Built global dictionary with {len(self.global_dictionary)} entries")
        return self.global_dictionary

    # ==================== Tier Management Methods ====================

    def promote_chunk_tier(self, chunk_id: str) -> bool:
        """Promote chunk to higher storage tier based on access patterns."""
        if chunk_id not in self.chunk_registry:
            return False

        chunk = self.chunk_registry[chunk_id]
        current_tier = chunk.storage_tier

        # Determine new tier based on access frequency
        tier_order = [StorageTier.ARCHIVE, StorageTier.COLD, StorageTier.WARM, StorageTier.HOT]
        current_idx = tier_order.index(current_tier)

        if current_idx < len(tier_order) - 1:
            new_tier = tier_order[current_idx + 1]
            new_config = self.tier_configs[new_tier]

            # Check if promotion criteria met
            if chunk.access_frequency >= new_config.access_threshold:
                chunk.storage_tier = new_tier

                # Decompress if moving to hotter tier
                if new_tier in [StorageTier.HOT, StorageTier.WARM]:
                    self._decompress_chunk(chunk)

                self.metrics["tier_promotions"] += 1
                logger.debug(f"Promoted {chunk_id} from {current_tier} to {new_tier}")
                return True

        return False

    def demote_chunk_tier(self, chunk_id: str) -> bool:
        """Demote chunk to lower storage tier based on age and access."""
        if chunk_id not in self.chunk_registry:
            return False

        chunk = self.chunk_registry[chunk_id]
        current_tier = chunk.storage_tier

        tier_order = [StorageTier.ARCHIVE, StorageTier.COLD, StorageTier.WARM, StorageTier.HOT]
        current_idx = tier_order.index(current_tier)

        if current_idx > 0:
            new_tier = tier_order[current_idx - 1]
            config = self.tier_configs[current_tier]

            # Check if demotion criteria met
            age_hours = (datetime.now() - chunk.last_accessed).total_seconds() / 3600
            if age_hours >= config.age_threshold_hours:
                chunk.storage_tier = new_tier

                # Apply appropriate compression for new tier
                new_config = self.tier_configs[new_tier]
                if new_config.summarization_enabled and new_tier == StorageTier.ARCHIVE:
                    self._aggressive_compression(chunk)
                elif new_config.compression_level >= 7:
                    self._hybrid_compression(chunk)
                else:
                    self._adaptive_compression(chunk)

                self.metrics["tier_demotions"] += 1
                logger.debug(f"Demoted {chunk_id} from {current_tier} to {new_tier}")
                return True

        return False

    def rebalance_tiers(self) -> Dict[str, Any]:
        """Rebalance chunks across storage tiers for optimal memory usage."""
        results = {
            "promotions": 0,
            "demotions": 0,
            "tier_usage": {},
            "space_recovered": 0
        }

        # Calculate current tier usage
        tier_usage = {tier: 0 for tier in StorageTier}
        for chunk in self.chunk_registry.values():
            tier_usage[chunk.storage_tier] += chunk.compressed_length

        # Process chunks for tier changes
        for chunk_id in list(self.chunk_registry.keys()):
            chunk = self.chunk_registry[chunk_id]
            tier_config = self.tier_configs[chunk.storage_tier]

            # Check for promotion
            if chunk.access_frequency >= tier_config.access_threshold:
                if self.promote_chunk_tier(chunk_id):
                    results["promotions"] += 1

            # Check for demotion
            age_hours = (datetime.now() - chunk.last_accessed).total_seconds() / 3600
            if age_hours >= tier_config.age_threshold_hours:
                old_size = chunk.compressed_length
                if self.demote_chunk_tier(chunk_id):
                    results["demotions"] += 1
                    results["space_recovered"] += old_size - chunk.compressed_length

        # Recalculate tier usage
        for tier in StorageTier:
            results["tier_usage"][tier.value] = sum(
                c.compressed_length for c in self.chunk_registry.values()
                if c.storage_tier == tier
            )

        return results

    # ==================== Streaming/Pagination Methods ====================

    def create_streaming_window(self, content_iterator: Iterator[str],
                               window_id: Optional[str] = None) -> str:
        """
        Create a streaming context window for massive content.
        Processes content in chunks without loading all into memory.
        """
        if window_id is None:
            window_id = f"stream_{int(datetime.now().timestamp())}"

        # Initialize streaming window
        window = ContextWindow(
            window_id=window_id,
            max_token_length=self.effective_max_context,
            current_token_length=0,
            chunks={},
            chunk_hierarchy={},
            compression_statistics={},
            performance_metrics={},
            created_at=datetime.now(),
            last_optimized=datetime.now(),
            streaming_enabled=True,
            pagination_state={"current_page": 0, "total_pages": 0}
        )

        self.active_windows[window_id] = window

        # Process content stream
        chunk_index = 0
        buffer = ""

        for content_piece in content_iterator:
            buffer += content_piece

            # Create chunks when buffer is large enough
            while len(buffer) >= self.streaming_chunk_size:
                chunk_content = buffer[:self.streaming_chunk_size]
                buffer = buffer[self.streaming_chunk_size:]

                chunk = self._create_streaming_chunk(
                    chunk_content, window_id, chunk_index
                )
                window.chunks[chunk.chunk_id] = chunk
                self.chunk_registry[chunk.chunk_id] = chunk
                window.current_token_length += chunk.compressed_length
                chunk_index += 1

                # Apply tier-based eviction if needed
                if window.current_token_length > self.effective_max_context * 0.9:
                    self._evict_lowest_priority_chunks(window_id)

        # Process remaining buffer
        if buffer:
            chunk = self._create_streaming_chunk(buffer, window_id, chunk_index)
            window.chunks[chunk.chunk_id] = chunk
            self.chunk_registry[chunk.chunk_id] = chunk
            window.current_token_length += chunk.compressed_length

        window.pagination_state["total_pages"] = (
            len(window.chunks) // 10 + (1 if len(window.chunks) % 10 else 0)
        )

        logger.info(f"Created streaming window {window_id} with {len(window.chunks)} chunks")
        return window_id

    def _create_streaming_chunk(self, content: str, window_id: str,
                               index: int) -> ContextChunk:
        """Create a chunk for streaming window with immediate compression."""
        chunk_id = f"{window_id}_chunk_{index}"

        chunk = ContextChunk(
            chunk_id=chunk_id,
            content=content,
            original_length=len(content),
            compressed_length=len(content),
            compression_ratio=0.0,
            priority=PriorityLevel.MEDIUM,
            access_frequency=0,
            last_accessed=datetime.now(),
            semantic_signature=self._calculate_semantic_signature(content),
            retrieval_keys=self._extract_chunk_retrieval_keys(content),
            storage_tier=StorageTier.WARM,
            content_hash=_compute_content_hash(content),
            similarity_hash=_compute_similarity_hash(content)
        )

        # Immediately compress and deduplicate
        chunk, is_dup = self.deduplicate_chunk(chunk)
        if not is_dup:
            chunk = self._hybrid_compression(chunk)

        return chunk

    def get_paginated_context(self, window_id: str, page: int,
                             page_size: int = 10) -> Dict[str, Any]:
        """Get paginated context from a window."""
        if window_id not in self.active_windows:
            return {"error": "Window not found"}

        window = self.active_windows[window_id]
        chunk_list = list(window.chunks.values())

        start_idx = page * page_size
        end_idx = min(start_idx + page_size, len(chunk_list))

        page_chunks = []
        for chunk in chunk_list[start_idx:end_idx]:
            content = self._decompress_chunk(chunk)
            page_chunks.append({
                "chunk_id": chunk.chunk_id,
                "content": content,
                "compressed_size": chunk.compressed_length,
                "original_size": chunk.original_length,
                "tier": chunk.storage_tier.value
            })

        return {
            "window_id": window_id,
            "page": page,
            "page_size": page_size,
            "total_pages": (len(chunk_list) // page_size) +
                          (1 if len(chunk_list) % page_size else 0),
            "total_chunks": len(chunk_list),
            "chunks": page_chunks
        }

    # ==================== Eviction Methods ====================

    def _evict_lowest_priority_chunks(self, window_id: str,
                                      target_reduction: float = 0.2) -> int:
        """
        Evict lowest priority chunks to free space.
        Uses LRU/LFU hybrid scoring.
        """
        if window_id not in self.active_windows:
            return 0

        window = self.active_windows[window_id]
        target_size = int(window.current_token_length * (1 - target_reduction))

        # Score chunks for eviction (lower score = more likely to evict)
        chunk_scores = []
        for chunk_id, chunk in window.chunks.items():
            # LRU component
            age_hours = (datetime.now() - chunk.last_accessed).total_seconds() / 3600
            lru_score = 1.0 / (age_hours + 1)

            # LFU component
            lfu_score = chunk.access_frequency / max(
                max(c.access_frequency for c in window.chunks.values()), 1
            )

            # Priority component
            priority_score = chunk.priority.value / 5.0

            # Combined score (higher = keep, lower = evict)
            combined_score = (lru_score * 0.3 + lfu_score * 0.3 + priority_score * 0.4)
            chunk_scores.append((combined_score, chunk_id, chunk))

        # Sort by score (lowest first for eviction)
        chunk_scores.sort(key=lambda x: x[0])

        evicted_count = 0
        for score, chunk_id, chunk in chunk_scores:
            if window.current_token_length <= target_size:
                break

            # Archive instead of delete for important content
            if chunk.priority.value >= PriorityLevel.HIGH.value:
                # Move to archive tier with aggressive compression
                chunk.storage_tier = StorageTier.ARCHIVE
                self._aggressive_compression(chunk)
            else:
                # Evict completely
                window.current_token_length -= chunk.compressed_length
                del window.chunks[chunk_id]
                chunk.state = ChunkState.EVICTED

            evicted_count += 1

        logger.info(f"Evicted {evicted_count} chunks from window {window_id}")
        return evicted_count

    # ==================== Context Maximization Methods ====================

    def maximize_context_capacity(self, window_id: str) -> Dict[str, Any]:
        """
        Apply all available techniques to maximize context window capacity.
        Returns statistics about capacity gained.
        """
        if window_id not in self.active_windows:
            return {"error": "Window not found"}

        window = self.active_windows[window_id]
        initial_size = window.current_token_length
        initial_original = window.total_original_size or sum(
            c.original_length for c in window.chunks.values()
        )

        results = {
            "initial_size": initial_size,
            "initial_original_size": initial_original,
            "techniques_applied": [],
            "space_saved": 0,
            "effective_capacity_multiplier": 1.0
        }

        # 1. Deduplication pass
        dedup_savings = 0
        for chunk_id in list(window.chunks.keys()):
            chunk = window.chunks[chunk_id]
            if chunk.content and not chunk.reference_id:
                _, is_dup = self.deduplicate_chunk(chunk)
                if is_dup:
                    dedup_savings += chunk.original_length - chunk.compressed_length

        if dedup_savings > 0:
            results["techniques_applied"].append({
                "technique": "deduplication",
                "savings": dedup_savings
            })

        # 2. Build/update global dictionary
        if len(self.chunk_registry) > self.dictionary_update_threshold:
            self.build_global_dictionary()
            results["techniques_applied"].append({
                "technique": "dictionary_update",
                "entries": len(self.global_dictionary)
            })

        # 3. Apply hybrid compression to all chunks
        compression_savings = 0
        for chunk_id, chunk in window.chunks.items():
            if chunk.compression_ratio > 0.5:  # Not well compressed
                old_size = chunk.compressed_length
                self._hybrid_compression(chunk)
                compression_savings += old_size - chunk.compressed_length

        if compression_savings > 0:
            results["techniques_applied"].append({
                "technique": "hybrid_compression",
                "savings": compression_savings
            })

        # 4. Tier rebalancing
        tier_results = self.rebalance_tiers()
        if tier_results["space_recovered"] > 0:
            results["techniques_applied"].append({
                "technique": "tier_rebalancing",
                "savings": tier_results["space_recovered"]
            })

        # 5. Archive old, low-priority content
        archive_savings = 0
        for chunk_id, chunk in window.chunks.items():
            age_hours = (datetime.now() - chunk.last_accessed).total_seconds() / 3600
            if (age_hours > 24 and chunk.priority.value <= PriorityLevel.LOW.value
                    and chunk.state != ChunkState.SUMMARIZED):
                old_size = chunk.compressed_length
                self._aggressive_compression(chunk)
                chunk.storage_tier = StorageTier.ARCHIVE
                archive_savings += old_size - chunk.compressed_length

        if archive_savings > 0:
            results["techniques_applied"].append({
                "technique": "archival_summarization",
                "savings": archive_savings
            })

        # Calculate final statistics
        final_size = sum(c.compressed_length for c in window.chunks.values())
        window.current_token_length = final_size
        window.total_compressed_size = final_size
        window.total_original_size = initial_original

        results["final_size"] = final_size
        results["space_saved"] = initial_size - final_size
        results["total_space_saved"] = initial_original - final_size
        results["effective_capacity_multiplier"] = initial_original / max(final_size, 1)

        # Update window compression ratio
        window.effective_compression_ratio = final_size / max(initial_original, 1)

        # Update global metrics
        self.metrics["effective_capacity_multiplier"] = results["effective_capacity_multiplier"]

        logger.info(
            f"Maximized context capacity for {window_id}: "
            f"{results['effective_capacity_multiplier']:.2f}x multiplier"
        )

        return results

    def get_effective_context_capacity(self) -> Dict[str, Any]:
        """
        Calculate effective context capacity across all windows.
        Shows how much original content can be stored.
        """
        total_original = sum(
            chunk.original_length for chunk in self.chunk_registry.values()
        )
        total_compressed = sum(
            chunk.compressed_length for chunk in self.chunk_registry.values()
        )
        dedup_savings = self.metrics.get("deduplication_savings", 0)

        # Calculate effective capacity
        if total_compressed > 0:
            avg_compression_ratio = total_compressed / max(total_original, 1)
            effective_multiplier = 1 / max(avg_compression_ratio, 0.1)
        else:
            effective_multiplier = 1.0

        effective_capacity = int(self.max_context_length * effective_multiplier)

        return {
            "base_capacity": self.max_context_length,
            "effective_capacity": effective_capacity,
            "capacity_multiplier": effective_multiplier,
            "current_usage": {
                "original_size": total_original,
                "compressed_size": total_compressed,
                "compression_ratio": total_compressed / max(total_original, 1),
                "deduplication_savings": dedup_savings
            },
            "tier_distribution": {
                tier.value: sum(
                    c.compressed_length for c in self.chunk_registry.values()
                    if c.storage_tier == tier
                )
                for tier in StorageTier
            },
            "available_capacity": effective_capacity - total_compressed,
            "utilization_percentage": (total_compressed / effective_capacity) * 100
        }

    def get_relevant_context(self, query: str, max_tokens: int = 100000) -> List[Dict[str, Any]]:
        """
        Get relevant context for a query, optimized for the RAG orchestrator.
        Retrieves and decompresses most relevant chunks.
        """
        results = []
        total_tokens = 0

        # Score all chunks by relevance
        scored_chunks = []
        for chunk_id, chunk in self.chunk_registry.items():
            relevance = self._calculate_chunk_relevance(chunk, query)
            scored_chunks.append((relevance, chunk_id, chunk))

        # Sort by relevance
        scored_chunks.sort(reverse=True, key=lambda x: x[0])

        # Retrieve top chunks within token limit
        for relevance, chunk_id, chunk in scored_chunks:
            if total_tokens >= max_tokens:
                break

            content = self._decompress_chunk(chunk)
            chunk_tokens = len(content.split())  # Approximate token count

            if total_tokens + chunk_tokens <= max_tokens:
                results.append({
                    "chunk_id": chunk_id,
                    "content": content,
                    "relevance": relevance,
                    "tier": chunk.storage_tier.value,
                    "tokens": chunk_tokens
                })
                total_tokens += chunk_tokens

                # Update access patterns
                chunk.access_frequency += 1
                chunk.last_accessed = datetime.now()

        return results

    def _decompress_chunk(self, chunk: ContextChunk) -> str:
        """Decompress a chunk and return its content."""
        # If content is available directly
        if chunk.content and chunk.state == ChunkState.ACTIVE:
            return str(chunk.content)

        # If using reference
        if chunk.reference_id and chunk.reference_id in self.chunk_registry:
            ref_chunk = self.chunk_registry[chunk.reference_id]
            return self._decompress_chunk(ref_chunk)

        # If compressed data available
        if chunk.compressed_data:
            try:
                # First level decompression
                decompressed = zlib.decompress(chunk.compressed_data)

                # Check if it's a package (JSON)
                try:
                    package = json.loads(decompressed.decode('utf-8'))

                    if "compressed" in package:
                        # Aggressive compression format
                        inner_data = bytes.fromhex(package["compressed"])
                        return zlib.decompress(inner_data).decode('utf-8')
                    elif "data" in package:
                        # Hybrid compression format
                        inner_data = bytes.fromhex(package["data"])
                        return zlib.decompress(inner_data).decode('utf-8')
                    elif "text" in package:
                        # Dictionary compression format
                        return package["text"]
                    else:
                        return decompressed.decode('utf-8')
                except (json.JSONDecodeError, ValueError):
                    return decompressed.decode('utf-8')

            except Exception as e:
                logger.error(f"Decompression failed for {chunk.chunk_id}: {e}")
                return chunk.summary or ""

        # Fallback to summary
        return chunk.summary or ""

    def _calculate_chunk_relevance(self, chunk: ContextChunk, query: str) -> float:
        """Calculate relevance score of a chunk to a query."""
        query_lower = query.lower()
        query_words = set(query_lower.split())

        # Check retrieval keys overlap
        key_overlap = len(set(chunk.retrieval_keys) & query_words)
        key_score = key_overlap / max(len(query_words), 1)

        # Check content overlap if available
        content_score = 0.0
        if chunk.content:
            content_words = set(str(chunk.content).lower().split())
            content_overlap = len(query_words & content_words)
            content_score = content_overlap / max(len(query_words), 1)
        elif chunk.summary:
            summary_words = set(chunk.summary.lower().split())
            summary_overlap = len(query_words & summary_words)
            content_score = summary_overlap / max(len(query_words), 1) * 0.7

        # Boost for priority
        priority_boost = chunk.priority.value / 10.0

        # Boost for recent access
        recency_hours = (datetime.now() - chunk.last_accessed).total_seconds() / 3600
        recency_boost = 1.0 / (recency_hours + 1) * 0.1

        return key_score * 0.4 + content_score * 0.4 + priority_boost + recency_boost

    def _build_chunk_hierarchy(self, chunks: Dict[str, ContextChunk]) -> Dict[str, List[str]]:
        """Build hierarchy relationships between chunks."""
        hierarchy = {}
        chunk_list = list(chunks.values())

        for i, chunk in enumerate(chunk_list):
            hierarchy[chunk.chunk_id] = []
            # Link to adjacent chunks
            if i > 0:
                chunk.parent_chunks.append(chunk_list[i - 1].chunk_id)
            if i < len(chunk_list) - 1:
                chunk.child_chunks.append(chunk_list[i + 1].chunk_id)
                hierarchy[chunk.chunk_id].append(chunk_list[i + 1].chunk_id)

        return hierarchy

    def _calculate_compression_statistics(self, chunks: Dict[str, ContextChunk]) -> Dict[str, float]:
        """Calculate compression statistics for chunks."""
        if not chunks:
            return {}

        total_original = sum(c.original_length for c in chunks.values())
        total_compressed = sum(c.compressed_length for c in chunks.values())

        return {
            "total_original": total_original,
            "total_compressed": total_compressed,
            "overall_ratio": total_compressed / max(total_original, 1),
            "avg_chunk_ratio": statistics.mean(
                c.compression_ratio for c in chunks.values()
            ) if chunks else 0.0
        }

    def _optimize_window(self, window_id: str) -> Dict[str, Any]:
        """Optimize a specific window."""
        return self.maximize_context_capacity(window_id)

    def _perform_global_optimizations(self):
        """Perform optimizations across all windows."""
        # Rebuild dictionary periodically
        if len(self.chunk_registry) > self.dictionary_update_threshold:
            self.build_global_dictionary()

        # Rebalance tiers
        self.rebalance_tiers()

    def _update_priority_queue(self, chunk: ContextChunk):
        """Update priority queue for eviction management."""
        score = (
            chunk.priority.value * 0.4 +
            chunk.access_frequency * 0.3 +
            (1.0 / ((datetime.now() - chunk.last_accessed).total_seconds() / 3600 + 1)) * 0.3
        )
        heapq.heappush(self.priority_queue, (score, chunk.chunk_id))

    def _update_compression_metrics(self):
        """Update compression metrics."""
        if self.chunk_registry:
            ratios = [c.compression_ratio for c in self.chunk_registry.values()]
            self.metrics["avg_compression_ratio"] = statistics.mean(ratios)

    def _extract_learning_content(self, learning_pathway: Dict, user_profile: Dict) -> str:
        """Extract content for learning context window."""
        content_parts = []

        if "topics" in learning_pathway:
            content_parts.append("Topics: " + ", ".join(learning_pathway["topics"]))

        if "content" in learning_pathway:
            content_parts.append(str(learning_pathway["content"]))

        if "goals" in learning_pathway:
            content_parts.append("Goals: " + ", ".join(learning_pathway["goals"]))

        return "\n\n".join(content_parts)

    def _optimize_for_learning(self, window: ContextWindow, user_profile: Dict):
        """Optimize window specifically for learning patterns."""
        # Prioritize chunks based on learning goals
        learning_keywords = user_profile.get("interests", [])

        for chunk in window.chunks.values():
            # Boost priority for relevant content
            for keyword in learning_keywords:
                if keyword.lower() in str(chunk.retrieval_keys).lower():
                    if chunk.priority.value < PriorityLevel.HIGH.value:
                        chunk.priority = PriorityLevel(chunk.priority.value + 1)
