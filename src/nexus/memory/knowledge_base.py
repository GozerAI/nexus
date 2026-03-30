
"""
Enhanced Knowledge Base System for Nexus AI Platform.
Provides unified access to factual, skill, and contextual knowledge.

Production hardening:
- Thread-safe via threading.RLock
- SQLite persistence (optional) survives restarts
- Configurable max_items cap with LRU eviction
- TTL-based expiry for stale items
"""

import hashlib
import json
import logging
import sqlite3
import threading
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class KnowledgeType(Enum):
    FACTUAL = "factual"
    PROCEDURAL = "procedural"
    CONTEXTUAL = "contextual"
    EXPERIENTIAL = "experiential"
    PATTERN = "pattern"


class KnowledgeConfidence(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"


@dataclass
class KnowledgeItem:
    """Represents a single piece of knowledge."""
    id: str
    content: Any
    knowledge_type: KnowledgeType
    confidence: float
    source: str
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    verification_status: bool = False
    related_items: List[str] = field(default_factory=list)
    context_tags: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# SQLite persistence helpers
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS knowledge_items (
    id            TEXT PRIMARY KEY,
    content       TEXT NOT NULL,
    knowledge_type TEXT NOT NULL,
    confidence    REAL NOT NULL,
    source        TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    last_accessed TEXT NOT NULL,
    access_count  INTEGER NOT NULL DEFAULT 0,
    verification_status INTEGER NOT NULL DEFAULT 0,
    related_items TEXT NOT NULL DEFAULT '[]',
    context_tags  TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS knowledge_graph (
    from_id TEXT NOT NULL,
    to_id   TEXT NOT NULL,
    PRIMARY KEY (from_id, to_id)
);
"""


def _item_to_row(item: KnowledgeItem) -> tuple:
    return (
        item.id,
        json.dumps(item.content) if not isinstance(item.content, str) else item.content,
        item.knowledge_type.value,
        item.confidence,
        item.source,
        item.created_at.isoformat(),
        item.last_accessed.isoformat(),
        item.access_count,
        int(item.verification_status),
        json.dumps(item.related_items or []),
        json.dumps(item.context_tags or []),
    )


def _row_to_item(row: tuple) -> KnowledgeItem:
    (
        kid, content, ktype, confidence, source,
        created_at, last_accessed, access_count,
        verification, related_raw, tags_raw,
    ) = row
    # Try to deserialise JSON content; fall back to plain string
    try:
        content = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        pass
    return KnowledgeItem(
        id=kid,
        content=content,
        knowledge_type=KnowledgeType(ktype),
        confidence=confidence,
        source=source,
        created_at=datetime.fromisoformat(created_at),
        last_accessed=datetime.fromisoformat(last_accessed),
        access_count=access_count,
        verification_status=bool(verification),
        related_items=json.loads(related_raw) if related_raw else [],
        context_tags=json.loads(tags_raw) if tags_raw else [],
    )


# ---------------------------------------------------------------------------
# KnowledgeBase
# ---------------------------------------------------------------------------

# Defaults
DEFAULT_MAX_ITEMS = 100_000
DEFAULT_ITEM_TTL_SECONDS = 30 * 24 * 3600  # 30 days


class KnowledgeBase:
    """
    Unified knowledge base that integrates different types of knowledge
    and provides intelligent retrieval and reasoning capabilities.

    Parameters
    ----------
    db_path : str | Path | None
        Path to a SQLite database for persistence.  ``None`` or ``":memory:"``
        keeps everything in-memory only.
    max_items : int
        Maximum number of items before LRU eviction kicks in.
    item_ttl : int
        Seconds after last access before an item is eligible for eviction.
    """

    def __init__(
        self,
        memory_manager=None,
        factual_memory=None,
        skill_memory=None,
        *,
        db_path: Optional[str] = None,
        max_items: int = DEFAULT_MAX_ITEMS,
        item_ttl: int = DEFAULT_ITEM_TTL_SECONDS,
    ):
        self.memory_manager = memory_manager
        self.factual_memory = factual_memory
        self.skill_memory = skill_memory

        # Limits
        self.max_items = max_items
        self.item_ttl = item_ttl

        # Thread safety
        self._lock = threading.RLock()

        # Knowledge storage
        self.knowledge_items: Dict[str, KnowledgeItem] = {}
        self.knowledge_graph: Dict[str, List[str]] = {}  # Relationships
        self.topic_index: Dict[str, List[str]] = {}      # Topic to knowledge mapping
        self.pattern_library: Dict[str, Any] = {}        # Common patterns

        # Caching and optimization
        self.query_cache: Dict[str, Tuple[List[KnowledgeItem], float]] = {}
        self.cache_ttl = 300  # 5 minutes

        # Persistence
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        if db_path and db_path != ":memory:":
            self._init_db(db_path)

        # Vector search (optional Qdrant integration)
        self._vector_store = None

        self.initialized = False

    # -- persistence ---------------------------------------------------------

    def _init_db(self, db_path: str) -> None:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def _persist_item(self, item: KnowledgeItem) -> None:
        if self._conn is None:
            return
        self._conn.execute(
            "INSERT OR REPLACE INTO knowledge_items "
            "(id, content, knowledge_type, confidence, source, created_at, "
            "last_accessed, access_count, verification_status, related_items, context_tags) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            _item_to_row(item),
        )
        self._conn.commit()

    def _delete_item_db(self, kid: str) -> None:
        if self._conn is None:
            return
        self._conn.execute("DELETE FROM knowledge_items WHERE id = ?", (kid,))
        self._conn.commit()

    def _load_from_db(self) -> None:
        """Load all persisted items into memory."""
        if self._conn is None:
            return
        cursor = self._conn.execute(
            "SELECT id, content, knowledge_type, confidence, source, "
            "created_at, last_accessed, access_count, verification_status, "
            "related_items, context_tags FROM knowledge_items"
        )
        for row in cursor.fetchall():
            item = _row_to_item(row)
            self.knowledge_items[item.id] = item
            self._index_knowledge_item(item)
        # Load graph
        graph_cursor = self._conn.execute("SELECT from_id, to_id FROM knowledge_graph")
        for from_id, to_id in graph_cursor.fetchall():
            self.knowledge_graph.setdefault(from_id, []).append(to_id)
        logger.info("Loaded %d items from persistent storage", len(self.knowledge_items))

    def _persist_graph_edge(self, from_id: str, to_id: str) -> None:
        if self._conn is None:
            return
        self._conn.execute(
            "INSERT OR IGNORE INTO knowledge_graph (from_id, to_id) VALUES (?, ?)",
            (from_id, to_id),
        )
        self._conn.commit()

    def close(self) -> None:
        """Close the underlying database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # -- eviction ------------------------------------------------------------

    def _evict_expired(self) -> int:
        """Remove items that have not been accessed within ``item_ttl``.
        Returns the number of items evicted.  Caller must hold ``_lock``."""
        if self.item_ttl <= 0:
            return 0
        cutoff = datetime.now() - timedelta(seconds=self.item_ttl)
        expired = [
            kid for kid, item in self.knowledge_items.items()
            if item.last_accessed < cutoff and item.source != "built-in"
        ]
        for kid in expired:
            del self.knowledge_items[kid]
            self._delete_item_db(kid)
        if expired:
            logger.info("Evicted %d expired knowledge items", len(expired))
        return len(expired)

    def _evict_lru(self, need: int = 1) -> int:
        """Evict least-recently-used items to make room.
        Caller must hold ``_lock``."""
        if len(self.knowledge_items) + need <= self.max_items:
            return 0
        # Sort by last_accessed ascending, skip built-in items
        candidates = sorted(
            ((kid, item) for kid, item in self.knowledge_items.items() if item.source != "built-in"),
            key=lambda x: x[1].last_accessed,
        )
        to_remove = len(self.knowledge_items) + need - self.max_items
        removed = 0
        for kid, _ in candidates[:to_remove]:
            del self.knowledge_items[kid]
            self._delete_item_db(kid)
            removed += 1
        if removed:
            logger.info("LRU-evicted %d knowledge items (max_items=%d)", removed, self.max_items)
        return removed

    # -- public API ----------------------------------------------------------

    def initialize(self):
        """Initialize the knowledge base system."""
        with self._lock:
            if self.initialized:
                return

            logger.info("Initializing Knowledge Base...")

            # Load persisted items first
            self._load_from_db()

            # Load initial knowledge if available (skip if already loaded from DB)
            if not self.knowledge_items:
                self._load_core_knowledge()

            self._build_initial_patterns()

            # Initialize vector search (optional, best-effort)
            self._init_vector_store()

            self.initialized = True
            logger.info("Knowledge Base initialized with %d items", len(self.knowledge_items))

    def _init_vector_store(self) -> None:
        """Initialize Qdrant vector store for semantic search (best-effort)."""
        try:
            import asyncio
            from nexus.storage.qdrant_store import QdrantVectorStore

            store = QdrantVectorStore()
            # Run async init in sync context
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Can't await in sync init — defer to first query
                    self._vector_store = store
                    logger.info("Qdrant vector store deferred (will init on first query)")
                    return
            except RuntimeError:
                pass

            success = asyncio.run(store.initialize())
            if success:
                self._vector_store = store
                logger.info("Qdrant vector store initialized for semantic search")
            else:
                logger.info("Qdrant not available — using keyword search only")
        except ImportError:
            logger.debug("qdrant-client not installed — keyword search only")
        except Exception as e:
            logger.debug("Vector store init skipped: %s", e)

    def add_knowledge(self, content: Any, knowledge_type: KnowledgeType,
                     source: str, confidence: float = 0.8,
                     context_tags: List[str] = None) -> str:
        """Add new knowledge to the base."""
        with self._lock:
            # Evict if at capacity
            self._evict_expired()
            self._evict_lru(need=1)

            content_hash = hashlib.md5(str(content).encode()).hexdigest()[:8]
            knowledge_id = f"{knowledge_type.value}_{int(time.time())}_{content_hash}"

            item = KnowledgeItem(
                id=knowledge_id,
                content=content,
                knowledge_type=knowledge_type,
                confidence=confidence,
                source=source,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                context_tags=context_tags or [],
                related_items=[],
            )

            self.knowledge_items[knowledge_id] = item

            # Index by topics
            self._index_knowledge_item(item)

            # Persist
            self._persist_item(item)

            # Store in appropriate memory system
            if self.memory_manager:
                meta = {
                    'type': knowledge_type.value,
                    'confidence': confidence,
                    'source': source,
                }
                if knowledge_type in (KnowledgeType.FACTUAL, KnowledgeType.CONTEXTUAL):
                    self.memory_manager.store_factual_knowledge(knowledge_id, content, meta)
                else:
                    self.memory_manager.store_skill_knowledge(knowledge_id, content, meta)

            # Index in vector store (async, best-effort)
            if self._vector_store:
                try:
                    import asyncio
                    content_str = str(content)
                    metadata = {
                        "knowledge_type": knowledge_type.value,
                        "source": source,
                        "confidence": confidence,
                    }
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            asyncio.ensure_future(
                                self._vector_store.upsert(knowledge_id, content_str, metadata)
                            )
                        else:
                            asyncio.run(self._vector_store.upsert(knowledge_id, content_str, metadata))
                    except RuntimeError:
                        pass
                except Exception:
                    pass

            logger.info("Added knowledge item: %s", knowledge_id)
            return knowledge_id

    def query_knowledge(self, query: str, knowledge_types: List[KnowledgeType] = None,
                       min_confidence: float = 0.3, max_results: int = 10) -> List[KnowledgeItem]:
        """Query the knowledge base for relevant information."""
        with self._lock:
            # Check cache first
            cache_key = hashlib.md5(
                f"{query}|{knowledge_types}|{min_confidence}|{max_results}".encode()
            ).hexdigest()
            if cache_key in self.query_cache:
                cached_result, cached_time = self.query_cache[cache_key]
                if time.time() - cached_time < self.cache_ttl:
                    return cached_result

            # Perform search
            results = []
            query_lower = query.lower()
            query_terms = query_lower.split()

            for item in self.knowledge_items.values():
                # Type filtering
                if knowledge_types and item.knowledge_type not in knowledge_types:
                    continue

                # Confidence filtering
                if item.confidence < min_confidence:
                    continue

                # Content matching
                relevance_score = self._calculate_relevance(item, query_terms)
                if relevance_score > 0:
                    item.last_accessed = datetime.now()
                    item.access_count += 1
                    results.append((item, relevance_score))

            # Sort by relevance and confidence
            results.sort(key=lambda x: (x[1], x[0].confidence), reverse=True)
            final_results = [item for item, score in results[:max_results]]

            # Cache results
            self.query_cache[cache_key] = (final_results, time.time())

            return final_results

    def get_related_knowledge(self, knowledge_id: str, max_depth: int = 2) -> List[KnowledgeItem]:
        """Get knowledge items related to a specific item."""
        with self._lock:
            if knowledge_id not in self.knowledge_items:
                return []

            visited = set()
            to_visit = [(knowledge_id, 0)]
            related_items = []

            while to_visit:
                current_id, depth = to_visit.pop(0)

                if current_id in visited or depth > max_depth:
                    continue

                visited.add(current_id)

                if current_id != knowledge_id and current_id in self.knowledge_items:
                    related_items.append(self.knowledge_items[current_id])

                # Add related items to visit
                if current_id in self.knowledge_graph:
                    for related_id in self.knowledge_graph[current_id]:
                        if related_id not in visited:
                            to_visit.append((related_id, depth + 1))

            return related_items

    def learn_from_interaction(self, query: str, selected_knowledge: str,
                              feedback_positive: bool):
        """Learn from user interactions to improve knowledge retrieval."""
        with self._lock:
            if selected_knowledge in self.knowledge_items:
                item = self.knowledge_items[selected_knowledge]

                # Adjust confidence based on feedback
                if feedback_positive:
                    item.confidence = min(1.0, item.confidence + 0.05)
                else:
                    item.confidence = max(0.1, item.confidence - 0.1)

                self._persist_item(item)

                # Learn query patterns
                query_terms = query.lower().split()
                self._update_pattern_library(query_terms, item.knowledge_type, feedback_positive)

                # Clear relevant cache entries
                self._invalidate_cache(query)

    def synthesize_knowledge(self, topic: str) -> Dict[str, Any]:
        """Synthesize knowledge from multiple sources on a topic."""
        relevant_items = self.query_knowledge(topic, max_results=20)

        if not relevant_items:
            return {"topic": topic, "synthesis": "No relevant knowledge found"}

        # Group by knowledge type
        by_type: Dict[str, list] = {}
        for item in relevant_items:
            ktype = item.knowledge_type.value
            if ktype not in by_type:
                by_type[ktype] = []
            by_type[ktype].append(item)

        # Create synthesis
        synthesis = {
            "topic": topic,
            "knowledge_types_found": list(by_type.keys()),
            "total_items": len(relevant_items),
            "confidence_range": [min(item.confidence for item in relevant_items),
                               max(item.confidence for item in relevant_items)],
            "synthesis": {},
            "created_at": datetime.now().isoformat()
        }

        # Synthesize by type
        for ktype, items in by_type.items():
            synthesis["synthesis"][ktype] = {
                "count": len(items),
                "avg_confidence": sum(item.confidence for item in items) / len(items),
                "key_points": [str(item.content)[:100] + "..." if len(str(item.content)) > 100
                             else str(item.content) for item in items[:3]],
                "sources": list(set(item.source for item in items))
            }

        return synthesis

    def get_knowledge_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        with self._lock:
            if not self.knowledge_items:
                return {"total_items": 0}

            by_type: Dict[str, int] = {}
            confidence_sum = 0.0
            access_sum = 0

            for item in self.knowledge_items.values():
                ktype = item.knowledge_type.value
                by_type[ktype] = by_type.get(ktype, 0) + 1
                confidence_sum += item.confidence
                access_sum += item.access_count

            return {
                "total_items": len(self.knowledge_items),
                "max_items": self.max_items,
                "by_type": by_type,
                "avg_confidence": confidence_sum / len(self.knowledge_items),
                "total_accesses": access_sum,
                "cache_entries": len(self.query_cache),
                "relationships": len(self.knowledge_graph),
                "topics_indexed": len(self.topic_index),
                "persistent": self._conn is not None,
            }

    # -- internal helpers ----------------------------------------------------

    def _load_core_knowledge(self):
        """Load initial core knowledge across multiple domains."""

        # Mathematics and Science
        math_science_knowledge = [
            ("Mathematics: 2 + 2 = 4", KnowledgeType.FACTUAL, "built-in", 1.0),
            ("Mathematical operations follow order of operations (PEMDAS)", KnowledgeType.PATTERN, "built-in", 0.95),
            ("The value of pi (π) is approximately 3.14159", KnowledgeType.FACTUAL, "built-in", 1.0),
            ("The speed of light is approximately 299,792,458 meters per second", KnowledgeType.FACTUAL, "built-in", 1.0),
            ("Water boils at 100°C (212°F) at standard atmospheric pressure", KnowledgeType.FACTUAL, "built-in", 1.0),
            ("Gravity on Earth is approximately 9.8 m/s²", KnowledgeType.FACTUAL, "built-in", 1.0),
            ("DNA stands for Deoxyribonucleic Acid", KnowledgeType.FACTUAL, "built-in", 1.0),
            ("The periodic table organizes elements by atomic number", KnowledgeType.FACTUAL, "built-in", 0.95),
            ("Photosynthesis converts sunlight into chemical energy in plants", KnowledgeType.FACTUAL, "built-in", 0.9),
        ]

        # Geography and World Knowledge
        geography_knowledge = [
            ("The capital of Florida is Tallahassee", KnowledgeType.FACTUAL, "built-in", 1.0),
            ("The capital of California is Sacramento", KnowledgeType.FACTUAL, "built-in", 1.0),
            ("The capital of New York is Albany", KnowledgeType.FACTUAL, "built-in", 1.0),
            ("The capital of Texas is Austin", KnowledgeType.FACTUAL, "built-in", 1.0),
            ("The capital of France is Paris", KnowledgeType.FACTUAL, "built-in", 1.0),
            ("The capital of Japan is Tokyo", KnowledgeType.FACTUAL, "built-in", 1.0),
            ("The capital of Australia is Canberra", KnowledgeType.FACTUAL, "built-in", 1.0),
            ("Mount Everest is the highest mountain on Earth at 8,849 meters", KnowledgeType.FACTUAL, "built-in", 1.0),
            ("The Pacific Ocean is the largest ocean on Earth", KnowledgeType.FACTUAL, "built-in", 1.0),
            ("There are seven continents: Asia, Africa, North America, South America, Antarctica, Europe, and Australia", KnowledgeType.FACTUAL, "built-in", 1.0),
        ]

        # History
        history_knowledge = [
            ("World War II ended in 1945", KnowledgeType.FACTUAL, "built-in", 1.0),
            ("The American Civil War was fought from 1861 to 1865", KnowledgeType.FACTUAL, "built-in", 1.0),
            ("The Declaration of Independence was signed in 1776", KnowledgeType.FACTUAL, "built-in", 1.0),
            ("The Berlin Wall fell in 1989", KnowledgeType.FACTUAL, "built-in", 1.0),
            ("Christopher Columbus reached the Americas in 1492", KnowledgeType.FACTUAL, "built-in", 0.95),
        ]

        # Technology and Computing
        technology_knowledge = [
            ("HTML stands for HyperText Markup Language", KnowledgeType.FACTUAL, "built-in", 1.0),
            ("Python is a high-level programming language", KnowledgeType.FACTUAL, "built-in", 1.0),
            ("The Internet uses TCP/IP protocols for communication", KnowledgeType.FACTUAL, "built-in", 0.9),
            ("RAM stands for Random Access Memory", KnowledgeType.FACTUAL, "built-in", 1.0),
            ("SQL stands for Structured Query Language", KnowledgeType.FACTUAL, "built-in", 1.0),
            ("HTTP stands for HyperText Transfer Protocol", KnowledgeType.FACTUAL, "built-in", 1.0),
        ]

        # Literature and Arts
        literature_arts_knowledge = [
            ("William Shakespeare wrote Romeo and Juliet", KnowledgeType.FACTUAL, "built-in", 1.0),
            ("The Mona Lisa was painted by Leonardo da Vinci", KnowledgeType.FACTUAL, "built-in", 1.0),
            ("To Kill a Mockingbird was written by Harper Lee", KnowledgeType.FACTUAL, "built-in", 1.0),
            ("Beethoven composed nine symphonies", KnowledgeType.FACTUAL, "built-in", 1.0),
        ]

        # Human Body and Health
        health_knowledge = [
            ("The human body has 206 bones in adulthood", KnowledgeType.FACTUAL, "built-in", 1.0),
            ("The heart is a muscle that pumps blood throughout the body", KnowledgeType.FACTUAL, "built-in", 1.0),
            ("Vitamin C helps prevent scurvy", KnowledgeType.FACTUAL, "built-in", 0.95),
            ("The brain consumes about 20% of the body's energy", KnowledgeType.FACTUAL, "built-in", 0.9),
        ]

        # Language and Communication
        language_knowledge = [
            ("English has approximately 26 letters in its alphabet", KnowledgeType.FACTUAL, "built-in", 1.0),
            ("Mandarin Chinese is the most spoken language by native speakers", KnowledgeType.FACTUAL, "built-in", 0.95),
            ("A noun is a word that names a person, place, thing, or idea", KnowledgeType.FACTUAL, "built-in", 1.0),
        ]

        # General Problem-Solving and Patterns
        procedural_knowledge = [
            ("Problem solving requires breaking down complex issues", KnowledgeType.PROCEDURAL, "built-in", 0.9),
            ("User questions often require clarification", KnowledgeType.EXPERIENTIAL, "built-in", 0.8),
            ("Critical thinking involves analyzing information objectively", KnowledgeType.PROCEDURAL, "built-in", 0.9),
            ("Research involves gathering information from reliable sources", KnowledgeType.PROCEDURAL, "built-in", 0.9),
        ]

        # Combine all knowledge domains
        all_knowledge = (math_science_knowledge + geography_knowledge +
                        history_knowledge + technology_knowledge +
                        literature_arts_knowledge + health_knowledge +
                        language_knowledge + procedural_knowledge)

        for content, ktype, source, confidence in all_knowledge:
            self.add_knowledge(content, ktype, source, confidence)

        logger.info("Loaded %d core knowledge items across multiple domains", len(all_knowledge))

    def _build_initial_patterns(self):
        """Build initial pattern library."""
        self.pattern_library = {
            "question_patterns": {
                "what_is": ["what is", "what are", "define"],
                "how_to": ["how to", "how do", "how can"],
                "why": ["why", "why does", "explain why"],
                "calculate": ["calculate", "compute", "solve", "find"]
            },
            "response_patterns": {
                "factual_response": "Based on factual knowledge: {}",
                "procedural_response": "Here's the procedure: {}",
                "uncertain_response": "I'm not certain, but: {}"
            }
        }

    def _index_knowledge_item(self, item: KnowledgeItem):
        """Index a knowledge item by topics and tags."""
        content_str = str(item.content).lower()
        terms = content_str.split()

        # Index by content terms
        for term in terms:
            if len(term) > 3:  # Skip short words
                if term not in self.topic_index:
                    self.topic_index[term] = []
                self.topic_index[term].append(item.id)

        # Index by context tags
        if item.context_tags:
            for tag in item.context_tags:
                if tag not in self.topic_index:
                    self.topic_index[tag] = []
                self.topic_index[tag].append(item.id)

    def _calculate_relevance(self, item: KnowledgeItem, query_terms: List[str]) -> float:
        """Calculate relevance score between knowledge item and query."""
        content_str = str(item.content).lower()
        relevance = 0.0

        # Direct term matching
        for term in query_terms:
            if term in content_str:
                relevance += 1.0

        # Context tag matching
        if item.context_tags:
            for tag in item.context_tags:
                for term in query_terms:
                    if term in tag.lower():
                        relevance += 0.5

        # Normalize by query length
        if query_terms:
            relevance = relevance / len(query_terms)

        return relevance

    def _update_pattern_library(self, query_terms: List[str], knowledge_type: KnowledgeType,
                               positive_feedback: bool):
        """Update pattern library based on user interactions."""
        pattern_key = f"{knowledge_type.value}_patterns"
        if pattern_key not in self.pattern_library:
            self.pattern_library[pattern_key] = {}

        # Track successful query patterns
        query_pattern = " ".join(query_terms[:3])  # First 3 terms
        if positive_feedback:
            if query_pattern not in self.pattern_library[pattern_key]:
                self.pattern_library[pattern_key][query_pattern] = 0
            self.pattern_library[pattern_key][query_pattern] += 1

    def _invalidate_cache(self, query: str):
        """Invalidate cache entries related to a query."""
        keys_to_remove = []
        for cache_key in self.query_cache.keys():
            if query.lower() in cache_key.lower():
                keys_to_remove.append(cache_key)

        for key in keys_to_remove:
            del self.query_cache[key]
