"""
Content Relationships and Prerequisites Graph for Content Library.

Provides:
- Prerequisite management
- Content relationship tracking
- Learning path generation
- Dependency analysis
- Topological sorting for sequencing
"""

import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import defaultdict, deque
from enum import Enum

from .models import ContentItem, ContentFilters, DifficultyLevel

logger = logging.getLogger(__name__)


# =============================================================================
# Relationship Types
# =============================================================================

class RelationshipType(Enum):
    """Types of relationships between content items."""
    PREREQUISITE = "prerequisite"       # Must complete before
    RELATED = "related"                 # Conceptually related
    ALTERNATIVE = "alternative"         # Alternative coverage
    EXTENSION = "extension"             # Extends/deepens
    SIMPLIFICATION = "simplification"   # Simplified version
    TRANSLATION = "translation"         # Different language
    SEQUEL = "sequel"                   # Follows sequentially
    PART_OF = "part_of"                # Part of larger content


@dataclass
class ContentRelationship:
    """Represents a relationship between two content items."""
    source_id: str
    target_id: str
    relationship_type: RelationshipType
    weight: float = 1.0  # Relationship strength/importance
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class LearningPath:
    """A sequence of content items forming a learning path."""
    path_id: str
    title: str
    description: str
    content_ids: List[str]
    total_time_minutes: int
    difficulty_progression: List[str]
    topics_covered: List[str]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Content Graph
# =============================================================================

class ContentGraph:
    """
    Manage content relationships and prerequisites.

    Provides graph-based operations for:
    - Prerequisite chains
    - Related content discovery
    - Learning path generation
    - Dependency validation
    """

    def __init__(self, content_library=None, storage_backend=None):
        """
        Initialize content graph.

        Args:
            content_library: Reference to content library
            storage_backend: Optional storage backend (used if library not provided)
        """
        self.library = content_library
        self.storage = storage_backend

        # Graph structures
        # Forward edges: source -> list of (target, relationship)
        self.edges: Dict[str, List[ContentRelationship]] = defaultdict(list)
        # Reverse edges: target -> list of (source, relationship)
        self.reverse_edges: Dict[str, List[ContentRelationship]] = defaultdict(list)

        # Prerequisite-specific structures for efficiency
        self.prerequisites: Dict[str, Set[str]] = defaultdict(set)  # content -> prereqs
        self.dependents: Dict[str, Set[str]] = defaultdict(set)      # prereq -> dependents

        logger.info("ContentGraph initialized")

    def _get_content(self, content_id: str) -> Optional[ContentItem]:
        """Get content from library or storage."""
        if self.library:
            return self.library.get_content(content_id)
        elif self.storage:
            return self.storage.get(content_id)
        return None

    # =========================================================================
    # Relationship Management
    # =========================================================================

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: RelationshipType,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ContentRelationship:
        """
        Add a relationship between content items.

        Args:
            source_id: Source content ID
            target_id: Target content ID
            relationship_type: Type of relationship
            weight: Relationship weight/importance
            metadata: Additional metadata

        Returns:
            Created ContentRelationship
        """
        relationship = ContentRelationship(
            source_id=source_id,
            target_id=target_id,
            relationship_type=relationship_type,
            weight=weight,
            metadata=metadata or {}
        )

        self.edges[source_id].append(relationship)
        self.reverse_edges[target_id].append(relationship)

        # Update prerequisite structures
        if relationship_type == RelationshipType.PREREQUISITE:
            self.prerequisites[source_id].add(target_id)
            self.dependents[target_id].add(source_id)

        logger.debug(f"Added {relationship_type.value} relationship: {source_id} -> {target_id}")
        return relationship

    def add_prerequisite(self, content_id: str, prerequisite_id: str, weight: float = 1.0):
        """
        Add a prerequisite relationship.

        content_id requires prerequisite_id to be completed first.

        Args:
            content_id: Content that has the prerequisite
            prerequisite_id: Required prerequisite content
            weight: Importance of prerequisite
        """
        return self.add_relationship(
            content_id, prerequisite_id,
            RelationshipType.PREREQUISITE,
            weight
        )

    def remove_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: Optional[RelationshipType] = None
    ) -> bool:
        """
        Remove a relationship between content items.

        Args:
            source_id: Source content ID
            target_id: Target content ID
            relationship_type: Optional type filter (removes all types if None)

        Returns:
            True if any relationship was removed
        """
        removed = False

        # Remove from forward edges
        original_len = len(self.edges[source_id])
        self.edges[source_id] = [
            r for r in self.edges[source_id]
            if not (r.target_id == target_id and
                   (relationship_type is None or r.relationship_type == relationship_type))
        ]
        if len(self.edges[source_id]) < original_len:
            removed = True

        # Remove from reverse edges
        self.reverse_edges[target_id] = [
            r for r in self.reverse_edges[target_id]
            if not (r.source_id == source_id and
                   (relationship_type is None or r.relationship_type == relationship_type))
        ]

        # Update prerequisite structures
        if relationship_type is None or relationship_type == RelationshipType.PREREQUISITE:
            self.prerequisites[source_id].discard(target_id)
            self.dependents[target_id].discard(source_id)

        return removed

    def remove_prerequisite(self, content_id: str, prerequisite_id: str) -> bool:
        """Remove a prerequisite relationship."""
        return self.remove_relationship(
            content_id, prerequisite_id,
            RelationshipType.PREREQUISITE
        )

    def get_relationships(
        self,
        content_id: str,
        relationship_type: Optional[RelationshipType] = None,
        direction: str = "outgoing"  # "outgoing", "incoming", "both"
    ) -> List[ContentRelationship]:
        """
        Get relationships for a content item.

        Args:
            content_id: Content ID
            relationship_type: Optional type filter
            direction: Relationship direction

        Returns:
            List of ContentRelationships
        """
        relationships = []

        if direction in ("outgoing", "both"):
            for rel in self.edges[content_id]:
                if relationship_type is None or rel.relationship_type == relationship_type:
                    relationships.append(rel)

        if direction in ("incoming", "both"):
            for rel in self.reverse_edges[content_id]:
                if relationship_type is None or rel.relationship_type == relationship_type:
                    relationships.append(rel)

        return relationships

    # =========================================================================
    # Prerequisite Operations
    # =========================================================================

    def get_prerequisites(self, content_id: str) -> Set[str]:
        """Get direct prerequisites for content."""
        return self.prerequisites[content_id].copy()

    def get_all_prerequisites(self, content_id: str) -> Set[str]:
        """Get all prerequisites (transitive closure)."""
        all_prereqs = set()
        to_process = list(self.prerequisites[content_id])

        while to_process:
            prereq = to_process.pop()
            if prereq not in all_prereqs:
                all_prereqs.add(prereq)
                to_process.extend(self.prerequisites[prereq])

        return all_prereqs

    def get_dependents(self, content_id: str) -> Set[str]:
        """Get content that depends on this content as prerequisite."""
        return self.dependents[content_id].copy()

    def validate_prerequisites(
        self,
        content_id: str,
        user_id: str,
        completed_content: Optional[Set[str]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate if a user has met prerequisites for content.

        Args:
            content_id: Content to check
            user_id: User identifier
            completed_content: Set of content IDs the user has completed

        Returns:
            Tuple of (all_met: bool, missing_prerequisites: List[str])
        """
        if completed_content is None:
            completed_content = set()

        # Get user's completed content from library if available
        if self.library and hasattr(self.library, 'get_user_completed_content'):
            completed_content = completed_content.union(
                self.library.get_user_completed_content(user_id)
            )

        prerequisites = self.get_prerequisites(content_id)
        missing = [prereq for prereq in prerequisites if prereq not in completed_content]

        return len(missing) == 0, missing

    # =========================================================================
    # Path and Sequence Operations
    # =========================================================================

    def get_learning_path(
        self,
        target_content_id: str,
        user_id: str,
        completed_content: Optional[Set[str]] = None,
        include_target: bool = True
    ) -> List[str]:
        """
        Get ordered learning path to reach target content.

        Uses topological sort to order prerequisites.

        Args:
            target_content_id: Target content to reach
            user_id: User identifier
            completed_content: Already completed content
            include_target: Whether to include target in path

        Returns:
            Ordered list of content IDs forming the learning path
        """
        if completed_content is None:
            completed_content = set()

        # Get all prerequisites
        all_prereqs = self.get_all_prerequisites(target_content_id)

        # Filter out already completed
        needed = all_prereqs - completed_content

        if include_target and target_content_id not in completed_content:
            needed.add(target_content_id)

        if not needed:
            return []

        # Topological sort of needed content
        return self._topological_sort(needed)

    def _topological_sort(self, content_ids: Set[str]) -> List[str]:
        """Perform topological sort on content IDs based on prerequisites."""
        # Build in-degree map for subset
        in_degree: Dict[str, int] = {cid: 0 for cid in content_ids}
        subset_prereqs: Dict[str, Set[str]] = {}

        for cid in content_ids:
            prereqs_in_subset = self.prerequisites[cid] & content_ids
            subset_prereqs[cid] = prereqs_in_subset
            in_degree[cid] = len(prereqs_in_subset)

        # Kahn's algorithm
        queue = deque([cid for cid, degree in in_degree.items() if degree == 0])
        sorted_list = []

        while queue:
            # Get content with no pending prerequisites
            current = queue.popleft()
            sorted_list.append(current)

            # Reduce in-degree for dependents
            for dependent in self.dependents[current]:
                if dependent in in_degree:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)

        # Check for cycles
        if len(sorted_list) != len(content_ids):
            logger.warning("Cycle detected in prerequisite graph")
            # Return what we have plus remaining (cycle broken arbitrarily)
            remaining = content_ids - set(sorted_list)
            sorted_list.extend(remaining)

        return sorted_list

    def get_dependency_tree(self, content_id: str, max_depth: int = 10) -> Dict[str, Any]:
        """
        Get the dependency tree for content.

        Args:
            content_id: Root content ID
            max_depth: Maximum depth to traverse

        Returns:
            Nested dict representing dependency tree
        """
        def build_tree(cid: str, depth: int) -> Dict[str, Any]:
            if depth <= 0:
                return {"id": cid, "truncated": True}

            prereqs = self.get_prerequisites(cid)
            content = self._get_content(cid)

            return {
                "id": cid,
                "title": content.title if content else cid,
                "difficulty": content.difficulty.value if content else None,
                "prerequisites": [
                    build_tree(prereq, depth - 1)
                    for prereq in prereqs
                ]
            }

        return build_tree(content_id, max_depth)

    def detect_circular_dependencies(self) -> List[List[str]]:
        """
        Detect circular dependencies in the prerequisite graph.

        Returns:
            List of cycles (each cycle is a list of content IDs)
        """
        cycles = []
        visited = set()
        rec_stack = set()

        def dfs(content_id: str, path: List[str]) -> bool:
            visited.add(content_id)
            rec_stack.add(content_id)
            path.append(content_id)

            for prereq in self.prerequisites[content_id]:
                if prereq not in visited:
                    if dfs(prereq, path):
                        return True
                elif prereq in rec_stack:
                    # Found cycle
                    cycle_start = path.index(prereq)
                    cycles.append(path[cycle_start:] + [prereq])
                    return True

            path.pop()
            rec_stack.remove(content_id)
            return False

        for content_id in set(self.prerequisites.keys()) | set(self.dependents.keys()):
            if content_id not in visited:
                dfs(content_id, [])

        return cycles

    # =========================================================================
    # Content Recommendations
    # =========================================================================

    def suggest_next_content(
        self,
        user_id: str,
        current_content_id: str,
        completed_content: Optional[Set[str]] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Suggest next content based on current position and relationships.

        Args:
            user_id: User identifier
            current_content_id: Current content ID
            completed_content: Already completed content
            limit: Maximum suggestions

        Returns:
            List of suggested content with reasoning
        """
        if completed_content is None:
            completed_content = set()

        suggestions = []

        # Get content that has current as prerequisite (enabled content)
        enabled = self.dependents.get(current_content_id, set())
        for content_id in enabled:
            if content_id in completed_content:
                continue

            # Check if all prerequisites are met
            all_met, missing = self.validate_prerequisites(
                content_id, user_id, completed_content | {current_content_id}
            )

            if all_met:
                content = self._get_content(content_id)

                suggestions.append({
                    "content_id": content_id,
                    "title": content.title if content else content_id,
                    "reason": "Prerequisites completed",
                    "relationship": "sequel",
                    "score": 1.0
                })

        # Get related content
        related = self.get_relationships(current_content_id, RelationshipType.RELATED)
        for rel in related:
            if rel.target_id in completed_content:
                continue

            content = self._get_content(rel.target_id)

            suggestions.append({
                "content_id": rel.target_id,
                "title": content.title if content else rel.target_id,
                "reason": "Related topic",
                "relationship": "related",
                "score": 0.7 * rel.weight
            })

        # Sort by score and limit
        suggestions.sort(key=lambda x: x["score"], reverse=True)
        return suggestions[:limit]

    def get_related_content(
        self,
        content_id: str,
        relationship_types: Optional[List[RelationshipType]] = None,
        limit: int = 10
    ) -> List[ContentItem]:
        """
        Get content related to a given content item.

        Args:
            content_id: Source content ID
            relationship_types: Types of relationships to consider
            limit: Maximum results

        Returns:
            List of related ContentItems
        """
        if relationship_types is None:
            relationship_types = [RelationshipType.RELATED, RelationshipType.EXTENSION]

        related_ids = set()
        for rel_type in relationship_types:
            for rel in self.get_relationships(content_id, rel_type, "both"):
                if rel.source_id == content_id:
                    related_ids.add(rel.target_id)
                else:
                    related_ids.add(rel.source_id)

        related_content = []
        for cid in list(related_ids)[:limit]:
            content = self._get_content(cid)
            if content:
                related_content.append(content)

        return related_content

    # =========================================================================
    # Graph Analysis
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        total_content = len(set(self.edges.keys()) | set(self.reverse_edges.keys()))
        total_relationships = sum(len(rels) for rels in self.edges.values())

        relationship_counts = defaultdict(int)
        for relationships in self.edges.values():
            for rel in relationships:
                relationship_counts[rel.relationship_type.value] += 1

        # Find roots (no prerequisites) and leaves (no dependents)
        all_content = set(self.prerequisites.keys()) | set(self.dependents.keys())
        roots = [c for c in all_content if not self.prerequisites[c]]
        leaves = [c for c in all_content if not self.dependents[c]]

        # Calculate average depth
        depths = []
        for content_id in all_content:
            depth = len(self.get_all_prerequisites(content_id))
            depths.append(depth)
        avg_depth = sum(depths) / len(depths) if depths else 0

        return {
            "total_content_in_graph": total_content,
            "total_relationships": total_relationships,
            "relationship_counts": dict(relationship_counts),
            "root_content": len(roots),
            "leaf_content": len(leaves),
            "average_prerequisite_depth": avg_depth,
            "circular_dependencies": len(self.detect_circular_dependencies())
        }

    def export_to_dict(self) -> Dict[str, Any]:
        """Export graph to dictionary for serialization."""
        relationships = []
        for source_id, rels in self.edges.items():
            for rel in rels:
                relationships.append({
                    "source_id": rel.source_id,
                    "target_id": rel.target_id,
                    "relationship_type": rel.relationship_type.value,
                    "weight": rel.weight,
                    "metadata": rel.metadata,
                    "created_at": rel.created_at.isoformat()
                })

        return {
            "relationships": relationships,
            "exported_at": datetime.now(timezone.utc).isoformat()
        }

    def import_from_dict(self, data: Dict[str, Any]):
        """Import graph from dictionary."""
        for rel_data in data.get("relationships", []):
            self.add_relationship(
                source_id=rel_data["source_id"],
                target_id=rel_data["target_id"],
                relationship_type=RelationshipType(rel_data["relationship_type"]),
                weight=rel_data.get("weight", 1.0),
                metadata=rel_data.get("metadata", {})
            )
        logger.info(f"Imported {len(data.get('relationships', []))} relationships")
