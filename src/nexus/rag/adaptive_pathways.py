"""
Advanced Adaptive Learning Pathways for Nexus AI Platform.

Provides sophisticated personalized learning experiences with:
1. Spaced Repetition Scheduling - Optimal review timing based on forgetting curves
2. Mastery-Based Progression - Competency-driven advancement
3. Knowledge Dependency Graphs - Prerequisite-aware sequencing
4. Performance Prediction - ML-based learning outcome forecasting
5. Personalized Content Sequencing - Adaptive content ordering
6. Learning Style Adaptation - Multi-modal content delivery
7. Cognitive Load Management - Optimized information density
"""

import logging
import math
import time
import uuid
import hashlib
import statistics
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from collections import defaultdict
import heapq

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class LearningStyle(Enum):
    """Learning style preferences based on VARK model."""
    VISUAL = "visual"           # Diagrams, charts, videos
    AUDITORY = "auditory"       # Lectures, discussions, podcasts
    KINESTHETIC = "kinesthetic" # Hands-on, simulations, practice
    READING = "reading"         # Text, documentation, articles
    MULTIMODAL = "multimodal"   # Combination of styles


class DifficultyLevel(Enum):
    """Content difficulty levels."""
    NOVICE = "novice"
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

    @property
    def numeric(self) -> int:
        """Get numeric value for calculations."""
        return {
            "novice": 1, "beginner": 2, "intermediate": 3,
            "advanced": 4, "expert": 5
        }[self.value]

    @classmethod
    def from_numeric(cls, value: int) -> 'DifficultyLevel':
        """Create from numeric value."""
        mapping = {1: cls.NOVICE, 2: cls.BEGINNER, 3: cls.INTERMEDIATE,
                   4: cls.ADVANCED, 5: cls.EXPERT}
        return mapping.get(min(max(value, 1), 5), cls.INTERMEDIATE)


class MasteryLevel(Enum):
    """Knowledge mastery levels."""
    NOT_STARTED = "not_started"
    LEARNING = "learning"
    PRACTICING = "practicing"
    PROFICIENT = "proficient"
    MASTERED = "mastered"

    @property
    def threshold(self) -> float:
        """Get mastery score threshold."""
        return {
            "not_started": 0.0, "learning": 0.25, "practicing": 0.5,
            "proficient": 0.75, "mastered": 0.9
        }[self.value]


class ContentType(Enum):
    """Types of learning content."""
    CONCEPT = "concept"
    PROCEDURE = "procedure"
    FACT = "fact"
    PRINCIPLE = "principle"
    EXAMPLE = "example"
    EXERCISE = "exercise"
    ASSESSMENT = "assessment"
    PROJECT = "project"


class LearningPhase(Enum):
    """Phases of learning progression."""
    INTRODUCTION = "introduction"
    EXPLORATION = "exploration"
    PRACTICE = "practice"
    APPLICATION = "application"
    SYNTHESIS = "synthesis"
    MASTERY = "mastery"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SpacedRepetitionItem:
    """Item tracked for spaced repetition."""
    item_id: str
    content_id: str
    ease_factor: float = 2.5          # SM-2 ease factor
    interval_days: float = 1.0        # Current interval
    repetitions: int = 0              # Number of successful reviews
    next_review: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_review: Optional[datetime] = None
    quality_history: List[int] = field(default_factory=list)  # 0-5 ratings

    def calculate_next_interval(self, quality: int) -> float:
        """
        Calculate next review interval using SM-2 algorithm.

        Args:
            quality: Response quality (0-5)
                0 - Complete blackout
                1 - Incorrect, remembered upon seeing answer
                2 - Incorrect, easy to recall after seeing answer
                3 - Correct with serious difficulty
                4 - Correct with some hesitation
                5 - Perfect response

        Returns:
            New interval in days
        """
        self.quality_history.append(quality)

        if quality < 3:
            # Failed review - reset
            self.repetitions = 0
            self.interval_days = 1.0
        else:
            # Successful review
            if self.repetitions == 0:
                self.interval_days = 1.0
            elif self.repetitions == 1:
                self.interval_days = 6.0
            else:
                self.interval_days *= self.ease_factor

            self.repetitions += 1

            # Update ease factor
            self.ease_factor = max(1.3, self.ease_factor + 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))

        self.last_review = datetime.now(timezone.utc)
        self.next_review = self.last_review + timedelta(days=self.interval_days)

        return self.interval_days


@dataclass
class KnowledgeNode:
    """Node in knowledge dependency graph."""
    node_id: str
    topic: str
    difficulty: DifficultyLevel
    content_type: ContentType
    prerequisites: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)
    estimated_time_minutes: int = 30
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearnerProfile:
    """Comprehensive learner profile with learning analytics."""
    user_id: str
    learning_style: LearningStyle = LearningStyle.MULTIMODAL
    preferred_difficulty: DifficultyLevel = DifficultyLevel.INTERMEDIATE
    learning_pace: float = 1.0  # Multiplier (0.5 = slow, 2.0 = fast)
    interests: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    knowledge_gaps: List[str] = field(default_factory=list)

    # Learning history
    topics_mastered: Set[str] = field(default_factory=set)
    topics_in_progress: Set[str] = field(default_factory=set)
    mastery_scores: Dict[str, float] = field(default_factory=dict)

    # Performance metrics
    avg_session_duration: float = 30.0  # minutes
    preferred_session_time: str = "morning"  # morning, afternoon, evening
    total_learning_time: float = 0.0  # minutes
    sessions_completed: int = 0
    assessments_taken: int = 0
    avg_assessment_score: float = 0.0

    # Cognitive metrics
    attention_span: float = 25.0  # minutes before break recommended
    cognitive_load_tolerance: float = 0.7  # 0-1 scale
    retention_rate: float = 0.8  # Estimated retention

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_active: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def update_mastery(self, topic: str, score: float):
        """Update mastery score for a topic."""
        current = self.mastery_scores.get(topic, 0.0)
        # Weighted moving average
        self.mastery_scores[topic] = 0.7 * score + 0.3 * current

        if self.mastery_scores[topic] >= MasteryLevel.MASTERED.threshold:
            self.topics_mastered.add(topic)
            self.topics_in_progress.discard(topic)
        elif self.mastery_scores[topic] > MasteryLevel.NOT_STARTED.threshold:
            self.topics_in_progress.add(topic)


@dataclass
class LearningContent:
    """Learning content item with metadata."""
    content_id: str
    title: str
    content: str
    content_type: ContentType
    difficulty: DifficultyLevel
    topics: List[str]
    prerequisites: List[str] = field(default_factory=list)
    estimated_time_minutes: int = 15
    learning_styles: List[LearningStyle] = field(default_factory=list)
    format: str = "text"  # text, video, interactive, etc.
    source: str = "knowledge_base"
    quality_score: float = 0.8
    engagement_score: float = 0.7
    effectiveness_score: float = 0.75


@dataclass
class LearningModule:
    """A module in a learning pathway."""
    module_id: str
    title: str
    description: str
    phase: LearningPhase
    difficulty: DifficultyLevel
    topics: List[str]
    content_items: List[str] = field(default_factory=list)  # Content IDs
    estimated_hours: float = 2.0
    objectives: List[str] = field(default_factory=list)
    assessment_criteria: Dict[str, Any] = field(default_factory=dict)
    prerequisites: List[str] = field(default_factory=list)  # Module IDs


@dataclass
class LearningPathway:
    """Complete learning pathway with modules and progression."""
    pathway_id: str
    user_id: str
    goal: str
    modules: List[LearningModule]
    duration_weeks: int
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = "active"
    current_module_index: int = 0
    completion_percentage: float = 0.0
    adaptive_adjustments: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class StudySession:
    """A study session with items to review/learn."""
    session_id: str
    user_id: str
    items: List[Dict[str, Any]]  # Content items with metadata
    scheduled_duration: int  # minutes
    actual_duration: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    performance_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformancePrediction:
    """Prediction of learning performance."""
    topic: str
    predicted_score: float
    confidence: float
    factors: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


# =============================================================================
# Main Class
# =============================================================================

class AdaptiveLearningPathways:
    """
    Advanced adaptive learning system with multiple intelligent algorithms.

    Features:
    - Spaced Repetition: SM-2 algorithm for optimal review scheduling
    - Mastery Learning: Competency-based progression
    - Knowledge Graphs: Prerequisite-aware content sequencing
    - Performance Prediction: Forecasting learning outcomes
    - Cognitive Load Management: Optimized session planning
    """

    def __init__(self, knowledge_base=None, pattern_engine=None):
        """
        Initialize adaptive learning pathways.

        Args:
            knowledge_base: Knowledge base for content retrieval
            pattern_engine: Pattern recognition engine
        """
        self.knowledge_base = knowledge_base
        self.pattern_engine = pattern_engine

        # User data
        self.user_profiles: Dict[str, LearnerProfile] = {}
        self.learning_pathways: Dict[str, LearningPathway] = {}
        self.study_sessions: Dict[str, StudySession] = {}

        # Spaced repetition
        self.sr_items: Dict[str, Dict[str, SpacedRepetitionItem]] = {}  # user_id -> item_id -> item

        # Knowledge graph
        self.knowledge_graph: Dict[str, KnowledgeNode] = {}
        self.topic_prerequisites: Dict[str, Set[str]] = defaultdict(set)

        # Content library
        self.content_library: Dict[str, LearningContent] = {}

        # Performance tracking
        self.performance_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # Configuration
        self.config = {
            "min_mastery_to_advance": 0.75,
            "max_items_per_session": 20,
            "new_items_per_session": 5,
            "review_items_per_session": 15,
            "default_session_duration": 25,  # minutes (Pomodoro)
            "cognitive_load_limit": 0.8,
            "forgetting_curve_factor": 0.1,
        }

        # Analytics
        self.analytics = {
            "total_users": 0,
            "total_pathways": 0,
            "total_sessions": 0,
            "avg_completion_rate": 0.0,
            "avg_mastery_score": 0.0,
        }

        self.initialized = False
        logger.info("AdaptiveLearningPathways created")

    def initialize(self):
        """Initialize the learning pathways system."""
        if self.initialized:
            return

        logger.info("Initializing Adaptive Learning Pathways...")

        # Build initial knowledge graph from knowledge base
        if self.knowledge_base:
            self._build_knowledge_graph_from_kb()

        # Load default content library
        self._initialize_content_library()

        self.initialized = True
        logger.info("Adaptive Learning Pathways initialized")

    # =========================================================================
    # User Profile Management
    # =========================================================================

    def create_user_profile(
        self,
        user_id: str,
        preferences: Dict[str, Any]
    ) -> LearnerProfile:
        """
        Create or update a learner profile.

        Args:
            user_id: Unique user identifier
            preferences: User preferences including:
                - learning_style: visual, auditory, kinesthetic, reading
                - difficulty: novice, beginner, intermediate, advanced, expert
                - interests: List of interest topics
                - learning_pace: 0.5 (slow) to 2.0 (fast)

        Returns:
            Created or updated LearnerProfile
        """
        # Parse learning style
        style_str = preferences.get("learning_style", "multimodal").lower()
        try:
            learning_style = LearningStyle(style_str)
        except ValueError:
            learning_style = LearningStyle.MULTIMODAL

        # Parse difficulty
        diff_str = preferences.get("difficulty", "intermediate").lower()
        try:
            difficulty = DifficultyLevel(diff_str)
        except ValueError:
            difficulty = DifficultyLevel.INTERMEDIATE

        # Get existing profile or create new
        existing = self.user_profiles.get(user_id)

        profile = LearnerProfile(
            user_id=user_id,
            learning_style=learning_style,
            preferred_difficulty=difficulty,
            learning_pace=preferences.get("learning_pace", 1.0),
            interests=preferences.get("interests", []),
            strengths=existing.strengths if existing else [],
            knowledge_gaps=self._assess_knowledge_gaps(
                preferences.get("interests", []),
                preferences.get("domain", "general")
            ),
            topics_mastered=existing.topics_mastered if existing else set(),
            topics_in_progress=existing.topics_in_progress if existing else set(),
            mastery_scores=existing.mastery_scores if existing else {},
        )

        # Copy over history if updating
        if existing:
            profile.total_learning_time = existing.total_learning_time
            profile.sessions_completed = existing.sessions_completed
            profile.assessments_taken = existing.assessments_taken
            profile.avg_assessment_score = existing.avg_assessment_score

        self.user_profiles[user_id] = profile
        self.analytics["total_users"] = len(self.user_profiles)

        logger.info(f"Created/updated profile for user: {user_id}")
        return profile

    def get_user_profile(self, user_id: str) -> Optional[LearnerProfile]:
        """Get user profile by ID."""
        return self.user_profiles.get(user_id)

    def update_user_mastery(
        self,
        user_id: str,
        topic: str,
        score: float,
        assessment_type: str = "practice"
    ):
        """
        Update user's mastery of a topic based on performance.

        Args:
            user_id: User identifier
            topic: Topic that was assessed
            score: Performance score (0-1)
            assessment_type: Type of assessment (practice, quiz, project)
        """
        profile = self.user_profiles.get(user_id)
        if not profile:
            return

        # Weight by assessment type
        weights = {
            "practice": 0.3,
            "quiz": 0.5,
            "project": 0.7,
            "exam": 1.0
        }
        weight = weights.get(assessment_type, 0.5)

        # Update mastery with weighted score
        weighted_score = score * weight + (1 - weight) * profile.mastery_scores.get(topic, 0)
        profile.update_mastery(topic, weighted_score)

        # Record performance
        self.performance_history[user_id].append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "topic": topic,
            "score": score,
            "assessment_type": assessment_type,
            "mastery_after": profile.mastery_scores.get(topic, 0)
        })

        # Update analytics
        all_scores = [p.avg_assessment_score for p in self.user_profiles.values() if p.assessments_taken > 0]
        if all_scores:
            self.analytics["avg_mastery_score"] = statistics.mean(all_scores)

    # =========================================================================
    # Spaced Repetition System
    # =========================================================================

    def add_to_spaced_repetition(
        self,
        user_id: str,
        content_id: str,
        initial_ease: float = 2.5
    ) -> SpacedRepetitionItem:
        """
        Add an item to user's spaced repetition queue.

        Args:
            user_id: User identifier
            content_id: Content item ID
            initial_ease: Initial ease factor (default 2.5)

        Returns:
            Created SpacedRepetitionItem
        """
        if user_id not in self.sr_items:
            self.sr_items[user_id] = {}

        item_id = f"sr_{user_id}_{content_id}"

        item = SpacedRepetitionItem(
            item_id=item_id,
            content_id=content_id,
            ease_factor=initial_ease
        )

        self.sr_items[user_id][item_id] = item
        return item

    def record_review(
        self,
        user_id: str,
        item_id: str,
        quality: int
    ) -> Optional[float]:
        """
        Record a spaced repetition review result.

        Args:
            user_id: User identifier
            item_id: SR item ID
            quality: Review quality (0-5)

        Returns:
            New interval in days, or None if item not found
        """
        if user_id not in self.sr_items:
            return None

        item = self.sr_items[user_id].get(item_id)
        if not item:
            return None

        new_interval = item.calculate_next_interval(quality)

        logger.debug(f"SR review recorded: user={user_id}, item={item_id}, "
                    f"quality={quality}, new_interval={new_interval:.1f} days")

        return new_interval

    def get_due_reviews(
        self,
        user_id: str,
        limit: int = 20
    ) -> List[SpacedRepetitionItem]:
        """
        Get items due for review.

        Args:
            user_id: User identifier
            limit: Maximum items to return

        Returns:
            List of items due for review, sorted by urgency
        """
        if user_id not in self.sr_items:
            return []

        now = datetime.now(timezone.utc)
        due_items = []

        for item in self.sr_items[user_id].values():
            if item.next_review <= now:
                # Calculate overdue factor for priority
                overdue_days = (now - item.next_review).total_seconds() / 86400
                priority = overdue_days / max(item.interval_days, 0.1)
                due_items.append((priority, item))

        # Sort by priority (most overdue first)
        due_items.sort(key=lambda x: x[0], reverse=True)

        return [item for _, item in due_items[:limit]]

    def get_optimal_review_schedule(
        self,
        user_id: str,
        days_ahead: int = 7
    ) -> Dict[str, List[str]]:
        """
        Get optimal review schedule for upcoming days.

        Args:
            user_id: User identifier
            days_ahead: Days to plan ahead

        Returns:
            Dict mapping date strings to lists of item IDs
        """
        if user_id not in self.sr_items:
            return {}

        schedule = defaultdict(list)
        now = datetime.now(timezone.utc)

        for item in self.sr_items[user_id].values():
            days_until = (item.next_review - now).days
            if 0 <= days_until <= days_ahead:
                date_str = item.next_review.strftime("%Y-%m-%d")
                schedule[date_str].append(item.item_id)

        return dict(schedule)

    # =========================================================================
    # Knowledge Graph & Dependencies
    # =========================================================================

    def add_knowledge_node(self, node: KnowledgeNode):
        """Add a node to the knowledge graph."""
        self.knowledge_graph[node.node_id] = node

        # Update prerequisites mapping
        for prereq in node.prerequisites:
            self.topic_prerequisites[node.topic].add(prereq)

        # Update dependents of prerequisites
        for prereq_id in node.prerequisites:
            if prereq_id in self.knowledge_graph:
                self.knowledge_graph[prereq_id].dependents.append(node.node_id)

    def get_learning_sequence(
        self,
        target_topics: List[str],
        user_id: str
    ) -> List[KnowledgeNode]:
        """
        Get optimal learning sequence for target topics using topological sort.

        Considers:
        - Prerequisites (must come first)
        - User's existing mastery (skip mastered topics)
        - Difficulty progression

        Args:
            target_topics: Topics user wants to learn
            user_id: User identifier

        Returns:
            Ordered list of KnowledgeNodes
        """
        profile = self.user_profiles.get(user_id)
        mastered = profile.topics_mastered if profile else set()

        # Find all required nodes including prerequisites
        required_nodes = set()
        to_process = list(target_topics)

        while to_process:
            topic = to_process.pop()
            if topic in required_nodes or topic in mastered:
                continue

            required_nodes.add(topic)

            # Add prerequisites
            for node_id, node in self.knowledge_graph.items():
                if node.topic == topic:
                    to_process.extend(node.prerequisites)

        # Topological sort
        sorted_nodes = []
        visited = set()
        temp_visited = set()

        def visit(node_id: str):
            if node_id in temp_visited:
                raise ValueError(f"Circular dependency detected: {node_id}")
            if node_id in visited:
                return

            node = self.knowledge_graph.get(node_id)
            if not node or node.topic in mastered:
                return

            temp_visited.add(node_id)

            for prereq_id in node.prerequisites:
                visit(prereq_id)

            temp_visited.remove(node_id)
            visited.add(node_id)
            sorted_nodes.append(node)

        for topic in required_nodes:
            for node_id, node in self.knowledge_graph.items():
                if node.topic == topic and node_id not in visited:
                    visit(node_id)

        # Sort by difficulty within the topological order
        return sorted(sorted_nodes, key=lambda n: (
            len([p for p in n.prerequisites if p not in mastered]),
            n.difficulty.numeric
        ))

    def check_prerequisites_met(
        self,
        topic: str,
        user_id: str
    ) -> Tuple[bool, List[str]]:
        """
        Check if user has met prerequisites for a topic.

        Args:
            topic: Topic to check
            user_id: User identifier

        Returns:
            Tuple of (all_met: bool, missing_prerequisites: List[str])
        """
        profile = self.user_profiles.get(user_id)
        if not profile:
            return False, []

        prerequisites = self.topic_prerequisites.get(topic, set())
        missing = []

        for prereq in prerequisites:
            mastery = profile.mastery_scores.get(prereq, 0)
            if mastery < self.config["min_mastery_to_advance"]:
                missing.append(prereq)

        return len(missing) == 0, missing

    # =========================================================================
    # Learning Pathway Generation
    # =========================================================================

    def generate_learning_pathway(
        self,
        user_id: str,
        learning_goal: str,
        duration_weeks: int = 4,
        target_topics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate a personalized learning pathway.

        Args:
            user_id: User identifier
            learning_goal: Description of learning goal
            duration_weeks: Target duration in weeks
            target_topics: Specific topics to cover (optional)

        Returns:
            Dict containing pathway details
        """
        if not self.initialized:
            self.initialize()

        profile = self.user_profiles.get(user_id)
        if not profile:
            profile = self.create_user_profile(user_id, {})

        pathway_id = str(uuid.uuid4())

        # Analyze goal to determine topics if not specified
        if not target_topics:
            target_topics = self._extract_topics_from_goal(learning_goal)

        # Get optimal learning sequence
        learning_sequence = self.get_learning_sequence(target_topics, user_id)

        # Generate modules based on sequence and user profile
        modules = self._generate_pathway_modules(
            learning_sequence,
            profile,
            duration_weeks,
            learning_goal
        )

        pathway = LearningPathway(
            pathway_id=pathway_id,
            user_id=user_id,
            goal=learning_goal,
            modules=modules,
            duration_weeks=duration_weeks
        )

        self.learning_pathways[pathway_id] = pathway
        self.analytics["total_pathways"] = len(self.learning_pathways)

        logger.info(f"Generated pathway {pathway_id} for user {user_id}: "
                   f"{len(modules)} modules over {duration_weeks} weeks")

        return self._pathway_to_dict(pathway)

    def _generate_pathway_modules(
        self,
        learning_sequence: List[KnowledgeNode],
        profile: LearnerProfile,
        duration_weeks: int,
        goal: str
    ) -> List[LearningModule]:
        """Generate modules from learning sequence."""
        modules = []

        # Calculate items per week
        total_items = len(learning_sequence)
        items_per_week = max(1, total_items // duration_weeks)

        # Define phase progression
        phases = [
            LearningPhase.INTRODUCTION,
            LearningPhase.EXPLORATION,
            LearningPhase.PRACTICE,
            LearningPhase.APPLICATION,
            LearningPhase.SYNTHESIS,
            LearningPhase.MASTERY
        ]

        week = 1
        current_items = []

        for i, node in enumerate(learning_sequence):
            current_items.append(node)

            # Create module when we have enough items or at the end
            if len(current_items) >= items_per_week or i == len(learning_sequence) - 1:
                phase_index = min(int(week / duration_weeks * len(phases)), len(phases) - 1)
                phase = phases[phase_index]

                # Calculate module difficulty
                avg_difficulty = statistics.mean([n.difficulty.numeric for n in current_items])
                module_difficulty = DifficultyLevel.from_numeric(int(avg_difficulty))

                # Calculate estimated time
                total_minutes = sum(n.estimated_time_minutes for n in current_items)
                estimated_hours = (total_minutes / 60) * profile.learning_pace

                module = LearningModule(
                    module_id=f"mod_{week}_{uuid.uuid4().hex[:8]}",
                    title=f"Week {week}: {phase.value.title()}",
                    description=f"Focus: {', '.join(n.topic for n in current_items[:3])}",
                    phase=phase,
                    difficulty=module_difficulty,
                    topics=[n.topic for n in current_items],
                    content_items=[n.node_id for n in current_items],
                    estimated_hours=estimated_hours,
                    objectives=self._generate_objectives(current_items, phase),
                    prerequisites=[m.module_id for m in modules[-1:]] if modules else []
                )

                modules.append(module)
                current_items = []
                week += 1

        return modules

    def _generate_objectives(
        self,
        nodes: List[KnowledgeNode],
        phase: LearningPhase
    ) -> List[str]:
        """Generate learning objectives for module."""
        objectives = []

        phase_verbs = {
            LearningPhase.INTRODUCTION: ["Understand", "Identify", "Recognize"],
            LearningPhase.EXPLORATION: ["Explore", "Discover", "Compare"],
            LearningPhase.PRACTICE: ["Apply", "Practice", "Demonstrate"],
            LearningPhase.APPLICATION: ["Implement", "Create", "Build"],
            LearningPhase.SYNTHESIS: ["Integrate", "Analyze", "Evaluate"],
            LearningPhase.MASTERY: ["Master", "Teach", "Innovate"]
        }

        verbs = phase_verbs.get(phase, ["Learn"])

        for node in nodes[:3]:
            verb = verbs[len(objectives) % len(verbs)]
            objectives.append(f"{verb} {node.topic}")

        return objectives

    # =========================================================================
    # Study Session Management
    # =========================================================================

    def generate_study_session(
        self,
        user_id: str,
        duration_minutes: int = 25,
        focus_topics: Optional[List[str]] = None
    ) -> StudySession:
        """
        Generate an optimized study session.

        Balances:
        - New content introduction
        - Spaced repetition reviews
        - Cognitive load management

        Args:
            user_id: User identifier
            duration_minutes: Target session duration
            focus_topics: Optional specific topics to focus on

        Returns:
            StudySession with optimized content
        """
        profile = self.user_profiles.get(user_id)
        if not profile:
            profile = self.create_user_profile(user_id, {})

        session_id = str(uuid.uuid4())
        items = []

        # Get due reviews (prioritize these)
        due_reviews = self.get_due_reviews(user_id, self.config["review_items_per_session"])
        review_content_ids = {item.content_id for item in due_reviews}

        # Add review items
        for sr_item in due_reviews:
            content = self.content_library.get(sr_item.content_id)
            if content:
                items.append({
                    "type": "review",
                    "content_id": sr_item.content_id,
                    "sr_item_id": sr_item.item_id,
                    "title": content.title,
                    "estimated_minutes": max(2, content.estimated_time_minutes // 2),
                    "priority": 1
                })

        # Calculate remaining time for new content
        review_time = sum(item["estimated_minutes"] for item in items)
        remaining_time = duration_minutes - review_time

        # Add new content
        if remaining_time > 5:
            new_content = self._select_new_content(
                profile,
                focus_topics,
                remaining_time,
                exclude_ids=review_content_ids
            )

            for content in new_content:
                items.append({
                    "type": "new",
                    "content_id": content.content_id,
                    "title": content.title,
                    "estimated_minutes": content.estimated_time_minutes,
                    "difficulty": content.difficulty.value,
                    "priority": 2
                })

                # Add to spaced repetition
                self.add_to_spaced_repetition(user_id, content.content_id)

        # Sort by priority (reviews first)
        items.sort(key=lambda x: x["priority"])

        # Apply cognitive load ordering
        items = self._optimize_cognitive_load(items, profile)

        session = StudySession(
            session_id=session_id,
            user_id=user_id,
            items=items,
            scheduled_duration=duration_minutes
        )

        self.study_sessions[session_id] = session
        self.analytics["total_sessions"] = len(self.study_sessions)

        return session

    def _select_new_content(
        self,
        profile: LearnerProfile,
        focus_topics: Optional[List[str]],
        time_budget: int,
        exclude_ids: Set[str]
    ) -> List[LearningContent]:
        """Select new content optimized for user."""
        candidates = []

        for content_id, content in self.content_library.items():
            if content_id in exclude_ids:
                continue

            # Check prerequisites
            prereqs_met, _ = self.check_prerequisites_met(
                content.topics[0] if content.topics else "",
                profile.user_id
            )
            if not prereqs_met:
                continue

            # Calculate relevance score
            score = self._calculate_content_score(content, profile, focus_topics)
            candidates.append((score, content))

        # Sort by score
        candidates.sort(key=lambda x: x[0], reverse=True)

        # Select within time budget
        selected = []
        total_time = 0

        for score, content in candidates:
            if total_time + content.estimated_time_minutes <= time_budget:
                selected.append(content)
                total_time += content.estimated_time_minutes

                if len(selected) >= self.config["new_items_per_session"]:
                    break

        return selected

    def _calculate_content_score(
        self,
        content: LearningContent,
        profile: LearnerProfile,
        focus_topics: Optional[List[str]]
    ) -> float:
        """Calculate personalized content relevance score."""
        score = 0.0

        # Topic relevance
        if focus_topics:
            topic_match = sum(1 for t in content.topics if t in focus_topics)
            score += topic_match * 0.3

        # Interest alignment
        interest_match = sum(1 for t in content.topics if t in profile.interests)
        score += interest_match * 0.2

        # Knowledge gap filling
        gap_match = sum(1 for t in content.topics if t in profile.knowledge_gaps)
        score += gap_match * 0.25

        # Learning style match
        if profile.learning_style in content.learning_styles:
            score += 0.15

        # Difficulty appropriateness
        diff_delta = abs(content.difficulty.numeric - profile.preferred_difficulty.numeric)
        score += max(0, 0.1 - diff_delta * 0.02)

        return score

    def _optimize_cognitive_load(
        self,
        items: List[Dict[str, Any]],
        profile: LearnerProfile
    ) -> List[Dict[str, Any]]:
        """Optimize item order for cognitive load management."""
        # Group by difficulty
        easy = [i for i in items if i.get("difficulty", "intermediate") in ["novice", "beginner"]]
        medium = [i for i in items if i.get("difficulty", "intermediate") == "intermediate"]
        hard = [i for i in items if i.get("difficulty", "intermediate") in ["advanced", "expert"]]
        reviews = [i for i in items if i["type"] == "review"]

        # Interleave: Start with reviews, then easy, then interleave medium/hard
        optimized = []

        # Warm up with reviews
        optimized.extend(reviews[:3])

        # Easy content
        optimized.extend(easy)

        # Interleave medium and hard
        while medium or hard:
            if medium:
                optimized.append(medium.pop(0))
            if hard:
                optimized.append(hard.pop(0))

        # Remaining reviews at end
        optimized.extend(reviews[3:])

        return optimized

    def complete_study_session(
        self,
        session_id: str,
        performance_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Record completion of a study session.

        Args:
            session_id: Session identifier
            performance_data: Performance metrics including:
                - item_scores: Dict mapping content_id to score (0-5)
                - actual_duration: Actual time spent
                - engagement_rating: Self-reported engagement (1-5)

        Returns:
            Session summary with recommendations
        """
        session = self.study_sessions.get(session_id)
        if not session:
            return {"error": "Session not found"}

        profile = self.user_profiles.get(session.user_id)

        # Update session
        session.completed_at = datetime.now(timezone.utc)
        session.actual_duration = performance_data.get("actual_duration", session.scheduled_duration)
        session.performance_data = performance_data

        # Process item scores
        item_scores = performance_data.get("item_scores", {})
        for item in session.items:
            content_id = item["content_id"]
            score = item_scores.get(content_id, 3)  # Default to "okay"

            # Update spaced repetition if applicable
            if item["type"] == "review" and "sr_item_id" in item:
                self.record_review(session.user_id, item["sr_item_id"], score)

            # Update mastery
            content = self.content_library.get(content_id)
            if content and content.topics:
                normalized_score = score / 5.0
                self.update_user_mastery(
                    session.user_id,
                    content.topics[0],
                    normalized_score,
                    "practice"
                )

        # Update profile metrics
        if profile:
            profile.sessions_completed += 1
            profile.total_learning_time += session.actual_duration
            profile.last_active = datetime.now(timezone.utc)

            # Update average session duration
            profile.avg_session_duration = (
                profile.avg_session_duration * 0.8 +
                session.actual_duration * 0.2
            )

        # Generate summary
        avg_score = statistics.mean(item_scores.values()) if item_scores else 0
        return {
            "session_id": session_id,
            "completed": True,
            "items_completed": len(session.items),
            "avg_score": avg_score,
            "duration_minutes": session.actual_duration,
            "recommendations": self._generate_session_recommendations(session, avg_score)
        }

    def _generate_session_recommendations(
        self,
        session: StudySession,
        avg_score: float
    ) -> List[str]:
        """Generate recommendations based on session performance."""
        recommendations = []

        if avg_score >= 4:
            recommendations.append("Excellent session! Consider increasing difficulty.")
        elif avg_score >= 3:
            recommendations.append("Good progress. Continue with current pace.")
        elif avg_score >= 2:
            recommendations.append("Some areas need review. Focus on weak points.")
        else:
            recommendations.append("Consider reviewing prerequisite material.")

        # Time-based recommendations
        if session.actual_duration > session.scheduled_duration * 1.5:
            recommendations.append("Consider shorter, more frequent sessions.")
        elif session.actual_duration < session.scheduled_duration * 0.5:
            recommendations.append("Great focus! Try longer sessions if comfortable.")

        return recommendations

    # =========================================================================
    # Performance Prediction
    # =========================================================================

    def predict_performance(
        self,
        user_id: str,
        topic: str
    ) -> PerformancePrediction:
        """
        Predict user's performance on a topic.

        Uses historical data and learning patterns to forecast outcomes.

        Args:
            user_id: User identifier
            topic: Topic to predict performance for

        Returns:
            PerformancePrediction with score and confidence
        """
        profile = self.user_profiles.get(user_id)
        if not profile:
            return PerformancePrediction(
                topic=topic,
                predicted_score=0.5,
                confidence=0.1,
                recommendations=["Create a profile for personalized predictions"]
            )

        history = self.performance_history.get(user_id, [])
        factors = {}

        # Factor 1: Current mastery
        current_mastery = profile.mastery_scores.get(topic, 0)
        factors["current_mastery"] = current_mastery

        # Factor 2: Prerequisite strength
        prereqs = self.topic_prerequisites.get(topic, set())
        if prereqs:
            prereq_scores = [profile.mastery_scores.get(p, 0) for p in prereqs]
            factors["prerequisite_strength"] = statistics.mean(prereq_scores)
        else:
            factors["prerequisite_strength"] = 0.7

        # Factor 3: Learning velocity (recent improvement rate)
        topic_history = [h for h in history if h.get("topic") == topic]
        if len(topic_history) >= 2:
            recent_scores = [h["score"] for h in topic_history[-5:]]
            if len(recent_scores) >= 2:
                improvement = recent_scores[-1] - recent_scores[0]
                factors["learning_velocity"] = 0.5 + improvement
            else:
                factors["learning_velocity"] = 0.5
        else:
            factors["learning_velocity"] = 0.5

        # Factor 4: Retention estimate
        if topic in self.sr_items.get(user_id, {}):
            sr_item = list(self.sr_items[user_id].values())[0]  # Get relevant item
            days_since_review = (datetime.now(timezone.utc) - (sr_item.last_review or sr_item.next_review)).days
            # Forgetting curve approximation
            retention = math.exp(-self.config["forgetting_curve_factor"] * days_since_review)
            factors["retention_estimate"] = retention
        else:
            factors["retention_estimate"] = 0.5

        # Combine factors
        weights = {
            "current_mastery": 0.35,
            "prerequisite_strength": 0.25,
            "learning_velocity": 0.2,
            "retention_estimate": 0.2
        }

        predicted_score = sum(factors[k] * weights[k] for k in factors)
        predicted_score = max(0.0, min(1.0, predicted_score))

        # Calculate confidence based on data availability
        confidence = min(0.9, len(topic_history) * 0.15 + 0.2)

        # Generate recommendations
        recommendations = []
        if factors["prerequisite_strength"] < 0.6:
            recommendations.append("Review prerequisite topics first")
        if factors["retention_estimate"] < 0.5:
            recommendations.append("Schedule a review session")
        if factors["learning_velocity"] < 0.4:
            recommendations.append("Try different learning approaches")

        return PerformancePrediction(
            topic=topic,
            predicted_score=predicted_score,
            confidence=confidence,
            factors=factors,
            recommendations=recommendations
        )

    # =========================================================================
    # Pathway Adaptation
    # =========================================================================

    def adapt_pathway_based_on_progress(
        self,
        pathway_id: str,
        progress_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Adapt a learning pathway based on user progress.

        Args:
            pathway_id: Pathway identifier
            progress_data: Progress metrics including:
                - completion_rate: 0-1 completion percentage
                - assessment_scores: Dict of topic -> score
                - difficulty_rating: 1-5 self-reported difficulty
                - time_spent: Minutes spent learning

        Returns:
            Adaptation summary with changes made
        """
        pathway = self.learning_pathways.get(pathway_id)
        if not pathway:
            return {"error": "Pathway not found"}

        profile = self.user_profiles.get(pathway.user_id)
        if not profile:
            return {"error": "User profile not found"}

        adaptations = []

        # Analyze progress
        completion_rate = progress_data.get("completion_rate", 0.5)
        assessment_scores = progress_data.get("assessment_scores", {})
        difficulty_rating = progress_data.get("difficulty_rating", 3)

        # Update mastery scores
        for topic, score in assessment_scores.items():
            self.update_user_mastery(pathway.user_id, topic, score, "assessment")

        # Adaptation 1: Pace adjustment
        if completion_rate > 0.9 and difficulty_rating < 3:
            profile.learning_pace = min(2.0, profile.learning_pace * 1.2)
            adaptations.append({
                "type": "pace_increase",
                "reason": "High completion with low difficulty",
                "new_pace": profile.learning_pace
            })
        elif completion_rate < 0.5 or difficulty_rating > 4:
            profile.learning_pace = max(0.5, profile.learning_pace * 0.8)
            adaptations.append({
                "type": "pace_decrease",
                "reason": "Low completion or high difficulty",
                "new_pace": profile.learning_pace
            })

        # Adaptation 2: Difficulty adjustment
        avg_score = statistics.mean(assessment_scores.values()) if assessment_scores else 0.5
        if avg_score > 0.85:
            self._increase_pathway_difficulty(pathway)
            adaptations.append({
                "type": "difficulty_increase",
                "reason": "High assessment scores"
            })
        elif avg_score < 0.5:
            self._decrease_pathway_difficulty(pathway)
            adaptations.append({
                "type": "difficulty_decrease",
                "reason": "Low assessment scores"
            })

        # Adaptation 3: Content focus
        weak_topics = [t for t, s in assessment_scores.items() if s < 0.6]
        if weak_topics:
            self._add_remedial_content(pathway, weak_topics)
            adaptations.append({
                "type": "remedial_content",
                "topics": weak_topics
            })

        # Record adaptation
        pathway.adaptive_adjustments.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "adaptations": adaptations,
            "progress_data": progress_data
        })

        return {
            "pathway_id": pathway_id,
            "adaptations_made": len(adaptations),
            "adaptations": adaptations,
            "updated_pace": profile.learning_pace,
            "recommendations": self._generate_pathway_recommendations(pathway, progress_data)
        }

    def _increase_pathway_difficulty(self, pathway: LearningPathway):
        """Increase difficulty of remaining modules."""
        for module in pathway.modules[pathway.current_module_index:]:
            current = module.difficulty.numeric
            module.difficulty = DifficultyLevel.from_numeric(min(5, current + 1))

    def _decrease_pathway_difficulty(self, pathway: LearningPathway):
        """Decrease difficulty of remaining modules."""
        for module in pathway.modules[pathway.current_module_index:]:
            current = module.difficulty.numeric
            module.difficulty = DifficultyLevel.from_numeric(max(1, current - 1))

    def _add_remedial_content(self, pathway: LearningPathway, topics: List[str]):
        """Add remedial content for weak topics."""
        # Find relevant content
        remedial_items = []
        for topic in topics:
            for content_id, content in self.content_library.items():
                if topic in content.topics and content.difficulty.numeric <= 2:
                    remedial_items.append(content_id)

        # Insert remedial module if needed
        if remedial_items and pathway.current_module_index < len(pathway.modules):
            current_module = pathway.modules[pathway.current_module_index]
            current_module.content_items = remedial_items[:5] + current_module.content_items

    def _generate_pathway_recommendations(
        self,
        pathway: LearningPathway,
        progress_data: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations for pathway progress."""
        recommendations = []

        completion = progress_data.get("completion_rate", 0)
        if completion < 0.3:
            recommendations.append("Focus on completing current module before moving on")
        elif completion > 0.8:
            recommendations.append("Great progress! Consider attempting advanced content")

        time_spent = progress_data.get("time_spent", 0)
        expected_time = sum(m.estimated_hours * 60 for m in pathway.modules[:pathway.current_module_index + 1])
        if time_spent < expected_time * 0.5:
            recommendations.append("Consider spending more time with the material")

        return recommendations

    # =========================================================================
    # Content Recommendations
    # =========================================================================

    def get_personalized_content_recommendations(
        self,
        user_id: str,
        topic: str = "general",
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get personalized content recommendations.

        Args:
            user_id: User identifier
            topic: Topic focus (or "general" for broad recommendations)
            limit: Maximum recommendations to return

        Returns:
            List of recommended content with personalization details
        """
        profile = self.user_profiles.get(user_id)
        if not profile:
            profile = self.create_user_profile(user_id, {})

        recommendations = []

        # Query knowledge base if available
        if self.knowledge_base and topic != "general":
            try:
                kb_results = self.knowledge_base.query_knowledge(topic, max_results=20)
                for item in kb_results:
                    score = self._calculate_kb_content_score(item, profile, topic)
                    recommendations.append({
                        "content": str(item.content),
                        "type": item.knowledge_type.value if hasattr(item.knowledge_type, 'value') else str(item.knowledge_type),
                        "relevance_score": score,
                        "source": "knowledge_base",
                        "personalization_reason": self._explain_recommendation(item, profile),
                        "estimated_time": self._estimate_learning_time(item, profile),
                        "difficulty_level": self._assess_content_difficulty(item),
                        "learning_style_match": self._assess_learning_style_match(item, profile)
                    })
            except Exception as e:
                logger.warning(f"Knowledge base query failed: {e}")

        # Add from content library
        for content_id, content in self.content_library.items():
            if topic == "general" or topic in content.topics:
                score = self._calculate_content_score(content, profile, [topic] if topic != "general" else None)
                recommendations.append({
                    "content_id": content_id,
                    "title": content.title,
                    "content": content.content[:200] + "..." if len(content.content) > 200 else content.content,
                    "type": content.content_type.value,
                    "relevance_score": score,
                    "source": content.source,
                    "estimated_time": content.estimated_time_minutes,
                    "difficulty_level": content.difficulty.value,
                    "learning_style_match": 1.0 if profile.learning_style in content.learning_styles else 0.5
                })

        # Sort by relevance and return top N
        recommendations.sort(key=lambda x: x["relevance_score"], reverse=True)
        return recommendations[:limit]

    def _calculate_kb_content_score(self, item, profile: LearnerProfile, topic: str) -> float:
        """Calculate relevance score for knowledge base item."""
        score = 0.0
        content_str = str(item.content).lower()

        # Topic match
        if topic.lower() in content_str:
            score += 0.4

        # Interest alignment
        for interest in profile.interests:
            if interest.lower() in content_str:
                score += 0.15

        # Gap filling
        for gap in profile.knowledge_gaps:
            if gap.lower() in content_str:
                score += 0.2

        # Confidence bonus
        if hasattr(item, 'confidence'):
            score += item.confidence * 0.1

        return min(1.0, score)

    def _explain_recommendation(self, item, profile: LearnerProfile) -> str:
        """Explain why content is recommended."""
        content_str = str(item.content).lower()
        reasons = []

        for interest in profile.interests:
            if interest.lower() in content_str:
                reasons.append(f"matches interest: {interest}")

        for gap in profile.knowledge_gaps:
            if gap.lower() in content_str:
                reasons.append(f"addresses gap: {gap}")

        return "; ".join(reasons) if reasons else "general relevance"

    def _estimate_learning_time(self, item, profile: LearnerProfile) -> int:
        """Estimate learning time in minutes."""
        # Base estimate on content length
        content_length = len(str(item.content))
        base_time = max(5, content_length // 100)

        # Adjust for learning pace
        adjusted_time = int(base_time / profile.learning_pace)

        return max(5, min(60, adjusted_time))

    def _assess_content_difficulty(self, item) -> str:
        """Assess content difficulty level."""
        content_str = str(item.content).lower()

        advanced_indicators = ["advanced", "complex", "sophisticated", "expert"]
        beginner_indicators = ["basic", "introduction", "simple", "fundamental"]

        if any(ind in content_str for ind in advanced_indicators):
            return "advanced"
        elif any(ind in content_str for ind in beginner_indicators):
            return "beginner"
        return "intermediate"

    def _assess_learning_style_match(self, item, profile: LearnerProfile) -> float:
        """Assess how well content matches learning style."""
        content_str = str(item.content).lower()

        style_indicators = {
            LearningStyle.VISUAL: ["diagram", "chart", "image", "visual", "graph"],
            LearningStyle.AUDITORY: ["audio", "podcast", "discussion", "lecture"],
            LearningStyle.KINESTHETIC: ["exercise", "practice", "hands-on", "interactive"],
            LearningStyle.READING: ["read", "text", "article", "documentation"]
        }

        indicators = style_indicators.get(profile.learning_style, [])
        matches = sum(1 for ind in indicators if ind in content_str)

        return min(1.0, 0.5 + matches * 0.2)

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _assess_knowledge_gaps(self, interests: List[str], domain: str) -> List[str]:
        """Assess knowledge gaps based on interests and domain."""
        gaps = []

        # Domain-specific common gaps
        domain_gaps = {
            "technology": ["algorithms", "data structures", "system design", "security"],
            "science": ["scientific method", "data analysis", "statistics", "research"],
            "business": ["financial analysis", "strategy", "management", "marketing"],
            "general": ["critical thinking", "problem solving", "communication"]
        }

        gaps.extend(domain_gaps.get(domain, domain_gaps["general"]))

        # Check knowledge base coverage for interests
        if self.knowledge_base:
            for interest in interests:
                try:
                    results = self.knowledge_base.query_knowledge(interest, max_results=5)
                    if len(results) < 3:
                        gaps.append(f"{interest} fundamentals")
                except Exception:
                    pass

        return list(set(gaps))

    def _extract_topics_from_goal(self, goal: str) -> List[str]:
        """Extract topics from learning goal description."""
        # Simple keyword extraction
        topics = []
        goal_lower = goal.lower()

        # Check against known topics in knowledge graph
        for node in self.knowledge_graph.values():
            if node.topic.lower() in goal_lower:
                topics.append(node.topic)

        # If no matches, use words from goal
        if not topics:
            words = goal.split()
            topics = [w for w in words if len(w) > 4 and w.isalpha()][:5]

        return topics or ["general"]

    def _build_knowledge_graph_from_kb(self):
        """Build knowledge graph from knowledge base content."""
        if not self.knowledge_base:
            return

        try:
            # Get all knowledge items
            if hasattr(self.knowledge_base, 'knowledge_items'):
                for item_id, item in self.knowledge_base.knowledge_items.items():
                    node = KnowledgeNode(
                        node_id=item_id,
                        topic=str(item.content)[:50],
                        difficulty=DifficultyLevel.INTERMEDIATE,
                        content_type=ContentType.CONCEPT,
                        tags=item.context_tags if hasattr(item, 'context_tags') and item.context_tags else []
                    )
                    self.knowledge_graph[item_id] = node
        except Exception as e:
            logger.warning(f"Failed to build knowledge graph from KB: {e}")

    def _initialize_content_library(self):
        """Initialize default content library."""
        # Add sample content for testing
        sample_topics = [
            ("programming_basics", "Programming Fundamentals", ContentType.CONCEPT, DifficultyLevel.BEGINNER),
            ("data_structures", "Data Structures", ContentType.CONCEPT, DifficultyLevel.INTERMEDIATE),
            ("algorithms", "Algorithm Design", ContentType.PROCEDURE, DifficultyLevel.INTERMEDIATE),
            ("system_design", "System Design Principles", ContentType.PRINCIPLE, DifficultyLevel.ADVANCED),
        ]

        for topic_id, title, content_type, difficulty in sample_topics:
            self.content_library[topic_id] = LearningContent(
                content_id=topic_id,
                title=title,
                content=f"Learn about {title}",
                content_type=content_type,
                difficulty=difficulty,
                topics=[topic_id],
                learning_styles=[LearningStyle.READING, LearningStyle.VISUAL]
            )

    def _pathway_to_dict(self, pathway: LearningPathway) -> Dict[str, Any]:
        """Convert pathway to dictionary format."""
        return {
            "pathway_id": pathway.pathway_id,
            "user_id": pathway.user_id,
            "goal": pathway.goal,
            "duration_weeks": pathway.duration_weeks,
            "status": pathway.status,
            "completion_percentage": pathway.completion_percentage,
            "current_module_index": pathway.current_module_index,
            "weekly_modules": [
                {
                    "module_id": m.module_id,
                    "week": i + 1,
                    "title": m.title,
                    "description": m.description,
                    "phase": m.phase.value,
                    "difficulty": m.difficulty.value,
                    "topics": m.topics,
                    "estimated_hours": m.estimated_hours,
                    "objectives": m.objectives
                }
                for i, m in enumerate(pathway.modules)
            ],
            "created_at": pathway.created_at.isoformat()
        }

    # =========================================================================
    # Analytics & Statistics
    # =========================================================================

    def get_user_analytics(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive analytics for a user."""
        profile = self.user_profiles.get(user_id)
        if not profile:
            return {"error": "User not found"}

        # Calculate learning velocity
        history = self.performance_history.get(user_id, [])
        velocity = 0.0
        if len(history) >= 2:
            recent_scores = [h["score"] for h in history[-10:]]
            velocity = (recent_scores[-1] - recent_scores[0]) / len(recent_scores)

        # Calculate retention estimate
        sr_items = self.sr_items.get(user_id, {})
        avg_ease = statistics.mean([i.ease_factor for i in sr_items.values()]) if sr_items else 2.5

        return {
            "user_id": user_id,
            "profile_summary": {
                "learning_style": profile.learning_style.value,
                "difficulty_level": profile.preferred_difficulty.value,
                "learning_pace": profile.learning_pace,
                "topics_mastered": len(profile.topics_mastered),
                "topics_in_progress": len(profile.topics_in_progress)
            },
            "performance_metrics": {
                "total_learning_time_hours": profile.total_learning_time / 60,
                "sessions_completed": profile.sessions_completed,
                "avg_assessment_score": profile.avg_assessment_score,
                "learning_velocity": velocity
            },
            "spaced_repetition": {
                "items_tracked": len(sr_items),
                "avg_ease_factor": avg_ease,
                "due_reviews": len(self.get_due_reviews(user_id, 100))
            },
            "recommendations": self._generate_user_recommendations(profile)
        }

    def _generate_user_recommendations(self, profile: LearnerProfile) -> List[str]:
        """Generate personalized recommendations for user."""
        recommendations = []

        if profile.sessions_completed < 5:
            recommendations.append("Complete more study sessions to unlock personalized insights")

        if profile.learning_pace > 1.5:
            recommendations.append("Consider challenging yourself with advanced content")
        elif profile.learning_pace < 0.7:
            recommendations.append("Focus on mastering fundamentals before advancing")

        if profile.knowledge_gaps:
            recommendations.append(f"Priority topics to address: {', '.join(profile.knowledge_gaps[:3])}")

        return recommendations

    def get_system_statistics(self) -> Dict[str, Any]:
        """Get overall system statistics."""
        return {
            **self.analytics,
            "content_library_size": len(self.content_library),
            "knowledge_graph_nodes": len(self.knowledge_graph),
            "active_pathways": len([p for p in self.learning_pathways.values() if p.status == "active"])
        }
