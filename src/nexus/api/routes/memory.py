"""
Memory API Routes

Provides RESTful endpoints for Nexus's memory system.
"""

from flask import Blueprint, request, jsonify
from typing import Dict, Any, Optional
import logging

from nexus.memory import (
    KnowledgeBase,
    FactualMemoryEngine,
    SkillMemoryEngine,
    PatternRecognitionEngine,
    MemoryBlockManager,
    KnowledgeValidator,
    KnowledgeGapTracker,
    KnowledgeExpander,
    MemoryAnalytics,
    KnowledgeType,
    KnowledgeConfidence,
)

logger = logging.getLogger(__name__)

memory_bp = Blueprint('memory', __name__, url_prefix='/api/v1/memory')

# Global memory instances (initialized by app)
knowledge_base: Optional[KnowledgeBase] = None
factual_memory: Optional[FactualMemoryEngine] = None
skill_memory: Optional[SkillMemoryEngine] = None
pattern_engine: Optional[PatternRecognitionEngine] = None
memory_manager: Optional[MemoryBlockManager] = None
knowledge_validator: Optional[KnowledgeValidator] = None
gap_tracker: Optional[KnowledgeGapTracker] = None
knowledge_expander: Optional[KnowledgeExpander] = None
memory_analytics: Optional[MemoryAnalytics] = None


def initialize_memory_system(config: Dict[str, Any]):
    """Initialize memory system components"""
    global knowledge_base, factual_memory, skill_memory, pattern_engine
    global memory_manager, knowledge_validator, gap_tracker, knowledge_expander
    global memory_analytics

    logger.info("Initializing Nexus memory system...")

    # Initialize core components
    memory_manager = MemoryBlockManager()
    factual_memory = FactualMemoryEngine()
    skill_memory = SkillMemoryEngine()
    pattern_engine = PatternRecognitionEngine()

    # Initialize knowledge base with components
    knowledge_base = KnowledgeBase(
        memory_manager=memory_manager,
        factual_memory=factual_memory,
        skill_memory=skill_memory
    )

    # Initialize advanced features
    knowledge_validator = KnowledgeValidator()
    gap_tracker = KnowledgeGapTracker()
    knowledge_expander = KnowledgeExpander()
    memory_analytics = MemoryAnalytics()

    logger.info("Memory system initialized successfully")


# ===== Knowledge Base Endpoints =====

@memory_bp.route('/knowledge', methods=['POST'])
def add_knowledge():
    """Add knowledge to the knowledge base"""
    try:
        data = request.json

        result = knowledge_base.add_knowledge(
            content=data['content'],
            knowledge_type=KnowledgeType(data.get('knowledge_type', 'factual')),
            confidence=data.get('confidence', 0.8),
            source=data.get('source', 'user'),
            context_tags=data.get('context_tags', [])
        )

        return jsonify({
            "status": "success",
            "knowledge_id": result.get('id'),
            "message": "Knowledge added successfully"
        }), 201

    except Exception as e:
        logger.error(f"Error adding knowledge: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@memory_bp.route('/knowledge/<knowledge_id>', methods=['GET'])
def get_knowledge(knowledge_id: str):
    """Retrieve knowledge by ID"""
    try:
        knowledge_item = knowledge_base.get_knowledge(knowledge_id)

        if not knowledge_item:
            return jsonify({"status": "error", "message": "Knowledge not found"}), 404

        return jsonify({
            "status": "success",
            "knowledge": {
                "id": knowledge_item.id,
                "content": knowledge_item.content,
                "type": knowledge_item.knowledge_type.value,
                "confidence": knowledge_item.confidence,
                "source": knowledge_item.source,
                "created_at": knowledge_item.created_at.isoformat(),
                "access_count": knowledge_item.access_count,
                "verification_status": knowledge_item.verification_status
            }
        })

    except Exception as e:
        logger.error(f"Error retrieving knowledge: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@memory_bp.route('/knowledge/search', methods=['POST'])
def search_knowledge():
    """Search knowledge base"""
    try:
        data = request.json
        query = data.get('query')
        knowledge_type = data.get('knowledge_type')
        min_confidence = data.get('min_confidence', 0.0)
        limit = data.get('limit', 10)

        results = knowledge_base.search(
            query=query,
            knowledge_type=KnowledgeType(knowledge_type) if knowledge_type else None,
            min_confidence=min_confidence,
            limit=limit
        )

        return jsonify({
            "status": "success",
            "results": [
                {
                    "id": item.id,
                    "content": item.content,
                    "type": item.knowledge_type.value,
                    "confidence": item.confidence,
                    "relevance_score": item.get('relevance_score', 1.0)
                }
                for item in results
            ],
            "count": len(results)
        })

    except Exception as e:
        logger.error(f"Error searching knowledge: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# ===== Factual Memory Endpoints =====

@memory_bp.route('/facts', methods=['POST'])
def add_fact():
    """Add a fact to factual memory"""
    try:
        data = request.json

        fact_id = factual_memory.add_fact(
            fact=data['fact'],
            source=data.get('source', 'user'),
            confidence=data.get('confidence', 0.8),
            provenance=data.get('provenance', {}),
            domain=data.get('domain')
        )

        return jsonify({
            "status": "success",
            "fact_id": fact_id,
            "message": "Fact added successfully"
        }), 201

    except Exception as e:
        logger.error(f"Error adding fact: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@memory_bp.route('/facts', methods=['GET'])
def get_facts():
    """Retrieve facts (optionally filtered by domain)"""
    try:
        domain = request.args.get('domain')
        min_confidence = float(request.args.get('min_confidence', 0.0))
        limit = int(request.args.get('limit', 50))

        facts = factual_memory.get_facts(
            domain=domain,
            min_confidence=min_confidence,
            limit=limit
        )

        return jsonify({
            "status": "success",
            "facts": facts,
            "count": len(facts)
        })

    except Exception as e:
        logger.error(f"Error retrieving facts: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# ===== Skill Memory Endpoints =====

@memory_bp.route('/skills', methods=['POST'])
def add_skill():
    """Add a skill to skill memory"""
    try:
        data = request.json

        skill_id = skill_memory.add_skill(
            skill_name=data['skill_name'],
            description=data.get('description', ''),
            procedure=data.get('procedure', []),
            prerequisites=data.get('prerequisites', []),
            proficiency=data.get('proficiency', 0.5)
        )

        return jsonify({
            "status": "success",
            "skill_id": skill_id,
            "message": "Skill added successfully"
        }), 201

    except Exception as e:
        logger.error(f"Error adding skill: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@memory_bp.route('/skills', methods=['GET'])
def get_skills():
    """Retrieve skills"""
    try:
        category = request.args.get('category')
        min_proficiency = float(request.args.get('min_proficiency', 0.0))

        skills = skill_memory.get_skills(
            category=category,
            min_proficiency=min_proficiency
        )

        return jsonify({
            "status": "success",
            "skills": skills,
            "count": len(skills)
        })

    except Exception as e:
        logger.error(f"Error retrieving skills: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# ===== Pattern Recognition Endpoints =====

@memory_bp.route('/patterns/detect', methods=['POST'])
def detect_patterns():
    """Detect patterns in data"""
    try:
        data = request.json

        patterns = pattern_engine.detect_patterns(
            data=data['data'],
            pattern_type=data.get('pattern_type'),
            threshold=data.get('threshold', 0.7)
        )

        return jsonify({
            "status": "success",
            "patterns": patterns,
            "count": len(patterns)
        })

    except Exception as e:
        logger.error(f"Error detecting patterns: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@memory_bp.route('/patterns', methods=['GET'])
def get_patterns():
    """Retrieve discovered patterns"""
    try:
        pattern_type = request.args.get('pattern_type')
        min_confidence = float(request.args.get('min_confidence', 0.0))

        patterns = pattern_engine.get_patterns(
            pattern_type=pattern_type,
            min_confidence=min_confidence
        )

        return jsonify({
            "status": "success",
            "patterns": patterns,
            "count": len(patterns)
        })

    except Exception as e:
        logger.error(f"Error retrieving patterns: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# ===== Knowledge Validation Endpoints =====

@memory_bp.route('/validate', methods=['POST'])
def validate_knowledge():
    """Validate knowledge against sources"""
    try:
        data = request.json

        validation_result = knowledge_validator.validate(
            knowledge=data['knowledge'],
            sources=data.get('sources', []),
            min_consensus=data.get('min_consensus', 0.7)
        )

        return jsonify({
            "status": "success",
            "validation": validation_result
        })

    except Exception as e:
        logger.error(f"Error validating knowledge: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# ===== Knowledge Gap Endpoints =====

@memory_bp.route('/gaps', methods=['GET'])
def get_knowledge_gaps():
    """Identify knowledge gaps"""
    try:
        domain = request.args.get('domain')

        gaps = gap_tracker.identify_gaps(domain=domain)

        return jsonify({
            "status": "success",
            "gaps": gaps,
            "count": len(gaps)
        })

    except Exception as e:
        logger.error(f"Error identifying gaps: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@memory_bp.route('/gaps/curriculum', methods=['POST'])
def generate_curriculum():
    """Generate learning curriculum for gaps"""
    try:
        data = request.json

        curriculum = gap_tracker.generate_curriculum(
            gaps=data.get('gaps', []),
            priority=data.get('priority', 'high')
        )

        return jsonify({
            "status": "success",
            "curriculum": curriculum
        })

    except Exception as e:
        logger.error(f"Error generating curriculum: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# ===== Knowledge Expansion Endpoints =====

@memory_bp.route('/expand', methods=['POST'])
def expand_knowledge():
    """Expand knowledge based on existing knowledge"""
    try:
        data = request.json

        expanded = knowledge_expander.expand(
            knowledge_id=data.get('knowledge_id'),
            topic=data.get('topic'),
            depth=data.get('depth', 1)
        )

        return jsonify({
            "status": "success",
            "expanded_knowledge": expanded,
            "count": len(expanded) if isinstance(expanded, list) else 1
        })

    except Exception as e:
        logger.error(f"Error expanding knowledge: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# ===== Analytics Endpoints =====

@memory_bp.route('/analytics', methods=['GET'])
def get_memory_analytics():
    """Get memory system analytics"""
    try:
        analytics = memory_analytics.get_analytics()

        return jsonify({
            "status": "success",
            "analytics": analytics
        })

    except Exception as e:
        logger.error(f"Error retrieving analytics: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@memory_bp.route('/analytics/usage', methods=['GET'])
def get_usage_analytics():
    """Get memory usage analytics"""
    try:
        time_period = request.args.get('period', '24h')

        usage = memory_analytics.get_usage(time_period=time_period)

        return jsonify({
            "status": "success",
            "usage": usage
        })

    except Exception as e:
        logger.error(f"Error retrieving usage analytics: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# ===== Health Check =====

@memory_bp.route('/health', methods=['GET'])
def memory_health():
    """Check memory system health"""
    try:
        health = {
            "status": "healthy",
            "components": {
                "knowledge_base": knowledge_base is not None,
                "factual_memory": factual_memory is not None,
                "skill_memory": skill_memory is not None,
                "pattern_engine": pattern_engine is not None,
                "memory_manager": memory_manager is not None,
                "knowledge_validator": knowledge_validator is not None,
                "gap_tracker": gap_tracker is not None,
                "knowledge_expander": knowledge_expander is not None,
                "memory_analytics": memory_analytics is not None,
            }
        }

        all_healthy = all(health["components"].values())
        if not all_healthy:
            health["status"] = "degraded"

        return jsonify(health)

    except Exception as e:
        logger.error(f"Error checking memory health: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500
