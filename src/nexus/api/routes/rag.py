"""
RAG API Routes

Provides RESTful endpoints for Nexus's RAG system.
"""

from flask import Blueprint, request, jsonify
from typing import Dict, Any, Optional
import logging

from nexus.rag import (
    RAGVectorEngine,
    AdaptiveRAGOrchestrator,
    ContextWindowManager,
    AdaptivePathways,
)

logger = logging.getLogger(__name__)

rag_bp = Blueprint('rag', __name__, url_prefix='/api/v1/rag')

# Global RAG instances (initialized by app)
rag_engine: Optional[RAGVectorEngine] = None
rag_orchestrator: Optional[AdaptiveRAGOrchestrator] = None
context_manager: Optional[ContextWindowManager] = None
pathways: Optional[AdaptivePathways] = None


def initialize_rag_system(config: Dict[str, Any]):
    """Initialize RAG system components"""
    global rag_engine, rag_orchestrator, context_manager, pathways

    logger.info("Initializing Nexus RAG system...")

    # Initialize components with configuration
    context_window_size = config.get('rag', {}).get('context_window', 150_000_000)
    vector_store_type = config.get('rag', {}).get('vector_store', 'faiss')

    # Initialize RAG engine
    rag_engine = RAGVectorEngine(
        context_window=context_window_size,
        vector_store=vector_store_type
    )

    # Initialize context manager
    context_manager = ContextWindowManager(max_tokens=context_window_size)

    # Initialize adaptive orchestrator
    rag_orchestrator = AdaptiveRAGOrchestrator()

    # Initialize learning pathways
    pathways = AdaptivePathways()

    logger.info(f"RAG system initialized with {context_window_size} token context")


# ===== Query Endpoints =====

@rag_bp.route('/query', methods=['POST'])
def rag_query():
    """Query using RAG with context retrieval"""
    try:
        data = request.json

        query = data.get('query')
        if not query:
            return jsonify({"status": "error", "message": "Query is required"}), 400

        # Retrieve context
        context = rag_engine.retrieve_context(
            query=query,
            top_k=data.get('top_k', 10),
            min_similarity=data.get('min_similarity', 0.7)
        )

        # Optional: Use adaptive orchestration
        if data.get('use_adaptive', False):
            result = rag_orchestrator.query_with_adaptation(
                query=query,
                context=context,
                strategy=data.get('strategy', 'auto')
            )
        else:
            result = {
                "query": query,
                "context": context,
                "retrieved_count": len(context) if isinstance(context, list) else 1
            }

        return jsonify({
            "status": "success",
            "result": result
        })

    except Exception as e:
        logger.error(f"Error in RAG query: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@rag_bp.route('/embed', methods=['POST'])
def embed_text():
    """Generate embeddings for text"""
    try:
        data = request.json

        text = data.get('text')
        if not text:
            return jsonify({"status": "error", "message": "Text is required"}), 400

        embedding = rag_engine.embed(text)

        return jsonify({
            "status": "success",
            "embedding": embedding.tolist() if hasattr(embedding, 'tolist') else embedding,
            "dimension": len(embedding) if hasattr(embedding, '__len__') else None
        })

    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@rag_bp.route('/add_documents', methods=['POST'])
def add_documents():
    """Add documents to RAG vector store"""
    try:
        data = request.json

        documents = data.get('documents', [])
        if not documents:
            return jsonify({"status": "error", "message": "Documents are required"}), 400

        # Add documents to vector store
        doc_ids = rag_engine.add_documents(
            documents=documents,
            metadata=data.get('metadata', [])
        )

        return jsonify({
            "status": "success",
            "message": f"Added {len(doc_ids)} documents",
            "document_ids": doc_ids
        }), 201

    except Exception as e:
        logger.error(f"Error adding documents: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# ===== Context Management Endpoints =====

@rag_bp.route('/context/manage', methods=['POST'])
def manage_context():
    """Manage context window"""
    try:
        data = request.json

        context = data.get('context')
        operation = data.get('operation', 'optimize')

        if operation == 'optimize':
            optimized = context_manager.optimize_context(context)
            result = {
                "operation": "optimize",
                "optimized_context": optimized,
                "token_count": context_manager.count_tokens(optimized)
            }
        elif operation == 'compress':
            compressed = context_manager.compress_context(
                context,
                target_size=data.get('target_size')
            )
            result = {
                "operation": "compress",
                "compressed_context": compressed,
                "original_tokens": context_manager.count_tokens(context),
                "compressed_tokens": context_manager.count_tokens(compressed)
            }
        elif operation == 'split':
            chunks = context_manager.split_context(
                context,
                chunk_size=data.get('chunk_size', 4000)
            )
            result = {
                "operation": "split",
                "chunks": chunks,
                "chunk_count": len(chunks)
            }
        else:
            return jsonify({"status": "error", "message": f"Unknown operation: {operation}"}), 400

        return jsonify({
            "status": "success",
            "result": result
        })

    except Exception as e:
        logger.error(f"Error managing context: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@rag_bp.route('/context/window', methods=['GET'])
def get_context_window_info():
    """Get context window information"""
    try:
        info = {
            "max_tokens": context_manager.max_tokens if context_manager else 150_000_000,
            "current_usage": context_manager.get_current_usage() if context_manager else 0,
            "utilization": context_manager.get_utilization() if context_manager else 0.0
        }

        return jsonify({
            "status": "success",
            "context_window": info
        })

    except Exception as e:
        logger.error(f"Error getting context window info: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# ===== Adaptive RAG Endpoints =====

@rag_bp.route('/adaptive/query', methods=['POST'])
def adaptive_query():
    """Query using adaptive RAG orchestration"""
    try:
        data = request.json

        query = data.get('query')
        if not query:
            return jsonify({"status": "error", "message": "Query is required"}), 400

        result = rag_orchestrator.adaptive_query(
            query=query,
            context_limit=data.get('context_limit'),
            strategy=data.get('strategy', 'auto'),
            confidence_threshold=data.get('confidence_threshold', 0.7)
        )

        return jsonify({
            "status": "success",
            "result": result
        })

    except Exception as e:
        logger.error(f"Error in adaptive query: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@rag_bp.route('/adaptive/strategies', methods=['GET'])
def get_adaptive_strategies():
    """Get available adaptive RAG strategies"""
    try:
        strategies = rag_orchestrator.get_available_strategies() if rag_orchestrator else []

        return jsonify({
            "status": "success",
            "strategies": strategies
        })

    except Exception as e:
        logger.error(f"Error getting strategies: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# ===== Learning Pathways Endpoints =====

@rag_bp.route('/pathways/optimize', methods=['POST'])
def optimize_pathway():
    """Optimize learning pathway"""
    try:
        data = request.json

        topic = data.get('topic')
        if not topic:
            return jsonify({"status": "error", "message": "Topic is required"}), 400

        pathway = pathways.optimize_pathway(
            topic=topic,
            depth=data.get('depth', 3),
            breadth=data.get('breadth', 5)
        )

        return jsonify({
            "status": "success",
            "pathway": pathway
        })

    except Exception as e:
        logger.error(f"Error optimizing pathway: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@rag_bp.route('/pathways', methods=['GET'])
def get_pathways():
    """Get existing learning pathways"""
    try:
        topic = request.args.get('topic')

        pathway_list = pathways.get_pathways(topic=topic) if pathways else []

        return jsonify({
            "status": "success",
            "pathways": pathway_list,
            "count": len(pathway_list)
        })

    except Exception as e:
        logger.error(f"Error getting pathways: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# ===== Search Endpoints =====

@rag_bp.route('/search', methods=['POST'])
def semantic_search():
    """Semantic search using vector similarity"""
    try:
        data = request.json

        query = data.get('query')
        if not query:
            return jsonify({"status": "error", "message": "Query is required"}), 400

        results = rag_engine.semantic_search(
            query=query,
            top_k=data.get('top_k', 10),
            filters=data.get('filters', {}),
            min_similarity=data.get('min_similarity', 0.7)
        )

        return jsonify({
            "status": "success",
            "results": results,
            "count": len(results)
        })

    except Exception as e:
        logger.error(f"Error in semantic search: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# ===== Analytics Endpoints =====

@rag_bp.route('/analytics', methods=['GET'])
def get_rag_analytics():
    """Get RAG system analytics"""
    try:
        analytics = {
            "total_documents": rag_engine.get_document_count() if rag_engine else 0,
            "total_embeddings": rag_engine.get_embedding_count() if rag_engine else 0,
            "context_window": {
                "max_tokens": context_manager.max_tokens if context_manager else 150_000_000,
                "average_usage": context_manager.get_average_usage() if context_manager else 0
            },
            "queries_processed": rag_orchestrator.get_query_count() if rag_orchestrator else 0,
            "active_pathways": len(pathways.get_pathways()) if pathways else 0
        }

        return jsonify({
            "status": "success",
            "analytics": analytics
        })

    except Exception as e:
        logger.error(f"Error getting RAG analytics: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# ===== Health Check =====

@rag_bp.route('/health', methods=['GET'])
def rag_health():
    """Check RAG system health"""
    try:
        health = {
            "status": "healthy",
            "components": {
                "rag_engine": rag_engine is not None,
                "rag_orchestrator": rag_orchestrator is not None,
                "context_manager": context_manager is not None,
                "pathways": pathways is not None,
            }
        }

        all_healthy = all(health["components"].values())
        if not all_healthy:
            health["status"] = "degraded"

        return jsonify(health)

    except Exception as e:
        logger.error(f"Error checking RAG health: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500
