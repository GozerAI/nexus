"""
Data API Routes

Provides RESTful endpoints for Nexus's data ingestion system.
"""

from flask import Blueprint, request, jsonify, current_app
from typing import Dict, Any, Optional
from urllib.parse import urlparse
import ipaddress
import logging
import socket

from nexus.licensing import license_gate
from nexus.data import (
    DataIngestion,
    AutoDataProcessor,
    InternetRetriever,
    HuggingFaceLoader,
)

logger = logging.getLogger(__name__)

_GATE = "nxs.discovery.intelligence"

data_bp = Blueprint('data', __name__, url_prefix='/api/v1/data')


@data_bp.before_request
def _require_auth():
    """Require API key for all non-health endpoints."""
    if request.endpoint and request.endpoint.endswith('.data_health'):
        return None
    auth = getattr(current_app, 'auth_middleware', None)
    if auth is None:
        return None
    api_key = request.headers.get("X-API-Key") or request.headers.get("Authorization")
    if not api_key:
        return jsonify({"error": "Missing API key", "message": "Provide API key in X-API-Key or Authorization header"}), 401
    if api_key.startswith("Bearer "):
        api_key = api_key[7:]
    api_key_obj = auth.api_key_manager.validate_key(api_key)
    if not api_key_obj:
        return jsonify({"error": "Invalid API key", "message": "The provided API key is invalid or expired"}), 401
    return None


def _validate_url(url: str) -> bool:
    """Block private/internal URLs to prevent SSRF."""
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        if not parsed.hostname:
            return False
        ip = socket.gethostbyname(parsed.hostname)
        addr = ipaddress.ip_address(ip)
        if addr.is_private or addr.is_loopback or addr.is_link_local:
            return False
        return True
    except Exception:
        return False


# Global data instances (initialized by app)
data_ingestion: Optional[DataIngestion] = None
auto_processor: Optional[AutoDataProcessor] = None
internet_retriever: Optional[InternetRetriever] = None
huggingface_loader: Optional[HuggingFaceLoader] = None


def initialize_data_system(config: Dict[str, Any]):
    """Initialize data system components"""
    global data_ingestion, auto_processor, internet_retriever, huggingface_loader

    logger.info("Initializing Nexus data system...")

    data_ingestion = DataIngestion()
    auto_processor = AutoDataProcessor()
    internet_retriever = InternetRetriever()
    huggingface_loader = HuggingFaceLoader()

    logger.info("Data system initialized successfully")


# ===== Data Ingestion Endpoints =====

@data_bp.route('/ingest', methods=['POST'])
def ingest_data():
    """Ingest data from various formats"""
    try:
        license_gate.gate(_GATE)
        data = request.json

        source = data.get('source')
        if not source:
            return jsonify({"status": "error", "message": "Source is required"}), 400

        result = data_ingestion.ingest(
            source=source,
            format=data.get('format', 'auto'),
            options=data.get('options', {})
        )

        return jsonify({
            "status": "success",
            "result": result
        }), 201

    except PermissionError as e:
        return jsonify({"status": "error", "message": str(e)}), 403
    except Exception as e:
        logger.error(f"Error ingesting data: {e}")
        return jsonify({"status": "error", "message": "An internal error occurred"}), 500


# ===== Auto Processing Endpoints =====

@data_bp.route('/process', methods=['POST'])
def process_data():
    """Automatically process ingested data"""
    try:
        license_gate.gate(_GATE)
        data = request.json

        input_data = data.get('data')
        if not input_data:
            return jsonify({"status": "error", "message": "Data is required"}), 400

        result = auto_processor.process(
            data=input_data,
            pipeline=data.get('pipeline', 'auto')
        )

        return jsonify({
            "status": "success",
            "result": result
        })

    except PermissionError as e:
        return jsonify({"status": "error", "message": str(e)}), 403
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        return jsonify({"status": "error", "message": "An internal error occurred"}), 500


# ===== Internet Retrieval Endpoints =====

@data_bp.route('/retrieve', methods=['POST'])
def retrieve_from_internet():
    """Retrieve data from the internet"""
    try:
        license_gate.gate(_GATE)
        data = request.json

        url = data.get('url')
        query = data.get('query')

        if not url and not query:
            return jsonify({"status": "error", "message": "URL or query is required"}), 400

        if url:
            if not _validate_url(url):
                return jsonify({"status": "error", "message": "URL is not allowed (private/internal addresses are blocked)"}), 400
            result = internet_retriever.retrieve_url(url)
        else:
            result = internet_retriever.search(query, limit=data.get('limit', 10))

        return jsonify({
            "status": "success",
            "result": result
        })

    except PermissionError as e:
        return jsonify({"status": "error", "message": str(e)}), 403
    except Exception as e:
        logger.error(f"Error retrieving data: {e}")
        return jsonify({"status": "error", "message": "An internal error occurred"}), 500


# ===== HuggingFace Integration Endpoints =====

@data_bp.route('/huggingface/load', methods=['POST'])
def load_huggingface_dataset():
    """Load dataset from HuggingFace"""
    try:
        license_gate.gate(_GATE)
        data = request.json

        dataset_name = data.get('dataset')
        if not dataset_name:
            return jsonify({"status": "error", "message": "Dataset name is required"}), 400

        result = huggingface_loader.load(
            dataset=dataset_name,
            split=data.get('split', 'train'),
            subset=data.get('subset')
        )

        return jsonify({
            "status": "success",
            "result": result
        })

    except PermissionError as e:
        return jsonify({"status": "error", "message": str(e)}), 403
    except Exception as e:
        logger.error(f"Error loading HuggingFace dataset: {e}")
        return jsonify({"status": "error", "message": "An internal error occurred"}), 500


# ===== Health Check =====

@data_bp.route('/health', methods=['GET'])
def data_health():
    """Check data system health"""
    try:
        health = {
            "status": "healthy",
            "components": {
                "data_ingestion": data_ingestion is not None,
                "auto_processor": auto_processor is not None,
                "internet_retriever": internet_retriever is not None,
                "huggingface_loader": huggingface_loader is not None,
            }
        }

        all_healthy = all(health["components"].values())
        if not all_healthy:
            health["status"] = "degraded"

        return jsonify(health)

    except Exception as e:
        logger.error(f"Error checking data health: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500
