"""
Nexus Data Ingestion & Processing System

Multi-format data ingestion and automated processing pipeline.

This system provides:
- Multi-format data ingestion (text, structured, code, documents, web)
- Automated data processing pipeline
- Internet and web data retrieval
- HuggingFace dataset integration
"""

from .data_ingestion import DataIngestionProcessor
from .auto_data_processor import AutoDataProcessor
from .internet_retriever import InternetKnowledgeRetriever
from .huggingface_loader import HuggingFaceLoader
from .wikipedia_collector import WikipediaCollector
from .pypi_collector import PyPIKnowledgeCollector
from .workflow_collector import WorkflowKnowledgeCollector
from .kh_graph_collector import KHGraphCollector
from .trend_signal_collector import TrendSignalCollector
from .arclane_collector import ArclaneCollector

# Aliases for backward compatibility
DataIngestion = DataIngestionProcessor
InternetRetriever = InternetKnowledgeRetriever

__all__ = [
    "DataIngestion",
    "DataIngestionProcessor",
    "AutoDataProcessor",
    "InternetRetriever",
    "InternetKnowledgeRetriever",
    "HuggingFaceLoader",
    "WikipediaCollector",
    "PyPIKnowledgeCollector",
    "WorkflowKnowledgeCollector",
    "KHGraphCollector",
    "TrendSignalCollector",
    "ArclaneCollector",
]

__version__ = "1.0.0"
