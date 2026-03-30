
"""
Data Ingestion System for Nexus AI Platform.
Processes uploaded files and extracts knowledge for learning.
"""

import logging
import os
import json
import csv
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
from nexus.memory.knowledge_base import KnowledgeBase, KnowledgeType
from nexus.memory.knowledge_expander import KnowledgeExpander

logger = logging.getLogger(__name__)

class DataIngestionProcessor:
    """
    Processes uploaded data files and extracts knowledge for learning.
    """
    
    def __init__(self, knowledge_base: KnowledgeBase, knowledge_expander: KnowledgeExpander):
        self.knowledge_base = knowledge_base
        self.knowledge_expander = knowledge_expander
        self.upload_dir = Path("nexus-unified/uploads")
        self.upload_dir.mkdir(exist_ok=True)
        
        # Supported file types and their processors
        self.processors = {
            '.txt': self._process_text_file,
            '.csv': self._process_csv_file,
            '.json': self._process_json_file,
            '.pdf': self._process_pdf_file,
            '.docx': self._process_docx_file,
            '.md': self._process_markdown_file
        }
        
        self.processed_files = {}
        self.extraction_stats = {}
        
    def process_uploaded_file(self, file_path: str, source_name: str = None, 
                            confidence: float = 0.8) -> Dict[str, Any]:
        """Process an uploaded file and extract knowledge."""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                return {"status": "error", "message": "File not found"}
            
            file_ext = file_path.suffix.lower()
            if file_ext not in self.processors:
                return {"status": "error", "message": f"Unsupported file type: {file_ext}"}
            
            logger.info(f"Processing file: {file_path.name}")
            
            # Process the file
            processor = self.processors[file_ext]
            extracted_data = processor(file_path)
            
            if not extracted_data:
                return {"status": "error", "message": "No data extracted from file"}
            
            # Add knowledge to the system
            knowledge_items = []
            source = source_name or file_path.name
            
            for item in extracted_data:
                if isinstance(item, dict):
                    content = item.get('content', str(item))
                    item_type = self._determine_knowledge_type(content)
                    tags = item.get('tags', []) + [file_ext[1:], 'uploaded']
                else:
                    content = str(item)
                    item_type = self._determine_knowledge_type(content)
                    tags = [file_ext[1:], 'uploaded']
                
                if len(content.strip()) > 10:  # Filter out very short content
                    knowledge_id = self.knowledge_base.add_knowledge(
                        content=content,
                        knowledge_type=item_type,
                        source=source,
                        confidence=confidence,
                        context_tags=tags
                    )
                    knowledge_items.append(knowledge_id)
            
            # Update processing statistics
            self.processed_files[str(file_path)] = {
                'processed_at': datetime.now().isoformat(),
                'items_extracted': len(knowledge_items),
                'file_type': file_ext,
                'source': source
            }
            
            result = {
                "status": "success",
                "file_name": file_path.name,
                "items_extracted": len(knowledge_items),
                "knowledge_ids": knowledge_items,
                "file_type": file_ext
            }
            
            logger.info(f"Successfully processed {file_path.name}: {len(knowledge_items)} items extracted")
            return result
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def process_text_data(self, text: str, source_name: str, 
                         data_type: str = "text", confidence: float = 0.8) -> Dict[str, Any]:
        """Process raw text data directly."""
        try:
            # Split text into meaningful chunks
            chunks = self._chunk_text(text)
            knowledge_items = []
            
            for chunk in chunks:
                if len(chunk.strip()) > 20:  # Filter out very short chunks
                    item_type = self._determine_knowledge_type(chunk)
                    knowledge_id = self.knowledge_base.add_knowledge(
                        content=chunk,
                        knowledge_type=item_type,
                        source=source_name,
                        confidence=confidence,
                        context_tags=[data_type, 'direct_input']
                    )
                    knowledge_items.append(knowledge_id)
            
            return {
                "status": "success",
                "items_extracted": len(knowledge_items),
                "knowledge_ids": knowledge_items
            }
            
        except Exception as e:
            logger.error(f"Error processing text data: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _process_text_file(self, file_path: Path) -> List[str]:
        """Process plain text files."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return self._chunk_text(content)
        except (FileNotFoundError, PermissionError) as e:
            logger.error("Cannot access text file %s: %s", file_path, e)
            return []
        except UnicodeDecodeError as e:
            logger.error("Encoding error in text file %s: %s", file_path, e)
            return []
        except Exception as e:
            logger.error("Unexpected error processing text file %s: %s: %s", file_path, type(e).__name__, e)
            return []

    def _process_csv_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process CSV files."""
        try:
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Convert each row to knowledge items
                    for key, value in row.items():
                        if value and len(str(value)) > 5:
                            data.append({
                                'content': f"{key}: {value}",
                                'tags': ['csv_data', key.lower()]
                            })
            return data
        except (FileNotFoundError, PermissionError) as e:
            logger.error("Cannot access CSV file %s: %s", file_path, e)
            return []
        except csv.Error as e:
            logger.error("CSV parse error in %s: %s", file_path, e)
            return []
        except UnicodeDecodeError as e:
            logger.error("Encoding error in CSV file %s: %s", file_path, e)
            return []
        except Exception as e:
            logger.error("Unexpected error processing CSV file %s: %s: %s", file_path, type(e).__name__, e)
            return []

    def _process_json_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process JSON files."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            data = []
            self._extract_json_knowledge(json_data, data)
            return data
        except (FileNotFoundError, PermissionError) as e:
            logger.error("Cannot access JSON file %s: %s", file_path, e)
            return []
        except json.JSONDecodeError as e:
            logger.error("JSON parse error in %s: %s", file_path, e)
            return []
        except Exception as e:
            logger.error("Unexpected error processing JSON file %s: %s: %s", file_path, type(e).__name__, e)
            return []

    def _process_pdf_file(self, file_path: Path) -> List[str]:
        """Process PDF files."""
        try:
            try:
                from pypdf import PdfReader
            except ImportError:
                from PyPDF2 import PdfReader

            text_content = []
            with open(file_path, 'rb') as f:
                pdf_reader = PdfReader(f)
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text.strip():
                        text_content.extend(self._chunk_text(text))
            return text_content
        except ImportError:
            logger.error("No PDF library available (install pypdf or PyPDF2)")
            return []
        except (FileNotFoundError, PermissionError) as e:
            logger.error("Cannot access PDF file %s: %s", file_path, e)
            return []
        except Exception as e:
            logger.error("Error processing PDF file %s: %s: %s", file_path, type(e).__name__, e)
            return []

    def _process_docx_file(self, file_path: Path) -> List[str]:
        """Process Word documents."""
        try:
            import docx  # lazy: python-docx is an optional dependency
            doc = docx.Document(file_path)
            text_content = []

            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_content.extend(self._chunk_text(paragraph.text))

            return text_content
        except (FileNotFoundError, PermissionError) as e:
            logger.error("Cannot access DOCX file %s: %s", file_path, e)
            return []
        except Exception as e:
            logger.error("Error processing DOCX file %s: %s: %s", file_path, type(e).__name__, e)
            return []

    def _process_markdown_file(self, file_path: Path) -> List[str]:
        """Process Markdown files."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return self._chunk_text(content)
        except (FileNotFoundError, PermissionError) as e:
            logger.error("Cannot access markdown file %s: %s", file_path, e)
            return []
        except UnicodeDecodeError as e:
            logger.error("Encoding error in markdown file %s: %s", file_path, e)
            return []
        except Exception as e:
            logger.error("Unexpected error processing markdown file %s: %s: %s", file_path, type(e).__name__, e)
            return []
    
    def _chunk_text(self, text: str, max_chunk_size: int = 500) -> List[str]:
        """Split text into meaningful chunks."""
        # Split by sentences first, then by paragraphs
        sentences = text.split('.')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(current_chunk) + len(sentence) < max_chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if len(chunk.strip()) > 20]
    
    def _extract_json_knowledge(self, data: Any, result: List[Dict[str, Any]], prefix: str = ""):
        """Recursively extract knowledge from JSON data."""
        if isinstance(data, dict):
            for key, value in data.items():
                current_prefix = f"{prefix}.{key}" if prefix else key
                if isinstance(value, (dict, list)):
                    self._extract_json_knowledge(value, result, current_prefix)
                else:
                    result.append({
                        'content': f"{current_prefix}: {value}",
                        'tags': ['json_data', key.lower()]
                    })
        elif isinstance(data, list):
            for i, item in enumerate(data):
                current_prefix = f"{prefix}[{i}]" if prefix else f"item_{i}"
                self._extract_json_knowledge(item, result, current_prefix)
        else:
            result.append({
                'content': f"{prefix}: {data}" if prefix else str(data),
                'tags': ['json_data']
            })
    
    def _determine_knowledge_type(self, content: str) -> KnowledgeType:
        """Determine the type of knowledge based on content."""
        content_lower = content.lower()
        
        # Check for procedural knowledge
        if any(word in content_lower for word in ['how to', 'step', 'procedure', 'process', 'method']):
            return KnowledgeType.PROCEDURAL
        
        # Check for factual knowledge
        elif any(word in content_lower for word in ['is', 'are', 'was', 'were', 'definition', 'means']):
            return KnowledgeType.FACTUAL
        
        # Check for experiential knowledge
        elif any(word in content_lower for word in ['experience', 'learned', 'discovered', 'found']):
            return KnowledgeType.EXPERIENTIAL
        
        # Default to factual
        return KnowledgeType.FACTUAL
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get statistics about processed files."""
        total_files = len(self.processed_files)
        total_items = sum(info['items_extracted'] for info in self.processed_files.values())
        
        file_types = {}
        for info in self.processed_files.values():
            file_type = info['file_type']
            file_types[file_type] = file_types.get(file_type, 0) + 1
        
        return {
            'total_files_processed': total_files,
            'total_knowledge_items_extracted': total_items,
            'file_types_processed': file_types,
            'avg_items_per_file': total_items / max(total_files, 1),
            'processed_files': self.processed_files
        }
    
    def clear_processing_history(self):
        """Clear the processing history."""
        self.processed_files.clear()
        logger.info("Processing history cleared")
