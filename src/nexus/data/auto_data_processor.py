
"""
Auto Data Processor for Nexus AI Platform.
Automatically detects, processes, and uploads data from various sources.
"""

import logging
import os
import json
import csv
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import mimetypes
import pandas as pd
from nexus.memory.knowledge_base import KnowledgeBase, KnowledgeType
from nexus.memory.knowledge_expander import KnowledgeExpander
from nexus.data.huggingface_loader import HuggingFaceLoader

logger = logging.getLogger(__name__)

class AutoDataProcessor:
    """
    Automatically detects, processes, and uploads data from various sources.
    """
    
    def __init__(self, knowledge_base: KnowledgeBase, knowledge_expander: KnowledgeExpander, hf_loader: 'HuggingFaceLoader'):
        self.knowledge_base = knowledge_base
        self.knowledge_expander = knowledge_expander
        self.hf_loader = hf_loader
        self.initialized = False
        
    def initialize(self):
        """Initialize the auto data processor."""
        if self.initialized:
            return
        logger.info("Auto Data Processor initialized")
        self.initialized = True

class AutoDataProcessor:
    """
    Automatically processes and uploads data from various sources.
    """
    
    def __init__(self, knowledge_base: KnowledgeBase, knowledge_expander: KnowledgeExpander,
                 hf_loader: HuggingFaceLoader):
        self.knowledge_base = knowledge_base
        self.knowledge_expander = knowledge_expander
        self.hf_loader = hf_loader
        
        # Set up directories
        self.watch_dir = Path("nexus-unified/data/auto_upload")
        self.processed_dir = Path("nexus-unified/data/processed")
        self.watch_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # File processing state
        self.processed_files = {}
        self.file_signatures = {}
        
        # Auto-detection patterns
        self.qa_patterns = {
            'question_keys': ['question', 'q', 'query', 'prompt', 'input'],
            'answer_keys': ['answer', 'a', 'response', 'output', 'target', 'label'],
            'context_keys': ['context', 'passage', 'text', 'content']
        }
        
        self.text_patterns = {
            'text_keys': ['text', 'content', 'body', 'description', 'summary'],
            'title_keys': ['title', 'subject', 'headline', 'name'],
            'metadata_keys': ['author', 'date', 'category', 'tags', 'source']
        }

    def auto_scan_and_process(self) -> Dict[str, Any]:
        """Scan the watch directory and automatically process new files."""
        results = {
            "files_found": 0,
            "files_processed": 0,
            "files_skipped": 0,
            "processing_results": [],
            "errors": []
        }
        
        try:
            # Scan for files
            for file_path in self.watch_dir.rglob('*'):
                if file_path.is_file():
                    results["files_found"] += 1
                    
                    # Check if already processed
                    file_signature = self._get_file_signature(file_path)
                    if file_signature in self.file_signatures:
                        results["files_skipped"] += 1
                        continue
                    
                    # Process the file
                    try:
                        process_result = self.auto_process_file(file_path)
                        if process_result["status"] == "success":
                            results["files_processed"] += 1
                            self.file_signatures[file_signature] = {
                                'file_path': str(file_path),
                                'processed_at': datetime.now().isoformat(),
                                'result': process_result
                            }
                            
                            # Move to processed directory
                            self._move_to_processed(file_path)
                        
                        results["processing_results"].append(process_result)
                        
                    except Exception as e:
                        error_msg = f"Error processing {file_path.name}: {str(e)}"
                        logger.error(error_msg)
                        results["errors"].append(error_msg)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in auto scan: {str(e)}")
            results["errors"].append(str(e))
            return results

    def auto_process_file(self, file_path: Path) -> Dict[str, Any]:
        """Automatically detect file type and content structure, then process."""
        try:
            logger.info(f"Auto-processing file: {file_path.name}")
            
            # Detect file type
            file_info = self._detect_file_type(file_path)
            
            # Analyze content structure
            content_analysis = self._analyze_content_structure(file_path, file_info)
            
            # Process based on detected structure
            processing_result = self._process_by_structure(file_path, file_info, content_analysis)
            
            return {
                "status": "success",
                "file_name": file_path.name,
                "file_type": file_info["type"],
                "content_structure": content_analysis,
                "processing_result": processing_result
            }
            
        except Exception as e:
            logger.error(f"Error auto-processing {file_path.name}: {str(e)}")
            return {
                "status": "error",
                "file_name": file_path.name,
                "message": str(e)
            }

    def _detect_file_type(self, file_path: Path) -> Dict[str, Any]:
        """Detect file type using multiple methods."""
        file_info = {
            "extension": file_path.suffix.lower(),
            "type": "unknown",
            "mime_type": None,
            "encoding": "utf-8"
        }
        
        try:
            # Use mimetypes for MIME type detection
            mime_type, _ = mimetypes.guess_type(str(file_path))
            file_info["mime_type"] = mime_type
            
            # Map to our internal types
            if file_path.suffix.lower() in ['.json', '.jsonl']:
                file_info["type"] = "json"
            elif file_path.suffix.lower() in ['.csv', '.tsv']:
                file_info["type"] = "csv"
            elif file_path.suffix.lower() in ['.txt', '.md']:
                file_info["type"] = "text"
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                file_info["type"] = "excel"
            elif mime_type and 'json' in mime_type:
                file_info["type"] = "json"
            elif mime_type and ('csv' in mime_type or 'text' in mime_type):
                file_info["type"] = "csv"
            
        except Exception as e:
            logger.warning(f"Could not detect MIME type for {file_path.name}: {e}")
        
        return file_info

    def _analyze_content_structure(self, file_path: Path, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the internal structure of the file content."""
        analysis = {
            "structure_type": "unknown",
            "columns": [],
            "sample_data": None,
            "qa_potential": False,
            "text_content": False,
            "structured_data": False
        }
        
        try:
            if file_info["type"] == "json":
                analysis = self._analyze_json_structure(file_path)
            elif file_info["type"] == "csv":
                analysis = self._analyze_csv_structure(file_path)
            elif file_info["type"] == "excel":
                analysis = self._analyze_excel_structure(file_path)
            elif file_info["type"] == "text":
                analysis = self._analyze_text_structure(file_path)
                
        except Exception as e:
            logger.warning(f"Could not analyze structure of {file_path.name}: {e}")
        
        return analysis

    def _analyze_json_structure(self, file_path: Path) -> Dict[str, Any]:
        """Analyze JSON file structure."""
        analysis = {
            "structure_type": "json",
            "columns": [],
            "sample_data": None,
            "qa_potential": False,
            "text_content": False,
            "structured_data": True
        }
        
        with open(file_path, 'r', encoding='utf-8') as f:
            # Try to load as JSON lines or regular JSON
            first_line = f.readline().strip()
            f.seek(0)
            
            if first_line.startswith('{'):
                # Might be JSONL
                sample_items = []
                for i, line in enumerate(f):
                    if i >= 5:  # Sample first 5 items
                        break
                    try:
                        item = json.loads(line.strip())
                        sample_items.append(item)
                    except json.JSONDecodeError:
                        # Not valid JSONL, try as regular JSON
                        break
                
                if sample_items:
                    analysis["sample_data"] = sample_items
                    analysis["columns"] = list(sample_items[0].keys()) if sample_items else []
                else:
                    # Regular JSON
                    f.seek(0)
                    data = json.load(f)
                    if isinstance(data, list) and data:
                        analysis["sample_data"] = data[:5]
                        analysis["columns"] = list(data[0].keys()) if isinstance(data[0], dict) else []
                    elif isinstance(data, dict):
                        analysis["sample_data"] = [data]
                        analysis["columns"] = list(data.keys())
            else:
                # Regular JSON
                data = json.load(f)
                if isinstance(data, list) and data:
                    analysis["sample_data"] = data[:5]
                    analysis["columns"] = list(data[0].keys()) if isinstance(data[0], dict) else []
                elif isinstance(data, dict):
                    analysis["sample_data"] = [data]
                    analysis["columns"] = list(data.keys())
        
        # Check for Q&A patterns
        analysis["qa_potential"] = self._detect_qa_pattern(analysis["columns"])
        analysis["text_content"] = self._detect_text_content(analysis["columns"])
        
        return analysis

    def _analyze_csv_structure(self, file_path: Path) -> Dict[str, Any]:
        """Analyze CSV file structure."""
        analysis = {
            "structure_type": "csv",
            "columns": [],
            "sample_data": None,
            "qa_potential": False,
            "text_content": False,
            "structured_data": True
        }
        
        try:
            # Use pandas for robust CSV handling
            df = pd.read_csv(file_path, nrows=5)  # Sample first 5 rows
            analysis["columns"] = df.columns.tolist()
            analysis["sample_data"] = df.to_dict('records')
            
            # Check for Q&A and text patterns
            analysis["qa_potential"] = self._detect_qa_pattern(analysis["columns"])
            analysis["text_content"] = self._detect_text_content(analysis["columns"])
            
        except Exception as e:
            logger.warning(f"Error analyzing CSV {file_path.name}: {e}")
        
        return analysis

    def _analyze_excel_structure(self, file_path: Path) -> Dict[str, Any]:
        """Analyze Excel file structure."""
        analysis = {
            "structure_type": "excel",
            "columns": [],
            "sample_data": None,
            "qa_potential": False,
            "text_content": False,
            "structured_data": True
        }
        
        try:
            df = pd.read_excel(file_path, nrows=5)
            analysis["columns"] = df.columns.tolist()
            analysis["sample_data"] = df.to_dict('records')
            
            analysis["qa_potential"] = self._detect_qa_pattern(analysis["columns"])
            analysis["text_content"] = self._detect_text_content(analysis["columns"])
            
        except Exception as e:
            logger.warning(f"Error analyzing Excel {file_path.name}: {e}")
        
        return analysis

    def _analyze_text_structure(self, file_path: Path) -> Dict[str, Any]:
        """Analyze plain text file structure."""
        analysis = {
            "structure_type": "text",
            "columns": [],
            "sample_data": None,
            "qa_potential": False,
            "text_content": True,
            "structured_data": False
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read(1000)  # Sample first 1000 chars
                analysis["sample_data"] = content
                
                # Check if it looks like structured text (Q&A pairs, etc.)
                if 'Q:' in content and 'A:' in content:
                    analysis["qa_potential"] = True
                    analysis["structure_type"] = "qa_text"
                
        except Exception as e:
            logger.warning(f"Error analyzing text {file_path.name}: {e}")
        
        return analysis

    def _detect_qa_pattern(self, columns: List[str]) -> bool:
        """Detect if columns suggest Q&A structure."""
        column_lower = [col.lower() for col in columns]
        
        has_question = any(q_key in col for col in column_lower for q_key in self.qa_patterns['question_keys'])
        has_answer = any(a_key in col for col in column_lower for a_key in self.qa_patterns['answer_keys'])
        
        return has_question and has_answer

    def _detect_text_content(self, columns: List[str]) -> bool:
        """Detect if columns contain significant text content."""
        column_lower = [col.lower() for col in columns]
        
        return any(t_key in col for col in column_lower for t_key in self.text_patterns['text_keys'])

    def _process_by_structure(self, file_path: Path, file_info: Dict[str, Any], 
                            content_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Process file based on detected structure."""
        if content_analysis["qa_potential"]:
            return self._process_as_qa_data(file_path, file_info, content_analysis)
        elif content_analysis["text_content"]:
            return self._process_as_text_data(file_path, file_info, content_analysis)
        elif content_analysis["structured_data"]:
            return self._process_as_structured_data(file_path, file_info, content_analysis)
        else:
            return self._process_as_general_data(file_path, file_info, content_analysis)

    def _process_as_qa_data(self, file_path: Path, file_info: Dict[str, Any], 
                          content_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Process file as Q&A training data."""
        try:
            qa_pairs = []
            
            if file_info["type"] == "json":
                qa_pairs = self._extract_qa_from_json(file_path, content_analysis)
            elif file_info["type"] == "csv":
                qa_pairs = self._extract_qa_from_csv(file_path, content_analysis)
            elif content_analysis["structure_type"] == "qa_text":
                qa_pairs = self._extract_qa_from_text(file_path)
            
            # Add to knowledge base
            knowledge_ids = []
            for qa in qa_pairs:
                if qa.get('question') and qa.get('answer'):
                    content = f"Q: {qa['question']}\nA: {qa['answer']}"
                    knowledge_id = self.knowledge_base.add_knowledge(
                        content=content,
                        knowledge_type=KnowledgeType.FACTUAL,
                        source=f"auto_processed_{file_path.name}",
                        confidence=0.8,
                        context_tags=['auto_processed', 'qa_pair', 'training_data']
                    )
                    knowledge_ids.append(knowledge_id)
            
            return {
                "processing_type": "qa_data",
                "qa_pairs_extracted": len(qa_pairs),
                "knowledge_ids": knowledge_ids
            }
            
        except Exception as e:
            logger.error(f"Error processing Q&A data from {file_path.name}: {e}")
            return {"processing_type": "qa_data", "error": str(e)}

    def _process_as_text_data(self, file_path: Path, file_info: Dict[str, Any], 
                            content_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Process file as text data."""
        try:
            text_chunks = []
            
            if file_info["type"] == "text":
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    text_chunks = self._chunk_text(content)
            elif file_info["type"] in ["json", "csv"]:
                text_chunks = self._extract_text_from_structured(file_path, file_info, content_analysis)
            
            # Add to knowledge base
            knowledge_ids = []
            for chunk in text_chunks:
                if len(chunk.strip()) > 20:
                    knowledge_id = self.knowledge_base.add_knowledge(
                        content=chunk,
                        knowledge_type=self._determine_knowledge_type(chunk),
                        source=f"auto_processed_{file_path.name}",
                        confidence=0.7,
                        context_tags=['auto_processed', 'text_data']
                    )
                    knowledge_ids.append(knowledge_id)
            
            return {
                "processing_type": "text_data",
                "text_chunks_extracted": len(text_chunks),
                "knowledge_ids": knowledge_ids
            }
            
        except Exception as e:
            logger.error(f"Error processing text data from {file_path.name}: {e}")
            return {"processing_type": "text_data", "error": str(e)}

    def _process_as_structured_data(self, file_path: Path, file_info: Dict[str, Any], 
                                  content_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Process file as structured data."""
        try:
            knowledge_ids = []
            processed_items = 0
            
            if file_info["type"] == "csv":
                df = pd.read_csv(file_path)
                for _, row in df.iterrows():
                    for col, value in row.items():
                        if pd.notna(value) and len(str(value)) > 10:
                            content = f"{col}: {value}"
                            knowledge_id = self.knowledge_base.add_knowledge(
                                content=content,
                                knowledge_type=KnowledgeType.FACTUAL,
                                source=f"auto_processed_{file_path.name}",
                                confidence=0.6,
                                context_tags=['auto_processed', 'structured_data', col.lower()]
                            )
                            knowledge_ids.append(knowledge_id)
                            processed_items += 1
            
            return {
                "processing_type": "structured_data",
                "items_processed": processed_items,
                "knowledge_ids": knowledge_ids
            }
            
        except Exception as e:
            logger.error(f"Error processing structured data from {file_path.name}: {e}")
            return {"processing_type": "structured_data", "error": str(e)}

    def _process_as_general_data(self, file_path: Path, file_info: Dict[str, Any], 
                               content_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Process file as general data."""
        try:
            # Fallback processing - extract any text content
            content = self._extract_any_text_content(file_path, file_info)
            
            if content:
                chunks = self._chunk_text(content)
                knowledge_ids = []
                
                for chunk in chunks:
                    if len(chunk.strip()) > 15:
                        knowledge_id = self.knowledge_base.add_knowledge(
                            content=chunk,
                            knowledge_type=KnowledgeType.FACTUAL,
                            source=f"auto_processed_{file_path.name}",
                            confidence=0.5,
                            context_tags=['auto_processed', 'general_data']
                        )
                        knowledge_ids.append(knowledge_id)
                
                return {
                    "processing_type": "general_data",
                    "chunks_extracted": len(chunks),
                    "knowledge_ids": knowledge_ids
                }
            else:
                return {
                    "processing_type": "general_data",
                    "message": "No extractable content found"
                }
                
        except Exception as e:
            logger.error(f"Error processing general data from {file_path.name}: {e}")
            return {"processing_type": "general_data", "error": str(e)}

    def _extract_qa_from_json(self, file_path: Path, content_analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract Q&A pairs from JSON data."""
        qa_pairs = []
        columns = content_analysis["columns"]
        
        # Find question and answer columns
        question_col = self._find_column_by_pattern(columns, self.qa_patterns['question_keys'])
        answer_col = self._find_column_by_pattern(columns, self.qa_patterns['answer_keys'])
        
        if not question_col or not answer_col:
            return qa_pairs
        
        with open(file_path, 'r', encoding='utf-8') as f:
            # Handle both regular JSON and JSONL
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    data = [data]
            except json.JSONDecodeError:
                # Try as JSONL format
                f.seek(0)
                data = []
                for line in f:
                    try:
                        data.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
        
        for item in data:
            if question_col in item and answer_col in item:
                qa_pairs.append({
                    'question': str(item[question_col]),
                    'answer': str(item[answer_col])
                })
        
        return qa_pairs

    def _extract_qa_from_csv(self, file_path: Path, content_analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract Q&A pairs from CSV data."""
        qa_pairs = []
        columns = content_analysis["columns"]
        
        question_col = self._find_column_by_pattern(columns, self.qa_patterns['question_keys'])
        answer_col = self._find_column_by_pattern(columns, self.qa_patterns['answer_keys'])
        
        if not question_col or not answer_col:
            return qa_pairs
        
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            if pd.notna(row[question_col]) and pd.notna(row[answer_col]):
                qa_pairs.append({
                    'question': str(row[question_col]),
                    'answer': str(row[answer_col])
                })
        
        return qa_pairs

    def _extract_qa_from_text(self, file_path: Path) -> List[Dict[str, str]]:
        """Extract Q&A pairs from text file."""
        qa_pairs = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by common Q&A patterns
        sections = content.split('\n')
        current_q = None
        current_a = None
        
        for line in sections:
            line = line.strip()
            if line.startswith('Q:') or line.startswith('Question:'):
                if current_q and current_a:
                    qa_pairs.append({'question': current_q, 'answer': current_a})
                current_q = line[2:].strip() if line.startswith('Q:') else line[9:].strip()
                current_a = None
            elif line.startswith('A:') or line.startswith('Answer:'):
                current_a = line[2:].strip() if line.startswith('A:') else line[7:].strip()
            elif current_a and line:
                current_a += " " + line
        
        if current_q and current_a:
            qa_pairs.append({'question': current_q, 'answer': current_a})
        
        return qa_pairs

    def _find_column_by_pattern(self, columns: List[str], patterns: List[str]) -> Optional[str]:
        """Find column matching any of the given patterns."""
        for col in columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in patterns):
                return col
        return None

    def _chunk_text(self, text: str, max_chunk_size: int = 500) -> List[str]:
        """Split text into meaningful chunks."""
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

    def _determine_knowledge_type(self, content: str) -> KnowledgeType:
        """Determine knowledge type from content."""
        content_lower = content.lower()
        
        if any(word in content_lower for word in ['how to', 'step', 'procedure', 'process']):
            return KnowledgeType.PROCEDURAL
        elif any(word in content_lower for word in ['experience', 'learned', 'discovered']):
            return KnowledgeType.EXPERIENTIAL
        else:
            return KnowledgeType.FACTUAL

    def _extract_text_from_structured(self, file_path: Path, file_info: Dict[str, Any], 
                                    content_analysis: Dict[str, Any]) -> List[str]:
        """Extract text content from structured files."""
        text_chunks = []
        columns = content_analysis["columns"]
        
        text_columns = [col for col in columns if self._find_column_by_pattern([col], self.text_patterns['text_keys'])]
        
        if file_info["type"] == "csv":
            df = pd.read_csv(file_path)
            for col in text_columns:
                if col in df.columns:
                    for text in df[col].dropna():
                        if len(str(text)) > 50:
                            text_chunks.extend(self._chunk_text(str(text)))
        
        return text_chunks

    def _extract_any_text_content(self, file_path: Path, file_info: Dict[str, Any]) -> str:
        """Extract any text content from file as fallback."""
        try:
            if file_info["type"] == "text":
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            elif file_info["type"] == "json":
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return json.dumps(data, indent=2)
        except (IOError, json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.debug(f"Could not extract text content from {file_path}: {e}")

        return ""

    def _get_file_signature(self, file_path: Path) -> str:
        """Get file signature for duplicate detection."""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def _move_to_processed(self, file_path: Path):
        """Move processed file to processed directory."""
        try:
            processed_path = self.processed_dir / file_path.name
            file_path.rename(processed_path)
        except Exception as e:
            logger.warning(f"Could not move {file_path.name} to processed directory: {e}")

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get statistics about auto-processing."""
        return {
            "total_files_processed": len(self.file_signatures),
            "watch_directory": str(self.watch_dir),
            "processed_directory": str(self.processed_dir),
            "file_signatures": len(self.file_signatures),
            "processing_history": list(self.file_signatures.values())[:10]  # Last 10
        }

    def clear_processed_history(self):
        """Clear processing history."""
        self.file_signatures.clear()
        self.processed_files.clear()
        logger.info("Auto-processing history cleared")
