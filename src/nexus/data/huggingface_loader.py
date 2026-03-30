
"""
Hugging Face dataset loader for Nexus AI Platform.
Loads datasets from Hugging Face Hub for training.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Union
from datasets import load_dataset, Dataset, DatasetDict
from nexus.memory.knowledge_base import KnowledgeBase, KnowledgeType
from nexus.memory.knowledge_expander import KnowledgeExpander

logger = logging.getLogger(__name__)

class HuggingFaceLoader:
    """
    Loads datasets from Hugging Face Hub and processes them for training.
    """
    
    def __init__(self, knowledge_base: KnowledgeBase, knowledge_expander: KnowledgeExpander):
        self.knowledge_base = knowledge_base
        self.knowledge_expander = knowledge_expander
        self.loaded_datasets = {}
        
    def load_dataset_from_hub(self, dataset_name: str, config_name: str = None, 
                             split: str = None, streaming: bool = False,
                             max_samples: int = 1000) -> Dict[str, Any]:
        """Load a dataset from Hugging Face Hub."""
        try:
            logger.info(f"Loading dataset: {dataset_name}")
            
            # Load dataset
            dataset = load_dataset(
                dataset_name, 
                config_name, 
                split=split, 
                streaming=streaming,
                trust_remote_code=True
            )
            
            # Limit samples if specified
            if not streaming and isinstance(dataset, Dataset) and len(dataset) > max_samples:
                dataset = dataset.select(range(max_samples))
            elif isinstance(dataset, DatasetDict):
                # For DatasetDict, limit each split
                for split_name in dataset.keys():
                    if len(dataset[split_name]) > max_samples:
                        dataset[split_name] = dataset[split_name].select(range(max_samples))
            
            # Store reference
            dataset_id = f"{dataset_name}_{int(time.time())}"
            self.loaded_datasets[dataset_id] = {
                'dataset': dataset,
                'name': dataset_name,
                'config': config_name,
                'split': split,
                'loaded_at': time.time()
            }
            
            return {
                "status": "success",
                "dataset_id": dataset_id,
                "dataset_name": dataset_name,
                "size": self._get_dataset_size(dataset),
                "features": self._get_dataset_features(dataset)
            }
            
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def process_qa_dataset(self, dataset_id: str, question_column: str = "question", 
                          answer_column: str = "answer", confidence: float = 0.8) -> Dict[str, Any]:
        """Process a Q&A dataset for training."""
        try:
            if dataset_id not in self.loaded_datasets:
                return {"status": "error", "message": "Dataset not found"}
            
            dataset_info = self.loaded_datasets[dataset_id]
            dataset = dataset_info['dataset']
            
            knowledge_items = []
            processed_count = 0
            
            # Handle different dataset types
            if isinstance(dataset, DatasetDict):
                # Process all splits
                for split_name, split_data in dataset.items():
                    items = self._process_split_qa(
                        split_data, question_column, answer_column, 
                        confidence, f"{dataset_info['name']}_{split_name}"
                    )
                    knowledge_items.extend(items)
                    processed_count += len(items)
            else:
                # Single dataset
                items = self._process_split_qa(
                    dataset, question_column, answer_column, 
                    confidence, dataset_info['name']
                )
                knowledge_items.extend(items)
                processed_count += len(items)
            
            return {
                "status": "success",
                "dataset_name": dataset_info['name'],
                "processed_count": processed_count,
                "knowledge_ids": knowledge_items
            }
            
        except Exception as e:
            logger.error(f"Error processing Q&A dataset: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def process_text_dataset(self, dataset_id: str, text_column: str = "text", 
                           confidence: float = 0.7) -> Dict[str, Any]:
        """Process a text dataset for training."""
        try:
            if dataset_id not in self.loaded_datasets:
                return {"status": "error", "message": "Dataset not found"}
            
            dataset_info = self.loaded_datasets[dataset_id]
            dataset = dataset_info['dataset']
            
            knowledge_items = []
            processed_count = 0
            
            # Handle different dataset types
            if isinstance(dataset, DatasetDict):
                for split_name, split_data in dataset.items():
                    items = self._process_split_text(
                        split_data, text_column, confidence, 
                        f"{dataset_info['name']}_{split_name}"
                    )
                    knowledge_items.extend(items)
                    processed_count += len(items)
            else:
                items = self._process_split_text(
                    dataset, text_column, confidence, dataset_info['name']
                )
                knowledge_items.extend(items)
                processed_count += len(items)
            
            return {
                "status": "success",
                "dataset_name": dataset_info['name'],
                "processed_count": processed_count,
                "knowledge_ids": knowledge_items
            }
            
        except Exception as e:
            logger.error(f"Error processing text dataset: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _process_split_qa(self, split_data: Dataset, question_col: str, 
                         answer_col: str, confidence: float, source: str) -> List[str]:
        """Process a single split of Q&A data."""
        knowledge_items = []
        
        for i, example in enumerate(split_data):
            try:
                question = example.get(question_col, '')
                answer = example.get(answer_col, '')
                
                if question and answer:
                    # Create Q&A knowledge item
                    qa_content = f"Q: {question}\nA: {answer}"
                    
                    knowledge_id = self.knowledge_base.add_knowledge(
                        content=qa_content,
                        knowledge_type=KnowledgeType.FACTUAL,
                        source=source,
                        confidence=confidence,
                        context_tags=['huggingface', 'qa_pair', 'training_data']
                    )
                    knowledge_items.append(knowledge_id)
                    
                    # Also add individual question and answer
                    if len(question) > 10:
                        q_id = self.knowledge_base.add_knowledge(
                            content=f"Question: {question}",
                            knowledge_type=KnowledgeType.FACTUAL,
                            source=source,
                            confidence=confidence * 0.8,
                            context_tags=['huggingface', 'question', 'training_data']
                        )
                        knowledge_items.append(q_id)
                    
                    if len(answer) > 10:
                        a_id = self.knowledge_base.add_knowledge(
                            content=f"Answer: {answer}",
                            knowledge_type=KnowledgeType.FACTUAL,
                            source=source,
                            confidence=confidence * 0.9,
                            context_tags=['huggingface', 'answer', 'training_data']
                        )
                        knowledge_items.append(a_id)
                
            except Exception as e:
                logger.warning(f"Error processing example {i}: {str(e)}")
                continue
        
        return knowledge_items
    
    def _process_split_text(self, split_data: Dataset, text_col: str, 
                          confidence: float, source: str) -> List[str]:
        """Process a single split of text data."""
        knowledge_items = []
        
        for i, example in enumerate(split_data):
            try:
                text = example.get(text_col, '')
                
                if text and len(text.strip()) > 20:
                    # Chunk long texts
                    chunks = self._chunk_text(text)
                    
                    for chunk in chunks:
                        knowledge_id = self.knowledge_base.add_knowledge(
                            content=chunk,
                            knowledge_type=self._determine_knowledge_type(chunk),
                            source=source,
                            confidence=confidence,
                            context_tags=['huggingface', 'text_data', 'training_data']
                        )
                        knowledge_items.append(knowledge_id)
                
            except Exception as e:
                logger.warning(f"Error processing text example {i}: {str(e)}")
                continue
        
        return knowledge_items
    
    def _chunk_text(self, text: str, max_chunk_size: int = 500) -> List[str]:
        """Split text into chunks."""
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
    
    def _get_dataset_size(self, dataset: Union[Dataset, DatasetDict]) -> int:
        """Get total size of dataset."""
        if isinstance(dataset, DatasetDict):
            return sum(len(split) for split in dataset.values())
        return len(dataset)
    
    def _get_dataset_features(self, dataset: Union[Dataset, DatasetDict]) -> List[str]:
        """Get feature names from dataset."""
        if isinstance(dataset, DatasetDict):
            # Get features from first split
            first_split = next(iter(dataset.values()))
            return list(first_split.features.keys())
        return list(dataset.features.keys())
    
    def list_popular_datasets(self) -> List[Dict[str, str]]:
        """List popular datasets suitable for training."""
        return [
            {
                "name": "squad",
                "description": "Stanford Question Answering Dataset",
                "type": "qa",
                "question_col": "question",
                "answer_col": "answers.text[0]"
            },
            {
                "name": "ms_marco",
                "description": "Microsoft Machine Reading Comprehension",
                "type": "qa",
                "question_col": "query",
                "answer_col": "passages.passage_text[0]"
            },
            {
                "name": "natural_questions",
                "description": "Natural Questions dataset",
                "type": "qa",
                "question_col": "question.text",
                "answer_col": "annotations.short_answers[0].text"
            },
            {
                "name": "trivia_qa",
                "description": "Trivia Questions dataset",
                "type": "qa",
                "question_col": "question",
                "answer_col": "answer.value"
            },
            {
                "name": "common_crawl",
                "description": "Web crawl text data",
                "type": "text",
                "text_col": "text"
            },
            {
                "name": "wikipedia",
                "description": "Wikipedia articles",
                "type": "text",
                "text_col": "text"
            }
        ]
    
    def get_dataset_info(self, dataset_id: str) -> Dict[str, Any]:
        """Get information about a loaded dataset."""
        if dataset_id not in self.loaded_datasets:
            return {"status": "error", "message": "Dataset not found"}
        
        info = self.loaded_datasets[dataset_id]
        return {
            "status": "success",
            "dataset_id": dataset_id,
            "name": info['name'],
            "config": info['config'],
            "split": info['split'],
            "loaded_at": info['loaded_at'],
            "size": self._get_dataset_size(info['dataset']),
            "features": self._get_dataset_features(info['dataset'])
        }
    
    def initialize(self):
        """Initialize the HuggingFace loader."""
        logger.info("HuggingFace Loader initialized")
        
    def clear_loaded_datasets(self):
        """Clear all loaded datasets from memory."""
        self.loaded_datasets.clear()
        logger.info("Cleared all loaded datasets")
