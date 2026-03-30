
"""
In-memory vector storage client for knowledge management.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class VectorClient:
    """In-memory vector storage client."""
    
    def __init__(self, dimension: int = 512):
        self.dimension = dimension
        self.vectors = {}
        self.metadata = {}
        self.index_counter = 0
        
    def insert(self, vector: List[float], metadata: Dict[str, Any] = None) -> str:
        """Insert a vector with optional metadata."""
        if len(vector) != self.dimension:
            raise ValueError(f"Vector dimension {len(vector)} doesn't match expected {self.dimension}")
            
        vector_id = f"vec_{self.index_counter}"
        self.index_counter += 1
        
        self.vectors[vector_id] = np.array(vector)
        self.metadata[vector_id] = metadata or {}
        
        logger.debug(f"Inserted vector {vector_id}")
        return vector_id
        
    def search(self, query_vector: List[float], k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar vectors."""
        if len(query_vector) != self.dimension:
            raise ValueError(f"Query vector dimension {len(query_vector)} doesn't match expected {self.dimension}")
            
        if not self.vectors:
            return []
            
        query_np = np.array(query_vector)
        similarities = []
        
        for vector_id, vector in self.vectors.items():
            # Calculate cosine similarity
            similarity = np.dot(query_np, vector) / (np.linalg.norm(query_np) * np.linalg.norm(vector))
            similarities.append((vector_id, float(similarity)))
            
        # Sort by similarity (descending) and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
        
    def get(self, vector_id: str) -> Optional[Tuple[List[float], Dict[str, Any]]]:
        """Get a vector and its metadata by ID."""
        if vector_id not in self.vectors:
            return None
            
        vector = self.vectors[vector_id].tolist()
        metadata = self.metadata[vector_id]
        
        return vector, metadata
        
    def delete(self, vector_id: str) -> bool:
        """Delete a vector by ID."""
        if vector_id not in self.vectors:
            return False
            
        del self.vectors[vector_id]
        del self.metadata[vector_id]
        
        logger.debug(f"Deleted vector {vector_id}")
        return True
        
    def count(self) -> int:
        """Get total number of vectors stored."""
        return len(self.vectors)
        
    def clear(self):
        """Clear all stored vectors."""
        self.vectors.clear()
        self.metadata.clear()
        self.index_counter = 0
        logger.info("Cleared all vectors")
