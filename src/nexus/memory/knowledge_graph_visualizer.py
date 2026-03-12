
"""
Knowledge graph visualization and exploration system.
"""

import logging
import json
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class KnowledgeGraphVisualizer:
    """
    System for visualizing and exploring knowledge connections.
    """
    
    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base
        self.graph_cache = {}
        self.connection_strength_cache = {}
        
    def generate_graph_data(self, center_concept: str = None, depth: int = 2) -> Dict[str, Any]:
        """Generate graph data for visualization."""
        nodes = []
        edges = []
        
        if center_concept:
            # Build graph around a specific concept
            visited = set()
            queue = deque([(center_concept, 0)])
            
            while queue:
                concept, current_depth = queue.popleft()
                if concept in visited or current_depth > depth:
                    continue
                    
                visited.add(concept)
                
                # Add node
                node_data = self._get_node_data(concept)
                if node_data:
                    nodes.append(node_data)
                    
                    # Find related concepts
                    related = self._find_related_concepts(concept)
                    for related_concept, strength in related:
                        if related_concept not in visited:
                            queue.append((related_concept, current_depth + 1))
                            
                        # Add edge
                        edges.append({
                            'source': concept,
                            'target': related_concept,
                            'strength': strength,
                            'type': 'semantic'
                        })
        else:
            # Generate overview graph
            nodes, edges = self._generate_overview_graph()
            
        return {
            'nodes': nodes,
            'edges': edges,
            'metadata': {
                'total_concepts': len(nodes),
                'total_connections': len(edges),
                'generation_method': 'centered' if center_concept else 'overview'
            }
        }
    
    def _get_node_data(self, concept: str) -> Optional[Dict[str, Any]]:
        """Get node data for a concept."""
        # Search knowledge base for this concept
        results = self.knowledge_base.query_knowledge(concept, max_results=1)
        
        if not results:
            return None
            
        item = results[0]
        return {
            'id': concept,
            'label': concept,
            'type': item.knowledge_type.value,
            'confidence': item.confidence,
            'source': item.source,
            'size': min(100, max(10, item.confidence * 50)),  # Node size based on confidence
            'color': self._get_node_color(item.knowledge_type.value)
        }
    
    def _find_related_concepts(self, concept: str) -> List[Tuple[str, float]]:
        """Find concepts related to the given concept."""
        # This is a simplified implementation
        # In a real system, you'd use more sophisticated NLP techniques
        
        related = []
        results = self.knowledge_base.query_knowledge(concept, max_results=10)
        
        for item in results[1:]:  # Skip the first match (same concept)
            # Calculate semantic similarity
            strength = self._calculate_semantic_similarity(concept, str(item.content))
            if strength > 0.3:  # Threshold for relevance
                related.append((self._extract_key_concept(str(item.content)), strength))
                
        return sorted(related, key=lambda x: x[1], reverse=True)[:5]
    
    def _calculate_semantic_similarity(self, concept1: str, text2: str) -> float:
        """Calculate semantic similarity between concept and text."""
        # Simple word overlap similarity
        words1 = set(concept1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _extract_key_concept(self, text: str) -> str:
        """Extract key concept from text."""
        # Simple extraction - take first few significant words
        words = text.split()[:3]
        return ' '.join(words).lower()
    
    def _get_node_color(self, knowledge_type: str) -> str:
        """Get color for node based on knowledge type."""
        colors = {
            'factual': '#4CAF50',      # Green
            'procedural': '#2196F3',   # Blue  
            'experiential': '#FF9800', # Orange
            'pattern': '#9C27B0',      # Purple
            'skill': '#F44336',        # Red
        }
        return colors.get(knowledge_type, '#757575')  # Default gray
    
    def _generate_overview_graph(self) -> Tuple[List[Dict], List[Dict]]:
        """Generate overview graph of all knowledge."""
        nodes = []
        edges = []
        
        # Group knowledge by type and source
        type_counts = defaultdict(int)
        source_counts = defaultdict(int)
        
        # Get sample from knowledge base
        sample_size = min(50, len(self.knowledge_base.knowledge_store))
        knowledge_items = list(self.knowledge_base.knowledge_store.values())[:sample_size]
        
        for item in knowledge_items:
            type_counts[item.knowledge_type.value] += 1
            source_counts[item.source] += 1
            
        # Create type nodes
        for ktype, count in type_counts.items():
            nodes.append({
                'id': f'type_{ktype}',
                'label': f'{ktype.title()} ({count})',
                'type': 'category',
                'size': min(100, max(20, count * 5)),
                'color': self._get_node_color(ktype)
            })
            
        # Create source nodes and connect to types
        for source, count in source_counts.items():
            source_id = f'source_{source}'
            nodes.append({
                'id': source_id,
                'label': f'{source} ({count})',
                'type': 'source',
                'size': min(80, max(15, count * 3)),
                'color': '#607D8B'  # Blue gray
            })
            
            # Connect sources to types they contribute to
            for item in knowledge_items:
                if item.source == source:
                    edges.append({
                        'source': source_id,
                        'target': f'type_{item.knowledge_type.value}',
                        'strength': 0.5,
                        'type': 'contribution'
                    })
                    
        return nodes, edges
    
    def get_concept_details(self, concept: str) -> Dict[str, Any]:
        """Get detailed information about a specific concept."""
        results = self.knowledge_base.query_knowledge(concept, max_results=5)
        
        if not results:
            return {'error': 'Concept not found'}
            
        return {
            'concept': concept,
            'primary_info': {
                'type': results[0].knowledge_type.value,
                'confidence': results[0].confidence,
                'source': results[0].source,
                'content': str(results[0].content)[:500]
            },
            'related_items': [{
                'content': str(item.content)[:200],
                'confidence': item.confidence,
                'source': item.source
            } for item in results[1:]],
            'connections': self._find_related_concepts(concept)
        }
