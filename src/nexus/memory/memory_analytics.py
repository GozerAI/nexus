
"""
Advanced memory analytics and optimization system.
"""

import logging
import time
import statistics
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class MemoryInsight:
    """Represents an insight about memory usage."""
    insight_type: str
    description: str
    severity: str  # 'info', 'warning', 'critical'
    recommendation: str
    metrics: Dict[str, Any]

class MemoryAnalytics:
    """
    Advanced analytics for memory system performance and optimization.
    """
    
    def __init__(self, memory_manager, knowledge_base):
        self.memory_manager = memory_manager
        self.knowledge_base = knowledge_base
        self.analytics_history = []
        self.optimization_log = []
        
    def analyze_memory_patterns(self) -> Dict[str, Any]:
        """Perform comprehensive memory pattern analysis."""
        analysis_timestamp = time.time()
        
        # Collect basic statistics
        memory_stats = self.memory_manager.get_memory_statistics()
        knowledge_stats = self.knowledge_base.get_knowledge_statistics()
        
        # Analyze access patterns
        access_patterns = self._analyze_access_patterns()
        
        # Detect memory hotspots
        hotspots = self._detect_memory_hotspots()
        
        # Analyze knowledge distribution
        knowledge_distribution = self._analyze_knowledge_distribution()
        
        # Generate insights
        insights = self._generate_insights(memory_stats, knowledge_stats, access_patterns, hotspots)
        
        analysis_result = {
            'timestamp': analysis_timestamp,
            'memory_efficiency': self._calculate_memory_efficiency(),
            'knowledge_quality_score': self._calculate_knowledge_quality_score(),
            'access_patterns': access_patterns,
            'memory_hotspots': hotspots,
            'knowledge_distribution': knowledge_distribution,
            'insights': [self._insight_to_dict(insight) for insight in insights],
            'optimization_recommendations': self._generate_optimization_recommendations()
        }
        
        # Store in history
        self.analytics_history.append(analysis_result)
        
        # Keep only recent history
        if len(self.analytics_history) > 100:
            self.analytics_history = self.analytics_history[-100:]
            
        return analysis_result
    
    def _analyze_access_patterns(self) -> Dict[str, Any]:
        """Analyze how memory is being accessed."""
        # Get memory blocks and their access patterns
        memory_blocks = self.memory_manager.blocks
        
        access_frequency = Counter()
        access_recency = {}
        block_sizes = []
        
        current_time = time.time()
        
        for block_id, block in memory_blocks.items():
            # Simulate access tracking (in real implementation, this would be tracked)
            last_access = getattr(block, 'last_accessed', current_time)
            access_count = getattr(block, 'access_count', 1)
            
            access_frequency[block_id] = access_count
            access_recency[block_id] = current_time - last_access
            block_sizes.append(len(str(block.content)) if hasattr(block, 'content') else 100)
        
        return {
            'most_accessed_blocks': access_frequency.most_common(10),
            'least_recently_used': sorted(access_recency.items(), key=lambda x: x[1], reverse=True)[:10],
            'average_block_size': statistics.mean(block_sizes) if block_sizes else 0,
            'total_blocks': len(memory_blocks),
            'size_distribution': {
                'small': len([s for s in block_sizes if s < 100]),
                'medium': len([s for s in block_sizes if 100 <= s < 1000]),
                'large': len([s for s in block_sizes if s >= 1000])
            }
        }
    
    def _detect_memory_hotspots(self) -> List[Dict[str, Any]]:
        """Detect memory usage hotspots."""
        hotspots = []
        
        # Analyze knowledge base categories
        category_counts = defaultdict(int)
        source_counts = defaultdict(int)
        confidence_ranges = defaultdict(int)
        
        for item in self.knowledge_base.knowledge_store.values():
            # Group by type (simulated category)
            category = item.knowledge_type.value
            category_counts[category] += 1
            
            # Group by source
            source_counts[item.source] += 1
            
            # Group by confidence range
            conf_range = self._get_confidence_range(item.confidence)
            confidence_ranges[conf_range] += 1
        
        # Create hotspot analysis
        if category_counts:
            max_category = max(category_counts, key=category_counts.get)
            hotspots.append({
                'type': 'knowledge_category',
                'category': max_category,
                'count': category_counts[max_category],
                'percentage': (category_counts[max_category] / sum(category_counts.values())) * 100
            })
        
        if source_counts:
            max_source = max(source_counts, key=source_counts.get)
            hotspots.append({
                'type': 'knowledge_source',
                'source': max_source,
                'count': source_counts[max_source],
                'percentage': (source_counts[max_source] / sum(source_counts.values())) * 100
            })
            
        return hotspots
    
    def _analyze_knowledge_distribution(self) -> Dict[str, Any]:
        """Analyze distribution of knowledge across different dimensions."""
        distribution = {
            'by_type': defaultdict(int),
            'by_source': defaultdict(int),
            'by_confidence': defaultdict(int),
            'by_age': defaultdict(int)
        }
        
        current_time = time.time()
        
        for item in self.knowledge_base.knowledge_store.values():
            distribution['by_type'][item.knowledge_type.value] += 1
            distribution['by_source'][item.source] += 1
            distribution['by_confidence'][self._get_confidence_range(item.confidence)] += 1
            
            # Age analysis (simulated)
            age_days = (current_time - getattr(item, 'created_at', current_time)) / (24 * 3600)
            age_category = self._get_age_category(age_days)
            distribution['by_age'][age_category] += 1
        
        # Convert to regular dicts for JSON serialization
        return {k: dict(v) for k, v in distribution.items()}
    
    def _generate_insights(self, memory_stats: Dict, knowledge_stats: Dict, 
                          access_patterns: Dict, hotspots: List) -> List[MemoryInsight]:
        """Generate actionable insights from analysis."""
        insights = []
        
        # Memory efficiency insights
        total_blocks = memory_stats.get('total_blocks', 0)
        if total_blocks > 5000:
            insights.append(MemoryInsight(
                insight_type='memory_usage',
                description=f'High memory block count: {total_blocks} blocks',
                severity='warning',
                recommendation='Consider implementing memory cleanup policies',
                metrics={'block_count': total_blocks, 'threshold': 5000}
            ))
        
        # Access pattern insights
        avg_block_size = access_patterns.get('average_block_size', 0)
        if avg_block_size > 1000:
            insights.append(MemoryInsight(
                insight_type='block_size',
                description=f'Large average block size: {avg_block_size:.0f} characters',
                severity='info',
                recommendation='Consider breaking down large knowledge blocks',
                metrics={'avg_size': avg_block_size, 'recommended_max': 1000}
            ))
        
        # Knowledge quality insights
        knowledge_items = len(self.knowledge_base.knowledge_store)
        if knowledge_items > 0:
            avg_confidence = statistics.mean(item.confidence for item in self.knowledge_base.knowledge_store.values())
            if avg_confidence < 0.6:
                insights.append(MemoryInsight(
                    insight_type='knowledge_quality',
                    description=f'Low average knowledge confidence: {avg_confidence:.2f}',
                    severity='warning',
                    recommendation='Review and validate low-confidence knowledge items',
                    metrics={'avg_confidence': avg_confidence, 'threshold': 0.6}
                ))
        
        return insights
    
    def _calculate_memory_efficiency(self) -> float:
        """Calculate overall memory efficiency score."""
        # Simplified efficiency calculation
        memory_stats = self.memory_manager.get_memory_statistics()
        total_blocks = memory_stats.get('total_blocks', 1)
        
        # Factor in various efficiency metrics
        block_efficiency = min(1.0, 1000 / total_blocks)  # Efficient if < 1000 blocks
        
        # Knowledge utilization efficiency
        knowledge_count = len(self.knowledge_base.knowledge_store)
        utilization_efficiency = min(1.0, knowledge_count / max(1, total_blocks))
        
        return (block_efficiency + utilization_efficiency) / 2
    
    def _calculate_knowledge_quality_score(self) -> float:
        """Calculate overall knowledge quality score."""
        if not self.knowledge_base.knowledge_store:
            return 0.0
            
        confidences = [item.confidence for item in self.knowledge_base.knowledge_store.values()]
        avg_confidence = statistics.mean(confidences)
        
        # Factor in confidence distribution
        high_conf_count = len([c for c in confidences if c > 0.8])
        quality_distribution = high_conf_count / len(confidences)
        
        return (avg_confidence + quality_distribution) / 2
    
    def _generate_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Generate specific optimization recommendations."""
        recommendations = []
        
        # Analyze current state
        memory_stats = self.memory_manager.get_memory_statistics()
        total_blocks = memory_stats.get('total_blocks', 0)
        
        if total_blocks > 3000:
            recommendations.append({
                'type': 'cleanup',
                'priority': 'high',
                'action': 'Implement memory cleanup for old, unused blocks',
                'expected_benefit': 'Reduce memory usage by 20-30%',
                'implementation': 'Add LRU eviction policy'
            })
        
        # Knowledge quality recommendations
        if self.knowledge_base.knowledge_store:
            low_conf_count = len([item for item in self.knowledge_base.knowledge_store.values() 
                                if item.confidence < 0.5])
            if low_conf_count > 10:
                recommendations.append({
                    'type': 'quality_improvement',
                    'priority': 'medium',
                    'action': f'Review {low_conf_count} low-confidence knowledge items',
                    'expected_benefit': 'Improve overall knowledge quality score',
                    'implementation': 'Add knowledge validation workflow'
                })
        
        return recommendations
    
    def _get_confidence_range(self, confidence: float) -> str:
        """Categorize confidence into ranges."""
        if confidence < 0.3:
            return 'low'
        elif confidence < 0.7:
            return 'medium'
        else:
            return 'high'
    
    def _get_age_category(self, age_days: float) -> str:
        """Categorize knowledge by age."""
        if age_days < 1:
            return 'new'
        elif age_days < 7:
            return 'recent'
        elif age_days < 30:
            return 'older'
        else:
            return 'old'
    
    def _insight_to_dict(self, insight: MemoryInsight) -> Dict[str, Any]:
        """Convert insight to dictionary."""
        return {
            'type': insight.insight_type,
            'description': insight.description,
            'severity': insight.severity,
            'recommendation': insight.recommendation,
            'metrics': insight.metrics
        }
    
    def get_analytics_trends(self, days: int = 7) -> Dict[str, Any]:
        """Get analytics trends over time."""
        cutoff_time = time.time() - (days * 24 * 3600)
        recent_analytics = [a for a in self.analytics_history if a['timestamp'] > cutoff_time]
        
        if len(recent_analytics) < 2:
            return {'message': 'Insufficient data for trend analysis'}
        
        # Calculate trends
        efficiency_trend = [a['memory_efficiency'] for a in recent_analytics]
        quality_trend = [a['knowledge_quality_score'] for a in recent_analytics]
        
        return {
            'period_days': days,
            'data_points': len(recent_analytics),
            'efficiency_trend': {
                'current': efficiency_trend[-1],
                'change': efficiency_trend[-1] - efficiency_trend[0],
                'average': statistics.mean(efficiency_trend)
            },
            'quality_trend': {
                'current': quality_trend[-1],
                'change': quality_trend[-1] - quality_trend[0],
                'average': statistics.mean(quality_trend)
            }
        }
