
"""
Internet Knowledge Retrieval System for Nexus AI Platform.
Fetches information from the web to expand the knowledge base.
"""

import logging
import requests
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from urllib.parse import quote_plus
from nexus.memory.knowledge_base import KnowledgeBase, KnowledgeType

logger = logging.getLogger(__name__)

class InternetKnowledgeRetriever:
    """
    System for retrieving knowledge from the internet to fill knowledge gaps.
    """
    
    def __init__(self, knowledge_base: KnowledgeBase, ensemble_core=None):
        self.knowledge_base = knowledge_base
        self.ensemble_core = ensemble_core
        self.search_engines = {
            'duckduckgo': self._search_duckduckgo,
            'wikipedia': self._search_wikipedia,
            'arxiv': self._search_arxiv,
            'news': self._search_news
        }
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour
        self.rate_limit_delay = 0.5  # 0.5 seconds between requests for faster learning
        self.last_request_time = 0
        self.proactive_search_enabled = True
        self.validation_threshold = 0.7  # Ensemble confidence threshold for auto-acceptance
        
    def retrieve_knowledge_for_query(self, query: str, max_results: int = 3) -> Dict[str, Any]:
        """Retrieve knowledge from the internet for a given query."""
        logger.info(f"Retrieving internet knowledge for query: {query}")
        
        # Check cache first
        if query in self.cache:
            cached_result, cached_time = self.cache[query]
            if time.time() - cached_time < self.cache_ttl:
                logger.info(f"Returning cached result for query: {query}")
                return cached_result
        
        # Rate limiting
        current_time = time.time()
        if current_time - self.last_request_time < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - (current_time - self.last_request_time))
        
        results = {
            'query': query,
            'sources': [],
            'facts_found': [],
            'confidence_scores': [],
            'timestamp': datetime.now().isoformat(),
            'success': False
        }
        
        try:
            # Try different search engines
            for engine_name, search_func in self.search_engines.items():
                try:
                    engine_results = search_func(query, max_results)
                    if engine_results and engine_results.get('success'):
                        results['sources'].append({
                            'engine': engine_name,
                            'results': engine_results.get('results', [])
                        })
                        
                        # Extract facts from results
                        facts = self._extract_facts_from_results(engine_results.get('results', []))
                        results['facts_found'].extend(facts)
                        
                        if facts:
                            results['success'] = True
                            break  # Use first successful engine
                            
                except Exception as e:
                    logger.warning(f"Error with {engine_name}: {e}")
                    continue
            
            # Process and score the facts
            if results['facts_found']:
                results['confidence_scores'] = self._score_facts(results['facts_found'], query)
                
                # Add high-confidence facts to knowledge base
                added_facts = self._add_facts_to_knowledge_base(
                    results['facts_found'], 
                    results['confidence_scores'], 
                    query
                )
                results['added_to_kb'] = added_facts
            
            # Cache the result
            self.cache[query] = (results, time.time())
            self.last_request_time = time.time()
            
        except Exception as e:
            logger.error(f"Error retrieving knowledge for query '{query}': {e}")
            results['error'] = str(e)
        
        return results
    
    def _search_duckduckgo(self, query: str, max_results: int = 3) -> Dict[str, Any]:
        """Search using DuckDuckGo Instant Answer API."""
        try:
            # DuckDuckGo Instant Answer API
            url = f"https://api.duckduckgo.com/"
            params = {
                'q': query,
                'format': 'json',
                'no_html': '1',
                'skip_disambig': '1'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            results = []
            
            # Extract instant answer
            if data.get('Answer'):
                results.append({
                    'type': 'instant_answer',
                    'content': data['Answer'],
                    'source': 'DuckDuckGo Instant Answer',
                    'confidence': 0.9
                })
            
            # Extract abstract
            if data.get('Abstract'):
                results.append({
                    'type': 'abstract',
                    'content': data['Abstract'],
                    'source': data.get('AbstractSource', 'DuckDuckGo'),
                    'confidence': 0.8
                })
            
            # Extract definition
            if data.get('Definition'):
                results.append({
                    'type': 'definition',
                    'content': data['Definition'],
                    'source': data.get('DefinitionSource', 'DuckDuckGo'),
                    'confidence': 0.85
                })
            
            return {
                'success': True,
                'results': results[:max_results],
                'raw_data': data
            }
            
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _search_wikipedia(self, query: str, max_results: int = 3) -> Dict[str, Any]:
        """Search Wikipedia for information."""
        try:
            # Wikipedia API search
            search_url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + quote_plus(query)
            
            response = requests.get(search_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                results = [{
                    'type': 'encyclopedia',
                    'content': data.get('extract', ''),
                    'title': data.get('title', ''),
                    'source': 'Wikipedia',
                    'url': data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                    'confidence': 0.85
                }]
                
                return {
                    'success': True,
                    'results': results,
                    'raw_data': data
                }
            else:
                # Fallback to search API
                search_api_url = "https://en.wikipedia.org/api/rest_v1/page/related/" + quote_plus(query)
                response = requests.get(search_api_url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    results = []
                    
                    for page in data.get('pages', [])[:max_results]:
                        results.append({
                            'type': 'encyclopedia',
                            'content': page.get('extract', ''),
                            'title': page.get('title', ''),
                            'source': 'Wikipedia',
                            'url': page.get('content_urls', {}).get('desktop', {}).get('page', ''),
                            'confidence': 0.8
                        })
                    
                    return {
                        'success': True,
                        'results': results,
                        'raw_data': data
                    }
                    
        except Exception as e:
            logger.error(f"Wikipedia search error: {e}")
            
        return {'success': False, 'error': 'Wikipedia search failed'}
    
    def _extract_facts_from_results(self, results: List[Dict]) -> List[str]:
        """Extract factual statements from search results."""
        facts = []
        
        for result in results:
            content = result.get('content', '')
            if not content:
                continue
            
            # Simple fact extraction - split by sentences and filter
            sentences = content.split('. ')
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20 and len(sentence) < 200:  # Reasonable fact length
                    # Basic quality filters
                    if any(word in sentence.lower() for word in ['is', 'are', 'was', 'were', 'has', 'have']):
                        if not any(word in sentence.lower() for word in ['maybe', 'possibly', 'might', 'could be']):
                            facts.append(sentence)
        
        return facts[:10]  # Limit to 10 facts per query
    
    def _score_facts(self, facts: List[str], query: str) -> List[float]:
        """Score facts based on relevance and confidence."""
        scores = []
        query_terms = set(query.lower().split())
        
        for fact in facts:
            score = 0.5  # Base score
            fact_lower = fact.lower()
            fact_terms = set(fact_lower.split())
            
            # Relevance scoring
            overlap = len(query_terms.intersection(fact_terms))
            if overlap > 0:
                score += min(0.4, overlap * 0.1)
            
            # Quality indicators
            if any(indicator in fact_lower for indicator in ['according to', 'research shows', 'studies indicate']):
                score += 0.1
            
            # Specificity bonus
            if any(char.isdigit() for char in fact):
                score += 0.05
            
            # Length penalty for very long facts
            if len(fact) > 150:
                score -= 0.1
            
            scores.append(min(1.0, max(0.1, score)))
        
        return scores
    
    def _add_facts_to_knowledge_base(self, facts: List[str], scores: List[float], query: str) -> List[str]:
        """Add high-confidence facts to the knowledge base."""
        added_facts = []
        
        for fact, score in zip(facts, scores):
            if score >= 0.7:  # Only add high-confidence facts
                try:
                    knowledge_id = self.knowledge_base.add_knowledge(
                        content=fact,
                        knowledge_type=KnowledgeType.FACTUAL,
                        source=f"internet_search:{query}",
                        confidence=score,
                        context_tags=['internet_retrieved', query.lower()]
                    )
                    added_facts.append(knowledge_id)
                    logger.info(f"Added internet fact to KB: {knowledge_id}")
                    
                except Exception as e:
                    logger.error(f"Error adding fact to knowledge base: {e}")
        
        return added_facts
    
    def proactive_knowledge_search(self, topics: List[str], max_results_per_topic: int = 5) -> Dict[str, Any]:
        """Proactively search for knowledge on multiple topics."""
        logger.info(f"Starting proactive search for {len(topics)} topics")
        
        results = {
            'topics_searched': topics,
            'total_facts_found': 0,
            'validated_facts': 0,
            'rejected_facts': 0,
            'search_results': {}
        }
        
        for topic in topics:
            topic_results = self.retrieve_knowledge_for_query(topic, max_results_per_topic)
            results['search_results'][topic] = topic_results
            
            if topic_results.get('success'):
                # Validate facts using ensemble
                for fact in topic_results.get('facts_found', []):
                    validation_result = self._validate_fact_with_ensemble(fact, topic)
                    
                    if validation_result['accepted']:
                        results['validated_facts'] += 1
                    else:
                        results['rejected_facts'] += 1
                        
                results['total_facts_found'] += len(topic_results.get('facts_found', []))
        
        return results
    
    def _validate_fact_with_ensemble(self, fact: str, context: str) -> Dict[str, Any]:
        """Validate a fact using the ensemble for truthfulness assessment."""
        if not self.ensemble_core:
            # If no ensemble available, use basic validation
            return {'accepted': True, 'confidence': 0.5, 'reason': 'no_ensemble_validation'}
        
        # Create validation prompt
        validation_prompt = f"Assess the truthfulness and accuracy of this statement: '{fact}' in the context of '{context}'. Is this factually correct?"
        
        try:
            ensemble_result = self.ensemble_core.predict(validation_prompt)
            confidence = ensemble_result.confidence
            prediction = str(ensemble_result.prediction).lower()
            
            # Determine if fact should be accepted
            accepted = (confidence >= self.validation_threshold and 
                       any(word in prediction for word in ['true', 'correct', 'accurate', 'yes']))
            
            logger.info(f"Ensemble validation - Fact: {fact[:50]}... | Accepted: {accepted} | Confidence: {confidence}")
            
            return {
                'accepted': accepted,
                'confidence': confidence,
                'ensemble_prediction': ensemble_result.prediction,
                'reason': f'ensemble_validation_confidence_{confidence:.2f}'
            }
            
        except Exception as e:
            logger.error(f"Error in ensemble validation: {e}")
            return {'accepted': False, 'confidence': 0.0, 'reason': f'validation_error_{str(e)}'}
    
    def _search_arxiv(self, query: str, max_results: int = 3) -> Dict[str, Any]:
        """Search arXiv for academic papers."""
        try:
            # arXiv API search
            url = "http://export.arxiv.org/api/query"
            params = {
                'search_query': f'all:{query}',
                'start': 0,
                'max_results': max_results,
                'sortBy': 'relevance'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            results = []
            # Basic XML parsing (could be enhanced with proper XML parser)
            content = response.text
            if '<entry>' in content:
                entries = content.split('<entry>')[1:max_results+1]
                for entry in entries:
                    if '<title>' in entry and '<summary>' in entry:
                        title_start = entry.find('<title>') + 7
                        title_end = entry.find('</title>')
                        title = entry[title_start:title_end].strip() if title_end > title_start else ''
                        
                        summary_start = entry.find('<summary>') + 9
                        summary_end = entry.find('</summary>')
                        summary = entry[summary_start:summary_end].strip() if summary_end > summary_start else ''
                        
                        if title and summary:
                            results.append({
                                'type': 'academic_paper',
                                'title': title,
                                'content': summary[:500],  # Limit summary length
                                'source': 'arXiv',
                                'confidence': 0.85
                            })
            
            return {
                'success': True,
                'results': results,
                'raw_data': {'query': query, 'total_results': len(results)}
            }
            
        except Exception as e:
            logger.error(f"arXiv search error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _search_news(self, query: str, max_results: int = 3) -> Dict[str, Any]:
        """Search for recent news (using a simple news aggregator approach)."""
        try:
            # Note: In a production system, you'd use a proper news API
            # This is a simplified approach using search engines
            news_query = f"{query} news recent"
            return self._search_duckduckgo(news_query, max_results)
            
        except Exception as e:
            logger.error(f"News search error: {e}")
            return {'success': False, 'error': str(e)}
    
    def continuous_learning_mode(self, learning_topics: List[str] = None) -> Dict[str, Any]:
        """Enable continuous learning mode with proactive searches."""
        if not learning_topics:
            learning_topics = [
                'artificial intelligence', 'machine learning', 'technology news',
                'scientific discoveries', 'programming', 'data science',
                'current events', 'research breakthroughs'
            ]
        
        logger.info(f"Starting continuous learning mode with {len(learning_topics)} topics")
        
        return self.proactive_knowledge_search(learning_topics, max_results_per_topic=3)
    
    def get_retrieval_statistics(self) -> Dict[str, Any]:
        """Get statistics about internet knowledge retrieval."""
        return {
            'cache_size': len(self.cache),
            'cache_ttl': self.cache_ttl,
            'rate_limit_delay': self.rate_limit_delay,
            'available_engines': list(self.search_engines.keys()),
            'last_request_time': self.last_request_time,
            'proactive_search_enabled': self.proactive_search_enabled,
            'validation_threshold': self.validation_threshold
        }
