"""
Prometheus metrics collection.
"""

import logging
from typing import Optional
from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Collects and exposes Prometheus metrics.
    
    Metrics tracked:
    - Request count and latency
    - Model usage and performance
    - Error rates
    - Cost metrics
    - Cache hit/miss rates
    """
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """
        Initialize metrics collector.
        
        Args:
            registry: Optional Prometheus registry
        """
        self.registry = registry or CollectorRegistry()
        
        # Request metrics
        self.request_count = Counter(
            'thenexus_requests_total',
            'Total number of requests',
            ['endpoint', 'method', 'status'],
            registry=self.registry
        )
        
        self.request_latency = Histogram(
            'thenexus_request_duration_seconds',
            'Request latency in seconds',
            ['endpoint', 'method'],
            registry=self.registry
        )
        
        # Model metrics
        self.model_requests = Counter(
            'thenexus_model_requests_total',
            'Total requests per model',
            ['model_name', 'provider'],
            registry=self.registry
        )
        
        self.model_latency = Histogram(
            'thenexus_model_latency_milliseconds',
            'Model response latency in milliseconds',
            ['model_name', 'provider'],
            registry=self.registry
        )
        
        self.model_tokens = Counter(
            'thenexus_model_tokens_total',
            'Total tokens used per model',
            ['model_name', 'provider'],
            registry=self.registry
        )
        
        self.model_errors = Counter(
            'thenexus_model_errors_total',
            'Total model errors',
            ['model_name', 'provider', 'error_type'],
            registry=self.registry
        )
        
        # Cost metrics
        self.total_cost = Counter(
            'thenexus_cost_usd_total',
            'Total cost in USD',
            ['model_name', 'provider'],
            registry=self.registry
        )
        
        self.monthly_budget = Gauge(
            'thenexus_monthly_budget_usd',
            'Monthly budget limit in USD',
            registry=self.registry
        )
        
        self.budget_remaining = Gauge(
            'thenexus_budget_remaining_usd',
            'Remaining budget in USD',
            registry=self.registry
        )
        
        # Cache metrics
        self.cache_hits = Counter(
            'thenexus_cache_hits_total',
            'Total cache hits',
            registry=self.registry
        )
        
        self.cache_misses = Counter(
            'thenexus_cache_misses_total',
            'Total cache misses',
            registry=self.registry
        )
        
        self.cache_size = Gauge(
            'thenexus_cache_size_bytes',
            'Current cache size in bytes',
            registry=self.registry
        )
        
        # Ensemble metrics
        self.ensemble_models_count = Gauge(
            'thenexus_ensemble_models_total',
            'Number of models in ensemble',
            registry=self.registry
        )
        
        self.ensemble_score = Histogram(
            'thenexus_ensemble_score',
            'Ensemble response scores',
            ['model_name'],
            registry=self.registry
        )
        
        # System info
        self.info = Info(
            'thenexus',
            'TheNexus system information',
            registry=self.registry
        )
        
        logger.info("MetricsCollector initialized")
    
    def record_request(self, endpoint: str, method: str, status: int, duration: float):
        """
        Record an HTTP request.
        
        Args:
            endpoint: API endpoint
            method: HTTP method
            status: HTTP status code
            duration: Request duration in seconds
        """
        self.request_count.labels(
            endpoint=endpoint,
            method=method,
            status=str(status)
        ).inc()
        
        self.request_latency.labels(
            endpoint=endpoint,
            method=method
        ).observe(duration)
    
    def record_model_request(
        self,
        model_name: str,
        provider: str,
        latency_ms: float,
        tokens_used: int,
        cost_usd: float,
        success: bool = True,
        error_type: Optional[str] = None
    ):
        """
        Record a model request.
        
        Args:
            model_name: Name of the model
            provider: Provider name
            latency_ms: Response latency in milliseconds
            tokens_used: Number of tokens used
            cost_usd: Cost in USD
            success: Whether request succeeded
            error_type: Type of error if failed
        """
        self.model_requests.labels(
            model_name=model_name,
            provider=provider
        ).inc()
        
        self.model_latency.labels(
            model_name=model_name,
            provider=provider
        ).observe(latency_ms)
        
        self.model_tokens.labels(
            model_name=model_name,
            provider=provider
        ).inc(tokens_used)
        
        self.total_cost.labels(
            model_name=model_name,
            provider=provider
        ).inc(cost_usd)
        
        if not success and error_type:
            self.model_errors.labels(
                model_name=model_name,
                provider=provider,
                error_type=error_type
            ).inc()
    
    def record_cache_hit(self):
        """Record a cache hit."""
        self.cache_hits.inc()
    
    def record_cache_miss(self):
        """Record a cache miss."""
        self.cache_misses.inc()
    
    def update_budget_metrics(self, budget_limit: float, current_spend: float):
        """
        Update budget metrics.
        
        Args:
            budget_limit: Monthly budget limit
            current_spend: Current spend this month
        """
        self.monthly_budget.set(budget_limit)
        self.budget_remaining.set(max(0, budget_limit - current_spend))
    
    def update_ensemble_size(self, count: int):
        """
        Update ensemble model count.
        
        Args:
            count: Number of models
        """
        self.ensemble_models_count.set(count)
    
    def record_ensemble_score(self, model_name: str, score: float):
        """
        Record an ensemble score.
        
        Args:
            model_name: Name of the model
            score: Response score
        """
        self.ensemble_score.labels(model_name=model_name).observe(score)
    
    def set_system_info(self, version: str, python_version: str, models: list):
        """
        Set system information.
        
        Args:
            version: TheNexus version
            python_version: Python version
            models: List of available models
        """
        self.info.info({
            'version': version,
            'python_version': python_version,
            'models': ','.join(models),
        })
