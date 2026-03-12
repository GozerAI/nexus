"""
Unified configuration system integrating settings from all component systems.
"""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Deployment environment."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class EnsembleStrategy(str, Enum):
    """Ensemble selection strategies from TheNexus."""

    WEIGHTED_VOTING = "weighted_voting"
    CASCADING = "cascading"
    DYNAMIC_WEIGHT = "dynamic_weight"
    MAJORITY_VOTING = "majority_voting"
    COST_OPTIMIZED = "cost_optimized"
    SYNTHESIZED = "synthesized"  # From combo1


class ModelConfig(BaseSettings):
    """Configuration for AI model providers."""

    model_config = SettingsConfigDict(env_prefix="MODEL_")

    # OpenAI
    openai_api_key: Optional[str] = None
    openai_models: List[str] = Field(default_factory=lambda: ["gpt-4", "gpt-3.5-turbo"])

    # Anthropic
    anthropic_api_key: Optional[str] = None
    anthropic_models: List[str] = Field(default_factory=lambda: ["claude-3-opus", "claude-3-sonnet"])

    # Local models
    local_models_enabled: bool = False
    ollama_base_url: str = "http://localhost:11434"
    ollama_models: List[str] = Field(default_factory=list)

    # Model settings
    default_temperature: float = 0.7
    max_tokens: int = 2048
    timeout_seconds: int = 30


class EnsembleConfig(BaseSettings):
    """Ensemble orchestration configuration."""

    model_config = SettingsConfigDict(env_prefix="ENSEMBLE_")

    # Strategy settings
    default_strategy: EnsembleStrategy = EnsembleStrategy.SYNTHESIZED
    enable_response_synthesis: bool = True  # From combo1
    enable_confidence_calibration: bool = True  # From combo1
    enable_epistemic_monitoring: bool = True  # From fluffy-eureka

    # Performance settings
    parallel_execution: bool = True
    max_parallel_models: int = 10
    enable_streaming: bool = True

    # Quality thresholds
    min_confidence_threshold: float = 0.65
    min_quality_score: float = 0.7

    # Cost optimization
    enable_cost_tracking: bool = True
    budget_limit_usd: Optional[float] = None
    alert_threshold_percent: float = 80.0


class MemoryConfig(BaseSettings):
    """Unified memory system configuration."""

    model_config = SettingsConfigDict(env_prefix="MEMORY_")

    # Storage backend
    backend: str = "postgresql"  # postgresql, sqlite, hybrid
    postgresql_url: Optional[str] = None
    sqlite_path: Path = Path("./data/memory.db")

    # Memory architecture
    enable_partitioned_memory: bool = True  # From fluffy-eureka
    enable_hierarchical_tiers: bool = True  # From nexus-system
    enable_semantic_search: bool = True  # From combo1

    # Partitions (fluffy-eureka)
    factual_memory_enabled: bool = True
    skill_memory_enabled: bool = True

    # Tiers (nexus-system)
    hot_tier_size_mb: int = 100
    warm_tier_size_mb: int = 500
    cold_tier_enabled: bool = True

    # Vector search
    vector_backend: str = "faiss"  # faiss, milvus, chroma
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # Cache settings
    enable_redis_cache: bool = True
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 300


class LearningConfig(BaseSettings):
    """Self-directed learning configuration."""

    model_config = SettingsConfigDict(env_prefix="LEARNING_")

    # Meta-learning (combo1)
    enable_meta_learning: bool = True
    learning_rate: float = 0.01
    feedback_buffer_size: int = 10000

    # Autonomous learning (nexus-system)
    enable_autonomous_learning: bool = True
    learning_modes: List[str] = Field(
        default_factory=lambda: [
            "exploration",
            "exploitation",
            "curiosity_driven",
            "goal_oriented",
        ]
    )

    # Skeptical learning (fluffy-eureka)
    enable_skeptical_learning: bool = True
    belief_threshold: float = 0.7
    evidence_accumulation_required: int = 3

    # A/B testing (combo1)
    enable_ab_testing: bool = True
    ab_test_algorithm: str = "thompson_sampling"  # epsilon_greedy, ucb, thompson_sampling


class MonitoringConfig(BaseSettings):
    """Monitoring and observability configuration."""

    model_config = SettingsConfigDict(env_prefix="MONITORING_")

    # Metrics
    enable_prometheus: bool = True
    prometheus_port: int = 9090
    metrics_prefix: str = "unified_intelligence"

    # Epistemic monitoring (fluffy-eureka)
    enable_drift_detection: bool = True
    drift_check_interval_seconds: int = 60
    drift_threshold: float = 0.3

    # Logging
    log_level: LogLevel = LogLevel.INFO
    structured_logging: bool = True
    log_format: str = "json"  # json, text

    # Tracing
    enable_distributed_tracing: bool = False
    jaeger_endpoint: Optional[str] = None


class SafetyConfig(BaseSettings):
    """Safety and governance configuration."""

    model_config = SettingsConfigDict(env_prefix="SAFETY_")

    # Governance (psychic-bassoon)
    enable_policy_enforcement: bool = True
    policy_file: Path = Path("./config/governance_policy.yaml")

    # Safety monitors (nexus-system)
    enable_safety_monitor: bool = True
    enable_virtue_learning: bool = True

    # Quarantine (fluffy-eureka, combo1)
    enable_quarantine_system: bool = True
    quarantine_threshold_failures: int = 5
    quarantine_cooldown_seconds: int = 300

    # Circuit breakers (combo1, psychic-bassoon)
    enable_circuit_breakers: bool = True
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_timeout_seconds: int = 60

    # Verification (fluffy-eureka, combo1)
    enable_verification_layer: bool = True
    require_fact_checking: bool = False


class APIConfig(BaseSettings):
    """API server configuration."""

    model_config = SettingsConfigDict(env_prefix="API_")

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = False

    # Authentication
    enable_auth: bool = True
    jwt_secret_key: Optional[str] = None
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 60

    # Rate limiting
    enable_rate_limiting: bool = True
    rate_limit_per_minute: int = 100
    rate_limit_burst: int = 150

    # CORS
    enable_cors: bool = True
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])


class UnifiedConfig(BaseSettings):
    """
    Unified configuration system integrating all component configurations.

    This combines settings from:
    - TheNexus (ensemble strategies, cost tracking)
    - 4cast (forecasting, data processing)
    - nexus-system (consciousness, autonomous learning)
    - fluffy-eureka (epistemic monitoring, partitioned memory)
    - psychic-bassoon (governance, safety)
    - combo1 (comprehensive features)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    # Environment
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    project_name: str = "Unified Intelligence System"
    version: str = "0.1.0"

    # Component configurations
    models: ModelConfig = Field(default_factory=ModelConfig)
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    learning: LearningConfig = Field(default_factory=LearningConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    api: APIConfig = Field(default_factory=APIConfig)

    @field_validator("environment", mode="before")
    @classmethod
    def parse_environment(cls, v: Any) -> Environment:
        """Parse environment from string."""
        if isinstance(v, str):
            return Environment(v.lower())
        return v

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == Environment.PRODUCTION

    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == Environment.DEVELOPMENT

    def get_model_list(self) -> List[str]:
        """Get list of all enabled models."""
        models = []
        if self.models.openai_api_key:
            models.extend(self.models.openai_models)
        if self.models.anthropic_api_key:
            models.extend(self.models.anthropic_models)
        if self.models.local_models_enabled:
            models.extend(self.models.ollama_models)
        return models

    def validate_production_settings(self) -> List[str]:
        """
        Validate settings for production deployment.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        if self.is_production:
            # Check required secrets
            if not self.api.jwt_secret_key:
                errors.append("JWT secret key is required in production")

            if not self.models.openai_api_key and not self.models.anthropic_api_key:
                errors.append("At least one model API key is required in production")

            # Check safety settings
            if not self.safety.enable_policy_enforcement:
                errors.append("Policy enforcement should be enabled in production")

            if not self.monitoring.enable_prometheus:
                errors.append("Prometheus monitoring should be enabled in production")

            # Check secure settings
            if self.debug:
                errors.append("Debug mode should be disabled in production")

            if self.api.reload:
                errors.append("API reload should be disabled in production")

        return errors


# Global configuration instance
_config: Optional[UnifiedConfig] = None


def get_config() -> UnifiedConfig:
    """
    Get global configuration instance (singleton pattern).

    Returns:
        Unified configuration object
    """
    global _config
    if _config is None:
        _config = UnifiedConfig()
    return _config


def reload_config() -> UnifiedConfig:
    """
    Reload configuration from environment.

    Returns:
        Fresh configuration object
    """
    global _config
    _config = UnifiedConfig()
    return _config
