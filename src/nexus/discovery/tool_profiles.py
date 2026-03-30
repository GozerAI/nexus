"""
Rich tool and service profiles for the Nexus discovery system.

Provides structured metadata about tools, APIs, and services — both
those already integrated into C-Suite and external marketplace tools
that could be adopted. Each profile describes what the tool does, what
it's best at, what it requires, and which executives/tasks benefit most.

Categories:
- INTERNAL: Tools already wired into C-Suite
- INTEGRATION: External APIs with existing adapters
- MARKETPLACE: External tools/services available for adoption

Usage:
    from nexus.discovery.tool_profiles import (
        get_tool_profile, find_tools_for_task, get_tools_by_category,
    )

    # Find tools for a task
    tools = find_tools_for_task("email marketing", category="communication")

    # Get profile for a specific tool
    profile = get_tool_profile("stripe")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ToolProfile:
    """Rich metadata profile for a tool or service."""

    name: str
    display_name: str
    category: str  # communication, data, devops, research, business, documents, ai, monitoring, security
    availability: str  # "internal" (in C-Suite), "integration" (adapter exists), "marketplace" (available to adopt)

    # What this tool does
    description: str = ""
    strengths: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)

    # What tasks/workflows benefit from this tool
    best_for: List[str] = field(default_factory=list)
    not_suitable_for: List[str] = field(default_factory=list)

    # Which C-Suite executives would use this tool
    recommended_executives: List[str] = field(default_factory=list)

    # Integration details
    auth_type: str = ""  # "api_key", "oauth2", "webhook", "none"
    pricing_model: str = ""  # "free", "freemium", "pay_per_use", "subscription"
    has_free_tier: bool = False

    # Language/format support
    supported_formats: List[str] = field(default_factory=list)  # "json", "csv", "xml", "pdf", etc.
    supported_languages: List[str] = field(default_factory=list)  # natural languages for content tools

    # Technical requirements
    requires_self_hosted: bool = False
    requires_network: bool = True
    api_docs_url: str = ""

    # Quality signals
    reliability: str = "unknown"  # "high", "medium", "low", "unknown"
    maturity: str = "unknown"  # "established", "growing", "emerging", "experimental"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "category": self.category,
            "availability": self.availability,
            "description": self.description,
            "strengths": self.strengths,
            "limitations": self.limitations,
            "best_for": self.best_for,
            "not_suitable_for": self.not_suitable_for,
            "recommended_executives": self.recommended_executives,
            "auth_type": self.auth_type,
            "pricing_model": self.pricing_model,
            "has_free_tier": self.has_free_tier,
            "supported_formats": self.supported_formats,
            "supported_languages": self.supported_languages,
            "requires_self_hosted": self.requires_self_hosted,
            "requires_network": self.requires_network,
            "api_docs_url": self.api_docs_url,
            "reliability": self.reliability,
            "maturity": self.maturity,
        }


# ---------------------------------------------------------------------------
# Tool Profile Registry
# ---------------------------------------------------------------------------

_TOOLS: Dict[str, ToolProfile] = {}


def _reg(profile: ToolProfile) -> None:
    _TOOLS[profile.name] = profile


# ===== COMMUNICATION =====

_reg(ToolProfile(
    name="slack", display_name="Slack",
    category="communication", availability="internal",
    description="Send messages to Slack channels and users",
    strengths=["real-time team communication", "channel-based organization", "rich formatting", "thread support"],
    limitations=["rate limits on API", "no email replacement", "requires workspace access"],
    best_for=["internal notifications", "team updates", "alert routing", "status reports"],
    not_suitable_for=["customer-facing communication", "formal documents"],
    recommended_executives=["CTO", "CEngO", "CMO", "CSO", "CPO"],
    auth_type="oauth2", pricing_model="freemium", has_free_tier=True,
    reliability="high", maturity="established",
    api_docs_url="https://api.slack.com/",
))

_reg(ToolProfile(
    name="discord", display_name="Discord",
    category="communication", availability="internal",
    description="Send messages to Discord channels",
    strengths=["community engagement", "voice channels", "bot ecosystem", "free for most use"],
    limitations=["less formal than Slack", "rate limits", "not enterprise-standard"],
    best_for=["community management", "developer communities", "casual team chat"],
    not_suitable_for=["enterprise communication", "formal business correspondence"],
    recommended_executives=["CMO", "CCO", "CPO"],
    auth_type="api_key", pricing_model="freemium", has_free_tier=True,
    reliability="high", maturity="established",
))

_reg(ToolProfile(
    name="email", display_name="Email (SMTP/API)",
    category="communication", availability="internal",
    description="Send emails via SMTP or API services (SendGrid, Resend, SES)",
    strengths=["universal reach", "formal communication", "attachments", "templates", "tracking"],
    limitations=["delivery delays", "spam filters", "reputation management needed"],
    best_for=["customer outreach", "formal notifications", "marketing campaigns", "transactional emails"],
    not_suitable_for=["real-time communication", "internal quick updates"],
    recommended_executives=["CMO", "CCO", "CRevO", "CPO"],
    auth_type="api_key", pricing_model="pay_per_use", has_free_tier=True,
    supported_languages=["english", "spanish", "french", "german", "chinese", "japanese", "korean",
                         "portuguese", "italian", "dutch", "arabic", "hindi", "russian"],
    reliability="high", maturity="established",
))

_reg(ToolProfile(
    name="notion", display_name="Notion",
    category="communication", availability="internal",
    description="Create and manage pages, databases, and wikis in Notion",
    strengths=["structured knowledge management", "rich content", "databases", "team wikis"],
    limitations=["API limitations on formatting", "no real-time collaboration via API"],
    best_for=["documentation", "project wikis", "meeting notes", "knowledge bases", "roadmaps"],
    not_suitable_for=["real-time messaging", "email replacement"],
    recommended_executives=["CPO", "CTO", "CMO", "CSO"],
    auth_type="oauth2", pricing_model="freemium", has_free_tier=True,
    reliability="high", maturity="established",
    api_docs_url="https://developers.notion.com/",
))

_reg(ToolProfile(
    name="twilio", display_name="Twilio",
    category="communication", availability="marketplace",
    description="SMS, voice calls, WhatsApp messaging, and video via API",
    strengths=["SMS globally", "voice calls", "WhatsApp Business", "programmable", "reliable delivery"],
    limitations=["cost per message/call", "phone number provisioning", "compliance requirements"],
    best_for=["customer notifications", "2FA/verification", "appointment reminders", "phone support"],
    not_suitable_for=["internal team chat", "long-form content"],
    recommended_executives=["CCO", "CRevO", "CMO"],
    auth_type="api_key", pricing_model="pay_per_use", has_free_tier=True,
    supported_languages=["english", "spanish", "french", "german", "chinese", "japanese",
                         "portuguese", "arabic", "hindi", "korean"],
    reliability="high", maturity="established",
    api_docs_url="https://www.twilio.com/docs",
))

# ===== BUSINESS =====

_reg(ToolProfile(
    name="stripe", display_name="Stripe",
    category="business", availability="internal",
    description="Payment processing, subscriptions, invoicing",
    strengths=["comprehensive payment API", "subscription management", "global payments",
               "fraud detection", "detailed reporting", "webhook events"],
    limitations=["fees per transaction", "complex for simple payments", "compliance overhead"],
    best_for=["payment processing", "subscription billing", "invoice generation",
              "revenue tracking", "refund management"],
    not_suitable_for=["cryptocurrency payments", "cash transactions"],
    recommended_executives=["CFO", "CRevO"],
    auth_type="api_key", pricing_model="pay_per_use",
    supported_formats=["json"],
    reliability="high", maturity="established",
    api_docs_url="https://stripe.com/docs/api",
))

_reg(ToolProfile(
    name="jira", display_name="Jira / Linear",
    category="business", availability="internal",
    description="Project management, issue tracking, sprint planning",
    strengths=["issue tracking", "sprint management", "workflows", "roadmaps", "integrations"],
    limitations=["complex configuration", "can be slow", "steep learning curve"],
    best_for=["bug tracking", "sprint planning", "project management", "task assignment"],
    not_suitable_for=["simple todo lists", "personal task management"],
    recommended_executives=["CTO", "CEngO", "CPO"],
    auth_type="api_key", pricing_model="freemium", has_free_tier=True,
    reliability="high", maturity="established",
))

_reg(ToolProfile(
    name="hubspot", display_name="HubSpot CRM",
    category="business", availability="marketplace",
    description="CRM, marketing automation, sales pipeline, customer service",
    strengths=["unified CRM", "marketing automation", "email sequences",
               "pipeline management", "reporting", "free tier"],
    limitations=["expensive at scale", "complex pricing tiers", "data migration challenges"],
    best_for=["lead management", "email marketing", "sales tracking", "customer lifecycle"],
    not_suitable_for=["simple contact lists", "enterprise with existing CRM"],
    recommended_executives=["CMO", "CRevO", "CCO"],
    auth_type="oauth2", pricing_model="freemium", has_free_tier=True,
    reliability="high", maturity="established",
    api_docs_url="https://developers.hubspot.com/docs/api",
))

_reg(ToolProfile(
    name="zapier", display_name="Zapier",
    category="business", availability="internal",
    description="Workflow automation connecting 5000+ apps via triggers and actions",
    strengths=["massive app ecosystem", "no-code automation", "multi-step workflows", "scheduling"],
    limitations=["cost scales with usage", "execution limits", "limited error handling"],
    best_for=["cross-app automation", "data sync", "notification routing", "lead capture workflows"],
    not_suitable_for=["complex data transformations", "real-time processing", "high-volume tasks"],
    recommended_executives=["CTO", "CMO", "CRevO"],
    auth_type="webhook", pricing_model="freemium", has_free_tier=True,
    reliability="high", maturity="established",
))

# ===== DATA =====

_reg(ToolProfile(
    name="sql_query", display_name="SQL Database Query",
    category="data", availability="internal",
    description="Execute SQL queries against PostgreSQL, MySQL, SQLite databases",
    strengths=["direct data access", "complex queries", "aggregations", "joins"],
    limitations=["requires SQL knowledge", "security risks if unvalidated", "read-only recommended"],
    best_for=["data analysis", "reporting", "metrics extraction", "data validation"],
    not_suitable_for=["unstructured data", "real-time streaming"],
    recommended_executives=["CDO", "CFO", "CSO", "CRO", "CRiO"],
    auth_type="none", pricing_model="free",
    supported_formats=["json", "csv"],
    reliability="high", maturity="established",
))

_reg(ToolProfile(
    name="elasticsearch", display_name="Elasticsearch",
    category="data", availability="marketplace",
    description="Full-text search, analytics, and log aggregation engine",
    strengths=["fast full-text search", "real-time analytics", "log aggregation",
               "horizontal scaling", "rich query DSL"],
    limitations=["complex operations", "resource-intensive", "eventual consistency"],
    best_for=["log analysis", "full-text search", "metrics dashboards", "anomaly detection"],
    not_suitable_for=["transactional data", "relational queries", "small datasets"],
    recommended_executives=["CDO", "CTO", "CSecO"],
    auth_type="api_key", pricing_model="freemium", has_free_tier=True,
    requires_self_hosted=True,
    reliability="high", maturity="established",
    api_docs_url="https://www.elastic.co/guide/en/elasticsearch/reference/current/",
))

_reg(ToolProfile(
    name="snowflake", display_name="Snowflake",
    category="data", availability="marketplace",
    description="Cloud data warehouse for analytics, data sharing, and ML",
    strengths=["massive scale", "separation of compute/storage", "data sharing",
               "semi-structured data support", "time travel"],
    limitations=["cost at scale", "vendor lock-in", "cold start latency"],
    best_for=["enterprise analytics", "data warehousing", "cross-org data sharing", "ML pipelines"],
    not_suitable_for=["OLTP workloads", "real-time streaming", "small datasets"],
    recommended_executives=["CDO", "CFO", "CSO"],
    auth_type="oauth2", pricing_model="pay_per_use",
    reliability="high", maturity="established",
))

# ===== DEVOPS =====

_reg(ToolProfile(
    name="github", display_name="GitHub",
    category="devops", availability="internal",
    description="Code hosting, issues, pull requests, CI/CD via Actions",
    strengths=["industry standard", "CI/CD", "code review", "issue tracking", "package registry"],
    limitations=["API rate limits", "Actions minutes cost", "large file handling"],
    best_for=["code management", "CI/CD pipelines", "issue tracking", "code review", "open source"],
    not_suitable_for=["non-code project management", "document storage"],
    recommended_executives=["CTO", "CEngO"],
    auth_type="api_key", pricing_model="freemium", has_free_tier=True,
    reliability="high", maturity="established",
    api_docs_url="https://docs.github.com/en/rest",
))

_reg(ToolProfile(
    name="docker", display_name="Docker",
    category="devops", availability="internal",
    description="Container runtime for building, shipping, and running applications",
    strengths=["environment isolation", "reproducible builds", "microservices", "CI/CD integration"],
    limitations=["security considerations", "resource overhead", "networking complexity"],
    best_for=["application deployment", "dev environments", "microservices", "CI/CD"],
    not_suitable_for=["GUI applications", "bare-metal performance requirements"],
    recommended_executives=["CTO", "CEngO"],
    auth_type="none", pricing_model="free", has_free_tier=True,
    requires_self_hosted=True, requires_network=False,
    reliability="high", maturity="established",
))

_reg(ToolProfile(
    name="terraform", display_name="Terraform / OpenTofu",
    category="devops", availability="marketplace",
    description="Infrastructure as Code for provisioning cloud resources",
    strengths=["multi-cloud", "declarative config", "state management", "plan/apply workflow"],
    limitations=["state management complexity", "learning curve", "provider-specific quirks"],
    best_for=["cloud provisioning", "infrastructure management", "multi-cloud deployment"],
    not_suitable_for=["application deployment", "config management"],
    recommended_executives=["CTO", "CEngO", "CIO"],
    auth_type="none", pricing_model="free", has_free_tier=True,
    requires_self_hosted=True, requires_network=False,
    reliability="high", maturity="established",
))

_reg(ToolProfile(
    name="kubernetes", display_name="Kubernetes",
    category="devops", availability="marketplace",
    description="Container orchestration for scaling and managing containerized applications",
    strengths=["auto-scaling", "self-healing", "service discovery", "rolling updates", "ecosystem"],
    limitations=["operational complexity", "resource overhead", "steep learning curve"],
    best_for=["microservices at scale", "auto-scaling workloads", "multi-service deployments"],
    not_suitable_for=["simple single-container apps", "small teams without ops expertise"],
    recommended_executives=["CTO", "CEngO", "CIO"],
    auth_type="none", pricing_model="free", has_free_tier=True,
    requires_self_hosted=True,
    reliability="high", maturity="established",
))

# ===== RESEARCH =====

_reg(ToolProfile(
    name="web_scrape", display_name="Web Scraper",
    category="research", availability="internal",
    description="Extract content from web pages using CSS selectors",
    strengths=["flexible extraction", "handles dynamic content", "CSS selector targeting"],
    limitations=["blocked by anti-scraping", "fragile to layout changes", "rate limiting"],
    best_for=["competitive research", "data collection", "content monitoring", "price tracking"],
    not_suitable_for=["structured API data", "high-volume real-time data"],
    recommended_executives=["CSO", "CRO", "CMO", "CRiO"],
    auth_type="none", pricing_model="free",
    requires_network=True,
    reliability="medium", maturity="established",
))

_reg(ToolProfile(
    name="arxiv_search", display_name="ArXiv Paper Search",
    category="research", availability="internal",
    description="Search academic papers on ArXiv across all scientific domains",
    strengths=["comprehensive academic coverage", "free access", "structured metadata"],
    limitations=["preprints (not peer-reviewed)", "STEM focus", "no full-text search"],
    best_for=["literature review", "technology scouting", "research trends", "citation analysis"],
    not_suitable_for=["business intelligence", "market research"],
    recommended_executives=["CRO", "CTO", "CSO"],
    auth_type="none", pricing_model="free", has_free_tier=True,
    reliability="high", maturity="established",
    api_docs_url="https://info.arxiv.org/help/api/",
))

_reg(ToolProfile(
    name="perplexity", display_name="Perplexity AI Search",
    category="research", availability="marketplace",
    description="AI-powered web search with source citations and real-time data",
    strengths=["real-time web data", "source citations", "AI-synthesized answers", "current events"],
    limitations=["API costs", "rate limits", "answer quality varies"],
    best_for=["current events research", "fact-checking", "market intelligence", "competitor analysis"],
    not_suitable_for=["historical research", "academic depth"],
    recommended_executives=["CSO", "CRO", "CMO", "CRiO"],
    auth_type="api_key", pricing_model="pay_per_use", has_free_tier=True,
    reliability="high", maturity="growing",
    api_docs_url="https://docs.perplexity.ai/",
))

# ===== DOCUMENTS =====

_reg(ToolProfile(
    name="pdf_extract", display_name="PDF Text Extraction",
    category="documents", availability="internal",
    description="Extract text and tables from PDF documents",
    strengths=["handles complex layouts", "table extraction", "metadata extraction"],
    limitations=["scanned PDFs need OCR", "complex formatting may lose structure"],
    best_for=["contract analysis", "report processing", "data extraction from PDFs"],
    not_suitable_for=["image-heavy PDFs", "handwritten documents"],
    recommended_executives=["CLO", "CComO", "CFO", "CDO"],
    auth_type="none", pricing_model="free",
    supported_formats=["pdf"],
    requires_network=False,
    reliability="high", maturity="established",
))

_reg(ToolProfile(
    name="docusign", display_name="DocuSign",
    category="documents", availability="marketplace",
    description="Electronic signature and agreement management platform",
    strengths=["legally binding e-signatures", "workflow automation", "audit trails", "templates"],
    limitations=["cost per envelope", "complex API", "limited free tier"],
    best_for=["contract signing", "NDA management", "onboarding documents", "legal agreements"],
    not_suitable_for=["internal document collaboration", "content creation"],
    recommended_executives=["CLO", "CComO", "CFO", "CCO"],
    auth_type="oauth2", pricing_model="subscription",
    reliability="high", maturity="established",
    api_docs_url="https://developers.docusign.com/",
))

# ===== AI / ML =====

_reg(ToolProfile(
    name="huggingface", display_name="HuggingFace",
    category="ai", availability="integration",
    description="ML model hub, datasets, and inference API",
    strengths=["largest model hub", "free inference API", "datasets", "spaces",
               "model cards", "community"],
    limitations=["inference API rate limits", "large model hosting costs"],
    best_for=["model discovery", "fine-tuning", "embeddings", "NLP tasks", "dataset search"],
    not_suitable_for=["production high-throughput inference"],
    recommended_executives=["CTO", "CDO", "CRO"],
    auth_type="api_key", pricing_model="freemium", has_free_tier=True,
    reliability="high", maturity="established",
    api_docs_url="https://huggingface.co/docs/api-inference/",
))

_reg(ToolProfile(
    name="wandb", display_name="Weights & Biases",
    category="ai", availability="marketplace",
    description="ML experiment tracking, model versioning, and collaboration",
    strengths=["experiment tracking", "hyperparameter sweeps", "model registry",
               "team collaboration", "visualization"],
    limitations=["learning curve", "data volume costs", "self-hosted complexity"],
    best_for=["ML experiment management", "model comparison", "hyperparameter optimization"],
    not_suitable_for=["non-ML projects", "simple logging"],
    recommended_executives=["CTO", "CDO", "CRO"],
    auth_type="api_key", pricing_model="freemium", has_free_tier=True,
    reliability="high", maturity="established",
    api_docs_url="https://docs.wandb.ai/",
))

# ===== MONITORING =====

_reg(ToolProfile(
    name="grafana", display_name="Grafana",
    category="monitoring", availability="marketplace",
    description="Observability platform for metrics, logs, and traces visualization",
    strengths=["beautiful dashboards", "multi-datasource", "alerting", "free/open-source"],
    limitations=["requires data sources", "dashboard management overhead"],
    best_for=["metrics dashboards", "system monitoring", "alerting", "SLA tracking"],
    not_suitable_for=["data storage", "log collection (use with Loki/ES)"],
    recommended_executives=["CTO", "CEngO", "CIO"],
    auth_type="api_key", pricing_model="freemium", has_free_tier=True,
    requires_self_hosted=True,
    reliability="high", maturity="established",
    api_docs_url="https://grafana.com/docs/grafana/latest/developers/http_api/",
))

_reg(ToolProfile(
    name="sentry", display_name="Sentry",
    category="monitoring", availability="marketplace",
    description="Error tracking and performance monitoring for applications",
    strengths=["real-time error tracking", "stack traces", "release tracking",
               "performance monitoring", "issue grouping"],
    limitations=["event volume costs", "SDK integration required"],
    best_for=["error monitoring", "crash reporting", "performance bottlenecks", "release health"],
    not_suitable_for=["infrastructure monitoring", "log aggregation"],
    recommended_executives=["CTO", "CEngO"],
    auth_type="api_key", pricing_model="freemium", has_free_tier=True,
    reliability="high", maturity="established",
    api_docs_url="https://docs.sentry.io/api/",
))

# ===== SECURITY =====

_reg(ToolProfile(
    name="vault", display_name="HashiCorp Vault",
    category="security", availability="marketplace",
    description="Secrets management, encryption, and identity-based access",
    strengths=["dynamic secrets", "encryption as a service", "audit logging",
               "multi-cloud", "identity-based access"],
    limitations=["operational complexity", "requires dedicated management"],
    best_for=["secrets management", "API key rotation", "encryption", "PKI"],
    not_suitable_for=["simple env var storage", "small teams without ops"],
    recommended_executives=["CSecO", "CTO", "CIO"],
    auth_type="api_key", pricing_model="freemium", has_free_tier=True,
    requires_self_hosted=True,
    reliability="high", maturity="established",
    api_docs_url="https://developer.hashicorp.com/vault/api-docs",
))


# ===== OPEN SOURCE — AI / ML =====

_reg(ToolProfile(
    name="ollama", display_name="Ollama",
    category="ai", availability="internal",
    description="Run open-source LLMs locally (Qwen, Llama, DeepSeek, Mistral, Gemma)",
    strengths=["zero API cost", "data stays local", "fast inference", "model switching",
               "supports GGUF/safetensors", "simple CLI and API"],
    limitations=["requires GPU for large models", "slower than cloud APIs for flagship models",
                 "no function calling on all models"],
    best_for=["routine LLM tasks", "development/testing", "privacy-sensitive data",
              "cost reduction", "offline operation"],
    not_suitable_for=["frontier-level reasoning", "tasks requiring 200K+ context"],
    recommended_executives=["CTO", "CEngO", "CDO", "CRO"],
    auth_type="none", pricing_model="free", has_free_tier=True,
    requires_self_hosted=True, requires_network=False,
    reliability="high", maturity="established",
    api_docs_url="https://github.com/ollama/ollama/blob/main/docs/api.md",
))

_reg(ToolProfile(
    name="qdrant", display_name="Qdrant",
    category="ai", availability="marketplace",
    description="Open-source vector database for semantic search and RAG",
    strengths=["fast similarity search", "filtering during search", "horizontal scaling",
               "REST + gRPC API", "payload storage", "quantization support"],
    limitations=["requires separate deployment", "memory-intensive for large collections"],
    best_for=["RAG pipelines", "semantic search", "recommendation systems",
              "knowledge base search", "document retrieval"],
    not_suitable_for=["relational queries", "transactional data", "small datasets"],
    recommended_executives=["CTO", "CDO", "CRO"],
    auth_type="api_key", pricing_model="free", has_free_tier=True,
    requires_self_hosted=True,
    reliability="high", maturity="growing",
    api_docs_url="https://qdrant.tech/documentation/",
))

_reg(ToolProfile(
    name="langfuse", display_name="Langfuse",
    category="ai", availability="marketplace",
    description="Open-source LLM observability: tracing, prompt management, cost tracking, evals",
    strengths=["traces every LLM call", "cost tracking per request", "prompt versioning",
               "evaluation pipelines", "self-hostable", "OpenAI-compatible SDK"],
    limitations=["requires deployment", "storage grows with usage"],
    best_for=["LLM cost tracking", "prompt debugging", "A/B testing prompts",
              "production monitoring", "eval pipelines"],
    not_suitable_for=["general application monitoring", "non-LLM workloads"],
    recommended_executives=["CTO", "CFO", "CDO"],
    auth_type="api_key", pricing_model="freemium", has_free_tier=True,
    requires_self_hosted=True,
    reliability="high", maturity="growing",
    api_docs_url="https://langfuse.com/docs",
))

# ===== OPEN SOURCE — DATA =====

_reg(ToolProfile(
    name="minio", display_name="MinIO",
    category="data", availability="marketplace",
    description="S3-compatible high-performance object storage",
    strengths=["S3-compatible API", "high performance", "self-hosted data sovereignty",
               "versioning", "bucket policies", "erasure coding"],
    limitations=["requires operations", "no managed service free tier"],
    best_for=["file storage", "artifact storage", "backup", "data lake",
              "ML model storage", "document management"],
    not_suitable_for=["database replacement", "real-time streaming"],
    recommended_executives=["CTO", "CEngO", "CDO"],
    auth_type="api_key", pricing_model="free", has_free_tier=True,
    requires_self_hosted=True,
    supported_formats=["any"],
    reliability="high", maturity="established",
    api_docs_url="https://min.io/docs/minio/linux/developers/minio-drivers.html",
))

_reg(ToolProfile(
    name="typesense", display_name="Typesense",
    category="data", availability="marketplace",
    description="Fast open-source search engine with typo tolerance and faceting",
    strengths=["instant search (<50ms)", "typo tolerance", "faceted search",
               "easy setup", "low resource usage", "geo search"],
    limitations=["smaller ecosystem than Elasticsearch", "limited analytics"],
    best_for=["product search", "site search", "autocomplete", "documentation search"],
    not_suitable_for=["log aggregation", "complex analytics", "very large datasets"],
    recommended_executives=["CTO", "CPO", "CDO"],
    auth_type="api_key", pricing_model="free", has_free_tier=True,
    requires_self_hosted=True,
    reliability="high", maturity="growing",
    api_docs_url="https://typesense.org/docs/",
))

_reg(ToolProfile(
    name="meilisearch", display_name="Meilisearch",
    category="data", availability="marketplace",
    description="Lightning-fast open-source search engine with relevancy tuning",
    strengths=["sub-50ms search", "typo tolerance", "multi-language",
               "easy deployment", "customizable relevancy", "filtering"],
    limitations=["dataset size limits vs Elasticsearch", "fewer advanced features"],
    best_for=["e-commerce search", "documentation search", "autocomplete",
              "content discovery"],
    not_suitable_for=["log analytics", "geospatial queries at scale"],
    recommended_executives=["CTO", "CPO"],
    auth_type="api_key", pricing_model="free", has_free_tier=True,
    requires_self_hosted=True,
    supported_languages=["english", "french", "german", "spanish", "italian",
                         "portuguese", "chinese", "japanese", "korean",
                         "dutch", "russian", "arabic", "hindi", "thai"],
    reliability="high", maturity="growing",
    api_docs_url="https://www.meilisearch.com/docs",
))

_reg(ToolProfile(
    name="redpanda", display_name="Redpanda",
    category="data", availability="marketplace",
    description="Kafka-compatible event streaming platform, no JVM required",
    strengths=["Kafka API compatible", "no JVM/ZooKeeper", "lower latency",
               "simpler operations", "built-in schema registry", "WebAssembly transforms"],
    limitations=["smaller ecosystem than Kafka", "fewer managed offerings"],
    best_for=["event streaming", "real-time data pipelines", "event sourcing",
              "microservice communication", "log aggregation"],
    not_suitable_for=["simple pub/sub (use Redis)", "small-scale messaging"],
    recommended_executives=["CTO", "CEngO", "CDO"],
    auth_type="none", pricing_model="free", has_free_tier=True,
    requires_self_hosted=True,
    reliability="high", maturity="growing",
    api_docs_url="https://docs.redpanda.com/",
))

_reg(ToolProfile(
    name="directus", display_name="Directus",
    category="data", availability="marketplace",
    description="Open-source headless CMS and data studio with instant REST/GraphQL API",
    strengths=["instant API from any SQL database", "admin UI", "role-based access",
               "webhooks", "file management", "GraphQL + REST"],
    limitations=["requires SQL database", "UI customization limits"],
    best_for=["content management", "API generation from existing databases",
              "admin dashboards", "data entry interfaces"],
    not_suitable_for=["complex application logic", "real-time apps"],
    recommended_executives=["CTO", "CPO", "CMO"],
    auth_type="api_key", pricing_model="free", has_free_tier=True,
    requires_self_hosted=True,
    reliability="high", maturity="established",
    api_docs_url="https://docs.directus.io/",
))

# ===== OPEN SOURCE — MONITORING =====

_reg(ToolProfile(
    name="posthog", display_name="PostHog",
    category="monitoring", availability="marketplace",
    description="Open-source product analytics, session replay, feature flags, A/B testing",
    strengths=["all-in-one product analytics", "session replay", "feature flags",
               "A/B experiments", "self-hostable", "event autocapture"],
    limitations=["resource-intensive self-hosting", "learning curve for advanced features"],
    best_for=["product analytics", "user behavior tracking", "feature flagging",
              "conversion funnels", "A/B testing"],
    not_suitable_for=["infrastructure monitoring", "error tracking (use Sentry)"],
    recommended_executives=["CPO", "CMO", "CRevO"],
    auth_type="api_key", pricing_model="freemium", has_free_tier=True,
    requires_self_hosted=True,
    reliability="high", maturity="growing",
    api_docs_url="https://posthog.com/docs/api",
))

_reg(ToolProfile(
    name="uptime_kuma", display_name="Uptime Kuma",
    category="monitoring", availability="marketplace",
    description="Self-hosted uptime monitoring with notifications",
    strengths=["simple setup", "beautiful dashboard", "multi-protocol monitoring",
               "notification integrations (Slack, Discord, email, Telegram)",
               "status pages", "very low resource usage"],
    limitations=["single-node only", "no distributed monitoring"],
    best_for=["service uptime monitoring", "status pages", "SSL certificate monitoring",
              "ping/port checks", "alerting"],
    not_suitable_for=["APM", "log analysis", "distributed tracing"],
    recommended_executives=["CTO", "CEngO"],
    auth_type="none", pricing_model="free", has_free_tier=True,
    requires_self_hosted=True, requires_network=True,
    reliability="high", maturity="established",
    api_docs_url="https://github.com/louislam/uptime-kuma/wiki/API",
))

_reg(ToolProfile(
    name="plausible", display_name="Plausible Analytics",
    category="monitoring", availability="marketplace",
    description="Privacy-friendly open-source web analytics (no cookies, GDPR compliant)",
    strengths=["no cookies needed", "GDPR/CCPA compliant", "lightweight script (<1KB)",
               "simple dashboard", "self-hostable", "goal tracking"],
    limitations=["fewer features than Google Analytics", "no user-level tracking"],
    best_for=["website analytics", "privacy-compliant tracking", "marketing attribution",
              "content performance"],
    not_suitable_for=["user-level analytics", "complex funnels", "e-commerce tracking"],
    recommended_executives=["CMO", "CPO"],
    auth_type="api_key", pricing_model="freemium", has_free_tier=True,
    requires_self_hosted=True,
    reliability="high", maturity="established",
    api_docs_url="https://plausible.io/docs/stats-api",
))

# ===== DISTRIBUTION CHANNELS =====
# Email/newsletter, blog/CMS, video, podcast, digital products, social APIs.
# These complement shopforge/mdusa (commerce) with owned-media publishing channels.

_reg(ToolProfile(
    name="beehiiv", display_name="Beehiiv",
    category="communication", availability="integration",
    description="Newsletter platform built for growth — subscriber management, monetisation, analytics, and referral programs",
    strengths=["subscriber segmentation", "built-in monetisation", "referral program",
               "detailed open/click analytics", "custom domain", "web archive", "API-first"],
    limitations=["email-only audience", "less editorial flexibility than Ghost"],
    best_for=["newsletter publishing", "email list growth", "subscriber monetisation",
              "drip campaigns", "audience analytics", "referral growth loops"],
    not_suitable_for=["transactional email", "e-commerce checkout", "full CMS"],
    recommended_executives=["CMO", "CRevO", "CCO"],
    auth_type="api_key", pricing_model="freemium", has_free_tier=True,
    supported_formats=["html", "markdown"],
    reliability="high", maturity="growing",
    api_docs_url="https://developers.beehiiv.com/",
))

_reg(ToolProfile(
    name="mailchimp", display_name="Mailchimp",
    category="communication", availability="marketplace",
    description="Established email marketing platform with automations, audience segmentation, and multi-channel campaigns",
    strengths=["large template library", "audience segmentation", "A/B testing",
               "marketing automations", "landing pages", "social posting", "wide integrations"],
    limitations=["expensive at scale", "deliverability lower than SendGrid", "complex pricing"],
    best_for=["email campaigns", "drip sequences", "list segmentation",
              "campaign A/B testing", "audience growth", "e-commerce emails"],
    not_suitable_for=["transactional emails at volume (use Resend/SendGrid)", "newsletters at scale (use Beehiiv)"],
    recommended_executives=["CMO", "CRevO"],
    auth_type="api_key", pricing_model="freemium", has_free_tier=True,
    supported_formats=["html", "plain_text"],
    reliability="high", maturity="established",
    api_docs_url="https://mailchimp.com/developer/marketing/api/",
))

_reg(ToolProfile(
    name="convertkit", display_name="Kit (ConvertKit)",
    category="communication", availability="marketplace",
    description="Creator-focused email marketing with visual automations, forms, and commerce integrations",
    strengths=["visual automation builder", "subscriber tagging", "commerce integrations",
               "landing page builder", "creator-friendly UX", "strong deliverability"],
    limitations=["fewer templates than Mailchimp", "limited analytics depth"],
    best_for=["creator email marketing", "lead nurture sequences", "content upgrades",
              "subscriber tagging", "product launch emails", "course delivery"],
    not_suitable_for=["enterprise bulk email", "transactional email"],
    recommended_executives=["CMO", "CPO"],
    auth_type="api_key", pricing_model="freemium", has_free_tier=True,
    supported_formats=["html", "plain_text"],
    reliability="high", maturity="established",
    api_docs_url="https://developers.convertkit.com/",
))

_reg(ToolProfile(
    name="ghost", display_name="Ghost",
    category="communication", availability="marketplace",
    description="Open-source headless CMS and publishing platform for blogs, newsletters, and membership sites",
    strengths=["self-hostable", "built-in newsletter (via Mailgun/Sendgrid)", "membership tiers",
               "REST/Admin API", "SEO-friendly", "Markdown/rich editor", "custom themes"],
    limitations=["requires self-hosting or Ghost(Pro)", "smaller plugin ecosystem than WordPress"],
    best_for=["blog publishing", "membership content", "newsletter + blog hybrid",
              "content monetisation", "SEO content", "headless CMS for Next.js/Gatsby"],
    not_suitable_for=["e-commerce (use Shopforge)", "heavy plugin ecosystem"],
    recommended_executives=["CMO", "CPO", "CRevO"],
    auth_type="api_key", pricing_model="freemium", has_free_tier=False,
    requires_self_hosted=True,
    supported_formats=["html", "markdown", "json"],
    reliability="high", maturity="established",
    api_docs_url="https://ghost.org/docs/content-api/",
))

_reg(ToolProfile(
    name="wordpress_api", display_name="WordPress REST API",
    category="communication", availability="marketplace",
    description="REST API for programmatic publishing to WordPress sites — posts, pages, media, and taxonomies",
    strengths=["massive ecosystem", "plugin extensibility", "full CRUD on posts/pages",
               "media upload", "taxonomy management", "WooCommerce integration"],
    limitations=["requires WordPress installation", "auth setup (App Passwords/JWT)"],
    best_for=["blog post publishing", "page creation", "bulk content import",
              "headless WordPress", "SEO content at scale", "existing WP sites"],
    not_suitable_for=["new site setup (use Ghost or Coolify)", "real-time feeds"],
    recommended_executives=["CMO", "CPO"],
    auth_type="api_key", pricing_model="free", has_free_tier=True,
    supported_formats=["html", "json"],
    reliability="high", maturity="established",
    api_docs_url="https://developer.wordpress.org/rest-api/",
))

_reg(ToolProfile(
    name="hashnode", display_name="Hashnode",
    category="communication", availability="marketplace",
    description="Developer-focused blogging platform with custom domain, newsletter, and GraphQL API",
    strengths=["GraphQL API", "custom domain (free)", "built-in newsletter",
               "tech community discovery", "Markdown import", "SEO features"],
    limitations=["tech/developer audience only", "limited customisation vs Ghost"],
    best_for=["technical blog posts", "developer content", "API docs", "tech tutorials",
              "thought leadership for SaaS", "community discovery"],
    not_suitable_for=["non-technical audiences", "membership gating"],
    recommended_executives=["CMO", "CTO", "CEngO"],
    auth_type="api_key", pricing_model="free", has_free_tier=True,
    supported_formats=["markdown"],
    reliability="high", maturity="growing",
    api_docs_url="https://api.hashnode.com/",
))

_reg(ToolProfile(
    name="youtube_studio", display_name="YouTube Data API",
    category="communication", availability="marketplace",
    description="YouTube API for uploading videos, managing metadata, and retrieving channel analytics",
    strengths=["largest video platform", "SEO via search", "chapter markers", "subtitles API",
               "analytics depth", "ad monetisation", "Shorts support"],
    limitations=["requires video file upload", "strict ToS on automation", "OAuth2 required"],
    best_for=["video content publishing", "product demos", "thought leadership video",
              "tutorial content", "channel analytics", "video SEO optimisation"],
    not_suitable_for=["live-only content (use Twitch)", "podcast audio only"],
    recommended_executives=["CMO", "CPO"],
    auth_type="oauth2", pricing_model="free", has_free_tier=True,
    supported_formats=["mp4", "mov", "json"],
    reliability="high", maturity="established",
    api_docs_url="https://developers.google.com/youtube/v3",
))

_reg(ToolProfile(
    name="buzzsprout", display_name="Buzzsprout",
    category="communication", availability="marketplace",
    description="Podcast hosting and distribution platform with API for episode management and analytics",
    strengths=["wide directory distribution (Spotify, Apple, Amazon)", "episode API",
               "dynamic ad insertion", "chapter markers", "transcript generation", "embeddable player"],
    limitations=["audio files required", "limited free plan storage"],
    best_for=["podcast episode publishing", "show notes distribution",
              "podcast analytics", "cross-platform podcast distribution", "audiogram creation"],
    not_suitable_for=["video content", "written content distribution"],
    recommended_executives=["CMO", "CCO"],
    auth_type="api_key", pricing_model="freemium", has_free_tier=True,
    supported_formats=["mp3", "wav", "json"],
    reliability="high", maturity="established",
    api_docs_url="https://www.buzzsprout.com/api",
))

_reg(ToolProfile(
    name="gumroad", display_name="Gumroad",
    category="business", availability="integration",
    description="Digital product marketplace for selling ebooks, courses, templates, and digital downloads",
    strengths=["zero setup", "instant product creation via API", "handles payments + delivery",
               "subscriber list", "analytics", "affiliate program", "global tax handling"],
    limitations=["10% platform fee on free plan", "limited store customisation", "no subscriptions on basic"],
    best_for=["ebook sales", "digital product launches", "course selling",
              "template marketplaces", "content monetisation", "pay-what-you-want pricing"],
    not_suitable_for=["physical goods (use Shopforge)", "SaaS subscriptions (use Stripe)"],
    recommended_executives=["CRevO", "CMO", "CPO"],
    auth_type="api_key", pricing_model="freemium", has_free_tier=True,
    supported_formats=["pdf", "epub", "zip", "mp4", "mp3"],
    reliability="high", maturity="established",
    api_docs_url="https://app.gumroad.com/api",
))

_reg(ToolProfile(
    name="lemon_squeezy", display_name="Lemon Squeezy",
    category="business", availability="integration",
    description="Merchant-of-record platform for digital products and SaaS — handles VAT, payments, and licensing",
    strengths=["merchant-of-record (handles tax globally)", "license key API", "subscription billing",
               "checkout customisation", "affiliate system", "customer portal", "webhook events"],
    limitations=["5-8% fee", "less brand flexibility than Stripe", "US-focused support"],
    best_for=["digital product sales", "SaaS licensing", "subscription products",
              "global tax compliance", "software licenses", "course + product bundles"],
    not_suitable_for=["physical goods", "enterprise custom billing (use Stripe)"],
    recommended_executives=["CRevO", "CFO", "CPO"],
    auth_type="api_key", pricing_model="pay_per_use", has_free_tier=False,
    supported_formats=["json", "pdf", "zip"],
    reliability="high", maturity="growing",
    api_docs_url="https://docs.lemonsqueezy.com/api",
))

_reg(ToolProfile(
    name="substack", display_name="Substack",
    category="communication", availability="marketplace",
    description="Newsletter and publication platform with built-in payments, subscriber discovery, and podcast support",
    strengths=["built-in paid subscriptions", "subscriber discovery network",
               "podcast episodes per post", "notes feed", "reader app", "simple setup"],
    limitations=["10% take on revenue", "limited API (no programmatic publishing)",
               "less customisable than Ghost", "platform lock-in risk"],
    best_for=["writer-first newsletters", "paid subscriber growth", "publication launches",
              "long-form content monetisation", "thought leadership"],
    not_suitable_for=["automated publishing (no API)", "brand-controlled channels"],
    recommended_executives=["CMO", "CRevO"],
    auth_type="none", pricing_model="freemium", has_free_tier=True,
    supported_formats=["html", "markdown"],
    reliability="high", maturity="established",
    api_docs_url="",
))

_reg(ToolProfile(
    name="twitter_api", display_name="Twitter / X API",
    category="communication", availability="integration",
    description="Twitter/X API for programmatic tweet posting, thread creation, scheduling, and analytics",
    strengths=["real-time reach", "thread support", "quote tweets", "analytics API",
               "media upload", "reply threading", "bookmark/save"],
    limitations=["strict rate limits on free tier", "paid API required for full features", "algorithm deprioritises links"],
    best_for=["micro-content distribution", "launch announcements", "thread storytelling",
              "real-time commentary", "product updates", "community engagement"],
    not_suitable_for=["long-form content (use Ghost/Hashnode)", "confidential comms"],
    recommended_executives=["CMO", "CRevO"],
    auth_type="oauth2", pricing_model="freemium", has_free_tier=True,
    supported_formats=["plain_text", "json"],
    reliability="medium", maturity="established",
    api_docs_url="https://developer.twitter.com/en/docs/twitter-api",
))

_reg(ToolProfile(
    name="linkedin_api", display_name="LinkedIn Marketing API",
    category="communication", availability="integration",
    description="LinkedIn API for publishing posts, articles, and company page updates with professional audience reach",
    strengths=["B2B audience", "long-form article support", "company page posting",
               "document / PDF posts", "analytics API", "high organic reach for thought leadership"],
    limitations=["OAuth2 required", "rate limits on post frequency", "consumer use restrictions"],
    best_for=["B2B content distribution", "thought leadership articles",
              "product announcements", "company updates", "recruitment content", "case studies"],
    not_suitable_for=["B2C casual content", "high-frequency posting"],
    recommended_executives=["CMO", "CCO", "CRevO"],
    auth_type="oauth2", pricing_model="free", has_free_tier=True,
    supported_formats=["plain_text", "html", "pdf", "json"],
    reliability="high", maturity="established",
    api_docs_url="https://learn.microsoft.com/en-us/linkedin/marketing/",
))

# ===== OPEN SOURCE — BUSINESS =====

_reg(ToolProfile(
    name="cal_com", display_name="Cal.com",
    category="business", availability="marketplace",
    description="Open-source scheduling and appointment booking platform",
    strengths=["self-hostable", "calendar integrations (Google, Outlook, Apple)",
               "team scheduling", "round-robin", "customizable booking pages"],
    limitations=["requires calendar API setup", "complex for simple scheduling"],
    best_for=["meeting scheduling", "appointment booking", "sales demos",
              "customer onboarding calls", "team availability management"],
    not_suitable_for=["project management", "task tracking"],
    recommended_executives=["CCO", "CRevO", "CMO"],
    auth_type="api_key", pricing_model="freemium", has_free_tier=True,
    requires_self_hosted=True,
    reliability="high", maturity="growing",
    api_docs_url="https://cal.com/docs/api-reference",
))

_reg(ToolProfile(
    name="shopforge", display_name="Shopforge Commerce Platform",
    category="business", availability="internal",
    description="Multi-storefront commerce management with pricing optimization, order routing, and autonomous analysis",
    strengths=["multi-storefront management", "pricing optimization", "margin analysis",
               "order routing", "inventory monitoring", "autonomous commerce analysis",
               "executive-tailored reports", "niche storefront provisioning"],
    limitations=["requires Shopify/Medusa backend", "single-tenant per instance"],
    best_for=["product creation", "pricing optimization", "order routing",
              "storefront management", "revenue tracking", "inventory management",
              "margin analysis", "storefront provisioning"],
    not_suitable_for=["payment processing (use Stripe)", "customer communication"],
    recommended_executives=["CRO", "CFO", "CMO", "CoS"],
    auth_type="service_token", pricing_model="internal", has_free_tier=False,
    supported_formats=["json"],
    reliability="high", maturity="established",
    api_docs_url="http://localhost:8003/docs",
))

_reg(ToolProfile(
    name="lago", display_name="Lago",
    category="business", availability="marketplace",
    description="Open-source metering and usage-based billing platform",
    strengths=["usage-based billing", "real-time metering", "invoice generation",
               "subscription management", "webhook events", "self-hostable"],
    limitations=["younger than Stripe Billing", "smaller ecosystem"],
    best_for=["usage-based pricing", "API metering", "SaaS billing",
              "consumption tracking", "invoice automation"],
    not_suitable_for=["one-time payments", "e-commerce checkout"],
    recommended_executives=["CFO", "CRevO", "CPO"],
    auth_type="api_key", pricing_model="free", has_free_tier=True,
    requires_self_hosted=True,
    reliability="medium", maturity="growing",
    api_docs_url="https://doc.getlago.com/api-reference/intro",
))

# ===== OPEN SOURCE — DOCUMENTS =====

_reg(ToolProfile(
    name="documenso", display_name="Documenso",
    category="documents", availability="marketplace",
    description="Open-source electronic signature platform (DocuSign alternative)",
    strengths=["self-hosted e-signatures", "signing workflows", "templates",
               "audit trail", "API access", "no per-envelope fees"],
    limitations=["less legal recognition than DocuSign in some jurisdictions",
                 "fewer integrations"],
    best_for=["contract signing", "NDA management", "employee onboarding docs",
              "vendor agreements"],
    not_suitable_for=["complex multi-party workflows", "notarized documents"],
    recommended_executives=["CLO", "CComO", "CFO"],
    auth_type="api_key", pricing_model="free", has_free_tier=True,
    requires_self_hosted=True,
    reliability="medium", maturity="growing",
    api_docs_url="https://documen.so/docs",
))

# ===== OPEN SOURCE — DEVOPS =====

_reg(ToolProfile(
    name="coolify", display_name="Coolify",
    category="devops", availability="marketplace",
    description="Open-source self-hostable PaaS (Heroku/Vercel alternative)",
    strengths=["one-click deployments", "Docker/Docker Compose support",
               "automatic SSL", "database provisioning", "git push deploy",
               "resource monitoring"],
    limitations=["single-server focus", "less mature than established PaaS"],
    best_for=["self-hosted application deployment", "staging environments",
              "hobby/indie projects", "Docker app hosting"],
    not_suitable_for=["large-scale multi-region deployment", "enterprise compliance"],
    recommended_executives=["CTO", "CEngO"],
    auth_type="none", pricing_model="free", has_free_tier=True,
    requires_self_hosted=True,
    reliability="medium", maturity="growing",
    api_docs_url="https://coolify.io/docs",
))

_reg(ToolProfile(
    name="gitea", display_name="Gitea",
    category="devops", availability="marketplace",
    description="Lightweight self-hosted Git service (GitHub alternative)",
    strengths=["lightweight", "fast", "self-hosted", "GitHub-like UI",
               "issue tracking", "pull requests", "CI/CD via Gitea Actions"],
    limitations=["smaller ecosystem than GitHub", "fewer integrations"],
    best_for=["self-hosted code hosting", "private repositories",
              "air-gapped environments", "data sovereignty"],
    not_suitable_for=["open-source community projects (use GitHub for visibility)"],
    recommended_executives=["CTO", "CEngO"],
    auth_type="api_key", pricing_model="free", has_free_tier=True,
    requires_self_hosted=True, requires_network=False,
    reliability="high", maturity="established",
    api_docs_url="https://gitea.com/api/swagger",
))

# ===== OPEN SOURCE — COMMUNICATION =====

_reg(ToolProfile(
    name="resend", display_name="Resend",
    category="communication", availability="integration",
    description="Modern email API built for developers (already used in Arclane)",
    strengths=["simple API", "React email templates", "webhooks",
               "deliverability tracking", "domain verification", "fast integration"],
    limitations=["newer service", "volume pricing at scale"],
    best_for=["transactional emails", "notification emails", "onboarding sequences",
              "password resets", "developer-friendly email"],
    not_suitable_for=["mass marketing campaigns", "complex email automation"],
    recommended_executives=["CTO", "CMO", "CCO"],
    auth_type="api_key", pricing_model="freemium", has_free_tier=True,
    reliability="high", maturity="growing",
    api_docs_url="https://resend.com/docs/api-reference",
))

# ===== OPEN SOURCE — SECURITY =====

_reg(ToolProfile(
    name="authentik", display_name="Authentik",
    category="security", availability="marketplace",
    description="Open-source identity provider with SSO, MFA, and user management",
    strengths=["SSO (SAML, OIDC, LDAP)", "MFA support", "user lifecycle management",
               "customizable flows", "self-hosted", "application proxy"],
    limitations=["complex initial setup", "resource requirements"],
    best_for=["single sign-on", "identity management", "access control",
              "user onboarding/offboarding", "multi-tenant auth"],
    not_suitable_for=["simple API key auth", "small single-app setups"],
    recommended_executives=["CSecO", "CIO", "CTO"],
    auth_type="oauth2", pricing_model="free", has_free_tier=True,
    requires_self_hosted=True,
    reliability="high", maturity="growing",
    api_docs_url="https://docs.goauthentik.io/docs/",
))

_reg(ToolProfile(
    name="trivy", display_name="Trivy",
    category="security", availability="marketplace",
    description="Comprehensive open-source security scanner (containers, IaC, code, deps)",
    strengths=["container image scanning", "IaC misconfiguration detection",
               "dependency vulnerability scanning", "SBOM generation",
               "CI/CD integration", "fast scanning"],
    limitations=["CLI-focused", "no web UI (use with Grafana/dashboard)"],
    best_for=["container security", "CI/CD security gates", "dependency auditing",
              "compliance scanning", "SBOM generation"],
    not_suitable_for=["runtime threat detection", "WAF replacement"],
    recommended_executives=["CSecO", "CEngO", "CTO"],
    auth_type="none", pricing_model="free", has_free_tier=True,
    requires_self_hosted=True, requires_network=False,
    reliability="high", maturity="established",
    api_docs_url="https://aquasecurity.github.io/trivy/",
))


# ===== CONTENT PRODUCTION (GozerAI Internal) =====

_reg(ToolProfile(
    name="content_production", display_name="Content Production",
    category="business", availability="internal",
    description=(
        "Autonomous content production pipeline. Generates full ebooks (DOCX, HTML, PDF, Markdown), "
        "60+ marketing assets, and distributes to 4 marketplaces. Fueled by Trendscope signals and "
        "Knowledge Harvester research. Supports 10 author personas with distinct voices and specialties."
    ),
    strengths=[
        "end-to-end ebook production (research, blueprint, chapters, assembly, distribution)",
        "authority-scored research with self-correcting critic loop",
        "60+ marketing assets per topic (social posts, video scripts, landing page, emails, ads)",
        "per-chapter Knowledge Harvester enrichment (3 KH integration points)",
        "per-section quality gates with auto-retry on rejection",
        "multi-format output (DOCX, HTML, PDF, Markdown)",
        "10 author personas with distinct writing voices and specialty matching",
        "multi-marketplace distribution (Gumroad, Etsy, Shopify, Amazon KDP)",
        "per-customer marketplace credentials for Arclane businesses",
        "revenue tracking with webhook forwarding",
        "23 pre-production roadmaps (~730 chapters) ready to produce",
        "blueprint generation (8-book series per topic, LLM-driven)",
    ],
    limitations=[
        "requires ANTHROPIC_API_KEY for LLM calls",
        "image generation requires separate CP_IMAGE_BACKEND config",
        "~12 minutes per ebook (~$0.78 per book at Claude Sonnet pricing)",
    ],
    best_for=[
        "ebook generation", "content marketing", "digital product creation",
        "marketplace listing", "revenue generation", "thought leadership content",
        "automated publishing", "series production", "lead magnet creation",
        "multi-platform distribution", "niche content at scale",
    ],
    not_suitable_for=["real-time chat content", "social media management", "video production"],
    recommended_executives=["CMO", "CRevO", "CPO", "CCO", "CoS"],
    auth_type="api_key", pricing_model="pay_per_use", has_free_tier=False,
    requires_self_hosted=True, requires_network=True,
    reliability="high", maturity="growing",
    api_docs_url="http://localhost:8013/docs",
))

_reg(ToolProfile(
    name="trendscope", display_name="Trendscope",
    category="research", availability="internal",
    description=(
        "Market trend intelligence service. Collects from Google Trends, Reddit, Hacker News, "
        "Product Hunt. Detects anomalies, generates buy/sell signals, identifies niche opportunities. "
        "Runs autonomously with hourly refresh. Internal services authenticate via X-Service-Token."
    ),
    strengths=[
        "real-time trend collection from 4+ public sources",
        "signal classification (strong_buy/buy/hold/sell/strong_sell)",
        "anomaly detection and drift analysis",
        "niche opportunity scoring",
        "executive-specific reports (CMO, CTO, CEO)",
        "autonomous scheduling (refresh every 60min, anomalies every 120min)",
        "internal service token auth for machine-to-machine calls",
    ],
    limitations=["signals need 2+ refresh cycles for velocity data", "no paid data sources yet"],
    best_for=[
        "market intelligence", "trend detection", "content opportunity discovery",
        "competitive analysis", "signal generation", "niche identification",
    ],
    not_suitable_for=["social media posting", "ad buying", "CRM"],
    recommended_executives=["CMO", "CSO", "CRevO", "CPO"],
    auth_type="api_key", pricing_model="free", has_free_tier=True,
    requires_self_hosted=True, requires_network=True,
    reliability="high", maturity="established",
    api_docs_url="http://localhost:8002/docs",
))

_reg(ToolProfile(
    name="knowledge_harvester", display_name="Knowledge Harvester",
    category="research", availability="internal",
    description=(
        "Knowledge artifact library. Harvests, classifies, scores, and embeds artifacts from 37+ sources. "
        "Maintains an intelligence graph with cluster analysis and coverage gap detection. "
        "Autonomous scheduling: auto-refresh stale artifacts, sync from Trendscope, generate recommendations."
    ),
    strengths=[
        "37+ source collectors",
        "authority scoring on sources",
        "intelligence graph with cluster analysis",
        "coverage gap detection (feeds Content Production topic selection)",
        "semantic search via pgvector embeddings",
        "artifact recommendations",
        "autonomous scheduling (refresh 4hr, TS sync 2hr, recommendations 6hr)",
        "feedback loop: Content Production publishes completed ebooks back as artifacts",
    ],
    limitations=["requires PostgreSQL with pgvector extension", "needs Docker for database"],
    best_for=[
        "knowledge management", "research context", "content enrichment",
        "coverage analysis", "artifact discovery", "intelligence graph",
    ],
    not_suitable_for=["real-time data processing", "transaction storage"],
    recommended_executives=["CTO", "CSO", "CPO", "CMO"],
    auth_type="none", pricing_model="free", has_free_tier=True,
    requires_self_hosted=True, requires_network=True,
    reliability="high", maturity="established",
    api_docs_url="http://localhost:8011/api/stats",
))


# ---------------------------------------------------------------------------
# Lookup API
# ---------------------------------------------------------------------------

def get_tool_profile(name: str) -> Optional[ToolProfile]:
    """Get profile for a specific tool by name."""
    return _TOOLS.get(name.lower())


def get_all_tool_profiles() -> Dict[str, ToolProfile]:
    """Return all registered tool profiles."""
    return dict(_TOOLS)


def get_tools_by_category(category: str) -> List[ToolProfile]:
    """Get all tools in a category."""
    return [t for t in _TOOLS.values() if t.category == category.lower()]


def get_tools_by_availability(availability: str) -> List[ToolProfile]:
    """Get tools by availability: 'internal', 'integration', 'marketplace'."""
    return [t for t in _TOOLS.values() if t.availability == availability.lower()]


def find_tools_for_task(
    task_description: str,
    *,
    category: Optional[str] = None,
    availability: Optional[str] = None,
    executive: Optional[str] = None,
    require_free: bool = False,
) -> List[ToolProfile]:
    """Find tools suited for a task.

    Matches against best_for, strengths, and description fields.
    Optionally filters by category, availability, executive, or pricing.
    """
    task_lower = task_description.lower()
    matches = []

    for tool in _TOOLS.values():
        # Apply filters
        if category and tool.category != category.lower():
            continue
        if availability and tool.availability != availability.lower():
            continue
        if executive and executive not in tool.recommended_executives:
            continue
        if require_free and not tool.has_free_tier:
            continue

        # Check not_suitable_for (exclude)
        if any(task_lower in ns.lower() for ns in tool.not_suitable_for):
            continue

        # Score relevance
        score = 0
        for bf in tool.best_for:
            if task_lower in bf.lower() or bf.lower() in task_lower:
                score += 2
        for s in tool.strengths:
            if task_lower in s.lower() or any(w in s.lower() for w in task_lower.split()):
                score += 1
        if task_lower in tool.description.lower():
            score += 1

        if score > 0:
            matches.append((score, tool))

    matches.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in matches]


def get_tools_for_executive(executive_code: str) -> List[ToolProfile]:
    """Get all tools recommended for a specific executive."""
    return [t for t in _TOOLS.values() if executive_code in t.recommended_executives]
