"""
Seed the Nexus Knowledge Base with foundational GozerAI knowledge.

Run this once after Nexus startup to populate the knowledge base with
product information, architectural decisions, and operational patterns
that executives need for informed decision-making.

Usage:
    python scripts/seed_knowledge.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nexus.memory.knowledge_base import KnowledgeBase, KnowledgeType

def seed():
    kb = KnowledgeBase(db_path=os.path.expanduser("~/.nexus/knowledge.db"))
    kb.initialize()

    items = [
        # GozerAI Product Portfolio
        ("GozerAI is the parent brand. Products: C-Suite (AI executive team), Arclane (AI startup incubator), Trendscope (market intelligence), Shopforge (e-commerce toolkit), Brandguard (brand monitoring), Taskpilot (task automation). All use GozerAI Telemetry for resilience.", KnowledgeType.FACTUAL, "gozerai:portfolio"),
        ("C-Suite has 15 executives + CoS (Overwatch) coordinator. Executives: CCO, CTO, CFO, CIO, CSO, CDO, CPO, CMO, CSecO, CComO, CEngO, CRiO, CRO, CRevO, CLO.", KnowledgeType.FACTUAL, "csuite:architecture"),
        ("Nexus is the shared AI infrastructure layer, not a strategic brain. It provides: model discovery (800+ models from OpenRouter), ensemble routing, RAG/knowledge base, research, tool/model profiles.", KnowledgeType.FACTUAL, "nexus:architecture"),
        ("Arclane is an AI-managed startup incubator. 90-day roadmap: Foundation→Validation→Growth→Scale-Ready. Post-graduation: adaptive optimizer. 5 billing plans from free to $499/month.", KnowledgeType.FACTUAL, "arclane:product"),
        ("Trendscope provides market intelligence via 6 intelligence domains. It integrates with Knowledge Harvester for data collection and Nexus for cross-product intelligence.", KnowledgeType.FACTUAL, "trendscope:product"),
        ("Zuultimate handles identity, authentication (JWT + API keys), and authorization. Vinzy-Engine manages licenses and activation. Together they form the auth/license stack.", KnowledgeType.FACTUAL, "auth:architecture"),

        # Architectural Decisions
        ("C-Suite uses OpenRouter as its primary LLM provider. For CRITICAL tasks, it routes through Nexus ensemble for multi-model consensus. Routine tasks use the local provider for cost efficiency.", KnowledgeType.PROCEDURAL, "csuite:llm_routing"),
        ("Inter-service communication between C-Suite and Nexus uses Redis pub/sub with channel prefix csuite:nexus. The NexusServiceClient sends requests, NexusServiceHandler dispatches to platform methods.", KnowledgeType.PROCEDURAL, "integration:architecture"),
        ("The persistent sync outbox (SQLite-backed) ensures no data loss during service outages. Events are written locally first, then delivered to Nexus. On reconnect, pending events drain automatically.", KnowledgeType.PROCEDURAL, "integration:resilience"),
        ("Directives v1.2 governs executive behavior: Balanced profile default ($2,500 max spend/task, 1000 tool invocations, 4hr compute). Absolute blocks: modify own directives, disable monitoring, delete logs, impersonate humans.", KnowledgeType.PROCEDURAL, "directives:policy"),
        ("New products/projects require CEO approval before first task execution. Budgets are per-task and travel through the delegation chain — CoS→CTO→CEngO all share the same task budget.", KnowledgeType.PROCEDURAL, "directives:budgets"),

        # Technology Stack
        ("C-Suite: Python 3.14, FastAPI, SQLAlchemy, PostgreSQL, Redis, Docker. Frontend: React/TypeScript.", KnowledgeType.FACTUAL, "csuite:tech_stack"),
        ("Nexus: Python 3.10+, Flask API, aiohttp service, SQLAlchemy, SQLite + Qdrant (vector search), Redis bridge.", KnowledgeType.FACTUAL, "nexus:tech_stack"),
        ("Infrastructure tools deployed: Langfuse (LLM observability, :3100), Qdrant (vector DB, :6333), Uptime Kuma (monitoring, :3001), Prometheus + Grafana (metrics).", KnowledgeType.FACTUAL, "infrastructure:tools"),
        ("All repos use GitHub Actions CI. C-Suite and Zuultimate include alembic migration verification. GozerAI Telemetry provides shared resilience (RetryPolicy, CircuitBreaker) across all Python services.", KnowledgeType.FACTUAL, "infrastructure:ci"),

        # Operational Patterns
        ("LLM cost optimization: use Ollama for local inference on routine tasks, OpenRouter for API-based models, Nexus ensemble for critical decisions only. Track costs via Langfuse when configured.", KnowledgeType.EXPERIENTIAL, "ops:cost_optimization"),
        ("When debugging C-Suite: check tool authorization first (common source of 'task failed' errors), then LLM provider connectivity, then Redis bridge status. Use /health endpoint for quick checks.", KnowledgeType.EXPERIENTIAL, "ops:debugging"),
        ("For cross-product intelligence: products publish to Nexus knowledge base, C-Suite consumes via search_knowledge. Arclane already uses NexusPublisher pattern. Other products should follow.", KnowledgeType.EXPERIENTIAL, "ops:data_flow"),

        # Business Context
        ("GozerAI targets SaaS businesses and startups. Pricing: subscription-based for standalone products (Trendscope, Shopforge, etc.), usage-based for Arclane (credits per cycle). Zuultimate handles all auth and licensing.", KnowledgeType.FACTUAL, "business:model"),
        ("The website gozerai.com is WordPress Multisite on Hostinger VPS. Each product has its own subsite. arclane.cloud is a separate domain.", KnowledgeType.FACTUAL, "business:web"),
        ("1450 Enterprises is the holding company. GozerAI is the AI brand. Chris Arsenault is CEO/Founder/sole shareholder.", KnowledgeType.FACTUAL, "business:ownership"),
    ]

    added = 0
    for content, ktype, source in items:
        try:
            kb.add_knowledge(content, ktype, source, confidence=0.95, context_tags=["seed", "gozerai"])
            added += 1
        except Exception as e:
            print(f"Failed to add: {e}")

    print(f"Seeded {added}/{len(items)} knowledge items")
    stats = kb.get_knowledge_statistics()
    print(f"Knowledge base stats: {stats['total_items']} items, {stats.get('persistent', False)} persistent")
    kb.close()


if __name__ == "__main__":
    seed()
