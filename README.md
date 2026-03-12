# Nexus

**AI Orchestration Platform -- Memory, RAG, Multi-Provider Inference**

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

Nexus is a production-ready platform for building AI applications with pluggable memory, retrieval-augmented generation, multi-provider model access, and cost tracking. Part of the [GozerAI](https://gozerai.com) ecosystem.

---

## Features

- **Multi-Provider Inference** -- Unified adapter layer for OpenAI, Anthropic, and more
- **Memory System** -- Factual memory, skill memory, pattern recognition, knowledge validation, gap detection
- **RAG Pipeline** -- Vector search with FAISS/HNSW, hybrid BM25 + semantic retrieval, cross-encoder reranking
- **Cost Tracking** -- Per-request cost metering across providers
- **Safety & Resilience** -- Circuit breakers, model quarantine, rate limiting
- **Monitoring** -- Prometheus metrics, health checks, usage analytics
- **API Server** -- RESTful endpoints for memory, RAG, and reasoning
- **CLI** -- Interactive chat interface
- **MCP Server** -- Model Context Protocol integration

---

## Quick Start

### Installation

```bash
# From PyPI
pip install nexus-ai

# From source
git clone https://github.com/GozerAI/nexus.git
cd nexus
pip install -e .

# With API server dependencies
pip install -e ".[server]"

# With embedding model support
pip install -e ".[embeddings]"
```

### Basic Usage

#### RAG Pipeline

```python
from nexus.rag import create_rag

# Create a RAG instance with sensible defaults
rag = create_rag()

# Add documents
rag.add_document("Nexus is an AI orchestration platform.")
rag.add_document("It supports multiple providers and memory systems.")

# Query
results = rag.query("What does Nexus do?")
for result in results:
    print(f"{result.text} (score: {result.score:.2f})")
```

#### Memory System

```python
from nexus.memory import KnowledgeBase, FactualMemoryEngine

# Knowledge base with 45+ built-in domains
kb = KnowledgeBase()
kb.add_knowledge("python", "Python supports async/await for concurrency.")
items = kb.search("async programming")

# Factual memory with provenance tracking
memory = FactualMemoryEngine()
memory.store_fact("The speed of light is 299,792,458 m/s", source="physics")
facts = memory.recall("speed of light")
```

#### Multi-Provider Inference

```python
from nexus.providers import OpenAIModelAdapter, AnthropicModelAdapter, CostTracker

# Use any supported provider through a unified interface
adapter = OpenAIModelAdapter(model="gpt-4o", api_key="sk-...")
response = await adapter.generate("Explain quantum computing briefly.")

# Track costs across providers
tracker = CostTracker()
summary = tracker.get_summary()
print(f"Total cost: ${summary['total_cost']:.4f}")
```

#### Safety & Resilience

```python
from nexus.providers import CircuitBreaker

# Circuit breaker prevents cascading failures
breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
```

### Run the API Server

```bash
# Install server dependencies
pip install -e ".[server]"

# Start the server (default: http://localhost:5000)
nexus-api
```

### API Endpoints

#### Memory (Community)

```bash
# Store knowledge
curl -X POST http://localhost:5000/api/v1/memory/knowledge \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"domain": "python", "content": "Python 3.12 adds type parameter syntax."}'

# Search knowledge
curl http://localhost:5000/api/v1/memory/knowledge/search?query=python \
  -H "X-API-Key: your-api-key"

# Health check
curl http://localhost:5000/api/v1/memory/health
```

#### RAG (Community)

```bash
# Add a document
curl -X POST http://localhost:5000/api/v1/rag/documents \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"text": "Your document content here."}'

# Query
curl -X POST http://localhost:5000/api/v1/rag/query \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is in my documents?"}'

# Health check
curl http://localhost:5000/api/v1/rag/health
```

---

## Feature Tiers

| Feature | Community | Pro | Enterprise |
|---------|:---------:|:---:|:----------:|
| Memory system (factual, skill, patterns) | Yes | Yes | Yes |
| RAG pipeline (FAISS, hybrid search) | Yes | Yes | Yes |
| Multi-provider adapters (OpenAI, Anthropic) | Yes | Yes | Yes |
| Cost tracking & usage analytics | Yes | Yes | Yes |
| Safety (circuit breaker, quarantine) | Yes | Yes | Yes |
| Monitoring & health checks | Yes | Yes | Yes |
| API server & CLI | Yes | Yes | Yes |
| MCP server | Yes | Yes | Yes |
| Orchestration & observatory | Yes | Yes | Yes |
| Multi-model ensemble | -- | Yes | Yes |
| Advanced reasoning chains | -- | Yes | Yes |
| Discovery & intelligence | -- | -- | Yes |
| Strategic analysis | -- | -- | Yes |

Community tier is fully functional for building AI applications. Pro and Enterprise unlock advanced orchestration and analysis capabilities.

---

## Unlocking Pro & Enterprise Features

Set your license key as an environment variable:

```bash
export VINZY_LICENSE_KEY="your-license-key"
export VINZY_SERVER="https://api.gozerai.com"  # or your self-hosted Vinzy instance
```

Without a license key, Nexus runs in Community mode. Gated features return a clear error with upgrade instructions. If the license server is unreachable, gated features remain locked (fail-closed).

Visit **[gozerai.com/pricing](https://gozerai.com/pricing)** to purchase a license.

---

## Configuration

Nexus uses environment variables for provider access and licensing:

```bash
# Required for provider access
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Optional: license key for Pro/Enterprise features
export VINZY_LICENSE_KEY="your-key"
export VINZY_SERVER="https://api.gozerai.com"
```

---

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## License

Nexus is dual-licensed:

- **[AGPL-3.0](LICENSE)** -- Free for open-source use with copyleft obligations
- **Commercial License** -- For proprietary use without AGPL requirements

Visit [gozerai.com/pricing](https://gozerai.com/pricing) for commercial licensing. See [LICENSING.md](LICENSING.md) for tier details.

---

## Contributing

We welcome contributions. Please see our [Contributing Guide](CONTRIBUTING.md) for details, including the required Contributor License Agreement (CLA).

---

## Links

- [GozerAI](https://gozerai.com) -- Main site
- [Pricing](https://gozerai.com/pricing) -- License tiers and pricing
- [Issues](https://github.com/GozerAI/nexus/issues) -- Bug reports and feature requests
- [Discussions](https://github.com/GozerAI/nexus/discussions) -- Community Q&A

Copyright (c) 2025-2026 GozerAI.
