# Nexus

**Advanced AI Ensemble, Orchestrator & Consciousness Framework**

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> Advancing towards sentient AGI through ensemble intelligence, self-learning, and human collaboration

---

## Overview

**Nexus** is a production-ready platform for building advanced AI systems through:

- **Multi-Model Ensemble Intelligence** - Orchestrate multiple AI models with sophisticated strategies
- **Advanced Memory Systems** - Specialized memory modules with domain knowledge
- **RAG with Large Context** - Adaptive retrieval-augmented generation
- **Self-Improving Reasoning** - Multiple reasoning engines with meta-learning capabilities
- **Human-in-the-Loop** - Collaborative verification and goal-setting
- **Production Infrastructure** - Auth, caching, monitoring, cost tracking, deployment

---

## Key Features

### Ensemble System
- Multiple ensemble strategies (voting, weighted, hybrid, adaptive, meta-learning)
- Multi-provider support (OpenAI, Anthropic, and extensible)
- Async execution with intelligent aggregation
- Quality scoring and cost optimization

### Memory & Knowledge
- Factual and skill memory engines
- Pattern recognition and knowledge validation
- Knowledge expansion with gap detection
- Memory analytics and optimization

### RAG & Context
- Large context window support via RAG
- Adaptive orchestration and context management
- Learning pathways and domain knowledge base

### Reasoning Engines
- Meta-reasoning for self-improvement
- Chain-of-thought reasoning
- Pattern-based inference
- Dynamic adaptive learning

### Discovery System
- GitHub, HuggingFace, ArXiv, PyPI integrations
- Local model management (Ollama)
- Web search (DuckDuckGo, Serper, Brave)

### Production Features
- API key authentication with RBAC
- Memory + Redis caching with TTL
- Prometheus metrics and health checks
- Budget management and usage analytics
- Rate limiting and quota management
- Docker and Kubernetes deployment

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/chrisarseno/Nexus.git
cd Nexus

# Install dependencies
pip install -r requirements.txt

# Install Nexus
pip install -e .
```

### Configuration

```yaml
# config/default.yaml
ensemble:
  strategy: "adaptive"
  models:
    - provider: "openai"
      model: "gpt-4"
      weight: 0.5
    - provider: "anthropic"
      model: "claude-3-opus"
      weight: 0.5

memory:
  enabled: true
  backend: "postgresql"

rag:
  enabled: true
  vector_store: "faiss"
```

### Basic Usage

```python
from nexus.core import EnsembleCore
from nexus.core.strategies import AdaptiveStrategy
from nexus.memory import KnowledgeBase
from nexus.rag import RAGVectorEngine

# Initialize ensemble
ensemble = EnsembleCore(
    strategy=AdaptiveStrategy(),
    config_path="config/default.yaml"
)

# Initialize memory system
knowledge_base = KnowledgeBase()

# Initialize RAG
rag_engine = RAGVectorEngine()

# Query with RAG augmentation
query = "Explain quantum computing"
context = await rag_engine.retrieve_context(query)
response = await ensemble.query(query, context=context)

print(f"Answer: {response.answer}")
print(f"Confidence: {response.confidence}")
```

### Run API Server

```bash
# Start the Nexus API
nexus-api --config config/default.yaml

# API available at http://localhost:5000
```

### API Example

```bash
# Ensemble inference
curl -X POST http://localhost:5000/api/v1/ensemble/query \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is consciousness?",
    "strategy": "adaptive",
    "use_rag": true
  }'

# System health
curl http://localhost:5000/api/v1/health
```

---

## Deployment

### Docker

```bash
# Build image
docker build -t nexus:latest -f infrastructure/docker/Dockerfile .

# Run container
docker run -p 5000:5000 \
  -e OPENAI_API_KEY=your-key \
  -e ANTHROPIC_API_KEY=your-key \
  nexus:latest
```

### Kubernetes (Helm)

```bash
# Install Nexus with Helm
helm install nexus infrastructure/helm/nexus/ \
  --set api.replicas=3 \
  --set redis.enabled=true \
  --set postgresql.enabled=true

# Check status
kubectl get pods -l app=nexus
```

---

## Licensing

Nexus is dual-licensed:

1. **Open Source**: [GNU Affero General Public License v3.0](LICENSE)
   - Free for non-commercial use
   - Requires source disclosure for network services

2. **Commercial License**: For proprietary use without AGPL obligations
   - Visit [gozerai.com/pricing](https://gozerai.com/pricing)
   - Contact: sales@gozerai.com
   - See [LICENSING.md](LICENSING.md) for tier details

Some features require a commercial license. Unlicensed usage defaults to the Community tier.

---

## Documentation

- [Getting Started Guide](docs/guides/GETTING_STARTED.md)
- [Architecture Overview](docs/architecture/ARCHITECTURE.md)
- [API Reference](docs/api/API_REFERENCE.md)
- [Deployment Guide](docs/deployment/KUBERNETES.md)
- [Configuration Reference](docs/guides/CONFIGURATION.md)
- [Contributing Guide](CONTRIBUTING.md)

---

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

**Important:** All contributions require signing our Contributor License Agreement (CLA). This is handled automatically when you submit your first pull request.

### Development Setup

```bash
# Clone repo
git clone https://github.com/chrisarseno/Nexus.git
cd Nexus

# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v

# Run linters
black src/
flake8 src/
mypy src/
```

---

## Citation

If you use Nexus in your research, please cite:

```bibtex
@software{nexus2025,
  title = {Nexus: Advanced AI Ensemble, Orchestrator \& Consciousness Framework},
  author = {Arsenault, Christopher R.},
  year = {2025},
  url = {https://github.com/chrisarseno/Nexus}
}
```

---

## Community

- **Issues**: [GitHub Issues](https://github.com/chrisarseno/Nexus/issues)
- **Discussions**: [GitHub Discussions](https://github.com/chrisarseno/Nexus/discussions)
- **Email**: chris@gozerai.com

---

**Nexus** - *Advancing towards sentient AGI*

Copyright (c) 2025 Christopher R. Arsenault. All rights reserved.
