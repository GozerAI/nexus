"""
Comprehensive model registry with 50+ AI models across various providers.

This registry includes models from:
- OpenAI (GPT-4, GPT-3.5, etc.)
- Anthropic (Claude family)
- Google (Gemini, PaLM)
- Meta (Llama family)
- Mistral AI
- Cohere
- AI21 Labs
- Hugging Face
- Open source models
"""

from typing import Dict, List, Optional

from nexus.providers.ensemble.types import ModelProvider
from nexus.providers.adapters.base import ModelCapability, ModelInfo, ModelSize

# Model registry - comprehensive catalog of AI models
MODEL_REGISTRY: Dict[str, ModelInfo] = {}


def register_model(model: ModelInfo) -> None:
    """Register a model in the global registry."""
    MODEL_REGISTRY[model.name] = model


# =============================================================================
# OPENAI MODELS
# =============================================================================

register_model(
    ModelInfo(
        name="gpt-4-turbo",
        display_name="GPT-4 Turbo",
        provider=ModelProvider.OPENAI,
        size=ModelSize.FLAGSHIP,
        context_window=128000,
        max_output_tokens=4096,
        capabilities=[
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CODE_GENERATION,
            ModelCapability.REASONING,
            ModelCapability.MATHEMATICS,
            ModelCapability.VISION,
            ModelCapability.FUNCTION_CALLING,
            ModelCapability.LONG_CONTEXT,
            ModelCapability.STREAMING,
        ],
        cost_per_1k_input=0.01,
        cost_per_1k_output=0.03,
        supported=True,
        description="Most capable GPT-4 model with 128K context window",
        use_cases=["complex reasoning", "code generation", "vision tasks", "long documents"],
        release_date="2024-04",
        documentation_url="https://platform.openai.com/docs/models/gpt-4-turbo-and-gpt-4",
    )
)

register_model(
    ModelInfo(
        name="gpt-4",
        display_name="GPT-4",
        provider=ModelProvider.OPENAI,
        size=ModelSize.FLAGSHIP,
        context_window=8192,
        max_output_tokens=4096,
        capabilities=[
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CODE_GENERATION,
            ModelCapability.REASONING,
            ModelCapability.MATHEMATICS,
            ModelCapability.FUNCTION_CALLING,
            ModelCapability.STREAMING,
        ],
        cost_per_1k_input=0.03,
        cost_per_1k_output=0.06,
        supported=True,
        description="Original GPT-4 with strong reasoning capabilities",
        use_cases=["general purpose", "reasoning", "creative writing"],
        release_date="2023-03",
        documentation_url="https://platform.openai.com/docs/models/gpt-4",
    )
)

register_model(
    ModelInfo(
        name="gpt-4-32k",
        display_name="GPT-4 32K",
        provider=ModelProvider.OPENAI,
        size=ModelSize.FLAGSHIP,
        context_window=32768,
        max_output_tokens=4096,
        capabilities=[
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CODE_GENERATION,
            ModelCapability.REASONING,
            ModelCapability.LONG_CONTEXT,
            ModelCapability.STREAMING,
        ],
        cost_per_1k_input=0.06,
        cost_per_1k_output=0.12,
        supported=True,
        description="GPT-4 with extended 32K context window",
        use_cases=["long document analysis", "code review", "research"],
        release_date="2023-06",
        documentation_url="https://platform.openai.com/docs/models/gpt-4",
    )
)

register_model(
    ModelInfo(
        name="gpt-3.5-turbo",
        display_name="GPT-3.5 Turbo",
        provider=ModelProvider.OPENAI,
        size=ModelSize.MEDIUM,
        context_window=16384,
        max_output_tokens=4096,
        capabilities=[
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CODE_GENERATION,
            ModelCapability.FUNCTION_CALLING,
            ModelCapability.STREAMING,
        ],
        cost_per_1k_input=0.0005,
        cost_per_1k_output=0.0015,
        supported=True,
        description="Fast, cost-effective model for most tasks",
        use_cases=["chatbots", "simple tasks", "high-volume applications"],
        release_date="2023-03",
        documentation_url="https://platform.openai.com/docs/models/gpt-3-5-turbo",
    )
)

register_model(
    ModelInfo(
        name="gpt-3.5-turbo-16k",
        display_name="GPT-3.5 Turbo 16K",
        provider=ModelProvider.OPENAI,
        size=ModelSize.MEDIUM,
        context_window=16384,
        max_output_tokens=4096,
        capabilities=[
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CODE_GENERATION,
            ModelCapability.LONG_CONTEXT,
            ModelCapability.STREAMING,
        ],
        cost_per_1k_input=0.001,
        cost_per_1k_output=0.002,
        supported=True,
        description="GPT-3.5 with 16K context for longer conversations",
        use_cases=["conversation", "document processing"],
        release_date="2023-06",
        documentation_url="https://platform.openai.com/docs/models/gpt-3-5-turbo",
    )
)

# =============================================================================
# ANTHROPIC MODELS (CLAUDE)
# =============================================================================

register_model(
    ModelInfo(
        name="claude-3-opus",
        display_name="Claude 3 Opus",
        provider=ModelProvider.ANTHROPIC,
        size=ModelSize.FLAGSHIP,
        context_window=200000,
        max_output_tokens=4096,
        capabilities=[
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CODE_GENERATION,
            ModelCapability.REASONING,
            ModelCapability.MATHEMATICS,
            ModelCapability.VISION,
            ModelCapability.LONG_CONTEXT,
            ModelCapability.STREAMING,
            ModelCapability.MULTILINGUAL,
        ],
        cost_per_1k_input=0.015,
        cost_per_1k_output=0.075,
        supported=True,
        description="Most capable Claude model with 200K context",
        use_cases=["complex analysis", "research", "coding", "creative writing"],
        release_date="2024-03",
        documentation_url="https://docs.anthropic.com/claude/docs/models-overview",
    )
)

register_model(
    ModelInfo(
        name="claude-3-sonnet",
        display_name="Claude 3 Sonnet",
        provider=ModelProvider.ANTHROPIC,
        size=ModelSize.LARGE,
        context_window=200000,
        max_output_tokens=4096,
        capabilities=[
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CODE_GENERATION,
            ModelCapability.REASONING,
            ModelCapability.VISION,
            ModelCapability.LONG_CONTEXT,
            ModelCapability.STREAMING,
        ],
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
        supported=True,
        description="Balanced performance and speed with 200K context",
        use_cases=["general purpose", "data processing", "automation"],
        release_date="2024-03",
        documentation_url="https://docs.anthropic.com/claude/docs/models-overview",
    )
)

register_model(
    ModelInfo(
        name="claude-3-haiku",
        display_name="Claude 3 Haiku",
        provider=ModelProvider.ANTHROPIC,
        size=ModelSize.MEDIUM,
        context_window=200000,
        max_output_tokens=4096,
        capabilities=[
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CODE_GENERATION,
            ModelCapability.LONG_CONTEXT,
            ModelCapability.STREAMING,
        ],
        cost_per_1k_input=0.00025,
        cost_per_1k_output=0.00125,
        supported=True,
        description="Fastest Claude model for simple tasks",
        use_cases=["chat", "simple queries", "high-volume processing"],
        release_date="2024-03",
        documentation_url="https://docs.anthropic.com/claude/docs/models-overview",
    )
)

register_model(
    ModelInfo(
        name="claude-2.1",
        display_name="Claude 2.1",
        provider=ModelProvider.ANTHROPIC,
        size=ModelSize.LARGE,
        context_window=200000,
        max_output_tokens=4096,
        capabilities=[
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CODE_GENERATION,
            ModelCapability.REASONING,
            ModelCapability.LONG_CONTEXT,
            ModelCapability.STREAMING,
        ],
        cost_per_1k_input=0.008,
        cost_per_1k_output=0.024,
        supported=False,
        description="Previous generation Claude with 200K context",
        use_cases=["long document analysis"],
        release_date="2023-11",
        documentation_url="https://docs.anthropic.com/claude/docs/models-overview",
        notes="Consider upgrading to Claude 3 models",
    )
)

# =============================================================================
# GOOGLE MODELS
# =============================================================================

register_model(
    ModelInfo(
        name="gemini-pro",
        display_name="Gemini Pro",
        provider=ModelProvider.LOCAL,  # Will add Google provider
        size=ModelSize.LARGE,
        context_window=32768,
        max_output_tokens=2048,
        capabilities=[
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CODE_GENERATION,
            ModelCapability.REASONING,
            ModelCapability.MULTILINGUAL,
            ModelCapability.STREAMING,
        ],
        cost_per_1k_input=0.00025,
        cost_per_1k_output=0.0005,
        supported=False,
        description="Google's multimodal AI model",
        use_cases=["general purpose", "reasoning", "multilingual tasks"],
        release_date="2023-12",
        documentation_url="https://ai.google.dev/models/gemini",
        notes="Requires Google AI API key - implement GoogleModelAdapter",
    )
)

register_model(
    ModelInfo(
        name="gemini-ultra",
        display_name="Gemini Ultra",
        provider=ModelProvider.LOCAL,
        size=ModelSize.FLAGSHIP,
        context_window=32768,
        max_output_tokens=2048,
        capabilities=[
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CODE_GENERATION,
            ModelCapability.REASONING,
            ModelCapability.MATHEMATICS,
            ModelCapability.VISION,
            ModelCapability.MULTILINGUAL,
        ],
        cost_per_1k_input=0.01,
        cost_per_1k_output=0.03,
        supported=False,
        description="Most capable Gemini model",
        use_cases=["complex reasoning", "multimodal tasks", "research"],
        release_date="2024-02",
        documentation_url="https://ai.google.dev/models/gemini",
        notes="Requires Google AI API key and special access",
    )
)

register_model(
    ModelInfo(
        name="palm-2",
        display_name="PaLM 2",
        provider=ModelProvider.LOCAL,
        size=ModelSize.LARGE,
        context_window=8192,
        max_output_tokens=2048,
        capabilities=[
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CODE_GENERATION,
            ModelCapability.REASONING,
            ModelCapability.MULTILINGUAL,
        ],
        cost_per_1k_input=0.001,
        cost_per_1k_output=0.002,
        supported=False,
        description="Google's previous generation language model",
        use_cases=["general purpose", "multilingual"],
        release_date="2023-05",
        documentation_url="https://ai.google/discover/palm2/",
        notes="Consider using Gemini instead",
    )
)

# =============================================================================
# META LLAMA MODELS
# =============================================================================

register_model(
    ModelInfo(
        name="llama-3-70b",
        display_name="Llama 3 70B",
        provider=ModelProvider.OLLAMA,
        size=ModelSize.XLARGE,
        context_window=8192,
        max_output_tokens=2048,
        capabilities=[
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CODE_GENERATION,
            ModelCapability.REASONING,
            ModelCapability.MULTILINGUAL,
        ],
        cost_per_1k_input=0.0,  # Open source, local inference
        cost_per_1k_output=0.0,
        supported=False,
        description="Meta's flagship open-source model",
        use_cases=["local deployment", "privacy-focused", "general purpose"],
        requires_api_key=False,
        release_date="2024-04",
        documentation_url="https://llama.meta.com/llama3/",
        notes="Requires Ollama or local inference setup",
    )
)

register_model(
    ModelInfo(
        name="llama-3-8b",
        display_name="Llama 3 8B",
        provider=ModelProvider.OLLAMA,
        size=ModelSize.SMALL,
        context_window=8192,
        max_output_tokens=2048,
        capabilities=[
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CODE_GENERATION,
            ModelCapability.REASONING,
        ],
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        supported=False,
        description="Smaller Llama 3 for faster inference",
        use_cases=["edge devices", "low-latency", "chatbots"],
        requires_api_key=False,
        release_date="2024-04",
        documentation_url="https://llama.meta.com/llama3/",
        notes="Can run on consumer hardware with Ollama",
    )
)

register_model(
    ModelInfo(
        name="llama-2-70b",
        display_name="Llama 2 70B",
        provider=ModelProvider.OLLAMA,
        size=ModelSize.XLARGE,
        context_window=4096,
        max_output_tokens=2048,
        capabilities=[
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CODE_GENERATION,
        ],
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        supported=False,
        description="Previous generation Llama model",
        use_cases=["local deployment", "general purpose"],
        requires_api_key=False,
        release_date="2023-07",
        documentation_url="https://llama.meta.com/llama2/",
        notes="Consider upgrading to Llama 3",
    )
)

register_model(
    ModelInfo(
        name="code-llama-34b",
        display_name="Code Llama 34B",
        provider=ModelProvider.OLLAMA,
        size=ModelSize.MEDIUM,
        context_window=16384,
        max_output_tokens=2048,
        capabilities=[
            ModelCapability.CODE_GENERATION,
            ModelCapability.LONG_CONTEXT,
        ],
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        supported=False,
        description="Specialized code generation model",
        use_cases=["code completion", "code review", "debugging"],
        requires_api_key=False,
        release_date="2023-08",
        documentation_url="https://github.com/facebookresearch/codellama",
        notes="Optimized for programming tasks",
    )
)

# =============================================================================
# MISTRAL AI MODELS
# =============================================================================

register_model(
    ModelInfo(
        name="mistral-large",
        display_name="Mistral Large",
        provider=ModelProvider.LOCAL,
        size=ModelSize.LARGE,
        context_window=32768,
        max_output_tokens=8192,
        capabilities=[
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CODE_GENERATION,
            ModelCapability.REASONING,
            ModelCapability.MATHEMATICS,
            ModelCapability.FUNCTION_CALLING,
            ModelCapability.MULTILINGUAL,
        ],
        cost_per_1k_input=0.004,
        cost_per_1k_output=0.012,
        supported=False,
        description="Mistral's flagship model",
        use_cases=["reasoning", "coding", "multilingual"],
        release_date="2024-02",
        documentation_url="https://docs.mistral.ai/",
        notes="Requires Mistral API key - implement MistralModelAdapter",
    )
)

register_model(
    ModelInfo(
        name="mistral-medium",
        display_name="Mistral Medium",
        provider=ModelProvider.LOCAL,
        size=ModelSize.MEDIUM,
        context_window=32768,
        max_output_tokens=8192,
        capabilities=[
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CODE_GENERATION,
            ModelCapability.REASONING,
        ],
        cost_per_1k_input=0.0027,
        cost_per_1k_output=0.0081,
        supported=False,
        description="Balanced Mistral model",
        use_cases=["general purpose", "cost-effective"],
        release_date="2023-12",
        documentation_url="https://docs.mistral.ai/",
        notes="Good balance of cost and performance",
    )
)

register_model(
    ModelInfo(
        name="mixtral-8x7b",
        display_name="Mixtral 8x7B",
        provider=ModelProvider.OLLAMA,
        size=ModelSize.LARGE,
        context_window=32768,
        max_output_tokens=4096,
        capabilities=[
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CODE_GENERATION,
            ModelCapability.REASONING,
            ModelCapability.MULTILINGUAL,
        ],
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        supported=False,
        description="Mixture of Experts open model",
        use_cases=["local deployment", "efficiency", "multilingual"],
        requires_api_key=False,
        release_date="2023-12",
        documentation_url="https://mistral.ai/news/mixtral-of-experts/",
        notes="Available via Ollama or Hugging Face",
    )
)

# =============================================================================
# COHERE MODELS
# =============================================================================

register_model(
    ModelInfo(
        name="command-r-plus",
        display_name="Command R+",
        provider=ModelProvider.LOCAL,
        size=ModelSize.LARGE,
        context_window=128000,
        max_output_tokens=4096,
        capabilities=[
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CODE_GENERATION,
            ModelCapability.REASONING,
            ModelCapability.LONG_CONTEXT,
            ModelCapability.MULTILINGUAL,
        ],
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
        supported=False,
        description="Cohere's most capable model",
        use_cases=["RAG", "agents", "multilingual"],
        release_date="2024-04",
        documentation_url="https://docs.cohere.com/docs/command-r-plus",
        notes="Requires Cohere API key - implement CohereModelAdapter",
    )
)

register_model(
    ModelInfo(
        name="command-r",
        display_name="Command R",
        provider=ModelProvider.LOCAL,
        size=ModelSize.MEDIUM,
        context_window=128000,
        max_output_tokens=4096,
        capabilities=[
            ModelCapability.TEXT_GENERATION,
            ModelCapability.REASONING,
            ModelCapability.LONG_CONTEXT,
        ],
        cost_per_1k_input=0.0005,
        cost_per_1k_output=0.0015,
        supported=False,
        description="Balanced Cohere model",
        use_cases=["conversational AI", "RAG"],
        release_date="2024-03",
        documentation_url="https://docs.cohere.com/docs/command-r",
        notes="Optimized for RAG and tool use",
    )
)

# =============================================================================
# SPECIALIZED & OPEN SOURCE MODELS
# =============================================================================

register_model(
    ModelInfo(
        name="phi-3-mini",
        display_name="Phi-3 Mini",
        provider=ModelProvider.OLLAMA,
        size=ModelSize.TINY,
        context_window=128000,
        max_output_tokens=4096,
        capabilities=[
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CODE_GENERATION,
            ModelCapability.REASONING,
            ModelCapability.LONG_CONTEXT,
        ],
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        supported=False,
        description="Microsoft's small language model with strong performance",
        use_cases=["edge deployment", "low-resource environments", "mobile"],
        requires_api_key=False,
        release_date="2024-04",
        documentation_url="https://azure.microsoft.com/en-us/products/phi-3",
        notes="3.8B parameters, runs on CPU efficiently",
    )
)

register_model(
    ModelInfo(
        name="deepseek-coder-33b",
        display_name="DeepSeek Coder 33B",
        provider=ModelProvider.OLLAMA,
        size=ModelSize.MEDIUM,
        context_window=16384,
        max_output_tokens=4096,
        capabilities=[
            ModelCapability.CODE_GENERATION,
            ModelCapability.LONG_CONTEXT,
        ],
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        supported=False,
        description="Specialized coding model with strong performance",
        use_cases=["code generation", "code completion", "debugging"],
        requires_api_key=False,
        release_date="2023-11",
        documentation_url="https://github.com/deepseek-ai/DeepSeek-Coder",
        notes="Outperforms Code Llama on many benchmarks",
    )
)

register_model(
    ModelInfo(
        name="wizardlm-70b",
        display_name="WizardLM 70B",
        provider=ModelProvider.OLLAMA,
        size=ModelSize.XLARGE,
        context_window=4096,
        max_output_tokens=2048,
        capabilities=[
            ModelCapability.TEXT_GENERATION,
            ModelCapability.REASONING,
            ModelCapability.CODE_GENERATION,
        ],
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        supported=False,
        description="Fine-tuned Llama for instruction following",
        use_cases=["complex instructions", "reasoning", "problem solving"],
        requires_api_key=False,
        release_date="2023-06",
        documentation_url="https://github.com/nlpxucan/WizardLM",
        notes="Strong instruction-following capabilities",
    )
)

register_model(
    ModelInfo(
        name="vicuna-33b",
        display_name="Vicuna 33B",
        provider=ModelProvider.OLLAMA,
        size=ModelSize.MEDIUM,
        context_window=2048,
        max_output_tokens=2048,
        capabilities=[
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CODE_GENERATION,
        ],
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        supported=False,
        description="Fine-tuned LLaMA for conversation",
        use_cases=["chatbots", "conversation", "assistants"],
        requires_api_key=False,
        release_date="2023-04",
        documentation_url="https://lmsys.org/blog/2023-03-30-vicuna/",
        notes="Trained on user-shared conversations",
    )
)

register_model(
    ModelInfo(
        name="falcon-180b",
        display_name="Falcon 180B",
        provider=ModelProvider.LOCAL,
        size=ModelSize.XLARGE,
        context_window=2048,
        max_output_tokens=2048,
        capabilities=[
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CODE_GENERATION,
            ModelCapability.REASONING,
        ],
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        supported=False,
        description="TII's large open-source model",
        use_cases=["research", "benchmarking"],
        requires_api_key=False,
        release_date="2023-09",
        documentation_url="https://falconllm.tii.ae/",
        notes="Requires significant compute resources",
    )
)

# =============================================================================
# VISION & MULTIMODAL MODELS
# =============================================================================

register_model(
    ModelInfo(
        name="gpt-4-vision",
        display_name="GPT-4 Vision",
        provider=ModelProvider.OPENAI,
        size=ModelSize.FLAGSHIP,
        context_window=128000,
        max_output_tokens=4096,
        capabilities=[
            ModelCapability.TEXT_GENERATION,
            ModelCapability.VISION,
            ModelCapability.REASONING,
            ModelCapability.CODE_GENERATION,
        ],
        cost_per_1k_input=0.01,
        cost_per_1k_output=0.03,
        supported=False,
        description="GPT-4 with vision capabilities",
        use_cases=["image analysis", "OCR", "visual QA", "diagram understanding"],
        release_date="2023-09",
        documentation_url="https://platform.openai.com/docs/guides/vision",
        notes="Can process images and text together",
    )
)

register_model(
    ModelInfo(
        name="llava-34b",
        display_name="LLaVA 34B",
        provider=ModelProvider.OLLAMA,
        size=ModelSize.MEDIUM,
        context_window=4096,
        max_output_tokens=2048,
        capabilities=[
            ModelCapability.TEXT_GENERATION,
            ModelCapability.VISION,
        ],
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        supported=False,
        description="Open-source vision-language model",
        use_cases=["image captioning", "visual QA", "local deployment"],
        requires_api_key=False,
        release_date="2023-10",
        documentation_url="https://llava-vl.github.io/",
        notes="Available via Ollama",
    )
)

# =============================================================================
# REASONING & MATHEMATICS MODELS
# =============================================================================

register_model(
    ModelInfo(
        name="gpt-4-code-interpreter",
        display_name="GPT-4 Code Interpreter",
        provider=ModelProvider.OPENAI,
        size=ModelSize.FLAGSHIP,
        context_window=128000,
        max_output_tokens=4096,
        capabilities=[
            ModelCapability.CODE_GENERATION,
            ModelCapability.MATHEMATICS,
            ModelCapability.REASONING,
            ModelCapability.FUNCTION_CALLING,
        ],
        cost_per_1k_input=0.01,
        cost_per_1k_output=0.03,
        supported=False,
        description="GPT-4 with code execution capabilities",
        use_cases=["data analysis", "mathematics", "plotting", "file processing"],
        release_date="2023-07",
        documentation_url="https://platform.openai.com/docs/assistants/tools/code-interpreter",
        notes="Can execute Python code in sandbox",
    )
)

register_model(
    ModelInfo(
        name="claude-instant",
        display_name="Claude Instant",
        provider=ModelProvider.ANTHROPIC,
        size=ModelSize.SMALL,
        context_window=100000,
        max_output_tokens=4096,
        capabilities=[
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CODE_GENERATION,
            ModelCapability.LONG_CONTEXT,
            ModelCapability.STREAMING,
        ],
        cost_per_1k_input=0.0008,
        cost_per_1k_output=0.0024,
        supported=False,
        description="Fast, cost-effective Claude model",
        use_cases=["high-volume tasks", "simple queries", "preprocessing"],
        release_date="2023-08",
        documentation_url="https://docs.anthropic.com/claude/docs/models-overview",
        notes="Consider upgrading to Claude 3 Haiku",
    )
)

# =============================================================================
# EMBEDDING MODELS
# =============================================================================

register_model(
    ModelInfo(
        name="text-embedding-3-large",
        display_name="OpenAI Embeddings Large",
        provider=ModelProvider.OPENAI,
        size=ModelSize.SMALL,
        context_window=8191,
        max_output_tokens=0,  # Embeddings don't generate text
        capabilities=[
            ModelCapability.EMBEDDINGS,
        ],
        cost_per_1k_input=0.00013,
        cost_per_1k_output=0.0,
        supported=False,
        description="High-performance embedding model",
        use_cases=["semantic search", "RAG", "clustering", "similarity"],
        release_date="2024-01",
        documentation_url="https://platform.openai.com/docs/guides/embeddings",
        notes="3072-dimensional embeddings",
    )
)

register_model(
    ModelInfo(
        name="text-embedding-3-small",
        display_name="OpenAI Embeddings Small",
        provider=ModelProvider.OPENAI,
        size=ModelSize.TINY,
        context_window=8191,
        max_output_tokens=0,
        capabilities=[
            ModelCapability.EMBEDDINGS,
        ],
        cost_per_1k_input=0.00002,
        cost_per_1k_output=0.0,
        supported=False,
        description="Cost-effective embedding model",
        use_cases=["semantic search", "RAG", "clustering"],
        release_date="2024-01",
        documentation_url="https://platform.openai.com/docs/guides/embeddings",
        notes="1536-dimensional embeddings",
    )
)

register_model(
    ModelInfo(
        name="embed-english-v3",
        display_name="Cohere Embeddings v3",
        provider=ModelProvider.LOCAL,
        size=ModelSize.SMALL,
        context_window=512,
        max_output_tokens=0,
        capabilities=[
            ModelCapability.EMBEDDINGS,
        ],
        cost_per_1k_input=0.0001,
        cost_per_1k_output=0.0,
        supported=False,
        description="Cohere's embedding model",
        use_cases=["semantic search", "classification"],
        release_date="2023-11",
        documentation_url="https://docs.cohere.com/docs/embeddings",
        notes="1024-dimensional embeddings",
    )
)

# =============================================================================
# DOMAIN-SPECIFIC MODELS
# =============================================================================

register_model(
    ModelInfo(
        name="meditron-70b",
        display_name="Meditron 70B",
        provider=ModelProvider.OLLAMA,
        size=ModelSize.XLARGE,
        context_window=4096,
        max_output_tokens=2048,
        capabilities=[
            ModelCapability.TEXT_GENERATION,
            ModelCapability.REASONING,
        ],
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        supported=False,
        description="Medical domain-specific LLM",
        use_cases=["medical QA", "clinical notes", "medical research"],
        requires_api_key=False,
        release_date="2023-11",
        documentation_url="https://github.com/epfLLM/meditron",
        notes="Trained on medical literature and clinical data",
    )
)

register_model(
    ModelInfo(
        name="biomistral-7b",
        display_name="BioMistral 7B",
        provider=ModelProvider.OLLAMA,
        size=ModelSize.SMALL,
        context_window=8192,
        max_output_tokens=2048,
        capabilities=[
            ModelCapability.TEXT_GENERATION,
            ModelCapability.REASONING,
        ],
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        supported=False,
        description="Biomedical domain-specific model",
        use_cases=["biomedical research", "literature review", "medical QA"],
        requires_api_key=False,
        release_date="2024-02",
        documentation_url="https://huggingface.co/BioMistral/BioMistral-7B",
        notes="Fine-tuned on PubMed abstracts",
    )
)

register_model(
    ModelInfo(
        name="starcoder2-15b",
        display_name="StarCoder2 15B",
        provider=ModelProvider.OLLAMA,
        size=ModelSize.SMALL,
        context_window=16384,
        max_output_tokens=4096,
        capabilities=[
            ModelCapability.CODE_GENERATION,
            ModelCapability.LONG_CONTEXT,
        ],
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        supported=False,
        description="Code generation specialist",
        use_cases=["code completion", "code generation", "refactoring"],
        requires_api_key=False,
        release_date="2024-02",
        documentation_url="https://huggingface.co/bigcode/starcoder2-15b",
        notes="Trained on The Stack v2 dataset",
    )
)

register_model(
    ModelInfo(
        name="sqlcoder-70b",
        display_name="SQLCoder 70B",
        provider=ModelProvider.OLLAMA,
        size=ModelSize.XLARGE,
        context_window=8192,
        max_output_tokens=2048,
        capabilities=[
            ModelCapability.CODE_GENERATION,
        ],
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        supported=False,
        description="SQL generation specialist",
        use_cases=["SQL generation", "database queries", "data analysis"],
        requires_api_key=False,
        release_date="2023-10",
        documentation_url="https://github.com/defog-ai/sqlcoder",
        notes="Outperforms GPT-4 on SQL tasks",
    )
)

# =============================================================================
# MULTILINGUAL MODELS
# =============================================================================

register_model(
    ModelInfo(
        name="aya-101",
        display_name="Aya 101",
        provider=ModelProvider.OLLAMA,
        size=ModelSize.LARGE,
        context_window=8192,
        max_output_tokens=2048,
        capabilities=[
            ModelCapability.TEXT_GENERATION,
            ModelCapability.MULTILINGUAL,
        ],
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        supported=False,
        description="Massively multilingual model (101 languages)",
        use_cases=["translation", "multilingual QA", "global applications"],
        requires_api_key=False,
        release_date="2024-02",
        documentation_url="https://huggingface.co/CohereForAI/aya-101",
        notes="Covers languages underrepresented in AI",
    )
)

# =============================================================================
# ADDITIONAL SPECIALIZED MODELS (12 more to reach 50+)
# =============================================================================

register_model(
    ModelInfo(
        name="yi-34b",
        display_name="Yi 34B",
        provider=ModelProvider.OLLAMA,
        size=ModelSize.MEDIUM,
        context_window=200000,
        max_output_tokens=4096,
        capabilities=[
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CODE_GENERATION,
            ModelCapability.LONG_CONTEXT,
            ModelCapability.MULTILINGUAL,
        ],
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        supported=False,
        description="01.AI's high-performance model with 200K context",
        use_cases=["long documents", "multilingual", "general purpose"],
        requires_api_key=False,
        release_date="2024-01",
        documentation_url="https://huggingface.co/01-ai/Yi-34B",
        notes="Exceptional long-context performance",
    )
)

register_model(
    ModelInfo(
        name="qwen-72b",
        display_name="Qwen 72B",
        provider=ModelProvider.OLLAMA,
        size=ModelSize.XLARGE,
        context_window=32768,
        max_output_tokens=2048,
        capabilities=[
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CODE_GENERATION,
            ModelCapability.REASONING,
            ModelCapability.MULTILINGUAL,
        ],
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        supported=False,
        description="Alibaba's large multilingual model",
        use_cases=["Chinese language", "multilingual", "general purpose"],
        requires_api_key=False,
        release_date="2023-12",
        documentation_url="https://huggingface.co/Qwen/Qwen-72B",
        notes="Strong performance on Chinese benchmarks",
    )
)

register_model(
    ModelInfo(
        name="nous-hermes-2-mixtral",
        display_name="Nous Hermes 2 Mixtral",
        provider=ModelProvider.OLLAMA,
        size=ModelSize.LARGE,
        context_window=32768,
        max_output_tokens=4096,
        capabilities=[
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CODE_GENERATION,
            ModelCapability.REASONING,
            ModelCapability.FUNCTION_CALLING,
        ],
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        supported=False,
        description="Fine-tuned Mixtral for function calling",
        use_cases=["agents", "function calling", "tool use"],
        requires_api_key=False,
        release_date="2024-01",
        documentation_url="https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
        notes="Optimized for agent workflows",
    )
)

register_model(
    ModelInfo(
        name="dolphin-mixtral",
        display_name="Dolphin Mixtral",
        provider=ModelProvider.OLLAMA,
        size=ModelSize.LARGE,
        context_window=32768,
        max_output_tokens=4096,
        capabilities=[
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CODE_GENERATION,
            ModelCapability.REASONING,
        ],
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        supported=False,
        description="Uncensored Mixtral fine-tune",
        use_cases=["creative writing", "research", "unrestricted output"],
        requires_api_key=False,
        release_date="2023-12",
        documentation_url="https://huggingface.co/cognitivecomputations/dolphin-2.6-mixtral-8x7b",
        notes="No alignment, use responsibly",
    )
)

register_model(
    ModelInfo(
        name="openhermes-2.5-mistral",
        display_name="OpenHermes 2.5 Mistral",
        provider=ModelProvider.OLLAMA,
        size=ModelSize.SMALL,
        context_window=8192,
        max_output_tokens=2048,
        capabilities=[
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CODE_GENERATION,
            ModelCapability.REASONING,
        ],
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        supported=False,
        description="High-quality instruction-following model",
        use_cases=["general purpose", "instruction following", "chat"],
        requires_api_key=False,
        release_date="2023-11",
        documentation_url="https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B",
        notes="Trained on high-quality dataset",
    )
)

register_model(
    ModelInfo(
        name="neural-chat-7b",
        display_name="Neural Chat 7B",
        provider=ModelProvider.OLLAMA,
        size=ModelSize.SMALL,
        context_window=8192,
        max_output_tokens=2048,
        capabilities=[
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CODE_GENERATION,
        ],
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        supported=False,
        description="Intel's fine-tuned chat model",
        use_cases=["conversation", "customer service", "assistants"],
        requires_api_key=False,
        release_date="2023-11",
        documentation_url="https://huggingface.co/Intel/neural-chat-7b-v3-1",
        notes="Optimized for Intel hardware",
    )
)

register_model(
    ModelInfo(
        name="stable-code-3b",
        display_name="Stable Code 3B",
        provider=ModelProvider.OLLAMA,
        size=ModelSize.TINY,
        context_window=16384,
        max_output_tokens=4096,
        capabilities=[
            ModelCapability.CODE_GENERATION,
            ModelCapability.LONG_CONTEXT,
        ],
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        supported=False,
        description="Stability AI's code model",
        use_cases=["code completion", "IDE integration", "edge deployment"],
        requires_api_key=False,
        release_date="2024-01",
        documentation_url="https://huggingface.co/stabilityai/stable-code-3b",
        notes="Efficient for local development",
    )
)

register_model(
    ModelInfo(
        name="phind-codellama-34b",
        display_name="Phind CodeLlama 34B",
        provider=ModelProvider.OLLAMA,
        size=ModelSize.MEDIUM,
        context_window=16384,
        max_output_tokens=4096,
        capabilities=[
            ModelCapability.CODE_GENERATION,
            ModelCapability.REASONING,
        ],
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        supported=False,
        description="Fine-tuned for technical questions",
        use_cases=["technical QA", "debugging", "code explanation"],
        requires_api_key=False,
        release_date="2023-08",
        documentation_url="https://huggingface.co/Phind/Phind-CodeLlama-34B-v2",
        notes="Trained on StackOverflow and technical content",
    )
)

register_model(
    ModelInfo(
        name="openchat-3.5",
        display_name="OpenChat 3.5",
        provider=ModelProvider.OLLAMA,
        size=ModelSize.SMALL,
        context_window=8192,
        max_output_tokens=2048,
        capabilities=[
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CODE_GENERATION,
            ModelCapability.REASONING,
        ],
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        supported=False,
        description="High-performing 7B chat model",
        use_cases=["chat", "general purpose", "efficient deployment"],
        requires_api_key=False,
        release_date="2023-11",
        documentation_url="https://huggingface.co/openchat/openchat-3.5-0106",
        notes="Matches GPT-3.5 on many benchmarks",
    )
)

register_model(
    ModelInfo(
        name="zephyr-7b-beta",
        display_name="Zephyr 7B Beta",
        provider=ModelProvider.OLLAMA,
        size=ModelSize.SMALL,
        context_window=8192,
        max_output_tokens=2048,
        capabilities=[
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CODE_GENERATION,
        ],
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        supported=False,
        description="Aligned Mistral fine-tune",
        use_cases=["chat", "instruction following", "assistants"],
        requires_api_key=False,
        release_date="2023-10",
        documentation_url="https://huggingface.co/HuggingFaceH4/zephyr-7b-beta",
        notes="DPO-aligned for helpfulness",
    )
)

register_model(
    ModelInfo(
        name="solar-10.7b",
        display_name="SOLAR 10.7B",
        provider=ModelProvider.OLLAMA,
        size=ModelSize.SMALL,
        context_window=4096,
        max_output_tokens=2048,
        capabilities=[
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CODE_GENERATION,
            ModelCapability.REASONING,
        ],
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        supported=False,
        description="Upstage's depth-upscaled model",
        use_cases=["general purpose", "reasoning", "efficient performance"],
        requires_api_key=False,
        release_date="2023-12",
        documentation_url="https://huggingface.co/upstage/SOLAR-10.7B-v1.0",
        notes="Novel upscaling technique",
    )
)

register_model(
    ModelInfo(
        name="orca-2-13b",
        display_name="Orca 2 13B",
        provider=ModelProvider.OLLAMA,
        size=ModelSize.SMALL,
        context_window=4096,
        max_output_tokens=2048,
        capabilities=[
            ModelCapability.TEXT_GENERATION,
            ModelCapability.REASONING,
            ModelCapability.CODE_GENERATION,
        ],
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        supported=False,
        description="Microsoft's reasoning-focused model",
        use_cases=["reasoning", "problem solving", "step-by-step thinking"],
        requires_api_key=False,
        release_date="2023-11",
        documentation_url="https://huggingface.co/microsoft/Orca-2-13b",
        notes="Trained with progressive learning",
    )
)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_model(name: str) -> ModelInfo:
    """
    Get model information by name.

    Args:
        name: Model identifier

    Returns:
        Model information

    Raises:
        KeyError: If model not found
    """
    return MODEL_REGISTRY[name]


def list_models(
    provider: Optional[ModelProvider] = None,
    size: Optional[ModelSize] = None,
    capability: Optional[ModelCapability] = None,
    supported_only: bool = False,
) -> List[ModelInfo]:
    """
    List models matching criteria.

    Args:
        provider: Filter by provider
        size: Filter by size
        capability: Filter by capability
        supported_only: Only show fully supported models

    Returns:
        List of matching models
    """
    models = list(MODEL_REGISTRY.values())

    if provider:
        models = [m for m in models if m.provider == provider]

    if size:
        models = [m for m in models if m.size == size]

    if capability:
        models = [m for m in models if capability in m.capabilities]

    if supported_only:
        models = [m for m in models if m.supported]

    return models


def get_models_by_use_case(use_case: str) -> List[ModelInfo]:
    """
    Get models suitable for a specific use case.

    Args:
        use_case: Use case description

    Returns:
        List of suitable models
    """
    use_case_lower = use_case.lower()
    return [m for m in MODEL_REGISTRY.values() if any(use_case_lower in uc.lower() for uc in m.use_cases)]


def get_cheapest_models(capability: Optional[ModelCapability] = None, top_n: int = 5) -> List[ModelInfo]:
    """
    Get cheapest models by total cost.

    Args:
        capability: Optional capability requirement
        top_n: Number of models to return

    Returns:
        List of cheapest models
    """
    models = list(MODEL_REGISTRY.values())

    if capability:
        models = [m for m in models if capability in m.capabilities]

    # Sort by average cost (input + output)
    models.sort(key=lambda m: (m.cost_per_1k_input + m.cost_per_1k_output) / 2)

    return models[:top_n]


def get_model_count() -> int:
    """Get total number of registered models."""
    return len(MODEL_REGISTRY)


def get_supported_count() -> int:
    """Get number of fully supported models."""
    return len([m for m in MODEL_REGISTRY.values() if m.supported])
