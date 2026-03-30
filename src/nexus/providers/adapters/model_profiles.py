"""
Rich model profiles for Nexus model registry.

Provides structured metadata about model families: what they're good at,
what they're not, which programming and natural languages they support,
and what tasks they specialize in. This data is used by the intelligent
selector to make better routing decisions.

Profiles are matched by model ID pattern (prefix matching). When a model
is discovered from OpenRouter/OpenAI/etc, its ID is matched against
known families to enrich the registration with curated metadata.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ModelProfile:
    """Rich metadata profile for a model or model family."""

    family: str  # e.g., "gpt-5", "claude-4", "qwen3"

    # What this model excels at
    strengths: List[str] = field(default_factory=list)

    # Known limitations
    weaknesses: List[str] = field(default_factory=list)

    # Tasks this model is particularly good for
    specializations: List[str] = field(default_factory=list)

    # Tasks where other models are better choices
    not_recommended_for: List[str] = field(default_factory=list)

    # Programming language support (what it can read/write well)
    programming_languages: List[str] = field(default_factory=list)

    # Natural language support (which human languages it handles well)
    natural_languages: List[str] = field(default_factory=list)

    # Weak natural languages (can attempt but quality is poor)
    weak_natural_languages: List[str] = field(default_factory=list)

    # Training data cutoff (if known)
    knowledge_cutoff: Optional[str] = None

    # Quality tier: "frontier", "strong", "good", "basic"
    quality_tier: str = "good"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": self.family,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "specializations": self.specializations,
            "not_recommended_for": self.not_recommended_for,
            "programming_languages": self.programming_languages,
            "natural_languages": self.natural_languages,
            "weak_natural_languages": self.weak_natural_languages,
            "knowledge_cutoff": self.knowledge_cutoff,
            "quality_tier": self.quality_tier,
        }


# ---------------------------------------------------------------------------
# Common language sets
# ---------------------------------------------------------------------------

_ALL_MAJOR_PROG_LANGS = [
    "python", "javascript", "typescript", "java", "c", "cpp", "csharp",
    "go", "rust", "ruby", "php", "swift", "kotlin", "scala", "r",
    "sql", "bash", "powershell", "html", "css",
]

_EXTENDED_PROG_LANGS = _ALL_MAJOR_PROG_LANGS + [
    "haskell", "elixir", "clojure", "lua", "perl", "dart", "julia",
    "matlab", "fortran", "cobol", "assembly", "vhdl", "verilog",
    "solidity", "zig", "nim", "ocaml", "f#", "groovy",
]

_WESTERN_LANGUAGES = [
    "english", "spanish", "french", "german", "italian", "portuguese",
    "dutch", "swedish", "norwegian", "danish", "finnish", "polish",
    "czech", "romanian", "hungarian", "greek",
]

_CJK_LANGUAGES = ["chinese", "japanese", "korean"]

_BROAD_LANGUAGES = _WESTERN_LANGUAGES + _CJK_LANGUAGES + [
    "arabic", "hindi", "turkish", "thai", "vietnamese", "indonesian",
    "russian", "ukrainian",
]


# ---------------------------------------------------------------------------
# Curated Model Family Profiles
# ---------------------------------------------------------------------------

_PROFILES: Dict[str, ModelProfile] = {}


def _register(pattern: str, profile: ModelProfile) -> None:
    _PROFILES[pattern] = profile


# --- OpenAI GPT-5 family ---
_register("openai/gpt-5", ModelProfile(
    family="gpt-5",
    strengths=[
        "complex multi-step reasoning", "instruction following",
        "code generation and debugging", "creative writing",
        "structured output (JSON/XML)", "tool/function calling",
        "long document analysis", "mathematical reasoning",
    ],
    weaknesses=[
        "high cost for simple tasks", "can be verbose",
        "occasional hallucination on niche topics",
    ],
    specializations=[
        "enterprise applications", "code review", "technical writing",
        "data analysis", "strategic planning", "legal document review",
    ],
    not_recommended_for=[
        "cost-sensitive high-volume tasks", "simple classification",
        "latency-critical real-time applications",
    ],
    programming_languages=_EXTENDED_PROG_LANGS,
    natural_languages=_BROAD_LANGUAGES,
    knowledge_cutoff="2025-10",
    quality_tier="frontier",
))

# --- OpenAI o3/o4 reasoning ---
_register("openai/o3", ModelProfile(
    family="o3",
    strengths=[
        "deep multi-step reasoning", "mathematical proofs",
        "scientific analysis", "complex code architecture",
    ],
    weaknesses=[
        "very high cost", "slow response time", "overkill for simple tasks",
    ],
    specializations=[
        "research", "theorem proving", "complex debugging",
        "system design", "strategic analysis",
    ],
    not_recommended_for=[
        "chat", "simple Q&A", "content generation", "real-time applications",
    ],
    programming_languages=_EXTENDED_PROG_LANGS,
    natural_languages=_BROAD_LANGUAGES,
    knowledge_cutoff="2025-10",
    quality_tier="frontier",
))

_register("openai/o4", ModelProfile(
    family="o4",
    strengths=[
        "fast reasoning", "code generation", "mathematical reasoning",
    ],
    weaknesses=["less thorough than o3 on complex problems"],
    specializations=["code debugging", "quick analysis", "math"],
    not_recommended_for=["deep research", "long-form writing"],
    programming_languages=_EXTENDED_PROG_LANGS,
    natural_languages=_BROAD_LANGUAGES,
    knowledge_cutoff="2025-10",
    quality_tier="frontier",
))

# --- Anthropic Claude 4.x family ---
_register("anthropic/claude-opus-4", ModelProfile(
    family="claude-opus-4",
    strengths=[
        "nuanced reasoning", "careful instruction following",
        "long document analysis (200K+ context)", "code generation",
        "safety-conscious responses", "structured outputs",
        "agentic tool use", "complex multi-turn conversations",
    ],
    weaknesses=[
        "high cost", "conservative on ambiguous requests",
    ],
    specializations=[
        "legal analysis", "compliance review", "code review",
        "technical documentation", "research synthesis",
        "autonomous agent workflows",
    ],
    not_recommended_for=[
        "cost-sensitive bulk processing", "real-time chat",
    ],
    programming_languages=_EXTENDED_PROG_LANGS,
    natural_languages=_BROAD_LANGUAGES,
    knowledge_cutoff="2025-05",
    quality_tier="frontier",
))

_register("anthropic/claude-sonnet-4", ModelProfile(
    family="claude-sonnet-4",
    strengths=[
        "balanced cost/quality", "strong code generation",
        "good instruction following", "reliable structured output",
        "tool use", "200K context",
    ],
    weaknesses=[
        "less thorough than Opus on complex reasoning",
    ],
    specializations=[
        "code generation", "code review", "API integration",
        "content writing", "data analysis",
    ],
    not_recommended_for=["deep research requiring maximum reasoning"],
    programming_languages=_EXTENDED_PROG_LANGS,
    natural_languages=_BROAD_LANGUAGES,
    knowledge_cutoff="2025-05",
    quality_tier="frontier",
))

_register("anthropic/claude-haiku-4", ModelProfile(
    family="claude-haiku-4",
    strengths=[
        "fast response", "low cost", "good for simple tasks",
        "classification", "extraction", "summarization",
    ],
    weaknesses=[
        "weaker complex reasoning", "shorter outputs",
        "less reliable on ambiguous instructions",
    ],
    specializations=[
        "classification", "entity extraction", "summarization",
        "simple Q&A", "content moderation",
    ],
    not_recommended_for=[
        "complex code generation", "deep analysis", "creative writing",
    ],
    programming_languages=_ALL_MAJOR_PROG_LANGS,
    natural_languages=_BROAD_LANGUAGES,
    knowledge_cutoff="2025-05",
    quality_tier="good",
))

# --- Google Gemini family ---
_register("google/gemini-3", ModelProfile(
    family="gemini-3",
    strengths=[
        "multimodal (text, image, video, audio)", "very long context (1M+)",
        "strong reasoning", "fast response", "good code generation",
        "native tool/function calling",
    ],
    weaknesses=[
        "occasional instruction drift on long conversations",
    ],
    specializations=[
        "multimodal analysis", "document understanding", "video analysis",
        "code generation", "data analysis",
    ],
    not_recommended_for=["tasks requiring strict output format control"],
    programming_languages=_EXTENDED_PROG_LANGS,
    natural_languages=_BROAD_LANGUAGES,
    knowledge_cutoff="2025-06",
    quality_tier="frontier",
))

_register("google/gemini-2.5", ModelProfile(
    family="gemini-2.5",
    strengths=[
        "multimodal", "1M context", "fast", "good reasoning",
        "strong code generation", "cost-effective",
    ],
    weaknesses=["slightly less capable than Gemini 3 on complex tasks"],
    specializations=[
        "multimodal analysis", "code generation", "summarization",
    ],
    not_recommended_for=[],
    programming_languages=_EXTENDED_PROG_LANGS,
    natural_languages=_BROAD_LANGUAGES,
    knowledge_cutoff="2025-03",
    quality_tier="strong",
))

# --- Qwen family ---
_register("qwen/qwen3", ModelProfile(
    family="qwen3",
    strengths=[
        "strong multilingual (especially CJK)", "good reasoning",
        "cost-effective", "strong code generation",
        "good mathematical reasoning", "MoE efficiency",
    ],
    weaknesses=[
        "less strong on nuanced English creative writing",
        "smaller community/ecosystem than GPT/Claude",
    ],
    specializations=[
        "Chinese/Japanese/Korean content", "code generation",
        "mathematical reasoning", "translation", "multilingual tasks",
    ],
    not_recommended_for=[
        "English-only creative writing where nuance matters",
    ],
    programming_languages=_EXTENDED_PROG_LANGS,
    natural_languages=_BROAD_LANGUAGES + ["malay", "bengali", "urdu", "persian"],
    knowledge_cutoff="2025-04",
    quality_tier="strong",
))

# --- DeepSeek family ---
_register("deepseek/deepseek-v3", ModelProfile(
    family="deepseek-v3",
    strengths=[
        "excellent code generation", "strong reasoning",
        "very cost-effective", "good at math", "strong Chinese support",
    ],
    weaknesses=[
        "less reliable on creative/open-ended tasks",
        "weaker on some Western languages",
    ],
    specializations=[
        "code generation", "code review", "debugging",
        "mathematical reasoning", "technical analysis",
    ],
    not_recommended_for=[
        "creative writing", "marketing copy", "nuanced cultural content",
    ],
    programming_languages=_EXTENDED_PROG_LANGS,
    natural_languages=["english", "chinese"] + _WESTERN_LANGUAGES[:6],
    weak_natural_languages=["arabic", "hindi", "thai"],
    knowledge_cutoff="2025-05",
    quality_tier="strong",
))

_register("deepseek/deepseek-r1", ModelProfile(
    family="deepseek-r1",
    strengths=[
        "strong chain-of-thought reasoning", "mathematical proofs",
        "code debugging", "logical analysis",
    ],
    weaknesses=["slow", "verbose reasoning traces", "high cost"],
    specializations=["math", "logic", "complex debugging", "research"],
    not_recommended_for=["chat", "simple tasks", "content generation"],
    programming_languages=_EXTENDED_PROG_LANGS,
    natural_languages=["english", "chinese"],
    knowledge_cutoff="2025-01",
    quality_tier="strong",
))

# --- Mistral family ---
_register("mistral/mistral-large", ModelProfile(
    family="mistral-large",
    strengths=[
        "strong European language support", "good reasoning",
        "function calling", "balanced cost/performance",
    ],
    weaknesses=["less capable than GPT-5/Claude on complex tasks"],
    specializations=[
        "European multilingual tasks", "business applications",
        "structured data extraction", "function calling",
    ],
    not_recommended_for=["CJK languages", "deep research"],
    programming_languages=_ALL_MAJOR_PROG_LANGS,
    natural_languages=_WESTERN_LANGUAGES + ["arabic", "russian"],
    weak_natural_languages=_CJK_LANGUAGES,
    knowledge_cutoff="2025-06",
    quality_tier="strong",
))

_register("mistral/codestral", ModelProfile(
    family="codestral",
    strengths=[
        "excellent code generation", "code completion",
        "multi-language code support", "fast inference",
    ],
    weaknesses=["not designed for general conversation"],
    specializations=[
        "code generation", "code completion", "refactoring",
        "code review", "test generation",
    ],
    not_recommended_for=[
        "general chat", "creative writing", "research",
    ],
    programming_languages=_EXTENDED_PROG_LANGS,
    natural_languages=["english", "french"],
    knowledge_cutoff="2025-06",
    quality_tier="strong",
))

# --- Google Gemma 3 family ---
_register("google/gemma-3", ModelProfile(
    family="gemma-3",
    strengths=[
        "open source", "strong instruction following", "efficient inference",
        "good multilingual support", "local deployment possible",
    ],
    weaknesses=[
        "less capable than frontier proprietary models",
        "limited tool/function calling",
    ],
    specializations=[
        "local deployment", "content analysis", "classification",
        "summarization", "evaluation and scoring",
    ],
    not_recommended_for=[
        "complex multi-step reasoning", "large code generation",
    ],
    programming_languages=_ALL_MAJOR_PROG_LANGS,
    natural_languages=_WESTERN_LANGUAGES + ["chinese", "japanese", "korean"],
    knowledge_cutoff="2025-02",
    quality_tier="good",
))

# --- Mistral Small family ---
_register("mistralai/mistral-small", ModelProfile(
    family="mistral-small",
    strengths=[
        "efficient reasoning", "good structured output",
        "fast inference", "strong European language support",
        "cost-effective", "good instruction following",
    ],
    weaknesses=[
        "less capable than Mistral Large on complex tasks",
        "limited creative writing",
    ],
    specializations=[
        "structured analysis", "classification", "evaluation",
        "adversarial review", "recommendation generation",
    ],
    not_recommended_for=[
        "frontier-level reasoning", "creative writing", "CJK languages",
    ],
    programming_languages=_ALL_MAJOR_PROG_LANGS,
    natural_languages=_WESTERN_LANGUAGES + ["arabic", "russian"],
    weak_natural_languages=_CJK_LANGUAGES,
    knowledge_cutoff="2025-06",
    quality_tier="good",
))

# --- Meta Llama family ---
_register("meta-llama/llama", ModelProfile(
    family="llama",
    strengths=[
        "open source", "good general capability", "fast inference",
        "local deployment possible", "fine-tunable",
    ],
    weaknesses=[
        "less capable than proprietary frontier models",
        "weaker on complex reasoning",
    ],
    specializations=[
        "local deployment", "fine-tuning base", "cost-effective inference",
    ],
    not_recommended_for=[
        "tasks requiring frontier-level reasoning",
    ],
    programming_languages=_ALL_MAJOR_PROG_LANGS,
    natural_languages=_WESTERN_LANGUAGES + ["chinese"],
    weak_natural_languages=["japanese", "korean", "arabic"],
    knowledge_cutoff="2025-03",
    quality_tier="good",
))

# --- xAI Grok family ---
_register("x-ai/grok-4", ModelProfile(
    family="grok-4",
    strengths=[
        "strong reasoning", "real-time knowledge", "code generation",
        "mathematical reasoning", "long context",
    ],
    weaknesses=["less established ecosystem"],
    specializations=[
        "current events analysis", "code generation",
        "mathematical reasoning", "scientific analysis",
    ],
    not_recommended_for=[],
    programming_languages=_EXTENDED_PROG_LANGS,
    natural_languages=_BROAD_LANGUAGES,
    knowledge_cutoff="2025-06",
    quality_tier="frontier",
))

# --- Cohere family ---
_register("cohere/command", ModelProfile(
    family="cohere-command",
    strengths=[
        "enterprise RAG", "strong embeddings",
        "good structured generation", "multilingual",
    ],
    weaknesses=["less capable on creative tasks", "smaller model ecosystem"],
    specializations=[
        "RAG/retrieval", "enterprise search", "document processing",
        "classification", "embeddings",
    ],
    not_recommended_for=["creative writing", "complex code generation"],
    programming_languages=_ALL_MAJOR_PROG_LANGS[:10],
    natural_languages=_BROAD_LANGUAGES,
    knowledge_cutoff="2025-01",
    quality_tier="good",
))


# ---------------------------------------------------------------------------
# Lookup API
# ---------------------------------------------------------------------------

def get_model_profile(model_id: str) -> Optional[ModelProfile]:
    """Find the best matching profile for a model ID.

    Matches by longest prefix: 'openai/gpt-5.2' matches 'openai/gpt-5'.
    """
    model_lower = model_id.lower()
    best_match = None
    best_len = 0

    for pattern, profile in _PROFILES.items():
        pattern_lower = pattern.lower()
        if model_lower.startswith(pattern_lower) and len(pattern_lower) > best_len:
            best_match = profile
            best_len = len(pattern_lower)

    return best_match


def get_all_profiles() -> Dict[str, ModelProfile]:
    """Return all registered profiles."""
    return dict(_PROFILES)


def get_profile_for_task(
    task_type: str,
    required_languages: Optional[List[str]] = None,
    required_prog_languages: Optional[List[str]] = None,
) -> List[ModelProfile]:
    """Find profiles that specialize in a given task type and language requirements."""
    matches = []
    for profile in _PROFILES.values():
        # Check specialization match
        task_match = any(
            task_type.lower() in s.lower() for s in profile.specializations
        )
        if not task_match:
            # Also check strengths
            task_match = any(
                task_type.lower() in s.lower() for s in profile.strengths
            )

        if not task_match:
            continue

        # Check language requirements
        if required_languages:
            supported = {l.lower() for l in profile.natural_languages}
            if not all(l.lower() in supported for l in required_languages):
                continue

        if required_prog_languages:
            supported = {l.lower() for l in profile.programming_languages}
            if not all(l.lower() in supported for l in required_prog_languages):
                continue

        matches.append(profile)

    return matches
