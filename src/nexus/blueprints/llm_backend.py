"""
LLM Backend - Configurable provider for chapter generation

Supports:
- Anthropic (Claude) - fast, API-based
- Ollama - local, free, slower
"""

import os
import logging
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import asyncio


logger = logging.getLogger(__name__)
_ENV_BOOTSTRAPPED = False


def _ensure_environment_loaded() -> None:
    """Load project-level .env once so query backends do not depend on unrelated init."""
    global _ENV_BOOTSTRAPPED
    if _ENV_BOOTSTRAPPED:
        return

    _ENV_BOOTSTRAPPED = True

    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    env_paths = (
        Path(__file__).resolve().parents[3] / ".env",
        Path(__file__).resolve().parents[2] / ".env",
        Path(__file__).resolve().parent / ".env",
    )

    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path, override=False)
            logger.info("Loaded environment from %s", env_path)
            break


@dataclass
class LLMResponse:
    """Response from LLM generation."""
    content: str
    tokens_used: int
    model: str
    provider: str
    duration_seconds: float = 0
    input_tokens: int = 0
    output_tokens: int = 0


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 8000,
    ) -> LLMResponse:
        """Generate text from prompt."""
        pass
    
    @abstractmethod
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost in USD."""
        pass
    
    @property
    def name(self) -> str:
        """Human-readable backend name."""
        return f"{self.__class__.__name__}"


class AnthropicBackend(LLMBackend):
    """Anthropic Claude backend."""
    
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None
    ):
        _ensure_environment_loaded()
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client = None
    
    @property
    def name(self) -> str:
        return f"Anthropic ({self.model})"
    
    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client
    
    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 8000,
    ) -> LLMResponse:
        import time
        start = time.time()
        
        client = self._get_client()
        
        # Run sync client in thread pool
        loop = asyncio.get_event_loop()
        message = await loop.run_in_executor(
            None,
            lambda: client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}]
            )
        )
        
        duration = time.time() - start
        
        return LLMResponse(
            content=message.content[0].text,
            tokens_used=message.usage.input_tokens + message.usage.output_tokens,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            model=message.model,
            provider="anthropic",
            duration_seconds=duration
        )
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        # Claude Sonnet pricing (as of late 2024)
        # $3 per 1M input, $15 per 1M output
        input_cost = (input_tokens / 1_000_000) * 3.0
        output_cost = (output_tokens / 1_000_000) * 15.0
        return input_cost + output_cost

    async def check_available(self) -> bool:
        """Check whether the backend is configured for use."""
        return bool(self.api_key)


class OllamaBackend(LLMBackend):
    """Ollama local backend with Qwen3 thinking mode support."""
    
    def __init__(
        self,
        model: str = "qwen3:30b-a3b",
        host: str = "http://localhost:11434",
        thinking_mode: bool = True,  # Enable Qwen3 deep thinking by default
        strip_thinking: bool = True,  # Remove <think> tags from output
    ):
        _ensure_environment_loaded()
        self.model = model
        self.host = host
        self.thinking_mode = thinking_mode
        self.strip_thinking = strip_thinking
    
    @property
    def name(self) -> str:
        mode = " [thinking]" if self.thinking_mode and "qwen3" in self.model.lower() else ""
        return f"Ollama ({self.model}{mode})"
    
    def _is_qwen3(self) -> bool:
        """Check if this is a Qwen3 model."""
        return "qwen3" in self.model.lower()
    
    def _extract_content(self, raw_content: str) -> tuple[str, str]:
        """Extract thinking and final content from Qwen3 response.
        
        Returns:
            tuple: (thinking_content, final_content)
        """
        import re
        
        thinking = ""
        content = raw_content
        
        # Extract <think>...</think> blocks
        think_pattern = r'<think>(.*?)</think>'
        think_matches = re.findall(think_pattern, raw_content, re.DOTALL)
        
        if think_matches:
            thinking = "\n".join(think_matches)
            if self.strip_thinking:
                content = re.sub(think_pattern, '', raw_content, flags=re.DOTALL).strip()
        
        return thinking, content
    
    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 8000,
    ) -> LLMResponse:
        import time
        import aiohttp
        
        start = time.time()
        
        # Use chat API with system message for better results
        url = f"{self.host}/api/chat"
        messages = []
        
        # For Qwen3: Add thinking mode instruction to system prompt
        effective_system = system_prompt
        if self._is_qwen3() and self.thinking_mode:
            thinking_instruction = "You are a thoughtful assistant. Use /think to reason deeply before responding."
            if system_prompt:
                effective_system = f"{thinking_instruction}\n\n{system_prompt}"
            else:
                effective_system = thinking_instruction
        
        if effective_system:
            messages.append({"role": "system", "content": effective_system})
        
        # For Qwen3: Prepend /think to enable thinking mode
        effective_prompt = prompt
        if self._is_qwen3() and self.thinking_mode:
            effective_prompt = f"/think\n{prompt}"
        
        messages.append({"role": "user", "content": effective_prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": "30m",
            "options": {
                "num_predict": max_tokens,
                "num_ctx": 32768,  # Qwen3 supports up to 32K context
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=3600)) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    raise RuntimeError(f"Ollama error: {error}")
                
                data = await resp.json()
        
        duration = time.time() - start
        
        # Extract response from chat format
        raw_content = data.get("message", {}).get("content", "")
        
        # Process Qwen3 thinking output
        thinking, content = self._extract_content(raw_content)
        
        # Ollama returns eval_count for output tokens, prompt_eval_count for input
        input_tokens = data.get("prompt_eval_count", 0)
        output_tokens = data.get("eval_count", 0)
        
        return LLMResponse(
            content=content,
            tokens_used=input_tokens + output_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model,
            provider="ollama",
            duration_seconds=duration
        )
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        # Local = free (just electricity)
        return 0.0
    
    async def check_available(self) -> bool:
        """Check if Ollama is running and model is available."""
        import aiohttp
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.host}/api/tags", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        models = [m["name"] for m in data.get("models", [])]
                        # Check if our model (or base name) is available
                        model_base = self.model.split(":")[0]
                        return any(model_base in m for m in models)
            return False
        except Exception:
            return False
    
    async def list_models(self) -> list[str]:
        """List available models."""
        import aiohttp
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.host}/api/tags", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return [m["name"] for m in data.get("models", [])]
            return []
        except Exception:
            return []
    
    async def pull_model(self, progress_callback=None) -> bool:
        """Pull model if not available."""
        import aiohttp
        url = f"{self.host}/api/pull"
        payload = {"name": self.model, "stream": True}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=7200)) as resp:
                    if resp.status != 200:
                        return False
                    
                    # Stream progress
                    async for line in resp.content:
                        if line and progress_callback:
                            try:
                                import json
                                data = json.loads(line)
                                status = data.get("status", "")
                                if "pulling" in status or "downloading" in status:
                                    completed = data.get("completed", 0)
                                    total = data.get("total", 0)
                                    if total > 0:
                                        pct = (completed / total) * 100
                                        progress_callback(f"{status}: {pct:.1f}%")
                                else:
                                    progress_callback(status)
                            except Exception:
                                pass
                    
                    return True
        except Exception as e:
            print(f"Failed to pull model: {e}")
            return False


def get_backend(
    provider: str = "anthropic",
    model: Optional[str] = None,
    **kwargs
) -> LLMBackend:
    """Factory function to get LLM backend.
    
    Args:
        provider: "anthropic", "openai", or "ollama"
        model: Model name (provider-specific)
        **kwargs: Additional provider-specific args
        
    Returns:
        LLMBackend instance
    """
    if provider == "anthropic":
        return AnthropicBackend(
            model=model or "claude-sonnet-4-20250514",
            **kwargs
        )
    elif provider == "openai":
        return OpenAIBackend(
            model=model or "gpt-4o",
            **kwargs
        )
    elif provider == "ollama":
        return OllamaBackend(
            model=model or "qwen3:30b-a3b",
            thinking_mode=kwargs.pop("thinking_mode", True),
            strip_thinking=kwargs.pop("strip_thinking", True),
            **kwargs
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


class OpenAIBackend(LLMBackend):
    """OpenAI GPT backend - ideal for creative content like blueprint generation."""
    
    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None
    ):
        _ensure_environment_loaded()
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client = None
    
    @property
    def name(self) -> str:
        return f"OpenAI ({self.model})"
    
    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        return self._client
    
    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 8000,
    ) -> LLMResponse:
        import time
        start = time.time()
        
        client = self._get_client()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Run sync client in thread pool
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=messages
            )
        )
        
        duration = time.time() - start
        
        return LLMResponse(
            content=response.choices[0].message.content,
            tokens_used=response.usage.prompt_tokens + response.usage.completion_tokens,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            model=response.model,
            provider="openai",
            duration_seconds=duration
        )
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        # GPT-4o pricing (as of late 2024)
        # $2.50 per 1M input, $10 per 1M output
        input_cost = (input_tokens / 1_000_000) * 2.5
        output_cost = (output_tokens / 1_000_000) * 10.0
        return input_cost + output_cost

    async def check_available(self) -> bool:
        """Check whether the backend is configured for use."""
        return bool(self.api_key)


class OpenRouterBackend(LLMBackend):
    """OpenRouter backend — free and paid models via unified API."""

    def __init__(self, model: str = "qwen/qwen3-coder:free", api_key: Optional[str] = None):
        _ensure_environment_loaded()
        self.model = model
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self._client = None

    @property
    def name(self) -> str:
        return f"OpenRouter ({self.model})"

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key,
                base_url="https://openrouter.ai/api/v1",
            )
        return self._client

    async def generate(self, prompt: str, system_prompt: str = "", max_tokens: int = 8000) -> LLMResponse:
        import time
        start = time.time()
        client = self._get_client()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(model=self.model, max_tokens=max_tokens, messages=messages),
        )
        duration = time.time() - start
        usage = response.usage or type("U", (), {"prompt_tokens": 0, "completion_tokens": 0})()
        return LLMResponse(
            content=response.choices[0].message.content,
            tokens_used=usage.prompt_tokens + usage.completion_tokens,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            model=response.model,
            provider="openrouter",
            duration_seconds=duration,
        )

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        if ":free" in self.model:
            return 0.0
        return (input_tokens + output_tokens) / 1_000_000 * 1.0

    async def check_available(self) -> bool:
        return bool(self.api_key)




class GoogleBackend(LLMBackend):
    """Google Gemini backend."""

    def __init__(self, model: str = "gemini-2.5-flash", api_key: Optional[str] = None):
        _ensure_environment_loaded()
        self.model = model
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self._client = None

    @property
    def name(self) -> str:
        return f"Google ({self.model})"

    def _get_client(self):
        if self._client is None:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._client = genai.GenerativeModel(self.model)
        return self._client

    async def generate(self, prompt: str, system_prompt: str = "", max_tokens: int = 8000) -> LLMResponse:
        import time
        start = time.time()
        model = self._get_client()
        if system_prompt:
            # Recreate with system instruction
            import google.generativeai as genai
            model = genai.GenerativeModel(self.model, system_instruction=system_prompt)
        config = {"max_output_tokens": max_tokens}
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: model.generate_content(prompt, generation_config=config),
        )
        duration = time.time() - start
        content = response.text if hasattr(response, "text") else ""
        usage = getattr(response, "usage_metadata", None)
        input_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
        output_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0
        return LLMResponse(
            content=content,
            tokens_used=input_tokens + output_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model,
            provider="google",
            duration_seconds=duration,
        )

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens / 1_000_000) * 0.15 + (output_tokens / 1_000_000) * 0.60

    async def check_available(self) -> bool:
        return bool(self.api_key)


class ClaudeCodeBackend(LLMBackend):
    """Anthropic backend via Claude Code CLI — uses Max subscription (no per-token cost).

    Shells out to `claude -p` which routes through the user's Max account.
    """

    def __init__(self, model: str = "sonnet", claude_path: Optional[str] = None):
        _ensure_environment_loaded()
        self.model = model  # "sonnet", "opus", "haiku"
        self.claude_path = claude_path or os.getenv(
            "CLAUDE_CLI_PATH",
            shutil.which("claude") or "/usr/local/bin/claude",
        )

    @property
    def name(self) -> str:
        return f"Claude Max ({self.model})"

    async def generate(self, prompt: str, system_prompt: str = "", max_tokens: int = 8000) -> LLMResponse:
        import time
        import subprocess
        import json as _json
        start = time.time()

        cmd = [
            self.claude_path, "-p",
            "--output-format", "json",
            "--model", self.model,
            "--no-session-persistence",
        ]
        if system_prompt:
            cmd.extend(["--append-system-prompt", system_prompt])

        loop = asyncio.get_event_loop()

        def _run():
            result = subprocess.run(
                cmd, input=prompt, capture_output=True, text=True, timeout=120,
            )
            return result.stdout, result.stderr, result.returncode

        stdout, stderr, returncode = await loop.run_in_executor(None, _run)
        duration = time.time() - start

        content = ""
        if returncode == 0 and stdout.strip():
            try:
                data = _json.loads(stdout)
                content = data.get("result", data.get("content", stdout.strip()))
            except _json.JSONDecodeError:
                content = stdout.strip()
        elif stderr:
            raise RuntimeError(f"Claude CLI error: {stderr[:300]}")

        return LLMResponse(
            content=content,
            tokens_used=0,  # CLI doesn't report token counts
            model=self.model,
            provider="claude-max",
            duration_seconds=duration,
        )

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return 0.0  # Max subscription — no per-token cost

    async def check_available(self) -> bool:
        return os.path.exists(self.claude_path)


class OpenAICompatibleBackend(LLMBackend):
    """Generic backend for any OpenAI-compatible API (DeepSeek, Mistral, Together, etc.)."""

    def __init__(
        self,
        provider_name: str,
        model: str,
        base_url: str,
        api_key: Optional[str] = None,
        env_var: str = "",
    ):
        _ensure_environment_loaded()
        self.provider_name = provider_name
        self.model = model
        self.base_url = base_url
        self.api_key = api_key or (os.getenv(env_var) if env_var else None)
        self._client = None

    @property
    def name(self) -> str:
        return f"{self.provider_name} ({self.model})"

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    async def generate(self, prompt: str, system_prompt: str = "", max_tokens: int = 8000) -> LLMResponse:
        import time
        start = time.time()
        client = self._get_client()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(model=self.model, max_tokens=max_tokens, messages=messages),
        )
        duration = time.time() - start
        usage = response.usage or type("U", (), {"prompt_tokens": 0, "completion_tokens": 0})()
        return LLMResponse(
            content=response.choices[0].message.content or "",
            tokens_used=usage.prompt_tokens + usage.completion_tokens,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            model=response.model or self.model,
            provider=self.provider_name.lower(),
            duration_seconds=duration,
        )

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens + output_tokens) / 1_000_000 * 1.0

    async def check_available(self) -> bool:
        return bool(self.api_key)


class CohereBackend(LLMBackend):
    """Cohere backend — strong for RAG and enterprise tasks."""

    def __init__(self, model: str = "command-r-plus", api_key: Optional[str] = None):
        _ensure_environment_loaded()
        self.model = model
        self.api_key = api_key or os.getenv("COHERE_API_KEY")

    @property
    def name(self) -> str:
        return f"Cohere ({self.model})"

    async def generate(self, prompt: str, system_prompt: str = "", max_tokens: int = 8000) -> LLMResponse:
        import time
        import urllib.request
        import json as _json
        start = time.time()
        payload = {
            "model": self.model,
            "message": prompt,
            "max_tokens": max_tokens,
        }
        if system_prompt:
            payload["preamble"] = system_prompt
        data = _json.dumps(payload).encode()
        req = urllib.request.Request(
            "https://api.cohere.com/v1/chat",
            data=data,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        loop = asyncio.get_event_loop()
        resp_bytes = await loop.run_in_executor(None, lambda: urllib.request.urlopen(req, timeout=120).read())
        result = _json.loads(resp_bytes)
        duration = time.time() - start
        content = result.get("text", "")
        meta = result.get("meta", {}).get("tokens", {})
        return LLMResponse(
            content=content,
            tokens_used=meta.get("input_tokens", 0) + meta.get("output_tokens", 0),
            input_tokens=meta.get("input_tokens", 0),
            output_tokens=meta.get("output_tokens", 0),
            model=self.model,
            provider="cohere",
            duration_seconds=duration,
        )

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens / 1_000_000) * 2.5 + (output_tokens / 1_000_000) * 10.0

    async def check_available(self) -> bool:
        return bool(self.api_key)


class ReplicateBackend(LLMBackend):
    """Replicate backend — run any open model on demand."""

    def __init__(self, model: str = "meta/llama-4-maverick-instruct", api_key: Optional[str] = None):
        _ensure_environment_loaded()
        self.model = model
        self.api_key = api_key or os.getenv("REPLICATE_API_TOKEN")

    @property
    def name(self) -> str:
        return f"Replicate ({self.model})"

    async def generate(self, prompt: str, system_prompt: str = "", max_tokens: int = 8000) -> LLMResponse:
        import time
        import urllib.request
        import json as _json
        start = time.time()
        inp = {"prompt": prompt, "max_tokens": max_tokens}
        if system_prompt:
            inp["system_prompt"] = system_prompt
        payload = {"input": inp}
        data = _json.dumps(payload).encode()
        req = urllib.request.Request(
            f"https://api.replicate.com/v1/models/{self.model}/predictions",
            data=data,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json", "Prefer": "wait"},
        )
        loop = asyncio.get_event_loop()
        resp_bytes = await loop.run_in_executor(None, lambda: urllib.request.urlopen(req, timeout=120).read())
        result = _json.loads(resp_bytes)
        duration = time.time() - start
        output = result.get("output", "")
        if isinstance(output, list):
            output = "".join(output)
        return LLMResponse(
            content=output,
            tokens_used=0,
            model=self.model,
            provider="replicate",
            duration_seconds=duration,
        )

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens + output_tokens) / 1_000_000 * 0.5

    async def check_available(self) -> bool:
        return bool(self.api_key)


class HuggingFaceBackend(LLMBackend):
    """HuggingFace Inference API backend."""

    def __init__(self, model: str = "mistralai/Mistral-7B-Instruct-v0.3", api_key: Optional[str] = None):
        _ensure_environment_loaded()
        self.model = model
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_TOKEN") or os.getenv("HF_TOKEN")

    @property
    def name(self) -> str:
        return f"HuggingFace ({self.model})"

    async def generate(self, prompt: str, system_prompt: str = "", max_tokens: int = 8000) -> LLMResponse:
        import time
        import urllib.request
        import json as _json
        start = time.time()
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        payload = {"inputs": full_prompt, "parameters": {"max_new_tokens": min(max_tokens, 4096), "return_full_text": False}}
        data = _json.dumps(payload).encode()
        req = urllib.request.Request(
            f"https://api-inference.huggingface.co/models/{self.model}",
            data=data,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        loop = asyncio.get_event_loop()
        resp_bytes = await loop.run_in_executor(None, lambda: urllib.request.urlopen(req, timeout=120).read())
        result = _json.loads(resp_bytes)
        duration = time.time() - start
        content = ""
        if isinstance(result, list) and result:
            content = result[0].get("generated_text", "")
        elif isinstance(result, dict):
            content = result.get("generated_text", result.get("error", ""))
        return LLMResponse(
            content=content,
            tokens_used=0,
            model=self.model,
            provider="huggingface",
            duration_seconds=duration,
        )

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return 0.0  # Free tier

    async def check_available(self) -> bool:
        return bool(self.api_key)


# Convenience presets
BACKENDS = {
    # Cloud providers
    "anthropic-sonnet": lambda: AnthropicBackend(model="claude-sonnet-4-20250514"),
    "anthropic-haiku": lambda: AnthropicBackend(model="claude-haiku-4-20250514"),
    "openai-gpt4o": lambda: OpenAIBackend(model="gpt-4o"),
    "openai-gpt4o-mini": lambda: OpenAIBackend(model="gpt-4o-mini"),
    
    # Qwen3 with thinking mode (recommended for creative writing)
    "ollama-qwen3-30b": lambda: OllamaBackend(model="qwen3:30b-a3b", thinking_mode=True),  # MoE - best efficiency
    "ollama-qwen3-8b": lambda: OllamaBackend(model="qwen3:8b", thinking_mode=True),        # Fast, laptop-friendly
    "ollama-qwen3-14b": lambda: OllamaBackend(model="qwen3:14b", thinking_mode=True),      # Middle ground
    "ollama-qwen3-32b": lambda: OllamaBackend(model="qwen3:32b", thinking_mode=True),      # Dense, high quality
    
    # Qwen3 without thinking mode (faster, less reasoning)
    "ollama-qwen3-30b-fast": lambda: OllamaBackend(model="qwen3:30b-a3b", thinking_mode=False),
    "ollama-qwen3-8b-fast": lambda: OllamaBackend(model="qwen3:8b", thinking_mode=False),
    
    # Qwen2.5 (legacy)
    "ollama-qwen-32b": lambda: OllamaBackend(model="qwen2.5:32b"),
    "ollama-qwen-14b": lambda: OllamaBackend(model="qwen2.5:14b"),
    "ollama-qwen-7b": lambda: OllamaBackend(model="qwen2.5:7b"),
    
    # C-Suite fine-tuned models on RunPod GPU pod (L40S, Ollama)
    # All 11 domain specialists served from a single remote Ollama instance
    "runpod-csuite-merged": lambda: OllamaBackend(model="csuite-merged:latest", host=os.environ.get("RUNPOD_OLLAMA_URL", "http://localhost:11434"), thinking_mode=False),
    "runpod-csuite-tech-core": lambda: OllamaBackend(model="csuite-tech_core:latest", host=os.environ.get("RUNPOD_OLLAMA_URL", "http://localhost:11434"), thinking_mode=False),
    "runpod-csuite-security-compliance": lambda: OllamaBackend(model="csuite-security_compliance:latest", host=os.environ.get("RUNPOD_OLLAMA_URL", "http://localhost:11434"), thinking_mode=False),
    "runpod-csuite-revenue-finance": lambda: OllamaBackend(model="csuite-revenue_finance:latest", host=os.environ.get("RUNPOD_OLLAMA_URL", "http://localhost:11434"), thinking_mode=False),
    "runpod-csuite-product-strategy": lambda: OllamaBackend(model="csuite-product_strategy:latest", host=os.environ.get("RUNPOD_OLLAMA_URL", "http://localhost:11434"), thinking_mode=False),
    "runpod-csuite-data-research": lambda: OllamaBackend(model="csuite-data_research:latest", host=os.environ.get("RUNPOD_OLLAMA_URL", "http://localhost:11434"), thinking_mode=False),
    "runpod-csuite-operations-coordination": lambda: OllamaBackend(model="csuite-operations_coordination:latest", host=os.environ.get("RUNPOD_OLLAMA_URL", "http://localhost:11434"), thinking_mode=False),
    "runpod-csuite-governance": lambda: OllamaBackend(model="csuite-governance:latest", host=os.environ.get("RUNPOD_OLLAMA_URL", "http://localhost:11434"), thinking_mode=False),
    "runpod-csuite-operations": lambda: OllamaBackend(model="csuite-operations:latest", host=os.environ.get("RUNPOD_OLLAMA_URL", "http://localhost:11434"), thinking_mode=False),
    "runpod-csuite-personality": lambda: OllamaBackend(model="csuite-personality:latest", host=os.environ.get("RUNPOD_OLLAMA_URL", "http://localhost:11434"), thinking_mode=False),
    "runpod-csuite-technical": lambda: OllamaBackend(model="csuite-technical:latest", host=os.environ.get("RUNPOD_OLLAMA_URL", "http://localhost:11434"), thinking_mode=False),

    # Local Ollama models (fallback when RunPod pod is down)
    "ollama-csuite-model": lambda: OllamaBackend(model="csuite-model:latest", thinking_mode=False),
    "ollama-devstral": lambda: OllamaBackend(model="devstral:latest"),
    "ollama-command-r-35b": lambda: OllamaBackend(model="command-r:35b"),
    "ollama-deepseek-r1-14b": lambda: OllamaBackend(model="deepseek-r1:14b"),

    # Other Ollama models
    "ollama-llama-8b": lambda: OllamaBackend(model="llama3.1:8b"),
    "ollama-llama-70b": lambda: OllamaBackend(model="llama3.1:70b"),

    # OpenRouter (free models)
    "openrouter-qwen3-coder": lambda: OpenRouterBackend(model="qwen/qwen3-coder:free"),
    "openrouter-nemotron-120b": lambda: OpenRouterBackend(model="nvidia/nemotron-3-super-120b-a12b:free"),
    "openrouter-qwen3-80b": lambda: OpenRouterBackend(model="qwen/qwen3-next-80b-a3b-instruct:free"),
    "openrouter-gpt-oss-120b": lambda: OpenRouterBackend(model="openai/gpt-oss-120b:free"),

    # Claude Max (via CLI — no per-token cost)
    "claude-max-sonnet": lambda: ClaudeCodeBackend(model="sonnet"),
    "claude-max-opus": lambda: ClaudeCodeBackend(model="opus"),
    "claude-max-haiku": lambda: ClaudeCodeBackend(model="haiku"),

    # Google
    "google-gemini-flash": lambda: GoogleBackend(model="gemini-2.5-flash"),
    "google-gemini-pro": lambda: GoogleBackend(model="gemini-2.5-pro"),

    # DeepSeek (OpenAI-compatible)
    "deepseek-chat": lambda: OpenAICompatibleBackend("DeepSeek", "deepseek-chat", "https://api.deepseek.com/v1", env_var="DEEPSEEK_API_KEY"),
    "deepseek-reasoner": lambda: OpenAICompatibleBackend("DeepSeek", "deepseek-reasoner", "https://api.deepseek.com/v1", env_var="DEEPSEEK_API_KEY"),

    # Mistral (OpenAI-compatible)
    "mistral-large": lambda: OpenAICompatibleBackend("Mistral", "mistral-large-latest", "https://api.mistral.ai/v1", env_var="MISTRAL_API_KEY"),
    "mistral-small": lambda: OpenAICompatibleBackend("Mistral", "mistral-small-latest", "https://api.mistral.ai/v1", env_var="MISTRAL_API_KEY"),
    "mistral-codestral": lambda: OpenAICompatibleBackend("Mistral", "codestral-latest", "https://api.mistral.ai/v1", env_var="MISTRAL_API_KEY"),

    # Perplexity (OpenAI-compatible, has web search)
    "perplexity-sonar": lambda: OpenAICompatibleBackend("Perplexity", "sonar", "https://api.perplexity.ai", env_var="PERPLEXITY_API_KEY"),
    "perplexity-sonar-pro": lambda: OpenAICompatibleBackend("Perplexity", "sonar-pro", "https://api.perplexity.ai", env_var="PERPLEXITY_API_KEY"),

    # Together AI (OpenAI-compatible, cheap open models)
    "together-llama-70b": lambda: OpenAICompatibleBackend("Together", "meta-llama/Llama-3.3-70B-Instruct-Turbo", "https://api.together.xyz/v1", env_var="TOGETHER_AI_API_KEY"),
    "together-qwen-72b": lambda: OpenAICompatibleBackend("Together", "Qwen/Qwen2.5-72B-Instruct-Turbo", "https://api.together.xyz/v1", env_var="TOGETHER_AI_API_KEY"),
    "together-deepseek-r1": lambda: OpenAICompatibleBackend("Together", "deepseek-ai/DeepSeek-R1", "https://api.together.xyz/v1", env_var="TOGETHER_AI_API_KEY"),

    # GitHub Models (OpenAI-compatible via Azure)
    "github-gpt4o": lambda: OpenAICompatibleBackend("GitHub", "gpt-4o", "https://models.inference.ai.azure.com", env_var="GITHUB_PERSONAL_ACCESS_TOKEN"),
    "github-gpt4o-mini": lambda: OpenAICompatibleBackend("GitHub", "gpt-4o-mini", "https://models.inference.ai.azure.com", env_var="GITHUB_PERSONAL_ACCESS_TOKEN"),

    # Cohere
    "cohere-command-r-plus": lambda: CohereBackend(model="command-r-plus"),
    "cohere-command-r": lambda: CohereBackend(model="command-r"),

    # Replicate
    "replicate-llama-maverick": lambda: ReplicateBackend(model="meta/llama-4-maverick-instruct"),

    # HuggingFace Inference
    "hf-mistral-7b": lambda: HuggingFaceBackend(model="mistralai/Mistral-7B-Instruct-v0.3"),
}


def get_preset_backend(preset: str) -> LLMBackend:
    """Get a backend from preset name."""
    if preset not in BACKENDS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(BACKENDS.keys())}")
    return BACKENDS[preset]()


def list_presets() -> list[str]:
    """List available backend presets."""
    return list(BACKENDS.keys())
