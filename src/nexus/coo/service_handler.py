"""
Nexus Service Handler — handles service requests from C-Suite.

Listens on {prefix}:service:request for incoming service calls and
dispatches to the appropriate NexusPlatform method. Responses are
published to the per-request response channel.

This replaces the "strategic brain" pattern with direct service access
to Nexus's real capabilities: discovery, ensemble, research, knowledge.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)
import os

import httpx

SHOPFORGE_BASE_URL = os.environ.get("SHOPFORGE_BASE_URL", "http://localhost:8003")
SHOPFORGE_SERVICE_TOKEN = os.environ.get("SHOPFORGE_SERVICE_TOKEN", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"



class _TTLCache:
    """Simple TTL cache for discovery results."""

    def __init__(self, ttl_seconds: int = 900):
        self._cache: Dict[str, Tuple[float, Any]] = {}
        self._ttl = ttl_seconds

    def get(self, key: str) -> Optional[Any]:
        entry = self._cache.get(key)
        if entry and time.monotonic() - entry[0] < self._ttl:
            return entry[1]
        return None

    def set(self, key: str, value: Any) -> None:
        self._cache[key] = (time.monotonic(), value)

    @property
    def size(self) -> int:
        return len(self._cache)


def _utcnow():
    return datetime.now(timezone.utc)


# Prometheus metrics (optional — no-op if prometheus_client not installed)
try:
    from prometheus_client import Counter, Histogram

    SERVICE_REQUESTS = Counter(
        "nexus_service_requests_total",
        "Total service requests handled",
        ["service", "status"],
    )
    SERVICE_LATENCY = Histogram(
        "nexus_service_latency_seconds",
        "Service request latency",
        ["service"],
        buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 30.0],
    )
    _PROM_ENABLED = True
except ImportError:
    _PROM_ENABLED = False


class NexusServiceHandler:
    """Handles service requests from C-Suite via Redis pub/sub."""

    def __init__(self, platform, redis_client, channel_prefix: str = "csuite:nexus",
                 cache_ttl: int = 900):
        self._platform = platform
        self._redis = redis_client
        self._prefix = channel_prefix
        self._listening = False
        self._listener_task: Optional[asyncio.Task] = None
        self._requests_handled = 0
        self._errors = 0
        self._cache = _TTLCache(ttl_seconds=cache_ttl)
        self._cache_hits = 0

    async def start(self) -> None:
        """Start listening for service requests."""
        if self._listening:
            return
        self._listening = True
        self._listener_task = asyncio.create_task(self._listen())
        logger.info("NexusServiceHandler started on %s:service:request", self._prefix)

    async def stop(self) -> None:
        """Stop listening."""
        self._listening = False
        if self._listener_task and not self._listener_task.done():
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
        logger.info(
            "NexusServiceHandler stopped (handled=%d, errors=%d)",
            self._requests_handled, self._errors,
        )

    async def _listen(self) -> None:
        channel = f"{self._prefix}:service:request"
        pubsub = self._redis.pubsub()
        await pubsub.subscribe(channel)
        logger.info("Subscribed to %s", channel)

        try:
            async for message in pubsub.listen():
                if not self._listening:
                    break
                if message["type"] != "message":
                    continue
                try:
                    request = json.loads(message["data"])
                    asyncio.create_task(self._handle_request(request))
                except json.JSONDecodeError as e:
                    logger.error("Invalid JSON in service request: %s", e)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Service listener error: %s", e)
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.close()

    async def _handle_request(self, request: Dict[str, Any]) -> None:
        """Dispatch a service request to the appropriate handler."""
        request_id = request.get("request_id", "unknown")
        service = request.get("service", "")
        params = request.get("params", {})
        response_channel = request.get("response_channel", "")

        if not response_channel:
            logger.warning("Service request %s missing response_channel", request_id)
            return

        logger.debug("Handling service request: %s (service=%s)", request_id, service)

        _start = time.monotonic()
        try:
            handler = self._get_handler(service)
            if handler is None:
                if _PROM_ENABLED:
                    SERVICE_REQUESTS.labels(service=service, status="not_found").inc()
                await self._respond(response_channel, request_id, False,
                                    error=f"Unknown service: {service}")
                return

            # Check cache for discovery/search services
            cacheable = service in (
                "search_models", "search_github", "search_arxiv",
                "search_pypi", "web_search", "discover_resources",
                "platform_status", "get_model_profile",
                "find_models_for_task", "list_model_profiles",
                "get_tool_profile", "find_tools_for_task",
                "list_tool_profiles", "get_tools_for_executive",
                "list_distribution_channels", "get_distribution_channel",
                "list_active_channels",
                "select_model_for_task",
            )
            cache_key = f"{service}:{json.dumps(params, sort_keys=True)}" if cacheable else None

            if cache_key:
                cached = self._cache.get(cache_key)
                if cached is not None:
                    self._cache_hits += 1
                    self._requests_handled += 1
                    await self._respond(response_channel, request_id, True, data=cached)
                    return

            result = await handler(params)
            self._requests_handled += 1

            if cache_key and result is not None:
                self._cache.set(cache_key, result)

            if _PROM_ENABLED:
                SERVICE_REQUESTS.labels(service=service, status="success").inc()
                SERVICE_LATENCY.labels(service=service).observe(time.monotonic() - _start)

            await self._respond(response_channel, request_id, True, data=result)

        except Exception as e:
            self._errors += 1
            if _PROM_ENABLED:
                SERVICE_REQUESTS.labels(service=service, status="error").inc()
            logger.error("Service %s failed: %s: %s", service, type(e).__name__, e)
            await self._respond(response_channel, request_id, False, error=str(e))

    def _get_handler(self, service: str):
        """Map service name to handler coroutine."""
        handlers = {
            "search_models": self._search_models,
            "discover_resources": self._discover_resources,
            "search_github": self._search_github,
            "search_arxiv": self._search_arxiv,
            "search_pypi": self._search_pypi,
            "web_search": self._web_search,
            "ensemble_query": self._ensemble_query,
            "expert_opinion": self._expert_opinion,
            "research": self._research,
            "search_knowledge": self._search_knowledge,
            "add_knowledge": self._add_knowledge,
            "platform_status": self._platform_status,
            "get_model_profile": self._get_model_profile,
            "find_models_for_task": self._find_models_for_task,
            "list_model_profiles": self._list_model_profiles,
            "get_tool_profile": self._get_tool_profile,
            "find_tools_for_task": self._find_tools_for_task,
            "list_tool_profiles": self._list_tool_profiles,
            "get_tools_for_executive": self._get_tools_for_executive,
            "vector_store_stats": self._vector_store_stats,
            # Distribution channel discovery
            "list_distribution_channels": self._list_distribution_channels,
            "get_distribution_channel": self._get_distribution_channel,
            # Channel activation registry (mirrors C-Suite ChannelRegistry)
            "register_active_channel": self._register_active_channel,
            "list_active_channels": self._list_active_channels,
            # Shopforge commerce endpoints
            "shopforge_list_storefronts": self._shopforge_list_storefronts,
            "shopforge_get_products": self._shopforge_get_products,
            "shopforge_optimize_pricing": self._shopforge_optimize_pricing,
            "shopforge_apply_price": self._shopforge_apply_price,
            "shopforge_get_analytics": self._shopforge_get_analytics,
            "shopforge_get_margins": self._shopforge_get_margins,
            "shopforge_provision_storefront": self._shopforge_provision_storefront,
            "shopforge_run_analysis": self._shopforge_run_analysis,
            "shopforge_executive_report": self._shopforge_executive_report,
            "shopforge_revenue_summary": self._shopforge_revenue_summary,
            # Outcome feedback loop — multi-LLM routing
            "select_model_for_task": self._select_model_for_task,
            "outcome_loop_query": self._outcome_loop_query,
        }
        return handlers.get(service)

    async def _respond(
        self,
        channel: str,
        request_id: str,
        success: bool,
        data: Any = None,
        error: Optional[str] = None,
    ) -> None:
        response = {
            "type": "service_response",
            "request_id": request_id,
            "success": success,
            "data": data,
            "error": error,
            "timestamp": _utcnow().isoformat(),
        }
        await self._redis.publish(channel, json.dumps(response, default=str))

    # ------------------------------------------------------------------
    # Service Handlers
    # ------------------------------------------------------------------

    async def _search_models(self, params: Dict[str, Any]) -> Any:
        await self._platform.initialize_discovery_path()
        results = await self._platform.search_models(
            query=params.get("query", ""),
            capabilities=params.get("capabilities"),
            max_price=params.get("max_price"),
            min_context=params.get("min_context"),
        )
        return [r.to_dict() if hasattr(r, "to_dict") else r for r in results]

    async def _discover_resources(self, params: Dict[str, Any]) -> Any:
        await self._platform.initialize_discovery_path()
        return await self._platform.discover_resources()

    async def _search_github(self, params: Dict[str, Any]) -> Any:
        await self._platform.initialize_discovery_path()
        results = await self._platform.search_github(
            query=params.get("query", ""),
            limit=params.get("limit", 30),
        )
        return [r.to_dict() if hasattr(r, "to_dict") else r for r in results]

    async def _search_arxiv(self, params: Dict[str, Any]) -> Any:
        await self._platform.initialize_discovery_path()
        results = await self._platform.search_arxiv(
            query=params.get("query", ""),
            max_results=params.get("max_results", 20),
        )
        return [r.to_dict() if hasattr(r, "to_dict") else r for r in results]

    async def _search_pypi(self, params: Dict[str, Any]) -> Any:
        await self._platform.initialize_discovery_path()
        results = await self._platform.search_pypi(
            query=params.get("query", ""),
            max_results=params.get("max_results", 20),
        )
        return [r.to_dict() if hasattr(r, "to_dict") else r for r in results]

    async def _web_search(self, params: Dict[str, Any]) -> Any:
        await self._platform.initialize_discovery_path()
        results = await self._platform.web_search(
            query=params.get("query", ""),
            num_results=params.get("num_results", 10),
        )
        return [r.to_dict() if hasattr(r, "to_dict") else r for r in results]

    async def _ensemble_query(self, params: Dict[str, Any]) -> Any:
        await self._platform.initialize_query_path()
        return await self._platform.query(
            prompt=params.get("prompt", ""),
            strategy=params.get("strategy"),
            model=params.get("model"),
            system_prompt=params.get("system_prompt"),
            max_tokens=params.get("max_tokens", 8000),
        )

    async def _expert_opinion(self, params: Dict[str, Any]) -> Any:
        await self._platform.initialize_expert_path()
        # Simple dict pass-through since we don't have a Task on this side
        return await self._platform.get_expert_opinion(params.get("task_description", ""))

    async def _research(self, params: Dict[str, Any]) -> Any:
        await self._platform.initialize_research_path()
        return await self._platform.research(
            topic=params.get("topic", ""),
        )

    async def _search_knowledge(self, params: Dict[str, Any]) -> Any:
        query = params.get("query", "")
        top_k = params.get("top_k", 10)

        # Try Qdrant semantic search first (if available)
        kb = getattr(self._platform, "_knowledge_base", None) if hasattr(self._platform, "_knowledge_base") else None
        if kb and getattr(kb, "_vector_store", None) and kb._vector_store.is_available:
            try:
                vector_results = await kb._vector_store.search(query, top_k=top_k)
                if vector_results:
                    return vector_results  # Already formatted as dicts
            except Exception:
                pass  # Fall through to keyword search

        # Fallback to keyword search
        if not kb:
            return []
        results = kb.query_knowledge(
            query=query,
            max_results=top_k,
        )
        return [
            {
                "id": item.id,
                "content": item.content,
                "type": item.knowledge_type.value,
                "confidence": item.confidence,
                "source": item.source,
            }
            for item in results
        ]

    async def _add_knowledge(self, params: Dict[str, Any]) -> Any:
        if not hasattr(self._platform, "_knowledge_base"):
            return None
        from nexus.memory.knowledge_base import KnowledgeType
        return self._platform._knowledge_base.add_knowledge(
            content=params.get("content", ""),
            knowledge_type=KnowledgeType(params.get("knowledge_type", "factual")),
            source=params.get("source", "csuite"),
            confidence=params.get("confidence", 0.8),
        )

    async def _platform_status(self, params: Dict[str, Any]) -> Any:
        return self._platform.get_status()

    async def _get_model_profile(self, params: Dict[str, Any]) -> Any:
        """Get rich profile for a specific model."""
        from nexus.providers.adapters.model_profiles import get_model_profile
        model_id = params.get("model_id", "")
        profile = get_model_profile(model_id)
        return profile.to_dict() if profile else None

    async def _find_models_for_task(self, params: Dict[str, Any]) -> Any:
        """Find model profiles suited for a task type + language requirements."""
        from nexus.providers.adapters.model_profiles import get_profile_for_task
        profiles = get_profile_for_task(
            task_type=params.get("task_type", ""),
            required_languages=params.get("required_languages"),
            required_prog_languages=params.get("required_prog_languages"),
        )
        return [p.to_dict() for p in profiles]

    async def _list_model_profiles(self, params: Dict[str, Any]) -> Any:
        """List all known model profiles with their capabilities."""
        from nexus.providers.adapters.model_profiles import get_all_profiles
        return {k: v.to_dict() for k, v in get_all_profiles().items()}

    async def _get_tool_profile(self, params: Dict[str, Any]) -> Any:
        """Get rich profile for a specific tool/service."""
        from nexus.discovery.tool_profiles import get_tool_profile
        profile = get_tool_profile(params.get("name", ""))
        return profile.to_dict() if profile else None

    async def _find_tools_for_task(self, params: Dict[str, Any]) -> Any:
        """Find tools suited for a task with optional filters."""
        from nexus.discovery.tool_profiles import find_tools_for_task
        tools = find_tools_for_task(
            params.get("task_description", ""),
            category=params.get("category"),
            availability=params.get("availability"),
            executive=params.get("executive"),
            require_free=params.get("require_free", False),
        )
        return [t.to_dict() for t in tools]

    async def _list_tool_profiles(self, params: Dict[str, Any]) -> Any:
        """List all tool profiles, optionally filtered by category or availability."""
        from nexus.discovery.tool_profiles import (
            get_all_tool_profiles, get_tools_by_category, get_tools_by_availability,
        )
        category = params.get("category")
        availability = params.get("availability")
        if category:
            tools = get_tools_by_category(category)
        elif availability:
            tools = get_tools_by_availability(availability)
        else:
            tools = list(get_all_tool_profiles().values())
        return [t.to_dict() for t in tools]

    async def _get_tools_for_executive(self, params: Dict[str, Any]) -> Any:
        """Get all tools recommended for a specific executive."""
        from nexus.discovery.tool_profiles import get_tools_for_executive
        tools = get_tools_for_executive(params.get("executive", ""))
        return [t.to_dict() for t in tools]

    async def _vector_store_stats(self, params: Dict[str, Any]) -> Any:
        """Get Qdrant vector store statistics."""
        kb = getattr(self._platform, "_knowledge_base", None) if hasattr(self._platform, "_knowledge_base") else None
        if kb and getattr(kb, "_vector_store", None):
            return await kb._vector_store.get_stats()
        return {"available": False, "reason": "vector store not initialized"}


    # ------------------------------------------------------------------
    # Distribution Channel Handlers
    # ------------------------------------------------------------------

    # Canonical channel names → category grouping for C-Suite routing
    _DISTRIBUTION_CHANNEL_GROUPS = {
        "email": ["beehiiv", "mailchimp", "convertkit", "resend"],
        "newsletter": ["beehiiv", "substack", "convertkit"],
        "blog": ["ghost", "wordpress_api", "hashnode"],
        "video": ["youtube_studio"],
        "podcast": ["buzzsprout"],
        "digital_products": ["gumroad", "lemon_squeezy"],
        "social": ["twitter_api", "linkedin_api"],
        "commerce": ["shopforge", "mdusa"],
    }

    async def _register_active_channel(self, params: dict) -> Any:
        """
        Register a newly-activated distribution channel in Nexus knowledge base.

        C-Suite calls this after a channel transitions to ACTIVE so Nexus can
        factor it into tool recommendations and research context.

        Params:
            channel (str): channel name, e.g. "beehiiv"
            product_line (str): brand/product line
            channel_url (str, optional): public URL
            activation_id (str, optional): C-Suite activation record ID
        """
        channel = params.get("channel", "")
        product_line = params.get("product_line", "")
        if not channel or not product_line:
            return {"success": False, "error": "channel and product_line required"}

        kb = getattr(self._platform, "_knowledge_base", None)
        if kb:
            try:
                from nexus.memory.knowledge_base import KnowledgeType
                kb.add_knowledge(
                    content=(
                        f"Active distribution channel: {channel} for '{product_line}'. "
                        f"URL: {params.get('channel_url', 'pending')}. "
                        f"Activation ID: {params.get('activation_id', '')}."
                    ),
                    knowledge_type=KnowledgeType("factual"),
                    source="channel_activation",
                    confidence=0.95,
                )
            except Exception as e:
                logger.warning("Failed to store channel in knowledge base: %s", e)

        return {
            "success": True,
            "channel": channel,
            "product_line": product_line,
            "registered": True,
        }

    async def _list_active_channels(self, params: dict) -> Any:
        """
        Return active channel registrations from Nexus knowledge base.

        Params:
            product_line (str, optional): filter by product line
        """
        kb = getattr(self._platform, "_knowledge_base", None)
        if not kb:
            return {"channels": [], "source": "no_knowledge_base"}

        product_line = params.get("product_line", "")
        query = (
            f"active distribution channel {product_line}"
            if product_line else "active distribution channel"
        )
        results = kb.query_knowledge(query=query, max_results=50)
        channels = [
            {"content": r.content, "source": r.source, "confidence": r.confidence}
            for r in results
            if "channel_activation" in (r.source or "")
        ]
        return {"channels": channels, "count": len(channels)}

    async def _list_distribution_channels(self, params: dict) -> Any:
        """
        Return all distribution channel tool profiles, grouped by channel type.

        Params:
            group (str, optional): filter to a specific group
                (email|newsletter|blog|video|podcast|digital_products|social|commerce)
        """
        from nexus.discovery.tool_profiles import get_tool_profile

        group_filter = params.get("group")
        groups = (
            {group_filter: self._DISTRIBUTION_CHANNEL_GROUPS[group_filter]}
            if group_filter and group_filter in self._DISTRIBUTION_CHANNEL_GROUPS
            else self._DISTRIBUTION_CHANNEL_GROUPS
        )

        result: dict = {}
        for group, names in groups.items():
            channels = []
            for name in names:
                profile = get_tool_profile(name)
                if profile:
                    channels.append(profile.to_dict())
                else:
                    # Minimal stub for channels without a full profile yet
                    channels.append({"name": name, "display_name": name, "available": False})
            result[group] = channels

        return result

    async def _get_distribution_channel(self, params: dict) -> Any:
        """
        Return the full tool profile for a single distribution channel.

        Params:
            name (str): channel name, e.g. "beehiiv", "ghost", "gumroad"
        """
        from nexus.discovery.tool_profiles import get_tool_profile

        name = params.get("name", "")
        if not name:
            return None
        profile = get_tool_profile(name)
        return profile.to_dict() if profile else None

    # ------------------------------------------------------------------
    # Shopforge Commerce Handlers
    # ------------------------------------------------------------------

    async def _shopforge_request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Shared helper for Shopforge HTTP requests."""
        headers = {}
        if SHOPFORGE_SERVICE_TOKEN:
            headers["X-Service-Token"] = SHOPFORGE_SERVICE_TOKEN
        url = f"{SHOPFORGE_BASE_URL}{path}"
        async with httpx.AsyncClient(timeout=15.0) as client:
            if method == "GET":
                resp = await client.get(url, headers=headers, params=params)
            elif method == "POST":
                resp = await client.post(url, headers=headers, json=json_body, params=params)
            elif method == "PUT":
                resp = await client.put(url, headers=headers, json=json_body, params=params)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            resp.raise_for_status()
            return resp.json()

    async def _shopforge_list_storefronts(self, params: Dict[str, Any]) -> Any:
        return await self._shopforge_request("GET", "/v1/storefronts")

    async def _shopforge_get_products(self, params: Dict[str, Any]) -> Any:
        storefront_key = params.get("storefront_key", "")
        limit = params.get("limit", 100)
        return await self._shopforge_request(
            "GET", f"/v1/products/{storefront_key}", params={"limit": limit},
        )

    async def _shopforge_optimize_pricing(self, params: Dict[str, Any]) -> Any:
        return await self._shopforge_request(
            "POST", "/v1/pricing/optimize",
            params={
                "storefront_key": params.get("storefront_key", ""),
                "target_margin": params.get("target_margin", 40.0),
                "strategy": params.get("strategy", "cost_plus"),
            },
        )

    async def _shopforge_apply_price(self, params: Dict[str, Any]) -> Any:
        storefront_key = params.get("storefront_key", "")
        return await self._shopforge_request(
            "PUT", f"/v1/pricing/update/{storefront_key}",
            json_body={
                "product_id": params.get("product_id", ""),
                "variant_id": params.get("variant_id", ""),
                "new_price": params.get("new_price", 0),
                "compare_at_price": params.get("compare_at_price"),
            },
        )

    async def _shopforge_get_analytics(self, params: Dict[str, Any]) -> Any:
        return await self._shopforge_request("GET", "/v1/analytics")

    async def _shopforge_get_margins(self, params: Dict[str, Any]) -> Any:
        storefront_key = params.get("storefront_key")
        qp = {"storefront_key": storefront_key} if storefront_key else None
        return await self._shopforge_request("GET", "/v1/margins", params=qp)

    async def _shopforge_provision_storefront(self, params: Dict[str, Any]) -> Any:
        return await self._shopforge_request(
            "POST", "/v1/storefronts/provision", json_body=params,
        )

    async def _shopforge_run_analysis(self, params: Dict[str, Any]) -> Any:
        return await self._shopforge_request("POST", "/v1/autonomous/analyze")

    async def _shopforge_executive_report(self, params: Dict[str, Any]) -> Any:
        executive_code = params.get("executive_code", "CRO")
        return await self._shopforge_request(
            "GET", f"/v1/executive/{executive_code}",
        )

    async def _shopforge_revenue_summary(self, params: Dict[str, Any]) -> Any:
        return await self._shopforge_request("GET", "/v1/revenue/summary")

    # ------------------------------------------------------------------
    # Outcome feedback loop — multi-LLM routing
    # ------------------------------------------------------------------

    def _get_task_model_router(self):
        """Lazy-init the TaskModelRouter singleton."""
        if not hasattr(self, "_task_model_router"):
            from nexus.core.task_model_router import TaskModelRouter
            self._task_model_router = TaskModelRouter()
        return self._task_model_router

    async def _select_model_for_task(self, params: Dict[str, Any]) -> Any:
        """Return the best model for a given outcome loop step."""
        task_step = params.get("task_step", "decide")
        context = params.get("context", {})
        router = self._get_task_model_router()
        selection = router.select_model_for_task(task_step, context)
        return {
            "model_id": selection.model_id,
            "task_step": selection.task_step,
            "fallback_chain": selection.fallback_chain,
            "reasoning": selection.reasoning,
        }

    async def _outcome_loop_query(self, params: Dict[str, Any]) -> Any:
        """Select model for a task step AND execute the LLM call in one round-trip.

        This avoids 2 Redis round-trips (select + query) per step. Tries the
        primary model first, then walks fallbacks on failure.
        """
        task_step = params.get("task_step", "decide")
        prompt = params.get("prompt", "")
        system_prompt = params.get("system_prompt", "")
        max_tokens = params.get("max_tokens", 4000)

        if not prompt:
            return {"error": "prompt is required", "response": None}

        router = self._get_task_model_router()
        selection = router.select_model_for_task(task_step, params.get("context"))

        # Try primary, then fallbacks
        models_to_try = [selection.model_id] + selection.fallback_chain
        last_error = None

        for model_id in models_to_try:
            try:
                response_text = await self._call_openrouter(
                    model_id, prompt, system_prompt, max_tokens,
                )
                router.record_success(model_id)
                return {
                    "response": response_text,
                    "model_used": model_id,
                    "task_step": task_step,
                    "fallback_used": model_id != selection.model_id,
                }
            except Exception as e:
                last_error = str(e)
                router.record_failure(model_id)
                logger.warning(
                    "Outcome loop query failed for %s with %s: %s",
                    task_step, model_id, e,
                )

        return {
            "response": None,
            "model_used": None,
            "task_step": task_step,
            "fallback_used": True,
            "error": f"All models failed for step '{task_step}': {last_error}",
        }

    async def _call_openrouter(
        self,
        model_id: str,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 4000,
    ) -> str:
        """Make a chat completion call to OpenRouter."""
        if not OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY not set")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                OPENROUTER_BASE_URL,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model_id,
                    "messages": messages,
                    "max_tokens": max_tokens,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                raise ValueError("No choices returned from OpenRouter")
            return choices[0].get("message", {}).get("content", "")

    def get_stats(self) -> Dict[str, Any]:
        stats = {
            "requests_handled": self._requests_handled,
            "errors": self._errors,
            "cache_hits": self._cache_hits,
            "cache_size": self._cache.size,
            "listening": self._listening,
        }
        if hasattr(self, "_task_model_router"):
            stats["task_model_router"] = self._task_model_router.get_stats()
        return stats
