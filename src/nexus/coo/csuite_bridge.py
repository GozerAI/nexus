"""Legacy bridge for older csuite/Nexus orchestration experiments.

The bridge is retained so existing integrations keep working, but its role is
now compatibility-oriented: it connects external control planes to shared
Nexus infrastructure and optional legacy coordination flows.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from nexus.coo.core import AutonomousCOO

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Get current UTC time as timezone-aware datetime."""
    return datetime.now(timezone.utc)


@dataclass
class CSuiteBridgeConfig:
    """Configuration for the csuite bridge listener."""
    redis_url: str = "redis://localhost:6379"
    channel_prefix: str = "csuite:nexus"
    response_timeout_seconds: int = 30
    reconnect_delay_seconds: float = 5.0
    max_reconnect_attempts: int = 10


class CSuiteBridgeListener:
    """
    Listener for csuite requests.

    This class runs on the Nexus side and:
    1. Listens for compatibility-era guidance requests from CoS
    2. Receives outcome reports for cross-system learning
    3. Receives health status updates from csuite
    4. Can publish compatibility directives back to CoS

    Communication uses Redis pub/sub with the following channels:
    - {prefix}:guidance:request - CoS requests strategic guidance
    - {prefix}:guidance:response - Nexus sends guidance back
    - {prefix}:outcomes - CoS reports task outcomes
    - {prefix}:health - CoS publishes health status
    - {prefix}:directives - Nexus sends directives to CoS
    """

    def __init__(
        self,
        coo: "AutonomousCOO",
        config: Optional[CSuiteBridgeConfig] = None,
    ):
        """
        Initialize the csuite bridge listener.

        Args:
            coo: The legacy AutonomousCOO instance this listener belongs to
            config: Bridge configuration
        """
        self._coo = coo
        self.config = config or CSuiteBridgeConfig()
        self._redis = None
        self._connected = False
        self._listening = False
        self._listener_task: Optional[asyncio.Task] = None
        self._health_listener_task: Optional[asyncio.Task] = None
        self._outcome_listener_task: Optional[asyncio.Task] = None

        # Metrics
        self._guidance_requests_received = 0
        self._guidance_responses_sent = 0
        self._outcomes_received = 0
        self._health_updates_received = 0
        self._directives_sent = 0

        # Last received health status from csuite
        self._last_csuite_health: Optional[Dict[str, Any]] = None
        self._last_health_timestamp: Optional[datetime] = None

    @property
    def is_connected(self) -> bool:
        """Check if connected to Redis."""
        return self._connected

    @property
    def is_listening(self) -> bool:
        """Check if actively listening for messages."""
        return self._listening

    async def connect(self) -> bool:
        """Connect to Redis."""
        try:
            import redis.asyncio as redis
            self._redis = redis.from_url(self.config.redis_url)
            await self._redis.ping()
            self._connected = True
            logger.info(f"CSuiteBridgeListener connected to Redis: {self.config.redis_url}")
            return True
        except ImportError:
            logger.error("redis package not installed. Install with: pip install redis")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from Redis and stop listening."""
        await self.stop_listening()
        if self._redis:
            await self._redis.close()
            self._connected = False
            logger.info("CSuiteBridgeListener disconnected from Redis")

    async def start_listening(self) -> bool:
        """
        Start listening for messages from csuite.

        Returns:
            True if listening started successfully
        """
        if not self._connected:
            success = await self.connect()
            if not success:
                return False

        if self._listening:
            logger.warning("Already listening for csuite messages")
            return True

        self._listening = True

        # Start listener tasks
        self._listener_task = asyncio.create_task(
            self._listen_for_guidance_requests()
        )
        self._outcome_listener_task = asyncio.create_task(
            self._listen_for_outcomes()
        )
        self._health_listener_task = asyncio.create_task(
            self._listen_for_health_updates()
        )

        logger.info("CSuiteBridgeListener started listening for messages")
        return True

    async def stop_listening(self) -> None:
        """Stop listening for messages."""
        self._listening = False

        # Cancel listener tasks
        for task in [
            self._listener_task,
            self._outcome_listener_task,
            self._health_listener_task,
        ]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self._listener_task = None
        self._outcome_listener_task = None
        self._health_listener_task = None

        logger.info("CSuiteBridgeListener stopped listening")

    async def _listen_for_guidance_requests(self) -> None:
        """Listen for guidance requests from CoS."""
        channel = f"{self.config.channel_prefix}:guidance:request"

        while self._listening:
            try:
                pubsub = self._redis.pubsub()
                await pubsub.subscribe(channel)
                logger.debug(f"Subscribed to {channel}")

                async for message in pubsub.listen():
                    if not self._listening:
                        break

                    if message["type"] == "message":
                        try:
                            request = json.loads(message["data"])
                            await self.handle_guidance_request(request)
                        except json.JSONDecodeError as e:
                            logger.error(f"Invalid JSON in guidance request: {e}")
                        except Exception as e:
                            logger.error(f"Error handling guidance request: {e}")

                await pubsub.unsubscribe(channel)
                await pubsub.close()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in guidance listener: {e}")
                if self._listening:
                    await asyncio.sleep(self.config.reconnect_delay_seconds)

    async def _listen_for_outcomes(self) -> None:
        """Listen for outcome reports from CoS."""
        channel = f"{self.config.channel_prefix}:outcomes"

        while self._listening:
            try:
                pubsub = self._redis.pubsub()
                await pubsub.subscribe(channel)
                logger.debug(f"Subscribed to {channel}")

                async for message in pubsub.listen():
                    if not self._listening:
                        break

                    if message["type"] == "message":
                        try:
                            report = json.loads(message["data"])
                            await self.handle_outcome_report(report)
                        except json.JSONDecodeError as e:
                            logger.error(f"Invalid JSON in outcome report: {e}")
                        except Exception as e:
                            logger.error(f"Error handling outcome report: {e}")

                await pubsub.unsubscribe(channel)
                await pubsub.close()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in outcome listener: {e}")
                if self._listening:
                    await asyncio.sleep(self.config.reconnect_delay_seconds)

    async def _listen_for_health_updates(self) -> None:
        """Listen for health status updates from CoS."""
        channel = f"{self.config.channel_prefix}:health"

        while self._listening:
            try:
                pubsub = self._redis.pubsub()
                await pubsub.subscribe(channel)
                logger.debug(f"Subscribed to {channel}")

                async for message in pubsub.listen():
                    if not self._listening:
                        break

                    if message["type"] == "message":
                        try:
                            health_data = json.loads(message["data"])
                            await self._handle_health_update(health_data)
                        except json.JSONDecodeError as e:
                            logger.error(f"Invalid JSON in health update: {e}")
                        except Exception as e:
                            logger.error(f"Error handling health update: {e}")

                await pubsub.unsubscribe(channel)
                await pubsub.close()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health listener: {e}")
                if self._listening:
                    await asyncio.sleep(self.config.reconnect_delay_seconds)

    async def handle_guidance_request(self, request: Dict[str, Any]) -> None:
        """
        Process a guidance request from CoS.

        When CoS detects drift or needs strategic input, it sends a request
        to Nexus. This method processes that request and sends back guidance.

        Args:
            request: The guidance request containing drift context
        """
        self._guidance_requests_received += 1
        logger.info(f"Received guidance request: {request.get('type', 'unknown')}")

        drift_context = request.get("drift_context", {})
        timestamp = request.get("timestamp")

        # Generate strategic guidance based on drift context
        guidance = await self._generate_guidance(drift_context)

        # Send response
        await self._send_guidance_response(guidance)

    async def _generate_guidance(
        self,
        drift_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate strategic guidance based on drift context.

        This uses the AutonomousCOO's intelligence to provide guidance.

        Args:
            drift_context: Context about the detected drift

        Returns:
            Strategic guidance with updated context/priorities
        """
        # Extract drift information
        drift_type = drift_context.get("drift_type", "unknown")
        drift_level = drift_context.get("drift_level", 0.0)
        current_metrics = drift_context.get("metrics", {})

        guidance = {
            "generated_at": _utcnow().isoformat(),
            "context": None,
            "recommendations": [],
            "priority_adjustments": {},
        }

        # Get current COO state for context
        if self._coo:
            coo_status = self._coo.get_status()

            # Generate recommendations based on drift type
            if drift_type == "performance":
                guidance["recommendations"].append(
                    "Consider reducing concurrent task load"
                )
                if drift_level > 0.7:
                    guidance["priority_adjustments"]["max_concurrent"] = (
                        max(1, self._coo.config.max_concurrent_executions - 1)
                    )

            elif drift_type == "accuracy":
                guidance["recommendations"].append(
                    "Increase confidence threshold for auto-execution"
                )
                if drift_level > 0.5:
                    guidance["priority_adjustments"]["confidence_threshold"] = (
                        min(0.95, self._coo.config.auto_execute_confidence + 0.05)
                    )

            elif drift_type == "resource":
                guidance["recommendations"].append(
                    "Monitor resource usage, consider throttling"
                )

            # Provide strategic context update
            guidance["context"] = {
                "routing_priorities": self._calculate_routing_priorities(drift_context),
                "success_rate_threshold": self._calculate_success_threshold(drift_level),
                "execution_mode": coo_status.mode.value,
            }

        return guidance

    def _calculate_routing_priorities(
        self,
        drift_context: Dict[str, Any],
    ) -> Dict[str, float]:
        """Calculate routing priorities based on drift context."""
        # Default priorities
        priorities = {
            "urgent": 1.0,
            "high": 0.8,
            "medium": 0.5,
            "low": 0.2,
        }

        # Adjust based on drift
        if drift_context.get("drift_level", 0) > 0.5:
            # When drifting, prioritize urgent tasks more heavily
            priorities["urgent"] = 1.0
            priorities["high"] = 0.6
            priorities["medium"] = 0.3
            priorities["low"] = 0.1

        return priorities

    def _calculate_success_threshold(self, drift_level: float) -> float:
        """Calculate success rate threshold based on drift level."""
        # Higher drift = stricter threshold
        base_threshold = 0.8
        return min(0.95, base_threshold + (drift_level * 0.15))

    async def _send_guidance_response(self, guidance: Dict[str, Any]) -> None:
        """Send guidance response back to CoS."""
        channel = f"{self.config.channel_prefix}:guidance:response"

        try:
            await self._redis.publish(channel, json.dumps({
                "type": "guidance_response",
                "guidance": guidance,
                "timestamp": _utcnow().isoformat(),
            }))
            self._guidance_responses_sent += 1
            logger.debug("Sent guidance response to CoS")
        except Exception as e:
            logger.error(f"Failed to send guidance response: {e}")

    async def handle_outcome_report(self, report: Dict[str, Any]) -> None:
        """
        Process an outcome report from CoS.

        Outcome reports are used for cross-system learning. They contain
        information about task execution results that Nexus can use to
        improve its strategic decisions.

        Args:
            report: The outcome report with task metrics
        """
        self._outcomes_received += 1
        metrics = report.get("metrics", {})

        logger.info(
            f"Received outcome report: task_id={metrics.get('task_id', 'unknown')}, "
            f"success={metrics.get('success', 'unknown')}"
        )

        # Forward to COO learning system if available
        if self._coo and hasattr(self._coo, '_learning') and self._coo._learning:
            try:
                # Create a minimal outcome record for learning
                await self._coo._learning.record_external_outcome(
                    source="csuite",
                    task_id=metrics.get("task_id"),
                    success=metrics.get("success", False),
                    metrics=metrics,
                    timestamp=report.get("timestamp"),
                )
                logger.debug("Recorded outcome in COO learning system")
            except AttributeError:
                # record_external_outcome may not exist yet
                logger.debug("COO learning system doesn't support external outcomes yet")
            except Exception as e:
                logger.error(f"Failed to record outcome in learning system: {e}")

    async def _handle_health_update(self, health_data: Dict[str, Any]) -> None:
        """
        Process a health status update from CoS.

        Args:
            health_data: Health data from csuite
        """
        self._health_updates_received += 1
        self._last_csuite_health = health_data.get("data", {})
        self._last_health_timestamp = _utcnow()

        logger.debug(
            f"Received health update from csuite: "
            f"source={health_data.get('source', 'unknown')}"
        )

        # Check for concerning health indicators
        if self._last_csuite_health:
            drift_status = self._last_csuite_health.get("drift_status", {})
            if drift_status.get("is_drifting"):
                logger.warning(
                    f"csuite reports drift detected: "
                    f"type={drift_status.get('drift_type')}, "
                    f"level={drift_status.get('drift_level')}"
                )

    async def publish_directive(self, directive: Dict[str, Any]) -> bool:
        """
        Publish a directive to CoS.

        Directives are strategic instructions from Nexus to CoS.

        Args:
            directive: The directive to send

        Returns:
            True if published successfully
        """
        if not self._connected:
            logger.warning("Cannot publish directive: not connected")
            return False

        channel = f"{self.config.channel_prefix}:directives"

        try:
            await self._redis.publish(channel, json.dumps({
                "type": "directive",
                "directive": directive,
                "timestamp": _utcnow().isoformat(),
            }))
            self._directives_sent += 1
            logger.info(f"Published directive to CoS: {directive.get('type', 'unknown')}")
            return True
        except Exception as e:
            logger.error(f"Failed to publish directive: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the bridge listener."""
        return {
            "connected": self._connected,
            "listening": self._listening,
            "config": {
                "redis_url": self.config.redis_url,
                "channel_prefix": self.config.channel_prefix,
            },
            "metrics": {
                "guidance_requests_received": self._guidance_requests_received,
                "guidance_responses_sent": self._guidance_responses_sent,
                "outcomes_received": self._outcomes_received,
                "health_updates_received": self._health_updates_received,
                "directives_sent": self._directives_sent,
            },
            "csuite_health": {
                "last_update": (
                    self._last_health_timestamp.isoformat()
                    if self._last_health_timestamp
                    else None
                ),
                "data": self._last_csuite_health,
            },
        }

    def get_csuite_health(self) -> Optional[Dict[str, Any]]:
        """Get the last received health status from csuite."""
        return self._last_csuite_health
