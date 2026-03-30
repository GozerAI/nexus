"""
Offline Agent Communication (Item 788)

Provides a local message bus for inter-agent communication that works
entirely offline. Uses in-process queues with optional file-system persistence
for durability across restarts.
"""

import logging
import time
import json
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class MessagePriority(str, Enum):
    """Message priority levels."""

    URGENT = "urgent"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class DeliveryStatus(str, Enum):
    """Message delivery status."""

    QUEUED = "queued"
    DELIVERED = "delivered"
    ACKNOWLEDGED = "acknowledged"
    EXPIRED = "expired"
    FAILED = "failed"


@dataclass
class AgentMessage:
    """A message between agents."""

    message_id: str
    sender: str
    recipient: str  # agent name or "*" for broadcast
    topic: str
    payload: Dict[str, Any]
    priority: MessagePriority = MessagePriority.NORMAL
    ttl_seconds: float = 300.0
    status: DeliveryStatus = DeliveryStatus.QUEUED
    created_at: float = field(default_factory=time.time)
    delivered_at: Optional[float] = None
    reply_to: Optional[str] = None  # message_id this replies to

    @property
    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl_seconds

    @property
    def is_broadcast(self) -> bool:
        return self.recipient == "*"


@dataclass
class Subscription:
    """A topic subscription by an agent."""

    agent_name: str
    topic: str
    handler: Optional[Callable] = None
    created_at: float = field(default_factory=time.time)


class OfflineAgentCommunication:
    """
    Local message bus for offline inter-agent communication.

    Features:
    - Point-to-point messaging between named agents
    - Broadcast messages to all agents
    - Topic-based pub/sub subscriptions
    - Priority-based message ordering
    - TTL-based message expiration
    - Optional file-system persistence
    - Request/reply pattern support
    """

    PRIORITY_ORDER = {
        MessagePriority.URGENT: 0,
        MessagePriority.HIGH: 1,
        MessagePriority.NORMAL: 2,
        MessagePriority.LOW: 3,
    }

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        max_queue_per_agent: int = 1000,
        max_history: int = 5000,
    ):
        self._queues: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_queue_per_agent)
        )
        self._subscriptions: Dict[str, List[Subscription]] = defaultdict(list)
        self._registered_agents: Set[str] = set()
        self._message_history: deque = deque(maxlen=max_history)
        self._pending_replies: Dict[str, AgentMessage] = {}
        self._lock = threading.RLock()
        self._persist_dir = Path(persist_dir) if persist_dir else None
        self._max_queue = max_queue_per_agent
        self._msg_counter = 0

        if self._persist_dir:
            self._persist_dir.mkdir(parents=True, exist_ok=True)

    # ── Agent Registration ──────────────────────────────────────────

    def register_agent(self, agent_name: str) -> None:
        """Register an agent with the communication bus."""
        with self._lock:
            self._registered_agents.add(agent_name)
            if agent_name not in self._queues:
                self._queues[agent_name] = deque(maxlen=self._max_queue)
        logger.info("Agent registered: %s", agent_name)

    def unregister_agent(self, agent_name: str) -> None:
        """Unregister an agent."""
        with self._lock:
            self._registered_agents.discard(agent_name)
            # Keep queue for potential message recovery

    def is_registered(self, agent_name: str) -> bool:
        return agent_name in self._registered_agents

    # ── Messaging ───────────────────────────────────────────────────

    def send(
        self,
        sender: str,
        recipient: str,
        topic: str,
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        ttl_seconds: float = 300.0,
        reply_to: Optional[str] = None,
    ) -> AgentMessage:
        """Send a message to a specific agent or broadcast ('*')."""
        with self._lock:
            self._msg_counter += 1
            msg_id = f"msg_{int(time.time())}_{self._msg_counter}"

        msg = AgentMessage(
            message_id=msg_id,
            sender=sender,
            recipient=recipient,
            topic=topic,
            payload=payload,
            priority=priority,
            ttl_seconds=ttl_seconds,
            reply_to=reply_to,
        )

        with self._lock:
            if msg.is_broadcast:
                for agent_name in self._registered_agents:
                    if agent_name != sender:
                        self._queues[agent_name].append(msg)
            else:
                self._queues[recipient].append(msg)

            # Notify topic subscribers
            self._notify_subscribers(msg)

            self._message_history.append(msg)

        if self._persist_dir:
            self._persist_message(msg)

        logger.debug(
            "Message %s: %s -> %s [%s] (%s)",
            msg_id,
            sender,
            recipient,
            topic,
            priority.value,
        )
        return msg

    def receive(
        self,
        agent_name: str,
        topic: Optional[str] = None,
        limit: int = 10,
    ) -> List[AgentMessage]:
        """
        Receive messages for an agent, optionally filtered by topic.
        Marks received messages as delivered.
        """
        with self._lock:
            queue = self._queues.get(agent_name, deque())
            messages: List[AgentMessage] = []
            remaining: List[AgentMessage] = []

            for msg in queue:
                if msg.is_expired:
                    msg.status = DeliveryStatus.EXPIRED
                    continue

                if topic and msg.topic != topic:
                    remaining.append(msg)
                    continue

                if len(messages) < limit:
                    msg.status = DeliveryStatus.DELIVERED
                    msg.delivered_at = time.time()
                    messages.append(msg)
                else:
                    remaining.append(msg)

            self._queues[agent_name] = deque(remaining, maxlen=self._max_queue)

        # Sort by priority then timestamp
        messages.sort(
            key=lambda m: (
                self.PRIORITY_ORDER.get(m.priority, 99),
                m.created_at,
            )
        )
        return messages

    def acknowledge(self, message_id: str) -> bool:
        """Acknowledge receipt of a message."""
        with self._lock:
            for msg in self._message_history:
                if msg.message_id == message_id:
                    msg.status = DeliveryStatus.ACKNOWLEDGED
                    return True
        return False

    # ── Request/Reply ───────────────────────────────────────────────

    def send_request(
        self,
        sender: str,
        recipient: str,
        topic: str,
        payload: Dict[str, Any],
        ttl_seconds: float = 60.0,
    ) -> AgentMessage:
        """Send a message expecting a reply."""
        msg = self.send(
            sender=sender,
            recipient=recipient,
            topic=topic,
            payload=payload,
            ttl_seconds=ttl_seconds,
        )
        with self._lock:
            self._pending_replies[msg.message_id] = msg
        return msg

    def send_reply(
        self,
        sender: str,
        original_message_id: str,
        payload: Dict[str, Any],
    ) -> Optional[AgentMessage]:
        """Send a reply to a previous message."""
        with self._lock:
            original = self._pending_replies.get(original_message_id)
            if not original:
                # Search history
                for msg in self._message_history:
                    if msg.message_id == original_message_id:
                        original = msg
                        break

        if not original:
            return None

        return self.send(
            sender=sender,
            recipient=original.sender,
            topic=f"reply:{original.topic}",
            payload=payload,
            reply_to=original_message_id,
        )

    def get_reply(
        self, original_message_id: str, agent_name: str
    ) -> Optional[AgentMessage]:
        """Check for a reply to a specific message."""
        with self._lock:
            queue = self._queues.get(agent_name, deque())
            for msg in queue:
                if msg.reply_to == original_message_id:
                    msg.status = DeliveryStatus.DELIVERED
                    msg.delivered_at = time.time()
                    return msg
        return None

    # ── Subscriptions ───────────────────────────────────────────────

    def subscribe(
        self,
        agent_name: str,
        topic: str,
        handler: Optional[Callable] = None,
    ) -> Subscription:
        """Subscribe to a topic."""
        sub = Subscription(agent_name=agent_name, topic=topic, handler=handler)
        with self._lock:
            self._subscriptions[topic].append(sub)
        logger.debug("Agent %s subscribed to topic: %s", agent_name, topic)
        return sub

    def unsubscribe(self, agent_name: str, topic: str) -> bool:
        """Unsubscribe from a topic."""
        with self._lock:
            subs = self._subscriptions.get(topic, [])
            original_len = len(subs)
            self._subscriptions[topic] = [
                s for s in subs if s.agent_name != agent_name
            ]
            return len(self._subscriptions[topic]) < original_len

    def _notify_subscribers(self, msg: AgentMessage) -> None:
        """Notify topic subscribers of a new message."""
        subs = self._subscriptions.get(msg.topic, [])
        for sub in subs:
            if sub.agent_name != msg.sender:
                if sub.handler:
                    try:
                        sub.handler(msg)
                    except Exception as e:
                        logger.warning(
                            "Subscription handler error for %s: %s",
                            sub.agent_name,
                            e,
                        )

    # ── Persistence ─────────────────────────────────────────────────

    def _persist_message(self, msg: AgentMessage) -> None:
        try:
            path = self._persist_dir / f"{msg.message_id}.json"
            data = {
                "message_id": msg.message_id,
                "sender": msg.sender,
                "recipient": msg.recipient,
                "topic": msg.topic,
                "payload": msg.payload,
                "priority": msg.priority.value,
                "ttl_seconds": msg.ttl_seconds,
                "created_at": msg.created_at,
                "reply_to": msg.reply_to,
            }
            path.write_text(json.dumps(data, default=str))
        except Exception as e:
            logger.debug("Failed to persist message: %s", e)

    # ── Reporting ───────────────────────────────────────────────────

    def get_queue_stats(self) -> Dict[str, Any]:
        """Get statistics about the communication bus."""
        with self._lock:
            queue_sizes = {
                agent: len(q) for agent, q in self._queues.items()
            }
            total_messages = len(self._message_history)

            status_counts: Dict[str, int] = defaultdict(int)
            for msg in self._message_history:
                status_counts[msg.status.value] += 1

        return {
            "registered_agents": list(self._registered_agents),
            "queue_sizes": queue_sizes,
            "total_messages": total_messages,
            "status_counts": dict(status_counts),
            "active_subscriptions": {
                topic: len(subs)
                for topic, subs in self._subscriptions.items()
            },
            "pending_replies": len(self._pending_replies),
        }
