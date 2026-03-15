"""
Async response with webhook callback.

For long-running operations, immediately returns a job ID and
delivers the result to a registered webhook URL when complete.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


@dataclass
class WebhookCallback:
    """Configuration for a webhook callback."""
    url: str
    headers: Dict[str, str] = field(default_factory=dict)
    secret: Optional[str] = None
    retry_count: int = 3
    retry_delay_seconds: float = 5.0
    timeout_seconds: float = 30.0

    def sign_payload(self, payload: bytes) -> str:
        if not self.secret:
            return ""
        return hmac.new(self.secret.encode(), payload, hashlib.sha256).hexdigest()


@dataclass
class AsyncJob:
    """Represents an async job with webhook delivery."""
    job_id: str
    status: JobStatus = JobStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    webhook: Optional[WebhookCallback] = None
    progress: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_status_dict(self) -> dict:
        d = {
            "job_id": self.job_id,
            "status": self.status.value,
            "created_at": datetime.fromtimestamp(self.created_at, tz=timezone.utc).isoformat(),
            "progress": self.progress,
        }
        if self.started_at:
            d["started_at"] = datetime.fromtimestamp(self.started_at, tz=timezone.utc).isoformat()
        if self.completed_at:
            d["completed_at"] = datetime.fromtimestamp(self.completed_at, tz=timezone.utc).isoformat()
            d["duration_seconds"] = round(self.completed_at - (self.started_at or self.created_at), 3)
        if self.error:
            d["error"] = self.error
        return d


class AsyncWebhookDispatcher:
    """Manages async job execution with webhook result delivery."""

    def __init__(self, max_concurrent=50, job_ttl_seconds=86400, default_timeout_seconds=600.0):
        self._jobs: Dict[str, AsyncJob] = {}
        self._tasks: Dict[str, asyncio.Task] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._job_ttl = job_ttl_seconds
        self._default_timeout = default_timeout_seconds
        self._stats = {
            "submitted": 0, "completed": 0, "failed": 0,
            "webhook_deliveries": 0, "webhook_failures": 0,
        }

    async def submit(self, coroutine_fn, args=(), kwargs=None, webhook=None,
                     timeout_seconds=None, metadata=None):
        """Submit an async job and optionally register a webhook callback."""
        job_id = uuid.uuid4().hex[:16]
        job = AsyncJob(job_id=job_id, webhook=webhook, metadata=metadata or {})
        self._jobs[job_id] = job
        self._stats["submitted"] += 1
        effective_timeout = timeout_seconds or self._default_timeout
        task = asyncio.create_task(
            self._run_job(job, coroutine_fn, args, kwargs or {}, effective_timeout)
        )
        self._tasks[job_id] = task
        return job

    async def _run_job(self, job, coroutine_fn, args, kwargs, timeout):
        async with self._semaphore:
            job.status = JobStatus.RUNNING
            job.started_at = time.time()
            try:
                result = await asyncio.wait_for(coroutine_fn(*args, **kwargs), timeout=timeout)
                job.result = result
                job.status = JobStatus.COMPLETED
                job.progress = 1.0
                self._stats["completed"] += 1
            except asyncio.TimeoutError:
                job.status = JobStatus.TIMED_OUT
                job.error = f"Job timed out after {timeout}s"
                self._stats["failed"] += 1
            except Exception as exc:
                job.status = JobStatus.FAILED
                job.error = str(exc)
                self._stats["failed"] += 1
            finally:
                job.completed_at = time.time()
            if job.webhook:
                await self._deliver_webhook(job)

    async def _deliver_webhook(self, job):
        webhook = job.webhook
        if not webhook:
            return False
        payload_dict = {
            "event": "job.completed" if job.status == JobStatus.COMPLETED else "job.failed",
            "job": job.to_status_dict(),
        }
        if job.status == JobStatus.COMPLETED and job.result is not None:
            payload_dict["result"] = job.result
        try:
            payload_bytes = json.dumps(payload_dict, default=str).encode()
        except (TypeError, ValueError):
            payload_bytes = json.dumps({"job": job.to_status_dict()}).encode()
        headers = {"Content-Type": "application/json", "X-Job-Id": job.job_id}
        headers.update(webhook.headers)
        if webhook.secret:
            headers["X-Webhook-Signature"] = webhook.sign_payload(payload_bytes)
        for attempt in range(webhook.retry_count):
            try:
                import httpx
                async with httpx.AsyncClient(timeout=webhook.timeout_seconds) as client:
                    resp = await client.post(webhook.url, content=payload_bytes, headers=headers)
                    if 200 <= resp.status_code < 300:
                        self._stats["webhook_deliveries"] += 1
                        return True
            except ImportError:
                pass
            except Exception:
                pass
            if attempt < webhook.retry_count - 1:
                await asyncio.sleep(webhook.retry_delay_seconds * (attempt + 1))
        self._stats["webhook_failures"] += 1
        return False

    def get_status(self, job_id):
        """Get job status by ID."""
        job = self._jobs.get(job_id)
        return job.to_status_dict() if job else None

    def get_result(self, job_id):
        """Get job result if completed."""
        job = self._jobs.get(job_id)
        return job.result if job and job.status == JobStatus.COMPLETED else None

    async def cancel(self, job_id):
        """Cancel a running job."""
        job = self._jobs.get(job_id)
        task = self._tasks.get(job_id)
        if not job or not task:
            return False
        if job.status in (JobStatus.PENDING, JobStatus.RUNNING):
            task.cancel()
            job.status = JobStatus.CANCELLED
            job.completed_at = time.time()
            return True
        return False

    def cleanup_expired(self):
        """Remove expired job records."""
        now = time.time()
        expired = [jid for jid, job in self._jobs.items()
                   if job.completed_at and (now - job.completed_at) > self._job_ttl]
        for jid in expired:
            del self._jobs[jid]
            self._tasks.pop(jid, None)
        return len(expired)

    def get_stats(self):
        active = sum(1 for j in self._jobs.values() if j.status in (JobStatus.PENDING, JobStatus.RUNNING))
        return {**self._stats, "active_jobs": active, "total_jobs": len(self._jobs)}
