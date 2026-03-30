"""
Async response with webhook callback.

For long-running operations (inference, pipeline execution, batch processing),
the API returns immediately with a job ID and delivers the result via webhook
when the operation completes.

Usage::

    # Client submits request
    POST /api/v1/inference
    {
        "prompt": "...",
        "webhook_url": "https://client.example.com/hooks/nexus",
        "webhook_secret": "hmac-secret-123"
    }

    # Server responds immediately
    202 Accepted
    {"job_id": "abc-123", "status": "queued", "estimated_seconds": 30}

    # When done, server POSTs to webhook_url with HMAC signature
"""

import asyncio
import hashlib
import hmac
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


@dataclass
class WebhookJob:
    """A tracked async job with webhook delivery."""
    job_id: str
    webhook_url: str
    webhook_secret: Optional[str] = None
    status: JobStatus = JobStatus.QUEUED
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Any = None
    error: Optional[str] = None
    retries: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)
    task: Optional[asyncio.Task] = field(default=None, repr=False, compare=False)
    coroutine: Optional[Coroutine] = field(default=None, repr=False, compare=False)
    coroutine_started: bool = field(default=False, repr=False, compare=False)

    @property
    def elapsed_seconds(self) -> float:
        end = self.completed_at or time.time()
        start = self.started_at or self.created_at
        return end - start

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "elapsed_seconds": self.elapsed_seconds,
            "retries": self.retries,
            "metadata": self.metadata,
        }


@dataclass
class WebhookDelivery:
    """Record of a webhook delivery attempt."""
    job_id: str
    url: str
    attempt: int
    status_code: Optional[int] = None
    sent_at: float = field(default_factory=time.time)
    response_ms: float = 0.0
    success: bool = False
    error: Optional[str] = None


class AsyncWebhookDispatcher:
    """
    Manages async job execution and webhook result delivery.

    Features:
    - Accepts long-running coroutines and tracks their lifecycle
    - Delivers results via webhook POST with HMAC-SHA256 signature
    - Automatic retry with exponential backoff on delivery failure
    - Job status polling via job_id
    - Configurable concurrency limits
    """

    DEFAULT_TIMEOUT = 300  # 5 minutes
    MAX_CONCURRENT = 50
    DELIVERY_TIMEOUT = 30  # seconds

    def __init__(
        self,
        http_post: Optional[Callable[..., Coroutine]] = None,
        max_concurrent: int = MAX_CONCURRENT,
        default_timeout: float = DEFAULT_TIMEOUT,
    ):
        """
        Args:
            http_post: Async callable for HTTP POST (for DI/testing).
                       Signature: ``async def post(url, data, headers, timeout) -> (status_code, body)``
            max_concurrent: Max concurrent jobs
            default_timeout: Default job timeout in seconds
        """
        self._http_post = http_post or self._default_http_post
        self._max_concurrent = max_concurrent
        self._default_timeout = default_timeout
        self._jobs: Dict[str, WebhookJob] = {}
        self._deliveries: List[WebhookDelivery] = []
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._stats = {
            "jobs_submitted": 0,
            "jobs_completed": 0,
            "jobs_failed": 0,
            "webhooks_delivered": 0,
            "webhooks_failed": 0,
        }

    @staticmethod
    async def _default_http_post(
        url: str, data: bytes, headers: Dict[str, str], timeout: float
    ) -> tuple:
        """Default HTTP POST using urllib (no external deps)."""
        import urllib.request
        import urllib.error

        req = urllib.request.Request(
            url, data=data, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()
        except Exception as e:
            raise ConnectionError(f"Webhook delivery failed: {e}") from e

    def submit(
        self,
        coroutine: Coroutine,
        webhook_url: str,
        webhook_secret: Optional[str] = None,
        timeout: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> WebhookJob:
        """
        Submit an async job for execution with webhook callback.

        Args:
            coroutine: The async operation to execute
            webhook_url: URL to POST results to
            webhook_secret: HMAC secret for signing payloads
            timeout: Job timeout in seconds
            metadata: Additional metadata to include in webhook payload

        Returns:
            WebhookJob with job_id for status polling
        """
        job_id = str(uuid.uuid4())
        job = WebhookJob(
            job_id=job_id,
            webhook_url=webhook_url,
            webhook_secret=webhook_secret,
            metadata=metadata or {},
            coroutine=coroutine,
        )
        self._jobs[job_id] = job
        self._stats["jobs_submitted"] += 1

        # Schedule execution
        effective_timeout = timeout or self._default_timeout
        job.task = asyncio.ensure_future(
            self._execute(job, coroutine, effective_timeout)
        )

        logger.info("Job %s submitted, webhook=%s", job_id, webhook_url)
        return job

    async def _execute(
        self, job: WebhookJob, coroutine: Coroutine, timeout: float
    ) -> None:
        """Execute job and deliver result via webhook."""
        def _close_coroutine() -> None:
            coroutine_obj = job.coroutine or coroutine
            close = getattr(coroutine_obj, "close", None)
            if callable(close):
                close()

        if job.status == JobStatus.CANCELLED:
            _close_coroutine()
            return

        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._max_concurrent)

        try:
            async with self._semaphore:
                if job.status == JobStatus.CANCELLED:
                    _close_coroutine()
                    return

                job.status = JobStatus.RUNNING
                job.started_at = time.time()

                try:
                    job.coroutine_started = True
                    job.result = await asyncio.wait_for(coroutine, timeout=timeout)
                    if job.status == JobStatus.CANCELLED:
                        return
                    job.status = JobStatus.COMPLETED
                    job.completed_at = time.time()
                    self._stats["jobs_completed"] += 1
                    logger.info(
                        "Job %s completed in %.1fs", job.job_id, job.elapsed_seconds
                    )
                except asyncio.TimeoutError:
                    job.status = JobStatus.FAILED
                    job.error = f"Job timed out after {timeout}s"
                    job.completed_at = time.time()
                    self._stats["jobs_failed"] += 1
                    logger.warning("Job %s timed out", job.job_id)
                except Exception as e:
                    job.status = JobStatus.FAILED
                    job.error = str(e)
                    job.completed_at = time.time()
                    self._stats["jobs_failed"] += 1
                    logger.error("Job %s failed: %s", job.job_id, e)

                if job.status != JobStatus.CANCELLED:
                    await self._deliver_webhook(job)
        except asyncio.CancelledError:
            if not job.coroutine_started:
                _close_coroutine()
            if job.status != JobStatus.CANCELLED:
                job.status = JobStatus.CANCELLED
                job.completed_at = time.time()
            logger.info("Job %s cancelled", job.job_id)
        finally:
            job.coroutine = None
            if job.task is not None and job.task.done():
                job.task = None

    async def _deliver_webhook(self, job: WebhookJob) -> bool:
        """Deliver job result via webhook with retries."""
        payload = {
            "event": "job.completed" if job.status == JobStatus.COMPLETED else "job.failed",
            "job": job.to_dict(),
            "result": job.result if job.status == JobStatus.COMPLETED else None,
            "error": job.error,
            "timestamp": time.time(),
        }
        body = json.dumps(payload, default=str).encode("utf-8")

        headers = {"Content-Type": "application/json"}
        if job.webhook_secret:
            sig = hmac.new(
                job.webhook_secret.encode(), body, hashlib.sha256
            ).hexdigest()
            headers["X-Nexus-Signature"] = f"sha256={sig}"

        max_attempts = job.max_retries + 1
        for attempt in range(1, max_attempts + 1):
            delivery = WebhookDelivery(
                job_id=job.job_id,
                url=job.webhook_url,
                attempt=attempt,
            )
            try:
                start = time.time()
                status_code, _ = await self._http_post(
                    job.webhook_url, body, headers, self.DELIVERY_TIMEOUT
                )
                delivery.response_ms = (time.time() - start) * 1000
                delivery.status_code = status_code
                delivery.success = 200 <= status_code < 300

                self._deliveries.append(delivery)

                if delivery.success:
                    self._stats["webhooks_delivered"] += 1
                    logger.info(
                        "Webhook delivered for job %s (attempt %d)",
                        job.job_id, attempt,
                    )
                    return True
                else:
                    logger.warning(
                        "Webhook delivery returned %d for job %s (attempt %d)",
                        status_code, job.job_id, attempt,
                    )
            except Exception as e:
                delivery.error = str(e)
                self._deliveries.append(delivery)
                logger.warning(
                    "Webhook delivery failed for job %s (attempt %d): %s",
                    job.job_id, attempt, e,
                )

            # Exponential backoff between retries
            if attempt < max_attempts:
                backoff = min(2 ** attempt, 30)
                await asyncio.sleep(backoff)

        self._stats["webhooks_failed"] += 1
        return False

    def get_job(self, job_id: str) -> Optional[WebhookJob]:
        """Get job status by ID."""
        return self._jobs.get(job_id)

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a queued or running job."""
        job = self._jobs.get(job_id)
        if not job:
            return False
        if job.status in (JobStatus.QUEUED, JobStatus.RUNNING):
            job.status = JobStatus.CANCELLED
            job.completed_at = time.time()
            if not job.coroutine_started and job.coroutine is not None:
                close = getattr(job.coroutine, "close", None)
                if callable(close):
                    close()
                job.coroutine = None
            if job.task is not None and not job.task.done():
                job.task.cancel()
            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        return {
            **self._stats,
            "active_jobs": sum(
                1 for j in self._jobs.values()
                if j.status in (JobStatus.QUEUED, JobStatus.RUNNING)
            ),
            "total_jobs": len(self._jobs),
        }

    def cleanup(self, max_age_seconds: float = 3600) -> int:
        """Remove completed jobs older than max_age_seconds."""
        cutoff = time.time() - max_age_seconds
        to_remove = [
            jid for jid, job in self._jobs.items()
            if job.completed_at and job.completed_at < cutoff
        ]
        for jid in to_remove:
            del self._jobs[jid]
        return len(to_remove)
