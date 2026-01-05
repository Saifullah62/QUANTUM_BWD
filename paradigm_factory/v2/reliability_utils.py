#!/usr/bin/env python3
"""
Reliability Utilities for Bundle Generation
============================================

Provides retry wrappers and failure sinks for robust pipeline operation.
Prevents single flaky calls from taking down entire jobs.
"""

import time
import random
import json
from pathlib import Path
from typing import Callable, TypeVar, Optional, Any, Dict, List
from functools import wraps
from datetime import datetime

T = TypeVar('T')


def with_retries(
    fn: Callable[[], T],
    *,
    tries: int = 5,
    base_sleep: float = 0.5,
    max_sleep: float = 8.0,
    jitter: float = 0.2,
    on_fail: Optional[Callable[[Exception], T]] = None,
    name: str = "operation"
) -> T:
    """
    Execute a function with exponential backoff retries.

    Args:
        fn: Zero-argument callable to execute
        tries: Maximum number of attempts
        base_sleep: Initial sleep time in seconds
        max_sleep: Maximum sleep time in seconds
        jitter: Random jitter factor (0.0-1.0)
        on_fail: Fallback function called with the last exception if all retries fail
                 If None and all retries fail, raises the last exception
        name: Operation name for logging

    Returns:
        Result of fn() or on_fail(exception)

    Example:
        emb = with_retries(
            lambda: embedder.embed(text),
            on_fail=lambda e: None,
            name="embed"
        )
        if emb is None:
            stats["embed_fail"] += 1
    """
    last_exception = None

    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:
            last_exception = e

            if attempt < tries - 1:
                # Calculate sleep with exponential backoff + jitter
                sleep = min(max_sleep, base_sleep * (2 ** attempt))
                sleep = sleep * (1.0 + random.uniform(-jitter, jitter))
                sleep = max(0.0, sleep)

                print(f"[!] {name} failed (attempt {attempt + 1}/{tries}): {e}")
                print(f"    Retrying in {sleep:.1f}s...")
                time.sleep(sleep)

    # All retries exhausted
    if on_fail is not None:
        return on_fail(last_exception)

    raise last_exception


def retry_decorator(
    tries: int = 3,
    base_sleep: float = 0.5,
    max_sleep: float = 8.0,
    jitter: float = 0.2
):
    """
    Decorator version of with_retries.

    Usage:
        @retry_decorator(tries=3)
        def flaky_operation(x, y):
            return api_call(x, y)
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            return with_retries(
                lambda: fn(*args, **kwargs),
                tries=tries,
                base_sleep=base_sleep,
                max_sleep=max_sleep,
                jitter=jitter,
                name=fn.__name__
            )
        return wrapper
    return decorator


class FailureSink:
    """
    Records failed operations for later retry.

    Instead of failing the whole job, log failures to a JSONL file
    that can be processed in a separate retry run.
    """

    def __init__(self, path: Path, flush_every: int = 100):
        self.path = path
        self.flush_every = flush_every
        self.buffer: List[Dict] = []
        self.total_failures = 0

    def record(
        self,
        item_id: str,
        reason: str,
        item_data: Optional[Dict] = None,
        exception: Optional[Exception] = None
    ):
        """Record a failure for later retry."""
        failure = {
            "item_id": item_id,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        }

        if item_data:
            failure["item_data"] = item_data

        if exception:
            failure["exception"] = str(exception)
            failure["exception_type"] = type(exception).__name__

        self.buffer.append(failure)
        self.total_failures += 1

        if len(self.buffer) >= self.flush_every:
            self.flush()

    def flush(self):
        """Write buffered failures to disk."""
        if not self.buffer:
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            for failure in self.buffer:
                f.write(json.dumps(failure, ensure_ascii=False) + "\n")

        self.buffer.clear()

    def close(self):
        """Flush and close the sink."""
        self.flush()
        if self.total_failures > 0:
            print(f"[!] {self.total_failures} failures recorded to {self.path}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class RateLimiter:
    """
    Simple rate limiter for API calls.

    Prevents hitting rate limits on external services.
    """

    def __init__(self, calls_per_second: float = 10.0):
        self.min_interval = 1.0 / calls_per_second
        self.last_call = 0.0

    def wait(self):
        """Wait if necessary to respect rate limit."""
        now = time.time()
        elapsed = now - self.last_call

        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

        self.last_call = time.time()

    def __call__(self, fn):
        """Decorator usage."""
        @wraps(fn)
        def wrapper(*args, **kwargs):
            self.wait()
            return fn(*args, **kwargs)
        return wrapper


# =============================================================================
# Checkpoint/Resume Utilities
# =============================================================================

class Checkpointer:
    """
    Manages checkpoints for resumable processing.

    Tracks which items have been processed and allows resuming
    from the last checkpoint after a crash.
    """

    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.progress_file = checkpoint_dir / "progress.json"
        self.done_file = checkpoint_dir / "done.txt"

        self._load_state()

    def _load_state(self):
        """Load existing state from disk."""
        self.done_ids = set()
        if self.done_file.exists():
            with self.done_file.open("r") as f:
                for line in f:
                    self.done_ids.add(line.strip())

        self.progress = {}
        if self.progress_file.exists():
            with self.progress_file.open("r") as f:
                self.progress = json.load(f)

    def is_done(self, item_id: str) -> bool:
        """Check if an item has been processed."""
        return item_id in self.done_ids

    def mark_done(self, item_id: str):
        """Mark an item as processed."""
        with self.done_file.open("a") as f:
            f.write(item_id + "\n")
        self.done_ids.add(item_id)

    def save_progress(self, key: str, value: Any):
        """Save arbitrary progress data."""
        self.progress[key] = value
        with self.progress_file.open("w") as f:
            json.dump(self.progress, f, indent=2)

    def get_progress(self, key: str, default: Any = None) -> Any:
        """Get saved progress data."""
        return self.progress.get(key, default)


# =============================================================================
# Example: Reliable Embedding with Fallback
# =============================================================================

def reliable_embed(
    embedder,
    text: str,
    fallback_cache: Optional[Dict[str, Any]] = None,
    failure_sink: Optional[FailureSink] = None,
    item_id: Optional[str] = None
) -> Optional[Any]:
    """
    Embed text with retries and fallback.

    Args:
        embedder: Embedding model with .encode() method
        text: Text to embed
        fallback_cache: Optional cache to check before embedding
        failure_sink: Optional sink to record failures
        item_id: Optional ID for failure tracking

    Returns:
        Embedding or None if all attempts fail
    """
    # Check cache first
    if fallback_cache is not None and text in fallback_cache:
        return fallback_cache[text]

    # Try with retries
    def do_embed():
        return embedder.encode(text, show_progress_bar=False)

    result = with_retries(
        do_embed,
        tries=3,
        on_fail=lambda e: None,
        name=f"embed({text[:30]}...)"
    )

    if result is None and failure_sink is not None:
        failure_sink.record(
            item_id=item_id or text[:50],
            reason="embed_failed",
            item_data={"text": text[:200]}
        )

    # Cache successful result
    if result is not None and fallback_cache is not None:
        fallback_cache[text] = result

    return result
