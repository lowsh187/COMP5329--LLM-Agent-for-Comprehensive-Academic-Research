import asyncio
import time

COOLDOWN_SECONDS = 30
MIN_REQUEST_INTERVAL_SECONDS = 1.2
_cooldowns: dict[str, float] = {}
_last_requests: dict[str, float] = {}
_locks: dict[str, asyncio.Lock] = {}


def cooldown_remaining(source: str) -> int:
    remaining = _cooldowns.get(source, 0) - time.monotonic()
    return max(0, int(remaining))


def is_in_cooldown(source: str) -> bool:
    return cooldown_remaining(source) > 0


def start_cooldown(source: str) -> None:
    _cooldowns[source] = time.monotonic() + COOLDOWN_SECONDS


async def wait_for_request_slot(source: str) -> None:
    lock = _locks.setdefault(source, asyncio.Lock())
    async with lock:
        now = time.monotonic()
        wait_seconds = _last_requests.get(source, 0) + MIN_REQUEST_INTERVAL_SECONDS - now
        if wait_seconds > 0:
            print(f"[{source}] waiting {wait_seconds:.1f}s for request rate limit", flush=True)
            await asyncio.sleep(wait_seconds)
        _last_requests[source] = time.monotonic()
