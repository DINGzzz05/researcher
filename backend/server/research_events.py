import asyncio
import json
import logging
import re
from contextlib import asynccontextmanager

from redis.asyncio import Redis
from redis.exceptions import RedisError
from redis.backoff import NoBackoff
from redis.asyncio.retry import Retry

logger = logging.getLogger(__name__)
FIRST_EVENT = "0-0"
ACTIVE_STATUSES = {"queued", "running"}

# A snapshot and its cursor always describe the same point in the stream.
APPEND_EVENT = """
local id = redis.call('XADD', KEYS[1], 'MAXLEN', '~', ARGV[1], '*',
    'kind', ARGV[2], 'data', ARGV[3])
local snapshot = KEYS[2]
if redis.call('HGET', snapshot, 'attempt_id') ~= ARGV[4] then
    redis.call('DEL', snapshot)
end
redis.call('HSET', snapshot, 'attempt_id', ARGV[4], 'event_id', id)
if ARGV[5] ~= '' then redis.call('HSET', snapshot, 'stage', ARGV[5]) end
if ARGV[6] ~= '' then redis.call('HSET', snapshot, 'completed', ARGV[6]) end
if ARGV[7] ~= '' then redis.call('HSET', snapshot, 'total', ARGV[7]) end
redis.call('EXPIRE', KEYS[1], ARGV[8])
redis.call('EXPIRE', snapshot, ARGV[8])
return id
"""


def valid_event_id(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9]{1,20}-[0-9]{1,20}", value)) and all(
        int(part) < 2**64 for part in value.split("-")
    )


class ResearchEvents:
    def __init__(self, redis: Redis, *, max_events: int = 2000, ttl: int = 86400):
        self.redis = redis
        self.max_events = max_events
        self.ttl = ttl

    @staticmethod
    def keys(research_id: str) -> tuple[str, str]:
        prefix = f"research:{{{research_id}}}"
        return f"{prefix}:events", f"{prefix}:progress"

    async def append(self, research_id: str, attempt_id: str, kind: str, data: dict) -> str:
        stage = data.get("stage", "") if kind == "stage_changed" else ""
        counts = (data.get("metadata") or {}) if kind in {"stage_changed", "progress"} else {}
        payload = json.dumps({**data, "attempt_id": attempt_id}, default=str)
        return await self.redis.eval(
            APPEND_EVENT, 2, *self.keys(research_id), self.max_events, kind, payload,
            attempt_id, stage, counts.get("completed", ""), counts.get("total", ""), self.ttl,
        )

    async def snapshot(self, research_id: str) -> dict:
        snapshot = await self.redis.hgetall(self.keys(research_id)[1])
        for field in ("completed", "total"):
            if field in snapshot:
                snapshot[field] = int(snapshot[field])
        return snapshot

    async def contains(self, research_id: str, cursor: str) -> bool:
        key = self.keys(research_id)[0]
        return bool(await self.redis.xrange(key, min=cursor, max=cursor, count=1))

    async def read(self, research_id: str, cursor: str, *, wait: bool) -> list:
        stream = await self.redis.xread(
            {self.keys(research_id)[0]: cursor}, count=64, block=2500 if wait else None,
        )
        return stream[0][1] if stream else []


class ResearchEventPublisher:
    """Accept the agent's existing send_json calls without a client connection."""

    stages = {
        "starting_research", "initial_research", "planning", "parallel_research",
        "writing_report", "publishing",
    }

    def __init__(self, events: ResearchEvents, research_id: str, attempt_id: str):
        self.events = events
        self.research_id = research_id
        self.attempt_id = attempt_id
        self._lock = asyncio.Lock()
        self._retry_at = 0.0

    async def send_json(self, data: dict) -> None:
        step = data.get("content")
        kind = "log"
        if data.get("type") == "report":
            kind = "report_delta"
        elif data.get("type") == "progress":
            kind = "progress"
        elif isinstance(step, str) and step in self.stages:
            kind = "stage_changed"
            data = {**data, "stage": step}
        async with self._lock:
            loop = asyncio.get_running_loop()
            if loop.time() < self._retry_at:
                return
            try:
                async with asyncio.timeout(1):
                    await self.events.append(self.research_id, self.attempt_id, kind, data)
                self._retry_at = 0.0
            except (RedisError, TimeoutError):
                if not self._retry_at:
                    logger.warning("Research progress unavailable for %s", self.research_id)
                self._retry_at = loop.time() + 5


@asynccontextmanager
async def open_research_events(url: str | None):
    if not url:
        yield None
        return

    async with Redis.from_url(
        url, decode_responses=True, protocol=2, socket_connect_timeout=1, socket_timeout=4,
        retry=Retry(NoBackoff(), 0),
    ) as redis:
        yield ResearchEvents(redis)


def sse(kind: str, data: dict, cursor: str | None = None) -> str:
    event_id = "" if cursor is None else f"id: {cursor}\n"
    return f"{event_id}event: {kind}\ndata: {json.dumps(data, default=str)}\n\n"


async def stream_research_events(events: ResearchEvents, store, research_id: str, cursor: str):
    last_status = None
    while True:
        task = await store.get_task(research_id)
        if task is None:
            return
        terminal = task["status"] not in ACTIVE_STATUSES
        try:
            if cursor != FIRST_EVENT and not await events.contains(research_id, cursor):
                snapshot = await events.snapshot(research_id)
                cursor = snapshot.get("event_id", FIRST_EVENT)
                yield sse("reset", {"progress": snapshot}, "" if cursor == FIRST_EVENT else cursor)

            entries = await events.read(research_id, cursor, wait=not terminal)
            if entries and cursor != FIRST_EVENT and not await events.contains(research_id, cursor):
                # Trimming can advance the stream while this reader is blocked.
                continue
            for cursor, fields in entries:
                yield sse(fields["kind"], json.loads(fields["data"]), cursor)
        except RedisError:
            entries = []

        status = (task["status"], task["attempt_id"], task["cancel_requested"])
        if status != last_status and (not terminal or not entries):
            last_status = status
            yield sse("task_status", task)
        if terminal and not entries:
            return
        if not entries:
            yield ": keepalive\n\n"
            await asyncio.sleep(0.5)
