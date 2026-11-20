import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING
from uuid import uuid4

from anyio import CancelScope
from fastapi.encoders import jsonable_encoder

if TYPE_CHECKING:
    from psycopg_pool import AsyncConnectionPool

logger = logging.getLogger(__name__)

RESUMABLE_STATUSES = ("queued", "failed", "cancelled", "interrupted")
TASK_COLUMNS = """
    research_id, COALESCE(query, result->>'query') AS query, report_type, tone,
    status, error_code, created_at, updated_at, attempt_id, cancel_requested,
    request_payload IS NOT NULL AS worker_managed
"""


class ResearchTaskConflict(Exception):
    """A task cannot be claimed for this execution attempt."""


class ResearchResultStore:
    def __init__(self, pool: "AsyncConnectionPool"):
        self._pool = pool
        # Reserve a connection for task registration and status queries.
        self._run_slots = asyncio.Semaphore(pool.max_size - 1)

    async def register(
        self, research_id: str, query: str, report_type: str, tone: str, *,
        request_payload: dict | None = None,
    ) -> None:
        async with self._pool.connection() as connection:
            await connection.execute(
                """
                INSERT INTO research_results
                    (research_id, query, report_type, tone, status, request_payload)
                VALUES (%s, %s, %s, %s, 'queued', %s::jsonb)
                """,
                (research_id, query, report_type, tone,
                 json.dumps(request_payload) if request_payload is not None else None),
            )

    async def queued_jobs(self, limit: int = 16) -> list[str]:
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                """
                SELECT research_id FROM research_results
                WHERE status = 'queued' AND request_payload IS NOT NULL
                ORDER BY updated_at, research_id LIMIT %s
                """,
                (limit,),
            )
            return [row["research_id"] for row in await cursor.fetchall()]

    async def enqueue_resume(self, research_id: str, attempt_id: str | None) -> bool:
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                """
                UPDATE research_results SET status = 'queued', resume_requested = true,
                    cancel_requested = false, error_code = NULL, attempt_id = NULL,
                    updated_at = now()
                WHERE research_id = %s AND request_payload IS NOT NULL
                    AND attempt_id IS NOT DISTINCT FROM %s
                    AND status IN ('failed', 'cancelled', 'interrupted')
                """,
                (research_id, attempt_id),
            )
            return cursor.rowcount == 1

    async def request_cancel(self, research_id: str) -> str | None:
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                """
                UPDATE research_results SET cancel_requested = true,
                    status = CASE WHEN status = 'queued' THEN 'cancelled' ELSE status END,
                    error_code = CASE WHEN status = 'queued' THEN 'user_cancelled' ELSE error_code END,
                    updated_at = now()
                WHERE research_id = %s AND request_payload IS NOT NULL
                    AND status IN ('queued', 'running')
                RETURNING status
                """,
                (research_id,),
            )
            row = await cursor.fetchone()
            return row["status"] if row else None

    async def cancellation_requested(self, research_id: str, attempt_id: str) -> bool:
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                """
                SELECT cancel_requested FROM research_results
                WHERE research_id = %s AND attempt_id = %s AND status = 'running'
                """,
                (research_id, attempt_id),
            )
            row = await cursor.fetchone()
            return bool(row and row["cancel_requested"])

    async def get(self, research_id: str) -> dict | None:
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                "SELECT result FROM research_results WHERE research_id = %s AND status = 'succeeded'",
                (research_id,),
            )
            row = await cursor.fetchone()
            return row["result"] if row else None

    async def get_task(self, research_id: str) -> dict | None:
        async with self._pool.connection() as connection, connection.transaction():
            select_task = f"SELECT {TASK_COLUMNS} FROM research_results WHERE research_id = %s"
            cursor = await connection.execute(select_task, (research_id,))
            task = await cursor.fetchone()
            if task is None or task["status"] != "running":
                return task

            cursor = await connection.execute(
                "SELECT pg_try_advisory_xact_lock(hashtextextended(%s, 0)) AS acquired",
                (research_id,),
            )
            if not (await cursor.fetchone())["acquired"]:
                return task

            # Recheck the status after acquiring the lock in case the worker just finished.
            await connection.execute(
                """
                UPDATE research_results SET
                    status = CASE WHEN cancel_requested THEN 'cancelled' ELSE 'interrupted' END,
                    error_code = CASE WHEN cancel_requested THEN 'user_cancelled' ELSE 'worker_lost' END,
                    updated_at = now()
                WHERE research_id = %s AND status = 'running'
                """,
                (research_id,),
            )
            cursor = await connection.execute(select_task, (research_id,))
            return await cursor.fetchone()

    async def run(
        self, research_id: str, execute, *, resume: bool = False, worker_managed: bool = False,
    ) -> dict:
        attempt_id = uuid4().hex
        try:
            return await self._run_attempt(research_id, attempt_id, execute, resume, worker_managed)
        except asyncio.CancelledError as error:
            status, reason = "cancelled", "execution_cancelled"
            if error.args == ("websocket_closed",):
                reason = "websocket_closed"
            elif worker_managed:
                status, reason = "interrupted", "worker_stopped"
            with CancelScope(shield=True):
                await self._record_error(
                    research_id, attempt_id, status, reason,
                    include_queued=not resume and not worker_managed,
                )
            raise
        except ResearchTaskConflict:
            raise
        except Exception as error:
            await self._record_error(research_id, attempt_id, "failed", type(error).__name__)
            raise

    async def _run_attempt(
        self, research_id: str, attempt_id: str, execute, resume: bool, worker_managed: bool,
    ) -> dict:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        allowed = list(RESUMABLE_STATUSES if resume else ("queued",))
        async with self._run_slots, self._pool.connection() as connection:
            # The graph and the task row use the connection that owns the lock.
            # A worker that loses the connection cannot keep writing checkpoints.
            try:
                cursor = await connection.execute(
                    "SELECT pg_try_advisory_lock(hashtextextended(%s, 0)) AS acquired",
                    (research_id,),
                )
                if not (await cursor.fetchone())["acquired"]:
                    raise ResearchTaskConflict("Research task is already running.")

                cursor = await connection.execute(
                    """
                    UPDATE research_results SET status = 'running', attempt_id = %s,
                        error_code = NULL, updated_at = now()
                    WHERE research_id = %s AND status = ANY(%s)
                        AND (NOT %s OR report_type = 'multi_agents')
                        AND (request_payload IS NOT NULL) = %s
                    RETURNING report_type, query, request_payload, resume_requested
                    """,
                    (attempt_id, research_id, allowed, resume, worker_managed),
                )
                task = await cursor.fetchone()
                if task is None:
                    raise ResearchTaskConflict("Research task cannot be started in its current state.")

                checkpointer = (
                    AsyncPostgresSaver(connection) if task["report_type"] == "multi_agents" else None
                )
                if worker_managed:
                    result = await execute(
                        checkpointer, task["request_payload"], task["resume_requested"], attempt_id
                    )
                else:
                    result = await execute(checkpointer)
                await connection.execute(
                    """
                    UPDATE research_results SET
                        status = CASE WHEN cancel_requested THEN 'cancelled' ELSE 'succeeded' END,
                        result = CASE WHEN cancel_requested THEN NULL ELSE %s::jsonb END,
                        error_code = CASE WHEN cancel_requested THEN 'user_cancelled' ELSE NULL END,
                        updated_at = now()
                    WHERE research_id = %s AND attempt_id = %s AND status = 'running'
                    """,
                    (json.dumps(jsonable_encoder({**result, "query": task["query"]})), research_id, attempt_id),
                )
                return result
            finally:
                # Pooled connections must give up session locks before reuse.
                with CancelScope(shield=True):
                    if not connection.closed:
                        try:
                            await connection.execute("SELECT pg_advisory_unlock_all()")
                        except asyncio.CancelledError:
                            await connection.close()
                            raise
                        except Exception:
                            await connection.close()
                            logger.exception("Could not release lock for research task %s", research_id)

    async def _record_error(
        self, research_id: str, attempt_id: str, status: str, error_code: str,
        *, include_queued: bool = False,
    ) -> None:
        try:
            async with self._pool.connection() as connection:
                await connection.execute(
                    """
                    UPDATE research_results SET
                        status = CASE WHEN cancel_requested THEN 'cancelled' ELSE %s END,
                        error_code = CASE WHEN cancel_requested THEN 'user_cancelled' ELSE %s END,
                        updated_at = now()
                    WHERE research_id = %s AND (
                        (attempt_id = %s AND status IN ('running', 'interrupted'))
                        OR (%s AND status = 'queued' AND request_payload IS NULL)
                    )
                    """,
                    (status, error_code, research_id, attempt_id, include_queued),
                )
        except Exception:
            logger.exception("Could not persist terminal state for research task %s", research_id)


@asynccontextmanager
async def open_research_result_store(database_url: str | None):
    if not database_url:
        yield None
        return

    if isinstance(asyncio.get_running_loop(), getattr(asyncio, "ProactorEventLoop", ())):
        raise RuntimeError("On Windows, start the PostgreSQL-enabled backend with 'python main.py'.")

    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg.errors import UndefinedColumn, UndefinedTable
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    async with AsyncConnectionPool(
        database_url, min_size=1, max_size=5, open=False,
        kwargs={"autocommit": True, "row_factory": dict_row, "prepare_threshold": 0},
    ) as pool:
        async with pool.connection() as connection:
            try:
                await connection.execute("SELECT request_payload FROM research_results LIMIT 0")
            except (UndefinedTable, UndefinedColumn) as error:
                raise RuntimeError("Run 'alembic upgrade head' before starting the backend.") from error

            # Serialise the checkpointer's schema setup across application workers.
            await connection.execute("SELECT pg_advisory_lock(42512, 1)")
            try:
                await AsyncPostgresSaver(connection).setup()
            finally:
                await connection.execute("SELECT pg_advisory_unlock(42512, 1)")
        yield ResearchResultStore(pool)
