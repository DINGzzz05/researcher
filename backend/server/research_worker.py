"""Run durable multi-agent jobs independently of the API process."""

import asyncio
import logging
import os
from contextlib import suppress

from dotenv import load_dotenv
from psycopg import Error as PostgresError

from .research_events import ResearchEventPublisher, ResearchEvents, open_research_events
from .research_results import ResearchResultStore, ResearchTaskConflict, open_research_result_store

logger = logging.getLogger(__name__)


async def run_job(store: ResearchResultStore, events: ResearchEvents, research_id: str) -> bool:
    from .app import ResearchRequest, _generate_report

    async def execute(checkpointer, payload, resume, attempt_id):
        publisher = ResearchEventPublisher(events, research_id, attempt_id)
        execution = asyncio.current_task()

        async def watch_cancellation():
            while True:
                try:
                    if await store.cancellation_requested(research_id, attempt_id):
                        execution.cancel("user_cancelled")
                        return
                except PostgresError:
                    logger.warning("Could not check cancellation for %s", research_id)
                await asyncio.sleep(1)

        watcher = asyncio.create_task(watch_cancellation())
        try:
            await publisher.send_json({
                "type": "logs", "content": "starting_research",
                "output": "Resuming research" if resume else "Starting research",
            })
            request = ResearchRequest(**payload, repo_name="", branch_name="")
            return await _generate_report(
                request, research_id, checkpointer=checkpointer, resume=resume, event_sink=publisher,
            )
        finally:
            watcher.cancel()
            with suppress(asyncio.CancelledError):
                await watcher

    try:
        await store.run(research_id, execute, worker_managed=True)
    except ResearchTaskConflict:
        return False
    except asyncio.CancelledError as error:
        if error.args != ("user_cancelled",):
            raise
    except Exception:
        logger.exception("Research job %s failed", research_id)
    return True


async def run_worker():
    database_url, redis_url = os.getenv("DATABASE_URL"), os.getenv("REDIS_URL")
    if not database_url or not redis_url:
        raise SystemExit("The research worker requires DATABASE_URL and REDIS_URL.")
    async with (
        open_research_result_store(database_url) as store,
        open_research_events(redis_url) as events,
    ):
        logger.info("Research worker ready")
        while True:
            try:
                for research_id in await store.queued_jobs():
                    if await run_job(store, events, research_id):
                        break
                else:
                    await asyncio.sleep(1)
            except PostgresError:
                logger.exception("Could not poll for research jobs")
                await asyncio.sleep(2)


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    loop_factory = asyncio.SelectorEventLoop if os.name == "nt" else None
    with asyncio.Runner(loop_factory=loop_factory) as runner:
        runner.run(run_worker())
