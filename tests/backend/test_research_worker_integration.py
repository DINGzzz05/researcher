"""Run against disposable services configured by the two GPTR_TEST_* URLs."""

import asyncio
import json
import os
import socket
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TypedDict

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def services():
    database = os.getenv("GPTR_TEST_DATABASE_URL")
    redis = os.getenv("GPTR_TEST_REDIS_URL")
    if not database or not redis:
        pytest.skip("Set GPTR_TEST_DATABASE_URL and GPTR_TEST_REDIS_URL for live service tests")
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"], cwd=ROOT, check=True,
        env={**os.environ, "DATABASE_URL": database},
    )
    return database, redis


def run(coroutine):
    loop_factory = asyncio.SelectorEventLoop if os.name == "nt" else None
    with asyncio.Runner(loop_factory=loop_factory) as runner:
        return runner.run(coroutine)


@asynccontextmanager
async def api(database, redis):
    import httpx
    import uvicorn
    from backend.server.app import app
    from backend.server.research_events import open_research_events
    from backend.server.research_results import open_research_result_store

    async with (
        open_research_result_store(database) as store,
        open_research_events(redis) as events,
    ):
        app.state.research_results = store
        app.state.research_events = events
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            server = uvicorn.Server(uvicorn.Config(app, lifespan="off", log_level="warning"))
            serving = asyncio.create_task(server.serve(sockets=[listener]))
            try:
                async with asyncio.timeout(10):
                    while not server.started:
                        await asyncio.sleep(0.01)
                url = f"http://127.0.0.1:{listener.getsockname()[1]}"
                async with httpx.AsyncClient(base_url=url, timeout=10) as client:
                    yield client, store, events
            finally:
                server.should_exit = True
                await serving
                app.state.research_results = None
                app.state.research_events = None


async def frames(response):
    frame = {}
    async for line in response.aiter_lines():
        if line.startswith(("event:", "id:", "data:")):
            key, value = line.split(":", 1)
            frame[key] = json.loads(value) if key == "data" else value.strip()
        elif not line and frame:
            yield frame
            frame = {}


async def submit(client):
    response = await client.post("/api/research", json={"task": "Fault injection", "tone": "Formal"})
    assert response.status_code == 202
    return response.json()["research_id"]


def test_disconnect_replay_single_worker_and_expired_events(services, monkeypatch):
    from backend.server import app as api_module
    from backend.server.research_worker import run_job

    async def scenario():
        release = asyncio.Event()

        async def generate(request, research_id, *, checkpointer, resume, event_sink):
            assert request.tone == "Formal"
            await event_sink.send_json({
                "type": "progress", "content": "parallel_research", "output": "first",
                "metadata": {"completed": 1, "total": 2},
            })
            await release.wait()
            await event_sink.send_json({"type": "logs", "content": "test", "output": "second"})
            await event_sink.send_json({"type": "logs", "content": "test", "output": "third"})
            return {"report": "The finished report"}

        monkeypatch.setattr(api_module, "_generate_report", generate)
        async with api(*services) as (client, store, events):
            research_id = await submit(client)
            assert research_id in await store.queued_jobs()
            path = f"/api/research/{research_id}"
            job = asyncio.create_task(run_job(store, events, research_id))
            try:
                async with client.stream("GET", path + "/events") as response:
                    assert response.headers["content-type"].startswith("text/event-stream")
                    async for frame in frames(response):
                        if frame["event"] == "progress":
                            cursor = frame["id"]
                            break

                assert not job.done()
                status = (await client.get(path)).json()
                assert status["status"] == "running"
                assert status["progress"]["completed"] == 1
                assert not await run_job(store, events, research_id)
                release.set()
                assert await job

                async with client.stream(
                    "GET", path + "/events?after=0-0", headers={"Last-Event-ID": cursor},
                ) as response:
                    replay = [frame async for frame in frames(response)]
                assert [frame["data"]["output"] for frame in replay[:-1]] == ["second", "third"]
                assert replay[-1]["data"]["status"] == "succeeded"

                for key in events.keys(research_id):
                    await events.redis.expire(key, 0)
                async with client.stream("GET", path + f"/events?after={cursor}") as response:
                    expired = [frame async for frame in frames(response)]
                assert [frame["event"] for frame in expired] == ["reset", "task_status"]
                assert (await client.get(path + "/result")).json()["report"] == "The finished report"
            finally:
                release.set()
                await job

    run(scenario())


class GraphState(TypedDict):
    evidence: str
    report: str


async def checkpointed_report(request, research_id, *, checkpointer, resume, event_sink):
    from langgraph.graph import END, START, StateGraph

    async def research(state):
        assert not resume, "The research node already completed before the crash"
        return {"evidence": "Recovered evidence"}

    async def writer(state):
        if not resume:
            await event_sink.send_json({"type": "logs", "content": "test", "output": "checkpoint saved"})
            await asyncio.Event().wait()
        return {"report": state["evidence"]}

    workflow = StateGraph(GraphState)
    workflow.add_node("research", research)
    workflow.add_node("writer", writer)
    workflow.add_edge(START, "research")
    workflow.add_edge("research", "writer")
    workflow.add_edge("writer", END)
    graph = workflow.compile(checkpointer=checkpointer)
    return await graph.ainvoke(
        None if resume else {}, {"configurable": {"thread_id": research_id}}, durability="sync",
    )


async def child_worker(research_id):
    from backend.server import app as api_module
    from backend.server.research_events import open_research_events
    from backend.server.research_results import open_research_result_store
    from backend.server.research_worker import run_job

    api_module._generate_report = checkpointed_report
    async with (
        open_research_result_store(os.environ["GPTR_TEST_DATABASE_URL"]) as store,
        open_research_events(os.environ["GPTR_TEST_REDIS_URL"]) as events,
    ):
        await run_job(store, events, research_id)


def test_killed_worker_can_resume_from_postgres_checkpoint(services, monkeypatch):
    from backend.server import app as api_module
    from backend.server.research_worker import run_job

    async def scenario():
        async with api(*services) as (client, store, events):
            research_id = await submit(client)
            path = f"/api/research/{research_id}"
            code = (
                "import runpy, sys; test = runpy.run_path(sys.argv[1]); "
                "test['run'](test['child_worker'](sys.argv[2]))"
            )
            child = subprocess.Popen([sys.executable, "-c", code, str(Path(__file__).resolve()), research_id], cwd=ROOT)
            try:
                async with asyncio.timeout(30):
                    while True:
                        entries = await events.read(research_id, "0-0", wait=False)
                        if any(json.loads(fields["data"])["output"] == "checkpoint saved" for _, fields in entries):
                            break
                        assert child.poll() is None
                        await asyncio.sleep(0.1)
            finally:
                if child.poll() is None:
                    child.kill()
                await asyncio.to_thread(child.wait, 10)

            async with asyncio.timeout(10):
                while (await client.get(path)).json()["status"] == "running":
                    await asyncio.sleep(0.05)
            assert (await client.get(path)).json()["status"] == "interrupted"
            assert (await client.post(path + "/resume")).status_code == 202
            assert (await client.post(path + "/resume")).status_code == 409
            monkeypatch.setattr(api_module, "_generate_report", checkpointed_report)
            assert await run_job(store, events, research_id)
            assert (await client.get(path + "/result")).json()["report"] == "Recovered evidence"

    run(scenario())
