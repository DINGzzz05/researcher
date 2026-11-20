import json
import os
from typing import Dict, List, Any, Literal
import time
import logging
import sys
import warnings
from pathlib import Path
from uuid import uuid4

# Suppress Pydantic V2 migration warnings
warnings.filterwarnings("ignore", message="Valid config keys have changed in V2")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, File, UploadFile, BackgroundTasks, HTTPException
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from redis.exceptions import RedisError

# Add the parent directory to sys.path to make sure we can import from server
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from server.websocket_manager import WebSocketManager
from server.server_utils import (
    get_config_dict, sanitize_filename,
    update_environment_variables, handle_file_upload, handle_file_deletion,
    execute_multi_agents, handle_websocket_communication
)
from server.agent_discovery import build_agent_discovery_document

from server.websocket_manager import run_agent
from utils import write_md_to_word, write_md_to_pdf
from gpt_researcher.actions import stream_output
from gpt_researcher.retrievers.errors import InsufficientSearchEvidence
from gpt_researcher.utils.enum import ReportSource, Tone
from chat.chat import ChatAgentWithMemory

from server.report_store import ReportStore
from .research_results import (
    RESUMABLE_STATUSES,
    ResearchResultStore,
    ResearchTaskConflict,
    open_research_result_store,
)
from .research_events import (
    FIRST_EVENT, ResearchEvents, open_research_events, stream_research_events, valid_event_id,
)

# Setup logging
logger = logging.getLogger(__name__)

# Don't override parent logger settings
logger.propagate = True

# Silence uvicorn reload logs
logging.getLogger("uvicorn.supervisors.ChangeReload").setLevel(logging.WARNING)

# Models


class ResearchRequest(BaseModel):
    task: str
    report_type: str
    report_source: str
    tone: str
    headers: dict | None = None
    repo_name: str
    branch_name: str
    generate_in_background: bool = True


class ResumeResearchRequest(BaseModel):
    headers: dict | None = None


class QueuedResearchRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    task: str = Field(min_length=1)
    report_type: Literal["multi_agents"] = "multi_agents"
    report_source: ReportSource = ReportSource.Web
    tone: str = "Objective"


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")  # Allow extra fields in the request
    
    report: str
    messages: List[Dict[str, Any]]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    os.makedirs("outputs", exist_ok=True)
    app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
    
    # Mount frontend static files
    frontend_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "frontend")
    if os.path.exists(frontend_path):
        app.mount("/site", StaticFiles(directory=frontend_path), name="frontend")
        logger.debug(f"Frontend mounted from: {frontend_path}")
        
        # Also mount the static directory directly for assets referenced as /static/
        static_path = os.path.join(frontend_path, "static")
        if os.path.exists(static_path):
            app.mount("/static", StaticFiles(directory=static_path), name="static")
            logger.debug(f"Static assets mounted from: {static_path}")
    else:
        logger.warning(f"Frontend directory not found: {frontend_path}")
    
    async with (
        open_research_result_store(os.getenv("DATABASE_URL")) as result_store,
        open_research_events(os.getenv("REDIS_URL")) as research_events,
    ):
        app.state.research_results = result_store
        app.state.research_events = research_events
        logger.info("GPT Researcher API ready")
        try:
            yield
        finally:
            app.state.research_results = None
            app.state.research_events = None
            logger.info("Research API shutting down")

# App initialization
app = FastAPI(lifespan=lifespan)

# Configure allowed origins for CORS
allowed_origins_env = os.getenv("CORS_ALLOW_ORIGINS")
ALLOWED_ORIGINS = (
    [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
    if allowed_origins_env
    else [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://app.gptr.dev",
    ]
)

# Standard JSON response - no custom MongoDB encoding needed

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use default JSON response class

# Mount static files for frontend
# Get the absolute path to the frontend directory
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend"))

# Mount static directories
app.mount("/static", StaticFiles(directory=os.path.join(frontend_dir, "static")), name="static")
app.mount("/site", StaticFiles(directory=frontend_dir), name="site")

# WebSocket manager
manager = WebSocketManager()

report_store = ReportStore(Path(os.getenv('REPORT_STORE_PATH', os.path.join('data', 'reports.json'))))

# Constants
DOC_PATH = os.getenv("DOC_PATH", "./my-docs")

# Startup event


# Lifespan events now handled in the lifespan context manager above


# Routes
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main frontend HTML page."""
    frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend"))
    index_path = os.path.join(frontend_dir, "index.html")
    
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Frontend index.html not found")
    
    with open(index_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    return HTMLResponse(content=content)


@app.get("/.well-known/agent-discovery.json")
async def agent_discovery(request: Request):
    """Advertise GPT Researcher services via the Agent Discovery Protocol."""
    origin = str(request.base_url).rstrip("/")
    domain = request.url.hostname or request.headers.get("host", "")
    contact = os.getenv("AGENT_DISCOVERY_CONTACT")

    document = build_agent_discovery_document(origin=origin, domain=domain, contact=contact)
    response = JSONResponse(content=document)
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response

@app.get("/report/{research_id}")
async def read_report(request: Request, research_id: str):
    docx_path = os.path.join('outputs', f"{research_id}.docx")
    if not os.path.exists(docx_path):
        return {"message": "Report not found."}
    return FileResponse(docx_path)


# Simplified API routes - no database persistence
@app.get("/api/reports")
async def get_all_reports(report_ids: str = None):
    report_ids_list = report_ids.split(",") if report_ids else None
    reports = await report_store.list_reports(report_ids_list)
    return {"reports": reports}


@app.get("/api/reports/{research_id}")
async def get_report_by_id(research_id: str):
    report = await report_store.get_report(research_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    return {"report": report}


@app.post("/api/reports")
async def create_or_update_report(request: Request):
    try:
        data = await request.json()
        research_id = data.get("id", "temp_id")

        now_ms = int(time.time() * 1000)
        existing = await report_store.get_report(research_id)
        incoming_timestamp = data.get("timestamp")
        timestamp = incoming_timestamp if isinstance(incoming_timestamp, int) else now_ms
        if existing and isinstance(existing.get("timestamp"), int):
            timestamp = max(timestamp, existing["timestamp"])

        report = {
            "id": research_id,
            "question": data.get("question"),
            "answer": data.get("answer"),
            "orderedData": data.get("orderedData") or [],
            "chatMessages": data.get("chatMessages") or [],
            "timestamp": timestamp,
        }

        await report_store.upsert_report(research_id, report)
        return {"success": True, "id": research_id}
    except Exception as e:
        logger.error(f"Error processing report creation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/reports/{research_id}")
async def update_report(research_id: str, request: Request):
    existing = await report_store.get_report(research_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Report not found")

    data = await request.json()
    now_ms = int(time.time() * 1000)

    updated = {
        **existing,
        **{k: v for k, v in data.items() if v is not None},
        "id": research_id,
        "timestamp": now_ms,
    }

    await report_store.upsert_report(research_id, updated)
    return {"success": True, "id": research_id}


@app.delete("/api/reports/{research_id}")
async def delete_report(research_id: str):
    existed = await report_store.delete_report(research_id)
    if not existed:
        raise HTTPException(status_code=404, detail="Report not found")
    return {"success": True}


@app.get("/api/reports/{research_id}/chat")
async def get_report_chat(research_id: str):
    report = await report_store.get_report(research_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    return {"chatMessages": report.get("chatMessages") or []}


@app.post("/api/reports/{research_id}/chat")
async def research_report_chat(research_id: str, request: Request):
    """Chat against a stored report: LLM reply + persist user/assistant messages."""
    report = await report_store.get_report(research_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")

    try:
        data = await request.json()
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}

    # Accept either a bare chat message object or {messages, report, message}
    incoming_messages = data.get("messages")
    if not isinstance(incoming_messages, list):
        if any(key in data for key in ("role", "content")):
            incoming_messages = [data]
        elif isinstance(data.get("message"), dict):
            incoming_messages = [data["message"]]
        else:
            incoming_messages = []

    report_text = data.get("report") or report.get("answer") or ""
    history = report.get("chatMessages") or []
    if not isinstance(history, list):
        history = []

    messages_for_llm = list(history)
    for msg in incoming_messages:
        if isinstance(msg, dict):
            messages_for_llm.append(msg)

    try:
        chat_agent = ChatAgentWithMemory(
            report=report_text if isinstance(report_text, str) else str(report_text),
            config_path="default",
            headers=None,
        )
        response_content, tool_calls_metadata = await chat_agent.chat(messages_for_llm, None)
    except Exception as e:
        logger.error(f"Error in research report chat: {str(e)}", exc_info=True)
        return {"error": str(e)}

    now_ms = int(time.time() * 1000)
    response_message = {
        "role": "assistant",
        "content": response_content,
        "timestamp": now_ms,
        "metadata": {"tool_calls": tool_calls_metadata} if tool_calls_metadata else None,
    }

    new_chat = list(history)
    for msg in incoming_messages:
        if isinstance(msg, dict):
            new_chat.append(msg)
    new_chat.append(response_message)

    updated = {
        **report,
        "chatMessages": new_chat,
        "timestamp": now_ms,
    }
    await report_store.upsert_report(research_id, updated)
    return {"success": True, "id": research_id, "response": response_message}




async def write_report(
    research_request: ResearchRequest,
    research_id: str,
    result_store: ResearchResultStore | None = None,
    *,
    resume: bool = False,
):
    if result_store is None:
        return await _generate_report(research_request, research_id)

    async def execute(checkpointer):
        return await _generate_report(
            research_request, research_id, checkpointer=checkpointer, resume=resume
        )

    return await result_store.run(research_id, execute, resume=resume)


async def _generate_report(
    research_request: ResearchRequest, research_id: str, *, checkpointer=None, resume=False,
    event_sink=None,
):
    report_information = await run_agent(
        task=research_request.task,
        report_type=research_request.report_type,
        report_source=research_request.report_source,
        source_urls=[],
        document_urls=[],
        tone=Tone[research_request.tone],
        websocket=None,
        stream_output=stream_output if event_sink is not None else None,
        headers=research_request.headers,
        query_domains=[],
        config_path="",
        return_researcher=True,
        research_id=research_id,
        checkpointer=checkpointer,
        resume=resume,
        event_sink=event_sink,
    )

    if research_request.report_type == "multi_agents":
        report = report_information
        research_information = None
    else:
        report, researcher = report_information
        research_information = {
            "source_urls": researcher.get_source_urls(),
            "research_costs": researcher.get_costs(),
            "visited_urls": list(researcher.visited_urls),
            "research_images": researcher.get_research_images(),
            "search_errors": researcher.get_search_errors(),
        }

    response = {
        "research_id": research_id,
        "report": report,
        "docx_path": await write_md_to_word(report, research_id),
        "pdf_path": await write_md_to_pdf(report, research_id),
    }
    if research_information is not None:
        response["research_information"] = research_information
    return response

@app.post("/report/")
async def generate_report(
    research_request: ResearchRequest, background_tasks: BackgroundTasks, request: Request
):
    result_store = getattr(request.app.state, "research_results", None)
    research_id = (
        f"task_{uuid4().hex}" if result_store is not None
        else sanitize_filename(f"task_{int(time.time())}_{research_request.task}")
    )
    if result_store is not None:
        if research_request.tone not in Tone.__members__:
            raise HTTPException(status_code=422, detail="Unknown report tone.")
        await result_store.register(
            research_id, research_request.task, research_request.report_type, research_request.tone
        )

    if research_request.generate_in_background:
        background_tasks.add_task(write_report, research_request, research_id, result_store)
        return {"message": "Your report is being generated in the background. Please check back later.",
                "research_id": research_id}
    try:
        if result_store is None:
            return await write_report(research_request, research_id)
        return await write_report(research_request, research_id, result_store)
    except InsufficientSearchEvidence as error:
        raise HTTPException(status_code=422, detail={
            "message": str(error),
            "research_id": research_id,
            "search_errors": error.search_errors,
        }) from error
    except Exception as error:
        if result_store is None:
            raise
        logger.exception("Research task %s failed", research_id)
        raise HTTPException(
            status_code=500, detail={"message": "Research task failed.", "research_id": research_id}
        ) from error


def _require_research_store(request: Request) -> ResearchResultStore:
    result_store = getattr(request.app.state, "research_results", None)
    if result_store is None:
        raise HTTPException(status_code=503, detail="Research result storage is disabled.")
    return result_store


def _research_events(request: Request) -> ResearchEvents | None:
    return getattr(request.app.state, "research_events", None)


@app.post("/api/research", status_code=202)
async def submit_research(research_request: QueuedResearchRequest, request: Request):
    result_store = getattr(request.app.state, "research_results", None)
    if result_store is None or _research_events(request) is None:
        raise HTTPException(status_code=503, detail="Research worker is disabled.")
    if research_request.tone not in Tone.__members__:
        raise HTTPException(status_code=422, detail="Unknown report tone.")
    research_id = f"task_{uuid4().hex}"
    await result_store.register(
        research_id, research_request.task, research_request.report_type, research_request.tone,
        request_payload=research_request.model_dump(mode="json"),
    )
    return {"research_id": research_id, "status": "queued"}


@app.get("/api/research/{research_id}")
async def get_research_task(research_id: str, request: Request):
    task = await _require_research_store(request).get_task(research_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Research task not found.")
    task["can_resume"] = (
        task["report_type"] == "multi_agents"
        and task["status"] in RESUMABLE_STATUSES
        and (not task["worker_managed"] or task["status"] != "queued")
    )
    events = _research_events(request)
    if events and task["status"] == "running" and task["worker_managed"]:
        try:
            progress = await events.snapshot(research_id)
            if progress.get("attempt_id") == task["attempt_id"]:
                task["progress"] = progress
        except RedisError:
            pass  # PostgreSQL still provides the task status.
    return task


@app.get("/api/research/{research_id}/events")
async def research_event_stream(research_id: str, request: Request, after: str | None = None):
    result_store = _require_research_store(request)
    task = await result_store.get_task(research_id)
    if not task or not task["worker_managed"]:
        raise HTTPException(status_code=404, detail="Research event stream not found.")
    events = _research_events(request)
    if events is None:
        raise HTTPException(status_code=503, detail="Research events are disabled.")
    cursor = request.headers.get("last-event-id", after) or FIRST_EVENT
    if not valid_event_id(cursor):
        raise HTTPException(status_code=422, detail="Invalid research event cursor.")
    return StreamingResponse(
        stream_research_events(events, result_store, research_id, cursor),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/research/{research_id}/resume")
async def resume_research(
    research_id: str, request: Request, options: ResumeResearchRequest | None = None
):
    result_store = _require_research_store(request)
    task = await result_store.get_task(research_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Research task not found.")
    if task["report_type"] != "multi_agents" or task["status"] not in RESUMABLE_STATUSES:
        raise HTTPException(status_code=409, detail="Research task cannot be resumed.")

    if task["worker_managed"]:
        if options and options.headers:
            raise HTTPException(status_code=422, detail="Research workers use server credentials.")
        if not await result_store.enqueue_resume(research_id, task["attempt_id"]):
            raise HTTPException(status_code=409, detail="Research task cannot be resumed.")
        return JSONResponse({"research_id": research_id, "status": "queued"}, status_code=202)

    research_request = ResearchRequest(
        task=task["query"], report_type=task["report_type"], tone=task["tone"],
        report_source="web", repo_name="", branch_name="", generate_in_background=False,
        headers=options.headers if options is not None else None,
    )
    try:
        return await write_report(research_request, research_id, result_store, resume=True)
    except ResearchTaskConflict as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except Exception as error:
        logger.exception("Research task %s could not be resumed", research_id)
        raise HTTPException(status_code=500, detail="Research task failed during resume.") from error


@app.post("/api/research/{research_id}/cancel")
async def cancel_research(research_id: str, request: Request):
    store = _require_research_store(request)
    task = await store.get_task(research_id)
    if not task or not task["worker_managed"]:
        raise HTTPException(status_code=404, detail="Worker-managed research task not found.")
    status = await store.request_cancel(research_id)
    if status is None:
        raise HTTPException(status_code=409, detail="Research task cannot be cancelled.")
    return JSONResponse(
        {"research_id": research_id, "status": status, "cancel_requested": True},
        status_code=202 if status == "running" else 200,
    )


@app.get("/api/research/{research_id}/result")
async def get_research_result(research_id: str, request: Request):
    result = await _require_research_store(request).get(research_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Research result not found.")
    return result


@app.get("/files/")
async def list_files():
    if not os.path.exists(DOC_PATH):
        os.makedirs(DOC_PATH, exist_ok=True)
    files = os.listdir(DOC_PATH)
    print(f"Files in {DOC_PATH}: {files}")
    return {"files": files}


@app.post("/api/multi_agents")
async def run_multi_agents():
    return await execute_multi_agents(manager)


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    return await handle_file_upload(file, DOC_PATH)


@app.delete("/files/{filename}")
async def delete_file(filename: str):
    return await handle_file_deletion(filename, DOC_PATH)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        await handle_websocket_communication(
            websocket, manager, getattr(websocket.app.state, "research_results", None)
        )
    except WebSocketDisconnect as e:
        logger.info(f"WebSocket disconnected with code {e.code} and reason: '{e.reason}'")
    except Exception as e:
        logger.error(f"Unexpected WebSocket error: {str(e)}")
    finally:
        # Release the connection's queue and sender task however the loop ended.
        await manager.disconnect(websocket)

@app.post("/api/chat")
async def chat(chat_request: ChatRequest):
    """Process a chat request with a report and message history.

    Args:
        chat_request: ChatRequest object containing report text and message history

    Returns:
        JSON response with the assistant's message and any tool usage metadata
    """
    try:
        logger.info(f"Received chat request with {len(chat_request.messages)} messages")

        # Create chat agent with the report
        chat_agent = ChatAgentWithMemory(
            report=chat_request.report,
            config_path="default",
            headers=None
        )

        # Process the chat and get response with metadata
        response_content, tool_calls_metadata = await chat_agent.chat(chat_request.messages, None)
        logger.info(f"response_content: {response_content}")
        logger.info(f"Got chat response of length: {len(response_content) if response_content else 0}")
        
        if tool_calls_metadata:
            logger.info(f"Tool calls used: {json.dumps(tool_calls_metadata)}")

        # Format response as a ChatMessage object with role, content, timestamp and metadata
        response_message = {
            "role": "assistant",
            "content": response_content,
            "timestamp": int(time.time() * 1000),  # Current time in milliseconds
            "metadata": {
                "tool_calls": tool_calls_metadata
            } if tool_calls_metadata else None
        }

        logger.info(f"Returning formatted response: {json.dumps(response_message)[:100]}...")
        return {"response": response_message}
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}", exc_info=True)
        return {"error": str(e)}

