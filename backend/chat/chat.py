import logging
import os
import json
from fastapi import WebSocket
from typing import List, Dict, Any

from gpt_researcher.config.config import Config
from gpt_researcher.context.select import select_context
from gpt_researcher.utils.llm import create_chat_completion
from gpt_researcher.utils.tools import create_chat_completion_with_tools, create_search_tool
try:
    from tavily import TavilyClient
except ImportError:  # optional dependency for chat web search
    TavilyClient = None
from datetime import datetime

# Setup logging
# Get logger instance
logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()  # Only log to console
    ]
)

# Note: LLM client is now handled through GPT Researcher's unified LLM system
# This supports all configured providers (OpenAI, Google Gemini, Anthropic, etc.)

#: The system prompt is a module constant, byte-identical on every turn.
#:
#: Prompt caching only reuses a *token prefix*, so anything that changes between
#: turns must not sit in front of the conversation. This prompt used to embed
#: the report context and ``datetime.now()``, which changed the first tokens of
#: every request: the provider's cache could never hit, not even for the
#: conversation history that really was identical. The volatile parts (retrieved
#: report slices, current time) now go in the last user message, after all the
#: history, so the cacheable prefix is ``system + every earlier turn``.
_SYSTEM_PROMPT = """You are GPT Researcher, an autonomous research agent created by an open source community at https://github.com/assafelovic/gpt-researcher, homepage: https://gptr.dev.
To learn more about GPT Researcher you can suggest to check out: https://docs.gptr.dev.

This is a chat about a research report that you created. Answer based on the given context and report.
You must include citations to your answer based on the report.

You may use the quick_search tool when the user asks about information that might require current data not found in the report, such as recent events, updated statistics, or news. If there's no report available, you can use the quick_search tool to find information online.

You must respond in markdown format. You must make it readable with paragraphs, tables, etc when possible. Remember that you're answering in a chat not a report."""


def _usable_turns(messages) -> list[dict]:
    """role/content messages only, in order, with timestamps and metadata dropped.

    Client-supplied ``system`` turns are dropped: the agent owns the system
    prompt, and a second one mid-history breaks both providers and caching.
    """
    turns = []
    for msg in messages or []:
        if isinstance(msg, dict) and msg.get("role") in ("user", "assistant") and msg.get("content") is not None:
            turns.append({"role": str(msg["role"]), "content": str(msg["content"])})
    return turns


def build_chat_messages(report_context: str, messages, now: datetime | None = None) -> list[dict]:
    """Assemble one chat request: stable system prompt, verbatim history, volatile tail.

    Only the last user turn carries the retrieved report slices and the current
    time. Everything before it is byte-identical to what the previous turn sent,
    which is what lets a provider's prefix cache (and its ~10x discount on cached
    input) apply at all.
    """
    turns = _usable_turns(messages)
    stamp = (now or datetime.now()).isoformat(timespec="seconds")

    last_user = None
    for index, turn in enumerate(turns):
        if turn["role"] == "user":
            last_user = index

    if last_user is None:
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _volatile_tail(report_context, "", stamp)},
        ]

    prompt = [{"role": "system", "content": _SYSTEM_PROMPT}]
    for index, turn in enumerate(turns):
        if index == last_user:
            prompt.append({
                "role": "user",
                "content": _volatile_tail(report_context, turn["content"], stamp),
            })
        else:
            prompt.append(turn)
    return prompt


def _volatile_tail(report_context: str, question: str, stamp: str) -> str:
    """The part of the request that changes every turn."""
    blocks = []
    context = (report_context or "").strip()
    if context:
        blocks.append(f"<report_context>\n{context}\n</report_context>")
    blocks.append(f"<current_time>{stamp}</current_time>")
    if question:
        blocks.append(question)
    return "\n\n".join(blocks)


def prompt_layout(prompt: list[dict]) -> dict:
    """Token split of one request: the cacheable prefix against the volatile tail."""
    from gpt_researcher.context.budget import count_tokens

    if not prompt:
        return {"stable_prefix_tokens": 0, "volatile_tail_tokens": 0, "messages": 0}
    stable = sum(count_tokens(m.get("content")) for m in prompt[:-1])
    tail = count_tokens(prompt[-1].get("content"))
    return {
        "stable_prefix_tokens": stable,
        "volatile_tail_tokens": tail,
        "messages": len(prompt),
    }


def get_tools():
    """Define tools for LLM function calling (primarily for OpenAI-compatible providers)"""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "quick_search",
                "description": "Search for current events or online information when you need new knowledge that doesn't exist in the current context",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]
    return tools

class ChatAgentWithMemory:
    def __init__(
        self,
        report: str,
        config_path="default",
        headers=None,
        vector_store=None
    ):
        self.report = report
        self.headers = headers
        self.config = Config(config_path)
        self.vector_store = vector_store
        self.retriever = None
        self.search_metadata = None
        
        # Initialize Tavily client (optional - only if API key is available)
        tavily_api_key = os.environ.get("TAVILY_API_KEY")
        if tavily_api_key and TavilyClient is not None:
            self.tavily_client = TavilyClient(api_key=tavily_api_key)
        else:
            self.tavily_client = None
            if TavilyClient is None:
                logger.warning("tavily package not installed - web search in chat will be disabled")
            else:
                logger.warning("TAVILY_API_KEY not set - web search in chat will be disabled")
        
        # A caller-supplied vector store is theirs to query. Otherwise the report
        # goes through the same context filter as research (CONTEXT_FILTER).
        if self.vector_store is not None:
            try:
                self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
            except TypeError:
                self.retriever = self.vector_store.as_retriever()

    def quick_search(self, query):
        """Perform a web search for current information using Tavily"""
        try:
            # Check if Tavily client is available
            if self.tavily_client is None:
                logger.warning(f"Tavily client not available, skipping web search for: {query}")
                self.search_metadata = {
                    "query": query,
                    "sources": [],
                    "error": "Web search is disabled - TAVILY_API_KEY not configured"
                }
                return {
                    "error": "Web search is disabled - TAVILY_API_KEY not configured",
                    "results": []
                }
            
            logger.info(f"Performing web search for: {query}")
            results = self.tavily_client.search(query=query, max_results=5)
            
            # Store search metadata for frontend
            self.search_metadata = self._build_search_metadata(query, results)
            
            return results
        except Exception as e:
            logger.error(f"Error performing web search: {str(e)}", exc_info=True)
            results = {
                "error": str(e),
                "results": []
            }
            self.search_metadata = self._build_search_metadata(query, results)
            return results

    @staticmethod
    def _build_search_metadata(query, results):
        """Describe one search result without making another provider request."""
        metadata = {"query": query, "sources": []}
        for result in results.get("results", []):
            content = result.get("content", "")
            metadata["sources"].append({
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "content": content[:200] + "..." if len(content) > 200 else content,
            })
        if "error" in results:
            metadata["error"] = results["error"]
        return metadata


    async def process_chat_completion(self, messages: List[Dict[str, str]]):
        """Process chat completion using configured LLM provider with tool calling support"""
        processed_metadata = []

        def search_with_metadata(query):
            results = self.quick_search(query)
            # Keep sources local to this completion and tied to the evidence
            # returned to the model, including repeated queries and failures.
            processed_metadata.append({
                "tool": "quick_search",
                "query": query,
                "search_metadata": self._build_search_metadata(query, results),
            })
            return results

        search_tool = create_search_tool(search_with_metadata)
        
        # Use the tool-enabled chat completion utility
        response, tool_calls_metadata = await create_chat_completion_with_tools(
            messages=messages,
            tools=[search_tool],
            model=self.config.smart_llm_model,
            llm_provider=self.config.smart_llm_provider,
            llm_kwargs=self.config.llm_kwargs,
        )
        
        # The utility returns no tool metadata when it falls back to a plain
        # completion; that answer was not generated from these search results.
        return response, processed_metadata if tool_calls_metadata else []



    async def _retrieve_context(self, user_message: str) -> str:
        """The parts of the report relevant to the latest user message.

        Uses the injected vector store when there is one, otherwise the same
        context filter research uses. Falls back to the full report, so chat
        always has something to answer from.
        """
        if not self.report or not user_message:
            return self.report or ""
        if self.retriever is not None:
            try:
                docs = self.retriever.invoke(user_message)
                chunks = [str(getattr(d, "page_content", "") or "") for d in docs or []]
                if any(chunks):
                    return "\n\n".join(c for c in chunks if c)
            except Exception as exc:  # noqa: BLE001 - retrieval must not break chat
                logger.warning(f"Report retrieval failed, using the context filter: {exc}")
        try:
            context = await select_context(
                user_message,
                [{"url": "report", "title": "Research report", "raw_content": self.report}],
                self.config,
            )
        except Exception as exc:  # noqa: BLE001 - chat must keep working
            logger.warning(f"Report context selection failed, using full report: {exc}")
            return self.report
        return context or self.report

    async def chat(self, messages, websocket=None):
        """Chat with configured LLM provider (supports OpenAI, Google Gemini, Anthropic, etc.)
        
        Args:
            messages: List of chat messages with role and content
            websocket: Optional websocket for streaming responses
        
        Returns:
            tuple: (str: The AI response message, dict: metadata about tool usage)
        """
        try:
            
            # Prefer retrieved report slices over stuffing the entire report each turn
            last_user = ""
            for msg in reversed(messages or []):
                if isinstance(msg, dict) and msg.get("role") == "user" and msg.get("content"):
                    last_user = str(msg.get("content"))
                    break
            report_context = await self._retrieve_context(last_user)

            # Stable system prompt, verbatim history, report slices and the clock
            # in the volatile tail. See _SYSTEM_PROMPT for why the split matters.
            formatted_messages = build_chat_messages(report_context, messages)
            layout = prompt_layout(formatted_messages)
            logger.info(
                "Chat prompt: %s message(s), cacheable prefix %s tokens, volatile tail %s tokens",
                layout["messages"], layout["stable_prefix_tokens"], layout["volatile_tail_tokens"],
            )

            # Process the chat using configured LLM provider
            ai_message, tool_calls_metadata = await self.process_chat_completion(formatted_messages)
            
            # Provide fallback response if message is empty
            if not ai_message:
                logger.warning("No AI message content found in response, using fallback message")
                ai_message = "I apologize, but I couldn't generate a proper response. Please try asking your question again."
            
            logger.info(f"Generated response: {ai_message[:100]}..." if len(ai_message) > 100 else f"Generated response: {ai_message}")
            
            # Return both the message and any metadata about tools used
            return ai_message, tool_calls_metadata
            
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}", exc_info=True)
            raise

    def get_context(self):
        """return the current context of the chat"""
        return self.report
