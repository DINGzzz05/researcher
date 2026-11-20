from typing import Union, List, Dict, Any
from typing_extensions import TypedDict


class BaseConfig(TypedDict):
    RETRIEVER: str
    EMBEDDING: str
    SIMILARITY_THRESHOLD: float
    CONTEXT_FILTER: str
    CONTEXT_DEDUP: bool
    CONTEXT_DEDUP_THRESHOLD: float
    CONTEXT_TOKEN_BUDGET: int
    CONTEXT_SUMMARIZE: bool
    CONTEXT_RERANK: str
    CONTEXT_RERANK_TOP_N: int
    CONTEXT_ORDER: str
    LONG_TERM_MEMORY: bool
    MEMORY_DB_PATH: str
    MEMORY_TTL_DAYS: int
    MEMORY_FACT_TTL_DAYS: int
    MEMORY_SESSION_TTL_HOURS: int
    MEMORY_MAX_RECORDS: int
    MEMORY_MAX_FACTS: int
    MEMORY_FACTS_PER_RUN: int
    MEMORY_TOP_K: int
    MEMORY_USER_ID: str
    MAX_TOOL_CALLS: int
    MAX_DUPLICATE_TOOL_CALLS: int
    MAX_TOOL_RESULT_CHARS: int
    MEMORY_SUMMARY_CHARS: int
    MEMORY_MAX_SOURCES: int
    MEMORY_BLOCK_CHARS: int
    MEMORY_EMBEDDING: str
    MEMORY_RERANK: str
    MEMORY_RERANK_CANDIDATES: int
    MEMORY_VECTOR_MIN_COS: float
    CITATION_CHECK: bool
    FAST_LLM: str
    SMART_LLM: str
    STRATEGIC_LLM: str
    FAST_TOKEN_LIMIT: int
    SMART_TOKEN_LIMIT: int
    STRATEGIC_TOKEN_LIMIT: int
    BROWSE_CHUNK_MAX_LENGTH: int
    SUMMARY_TOKEN_LIMIT: int
    TEMPERATURE: float
    USER_AGENT: str
    MAX_SEARCH_RESULTS_PER_QUERY: int
    MEMORY_BACKEND: str
    TOTAL_WORDS: int
    REPORT_FORMAT: str
    CURATE_SOURCES: bool
    MAX_ITERATIONS: int
    LANGUAGE: str
    AGENT_ROLE: Union[str, None]
    SCRAPER: str
    MAX_SCRAPER_WORKERS: int
    SCRAPER_RATE_LIMIT_DELAY: float
    MAX_SUBTOPICS: int
    REPORT_SOURCE: Union[str, None]
    DOC_PATH: str
    PROMPT_FAMILY: str
    LLM_KWARGS: dict
    EMBEDDING_KWARGS: dict
    VERBOSE: bool
    DEEP_RESEARCH_CONCURRENCY: int
    DEEP_RESEARCH_DEPTH: int
    DEEP_RESEARCH_BREADTH: int
    MCP_SERVERS: List[Dict[str, Any]]
    MCP_AUTO_TOOL_SELECTION: bool
    MCP_USE_LLM_ARGS: bool
    MCP_ALLOWED_ROOT_PATHS: List[str]
    MCP_STRATEGY: str
    REASONING_EFFORT: str
    # Image generation settings
    IMAGE_GENERATION_MODEL: Union[str, None]
    IMAGE_GENERATION_MAX_IMAGES: int
    IMAGE_GENERATION_ENABLED: bool
    IMAGE_GENERATION_STYLE: str  # Image style: "dark", "light", or "auto"
    IMAGE_GENERATION_PROVIDER: str  # Image provider: "google" or "modelslab"
