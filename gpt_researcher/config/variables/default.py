from .base import BaseConfig

DEFAULT_CONFIG: BaseConfig = {
    "RETRIEVER": "tavily",
    "EMBEDDING": "openai:text-embedding-3-small",
    "SIMILARITY_THRESHOLD": 0.42,
    "CONTEXT_FILTER": "auto",  # auto | jev | keyword | embeddings | summarize | none
    "CONTEXT_DEDUP": True,  # drop duplicate pages and exact/near-duplicate chunks
    "CONTEXT_DEDUP_THRESHOLD": 0.9,  # Jaccard similarity at which chunks count as the same text
    "CONTEXT_TOKEN_BUDGET": 12000,  # tokens of context per sub-query; 0 disables the cap
    "CONTEXT_SUMMARIZE": False,  # compress the selected chunks into one summary before budgeting
    "CONTEXT_RERANK": "",  # cross-encoder model; empty falls back to RERANK_MODEL, "off" disables
    "CONTEXT_RERANK_TOP_N": 30,  # fused candidates scored before the token budget is applied
    "CONTEXT_ORDER": "relevance",  # relevance | lost_in_middle (best passages at both ends)
    # Layered memory (see gpt_researcher/memory/long_term.py): episodes are
    # what was researched, facts are what was learned, the profile is what the
    # user keeps working on.
    "LONG_TERM_MEMORY": False,  # recall past research and store this run after writing
    "MEMORY_DB_PATH": "",  # default: ~/.gpt_researcher/memory.sqlite3
    "MEMORY_TTL_DAYS": 30,  # episode TTL: runs stop being recalled after this
    "MEMORY_FACT_TTL_DAYS": 365,  # fact TTL: claims outlive the run that reported them
    "MEMORY_SESSION_TTL_HOURS": 24,  # window in which an episode counts as "this session"
    "MEMORY_MAX_RECORDS": 1000,  # per-user cap on episodes
    "MEMORY_MAX_FACTS": 1000,  # per-user cap on active facts
    "MEMORY_FACTS_PER_RUN": 6,  # claims distilled from one report
    "MEMORY_TOP_K": 5,  # records recalled per layer
    "MEMORY_USER_ID": "default",  # namespace for recall and preferences
    # Runtime tool guard (gpt_researcher/harness/guards.py).
    "MAX_TOOL_CALLS": 200,  # tool-call steps per research unit; 0 disables the cap
    "MAX_DUPLICATE_TOOL_CALLS": 0,  # extra executions allowed for an identical call; 0 serves it from cache
    "MAX_TOOL_RESULT_CHARS": 20000,  # characters kept per result field; 0 disables clipping
    "MEMORY_SUMMARY_CHARS": 800,  # characters kept from the conclusion/body summary
    "MEMORY_MAX_SOURCES": 12,  # cleaned source URLs kept per record
    "MEMORY_BLOCK_CHARS": 2000,  # characters of memory injected into the writer context
    "MEMORY_EMBEDDING": "",  # embedding route; empty reuses EMBEDDING, "off" disables
    "MEMORY_RERANK": "off",  # "cross-encoder:<model>" to rerank; requires sentence-transformers
    "MEMORY_RERANK_CANDIDATES": 24,  # fused candidates passed to the reranker
    "MEMORY_VECTOR_MIN_COS": 0.3,  # cosine below which the vector route did not match
    "CITATION_CHECK": False,  # audit report citations against retrieved sources after writing
    "FAST_LLM": "openai:gpt-5.4-mini",
    "SMART_LLM": "openai:gpt-5.4",  # Has support for long responses (2k+ words).
    "STRATEGIC_LLM": "openai:gpt-5.4",  # Reasoning model used for planning; tune REASONING_EFFORT for speed vs. depth.
    # Output token limits. For reasoning models (the default gpt-5.x family)
    # these map to max_completion_tokens, which also covers reasoning tokens -
    # hence the generous headroom on top of the visible output.
    "FAST_TOKEN_LIMIT": 6000,
    "SMART_TOKEN_LIMIT": 12000,
    "STRATEGIC_TOKEN_LIMIT": 8000,
    "BROWSE_CHUNK_MAX_LENGTH": 8192,
    "CURATE_SOURCES": False,
    "SUMMARY_TOKEN_LIMIT": 700,
    "TEMPERATURE": 0.4,
    "USER_AGENT": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0",
    "MAX_SEARCH_RESULTS_PER_QUERY": 5,
    "MEMORY_BACKEND": "local",
    "TOTAL_WORDS": 1200,
    "REPORT_FORMAT": "APA",
    "MAX_ITERATIONS": 3,
    "AGENT_ROLE": None,
    "SCRAPER": "bs",
    "MAX_SCRAPER_WORKERS": 15,
    "SCRAPER_RATE_LIMIT_DELAY": 0.0,  # Minimum seconds between scraper requests (0 = no limit, useful for API rate limiting)
    "MAX_SUBTOPICS": 3,
    "LANGUAGE": "english",
    "REPORT_SOURCE": "web",
    "DOC_PATH": "./my-docs",
    "PROMPT_FAMILY": "default",
    "LLM_KWARGS": {},
    "EMBEDDING_KWARGS": {},
    "VERBOSE": False,
    # Deep research specific settings
    "DEEP_RESEARCH_BREADTH": 3,
    "DEEP_RESEARCH_DEPTH": 2,
    "DEEP_RESEARCH_CONCURRENCY": 4,
    
    # MCP retriever specific settings
    "MCP_SERVERS": [],  # List of predefined MCP server configurations
    "MCP_AUTO_TOOL_SELECTION": True,  # Whether to automatically select the best tool for a query
    "MCP_ALLOWED_ROOT_PATHS": [],  # List of allowed root paths for local file access
    "MCP_STRATEGY": "fast",  # MCP execution strategy: "fast", "deep", "disabled"
    "REASONING_EFFORT": "medium",
    
    # Image generation settings (optional - requires GOOGLE_API_KEY)
    # Free tier models: gemini-2.5-flash-image, gemini-2.0-flash-exp-image-generation
    # Paid tier models: imagen-4.0-generate-001, imagen-4.0-fast-generate-001
    "IMAGE_GENERATION_MODEL": "models/gemini-2.5-flash-image",
    "IMAGE_GENERATION_MAX_IMAGES": 3,  # Maximum number of images to generate per report
    "IMAGE_GENERATION_ENABLED": False,  # Master switch for inline image generation
    "IMAGE_GENERATION_STYLE": "dark",  # Image style: "dark" (matches app theme), "light", or "auto"
    "IMAGE_GENERATION_PROVIDER": "google",  # Image provider: "google" or "modelslab"
}
