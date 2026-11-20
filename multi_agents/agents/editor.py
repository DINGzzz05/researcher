from datetime import datetime
import asyncio
from typing import Dict, List, Optional

from langgraph.graph import StateGraph, END
from gpt_researcher.concurrency.budget import current_budget, default_budget
from gpt_researcher.retrievers.errors import InsufficientSearchEvidence

from .utils.views import print_agent_output
from .utils.llms import call_model
from ..memory.draft import DraftState
from . import ResearchAgent, ReviewerAgent, ReviserAgent
from .claim_review import finalize_claims
from .drafts import persist_draft
from .draft_review import (
    DEFAULT_MAX_DRAFT_REVISIONS,
    MaxDraftRevisionsExceededError,
    route_draft_review,
)


class EditorAgent:
    """Agent responsible for editing and managing code."""

    def __init__(self, websocket=None, stream_output=None, tone=None, headers=None):
        self.websocket = websocket
        self.stream_output = stream_output
        self.tone = tone
        self.headers = headers or {}

    async def plan_research(self, research_state: Dict[str, any]) -> Dict[str, any]:
        """
        Plan the research outline based on initial research and task parameters.

        :param research_state: Dictionary containing research state information
        :return: Dictionary with title, date, and planned sections
        """
        initial_research = research_state.get("initial_research")
        task = research_state.get("task")
        include_human_feedback = task.get("include_human_feedback")
        human_feedback = research_state.get("human_feedback")
        max_sections = task.get("max_sections")

        prompt = self._create_planning_prompt(
            initial_research, include_human_feedback, human_feedback, max_sections)

        message = "Planning an outline layout based on initial research..."
        if self.websocket and self.stream_output:
            await self.stream_output("logs", "planning", message, self.websocket)
        else:
            print_agent_output(message, agent="EDITOR")
        plan = await call_model(
            prompt=prompt,
            model=task.get("model"),
            response_format="json",
        )

        return {
            "title": plan.get("title"),
            "date": plan.get("date"),
            "sections": plan.get("sections"),
        }

    async def run_parallel_research(self, research_state: Dict[str, any]) -> dict:
        """
        Execute parallel research tasks for each section.

        :param research_state: Dictionary containing research state information
        :return: Dictionary with research results
        """
        workflow = self._create_workflow()
        chain = workflow.compile()

        queries = research_state.get("sections")
        title = research_state.get("title")

        # The planner is asked for at most max_sections headers, but the limit
        # was only in the prompt: a model returning ten created ten concurrent
        # section graphs, each with its own sub-queries and scraper pool. Enforce
        # the ceiling here, where the tasks are actually created.
        budget = current_budget() or default_budget()
        queries = budget.allow_sections(queries)

        await self._log_parallel_research(queries)
        completed = 0
        drafts_dir = research_state.get("section_drafts_dir")

        async def research_section(index, query):
            nonlocal completed
            result = await chain.ainvoke(
                self._create_task_input(research_state, query, title, queries),
                config={"tags": ["gpt-researcher"]},
            )
            completed += 1
            if self.websocket and self.stream_output:
                await self.stream_output(
                    "progress", "parallel_research", f"Researched {completed} of {len(queries)} sections",
                    self.websocket, metadata={"completed": completed, "total": len(queries)},
                )
            draft, unresolved = finalize_claims(result)
            summaries: dict[str, str | None] = {}
            files: dict[str, str] = {}
            for section_title, body in (draft or {}).items():
                if drafts_dir and isinstance(body, str):
                    summary, name = persist_draft(drafts_dir, index, section_title, body)
                    summaries[section_title] = summary
                    files[section_title] = name
                else:
                    summaries[section_title] = body
            return summaries, files, unresolved, result.get("search_errors", [])

        async with asyncio.TaskGroup() as group:
            drafts = [group.create_task(research_section(index, query)) for index, query in enumerate(queries)]
        results = [draft.result() for draft in drafts]
        search_errors = [error for _, _, _, errors in results for error in errors]
        if results and not any(any(draft.values()) for draft, _, _, _ in results):
            raise InsufficientSearchEvidence(search_errors)
        return {
            "research_data": [summaries for summaries, _, _, _ in results],
            "section_drafts": {title: name for _, files, _, _ in results for title, name in files.items()},
            "search_errors": search_errors,
            "unresolved_claim_sections": [
                index for index, (_, _, unresolved, _) in enumerate(results) if unresolved
            ],
        }

    def _create_planning_prompt(self, initial_research: str, include_human_feedback: bool,
                                human_feedback: Optional[str], max_sections: int) -> List[Dict[str, str]]:
        """Create the prompt for research planning."""
        return [
            {
                "role": "system",
                "content": "You are a research editor. Your goal is to oversee the research project "
                           "from inception to completion. Your main task is to plan the article section "
                           "layout based on an initial research summary.\n ",
            },
            {
                "role": "user",
                "content": self._format_planning_instructions(initial_research, include_human_feedback,
                                                              human_feedback, max_sections),
            },
        ]

    def _format_planning_instructions(self, initial_research: str, include_human_feedback: bool,
                                      human_feedback: Optional[str], max_sections: int) -> str:
        """Format the instructions for research planning."""
        today = datetime.now().strftime('%d/%m/%Y')
        feedback_instruction = (
            f"Human feedback: {human_feedback}. You must plan the sections based on the human feedback."
            if include_human_feedback and human_feedback and human_feedback != 'no'
            else ''
        )

        return f"""Today's date is {today}
                   Research summary report: '{initial_research}'
                   {feedback_instruction}
                   \nYour task is to generate an outline of sections headers for the research project
                   based on the research summary report above.
                   You must generate a maximum of {max_sections} section headers.
                   You must focus ONLY on related research topics for subheaders and do NOT include introduction, conclusion and references.
                   You must return nothing but a JSON with the fields 'title' (str) and 
                   'sections' (maximum {max_sections} section headers) with the following structure:
                   '{{title: string research title, date: today's date, 
                   sections: ['section header 1', 'section header 2', 'section header 3' ...]}}'."""

    def _initialize_agents(self) -> Dict[str, any]:
        """Initialize the research, reviewer, and reviser skills."""
        researcher = ResearchAgent(self.websocket, self.stream_output, self.tone, self.headers)
        return {
            "research": researcher,
            "reviewer": ReviewerAgent(self.websocket, self.stream_output, self.headers),
            "reviser": ReviserAgent(
                self.websocket, self.stream_output, self.headers, research_claim=researcher.research_claim,
            ),
        }

    def _create_workflow(self) -> StateGraph:
        """Create the workflow for the research process."""
        agents = self._initialize_agents()
        workflow = StateGraph(DraftState)

        workflow.add_node("researcher", agents["research"].run_depth_research)
        workflow.add_node("reviewer", agents["reviewer"].review_section)
        workflow.add_node("reviser", agents["reviser"].run)

        workflow.set_entry_point("researcher")
        workflow.add_conditional_edges(
            "researcher",
            lambda state: "reviewer" if any((state.get("draft") or {}).values()) else END,
            {"reviewer": "reviewer", END: END},
        )
        workflow.add_edge("reviser", "reviewer")
        workflow.add_conditional_edges(
            "reviewer",
            self._route_draft_review,
            {"accept": END, "revise": "reviser"},
        )

        return workflow

    def _route_draft_review(self, draft: Dict[str, any]) -> str:
        """Route reviewer output; force-accept once the ceiling is passed.

        ``route_draft_review`` raises when max_draft_revisions is exceeded.
        Its own docstring says the edge should force-accept gracefully rather
        than hit LangGraph's recursion_limit, which is what this does.
        """
        task = draft.get("task") or {}
        max_draft_revisions = task.get(
            "max_draft_revisions", DEFAULT_MAX_DRAFT_REVISIONS)
        try:
            return route_draft_review(draft, max_draft_revisions)
        except MaxDraftRevisionsExceededError:
            return "accept"

    async def _log_parallel_research(self, queries: List[str]) -> None:
        """Log the start of parallel research tasks."""
        if self.websocket and self.stream_output:
            await self.stream_output(
                "logs",
                "parallel_research",
                f"Running parallel research for the following queries: {queries}",
                self.websocket,
                metadata={"completed": 0, "total": len(queries)},
            )
        else:
            print_agent_output(
                f"Running the following research tasks in parallel: {queries}...",
                agent="EDITOR",
            )

    def _create_task_input(self, research_state: Dict[str, any], query: str, title: str,
                           sections: Optional[List[str]] = None) -> Dict[str, any]:
        """Create the input for a single research task."""
        return {
            "task": research_state.get("task"),
            "topic": query,
            "title": title,
            "headers": self.headers,
            # Sections are researched in parallel, so none can see what the others
            # wrote. Naming them lets each writer leave their ground alone (#1495).
            "sibling_sections": [s for s in (sections or []) if s != query],
        }
