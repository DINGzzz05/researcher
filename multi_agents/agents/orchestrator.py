import os
import time
from langgraph.graph import StateGraph, END
from gpt_researcher.concurrency.budget import budget_scope
from .utils.views import print_agent_output
from ..memory.research import ResearchState
from .utils.utils import sanitize_filename
from .plan_review import (
    DEFAULT_MAX_PLAN_REVISIONS,
    route_human_feedback,
)
from .fact_review import (
    DEFAULT_MAX_FACT_CHECK_REVISIONS,
    MaxFactCheckRevisionsExceededError,
    route_fact_check,
)

# Import agent classes
from . import \
    WriterAgent, \
    EditorAgent, \
    PublisherAgent, \
    ResearchAgent, \
    HumanAgent, \
    FactCheckerAgent, \
    VisualizerAgent


class ChiefEditorAgent:
    """Agent responsible for managing and coordinating editing tasks."""

    def __init__(
        self, task: dict, websocket=None, stream_output=None, tone=None, headers=None,
        *, task_id=None, allow_console_feedback=True,
    ):
        self.task = task
        self.websocket = websocket
        self.stream_output = stream_output
        self.headers = headers or {}
        self.tone = tone
        self.allow_console_feedback = allow_console_feedback
        self.task_id = task_id if task_id is not None else self._generate_task_id()
        self.output_dir = self._create_output_directory()

    def _generate_task_id(self):
        # Currently time based, but can be any unique identifier
        return int(time.time())

    def _create_output_directory(self):
        output_dir = "./outputs/" + \
            sanitize_filename(
                f"run_{self.task_id}_{self.task.get('query')[0:40].strip()}")

        os.makedirs(output_dir, exist_ok=True)
        # Section drafts are written here so the shared graph state carries
        # summaries only; see agents/drafts.py.
        self.section_drafts_dir = os.path.join(output_dir, "sections")
        os.makedirs(self.section_drafts_dir, exist_ok=True)
        return output_dir

    def _initialize_agents(self):
        return {
            "writer": WriterAgent(self.websocket, self.stream_output, self.headers),
            "editor": EditorAgent(self.websocket, self.stream_output, self.tone, self.headers),
            "research": ResearchAgent(self.websocket, self.stream_output, self.tone, self.headers),
            "publisher": PublisherAgent(self.output_dir, self.websocket, self.stream_output, self.headers),
            "human": HumanAgent(
                self.websocket, self.stream_output, self.headers, allow_console=self.allow_console_feedback
            ),
            "fact_checker": FactCheckerAgent(self.websocket, self.stream_output, self.headers),
            "visualizer": VisualizerAgent(self.websocket, self.stream_output, self.headers)
        }

    def _create_workflow(self, agents):
        workflow = StateGraph(ResearchState)

        # Add nodes for each agent
        workflow.add_node("browser", agents["research"].run_initial_research)
        workflow.add_node("planner", agents["editor"].plan_research)
        workflow.add_node("researcher", agents["editor"].run_parallel_research)
        workflow.add_node("writer", agents["writer"].run)
        workflow.add_node("fact_checker", agents["fact_checker"].run)
        workflow.add_node("visualizer", agents["visualizer"].run)
        workflow.add_node("publisher", agents["publisher"].run)
        workflow.add_node("human", agents["human"].review_plan)

        # Add edges
        self._add_workflow_edges(workflow)

        return workflow

    def _add_workflow_edges(self, workflow):
        workflow.add_edge('browser', 'planner')
        workflow.add_edge('planner', 'human')
        workflow.add_edge('researcher', 'writer')
        workflow.add_edge('writer', 'fact_checker')
        workflow.add_edge('visualizer', 'publisher')
        workflow.set_entry_point("browser")
        workflow.add_edge('publisher', END)

        # Human loop: exact "no" approval (human agent) + bounded plan revisions
        # via plan_review.route_human_feedback (task.max_plan_revisions).
        workflow.add_conditional_edges(
            'human',
            self._route_human_feedback,
            {"accept": "researcher", "revise": "planner"},
        )

        # Fact-checker loop — bounded via task.max_fact_check_revisions
        workflow.add_conditional_edges(
            'fact_checker',
            self._route_fact_check,
            {"accept": "visualizer", "revise": "writer"}
        )

    def _route_human_feedback(self, review):
        """Route human plan feedback; force-accept after max_plan_revisions.

        ``route_human_feedback`` raises when the configured ceiling is exceeded.
        For the graph edge we treat that as accept so the run proceeds to
        research instead of dying at LangGraph's default recursion_limit.
        """
        from .plan_review import MaxPlanRevisionsExceededError

        max_plan_revisions = self.task.get(
            "max_plan_revisions", DEFAULT_MAX_PLAN_REVISIONS)
        try:
            return route_human_feedback(review, max_plan_revisions)
        except MaxPlanRevisionsExceededError:
            return "accept"

    def _route_fact_check(self, state):
        """Route fact-check results; force-accept once the ceiling is passed.

        ``route_fact_check`` raises when max_fact_check_revisions is exceeded.
        As with the human loop, the edge treats that as accept so the run
        proceeds instead of dying at LangGraph's recursion_limit.
        """
        max_fact_check_revisions = self.task.get(
            "max_fact_check_revisions", DEFAULT_MAX_FACT_CHECK_REVISIONS)
        try:
            return route_fact_check(state, max_fact_check_revisions)
        except MaxFactCheckRevisionsExceededError:
            return "accept"

    def init_research_team(self):
        """Initialize and create a workflow for the research team."""
        agents = self._initialize_agents()
        return self._create_workflow(agents)

    async def _log_research_start(self):
        message = f"Starting the research process for query '{self.task.get('query')}'..."
        if self.websocket and self.stream_output:
            await self.stream_output("logs", "starting_research", message, self.websocket)
        else:
            print_agent_output(message, "MASTER")

    async def run_research_task(self, task_id=None, *, checkpointer=None, resume=False):
        """Run a new graph or continue the saved thread for the same research task."""
        research_team = self.init_research_team()
        chain = research_team.compile(checkpointer=checkpointer)
        await self._log_research_start()

        config = {"configurable": {"thread_id": str(task_id or self.task_id)}}
        # One budget for the graph: bounds how many sections the planner may
        # turn into concurrent research graphs. Each section then installs its
        # own budget when it calls conduct_research.
        graph_input = {
            "task": self.task,
            "section_drafts_dir": self.section_drafts_dir,
        }
        with budget_scope():
            if checkpointer is None:
                return await chain.ainvoke(graph_input, config=config)

            if resume:
                state = await chain.aget_state(config)
                if state.metadata is not None:
                    if not state.next:
                        # Report generation may have finished before exporting or saving the result.
                        return state.values
                    self.task = state.values.get("task", self.task)
                    graph_input = None

            return await chain.ainvoke(graph_input, config=config, durability="sync")
