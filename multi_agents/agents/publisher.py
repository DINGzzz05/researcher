from gpt_researcher.retrievers.errors import format_search_limitations

from .utils.file_formats import \
    write_md_to_pdf, \
    write_md_to_word, \
    write_text_to_md

from .utils.views import print_agent_output
from .claim_review import UNVERIFIED_CLAIM
from .drafts import read_draft


class PublisherAgent:
    def __init__(self, output_dir: str, websocket=None, stream_output=None, headers=None):
        self.websocket = websocket
        self.stream_output = stream_output
        self.output_dir = output_dir
        self.headers = headers or {}
        
    async def publish_research_report(self, research_state: dict, publish_formats: dict):
        layout = self.generate_layout(research_state)
        await self.write_report_by_formats(layout, publish_formats)

        return layout

    def generate_layout(self, research_state: dict):
        sections = []
        research_data = research_state.get("research_data") or []
        if not isinstance(research_data, (list, tuple)):
            research_data = [research_data]
        unresolved_sections = set(research_state.get("unresolved_claim_sections") or [])
        drafts_dir = research_state.get("section_drafts_dir")
        draft_files = research_state.get("section_drafts") or {}
        for index, subheader in enumerate(research_data):
            sections_by_title = subheader.items() if isinstance(subheader, dict) else [("Section", subheader)]
            for title, body in sections_by_title:
                # The full draft lives on disk; the state only carries a summary.
                saved = read_draft(drafts_dir, draft_files.get(title))
                if saved is not None:
                    text = saved
                elif body is None:
                    sections.append(f"### {title}\n\nInsufficient evidence. Research produced no usable draft for this section.")
                    continue
                else:
                    text = str(body)
                if index in unresolved_sections:
                    text = text.replace(
                        UNVERIFIED_CLAIM, f"**[证据不足 / Insufficient evidence]** {UNVERIFIED_CLAIM}",
                    )
                sections.append(text)

        sections_text = "\n\n".join(sections)
        sources = research_state.get("sources") or []
        if not isinstance(sources, (list, tuple)):
            sources = [sources] if sources else []
        references = "\n".join(f"{reference}" for reference in sources)
        headers = research_state.get("headers") or {}
        if not isinstance(headers, dict):
            headers = {}

        diagrams_text = ""
        diagrams = research_state.get("diagrams") or []
        if not isinstance(diagrams, (list, tuple)):
            diagrams = [diagrams] if diagrams else []
        if diagrams:
            diagrams_text = (
                "\n## Visualizations\n" + "\n\n".join(str(d) for d in diagrams) + "\n"
            )

        layout = f"""# {headers.get('title')}
#### {headers.get("date")}: {research_state.get('date')}

## {headers.get("introduction")}
{research_state.get('introduction')}

## {headers.get("table_of_contents")}
{research_state.get('table_of_contents')}

{sections_text}

## {headers.get("conclusion")}
{research_state.get('conclusion')}
{diagrams_text}
## {headers.get("references")}
{references}
"""
        return layout + format_search_limitations(research_state.get("search_errors"))

    async def write_report_by_formats(self, layout:str, publish_formats: dict):
        if publish_formats.get("pdf"):
            await write_md_to_pdf(layout, self.output_dir)
        if publish_formats.get("docx"):
            await write_md_to_word(layout, self.output_dir)
        if publish_formats.get("markdown"):
            await write_text_to_md(layout, self.output_dir)

    async def run(self, research_state: dict):
        task = research_state.get("task")
        publish_formats = task.get("publish_formats")
        if self.websocket and self.stream_output:
            await self.stream_output("logs", "publishing", f"Publishing final research report based on retrieved data...", self.websocket)
        else:
            print_agent_output(output="Publishing final research report based on retrieved data...", agent="PUBLISHER")
        final_research_report = await self.publish_research_report(research_state, publish_formats)
        return {"report": final_research_report}
