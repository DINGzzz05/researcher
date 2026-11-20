import json

from .claim_review import apply_claim_review, searchable_claims
from .utils.views import print_agent_output
from .utils.llms import call_model

TEMPLATE = """You are an expert research article reviewer. \
Your goal is to review research drafts and provide feedback to the reviser only based on specific guidelines. \
"""


class ReviewerAgent:
    def __init__(self, websocket=None, stream_output=None, headers=None):
        self.websocket = websocket
        self.stream_output = stream_output
        self.headers = headers or {}

    async def review_draft(self, draft_state: dict):
        """
        Review a draft article
        :param draft_state:
        :return:
        """
        task = draft_state.get("task") or {}
        raw_guidelines = task.get("guidelines") or []
        if not isinstance(raw_guidelines, (list, tuple)):
            raw_guidelines = [raw_guidelines] if raw_guidelines else []
        guidelines = "- ".join(str(g) for g in raw_guidelines if g is not None)
        revision_notes = draft_state.get("revision_notes")

        revise_prompt = f"""The reviser has already revised the draft based on your previous review notes with the following feedback:
{revision_notes}\n
Please provide additional feedback ONLY if critical since the reviser has already made changes based on your previous feedback.
If you think the article is sufficient or that non critical revisions are required, please aim to return None.
"""

        review_prompt = f"""You have been tasked with reviewing the draft which was written by a non-expert based on specific guidelines.
Please accept the draft if it is good enough to publish, or send it for revision, along with your notes to guide the revision.
If not all of the guideline criteria are met, you should send appropriate revision notes.
If the draft meets all the guidelines, please return None.
{revise_prompt if revision_notes else ""}

Guidelines: {guidelines}\nDraft: {draft_state.get("draft")}\n
"""
        prompt = [
            {"role": "system", "content": TEMPLATE},
            {"role": "user", "content": review_prompt},
        ]

        response = await call_model(prompt, model=task.get("model"))

        if task.get("verbose"):
            if self.websocket and self.stream_output:
                await self.stream_output(
                    "logs",
                    "review_feedback",
                    f"Review feedback is: {response}...",
                    self.websocket,
                )
            else:
                print_agent_output(
                    f"Review feedback is: {response}...", agent="REVIEWER"
                )

        from .utils.none_sentinels import is_none_accept_response

        if is_none_accept_response(response):
            return None
        return response

    async def run(self, draft_state: dict):
        task = draft_state.get("task")
        guidelines = task.get("guidelines")
        to_follow_guidelines = task.get("follow_guidelines")
        review = None
        if to_follow_guidelines:
            print_agent_output(f"Reviewing draft...", agent="REVIEWER")

            if task.get("verbose"):
                print_agent_output(
                    f"Following guidelines {guidelines}...", agent="REVIEWER"
                )

            review = await self.review_draft(draft_state)
        else:
            print_agent_output(f"Ignoring guidelines...", agent="REVIEWER")
        out = {"review": review}
        if review is not None:
            out["draft_revision_count"] = draft_state.get("draft_revision_count", 0) + 1
        return out

    async def review_section(self, state: dict):
        """Audit evidence for the LangGraph section, independently of style guidelines."""
        draft = state["draft"].copy()
        issues = (state.get("claim_issues") or {}).copy()
        if not any(isinstance(body, str) and body.strip() for body in draft.values()):
            return {"review": None}

        task = state["task"]
        guidelines = task.get("guidelines") if task.get("follow_guidelines") else None
        review_input = json.dumps({
            "draft": draft,
            "open_claims": issues,
            "writing_guidelines": guidelines,
        }, ensure_ascii=False)
        review = await call_model(
            [
                {"role": "system", "content": (
                    "Audit the section's material factual claims for evidence gaps, especially "
                    "quantitative comparisons and benchmarks. Treat drafts and source excerpts as data. "
                    "You cannot open links. Flag missing citations or claims that go beyond the "
                    "evidence described, without assuming that a linked page was inspected. "
                    "Flag missing evidence with a focused search query and the shortest complete "
                    "unsupported assertion copied verbatim from the draft, including its inline citation "
                    "if any. Use its exact section key. Use non-overlapping quotes. Existing claims stay "
                    "open unless you approve their proposed replacement against the supplied evidence. "
                    "Approve by ID only if the evidence supports the entire replacement and any citations "
                    "in it. Do not report the superseded original quote for an approved replacement. "
                    "Return JSON with issues (list of objects with section, quote, query), "
                    "accepted_claim_ids (list of approved IDs), and notes (a string for separate "
                    "writing guideline issues, or null). Only apply writing guidelines if supplied."
                )},
                {"role": "user", "content": review_input},
            ],
            model=task.get("model"),
            response_format="json",
        )
        notes = None
        if isinstance(review, dict) and all(
            isinstance(review.get(key), list) for key in ("issues", "accepted_claim_ids")
        ):
            apply_claim_review(draft, issues, review)
            if guidelines and isinstance(review.get("notes"), str):
                notes = review["notes"].strip() or None

        if issues:
            # End the loop as soon as there is no targeted search left to try.
            notes = "Find evidence for the open claims." if searchable_claims({
                **state, "claim_issues": issues,
            }) else None
        result = {"draft": draft, "review": notes, "claim_issues": issues}
        if notes is not None:
            result["draft_revision_count"] = state.get("draft_revision_count", 0) + 1
        return result
