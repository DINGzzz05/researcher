"""Persist requests that can be executed by an independent research worker."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "0003_research_job_queue"
down_revision = "0002_research_task_status"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("research_results", sa.Column("request_payload", JSONB()))
    for name in ("resume_requested", "cancel_requested"):
        op.add_column(
            "research_results",
            sa.Column(name, sa.Boolean(), nullable=False, server_default=sa.false()),
        )
    op.create_index(
        "research_jobs_queued",
        "research_results",
        ["updated_at", "research_id"],
        postgresql_where=sa.text("status = 'queued' AND request_payload IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("research_jobs_queued", table_name="research_results")
    for name in ("cancel_requested", "resume_requested", "request_payload"):
        op.drop_column("research_results", name)
