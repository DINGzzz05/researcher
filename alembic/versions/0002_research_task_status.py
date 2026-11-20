"""Add execution state to research results."""

import sqlalchemy as sa
from alembic import op

revision = "0002_research_task_status"
down_revision = "0001_research_results"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column("research_results", "result", nullable=True)
    op.add_column(
        "research_results",
        sa.Column("status", sa.Text(), nullable=False, server_default="succeeded"),
    )
    op.add_column("research_results", sa.Column("query", sa.Text()))
    op.add_column("research_results", sa.Column("report_type", sa.Text()))
    op.add_column("research_results", sa.Column("tone", sa.Text()))
    op.add_column("research_results", sa.Column("error_code", sa.Text()))
    op.add_column("research_results", sa.Column("attempt_id", sa.Text()))
    for name in ("created_at", "updated_at"):
        op.add_column(
            "research_results",
            sa.Column(name, sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        )


def downgrade() -> None:
    op.execute("DELETE FROM research_results WHERE result IS NULL")
    op.alter_column("research_results", "result", nullable=False)
    for name in (
        "updated_at", "created_at", "attempt_id", "error_code", "tone", "report_type", "query", "status"
    ):
        op.drop_column("research_results", name)
