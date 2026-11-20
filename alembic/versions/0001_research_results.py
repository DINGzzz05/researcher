"""Create the research results table."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "0001_research_results"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "research_results",
        sa.Column("research_id", sa.Text(), primary_key=True),
        sa.Column("result", JSONB(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("research_results")
