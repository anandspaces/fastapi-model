"""Alembic environment.

Loads DATABASE_URL from .env via python-dotenv, hands it to SQLAlchemy after
the same scheme rewrite the app uses (postgresql:// → postgresql+psycopg://),
and points target_metadata at the declarative Base so --autogenerate sees every
ORM model.
"""

from logging.config import fileConfig

from alembic import context
from dotenv import load_dotenv
from sqlalchemy import pool

# Load .env BEFORE importing src.db so DATABASE_URL is set when base.py runs.
load_dotenv()

from src.db import engine  # noqa: E402  (after load_dotenv)
from src.db.base import Base  # noqa: E402
from src.db import models as _models  # noqa: F401, E402 — registers tables on Base.metadata


config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Emit SQL statements rather than executing them."""
    context.configure(
        url=str(engine.url),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against the live Postgres connection.

    Reuses the app's own ``engine`` so pool / SSL / driver settings are
    identical to what the app sees at runtime.
    """
    # NullPool keeps Alembic from holding the connection across migrations.
    connectable = engine.execution_options().execution_options(poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
