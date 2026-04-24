"""
NeuroAgent — Database Session Manager
Async SQLAlchemy engine with pgvector support and connection pooling.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger("db.session")
settings = get_settings()

_engine = create_async_engine(
    settings.database_url,
    pool_size=settings.database_pool_size,
    max_overflow=settings.database_max_overflow,
    pool_pre_ping=True,
    echo=settings.environment == "development",
)

AsyncSessionFactory = async_sessionmaker(
    bind=_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Async context manager that yields a database session."""
    async with AsyncSessionFactory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency: yields a session per request."""
    async with get_session() as session:
        yield session


async def dispose_engine() -> None:
    """Clean up the connection pool on shutdown."""
    await _engine.dispose()
    logger.info("Database engine disposed")


def get_engine():
    return _engine
