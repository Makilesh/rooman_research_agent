"""Shared fixtures. Nothing here touches the network or loads a model."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from research_agent import db
from research_agent.config import Config


class FakeClock:
    """A clock the limiter tests drive by hand, so no test ever sleeps."""

    def __init__(self, start: datetime | None = None) -> None:
        self.t = start or datetime(2026, 8, 20, 12, 0, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.t

    def advance(self, **kwargs: float) -> None:
        self.t += timedelta(**kwargs)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def cfg(tmp_path) -> Config:
    """A Config pointed entirely at a temp directory.

    Constructed rather than loaded, so a developer's real `.env` can never leak into
    a test run and change what the assertions mean.
    """
    return Config(
        db_path=tmp_path / "agent.db",
        llm_cache_dir=tmp_path / "llm_cache",
        embeddings_path=tmp_path / "embeddings.npy",
        outputs_dir=tmp_path / "outputs",
    )


@pytest.fixture
def conn(cfg: Config):
    c = db.connect(cfg)
    db.migrate(c)
    yield c
    c.close()
