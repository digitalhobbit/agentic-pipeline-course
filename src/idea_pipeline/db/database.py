from sqlalchemy import Engine
from sqlmodel import Session, create_engine

from idea_pipeline.core.settings import settings


def get_engine(db_url: str | None = None) -> Engine:
    if db_url is None:
        settings.db_path.parent.mkdir(parents=True, exist_ok=True)
        db_url = f"sqlite:///{settings.db_path}"
    return create_engine(db_url)


engine = get_engine()


def get_session() -> Session:
    return Session(engine)
