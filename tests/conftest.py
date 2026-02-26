import pytest
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.main import app
from database.database import get_db
from database.models import Base


SECURITY_TEST_MODULES = {
    "test_api_security.py",
    "test_api_security_integration.py",
}




@pytest.fixture
def db_session():
    """
    Function-scoped SQLite session for API integration/security tests.

    Uses StaticPool so the same in-memory DB is shared across connections.
    """
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    testing_session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    session = testing_session_local()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset in-memory rate limit state between tests to avoid cross-test bleed."""
    limiters = []

    app_limiter = getattr(app.state, "limiter", None)
    if app_limiter is not None:
        limiters.append(app_limiter)

    try:
        from api.auth_routes import limiter as auth_limiter
        limiters.append(auth_limiter)
    except Exception:
        pass

    for limiter in limiters:
        if hasattr(limiter, "reset"):
            limiter.reset()
    yield
    for limiter in limiters:
        if hasattr(limiter, "reset"):
            limiter.reset()


@pytest.fixture(autouse=True)
def override_db_for_security_modules(request):
    """
    Ensure global TestClient-based security tests use isolated test DB sessions.
    """
    module_name = Path(str(request.fspath)).name
    if module_name not in SECURITY_TEST_MODULES:
        yield
        return

    session = request.getfixturevalue("db_session")

    def _override_get_db():
        try:
            yield session
        finally:
            pass

    app.dependency_overrides[get_db] = _override_get_db
    try:
        yield
    finally:
        app.dependency_overrides.pop(get_db, None)
