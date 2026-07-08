import pytest

from utils.sample_fixture import ensure_sample


@pytest.fixture(scope="session")
def sample_video():
    """Provide the shared public ASD clip; skip the test if it can't be fetched."""
    try:
        return ensure_sample()
    except Exception as exc:  # network / CDN failure -> skip, never fail the suite
        pytest.skip(f"sample fixture unavailable: {exc}")
