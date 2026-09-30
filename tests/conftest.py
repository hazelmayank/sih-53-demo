import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.mapping.adaptive_grid import ResolutionSchedule  # noqa: E402

REFERENCE = ([10.0, 30.0, 60.0, 100.0], [0.05, 0.10, 0.25, 0.50])


@pytest.fixture
def schedule() -> ResolutionSchedule:
    return ResolutionSchedule(*REFERENCE, max_range_m=100.0)


@pytest.fixture
def uniform() -> ResolutionSchedule:
    return ResolutionSchedule.uniform(0.05, 100.0)
