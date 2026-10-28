from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).parents[1]


@pytest.fixture(scope="session")
def golden_cases() -> list[dict]:
    path = Path(__file__).parents[1] / "fixtures" / "golden" / "biomni-identity-v1.json"
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == "echo.golden-trajectories.v1"
    return payload["cases"]


@pytest.fixture(scope="session")
def golden_by_id(golden_cases: list[dict]) -> dict[str, dict]:
    return {case["id"]: case for case in golden_cases}
