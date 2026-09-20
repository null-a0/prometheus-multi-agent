from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run integration/model-backed tests",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-integration"):
        return

    skip_integration = pytest.mark.skip(
        reason="integration/model-backed test; pass --run-integration to run"
    )
    for item in items:
        if "integration" in item.keywords or "model_backed" in item.keywords:
            item.add_marker(skip_integration)
