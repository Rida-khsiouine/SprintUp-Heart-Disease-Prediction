from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]


def test_ci_uses_immutable_actions_and_required_checks() -> None:
    workflow = (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    uses = re.findall(r"uses:\s*[^@\s]+@([^\s#]+)", workflow)
    checkout_steps = workflow.count("uses: actions/checkout@")

    assert uses
    assert all(re.fullmatch(r"[0-9a-f]{40}", reference) for reference in uses)
    assert checkout_steps == 2
    assert workflow.count("fetch-depth: 0") == checkout_steps
    assert "contents: read" in workflow
    assert 'python-version: ["3.11", "3.12"]' in workflow
    assert "uv sync --frozen --all-extras" in workflow
    assert "uv run ruff check ." in workflow
    assert 'uv run pytest -m "not full" --cov=heart_disease' in workflow
    assert "uv run heart-disease reproduce --profile smoke" in workflow
    assert "workflow_dispatch:" in workflow
    assert "uv run heart-disease reproduce --profile full" in workflow
