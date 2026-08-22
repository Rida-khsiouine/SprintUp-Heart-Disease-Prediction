from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).parents[1]


def test_notebook_executes_from_clean_kernel(tmp_path: Path) -> None:
    notebook_path = PROJECT_ROOT / "notebooks" / "analysis.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
    )

    executed = client.execute()
    nbformat.write(executed, tmp_path / "analysis.executed.ipynb")

    code = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )
    assert ".fit(" not in code
    assert "GridSearchCV" not in code
    assert "reports/unsupervised/summary.json" in code
    assert "reports/unsupervised/cluster-selection.csv" in code
