import pytest
from solver_python import config

@pytest.fixture(autouse=True)
def isolated_results(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_ROOT", tmp_path / "results")
