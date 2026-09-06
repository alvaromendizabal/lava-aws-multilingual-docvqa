"""Resuming analysis must preserve completed work and reject stale/corrupt outputs."""

from unittest.mock import Mock

import nbformat
import pytest

from lava.notebook_execution import analysis_input_digest, execute_and_save


def successful_notebook():
    return nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                "print(42)",
                execution_count=1,
                outputs=[nbformat.v4.new_output("stream", name="stdout", text="42\n")],
            )
        ]
    )


def test_completed_notebook_is_reused_without_reexecuting(tmp_path):
    source = tmp_path / "analysis.ipynb"
    nbformat.write(successful_notebook(), source)
    original = source.read_bytes()
    execute = Mock(return_value=successful_notebook())
    args = {"input_digest": "a" * 64, "code_revision": "b" * 40, "execute": execute}
    first = execute_and_save(source, tmp_path / "outputs", **args)
    second = execute_and_save(source, tmp_path / "outputs", **args)
    assert first["reused"] is False and second["reused"] is True
    assert first["output_sha256"] == second["output_sha256"]
    execute.assert_called_once()
    assert source.read_bytes() == original


@pytest.mark.parametrize("change", ["source", "data", "output", "code"])
def test_changed_or_corrupt_notebooks_are_not_silently_reused(tmp_path, change):
    source = tmp_path / "analysis.ipynb"
    nbformat.write(successful_notebook(), source)
    execute = Mock(return_value=successful_notebook())
    args = {"input_digest": "a" * 64, "code_revision": "b" * 40, "execute": execute}
    execute_and_save(source, tmp_path / "outputs", **args)
    if change == "source":
        source.write_bytes(source.read_bytes() + b" ")
    elif change == "output":
        (tmp_path / "outputs/analysis.ipynb").write_text("corrupt")
    elif change == "data":
        args["input_digest"] = "c" * 64
    else:
        args["code_revision"] = "c" * 40
    with pytest.raises(ValueError):
        execute_and_save(source, tmp_path / "outputs", **args)
    execute.assert_called_once()


def test_failed_execution_is_not_committed_and_can_resume(tmp_path):
    source = tmp_path / "analysis.ipynb"
    nbformat.write(successful_notebook(), source)
    execute = Mock(side_effect=[RuntimeError("kernel stopped"), successful_notebook()])
    args = {"input_digest": "a" * 64, "code_revision": "b" * 40, "execute": execute}
    with pytest.raises(RuntimeError):
        execute_and_save(source, tmp_path / "outputs", **args)
    assert not (tmp_path / "outputs/analysis.manifest.json").exists()
    assert execute_and_save(source, tmp_path / "outputs", **args)["reused"] is False


def test_incomplete_outputs_and_source_overwrite_are_rejected(tmp_path):
    source = tmp_path / "analysis.ipynb"
    notebook = successful_notebook()
    nbformat.write(notebook, source)
    args = {"input_digest": "a" * 64, "code_revision": "b" * 40, "execute": lambda: notebook}
    with pytest.raises(ValueError, match="canonical"):
        execute_and_save(source, tmp_path, **args)
    notebook.cells[0].execution_count = None
    with pytest.raises(ValueError, match="fully executed"):
        execute_and_save(source, tmp_path / "outputs", **args)


def test_digest_tracks_data_and_code_but_not_transient_outputs(tmp_path):
    for name in ("uv.lock", "pyproject.toml"):
        (tmp_path / name).write_text("pinned")
    (tmp_path / "reports").mkdir()
    data = tmp_path / "reports/summary.json"
    data.write_text('{"score": 0.8}')
    before = analysis_input_digest(tmp_path)
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts/runtime.json").write_text("progress")
    assert analysis_input_digest(tmp_path) == before
    data.write_text('{"score": 0.9}')
    assert analysis_input_digest(tmp_path) != before


def test_canonical_runner_saves_and_resumes_outputs(tmp_path, monkeypatch, capsys):
    import runpy
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    command = runpy.run_path(str(root / "scripts/execute_notebook_smoke.py"))["main"]
    for name in ("uv.lock", "pyproject.toml"):
        (tmp_path / name).write_text("pinned")
    source = tmp_path / "analysis.ipynb"
    nbformat.write(successful_notebook(), source)
    original = source.read_bytes()
    execute = Mock(return_value=successful_notebook())
    monkeypatch.setitem(command.__globals__, "find_repo_root", lambda _: tmp_path)
    monkeypatch.setitem(command.__globals__, "git_snapshot", lambda _: {"git_commit_sha": "b" * 40})
    monkeypatch.setitem(command.__globals__, "execute_notebook", execute)
    monkeypatch.setattr(
        "sys.argv",
        ["execute_notebook_smoke.py", "analysis.ipynb", "--output-dir", "artifacts/analysis"],
    )
    assert command() == 0
    assert command() == 0
    execute.assert_called_once()
    assert source.read_bytes() == original
    assert (tmp_path / "artifacts/analysis/events.jsonl").exists()
    assert '"reused": true' in capsys.readouterr().out
