"""Exercise real local kernels, IPC transport and cleanup after cell failure."""

from __future__ import annotations

import os
import runpy
import socket
from pathlib import Path

import nbformat
import pytest
from nbclient.exceptions import CellExecutionError

from lava.notebook_execution import NOTEBOOK_STEMS

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(os.name != "posix", reason="Linux IPC transport regression")
@pytest.mark.parametrize("fail", [False, True])
def test_real_notebook_kernel_uses_ipc_and_cleans_up(tmp_path, capfd, fail):
    try:
        with socket.socket(socket.AF_UNIX) as probe:
            probe.bind(str(tmp_path / "socket"))
    except PermissionError:
        pytest.skip("Execution sandbox prohibits local IPC sockets; exercised in Linux CI")
    source = (
        "from pathlib import Path\n"
        "from ipykernel.kernelapp import IPKernelApp\n"
        "kernel = IPKernelApp.instance()\n"
        "assert kernel.transport == 'ipc'\n"
        "Path('kernel_socket_directory.txt').write_text(str(Path(kernel.ip).parent))\n"
        "assert 6 * 7 == 42\n"
    )
    if fail:
        source += "raise RuntimeError('intentional regression test')\n"
    path = tmp_path / "notebook.ipynb"
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source)])
    nbformat.write(notebook, path)
    original = path.read_bytes()
    execute = runpy.run_path(str(ROOT / "scripts/execute_notebooks.py"))["execute_notebook"]
    if fail:
        with pytest.raises(CellExecutionError, match="intentional regression test"):
            execute(path, tmp_path, 30, kernel_name="python3")
    else:
        result = execute(path, tmp_path, 30, kernel_name="python3")
        assert result.cells[0].execution_count == 1
    socket_directory = Path((tmp_path / "kernel_socket_directory.txt").read_text())
    assert not socket_directory.exists()
    assert path.read_bytes() == original
    captured = capfd.readouterr()
    assert "without encryption" not in captured.err
    assert "TqdmWarning" not in captured.err


@pytest.mark.skipif(os.name != "posix", reason="Linux notebook runtime")
@pytest.mark.parametrize("stem", NOTEBOOK_STEMS)
def test_walkthrough_executes_all_steps_without_changing_source(tmp_path, stem):
    """Run the actual employer-facing notebooks and verify completed stage evidence."""
    import json

    try:
        with socket.socket(socket.AF_UNIX) as probe:
            probe.bind(str(tmp_path / "notebook-socket"))
    except PermissionError:
        pytest.skip("Execution sandbox prohibits local IPC sockets; exercised in Linux CI")
    path = ROOT / "notebooks" / f"{stem}.ipynb"
    original = path.read_bytes()
    execute = runpy.run_path(str(ROOT / "scripts/execute_notebooks.py"))["execute_notebook"]
    result = execute(path, ROOT, 90, kernel_name="python3")
    code = [cell for cell in result.cells if cell.cell_type == "code"]
    assert [cell.execution_count for cell in code] == list(range(1, len(code) + 1))
    outputs = [output for cell in code for output in cell.outputs]
    assert not any(output.output_type == "error" for output in outputs)
    events = []
    for output in outputs:
        if output.output_type == "stream":
            assert output.name == "stdout"
            for line in output.text.splitlines():
                if line.startswith("{"):
                    events.append(json.loads(line))
    assert events[-1]["event"].endswith("walkthrough.completed")
    assert events[-1]["elapsed_seconds"] >= 0
    assert all("timestamp_utc" in event for event in events)
    assert not any(event["level"] != "INFO" for event in events)
    html_outputs = [
        output.data["text/html"] for output in outputs if "text/html" in output.get("data", {})
    ]
    expected = {
        "00": "Data audit complete",
        "01": "Three reader pilots scored",
        "02": "Full pilot scored",
        "03": "Full pilot scored",
        "04": "Full-document retrieval evaluated",
    }[stem[:2]]
    assert any(expected in text for text in html_outputs)
    assert path.read_bytes() == original
