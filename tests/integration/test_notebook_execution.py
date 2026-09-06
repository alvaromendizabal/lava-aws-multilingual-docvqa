"""Exercise real local kernels, IPC transport and cleanup after cell failure."""

from __future__ import annotations

import os
import runpy
import socket
from pathlib import Path

import nbformat
import pytest
from nbclient.exceptions import CellExecutionError

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
    execute = runpy.run_path(str(ROOT / "scripts/execute_notebook_smoke.py"))["execute_notebook"]
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
