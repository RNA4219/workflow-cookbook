"""CLI既定の共有readとwrite競合を一時DBで検証する。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / "tools/supervision/workspace_coordinator.py"


def test_cli_reads_share_without_global_wip_and_block_writer(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    base = [
        sys.executable,
        str(SCRIPT),
        "acquire",
        "--workspace",
        str(workspace),
        "--state-root",
        str(tmp_path / "state"),
    ]
    outputs = []
    for owner, mode in [("reader-a", "read"), ("reader-b", "read"), ("writer", "write")]:
        result = subprocess.run(
            base + ["--owner", owner, "--task-id", owner, "--mode", mode],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        outputs.append((result.returncode, json.loads(result.stdout)))
    assert [code for code, _ in outputs] == [0, 0, 2]
    assert outputs[0][1]["lease"]["wip_key"] is None
    assert outputs[1][1]["acquired"] is True
    assert outputs[2][1]["reason"] == "workspace_busy"
