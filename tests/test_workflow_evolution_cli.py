"""公開CLIを実プロセスで起動し、JSONと終了コードを検証する。"""

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "module,args",
    [
        ("tools.context.progressive", ["--repo-root", ".", "--query", "needle", "--budget-bytes", "100"]),
        ("tools.workflow_plugins.run_report", ["--traces", "absent.json", "--outcome", "absent.json"]),
        (
            "tools.workflow_plugins.checkpoint",
            [
                "status",
                "--plan",
                "absent.json",
                "--state-client",
                "absent.json",
                "--workspace",
                ".",
                "--coordinator-root",
                "unused",
            ],
        ),
    ],
)
def test_module_entrypoints_emit_json_and_exit_status(tmp_path, module, args):
    # ソースrootはimportだけに使い、入出力先はfixture workspaceへ限定する。
    import os

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    result = subprocess.run(
        [sys.executable, "-B", "-X", "utf8", "-m", module, *args],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=15,
    )
    payload = json.loads(result.stdout)
    if module == "tools.context.progressive":
        assert result.returncode == 0
        assert payload["status"] == "not_found" and payload["context"] == ""
    else:
        assert result.returncode == 1
        assert payload["status"] == "invalid" and payload["error"]
    assert "Traceback" not in result.stderr
    assert not (tmp_path / "unused").exists()
