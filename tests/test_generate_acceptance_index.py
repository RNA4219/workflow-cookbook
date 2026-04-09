from __future__ import annotations

import json
from pathlib import Path

from tools.ci import generate_acceptance_index


def _write_plugin(root: Path) -> Path:
    package_dir = root / "index_plugin"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "plugin.py").write_text(
        """
class Plugin:
    capabilities = ["acceptance.index"]

    def build_acceptance_index(self, *, repo_root):
        return {"markdown": "# Acceptance Index\\n\\n| A |\\n|---|\\n| x |\\n"}


def create_plugin():
    return Plugin()
""".strip(),
        encoding="utf-8",
    )
    config_path = root / "plugins.json"
    config_path.write_text(
        json.dumps(
            {
                "workflow_plugins": [
                    {
                        "factory": "index_plugin.plugin:create_plugin",
                        "python_paths": ["."],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return config_path


def test_generate_acceptance_index_writes_file(tmp_path: Path) -> None:
    config_path = _write_plugin(tmp_path)
    output_path = tmp_path / "INDEX.md"

    result = generate_acceptance_index.main(
        ["--plugin-config", str(config_path), "--output", str(output_path)]
    )

    assert result == 0
    assert output_path.read_text(encoding="utf-8").startswith("# Acceptance Index")
