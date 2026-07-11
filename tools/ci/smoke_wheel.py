# SPDX-License-Identifier: MIT
"""Install a built wheel into an isolated environment and smoke-test its CLIs."""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
import venv
from pathlib import Path

ENTRYPOINTS = (
    "wfc-governance-gate",
    "wfc-collect-metrics",
    "wfc-codemap-update",
    "wfc-context-pack",
    "wfc-five-tool-manifest",
)


def _venv_python(root: Path) -> Path:
    return root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _venv_script(root: Path, name: str) -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    directory = "Scripts" if os.name == "nt" else "bin"
    return root / directory / f"{name}{suffix}"


def smoke_wheel(wheel: Path) -> None:
    wheel = wheel.resolve()
    if not wheel.is_file() or wheel.suffix != ".whl":
        raise ValueError(f"Wheel does not exist: {wheel}")

    with tempfile.TemporaryDirectory(prefix="wfc-wheel-smoke-") as raw_temp:
        temp = Path(raw_temp)
        environment = temp / "venv"
        venv.EnvBuilder(with_pip=True, clear=True).create(environment)
        python = _venv_python(environment)

        subprocess.run(
            [str(python), "-m", "pip", "install", "--disable-pip-version-check", str(wheel)],
            check=True,
            cwd=temp,
        )

        clean_env = os.environ.copy()
        clean_env.pop("PYTHONPATH", None)
        for name in ENTRYPOINTS:
            executable = _venv_script(environment, name)
            completed = subprocess.run(
                [str(executable), "--help"],
                check=False,
                cwd=temp,
                env=clean_env,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"{name} --help failed with exit code {completed.returncode}:\n"
                    f"{completed.stdout}\n{completed.stderr}"
                )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args(argv)
    smoke_wheel(args.wheel)
    print(f"Wheel smoke test passed: {args.wheel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
