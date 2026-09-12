"""CIのskip禁止を独立したpytestプロセスで確認する。"""

from pathlib import Path

import pytest

pytest_plugins = ["pytester"]


@pytest.mark.parametrize(
    ("body", "strict", "code"),
    [
        ("def test_case(): pass", True, 0),
        ("def test_case(): pytest.skip('missing runtime')", False, 0),
        ("def test_case(): pytest.skip('missing runtime')", True, 1),
        ("@pytest.mark.skipif(True, reason='runtime absent')\ndef test_case(): pass", True, 1),
        ("pytest.skip('missing module', allow_module_level=True)", True, 1),
        ("def test_case(): assert False", True, 1),
    ],
)
def test_ci_skip_exit_status(pytester: pytest.Pytester, body: str, strict: bool, code: int) -> None:
    pytester.makeconftest(Path(__file__).with_name("conftest.py").read_text(encoding="utf-8"))
    pytester.makepyfile(test_ok="def test_ok(): pass", test_case="import pytest\n" + body)
    result = pytester.runpytest_subprocess("-q", *(["--fail-on-skip"] if strict else []))
    assert result.ret == code
    if strict and "skip" in body:
        result.stdout.fnmatch_lines(["*Required CI tests skipped: *test_case*"])
