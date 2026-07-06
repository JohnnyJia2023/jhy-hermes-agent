import subprocess
import sys

from gateway.run import (
    _build_free_claude_prompt,
    _free_claude_is_enabled,
    _run_free_claude_command,
)


def test_free_claude_disabled_without_enabled_flag():
    assert _free_claude_is_enabled({}) is False
    assert _free_claude_is_enabled({"free_claude": {"enabled": False}}) is False


def test_free_claude_enabled_requires_command():
    assert (
        _free_claude_is_enabled(
            {"free_claude": {"enabled": True, "command": ["/bin/echo"]}}
        )
        is True
    )
    assert _free_claude_is_enabled({"free_claude": {"enabled": True}}) is False


def test_build_free_claude_prompt_includes_history_and_message():
    prompt = _build_free_claude_prompt(
        "What did I ask?",
        [{"role": "user", "content": "Earlier context"}],
        max_history_chars=200,
    )

    assert "What did I ask?" in prompt
    assert "Earlier context" in prompt
    assert "text-only reasoning engine" in prompt
    assert "delegate_task" in prompt
    assert "goal" in prompt
    assert "toolsets" in prompt
    assert "Do not claim a sub-agent was run unless Hermes returns its result" in prompt


def test_run_free_claude_command_sends_prompt_to_stdin(tmp_path):
    script = tmp_path / "free_claude_fake.py"
    script.write_text(
        "import sys\n"
        "payload = sys.stdin.read()\n"
        "print('reply:' + payload.splitlines()[-1])\n",
        encoding="utf-8",
    )

    result = _run_free_claude_command(
        "hello",
        {
            "free_claude": {
                "enabled": True,
                "command": [sys.executable, str(script)],
                "timeout": 5,
            }
        },
    )

    assert result["ok"] is True
    assert result["text"] == "reply:hello"


def test_run_free_claude_command_reports_nonzero_exit(tmp_path):
    script = tmp_path / "free_claude_fail.py"
    script.write_text(
        "import sys\n"
        "print('bad', file=sys.stderr)\n"
        "sys.exit(7)\n",
        encoding="utf-8",
    )

    result = _run_free_claude_command(
        "hello",
        {
            "free_claude": {
                "enabled": True,
                "command": [sys.executable, str(script)],
                "timeout": 5,
            }
        },
    )

    assert result["ok"] is False
    assert result["exit_code"] == 7
    assert "bad" in result["error"]


def test_run_free_claude_command_reports_timeout(monkeypatch):
    def raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=kwargs.get("args") or args[0], timeout=0.1)

    monkeypatch.setattr(subprocess, "run", raise_timeout)

    result = _run_free_claude_command(
        "hello",
        {
            "free_claude": {
                "enabled": True,
                "command": [sys.executable, "-c", "pass"],
                "timeout": 0.1,
            }
        },
    )

    assert result["ok"] is False
    assert "timeout" in result, result
    assert result["timeout"] is True
