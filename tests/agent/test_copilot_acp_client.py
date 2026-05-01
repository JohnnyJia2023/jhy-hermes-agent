import io
import json
import signal
import subprocess

from agent import copilot_acp_client as module
from agent.copilot_acp_client import CopilotACPClient, probe_copilot_acp


class _FakeProc:
    def __init__(self, stdout_lines: list[str] | None = None):
        joined = "".join(f"{line}\n" for line in (stdout_lines or []))
        self.stdin = io.StringIO()
        self.stdout = io.StringIO(joined)
        self.stderr = io.StringIO()
        self.pid = 4242
        self._returncode = None
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_calls: list[float] = []

    def poll(self):
        return self._returncode

    def terminate(self):
        self.terminate_calls += 1
        self._returncode = 0

    def kill(self):
        self.kill_calls += 1
        self._returncode = 0

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        self._returncode = 0
        return 0


def test_close_terminates_copilot_process_group(monkeypatch):
    proc = _FakeProc()
    client = CopilotACPClient(acp_command="copilot", acp_args=["--acp", "--stdio"])
    client._active_process = proc

    killpg_calls = []
    monkeypatch.setattr(module.os, "getpgid", lambda pid: pid + 1)
    monkeypatch.setattr(module.os, "killpg", lambda pgid, sig: killpg_calls.append((pgid, sig)))

    client.close()

    assert killpg_calls == [(4243, signal.SIGTERM)]
    assert proc.terminate_calls == 0
    assert proc.kill_calls == 0
    assert proc.stdin.closed is True
    assert proc.stdout.closed is True
    assert proc.stderr.closed is True


def test_run_prompt_launches_acp_in_its_own_session(monkeypatch):
    responses = [
        json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}}),
        json.dumps({"jsonrpc": "2.0", "id": 2, "result": {"sessionId": "sess-1"}}),
        json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"text": "OK"},
                    }
                },
            }
        ),
        json.dumps({"jsonrpc": "2.0", "id": 3, "result": {}}),
    ]
    proc = _FakeProc(stdout_lines=responses)
    popen_kwargs = {}

    def _fake_popen(*args, **kwargs):
        popen_kwargs.update(kwargs)
        return proc

    killpg_calls = []
    monkeypatch.setattr(module.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(module.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(module.os, "killpg", lambda pgid, sig: killpg_calls.append((pgid, sig)))

    client = CopilotACPClient(acp_command="copilot", acp_args=["--acp", "--stdio"], acp_cwd="/tmp")

    reply_text, reasoning_text = client._run_prompt("Reply with OK.", timeout_seconds=1)

    assert popen_kwargs["start_new_session"] is True
    assert reply_text == "OK"
    assert reasoning_text == ""
    assert killpg_calls == [(4242, signal.SIGTERM)]


def test_close_escalates_to_sigkill_when_group_does_not_exit(monkeypatch):
    proc = _FakeProc()

    def _wait(timeout=None):
        proc.wait_calls.append(timeout)
        if len(proc.wait_calls) == 1:
            raise subprocess.TimeoutExpired(cmd="copilot", timeout=timeout)
        proc._returncode = -9
        return 0

    proc.wait = _wait
    client = CopilotACPClient(acp_command="copilot", acp_args=["--acp", "--stdio"])
    client._active_process = proc

    killpg_calls = []
    monkeypatch.setattr(module.os, "getpgid", lambda pid: pid + 7)
    monkeypatch.setattr(module.os, "killpg", lambda pgid, sig: killpg_calls.append((pgid, sig)))

    client.close()

    assert killpg_calls == [
        (4249, signal.SIGTERM),
        (4249, signal.SIGKILL),
    ]


def test_probe_copilot_acp_uses_minimal_default_prompt(monkeypatch):
    captured = {}

    def _fake_run_prompt(self, prompt_text, *, timeout_seconds):
        captured["prompt_text"] = prompt_text
        captured["timeout_seconds"] = timeout_seconds
        return "OK", ""

    monkeypatch.setattr(CopilotACPClient, "_run_prompt", _fake_run_prompt)

    result = probe_copilot_acp(timeout_seconds=12, cwd="/tmp")

    assert result == "OK"
    assert captured["prompt_text"] == "Reply with OK."
    assert captured["timeout_seconds"] == 12.0


# ── Copilot ACP Safety Tests (upstream) ──────────────────────────────────────
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
class _FakeProcess:
    def __init__(self) -> None:
        self.stdin = io.StringIO()


class CopilotACPClientSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = CopilotACPClient(acp_cwd="/tmp")

    def _dispatch(self, message: dict, *, cwd: str) -> dict:
        process = _FakeProcess()
        handled = self.client._handle_server_message(
            message,
            process=process,
            cwd=cwd,
            text_parts=[],
            reasoning_parts=[],
        )
        self.assertTrue(handled)
        payload = process.stdin.getvalue().strip()
        self.assertTrue(payload)
        return json.loads(payload)

    def test_request_permission_is_not_auto_allowed(self) -> None:
        response = self._dispatch(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "session/request_permission",
                "params": {},
            },
            cwd="/tmp",
        )

        outcome = (((response.get("result") or {}).get("outcome") or {}).get("outcome"))
        self.assertEqual(outcome, "cancelled")

    def test_read_text_file_blocks_internal_hermes_hub_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "home"
            blocked = home / ".hermes" / "skills" / ".hub" / "index-cache" / "entry.json"
            blocked.parent.mkdir(parents=True, exist_ok=True)
            blocked.write_text('{"token":"sk-test-secret-1234567890"}')

            with patch.dict(
                os.environ,
                {"HOME": str(home), "HERMES_HOME": str(home / ".hermes")},
                clear=False,
            ):
                response = self._dispatch(
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "fs/read_text_file",
                        "params": {"path": str(blocked)},
                    },
                    cwd=str(home),
                )

        self.assertIn("error", response)

    def test_read_text_file_redacts_sensitive_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            secret_file = root / "config.env"
            secret_file.write_text("OPENAI_API_KEY=sk-proj-abc123def456ghi789jkl012")

            # agent.redact snapshots HERMES_REDACT_SECRETS at import time into
            # _REDACT_ENABLED, so patching os.environ is a no-op. Flip the
            # module-level constant directly for the duration of the call.
            with patch("agent.redact._REDACT_ENABLED", True):
                response = self._dispatch(
                    {
                        "jsonrpc": "2.0",
                        "id": 3,
                        "method": "fs/read_text_file",
                        "params": {"path": str(secret_file)},
                    },
                    cwd=str(root),
                )

        content = ((response.get("result") or {}).get("content") or "")
        self.assertNotIn("abc123def456", content)
        self.assertIn("OPENAI_API_KEY=", content)

    def test_write_text_file_reuses_write_denylist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "home"
            target = home / ".ssh" / "id_rsa"
            target.parent.mkdir(parents=True, exist_ok=True)

            with patch("agent.copilot_acp_client.is_write_denied", return_value=True, create=True):
                response = self._dispatch(
                    {
                        "jsonrpc": "2.0",
                        "id": 4,
                        "method": "fs/write_text_file",
                        "params": {
                            "path": str(target),
                            "content": "fake-private-key",
                        },
                    },
                    cwd=str(home),
                )

        self.assertIn("error", response)
        self.assertFalse(target.exists())

    def test_write_text_file_respects_safe_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            safe_root = root / "workspace"
            safe_root.mkdir()
            outside = root / "outside.txt"

            with patch.dict(os.environ, {"HERMES_WRITE_SAFE_ROOT": str(safe_root)}, clear=False):
                response = self._dispatch(
                    {
                        "jsonrpc": "2.0",
                        "id": 5,
                        "method": "fs/write_text_file",
                        "params": {
                            "path": str(outside),
                            "content": "should-not-write",
                        },
                    },
                    cwd=str(root),
                )

        self.assertIn("error", response)
        self.assertFalse(outside.exists())


if __name__ == "__main__":
    unittest.main()


# ── HOME env propagation tests (from PR #11285) ─────────────────────

from unittest.mock import patch as _patch
import pytest


def _make_home_client(tmp_path):
    return CopilotACPClient(
        api_key="copilot-acp",
        base_url="acp://copilot",
        acp_command="copilot",
        acp_args=["--acp", "--stdio"],
        acp_cwd=str(tmp_path),
    )


def _fake_popen_capture(captured):
    def _fake(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        raise FileNotFoundError("copilot not found")
    return _fake


def test_run_prompt_prefers_profile_home_when_available(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes"
    profile_home = hermes_home / "home"
    profile_home.mkdir(parents=True)

    monkeypatch.delenv("HOME", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    captured = {}
    client = _make_home_client(tmp_path)

    with _patch("agent.copilot_acp_client.subprocess.Popen", side_effect=_fake_popen_capture(captured)):
        with pytest.raises(RuntimeError, match="Could not start Copilot ACP command"):
            client._run_prompt("hello", timeout_seconds=1)

    assert captured["kwargs"]["env"]["HOME"] == str(profile_home)


def test_run_prompt_passes_home_when_parent_env_is_clean(monkeypatch, tmp_path):
    monkeypatch.delenv("HOME", raising=False)
    monkeypatch.delenv("HERMES_HOME", raising=False)

    captured = {}
    client = _make_home_client(tmp_path)

    with _patch("agent.copilot_acp_client.subprocess.Popen", side_effect=_fake_popen_capture(captured)):
        with pytest.raises(RuntimeError, match="Could not start Copilot ACP command"):
            client._run_prompt("hello", timeout_seconds=1)

    assert "env" in captured["kwargs"]
    assert captured["kwargs"]["env"]["HOME"]
