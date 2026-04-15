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
