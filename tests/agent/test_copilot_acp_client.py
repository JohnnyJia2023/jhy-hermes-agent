from agent.copilot_acp_client import CopilotACPClient, probe_copilot_acp


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
