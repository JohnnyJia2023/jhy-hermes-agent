import threading

from gateway import run


def test_copilot_acp_refresh_enabled_only_for_copilot_acp(monkeypatch):
    monkeypatch.setenv("HERMES_COPILOT_ACP_REFRESH_INTERVAL_SECONDS", "7200")

    assert run._copilot_acp_refresh_enabled({"model": {"provider": "copilot-acp"}}) is True
    assert run._copilot_acp_refresh_enabled({"model": {"provider": "copilot"}}) is False


def test_copilot_acp_refresh_invalid_interval_disables_refresh(monkeypatch):
    monkeypatch.setenv("HERMES_COPILOT_ACP_REFRESH_INTERVAL_SECONDS", "abc")

    assert run._copilot_acp_refresh_interval_seconds() == 0


def test_copilot_acp_refresh_ticker_probes_immediately(monkeypatch):
    stop_event = threading.Event()
    calls = []

    def _fake_probe(*, timeout_seconds):
        calls.append(timeout_seconds)
        stop_event.set()

    monkeypatch.setattr("agent.copilot_acp_client.probe_copilot_acp", _fake_probe)

    run._start_copilot_acp_refresh_ticker(
        stop_event,
        interval_seconds=7200,
        timeout_seconds=9.5,
    )

    assert calls == [9.5]
