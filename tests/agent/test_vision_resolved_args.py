"""Regression tests for the vision resolver call path."""

from unittest.mock import AsyncMock, MagicMock, patch


def test_vision_call_uses_only_explicit_overrides():
    """Vision should re-read provider config instead of forcing resolved base_url."""
    from agent.auxiliary_client import call_llm

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="description"))],
        usage=MagicMock(prompt_tokens=10, completion_tokens=5),
    )

    with patch(
        "agent.auxiliary_client._resolve_task_provider_model",
        return_value=("my-resolved-provider", "my-resolved-model", "http://resolved", "resolved-key", "chat_completions"),
    ), patch(
        "agent.auxiliary_client.resolve_vision_provider_client",
        return_value=("my-resolved-provider", fake_client, "my-resolved-model"),
    ) as mock_vision:
        call_llm(
            "vision",
            provider="raw-provider",
            model="raw-model",
            base_url="http://raw",
            api_key="raw-key",
            messages=[{"role": "user", "content": "describe this"}],
        )

    call_args = mock_vision.call_args
    assert call_args.kwargs["provider"] == "raw-provider"
    assert call_args.kwargs["model"] == "raw-model"
    assert call_args.kwargs["base_url"] == "http://raw"
    assert call_args.kwargs["api_key"] == "raw-key"


def test_async_vision_call_uses_only_explicit_overrides():
    """Async vision path must not feed resolved base_url back as a custom override."""
    from agent.auxiliary_client import async_call_llm

    fake_client = MagicMock()
    fake_client.chat.completions.create = AsyncMock(
        return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content="description"))],
            usage=MagicMock(prompt_tokens=10, completion_tokens=5),
        )
    )

    with patch(
        "agent.auxiliary_client._resolve_task_provider_model",
        return_value=("my-resolved-provider", "my-resolved-model", "http://resolved", "resolved-key", "chat_completions"),
    ), patch(
        "agent.auxiliary_client.resolve_vision_provider_client",
        return_value=("my-resolved-provider", fake_client, "my-resolved-model"),
    ) as mock_vision:
        import asyncio

        asyncio.run(
            async_call_llm(
                "vision",
                provider="raw-provider",
                model="raw-model",
                base_url="http://raw",
                api_key="raw-key",
                messages=[{"role": "user", "content": "describe this"}],
            )
        )

    call_args = mock_vision.call_args
    assert call_args.kwargs["provider"] == "raw-provider"
    assert call_args.kwargs["model"] == "raw-model"
    assert call_args.kwargs["base_url"] == "http://raw"
    assert call_args.kwargs["api_key"] == "raw-key"


def test_vision_base_url_override_keeps_explicit_provider():
    """Explicit provider should still drive credential resolution with custom base_url."""
    from agent.auxiliary_client import resolve_vision_provider_client

    fake_client = MagicMock()
    with patch(
        "agent.auxiliary_client._resolve_task_provider_model",
        return_value=(
            "zai",
            "glm-4v",
            "https://open.bigmodel.cn/api/paas/v4",
            None,
            "chat_completions",
        ),
    ), patch(
        "agent.auxiliary_client.resolve_provider_client",
        return_value=(fake_client, "glm-4v"),
    ) as mock_resolve:
        provider, client, model = resolve_vision_provider_client()

    assert provider == "zai"
    assert client is fake_client
    assert model == "glm-4v"
    assert mock_resolve.call_args.args[0] == "zai"
    assert mock_resolve.call_args.kwargs["explicit_base_url"] == "https://open.bigmodel.cn/api/paas/v4"
