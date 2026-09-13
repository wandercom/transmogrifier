"""Exercise the declared Gemini SDK without invoking a provider."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

genai = pytest.importorskip("google.genai", reason="requires the gemini extra")

from transmogrifier.backends.gemini import GeminiBackend


def test_gemini_uses_declared_sdk_and_reuses_client(monkeypatch):
    client = Mock()
    client.models.generate_content.return_value = SimpleNamespace(text="translated")
    constructor = Mock(return_value=client)
    monkeypatch.setattr(genai, "Client", constructor)
    backend = GeminiBackend(api_key="fixture-only", model="fixture-model")

    result = backend.complete(
        system="Use direct language",
        messages=[{"role": "user", "content": "first"}, {"role": "assistant", "content": "ignore"},
                  {"role": "user", "content": "second"}],
        max_tokens=42,
    )
    assert result == "translated"
    constructor.assert_called_once_with(api_key="fixture-only")
    kwargs = client.models.generate_content.call_args.kwargs
    assert kwargs["model"] == "fixture-model"
    assert kwargs["contents"] == "first\n\nsecond"
    assert kwargs["config"].system_instruction == "Use direct language"
    assert kwargs["config"].max_output_tokens == 42
    assert kwargs["config"].temperature == 0

    backend.complete(system="", messages=[], max_tokens=8)
    constructor.assert_called_once()
    assert client.models.generate_content.call_args.kwargs["config"].system_instruction is None
